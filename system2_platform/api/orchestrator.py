"""
Transaction Orchestrator.

Wires all detection layers together for a single transaction event:

  TransactionEvent
       │
       ├─→ [S5] FeatureUpdater.update_and_extract()  → LiveFeatureVector
       ├─→ [S6] GraphUpdater.update_and_extract()    → LiveGraphFeatureVector
       │
       ├─→ [L1] RuleEngine.evaluate()               → RuleEngineOutput
       ├─→ [L3] BehavioralEngine.score()             → BehavioralAnomalyOutput
       ├─→ [L4] GNNEngine.score()                   → GNNInferenceOutput (Pydantic)
       │
       ├─→ [P1] RiskFusion.fuse()                   → FusedRiskOutput
       └─→ [P2] AlertManager.maybe_create_alert()   → Alert | None

All engines are loaded once at startup (Singleton via dependency injection).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np

from ..contracts.transaction_event import TransactionEvent
from ..contracts.live_feature_vector import LiveFeatureVector
from ..contracts.live_graph_feature_vector import LiveGraphFeatureVector
from ..contracts.rule_engine_output import RuleEngineOutput
from ..contracts.behavioral_anomaly_output import BehavioralAnomalyOutput
from ..contracts.gnn_inference_output import GNNInferenceOutput
from ..contracts.fused_risk_output import FusedRiskOutput
from ..contracts.alert import Alert

from ..layer1_rules.engine import RuleEngine
from ..layer3_behavioral.l3b_inference import BehavioralInferenceEngine
from ..layer4_gnn.l4b_inference import GNNInferenceEngine
from ..shared.s5_feature_update import FeatureUpdater
from ..shared.s6_graph_update import GraphUpdater
from ..post_detection.p1_risk_fusion import RiskFusion
from ..post_detection.p2_alert_manager import AlertManager


class TransactionOrchestrator:
    """
    Stateful orchestrator. Load once, call process() per transaction.
    """

    def __init__(
        self,
        rule_engine: RuleEngine,
        behavioral_engine: BehavioralInferenceEngine,
        gnn_engine: GNNInferenceEngine,
        feature_updater: FeatureUpdater,
        graph_updater: GraphUpdater,
        risk_fusion: RiskFusion,
        alert_manager: AlertManager,
    ) -> None:
        self._rules   = rule_engine
        self._behav   = behavioral_engine
        self._gnn     = gnn_engine
        self._feat    = feature_updater
        self._graph   = graph_updater
        self._fusion  = risk_fusion
        self._alerts  = alert_manager

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: Path | str,
        historical_parquet: Optional[Path | str] = None,
    ) -> "TransactionOrchestrator":
        """
        Factory: loads all engines from disk.
        historical_parquet seeds the graph updater if provided.
        """
        artifact_dir = Path(artifact_dir)
        print(f"[Orchestrator] Loading engines from {artifact_dir}...")

        rule_engine   = RuleEngine()
        behav_engine  = BehavioralInferenceEngine.load(artifact_dir)
        gnn_engine    = GNNInferenceEngine.load(artifact_dir)
        feat_updater  = FeatureUpdater(artifact_dir)
        graph_updater = GraphUpdater(artifact_dir=artifact_dir)
        risk_fusion   = RiskFusion()
        alert_mgr     = AlertManager()

        if historical_parquet and Path(historical_parquet).exists():
            graph_updater.seed_from_historical(historical_parquet)

        print("[Orchestrator] All engines loaded.")
        return cls(rule_engine, behav_engine, gnn_engine,
                   feat_updater, graph_updater, risk_fusion, alert_mgr)

    # ------------------------------------------------------------------
    # Main processing path
    # ------------------------------------------------------------------

    def process(
        self,
        event: TransactionEvent,
    ) -> dict:
        """
        Full pipeline for one transaction.

        Returns a dict with:
          fv, gfv, rule_out, behav_out, gnn_out, fused, alert (or None)
        """
        # Layer S5: Feature extraction
        fv: LiveFeatureVector = self._feat.update_and_extract(event)

        # Layer S6: Graph update + structural features
        gfv: LiveGraphFeatureVector = self._graph.update_and_extract(event)

        # Layer L1: Deterministic rules
        rule_out: RuleEngineOutput = self._rules.evaluate(fv, gfv)

        # Layer L3: Behavioral anomaly
        behav_out: BehavioralAnomalyOutput = self._behav.score(fv, gfv)

        # Layer L4: GNN structural
        gnn_raw = self._gnn.score(event.sender_account, event.receiver_account)
        gnn_out: GNNInferenceOutput = _convert_gnn(
            event.transaction_id, gnn_raw, gfv
        )

        # Layer P1: Fusion
        fused: FusedRiskOutput = self._fusion.fuse(
            rule_out, behav_out, gnn_out,
            gfv=gfv,
            sender_account=event.sender_account,
        )

        # Layer P2: Alert
        alert: Optional[Alert] = self._alerts.maybe_create_alert(
            fused, rule_out, behav_out, gnn_out, gfv, event=event
        )

        return {
            "fv":       fv,
            "gfv":      gfv,
            "rule_out": rule_out,
            "behav_out": behav_out,
            "gnn_out":  gnn_out,
            "fused":    fused,
            "alert":    alert,
        }

    # ------------------------------------------------------------------
    # Async processing path — runs L1, L3B, L4B concurrently
    # ------------------------------------------------------------------

    async def async_process(self, event: TransactionEvent) -> dict:
        """
        Async pipeline: S5 and S6 run sequentially (they mutate state),
        then L1, L3B, L4B run concurrently via asyncio.gather using
        asyncio.to_thread() so they don't block the event loop.

        Target: <500ms per transaction.
        """
        loop = asyncio.get_event_loop()

        # S5 + S6: stateful — must run sequentially first
        fv:  LiveFeatureVector      = await loop.run_in_executor(
            None, self._feat.update_and_extract, event
        )
        gfv: LiveGraphFeatureVector = await loop.run_in_executor(
            None, self._graph.update_and_extract, event
        )

        # L1, L3B, L4B: read-only — run concurrently
        rule_fut  = loop.run_in_executor(None, self._rules.evaluate, fv, gfv)
        behav_fut = loop.run_in_executor(None, self._behav.score, fv, gfv)
        gnn_raw_fut = loop.run_in_executor(
            None, self._gnn.score, event.sender_account, event.receiver_account
        )

        rule_out_raw, behav_out, gnn_raw = await asyncio.gather(
            rule_fut, behav_fut, gnn_raw_fut
        )

        rule_out: RuleEngineOutput        = rule_out_raw
        gnn_out:  GNNInferenceOutput      = _convert_gnn(event.transaction_id, gnn_raw, gfv)

        # P1 + P2: sequential (fast, no IO)
        fused: FusedRiskOutput = self._fusion.fuse(
            rule_out, behav_out, gnn_out,
            gfv=gfv,
            sender_account=event.sender_account,
        )
        alert: Optional[Alert] = self._alerts.maybe_create_alert(
            fused, rule_out, behav_out, gnn_out, gfv, event=event
        )

        return {
            "fv":        fv,
            "gfv":       gfv,
            "rule_out":  rule_out,
            "behav_out": behav_out,
            "gnn_out":   gnn_out,
            "fused":     fused,
            "alert":     alert,
        }

    @property
    def alert_manager(self) -> AlertManager:
        return self._alerts

    @property
    def graph_updater(self) -> GraphUpdater:
        return self._graph


# ---------------------------------------------------------------------------
# Helper: convert internal GNNResult dataclass → contracts GNNInferenceOutput
# ---------------------------------------------------------------------------

def _convert_gnn(
    transaction_id: str,
    raw,                         # GNNInferenceEngine._GNNResult dataclass
    gfv: Optional[LiveGraphFeatureVector],
) -> GNNInferenceOutput:
    """
    Convert l4b internal GNNInferenceOutput dataclass to the Pydantic contract.
    """
    # Structural explanations
    explanations: list[str] = []
    if gfv and gfv.edge_creates_cycle:
        explanations.append(f"cycle_closure (len={gfv.cycle_length})")
    if not raw.src_known:
        explanations.append("unknown_sender_in_training_graph")
    if not raw.dst_known:
        explanations.append("unknown_receiver_in_training_graph")
    if raw.gnn_anomaly_score > 60:
        explanations.append("high_embedding_dissimilarity")

    # Community anomaly flag
    community_anomaly = (
        gfv is not None and (
            gfv.sender_community_risk_score > 60
            or gfv.sender_benford_chi2_community > 10.0
        )
    )

    return GNNInferenceOutput(
        transaction_id=transaction_id,
        sender_embedding=raw.embedding_src.tolist(),
        receiver_embedding=raw.embedding_dst.tolist(),
        sender_embedding_drift=0.0,   # drift vs baseline not implemented in v1
        receiver_embedding_drift=0.0,
        link_anomaly_score=float(1.0 - raw.link_score),
        gnn_anomaly_score=raw.gnn_anomaly_score,
        structural_anomaly_explanations=explanations,
        community_anomaly_flag=community_anomaly,
    )
