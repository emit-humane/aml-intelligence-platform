"""
P1 — Risk Score Fusion.

Combines outputs from:
  L1 — Rule Engine       (rule_score ∈ [0, 100])
  L3 — Behavioral         (ensemble_anomaly_score ∈ [0, 100])
  L4 — GNN               (gnn_anomaly_score ∈ [0, 100])
  L2 — Graph (via gfv)   (community risk, cycle flags)

into a single FusedRiskOutput.

Fusion formula (static weights, v2 — calibrated on validation data):
  transaction_risk_score = 0.05 x rule_score
                         + 0.12 x behavioral_score
                         + 0.80 x gnn_score
                         + 0.03 x graph_boost

  Weights are calibrated to each detector's measured validation AUROC:
    gnn        AUROC=0.97  → dominant signal (highest weight)
    behavioral AUROC=0.67  → secondary signal
    rule       AUROC=0.41  → low weight, retained for explainability
    graph      AUROC=0.41  → low weight, retained for ring context
  This yields system AUROC≈0.96 vs 0.58 for the naive equal-ish weighting.

  graph_boost = community_risk_score (0–100, Z-score normalized) if available

  group_risk_score = max(transaction_risk_score across community, default = tx score)

Risk levels (recalibrated to v2 fused-score distribution):
  0–39   → Low       (no alert)
  40–44  → Medium    (alert; F1-optimal zone, recall≈0.96)
  45–48  → High       (precision≥0.90 zone)
  49–100 → Critical   (precision≈1.0 zone)
"""

from __future__ import annotations

from typing import Optional

from ..contracts.rule_engine_output import RuleEngineOutput
from ..contracts.behavioral_anomaly_output import BehavioralAnomalyOutput
from ..contracts.gnn_inference_output import GNNInferenceOutput
from ..contracts.live_graph_feature_vector import LiveGraphFeatureVector
from ..contracts.fused_risk_output import FusedRiskOutput, RiskLevel

# Static fusion weights, v2 — calibrated to per-detector validation AUROC.
# GNN is by far the strongest discriminator (AUROC 0.97) so it dominates;
# rule/graph are near-random (AUROC 0.41) so they are down-weighted but
# retained for human-readable explanations and ring context.
_W_RULE  = 0.05
_W_BEHAV = 0.12
_W_GNN   = 0.80
_W_GRAPH = 0.03


def _risk_level(score: float) -> RiskLevel:
    # Bands recalibrated to the v2 fused-score distribution
    # (normals center ~37, fraud ~46; F1-optimal decision point ~42).
    if score < 40:
        return "Low"
    if score < 45:
        return "Medium"
    if score < 49:
        return "High"
    return "Critical"


class RiskFusion:
    """
    Stateless risk fusion layer.

    Usage
    -----
    fusion = RiskFusion()
    output = fusion.fuse(rule_out, behav_out, gnn_out, gfv=gfv)
    """

    def fuse(
        self,
        rule_out: RuleEngineOutput,
        behav_out: BehavioralAnomalyOutput,
        gnn_out: GNNInferenceOutput,
        gfv: Optional[LiveGraphFeatureVector] = None,
        sender_account: str = "",
    ) -> FusedRiskOutput:

        # Graph boost from community risk
        graph_boost = 0.0
        if gfv is not None:
            graph_boost = float(gfv.sender_community_risk_score)
            if gfv.edge_creates_cycle:
                graph_boost = min(100.0, graph_boost + 20.0)

        tx_score = (
            _W_RULE  * rule_out.rule_score
            + _W_BEHAV * behav_out.ensemble_anomaly_score
            + _W_GNN   * gnn_out.gnn_anomaly_score
            + _W_GRAPH * graph_boost
        )
        tx_score = float(min(max(tx_score, 0.0), 100.0))

        # Group score: same as tx_score for single-transaction view
        # (the alert manager aggregates across communities separately)
        group_score = tx_score

        # Triggered patterns
        patterns: list[str] = list(rule_out.triggered_rules)
        if behav_out.ensemble_anomaly_score > 60:
            patterns.append("behavioral_anomaly")
        if gnn_out.gnn_anomaly_score > 60:
            patterns.append("gnn_structural_anomaly")
        if gfv is not None and gfv.edge_creates_cycle:
            patterns.append("cycle_closure")

        # Score breakdown
        breakdown = {
            "rule":        round(rule_out.rule_score, 2),
            "behavioral":  round(behav_out.ensemble_anomaly_score, 2),
            "gnn":         round(gnn_out.gnn_anomaly_score, 2),
            "graph_boost": round(graph_boost, 2),
            "weights": {
                "rule": _W_RULE, "behavioral": _W_BEHAV,
                "gnn": _W_GNN,   "graph": _W_GRAPH,
            },
        }

        # Human-readable explanation
        top_contributor = max(
            [("rule", rule_out.rule_score * _W_RULE),
             ("behavioral", behav_out.ensemble_anomaly_score * _W_BEHAV),
             ("gnn", gnn_out.gnn_anomaly_score * _W_GNN)],
            key=lambda x: x[1],
        )
        explanation = (
            f"Transaction risk {tx_score:.0f}/100 "
            f"(primary driver: {top_contributor[0]}). "
        )
        if rule_out.rule_explanations:
            explanation += rule_out.rule_explanations[0]

        return FusedRiskOutput(
            transaction_id=rule_out.transaction_id,
            sender_account=sender_account,
            transaction_risk_score=round(tx_score, 2),
            group_risk_score=round(group_score, 2),
            risk_level=_risk_level(tx_score),
            risk_level_group=_risk_level(group_score),
            score_breakdown=breakdown,
            triggered_patterns=patterns,
            explanation=explanation,
            fusion_mode="static_weights",
        )
