"""
P1 — Risk Score Fusion.

Combines outputs from:
  L1 — Rule Engine       (rule_score ∈ [0, 100])
  L3 — Behavioral         (ensemble_anomaly_score ∈ [0, 100])
  L4 — GNN               (gnn_anomaly_score ∈ [0, 100])
  L2 — Graph (via gfv)   (community risk, cycle flags)

into a single FusedRiskOutput.

Fusion formula (static weights, v1):
  transaction_risk_score = 0.30 x rule_score
                         + 0.25 x behavioral_score
                         + 0.20 x gnn_score
                         + 0.25 x graph_boost

  graph_boost = community_risk_score (0–100) if available, else 0

  group_risk_score = max(transaction_risk_score across community, default = tx score)

Risk levels:
  0–30   → Low
  31–60  → Medium
  61–80  → High
  81–100 → Critical
"""

from __future__ import annotations

from typing import Optional

from ..contracts.rule_engine_output import RuleEngineOutput
from ..contracts.behavioral_anomaly_output import BehavioralAnomalyOutput
from ..contracts.gnn_inference_output import GNNInferenceOutput
from ..contracts.live_graph_feature_vector import LiveGraphFeatureVector
from ..contracts.fused_risk_output import FusedRiskOutput, RiskLevel

# Static fusion weights  (rule=0.30, graph=0.25, behavioral=0.25, gnn=0.20)
_W_RULE  = 0.30
_W_BEHAV = 0.25
_W_GNN   = 0.20
_W_GRAPH = 0.25


def _risk_level(score: float) -> RiskLevel:
    if score <= 30:
        return "Low"
    if score <= 60:
        return "Medium"
    if score <= 80:
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
