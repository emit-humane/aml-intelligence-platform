"""
P1 — Risk Score Fusion.

Combines outputs from:
  L1 — Rule Engine       (rule_score ∈ [0, 100])
  L3 — Behavioral         (ensemble_anomaly_score ∈ [0, 100])
  L4 — GNN               (gnn_anomaly_score ∈ [0, 100])
  L2 — Graph (via gfv)   (community risk, cycle flags)

into a single FusedRiskOutput.

Fusion formula (static weights, v4 — recalibrated after the Level-2
eval-harness ordering fix corrected behavioral's measured AUROC):
  transaction_risk_score = 0.02 x rule_score
                         + 0.18 x behavioral_score
                         + 0.78 x gnn_score
                         + 0.02 x graph_boost

  Weights are calibrated to each detector's measured validation AUROC on the
  TRUE timestamp-ordered scoring path (scripts/replay_eval_ordered.py +
  scripts/tune_fusion_weights.py, 418 fraud / 1036 normal):
    gnn        AUROC=0.975 → strongest discriminator (highest weight)
    behavioral AUROC=0.845 → now correctly measured. The old two-phase
                             harness scored fraud/normal in separate
                             non-chronological batches, corrupting the
                             stateful rolling-window features and pinning
                             behavioral at a false 0.61 (so v3 under-weighted
                             it at 0.05). With true-order replay it is 0.845,
                             earning real weight. (0.996 reported earlier was
                             a behavioral-only isolation number that excluded
                             the 7 graph dims the production ensemble uses.)
    rule       AUROC=0.28  → near-random, small floor for explainability
    graph      AUROC=0.41  → near-random, small floor for ring context
  Grid search over the weight simplex: this mix gives fused validation
  AUROC 0.9773 (vs 0.9754 v3 / 0.9604 v2) at P≈0.91, R≈0.90.

  graph_boost = community_risk_score (0–100, Z-score normalized) if available

  group_risk_score = max(transaction_risk_score across community, default = tx score)

Score calibration (v5): the raw weighted sum ranks near-perfectly
(AUROC 0.977) but is crushed into a ~28–51 band, so fraud (~48) and normal
(~41) looked almost identical. A monotone logistic map
  transaction_risk_score = 100 * sigmoid(_CAL_A * raw + _CAL_B)
spreads it onto the full 0–100 scale (fraud median ~94, normal ~0.5) without
changing ranking/AUROC. The raw weighted sum is kept in
score_breakdown["raw_weighted"]. See scripts/fit_score_calibration.py.

Risk levels (on the CALIBRATED scale ≈ P(fraud)*100;
normal median ~0.5, fraud median ~94, F1-optimal decision point 57):
  0–56     → Low       (no alert)
  57–79    → Medium    (alert; F1-optimal zone, P≈0.92)
  80–94    → High       (high-precision zone)
  95–100   → Critical   (precision≈1.0)
"""

from __future__ import annotations

import math
from typing import Optional

from ..contracts.rule_engine_output import RuleEngineOutput
from ..contracts.behavioral_anomaly_output import BehavioralAnomalyOutput
from ..contracts.gnn_inference_output import GNNInferenceOutput
from ..contracts.live_graph_feature_vector import LiveGraphFeatureVector
from ..contracts.fused_risk_output import FusedRiskOutput, RiskLevel

# Static fusion weights, v4 — recalibrated after scripts/replay_eval_ordered.py
# fixed the eval-harness ordering bug. Behavioral's true measured AUROC is
# 0.845 (the old batched harness pinned it at a false 0.61, so v3 starved it
# at 0.05). GNN remains the strongest single detector (0.975) but behavioral
# now earns real weight; rule/graph keep a small explainability/ring floor.
_W_RULE  = 0.02
_W_BEHAV = 0.18
_W_GNN   = 0.78
_W_GRAPH = 0.02

# Score calibration (Platt / logistic), fit on the validation set by
# scripts/fit_score_calibration.py. The raw weighted sum discriminates
# near-perfectly by RANK (AUROC 0.977) but is crushed into a ~28-51 band
# (the dominant GNN component only spans ~25-50), so fraud (~48) and normal
# (~41) look almost identical. This monotone map spreads the score onto the
# full 0-100 range and makes it read as a calibrated fraud probability:
#   calibrated = 100 * sigmoid(_CAL_A * raw + _CAL_B)
# Monotone => AUROC/ranking is EXACTLY preserved (0.977 unchanged); only the
# scale changes (fraud median ~94, normal median ~0.5). The calibrated=50
# decision point corresponds to raw≈45.4 (the old v4 raw threshold).
_CAL_A = 1.118790
_CAL_B = -50.834839


def _calibrate(raw: float) -> float:
    """Logistic spread of the raw weighted sum -> calibrated 0-100 risk."""
    return 100.0 / (1.0 + math.exp(-(_CAL_A * raw + _CAL_B)))


def _risk_level(score: float) -> RiskLevel:
    # Bands on the CALIBRATED scale (≈ P(fraud)*100): normal median ~0.5,
    # fraud median ~94. F1-optimal decision point = 57; precision ≈1.0 at ≥95.
    if score < 57:
        return "Low"
    if score < 80:
        return "Medium"
    if score < 95:
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

        raw_weighted = (
            _W_RULE  * rule_out.rule_score
            + _W_BEHAV * behav_out.ensemble_anomaly_score
            + _W_GNN   * gnn_out.gnn_anomaly_score
            + _W_GRAPH * graph_boost
        )
        raw_weighted = float(min(max(raw_weighted, 0.0), 100.0))
        # Logistic calibration: spread the compressed raw band onto the full
        # 0-100 scale (monotone -> ranking/AUROC unchanged). See _calibrate.
        tx_score = float(min(max(_calibrate(raw_weighted), 0.0), 100.0))

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

        # Score breakdown (raw_weighted retained for transparency/debug —
        # transaction_risk_score is the calibrated value)
        breakdown = {
            "rule":         round(rule_out.rule_score, 2),
            "behavioral":   round(behav_out.ensemble_anomaly_score, 2),
            "gnn":          round(gnn_out.gnn_anomaly_score, 2),
            "graph_boost":  round(graph_boost, 2),
            "raw_weighted": round(raw_weighted, 2),
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
