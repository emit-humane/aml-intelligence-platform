"""
Layer 1 — Rule Engine.

Runs all 13 deterministic rules against a LiveFeatureVector (and optionally
a LiveGraphFeatureVector) and produces a RuleEngineOutput.

Design:
  - Each rule is a callable: rule(fv, gfv) -> (triggered, score, explanation)
  - Scores from triggered rules are summed and clamped to [0, 100]
  - rule_score represents suspicion level; individual scores are calibrated
    so a single critical rule (~40 pts) + a supporting rule (~20 pts) approaches
    the High band (61–80).
"""

from __future__ import annotations

from typing import Callable

from ..contracts.live_feature_vector import LiveFeatureVector
from ..contracts.live_graph_feature_vector import LiveGraphFeatureVector
from ..contracts.rule_engine_output import RuleEngineOutput

from .rules.r01_high_value import rule as r01
from .rules.r02_structuring import rule as r02
from .rules.r03_velocity import rule as r03
from .rules.r04_dormant_activation import rule as r04
from .rules.r05_impossible_travel import rule as r05
from .rules.r06_high_risk_jurisdiction import rule as r06
from .rules.r07_device_anomaly import rule as r07
from .rules.r08_shared_device import rule as r08
from .rules.r09_excessive_beneficiaries import rule as r09
from .rules.r10_cycle_closure import rule as r10
from .rules.r11_round_amount_structuring import rule as r11
from .rules.r12_kyc_mismatch import rule as r12
from .rules.r13_benford import rule as r13

# (rule_id, callable)
_RULES: list[tuple[str, Callable]] = [
    ("R01", r01),
    ("R02", r02),
    ("R03", r03),
    ("R04", r04),
    ("R05", r05),
    ("R06", r06),
    ("R07", r07),
    ("R08", r08),
    ("R09", r09),
    ("R10", r10),
    ("R11", r11),
    ("R12", r12),
    ("R13", r13),
]


class RuleEngine:
    """Stateless rule runner."""

    def evaluate(
        self,
        fv: LiveFeatureVector,
        gfv: LiveGraphFeatureVector | None = None,
    ) -> RuleEngineOutput:
        triggered_ids: list[str] = []
        explanations: list[str] = []
        total_score = 0.0

        for rule_id, rule_fn in _RULES:
            try:
                fired, score, explanation = rule_fn(fv, gfv)
            except Exception as exc:
                # Never let a single rule crash the engine
                fired, score, explanation = False, 0.0, f"[ERROR] {rule_id}: {exc}"

            if fired:
                triggered_ids.append(rule_id)
                explanations.append(explanation)
                total_score += score

        clamped = min(100.0, max(0.0, total_score))

        return RuleEngineOutput(
            transaction_id=fv.transaction_id,
            rule_score=round(clamped, 2),
            triggered_rules=triggered_ids,
            rule_explanations=explanations,
            rule_count=len(triggered_ids),
        )
