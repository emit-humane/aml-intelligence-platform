"""
Layer 1 — Rule Engine.

Runs all 15 deterministic rules against a LiveFeatureVector (and optionally
a LiveGraphFeatureVector) and produces a RuleEngineOutput.

Design:
  - Each rule is a callable: rule(fv, gfv) -> (triggered, weight, explanation)
  - weight is the fixed per-rule weight defined in WEIGHTS
  - rule_score = min(100.0, round((sum_of_fired_weights / MAX_POSSIBLE) * 100, 2))
  - MAX_POSSIBLE = sum of all rule weights = 12.2
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
from .rules.r14_fan_in_collector import rule as r14
from .rules.r15_passthrough_layering import rule as r15

# Per-rule weights as defined in spec
WEIGHTS: dict[str, float] = {
    "R01": 1.0,
    "R02": 1.0,
    "R03": 0.8,
    "R04": 0.9,
    "R05": 0.9,
    "R06": 0.7,
    "R07": 0.6,
    "R08": 0.7,
    "R09": 0.8,
    "R10": 1.0,
    "R11": 0.6,
    "R12": 0.8,
    "R13": 0.7,
    "R14": 0.8,
    "R15": 0.9,
}

# Sum of all weights: 1.0+1.0+0.8+0.9+0.9+0.7+0.6+0.7+0.8+1.0+0.6+0.8+0.7+0.8+0.9 = 12.2
MAX_POSSIBLE = 12.2

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
    ("R14", r14),
    ("R15", r15),
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
        raw_score = 0.0

        for rule_id, rule_fn in _RULES:
            try:
                fired, score, explanation = rule_fn(fv, gfv)
            except Exception as exc:
                # Never let a single rule crash the engine
                fired, score, explanation = False, 0.0, f"[ERROR] {rule_id}: {exc}"

            if fired:
                triggered_ids.append(rule_id)
                explanations.append(explanation)
                raw_score += score

        rule_score = min(100.0, round((raw_score / MAX_POSSIBLE) * 100, 2))

        return RuleEngineOutput(
            transaction_id=fv.transaction_id,
            rule_score=rule_score,
            triggered_rules=triggered_ids,
            rule_explanations=explanations,
            rule_count=len(triggered_ids),
        )
