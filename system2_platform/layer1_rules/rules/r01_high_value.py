"""R01 — High-Value Single Transaction."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_WEIGHT = 1.0


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.amount > 1_000_000:
        return True, _WEIGHT, f"R01: Transaction amount {fv.amount:,.0f} exceeds 1,000,000 threshold"
    return False, 0.0, ""
