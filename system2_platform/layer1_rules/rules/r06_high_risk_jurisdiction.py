"""R06 — High-Risk Jurisdiction."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_WEIGHT = 0.7


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.high_risk_country_flag and fv.is_international:
        return True, _WEIGHT, (
            f"R06: Transaction involves high-risk jurisdiction and is international"
        )
    return False, 0.0, ""
