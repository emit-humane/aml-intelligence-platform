"""R06 — High-Risk Jurisdiction."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if not fv.high_risk_country_flag:
        return False, 0.0, ""

    xb = fv.cross_border_ratio_30d
    ctr_sw = fv.country_switch_count_7d
    score = 20.0
    if xb >= 0.5:
        score += 10.0
    if ctr_sw >= 3:
        score += 5.0

    return True, min(score, 35.0), (
        f"R06: Transaction involves high-risk jurisdiction; "
        f"cross_border_ratio_30d={xb:.2f}, country_switches_7d={int(ctr_sw)}"
    )
