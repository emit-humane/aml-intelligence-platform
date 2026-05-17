"""R01 — High-Value Single Transaction."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    z = fv.amount_zscore
    avg_vol = fv.avg_daily_volume_30d

    if z >= 10.0:
        return True, 40.0, f"R01: Extreme amount z-score {z:.1f} vs 30d baseline; avg_daily_vol={avg_vol:.0f}"
    if z >= 5.0:
        return True, 25.0, f"R01: High amount z-score {z:.1f} vs account 30d baseline"
    if z >= 3.0:
        return True, 15.0, f"R01: Elevated amount z-score {z:.1f} vs account 30d baseline"
    return False, 0.0, ""
