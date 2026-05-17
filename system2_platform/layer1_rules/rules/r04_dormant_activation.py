"""R04 — Dormant Account Activation."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

# If gap since last tx is very large (> 30 days = 2592000 sec) and z-score is high
_GAP_DORMANT_SEC = 30 * 86400


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    gap = fv.tx_gap_seconds
    z = fv.amount_zscore
    v24 = fv.tx_velocity_24h

    # tx_gap_seconds > 30 days indicates account was dormant
    if gap < _GAP_DORMANT_SEC:
        return False, 0.0, ""

    gap_days = gap / 86400
    base_score = 20.0
    if z >= 3.0:
        base_score += 15.0
    if v24 >= 3:
        base_score += 10.0

    return True, min(base_score, 45.0), (
        f"R04: Account inactive {gap_days:.0f} days; now tx z-score={z:.1f}, "
        f"velocity_24h={int(v24)} (dormant activation)"
    )
