"""R04 — Dormant Account Activation."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

# 180 days = 15_552_000 seconds
_GAP_DORMANT_SEC = 15_552_000
_WEIGHT = 0.9


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.tx_gap_seconds > _GAP_DORMANT_SEC and fv.amount > 100_000:
        gap_days = fv.tx_gap_seconds / 86400
        return True, _WEIGHT, (
            f"R04: Account inactive {gap_days:.0f} days; transaction amount={fv.amount:,.0f} "
            f"(dormant activation)"
        )
    return False, 0.0, ""
