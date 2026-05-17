"""R03 — Velocity Burst."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_WEIGHT = 0.8


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.tx_velocity_1h > 10 or fv.tx_velocity_24h > 50:
        return True, _WEIGHT, (
            f"R03: Velocity spike — tx_velocity_1h={int(fv.tx_velocity_1h)}, "
            f"tx_velocity_24h={int(fv.tx_velocity_24h)}"
        )
    return False, 0.0, ""
