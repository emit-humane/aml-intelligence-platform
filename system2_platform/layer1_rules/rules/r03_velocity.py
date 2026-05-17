"""R03 — Velocity Burst."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_V1H_CRITICAL = 10
_V1H_HIGH = 5
_V6H_HIGH = 15
_V24H_HIGH = 30


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    v1h = fv.tx_velocity_1h
    v6h = fv.tx_velocity_6h
    v24h = fv.tx_velocity_24h

    if v1h >= _V1H_CRITICAL:
        return True, 40.0, f"R03: {int(v1h)} transactions in last hour (critical velocity)"
    if v1h >= _V1H_HIGH:
        return True, 25.0, f"R03: {int(v1h)} transactions in last hour"
    if v6h >= _V6H_HIGH:
        return True, 20.0, f"R03: {int(v6h)} transactions in last 6h"
    if v24h >= _V24H_HIGH:
        return True, 15.0, f"R03: {int(v24h)} transactions in last 24h"
    return False, 0.0, ""
