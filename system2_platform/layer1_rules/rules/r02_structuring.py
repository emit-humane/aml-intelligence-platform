"""R02 — Structuring (sub-threshold pattern)."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if not fv.sub_threshold_flag:
        return False, 0.0, ""

    # Amplify if multiple sub-threshold txns in 24h
    v24 = fv.tx_velocity_24h
    if v24 >= 5:
        return True, 35.0, (
            f"R02: Sub-threshold amount with {int(v24)} transactions in 24h "
            f"(structuring pattern)"
        )
    if v24 >= 3:
        return True, 25.0, (
            f"R02: Sub-threshold amount with {int(v24)} transactions in 24h"
        )
    return True, 15.0, "R02: Single sub-threshold transaction (potential structuring)"
