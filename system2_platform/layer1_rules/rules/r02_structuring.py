"""R02 — Structuring (sub-threshold pattern)."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_WEIGHT = 1.0


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.sub_threshold_flag and fv.tx_velocity_1h >= 3:
        return True, _WEIGHT, (
            f"R02: Sub-threshold amount with {int(fv.tx_velocity_1h)} transactions in 1h "
            f"(structuring pattern)"
        )
    return False, 0.0, ""
