"""R08 — Shared Device Across Multiple Accounts."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_WEIGHT = 0.7


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.shared_device_count > 5:
        return True, _WEIGHT, (
            f"R08: Device used by {fv.shared_device_count} distinct accounts "
            f"(shared device — possible mule network)"
        )
    return False, 0.0, ""
