"""R08 — Shared Device Across Multiple Accounts."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_SHARED_THRESHOLD = 3  # device used by >= 3 accounts is suspicious


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    count = fv.shared_device_count
    if count < _SHARED_THRESHOLD:
        return False, 0.0, ""

    score = min(10.0 + (count - _SHARED_THRESHOLD) * 5.0, 30.0)
    return True, score, (
        f"R08: Device used by {count} distinct accounts "
        f"(shared device — possible mule network)"
    )
