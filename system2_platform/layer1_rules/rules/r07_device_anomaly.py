"""R07 — New / Anomalous Device."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if not fv.new_device_flag:
        return False, 0.0, ""

    z = fv.amount_zscore
    score = 15.0
    if z >= 2.0:
        score += 10.0  # new device + large amount is worse

    return True, score, (
        f"R07: Transaction from a new/unseen device; amount_zscore={z:.1f}"
    )
