"""R07 — New / Anomalous Device."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_WEIGHT = 0.6


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.new_device_flag and fv.amount > 200_000:
        return True, _WEIGHT, (
            f"R07: Transaction from new/unseen device with amount={fv.amount:,.0f}"
        )
    return False, 0.0, ""
