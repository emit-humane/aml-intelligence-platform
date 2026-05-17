"""R11 — Round Amount Structuring."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_WEIGHT = 0.6


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.round_amount_flag and fv.tx_velocity_24h > 5:
        return True, _WEIGHT, (
            f"R11: Round-number amount with tx_velocity_24h={int(fv.tx_velocity_24h)} "
            f"(structuring indicator)"
        )
    return False, 0.0, ""
