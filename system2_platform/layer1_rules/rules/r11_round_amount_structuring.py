"""R11 — Round Amount Structuring."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if not fv.round_amount_flag:
        return False, 0.0, ""

    # Amplify if this happens repeatedly
    v7 = fv.tx_velocity_7d
    score = 10.0 + min(v7 * 1.5, 15.0)
    return True, min(score, 25.0), (
        f"R11: Round-number amount with {int(v7)} transactions in 7d "
        f"(structuring indicator)"
    )
