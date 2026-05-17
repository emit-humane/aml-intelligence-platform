"""R09 — Excessive Unique Beneficiaries (fan-out pattern)."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_THRESH_7D = 10
_THRESH_30D = 30


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    b7 = fv.beneficiary_count_7d
    b30 = fv.beneficiary_count_30d

    if b7 >= _THRESH_7D:
        score = min(15.0 + (b7 - _THRESH_7D) * 2.0, 35.0)
        return True, score, (
            f"R09: {int(b7)} unique beneficiaries in 7d (fan-out / dispersion pattern)"
        )
    if b30 >= _THRESH_30D:
        return True, 15.0, (
            f"R09: {int(b30)} unique beneficiaries in 30d"
        )
    return False, 0.0, ""
