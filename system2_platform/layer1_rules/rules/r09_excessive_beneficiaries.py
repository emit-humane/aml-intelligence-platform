"""R09 — Excessive Unique Beneficiaries (fan-out pattern)."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_WEIGHT = 0.8


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.beneficiary_count_7d > 20:
        return True, _WEIGHT, (
            f"R09: {int(fv.beneficiary_count_7d)} unique beneficiaries in 7d "
            f"(fan-out / dispersion pattern)"
        )
    return False, 0.0, ""
