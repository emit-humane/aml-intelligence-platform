"""R12 — KYC Level vs Transaction Volume Mismatch."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_WEIGHT = 0.8


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.kyc_level == 0 and fv.amount > 500_000:
        return True, _WEIGHT, (
            f"R12: KYC level 0 account transacting {fv.amount:,.0f} "
            f"(exceeds 500,000 threshold — KYC mismatch)"
        )
    return False, 0.0, ""
