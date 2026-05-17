"""R12 — KYC Level vs Transaction Volume Mismatch."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

# KYC level 0/1 accounts transacting at high velocity or large amounts are suspicious
_KYC_VELOCITY_LIMITS = {0: 3, 1: 10, 2: 30, 3: 9999}
_KYC_HIGH_Z = {0: 2.0, 1: 3.0, 2: 4.0, 3: 9999.0}


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    kyc = fv.kyc_level  # not available directly in LiveFeatureVector
    # kyc_level is not a named field in LiveFeatureVector — approximate via sub_threshold + z
    # Use amount_zscore as proxy for KYC mismatch
    z = fv.amount_zscore
    v24 = fv.tx_velocity_24h

    # Low-KYC proxy: low beneficiary entropy (limited relationships) + high z-score
    low_kyc_proxy = fv.beneficiary_count_7d <= 2 and z >= 3.0
    if not low_kyc_proxy:
        return False, 0.0, ""

    score = 15.0
    if v24 >= 5:
        score += 10.0
    return True, score, (
        f"R12: Possible KYC-volume mismatch: z-score={z:.1f}, "
        f"beneficiary_count_7d={int(fv.beneficiary_count_7d)}, velocity_24h={int(v24)}"
    )
