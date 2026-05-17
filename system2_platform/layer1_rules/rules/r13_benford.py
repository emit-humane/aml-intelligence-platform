"""R13 — Benford's Law Violation."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

# Chi-squared critical value (df=1, alpha=0.05): 3.84
_CHI2_THRESHOLD = 3.84
_WEIGHT = 0.7


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if fv.benford_chi2_score > _CHI2_THRESHOLD:
        return True, _WEIGHT, (
            f"R13: Benford's law violation chi2={fv.benford_chi2_score:.2f} > 3.84 "
            f"(transaction amounts do not follow natural distribution)"
        )
    return False, 0.0, ""
