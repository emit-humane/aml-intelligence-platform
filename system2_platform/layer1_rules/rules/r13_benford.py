"""R13 — Benford's Law Violation."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

# Chi-squared critical values (df=8): p<0.05 -> 15.507, p<0.01 -> 20.090
_CHI2_HIGH = 20.0
_CHI2_MEDIUM = 15.5


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    chi2 = fv.benford_chi2_score
    if chi2 >= _CHI2_HIGH:
        return True, 20.0, (
            f"R13: Benford's law violation chi2={chi2:.1f} (p<0.01) — "
            f"transaction amounts do not follow natural distribution"
        )
    if chi2 >= _CHI2_MEDIUM:
        return True, 10.0, (
            f"R13: Benford's law marginal violation chi2={chi2:.1f} (p<0.05)"
        )
    return False, 0.0, ""
