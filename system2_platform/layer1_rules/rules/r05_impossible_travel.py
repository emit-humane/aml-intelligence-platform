"""R05 — Impossible Travel."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if not fv.impossible_travel_flag:
        return False, 0.0, ""

    dist = fv.geo_distance_km
    # Weights must stay in [0, 1] — engine normalises: (sum_weights / 12.2) * 100.
    # WEIGHTS["R05"] = 0.9 is the ceiling; tier down for shorter displacements.
    if dist > 5000:
        score = 0.9   # intercontinental — very high confidence of account compromise
    elif dist > 2000:
        score = 0.75  # cross-country impossible travel
    else:
        score = 0.6   # sub-2000 km but still physically impossible given the gap
    return True, score, (
        f"R05: Impossible travel detected — {dist:.0f} km displacement "
        f"in too short a time (account compromise or proxy)"
    )
