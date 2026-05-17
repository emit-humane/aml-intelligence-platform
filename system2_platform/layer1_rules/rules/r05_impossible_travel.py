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
    score = 30.0 if dist > 5000 else 20.0
    return True, score, (
        f"R05: Impossible travel detected — {dist:.0f} km displacement "
        f"in too short a time (account compromise or proxy)"
    )
