"""R14 — Fan-In Collector (many unique senders converging on one receiver).

Detects accounts that act as collection hubs: multiple distinct accounts
all sending money to a single receiver within the live graph window.
This is the structural mirror of R09 (fan-out / dispersion).
"""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_THRESHOLD_LOW  = 3   # >= 3 unique senders to one receiver
_THRESHOLD_HIGH = 7   # >= 7 unique senders — strong signal
_WEIGHT = 0.8         # fixed per-rule weight used by the scoring formula


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if gfv is None:
        return False, 0.0, ""

    n = gfv.receiver_in_degree_unique
    if n < _THRESHOLD_LOW:
        return False, 0.0, ""

    return True, _WEIGHT, (
        f"R14: Receiver {gfv.receiver_account} has {n} unique incoming senders "
        f"(fan-in collector — possible layering hub or mule account)"
    )
