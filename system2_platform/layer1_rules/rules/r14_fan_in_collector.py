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


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if gfv is None:
        return False, 0.0, ""

    n = gfv.receiver_in_degree_unique
    if n < _THRESHOLD_LOW:
        return False, 0.0, ""

    if n >= _THRESHOLD_HIGH:
        score = min(35.0 + (n - _THRESHOLD_HIGH) * 3.0, 50.0)
    else:
        score = 20.0 + (n - _THRESHOLD_LOW) * 5.0   # 20 → 40

    return True, min(score, 50.0), (
        f"R14: Receiver {gfv.receiver_account} has {n} unique incoming senders "
        f"(fan-in collector — possible layering hub or mule account)"
    )
