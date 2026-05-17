"""R15 — Pass-Through / Layering Chain Detection.

Detects intermediate nodes in a layering chain: accounts that have
previously received money (sender_in_degree >= 1) and are now forwarding
a large amount onward. Combined with a round-amount or high-value flag
this is a strong layering indicator.

This rule fires on the sender side (when B in A->B->C is the sender).
"""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector

_MIN_AMOUNT         = 500_000.0   # 5 lakh INR minimum to be interesting
_MIN_IN_DEGREE      = 1           # sender must have received at least once
_WEIGHT             = 0.9         # fixed per-rule weight used by the scoring formula


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if gfv is None:
        return False, 0.0, ""

    # Sender must have previously received AND be sending forward now
    if gfv.sender_in_degree < _MIN_IN_DEGREE:
        return False, 0.0, ""

    # Amount must be large enough to be suspicious
    if fv.amount < _MIN_AMOUNT:
        return False, 0.0, ""

    fan_in  = gfv.sender_fan_in_score
    fan_out = gfv.sender_fan_out_score

    return True, _WEIGHT, (
        f"R15: Pass-through node — sender has {gfv.sender_in_degree} incoming edge(s) "
        f"and is forwarding {fv.amount:,.0f} INR "
        f"(fan_in={fan_in:.2f}, round={fv.round_amount_flag}) — layering chain indicator"
    )
