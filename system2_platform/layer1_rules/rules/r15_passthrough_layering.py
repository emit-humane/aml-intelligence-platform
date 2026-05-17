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

    # Base score — amplified by round-amount flag and amount magnitude
    score = 20.0
    if fv.round_amount_flag:
        score += 10.0                          # round amounts are a structuring signal
    score += min(fv.amount / 1_000_000 * 5.0, 15.0)   # up to +15 for very large amounts

    # Stronger signal if sender_fan_in_score is balanced (equal in/out = pure pass-through)
    fan_in  = gfv.sender_fan_in_score
    fan_out = gfv.sender_fan_out_score
    if 0.25 <= fan_in <= 0.75 and 0.25 <= fan_out <= 0.75:
        score += 10.0    # balanced in/out = classic pass-through node

    return True, min(score, 55.0), (
        f"R15: Pass-through node — sender has {gfv.sender_in_degree} incoming edge(s) "
        f"and is forwarding {fv.amount:,.0f} INR "
        f"(fan_in={fan_in:.2f}, round={fv.round_amount_flag}) — layering chain indicator"
    )
