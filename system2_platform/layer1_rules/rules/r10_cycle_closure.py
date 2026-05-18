"""R10 — Transaction Closes a Cycle in the Transaction Graph."""
from __future__ import annotations
from ...contracts.live_feature_vector import LiveFeatureVector
from ...contracts.live_graph_feature_vector import LiveGraphFeatureVector


def rule(
    fv: LiveFeatureVector,
    gfv: LiveGraphFeatureVector | None,
) -> tuple[bool, float, str]:
    if gfv is None or not gfv.edge_creates_cycle:
        return False, 0.0, ""

    cycle_len = gfv.cycle_length
    # Weight must stay in the same 0–1 range as all other rules so that
    # engine.py's normalization  (raw_score / MAX_POSSIBLE) * 100  works
    # correctly.  A 2-hop cycle is the strongest signal (full weight=1.0);
    # longer cycles decay but stay above the 0.7 baseline of a weak rule.
    score = 1.0 if cycle_len <= 3 else (0.85 if cycle_len <= 5 else 0.7)
    return True, score, (
        f"R10: Transaction closes a {cycle_len}-hop cycle in the transaction graph "
        f"(circular laundering pattern)"
    )
