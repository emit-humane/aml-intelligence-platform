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
    score = 35.0 if cycle_len <= 3 else (25.0 if cycle_len <= 5 else 15.0)
    return True, score, (
        f"R10: Transaction closes a {cycle_len}-hop cycle in the transaction graph "
        f"(circular laundering pattern)"
    )
