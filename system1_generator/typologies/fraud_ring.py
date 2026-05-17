"""
Fraud Ring typology.

A coordinated group of 4–8 accounts transact in a dense, partially connected
sub-graph. Money flows between all pairs in a short burst (24–96 hours),
creating a tightly coupled community detectable by graph algorithms.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import permutations

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row


class FraudRing(Typology):
    pattern_type = "fraud_ring"

    def inject(
        self,
        graph,
        accounts: pd.DataFrame,
        num_rings: int,
        rng: np.random.Generator,
        start_date: datetime,
        total_days: int,
        structuring_threshold: float,
    ) -> list[InjectedRing]:
        rings: list[InjectedRing] = []
        acc = accounts.reset_index(drop=True)

        for i in range(num_rings):
            ring_id = f"RING_FRD_{i:05d}"
            group_size = int(rng.integers(4, 9))

            indices = rng.choice(len(acc), size=group_size, replace=False)
            members = [acc.iloc[int(idx)] for idx in indices]

            # Select a random subset of directed pairs (not all — dense but not complete)
            all_pairs = list(permutations(range(group_size), 2))
            num_edges = int(rng.integers(group_size, min(len(all_pairs), group_size * 3)))
            edge_indices = rng.choice(len(all_pairs), size=num_edges, replace=False)
            edges = [all_pairs[int(j)] for j in edge_indices]

            total_value = float(rng.uniform(
                structuring_threshold * 0.3,
                structuring_threshold * 3.0,
            ))
            amounts = rng.uniform(10_000, structuring_threshold * 0.4, size=num_edges)

            window_sec = float(rng.uniform(24.0, 96.0)) * 3600
            day_offset = int(rng.integers(0, max(1, total_days - 4)))
            base_ts = start_date + timedelta(days=day_offset)
            offsets = np.sort(rng.uniform(0, window_sec, size=num_edges))
            timestamps = [base_ts + timedelta(seconds=float(s)) for s in offsets]

            txns = []
            for (s_idx, r_idx), amt, ts in zip(edges, amounts, timestamps):
                txns.append(make_tx_row(
                    sender=members[s_idx],
                    receiver=members[r_idx],
                    amount=round(float(amt), 2),
                    timestamp=ts,
                    tx_type=str(rng.choice(["TRANSFER", "PAYMENT"])),
                    channel=str(rng.choice(["UPI", "IMPS", "NEFT"])),
                    merchant_cat=str(rng.choice(["RETAIL", "TRANSFER", "OTHER"])),
                    is_fraud=True,
                    ring_id=ring_id,
                    pattern_type=self.pattern_type,
                    severity=_severity(total_value),
                    rng=rng,
                ))

            rings.append(InjectedRing(
                ring_id=ring_id,
                member_accounts=[m["account_id"] for m in members],
                transactions=txns,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                entry_account=members[edges[0][0]]["account_id"],
                exit_account=members[edges[-1][1]]["account_id"],
                num_hops=len(edges),
                total_value=total_value,
            ))

        return rings
