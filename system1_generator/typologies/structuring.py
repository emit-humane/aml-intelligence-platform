"""
Structuring (Smurfing) typology.

A single source account splits a large amount into 3–8 sub-threshold
transactions within a 24–48 hour window to evade reporting limits.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row


class Structuring(Typology):
    pattern_type = "structuring"

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
            ring_id = f"RING_STR_{i:05d}"
            num_txns = int(rng.integers(3, 9))  # 3–8 inclusive

            sender_idx = int(rng.integers(0, len(acc)))
            sender = acc.iloc[sender_idx]

            recv_pool = list(range(len(acc)))
            recv_pool.pop(sender_idx)
            recv_indices = rng.choice(recv_pool, size=num_txns, replace=False)
            receivers = [acc.iloc[int(idx)] for idx in recv_indices]

            total_value = float(rng.uniform(
                structuring_threshold * 1.05,
                structuring_threshold * 2.5,
            ))
            # Each split: sub-threshold
            amounts = rng.uniform(
                0.85 * structuring_threshold,
                0.999 * structuring_threshold,
                size=num_txns,
            )
            # Normalise to total but keep sub-threshold
            amounts = np.clip(
                amounts / amounts.sum() * total_value,
                0.85 * structuring_threshold,
                0.999 * structuring_threshold,
            )

            day_offset = int(rng.integers(0, max(1, total_days - 2)))
            window_sec = float(rng.uniform(24.0, 48.0)) * 3600
            base_ts = start_date + timedelta(days=day_offset)
            offsets = np.sort(rng.uniform(0, window_sec, size=num_txns))
            timestamps = [base_ts + timedelta(seconds=float(s)) for s in offsets]

            txns = []
            for recv, amt, ts in zip(receivers, amounts, timestamps):
                txns.append(make_tx_row(
                    sender=sender,
                    receiver=recv,
                    amount=float(amt),
                    timestamp=ts,
                    tx_type="TRANSFER",
                    channel=str(rng.choice(["NEFT", "IMPS", "UPI"])),
                    merchant_cat="TRANSFER",
                    is_fraud=True,
                    ring_id=ring_id,
                    pattern_type=self.pattern_type,
                    severity=_severity(total_value),
                    rng=rng,
                ))

            rings.append(InjectedRing(
                ring_id=ring_id,
                member_accounts=[sender["account_id"]] + [r["account_id"] for r in receivers],
                transactions=txns,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                entry_account=sender["account_id"],
                exit_account=receivers[-1]["account_id"],
                num_hops=1,
                total_value=total_value,
            ))

        return rings
