"""
Velocity Burst typology.

An account sends an unusually high number of transactions (10–30) within a
very short window (1–4 hours), far exceeding its normal velocity profile.
Individual amounts are modest but aggregate volume is suspicious.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row
from ..distributions import sample_structuring_amounts


class VelocityBurst(Typology):
    pattern_type = "velocity_burst"

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
            ring_id = f"RING_VEL_{i:05d}"
            num_txns = int(rng.integers(10, 31))

            sender_idx = int(rng.integers(0, len(acc)))
            sender = acc.iloc[sender_idx]

            recv_pool = [j for j in range(len(acc)) if j != sender_idx]
            recv_indices = rng.choice(recv_pool, size=num_txns, replace=True)
            receivers = [acc.iloc[int(idx)] for idx in recv_indices]

            # Sub-threshold individual amounts; high aggregate
            amounts = sample_structuring_amounts(rng, num_txns, structuring_threshold)
            total_value = float(amounts.sum())

            window_sec = float(rng.uniform(1.0, 4.0)) * 3600
            day_offset = int(rng.integers(0, max(1, total_days - 1)))
            base_ts = start_date + timedelta(days=day_offset)
            offsets = np.sort(rng.uniform(0, window_sec, size=num_txns))
            timestamps = [base_ts + timedelta(seconds=float(s)) for s in offsets]

            txns = []
            for recv, amt, ts in zip(receivers, amounts, timestamps):
                txns.append(make_tx_row(
                    sender=sender,
                    receiver=recv,
                    amount=round(float(amt), 2),
                    timestamp=ts,
                    tx_type="TRANSFER",
                    channel=str(rng.choice(["UPI", "IMPS"])),
                    merchant_cat="TRANSFER",
                    is_fraud=True,
                    ring_id=ring_id,
                    pattern_type=self.pattern_type,
                    severity=_severity(total_value),
                    rng=rng,
                ))

            rings.append(InjectedRing(
                ring_id=ring_id,
                member_accounts=[sender["account_id"]] + list({r["account_id"] for r in receivers}),
                transactions=txns,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                entry_account=sender["account_id"],
                exit_account=receivers[-1]["account_id"],
                num_hops=1,
                total_value=total_value,
            ))

        return rings
