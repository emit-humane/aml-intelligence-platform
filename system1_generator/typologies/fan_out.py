"""
Fan-Out (Dispersion) typology.

A single source disperses funds to 3–10 beneficiary accounts within a
short window (12–72 hours), splitting proceeds after integration.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row


class FanOut(Typology):
    pattern_type = "fan_out"

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
            ring_id = f"RING_FOU_{i:05d}"
            num_targets = int(rng.integers(3, 11))

            indices = rng.choice(len(acc), size=num_targets + 1, replace=False)
            source = acc.iloc[int(indices[0])]
            targets = [acc.iloc[int(idx)] for idx in indices[1:]]

            total_value = float(rng.uniform(
                structuring_threshold * 0.5,
                structuring_threshold * 5.0,
            ))
            amounts = rng.dirichlet(np.ones(num_targets)) * total_value

            window_sec = float(rng.uniform(12.0, 72.0)) * 3600
            day_offset = int(rng.integers(0, max(1, total_days - 3)))
            base_ts = start_date + timedelta(days=day_offset)
            offsets = np.sort(rng.uniform(0, window_sec, size=num_targets))
            timestamps = [base_ts + timedelta(seconds=float(s)) for s in offsets]

            txns = []
            for tgt, amt, ts in zip(targets, amounts, timestamps):
                txns.append(make_tx_row(
                    sender=source,
                    receiver=tgt,
                    amount=round(float(amt), 2),
                    timestamp=ts,
                    tx_type="TRANSFER",
                    channel=str(rng.choice(["UPI", "IMPS", "NEFT"])),
                    merchant_cat="TRANSFER",
                    is_fraud=True,
                    ring_id=ring_id,
                    pattern_type=self.pattern_type,
                    severity=_severity(total_value),
                    rng=rng,
                ))

            rings.append(InjectedRing(
                ring_id=ring_id,
                member_accounts=[source["account_id"]] + [t["account_id"] for t in targets],
                transactions=txns,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                entry_account=source["account_id"],
                exit_account=targets[-1]["account_id"],
                num_hops=1,
                total_value=total_value,
            ))

        return rings
