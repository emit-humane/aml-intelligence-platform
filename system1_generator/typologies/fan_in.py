"""
Fan-In (Aggregation) typology.

Multiple source accounts (3–8) funnel money into a single collector account
over a short window (12–72 hours), aggregating funds before onward transfer.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row


class FanIn(Typology):
    pattern_type = "fan_in"

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
            ring_id = f"RING_FIN_{i:05d}"
            num_sources = int(rng.integers(3, 9))

            indices = rng.choice(len(acc), size=num_sources + 1, replace=False)
            sources = [acc.iloc[int(idx)] for idx in indices[:num_sources]]
            collector = acc.iloc[int(indices[num_sources])]

            total_value = float(rng.uniform(
                structuring_threshold * 0.5,
                structuring_threshold * 5.0,
            ))
            amounts = rng.dirichlet(np.ones(num_sources)) * total_value

            window_sec = float(rng.uniform(12.0, 72.0)) * 3600
            day_offset = int(rng.integers(0, max(1, total_days - 3)))
            base_ts = start_date + timedelta(days=day_offset)
            offsets = np.sort(rng.uniform(0, window_sec, size=num_sources))
            timestamps = [base_ts + timedelta(seconds=float(s)) for s in offsets]

            txns = []
            for src, amt, ts in zip(sources, amounts, timestamps):
                txns.append(make_tx_row(
                    sender=src,
                    receiver=collector,
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
                member_accounts=[s["account_id"] for s in sources] + [collector["account_id"]],
                transactions=txns,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                entry_account=sources[0]["account_id"],
                exit_account=collector["account_id"],
                num_hops=1,
                total_value=total_value,
            ))

        return rings
