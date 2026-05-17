"""
Circular Laundering typology.

Money travels through a closed loop of 3–7 accounts and returns to origin,
obscuring the paper trail. Each hop transfers 90–98 % of the received amount
(simulating fees/skimming). Transactions span 3–14 days.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row


class CircularLaundering(Typology):
    pattern_type = "circular_laundering"

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
            ring_id = f"RING_CIR_{i:05d}"
            loop_size = int(rng.integers(3, 8))  # 3–7 hops + return = loop_size+1 txns

            indices = rng.choice(len(acc), size=loop_size, replace=False)
            members = [acc.iloc[int(idx)] for idx in indices]

            entry_amount = float(rng.uniform(
                structuring_threshold * 0.5,
                structuring_threshold * 4.0,
            ))
            total_value = entry_amount

            # Spread over 3–14 days
            span_days = int(rng.integers(3, 15))
            day_offset = int(rng.integers(0, max(1, total_days - span_days)))
            base_ts = start_date + timedelta(days=day_offset)
            hop_gap_sec = (span_days * 86400) / (loop_size + 1)

            txns = []
            current_amount = entry_amount
            for hop in range(loop_size):
                sender = members[hop]
                receiver = members[(hop + 1) % loop_size]
                ts = base_ts + timedelta(seconds=hop * hop_gap_sec + float(rng.uniform(0, hop_gap_sec * 0.5)))
                txns.append(make_tx_row(
                    sender=sender,
                    receiver=receiver,
                    amount=round(current_amount, 2),
                    timestamp=ts,
                    tx_type="TRANSFER",
                    channel=str(rng.choice(["NEFT", "RTGS", "WIRE"])),
                    merchant_cat="TRANSFER",
                    is_fraud=True,
                    ring_id=ring_id,
                    pattern_type=self.pattern_type,
                    severity=_severity(total_value),
                    rng=rng,
                ))
                current_amount *= float(rng.uniform(0.90, 0.98))

            # Closing hop — back to origin
            ts_final = base_ts + timedelta(seconds=loop_size * hop_gap_sec)
            txns.append(make_tx_row(
                sender=members[-1],
                receiver=members[0],
                amount=round(current_amount, 2),
                timestamp=ts_final,
                tx_type="TRANSFER",
                channel=str(rng.choice(["NEFT", "RTGS", "WIRE"])),
                merchant_cat="TRANSFER",
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
                entry_account=members[0]["account_id"],
                exit_account=members[0]["account_id"],
                num_hops=loop_size + 1,
                total_value=total_value,
            ))

        return rings
