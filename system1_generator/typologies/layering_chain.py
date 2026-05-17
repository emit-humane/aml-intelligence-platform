"""
Layering Chain typology.

Funds pass through a linear chain of 4–10 intermediary accounts before
reaching the final beneficiary, creating layers of separation between
the source and destination.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row


class LayeringChain(Typology):
    pattern_type = "layering_chain"

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
            ring_id = f"RING_LAY_{i:05d}"
            chain_len = int(rng.integers(4, 11))  # 4–10 intermediaries + source = chain_len+1 nodes

            indices = rng.choice(len(acc), size=chain_len + 1, replace=False)
            chain = [acc.iloc[int(idx)] for idx in indices]

            entry_amount = float(rng.uniform(
                structuring_threshold * 0.8,
                structuring_threshold * 6.0,
            ))
            total_value = entry_amount

            # Spread over 7–30 days
            span_days = int(rng.integers(7, 31))
            day_offset = int(rng.integers(0, max(1, total_days - span_days)))
            base_ts = start_date + timedelta(days=day_offset)
            hop_gap_sec = (span_days * 86400) / chain_len

            txns = []
            current_amount = entry_amount
            for hop in range(chain_len):
                sender = chain[hop]
                receiver = chain[hop + 1]
                ts = base_ts + timedelta(
                    seconds=hop * hop_gap_sec + float(rng.uniform(0, hop_gap_sec * 0.4))
                )
                txns.append(make_tx_row(
                    sender=sender,
                    receiver=receiver,
                    amount=round(current_amount, 2),
                    timestamp=ts,
                    tx_type=str(rng.choice(["TRANSFER", "WIRE"])),
                    channel=str(rng.choice(["NEFT", "RTGS", "SWIFT"])),
                    merchant_cat="TRANSFER",
                    is_fraud=True,
                    ring_id=ring_id,
                    pattern_type=self.pattern_type,
                    severity=_severity(total_value),
                    rng=rng,
                ))
                current_amount *= float(rng.uniform(0.92, 0.99))

            rings.append(InjectedRing(
                ring_id=ring_id,
                member_accounts=[c["account_id"] for c in chain],
                transactions=txns,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                entry_account=chain[0]["account_id"],
                exit_account=chain[-1]["account_id"],
                num_hops=chain_len,
                total_value=total_value,
            ))

        return rings
