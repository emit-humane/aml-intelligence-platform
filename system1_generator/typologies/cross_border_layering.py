"""
Cross-Border Layering typology.

Funds move through 3–6 accounts across at least 2 high-risk jurisdictions,
with each hop converting or routing through an international transfer.
Uses accounts from high-risk countries (AE, NG, PK, MU, CN).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row

_HIGH_RISK = {"AE", "NG", "PK", "MU", "CN"}


class CrossBorderLayering(Typology):
    pattern_type = "cross_border_layering"

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

        hr_mask = accounts["country"].isin(_HIGH_RISK)
        hr_pool = accounts[hr_mask].reset_index(drop=True)
        all_pool = accounts.reset_index(drop=True)

        # Fall back to any account if high-risk pool is thin
        if len(hr_pool) < 4:
            hr_pool = all_pool

        for i in range(num_rings):
            ring_id = f"RING_CBL_{i:05d}"
            chain_len = int(rng.integers(3, 7))

            # Guarantee at least 2 nodes from high-risk pool
            hr_count = int(rng.integers(2, min(chain_len, len(hr_pool)) + 1))
            hr_indices = rng.choice(len(hr_pool), size=hr_count, replace=False)
            hr_members = [hr_pool.iloc[int(idx)] for idx in hr_indices]

            normal_count = chain_len - hr_count
            remaining_ids = set(all_pool["account_id"]) - {m["account_id"] for m in hr_members}
            remaining = all_pool[all_pool["account_id"].isin(remaining_ids)].reset_index(drop=True)
            if len(remaining) < normal_count:
                normal_count = len(remaining)
            norm_indices = rng.choice(len(remaining), size=normal_count, replace=False)
            norm_members = [remaining.iloc[int(idx)] for idx in norm_indices]

            chain = hr_members[:1] + norm_members + hr_members[1:]
            if len(chain) < 2:
                continue

            entry_amount = float(rng.uniform(
                structuring_threshold * 1.0,
                structuring_threshold * 8.0,
            ))
            total_value = entry_amount

            span_days = int(rng.integers(5, 21))
            day_offset = int(rng.integers(0, max(1, total_days - span_days)))
            base_ts = start_date + timedelta(days=day_offset)
            hop_gap_sec = (span_days * 86400) / (len(chain) - 1)

            txns = []
            current_amount = entry_amount
            for hop in range(len(chain) - 1):
                ts = base_ts + timedelta(
                    seconds=hop * hop_gap_sec + float(rng.uniform(0, hop_gap_sec * 0.3))
                )
                txns.append(make_tx_row(
                    sender=chain[hop],
                    receiver=chain[hop + 1],
                    amount=round(current_amount, 2),
                    timestamp=ts,
                    tx_type="WIRE",
                    channel="SWIFT",
                    merchant_cat="TRANSFER",
                    is_fraud=True,
                    ring_id=ring_id,
                    pattern_type=self.pattern_type,
                    severity=_severity(total_value),
                    rng=rng,
                ))
                current_amount *= float(rng.uniform(0.88, 0.97))

            rings.append(InjectedRing(
                ring_id=ring_id,
                member_accounts=[c["account_id"] for c in chain],
                transactions=txns,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                entry_account=chain[0]["account_id"],
                exit_account=chain[-1]["account_id"],
                num_hops=len(chain) - 1,
                total_value=total_value,
            ))

        return rings
