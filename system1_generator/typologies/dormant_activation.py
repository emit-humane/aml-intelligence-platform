"""
Dormant Account Activation typology.

An account that has been inactive for a long period suddenly receives a large
inflow and immediately disperses it to 2–5 accounts within 6–24 hours,
suggesting takeover or shell activation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row


class DormantActivation(Typology):
    pattern_type = "dormant_activation"

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

        # Prefer old accounts (age > 365 days) for the dormant role
        old_mask = accounts["account_age_days"] > 365
        old_pool = accounts[old_mask].reset_index(drop=True)
        all_pool = accounts.reset_index(drop=True)

        if len(old_pool) < 5:
            old_pool = all_pool

        for i in range(num_rings):
            ring_id = f"RING_DRM_{i:05d}"
            num_dispersals = int(rng.integers(2, 6))

            dormant_idx = int(rng.integers(0, len(old_pool)))
            dormant = old_pool.iloc[dormant_idx]

            # Funder: any account
            funder_idx = int(rng.integers(0, len(all_pool)))
            while all_pool.iloc[funder_idx]["account_id"] == dormant["account_id"]:
                funder_idx = int(rng.integers(0, len(all_pool)))
            funder = all_pool.iloc[funder_idx]

            recv_indices = rng.choice(len(all_pool), size=num_dispersals, replace=False)
            receivers = [all_pool.iloc[int(idx)] for idx in recv_indices]

            entry_amount = float(rng.uniform(
                structuring_threshold * 0.5,
                structuring_threshold * 4.0,
            ))
            total_value = entry_amount

            day_offset = int(rng.integers(0, max(1, total_days - 2)))
            base_ts = start_date + timedelta(days=day_offset)

            # Step 1: funder -> dormant
            txns = [make_tx_row(
                sender=funder,
                receiver=dormant,
                amount=round(entry_amount, 2),
                timestamp=base_ts,
                tx_type="TRANSFER",
                channel=str(rng.choice(["NEFT", "RTGS"])),
                merchant_cat="TRANSFER",
                is_fraud=True,
                ring_id=ring_id,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                rng=rng,
            )]

            # Step 2: dormant disperses within 6–24 h
            window_sec = float(rng.uniform(6.0, 24.0)) * 3600
            offsets = np.sort(rng.uniform(3600, window_sec, size=num_dispersals))
            amounts = rng.dirichlet(np.ones(num_dispersals)) * entry_amount * 0.95

            for recv, amt, off in zip(receivers, amounts, offsets):
                txns.append(make_tx_row(
                    sender=dormant,
                    receiver=recv,
                    amount=round(float(amt), 2),
                    timestamp=base_ts + timedelta(seconds=float(off)),
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
                member_accounts=[funder["account_id"], dormant["account_id"]] + [r["account_id"] for r in receivers],
                transactions=txns,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                entry_account=funder["account_id"],
                exit_account=receivers[-1]["account_id"],
                num_hops=2,
                total_value=total_value,
            ))

        return rings
