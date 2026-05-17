"""
Round-Tripping typology.

Funds leave an account, pass through 2–4 intermediaries (often cross-border),
and return to the same or a closely affiliated account — giving the appearance
of legitimate foreign investment or trade income.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import InjectedRing, Typology, _severity, make_tx_row


class RoundTripping(Typology):
    pattern_type = "round_tripping"

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
            ring_id = f"RING_RTP_{i:05d}"
            num_intermediaries = int(rng.integers(2, 5))
            total_nodes = num_intermediaries + 2  # origin + intermediaries + return

            indices = rng.choice(len(acc), size=total_nodes, replace=False)
            origin = acc.iloc[int(indices[0])]
            intermediaries = [acc.iloc[int(idx)] for idx in indices[1:-1]]
            destination = acc.iloc[int(indices[-1])]  # affiliated / same beneficial owner

            entry_amount = float(rng.uniform(
                structuring_threshold * 1.0,
                structuring_threshold * 5.0,
            ))
            total_value = entry_amount

            # Full round trip spans 14–60 days
            span_days = int(rng.integers(14, 61))
            day_offset = int(rng.integers(0, max(1, total_days - span_days)))
            base_ts = start_date + timedelta(days=day_offset)

            chain = [origin] + intermediaries + [destination]
            hop_gap_sec = (span_days * 86400) / (len(chain) - 1)

            txns = []
            current_amount = entry_amount
            for hop in range(len(chain) - 1):
                ts = base_ts + timedelta(
                    seconds=hop * hop_gap_sec + float(rng.uniform(0, hop_gap_sec * 0.2))
                )
                # Final hop: return close to original amount (investment "return")
                if hop == len(chain) - 2:
                    current_amount *= float(rng.uniform(0.95, 1.15))  # slight markup
                txns.append(make_tx_row(
                    sender=chain[hop],
                    receiver=chain[hop + 1],
                    amount=round(current_amount, 2),
                    timestamp=ts,
                    tx_type=str(rng.choice(["WIRE", "TRANSFER"])),
                    channel=str(rng.choice(["SWIFT", "RTGS", "NEFT"])),
                    merchant_cat=str(rng.choice(["INVESTMENT", "TRANSFER"])),
                    is_fraud=True,
                    ring_id=ring_id,
                    pattern_type=self.pattern_type,
                    severity=_severity(total_value),
                    rng=rng,
                ))
                if hop < len(chain) - 2:
                    current_amount *= float(rng.uniform(0.90, 0.98))

            rings.append(InjectedRing(
                ring_id=ring_id,
                member_accounts=[c["account_id"] for c in chain],
                transactions=txns,
                pattern_type=self.pattern_type,
                severity=_severity(total_value),
                entry_account=origin["account_id"],
                exit_account=destination["account_id"],
                num_hops=len(chain) - 1,
                total_value=total_value,
            ))

        return rings
