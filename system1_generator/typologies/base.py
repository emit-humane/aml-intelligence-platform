"""
Base class for all laundering typologies.

Every typology subclasses Typology and implements:
    inject(graph, accounts, num_rings, rng) -> list[InjectedRing]

The returned InjectedRing.transactions are plain dicts with *all* CSV columns
pre-filled.  transaction_id is left as "" — G3 assigns final IDs.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from ..distributions import leading_digit, country_coords

# ---------------------------------------------------------------------------
# InjectedRing data container
# ---------------------------------------------------------------------------

@dataclass
class InjectedRing:
    ring_id: str
    member_accounts: list[str]
    transactions: list[dict]       # full row dicts; transaction_id = ""
    pattern_type: str
    severity: str                  # Low / Medium / High / Critical
    entry_account: str
    exit_account: str
    num_hops: int
    total_value: float


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

def _severity(total_value: float) -> str:
    if total_value >= 5_000_000:
        return "Critical"
    if total_value >= 1_000_000:
        return "High"
    if total_value >= 200_000:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Row builder helper
# ---------------------------------------------------------------------------

def make_tx_row(
    sender: pd.Series,
    receiver: pd.Series,
    amount: float,
    timestamp: datetime,
    tx_type: str,
    channel: str,
    merchant_cat: str,
    is_fraud: bool,
    ring_id: str,
    pattern_type: str,
    severity: str,
    rng: np.random.Generator,
) -> dict:
    """
    Assemble a single transaction row dict matching the CSV schema.
    transaction_id is left as "" — assigned by G3.
    """
    is_intl = sender["country"] != receiver["country"]
    currency = "INR" if sender["country"] == "IN" else "USD"
    scale = float(rng.uniform(1.5, 6.0))
    s_bal_before = round(max(amount * scale, amount + 10_000.0), 2)
    r_bal_before = round(float(rng.lognormal(12.0, 1.5)), 2)

    return {
        "transaction_id": "",
        "timestamp": timestamp,
        "sender_account": sender["account_id"],
        "receiver_account": receiver["account_id"],
        "sender_name": sender["account_name"],
        "receiver_name": receiver["account_name"],
        "sender_bank": sender["bank"],
        "receiver_bank": receiver["bank"],
        "sender_country": sender["country"],
        "receiver_country": receiver["country"],
        "amount": round(amount, 2),
        "currency": currency,
        "transaction_type": tx_type,
        "payment_channel": channel,
        "device_id": sender["device_id"],
        "ip_address": sender["ip_address"],
        "geo_latitude": sender["geo_latitude"],
        "geo_longitude": sender["geo_longitude"],
        "merchant_category": merchant_cat,
        "transaction_status": "Success",
        "sender_balance_before": s_bal_before,
        "sender_balance_after": round(s_bal_before - amount, 2),
        "receiver_balance_before": r_bal_before,
        "receiver_balance_after": round(r_bal_before + amount, 2),
        "kyc_level": int(sender["kyc_level"]),
        "is_international": bool(is_intl),
        "remarks": "",
        "amount_leading_digit": leading_digit(amount),
        "is_fraud": is_fraud,
        "fraud_ring_id": ring_id,
        "synthetic_pattern_type": pattern_type,
        "scenario_severity": severity,
    }


# ---------------------------------------------------------------------------
# Typology ABC
# ---------------------------------------------------------------------------

class Typology(abc.ABC):
    pattern_type: str = ""

    @abc.abstractmethod
    def inject(
        self,
        graph,                        # networkx.MultiDiGraph or None
        accounts: pd.DataFrame,
        num_rings: int,
        rng: np.random.Generator,
        start_date: datetime,
        total_days: int,
        structuring_threshold: float,
    ) -> list[InjectedRing]:
        """Return a list of InjectedRing (len == num_rings)."""
