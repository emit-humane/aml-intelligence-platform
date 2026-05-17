"""
G2 — Normal Transaction Generator.

Produces normal_transactions.parquet: benign transactions distributed over the
full generation window (history_days + stream_days).  No fraud injected here.
All randomness flows through the caller-supplied numpy Generator.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from .config import GeneratorConfig
from .distributions import (
    leading_digit,
    make_timestamps,
    sample_amounts,
    sample_structuring_amounts,
)
from .g1_account_builder import MERCHANT_CATEGORIES

# Reference start date: 100 days before a fixed anchor (deterministic)
_GEN_START = datetime(2026, 1, 7, tzinfo=timezone.utc)

# ~15 % of high-value normal transactions are near the structuring threshold
_STRUCTURING_NEAR_RATIO = 0.05


def _pick_pairs(
    rng: np.random.Generator,
    n: int,
    account_ids: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick (sender, receiver) pairs with power-law bias (weights) ensuring sender != receiver.
    Returns parallel arrays of indices into account_ids.
    """
    senders_idx = rng.choice(len(account_ids), size=n, p=weights)
    receivers_idx = rng.choice(len(account_ids), size=n, p=weights)
    # Fix same-party transactions (iterate only the collisions — usually < 0.1%)
    mask = senders_idx == receivers_idx
    while mask.any():
        receivers_idx[mask] = rng.choice(len(account_ids), size=mask.sum(), p=weights)
        mask = senders_idx == receivers_idx
    return senders_idx, receivers_idx


def build_normal_transactions(
    cfg: GeneratorConfig,
    accounts: pd.DataFrame,
    rng: np.random.Generator,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Build normal_transactions.parquet and return the DataFrame.
    n = (num_historical + num_stream) * (1 - fraud_ratio), rounded up.
    """
    total_days = cfg.history_days + cfg.stream_days
    n_total = cfg.num_historical_transactions + cfg.num_stream_transactions
    n_normal = int(n_total * (1.0 - cfg.fraud_ratio))

    fake = Faker()
    fake.seed_instance(int(rng.integers(0, 2**31)))

    acc_ids = accounts["account_id"].to_numpy()
    weights = accounts["activity_weight"].to_numpy().astype(float)
    weights /= weights.sum()

    acc_lookup = accounts.set_index("account_id")

    # --- Timestamps ---
    timestamps = make_timestamps(rng, _GEN_START, total_days, n_normal)

    # --- Amounts ---
    n_struct = int(n_normal * _STRUCTURING_NEAR_RATIO)
    n_regular = n_normal - n_struct
    amounts_regular = sample_amounts(
        rng, n_regular,
        cfg.amount_distribution.normal_mean,
        cfg.amount_distribution.normal_std,
        cfg.amount_distribution.min,
        cfg.amount_distribution.max,
    )
    amounts_struct = sample_structuring_amounts(rng, n_struct, cfg.structuring_threshold)
    amounts = np.concatenate([amounts_regular, amounts_struct])
    # Shuffle so structuring amounts aren't all at the end
    shuffle_idx = rng.permutation(n_normal)
    amounts = amounts[shuffle_idx]
    timestamps_sorted_idxs = np.argsort([t.timestamp() for t in timestamps])

    # --- Account pairs ---
    senders_idx, receivers_idx = _pick_pairs(rng, n_normal, acc_ids, weights)

    # --- Transaction metadata ---
    tx_types = rng.choice(cfg.transaction_types, size=n_normal)
    channels = rng.choice(cfg.payment_channels, size=n_normal)
    merch_cats = rng.choice(MERCHANT_CATEGORIES, size=n_normal)
    statuses = rng.choice(
        ["Success", "Failed", "Pending"],
        size=n_normal,
        p=[0.93, 0.04, 0.03],
    )

    # --- Build row-by-row using vectorised attribute lookups ---
    sender_accs = acc_ids[senders_idx]
    receiver_accs = acc_ids[receivers_idx]

    # Pull account attributes vectorised via merge
    sender_info = accounts.iloc[senders_idx].reset_index(drop=True)
    receiver_info = accounts.iloc[receivers_idx].reset_index(drop=True)

    # Balances: synthetic but plausible
    scale = rng.uniform(1.5, 6.0, size=n_normal)
    sender_bal_before = np.maximum(amounts * scale, amounts + 10_000.0)
    sender_bal_after = sender_bal_before - amounts
    recv_bal_before = rng.lognormal(12.0, 1.5, size=n_normal)
    recv_bal_before = np.clip(recv_bal_before, 500.0, 50_000_000.0)
    recv_bal_after = recv_bal_before + amounts

    is_international = (sender_info["country"].to_numpy() != receiver_info["country"].to_numpy())
    currencies = np.where(sender_info["country"].to_numpy() == "IN", "INR", "USD")

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:010d}" for i in range(1, n_normal + 1)],
        "timestamp": timestamps,
        "sender_account": sender_accs,
        "receiver_account": receiver_accs,
        "sender_name": sender_info["account_name"].to_numpy(),
        "receiver_name": receiver_info["account_name"].to_numpy(),
        "sender_bank": sender_info["bank"].to_numpy(),
        "receiver_bank": receiver_info["bank"].to_numpy(),
        "sender_country": sender_info["country"].to_numpy(),
        "receiver_country": receiver_info["country"].to_numpy(),
        "amount": np.round(amounts, 2),
        "currency": currencies,
        "transaction_type": tx_types,
        "payment_channel": channels,
        "device_id": sender_info["device_id"].to_numpy(),
        "ip_address": sender_info["ip_address"].to_numpy(),
        "geo_latitude": sender_info["geo_latitude"].to_numpy(),
        "geo_longitude": sender_info["geo_longitude"].to_numpy(),
        "merchant_category": merch_cats,
        "transaction_status": statuses,
        "sender_balance_before": np.round(sender_bal_before, 2),
        "sender_balance_after": np.round(sender_bal_after, 2),
        "receiver_balance_before": np.round(recv_bal_before, 2),
        "receiver_balance_after": np.round(recv_bal_after, 2),
        "kyc_level": sender_info["kyc_level"].to_numpy(),
        "is_international": is_international.astype(bool),
        "remarks": "",
        "amount_leading_digit": [leading_digit(a) for a in amounts],
        "is_fraud": False,
        "fraud_ring_id": "",
        "synthetic_pattern_type": "",
        "scenario_severity": "",
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "normal_transactions.parquet", index=False)
    print(f"[G2] wrote {len(df):,} normal transactions -> {out_dir / 'normal_transactions.parquet'}")
    return df
