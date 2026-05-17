"""
G1 — Account Builder.

Produces accounts.parquet with one row per synthetic account.
All randomness flows through the caller-supplied numpy Generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from .config import GeneratorConfig
from .distributions import country_coords_bulk, sample_amounts

# Merchant categories (used across G2/G3 as well)
MERCHANT_CATEGORIES = [
    "TRANSFER", "RETAIL", "UTILITY", "SALARY", "INVESTMENT",
    "INSURANCE", "LOAN_REPAYMENT", "RENT", "FOOD", "TRAVEL",
    "MEDICAL", "EDUCATION", "ENTERTAINMENT", "GOVERNMENT", "OTHER",
]

# KYC level distribution: 0=minimal, 1=basic, 2=standard, 3=full
_KYC_PROBS = np.array([0.05, 0.25, 0.50, 0.20])

# Country distribution — most accounts are Indian
_COUNTRY_WEIGHTS = {
    "IN": 0.75, "US": 0.04, "AE": 0.05, "SG": 0.03,
    "GB": 0.03, "CN": 0.03, "MU": 0.02, "NG": 0.02,
    "PK": 0.02, "CH": 0.01,
}


def build_accounts(cfg: GeneratorConfig, rng: np.random.Generator, out_dir: Path) -> pd.DataFrame:
    """
    Build the accounts DataFrame and write accounts.parquet.

    Returns the DataFrame so downstream modules can use it directly.
    """
    fake = Faker()
    fake.seed_instance(int(rng.integers(0, 2**31)))

    n = cfg.num_accounts
    countries_list = list(_COUNTRY_WEIGHTS.keys())
    country_probs = np.array([_COUNTRY_WEIGHTS[c] for c in countries_list], dtype=float)
    country_probs /= country_probs.sum()

    # --- IDs ---
    account_ids = [f"ACC{i:06d}" for i in range(1, n + 1)]
    device_ids = [f"DEV{i:06d}" for i in range(1, n + 1)]

    # --- Bank, country, KYC ---
    banks = rng.choice(cfg.banks, size=n)
    countries = rng.choice(countries_list, size=n, p=country_probs)
    kyc_levels = rng.choice([0, 1, 2, 3], size=n, p=_KYC_PROBS)

    # --- Geo coordinates ---
    lats, lons = country_coords_bulk(list(countries), rng, jitter=3.0)

    # --- Balances (initial) ---
    # Lognormal centred at ~500K INR
    init_balances = sample_amounts(rng, n, mean=500_000, std=1_000_000, min_amount=1_000, max_amount=50_000_000)

    # --- Account age in days (1–2000 days; some accounts are old) ---
    account_age_days = rng.integers(30, 2000, size=n)

    # --- Names and IPs (Faker) ---
    names = [fake.name() for _ in range(n)]
    ips = [fake.ipv4_private() for _ in range(n)]

    # --- Activity weight (Zipf-like: a few hubs, many low-activity) ---
    # Used by G2 to pick senders/receivers
    activity_weights = rng.zipf(1.5, size=n).astype(float)
    activity_weights /= activity_weights.sum()

    df = pd.DataFrame({
        "account_id": account_ids,
        "account_name": names,
        "bank": banks,
        "country": countries,
        "kyc_level": kyc_levels.astype(np.int8),
        "initial_balance": init_balances,
        "device_id": device_ids,
        "ip_address": ips,
        "geo_latitude": lats,
        "geo_longitude": lons,
        "account_age_days": account_age_days.astype(np.int16),
        "activity_weight": activity_weights,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "accounts.parquet", index=False)
    print(f"[G1] wrote {len(df)} accounts -> {out_dir / 'accounts.parquet'}")
    return df
