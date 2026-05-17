"""
G3 — Scenario Injector.

1. Determines how many fraud rings to generate (n_fraud transactions target).
2. Allocates rings per typology according to scenario_probabilities.
3. Calls each Typology.inject() to produce InjectedRing objects.
4. Flattens all ring transactions into a single DataFrame.
5. Assigns final transaction_ids (TXN_F...) and merges with the normal df.
6. Writes fraud_transactions.parquet and the combined all_transactions.parquet.
Returns (fraud_df, combined_df, rings).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GeneratorConfig
from .typologies import ALL_TYPOLOGIES, InjectedRing

_GEN_START = datetime(2026, 1, 7, tzinfo=timezone.utc)


def _allocate_rings(
    total_rings: int,
    probabilities: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Distribute total_rings across typologies by probability, no zeros."""
    names = list(probabilities.keys())
    probs = np.array([probabilities[n] for n in names], dtype=float)
    probs /= probs.sum()

    counts_raw = (probs * total_rings).astype(int)
    remainder = total_rings - counts_raw.sum()
    # Distribute leftover rings to typologies with highest fractional parts
    fracs = probs * total_rings - counts_raw
    top = np.argsort(fracs)[::-1][:remainder]
    counts_raw[top] += 1

    # Guarantee at least 1 ring per typology
    counts_raw = np.maximum(counts_raw, 1)
    return {n: int(c) for n, c in zip(names, counts_raw)}


def inject_scenarios(
    cfg: GeneratorConfig,
    accounts: pd.DataFrame,
    normal_df: pd.DataFrame,
    rng: np.random.Generator,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[InjectedRing]]:
    """
    Inject fraud rings into the transaction pool.

    Returns (fraud_df, combined_df, all_rings).
    """
    total_txns = cfg.num_historical_transactions + cfg.num_stream_transactions
    n_fraud_target = int(total_txns * cfg.fraud_ratio)
    total_days = cfg.history_days + cfg.stream_days

    # Estimate ~5 transactions per ring on average
    total_rings = max(n_fraud_target // 5, len(cfg.scenario_probabilities))
    allocation = _allocate_rings(total_rings, cfg.scenario_probabilities, rng)

    all_rings: list[InjectedRing] = []
    for pattern_name, num_rings in allocation.items():
        cls = ALL_TYPOLOGIES.get(pattern_name)
        if cls is None:
            print(f"[G3] WARNING: unknown typology '{pattern_name}', skipping")
            continue
        typology = cls()
        rings = typology.inject(
            graph=None,
            accounts=accounts,
            num_rings=num_rings,
            rng=rng,
            start_date=_GEN_START,
            total_days=total_days,
            structuring_threshold=cfg.structuring_threshold,
        )
        all_rings.extend(rings)
        print(f"[G3] {pattern_name}: {num_rings} rings -> {sum(len(r.transactions) for r in rings)} txns")

    # Flatten ring transactions
    all_tx_dicts = []
    for ring in all_rings:
        all_tx_dicts.extend(ring.transactions)

    if not all_tx_dicts:
        fraud_df = pd.DataFrame(columns=normal_df.columns)
        combined_df = normal_df.copy()
        out_dir.mkdir(parents=True, exist_ok=True)
        fraud_df.to_parquet(out_dir / "fraud_transactions.parquet", index=False)
        combined_df.to_parquet(out_dir / "all_transactions.parquet", index=False)
        return fraud_df, combined_df, all_rings

    fraud_df = pd.DataFrame(all_tx_dicts)

    # Assign final transaction IDs
    n_normal = len(normal_df)
    fraud_df["transaction_id"] = [
        f"TXN_F{i:08d}" for i in range(1, len(fraud_df) + 1)
    ]

    # Ensure consistent dtypes with normal_df
    for col in normal_df.columns:
        if col in fraud_df.columns and col != "transaction_id":
            try:
                fraud_df[col] = fraud_df[col].astype(normal_df[col].dtype)
            except Exception:
                pass

    # Add any missing columns from normal schema
    for col in normal_df.columns:
        if col not in fraud_df.columns:
            fraud_df[col] = normal_df[col].iloc[0] if len(normal_df) > 0 else None

    fraud_df = fraud_df[normal_df.columns]

    combined_df = pd.concat([normal_df, fraud_df], ignore_index=True)
    # Sort by timestamp
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fraud_df.to_parquet(out_dir / "fraud_transactions.parquet", index=False)
    combined_df.to_parquet(out_dir / "all_transactions.parquet", index=False)

    total_fraud = len(fraud_df)
    actual_ratio = total_fraud / len(combined_df)
    print(f"[G3] injected {total_fraud:,} fraud txns ({actual_ratio:.3%} of {len(combined_df):,} total)")
    print(f"[G3] wrote fraud_transactions.parquet and all_transactions.parquet -> {out_dir}")

    return fraud_df, combined_df, all_rings
