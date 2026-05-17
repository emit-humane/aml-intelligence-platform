"""
G4 — Splitter & Exporter.

Splits all_transactions into:
  - historical_transactions.csv  (first history_days days)
  - stream_transactions.csv      (last stream_days days, no fraud labels)
  - hidden_ground_truth.csv      (stream fraud rows with labels, for evaluation)

All three files are written to out_dir (data/raw/).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import GeneratorConfig
from .typologies import InjectedRing

_GEN_START = datetime(2026, 1, 7, tzinfo=timezone.utc)

# Columns to STRIP from stream (hide labels from live system)
_HIDDEN_COLS = ["is_fraud", "fraud_ring_id", "synthetic_pattern_type", "scenario_severity"]


def export_splits(
    cfg: GeneratorConfig,
    combined_df: pd.DataFrame,
    all_rings: list[InjectedRing],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split combined_df by date and export CSV files.

    Returns (historical_df, stream_df_public, ground_truth_df).
    """
    cutoff = _GEN_START + timedelta(days=cfg.history_days)
    stream_end = _GEN_START + timedelta(days=cfg.history_days + cfg.stream_days)

    timestamps = pd.to_datetime(combined_df["timestamp"], utc=True)

    hist_mask = timestamps < cutoff
    stream_mask = (timestamps >= cutoff) & (timestamps < stream_end)

    historical_df = combined_df[hist_mask].copy()
    stream_full_df = combined_df[stream_mask].copy()

    # Public stream: strip fraud labels
    stream_public_df = stream_full_df.drop(columns=_HIDDEN_COLS, errors="ignore")

    # Ground truth: only fraud rows from the stream window
    ground_truth_df = stream_full_df[stream_full_df["is_fraud"] == True].copy()  # noqa: E712

    out_dir.mkdir(parents=True, exist_ok=True)
    historical_df.to_csv(out_dir / "historical_transactions.csv", index=False)
    stream_public_df.to_csv(out_dir / "stream_transactions.csv", index=False)
    ground_truth_df.to_csv(out_dir / "hidden_ground_truth.csv", index=False)

    n_hist_fraud = int(historical_df["is_fraud"].sum())
    n_stream_fraud = int(stream_full_df["is_fraud"].sum())
    print(
        f"[G4] historical: {len(historical_df):,} txns "
        f"({n_hist_fraud} fraud, {len(historical_df) - n_hist_fraud} normal)"
    )
    print(
        f"[G4] stream:     {len(stream_full_df):,} txns "
        f"({n_stream_fraud} fraud hidden in ground truth)"
    )
    print(f"[G4] ground truth: {len(ground_truth_df)} fraud events")
    print(f"[G4] wrote CSVs -> {out_dir}")

    return historical_df, stream_public_df, ground_truth_df
