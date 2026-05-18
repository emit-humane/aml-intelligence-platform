"""
Retrain L3A behavioral models only (IsoForest + LOF + AutoEncoder).
Loads existing graph_features artifact so S2/L2A/L2B are skipped.

Usage:
    python -m scripts.retrain_l3_only
    .venv\Scripts\python.exe -m scripts.retrain_l3_only
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_DIR = Path("data/artifacts")
DATA_PATH    = Path("data/raw/all_transactions.parquet")
SEED         = 42
MAX_SAMPLE   = 20_000  # uniform-by-transaction sample


def main() -> None:
    from system2_platform.shared.s3_artifact_store import ArtifactStore
    from system2_platform.layer3_behavioral.l3a_training import train_models_only

    store = ArtifactStore(ARTIFACT_DIR)

    print("[retrain_l3] Loading graph_features from artifacts...")
    graph_features = store.load("graph_features")
    print(f"[retrain_l3] graph_features shape: {graph_features.shape}")

    print(f"[retrain_l3] Loading transactions from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"[retrain_l3] {len(df):,} transactions, {df['sender_account'].nunique():,} accounts")

    # Pass the FULL stream. train_models_only replays every row in timestamp
    # order to build true per-account rolling history, then fits on a
    # per-account-capped sample of NORMAL transactions whose features were
    # computed WITH that real history. Pre-capping before the replay (the old
    # behaviour) gave every account near-empty history and inverted L3.
    t0 = time.time()
    summary = train_models_only(
        history_df       = df,
        graph_features   = graph_features,
        artifact_store   = store,
        seed             = SEED,
        feature_sample_n = MAX_SAMPLE,
        max_per_account  = None,   # uniform-by-transaction (see _pick_sample_ids)
    )
    elapsed = time.time() - t0

    print(f"\n[retrain_l3] DONE in {elapsed:.0f}s")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Update manifest
    store.save_manifest()
    print("[retrain_l3] Manifest updated.")


if __name__ == "__main__":
    main()
