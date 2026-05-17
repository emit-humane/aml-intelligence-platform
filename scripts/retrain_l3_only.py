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
MAX_SAMPLE   = 30_000


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

    # Sample normal transactions — capped at MAX_PER_ACCOUNT per sender to
    # prevent a single hub account dominating and skewing the feature scaler.
    MAX_PER_ACCOUNT = 20
    normal_df = df[df["is_fraud"] == False].reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    capped_parts = []
    for _, grp in normal_df.groupby("sender_account"):
        n = min(len(grp), MAX_PER_ACCOUNT)
        capped_parts.append(grp.sample(n, random_state=SEED))
    capped = pd.concat(capped_parts, ignore_index=True)
    n_sample = min(MAX_SAMPLE, len(capped))
    idx = rng.choice(len(capped), size=n_sample, replace=False)
    sample_df = (
        capped.iloc[idx]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    top5 = sample_df["sender_account"].value_counts().head(3)
    print(f"[retrain_l3] Sampled {len(sample_df):,} normal txns (capped {MAX_PER_ACCOUNT}/acct)")
    print(f"[retrain_l3] Top-3 senders: {top5.to_dict()}")

    t0 = time.time()
    summary = train_models_only(
        sample_df      = sample_df,
        graph_features = graph_features,
        artifact_store = store,
        seed           = SEED,
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
