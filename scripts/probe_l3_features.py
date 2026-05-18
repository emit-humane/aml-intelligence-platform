"""
Probe: WHY does L3 stay inverted / AE-saturated after the history fix?

Compares, per 52-dim feature, the *scaled* value distribution between:
  TRAIN  - the sampled normal rows used to fit the scaler/models
  NORMAL - inference normal stream txns (full history)
  FRAUD  - inference fraud stream txns (full history)

For each dim prints train mean (should be ~0 post-scaling), and the
inference normal/fraud scaled means + the |scaled| magnitude. Dims with
huge |scaled| at inference are the ones blowing up AE reconstruction error
and dominating IsoForest/LOF — that pinpoints the real bug.

Run:
  python -m scripts.probe_l3_features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from system2_platform.shared.s3_artifact_store import ArtifactStore
from system2_platform.shared.s1_feature_engineering import FeatureStore
from system2_platform.layer3_behavioral.l3a_training import (
    _pick_sample_ids, _GRAPH_FEATURE_COLS,
)
from system2_platform.layer3_behavioral.l3b_inference import (
    _BEHAVIORAL_COLS,
)

ART = Path("data/artifacts")
ALL_PARQUET = Path("data/raw/all_transactions.parquet")
GT_CSV = Path("data/raw/hidden_ground_truth.csv")

NAMES = _BEHAVIORAL_COLS + _GRAPH_FEATURE_COLS  # 45 + 7 = 52


def _raw_with_history(history_df: pd.DataFrame, want_ids: set[str],
                      gf_indexed: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    fs = FeatureStore()
    hist = history_df.sort_values("timestamp")
    rawm, sender = [], []
    for row in hist.itertuples(index=False):
        tid = str(getattr(row, "transaction_id", ""))
        if tid in want_ids:
            rawm.append(fs._compute_raw(row))
            sender.append(str(getattr(row, "sender_account", "")))
        fs._ingest_row(row)
    X45 = np.array(rawm, dtype=float)
    X7 = np.zeros((len(sender), len(_GRAPH_FEATURE_COLS)))
    for i, a in enumerate(sender):
        if a in gf_indexed.index:
            X7[i] = gf_indexed.loc[a].values
    return np.hstack([X45, X7]), sender


def main() -> None:
    store = ArtifactStore(ART)
    scaler = store.load("scaler_behavioral")
    gf = store.load("graph_features").set_index("account_id")[_GRAPH_FEATURE_COLS]
    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    print(f"[probe] scaler n_features_in_={getattr(scaler,'n_features_in_','?')}")

    df = pd.read_parquet(ALL_PARQUET)
    gt_ids = set(pd.read_csv(GT_CSV, usecols=["transaction_id"])
                 ["transaction_id"].astype(str))
    df["transaction_id"] = df["transaction_id"].astype(str)

    rng = np.random.default_rng(42)
    normal_df = df[df["is_fraud"] == False].reset_index(drop=True)

    # Same sample the trainer used
    train_ids = _pick_sample_ids(normal_df, 30_000, 20, rng)

    # Inference test ids: all fraud + random 1500 normal NOT in train sample
    fraud_ids = set(df.loc[df["transaction_id"].isin(gt_ids), "transaction_id"])
    norm_pool = list(set(normal_df["transaction_id"]) - train_ids)
    norm_test = set(rng.choice(norm_pool, size=1500, replace=False).tolist())

    want = train_ids | fraud_ids | norm_test
    print(f"[probe] computing raw features (full history) for "
          f"{len(want):,} rows (train={len(train_ids):,}, "
          f"fraud={len(fraud_ids):,}, norm_test={len(norm_test):,})...")

    fs = FeatureStore()
    hist = df.sort_values("timestamp")
    buckets = {"train": [], "norm": [], "fraud": []}
    for row in hist.itertuples(index=False):
        tid = str(getattr(row, "transaction_id", ""))
        if tid in want:
            raw = fs._compute_raw(row)
            acc = str(getattr(row, "sender_account", ""))
            g = gf.loc[acc].values if acc in gf.index else np.zeros(7)
            vec = np.concatenate([raw, g])
            if tid in fraud_ids:
                buckets["fraud"].append(vec)
            elif tid in train_ids:
                buckets["train"].append(vec)
            elif tid in norm_test:
                buckets["norm"].append(vec)
        fs._ingest_row(row)

    Xtr = np.array(buckets["train"], dtype=float)
    Xno = np.array(buckets["norm"], dtype=float)
    Xfr = np.array(buckets["fraud"], dtype=float)
    print(f"[probe] shapes train={Xtr.shape} norm={Xno.shape} fraud={Xfr.shape}\n")

    def scaled(X):
        return (X - mean) / np.where(scale > 0, scale, 1.0)

    Str, Sno, Sfr = scaled(Xtr), scaled(Xno), scaled(Xfr)

    # Per-dim report, sorted by inference |scaled| magnitude
    rows = []
    for j in range(52):
        rows.append((
            j, NAMES[j],
            float(np.mean(np.abs(Str[:, j]))),
            float(np.mean(np.abs(Sno[:, j]))),
            float(np.mean(np.abs(Sfr[:, j]))),
            float(Sno[:, j].mean()),
            float(Sfr[:, j].mean()),
            float(scale[j]),
        ))
    rows.sort(key=lambda r: max(r[3], r[4]), reverse=True)

    print(f"{'dim':<4}{'feature':<26}{'|tr|':>8}{'|no|':>9}"
          f"{'|fr|':>9}{'no.mean':>10}{'fr.mean':>10}{'scale':>12}")
    print("-" * 88)
    for (j, nm, atr, ano, afr, nmean, fmean, sc) in rows:
        flag = "  <== blows up" if max(ano, afr) > 5 else ""
        print(f"{j:<4}{nm:<26}{atr:>8.2f}{ano:>9.2f}{afr:>9.2f}"
              f"{nmean:>10.2f}{fmean:>10.2f}{sc:>12.2g}{flag}")

    # Reconstruction-error proxy: mean squared scaled value (AE input energy)
    print(f"\n[probe] mean ||scaled||^2  train={np.mean(Str**2):.2f}  "
          f"norm={np.mean(Sno**2):.2f}  fraud={np.mean(Sfr**2):.2f}")
    print("[probe] If norm/fraud >> train, the scaler/feature distribution "
          "is shifted -> AE recon saturates for everyone (the bug).")


if __name__ == "__main__":
    main()
