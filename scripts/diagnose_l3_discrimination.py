"""
Diagnostic: does Layer 3 (behavioral anomaly) discriminate fraud vs normal?

Hypothesis under test
---------------------
The production orchestrator seeds the GRAPH updater from historical data
(s6_graph_update.seed_from_historical) but NOT the FEATURE store
(s5_feature_update.FeatureUpdater starts with an empty FeatureStore).

With no rolling-window history, every account looks brand-new at inference:
  tx_velocity_*        = 0
  avg_amount_7d        = 0
  std_amount_7d        = 0
  beneficiary_count_*  = 0
  amount_zscore        = 0
  tx_gap_seconds       = 86400  (brand-new-account default)

The scaler_behavioral was fit on transactions that DID have real history,
so these zeros become extreme z-scores -> IsoForest / LOF / Autoencoder all
see out-of-distribution inputs -> every transaction scores high, fraud and
normal alike (no discrimination).

This script proves it by scoring the SAME test set two ways:

  Scenario A  NO-SEED   empty FeatureStore  (current production behaviour)
  Scenario B  SEEDED    FeatureStore warmed up from historical_transactions.csv
                        (the proposed fix)

Test set:
  fraud  = all rows in data/raw/hidden_ground_truth.csv  (418)
  normal = random sample of stream rows NOT in ground truth (default 2000)

For each scenario it reports, per behavioral sub-score
(iso / lof / autoencoder / ensemble):
  mean(fraud)  mean(normal)  AUROC

A working detector has AUROC well above 0.5 and mean(fraud) > mean(normal).
The no-seed scenario is expected to collapse (AUROC ~0.5, both means high).

Run:
  python -m scripts.diagnose_l3_discrimination
  python -m scripts.diagnose_l3_discrimination --normal-sample 4000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from system2_platform.layer3_behavioral.l3b_inference import BehavioralInferenceEngine
from system2_platform.shared.s5_feature_update import FeatureUpdater
from system2_platform.shared.s1_feature_engineering import FeatureStore

HIST_CSV   = Path("data/raw/historical_transactions.csv")
STREAM_CSV = Path("data/raw/stream_transactions.csv")
GT_CSV     = Path("data/raw/hidden_ground_truth.csv")
ARTIFACTS  = Path("data/artifacts")


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC without sklearn dependency (rank-sum / Mann-Whitney)."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sum_rank = np.zeros(len(counts))
    np.add.at(sum_rank, inv, ranks)
    avg_rank = sum_rank / counts
    ranks = avg_rank[inv]
    r_pos = ranks[y_true == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _run_scenario(
    name: str,
    seed_history: bool,
    l3b: BehavioralInferenceEngine,
    fu: FeatureUpdater,
    hist_df: pd.DataFrame,
    stream_sorted: pd.DataFrame,
    record_ids: set[str],
    gt_ids: set[str],
) -> dict:
    print(f"\n{'='*70}\n[{name}]  seed_history={seed_history}\n{'='*70}")

    # Fresh feature store each scenario
    fu._store = FeatureStore()
    if seed_history:
        print(f"[{name}] Seeding FeatureStore from {len(hist_df):,} historical txns...")
        fu._store.fit(hist_df)  # replays sorted history via _ingest_row
        print(f"[{name}] Seeded {len(fu._store._states):,} account states.")

    rows: list[dict] = []
    n = len(stream_sorted)
    for i, row in enumerate(stream_sorted.itertuples(index=False)):
        tid = str(row.transaction_id)
        if tid in record_ids:
            fv = fu._store.compute(row)                       # raw (internal scaler None)
            scaled = fu._rescale(fv.scaled_feature_vector)    # trained scaler, 45 dims
            fv = fv.model_copy(update={"scaled_feature_vector": scaled})
            out = l3b.score(fv, gfv=None)                     # isolate behavioral signal
            rows.append({
                "transaction_id": tid,
                "is_fraud": int(tid in gt_ids),
                "iso": out.iso_forest_score,
                "lof": out.lof_score,
                "ae":  out.autoencoder_score,
                "ensemble": out.ensemble_anomaly_score,
            })
        fu._store.update(row)                                 # progress rolling state
        if (i + 1) % 10000 == 0:
            print(f"[{name}]   {i+1:,}/{n:,} stream rows processed "
                  f"({len(rows):,} scored)")

    df = pd.DataFrame(rows)
    fraud = df[df.is_fraud == 1]
    norm  = df[df.is_fraud == 0]
    print(f"[{name}] Scored {len(df):,} txns "
          f"({len(fraud):,} fraud / {len(norm):,} normal)")

    result = {"scenario": name}
    print(f"\n  {'metric':<10}{'mean(fraud)':>14}{'mean(normal)':>14}"
          f"{'AUROC':>9}")
    print(f"  {'-'*10}{'-'*14}{'-'*14}{'-'*9}")
    for col in ("iso", "lof", "ae", "ensemble"):
        mf = float(fraud[col].mean())
        mn = float(norm[col].mean())
        au = _auroc(df.is_fraud.values, df[col].values)
        flag = "  <-- discriminates" if au >= 0.65 else (
               "  <-- COLLAPSED" if au < 0.55 else "")
        print(f"  {col:<10}{mf:>14.2f}{mn:>14.2f}{au:>9.3f}{flag}")
        result[col] = {"mean_fraud": mf, "mean_normal": mn, "auroc": au}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="L3 discrimination diagnostic")
    ap.add_argument("--normal-sample", type=int, default=2000,
                    help="number of normal stream txns to score (default 2000)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--skip-noseed", action="store_true",
                    help="only run the SEEDED scenario")
    args = ap.parse_args()

    for p in (HIST_CSV, STREAM_CSV, GT_CSV):
        if not p.exists():
            print(f"[diag] Missing required file: {p}")
            sys.exit(1)

    print("[diag] Loading data...")
    hist_df   = pd.read_csv(HIST_CSV)
    stream_df = pd.read_csv(STREAM_CSV)
    gt_ids    = set(pd.read_csv(GT_CSV, usecols=["transaction_id"])
                    ["transaction_id"].astype(str))
    print(f"[diag] historical={len(hist_df):,}  stream={len(stream_df):,}  "
          f"ground_truth_fraud={len(gt_ids):,}")

    stream_df["transaction_id"] = stream_df["transaction_id"].astype(str)
    stream_df["_is_fraud"] = stream_df["transaction_id"].isin(gt_ids)
    stream_sorted = stream_df.sort_values("timestamp").reset_index(drop=True)

    fraud_ids = set(stream_sorted.loc[stream_sorted._is_fraud, "transaction_id"])
    normal_pool = stream_sorted.loc[~stream_sorted._is_fraud, "transaction_id"].tolist()
    rng = np.random.default_rng(args.seed)
    n_norm = min(args.normal_sample, len(normal_pool))
    normal_ids = set(rng.choice(normal_pool, size=n_norm, replace=False).tolist())
    record_ids = fraud_ids | normal_ids
    print(f"[diag] Test set: {len(fraud_ids):,} fraud + {len(normal_ids):,} normal "
          f"= {len(record_ids):,} scored rows "
          f"(replaying full {len(stream_sorted):,}-row stream for correct windows)")

    l3b = BehavioralInferenceEngine.load(ARTIFACTS)
    fu  = FeatureUpdater(ARTIFACTS)  # gives us the trained-scaler _rescale()

    results = []
    if not args.skip_noseed:
        results.append(_run_scenario(
            "NO-SEED (current production)", False,
            l3b, fu, hist_df, stream_sorted, record_ids, gt_ids))
    results.append(_run_scenario(
        "SEEDED (proposed fix)", True,
        l3b, fu, hist_df, stream_sorted, record_ids, gt_ids))

    print(f"\n{'='*70}\n[diag] VERDICT\n{'='*70}")
    seeded = next(r for r in results if r["scenario"].startswith("SEEDED"))
    ens_au = seeded["ensemble"]["auroc"]
    if ens_au >= 0.65:
        print(f"[diag] With history seeded, ensemble AUROC = {ens_au:.3f} "
              f"-> Layer 3 DOES discriminate. The fix is to seed the "
              f"FeatureStore from historical data at orchestrator startup.")
    else:
        print(f"[diag] Even seeded, ensemble AUROC = {ens_au:.3f} "
              f"-> the problem is NOT (only) the missing history; deeper "
              f"investigation of feature/scaler alignment needed.")
    if not args.skip_noseed:
        ns = next(r for r in results if r["scenario"].startswith("NO-SEED"))
        print(f"[diag] No-seed ensemble AUROC = {ns['ensemble']['auroc']:.3f}, "
              f"mean(normal) = {ns['ensemble']['mean_normal']:.1f} "
              f"(high + non-discriminating == the reported symptom).")


if __name__ == "__main__":
    main()
