"""
E1 — Alert-to-Ground-Truth Joiner.

Left-joins ALL scored transactions against the hidden ground truth on transaction_id.
Builds joined_evaluation_frame.parquet with columns:
  transaction_id, y_true (int), y_pred (int), y_score (float),
  risk_level, fraud_ring_id, synthetic_pattern_type, scenario_severity

Inputs:
  data/raw/hidden_ground_truth.csv   — transaction_id, fraud_ring_id, synthetic_pattern_type, ...
  data/generated_alerts.csv          — alerted transactions with risk scores
  data/raw/all_transactions.parquet  — full transaction universe (for non-alerted FP/TN)

The evaluation universe = ground truth fraud transactions  +  alerted transactions
(all alerted normals count as FP, all un-alerted fraud as FN).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def join(
    truth_csv: Path | str,
    alerts_csv: Path | str,
    scored_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Build the evaluation frame.

    Parameters
    ----------
    truth_csv   : path to hidden_ground_truth.csv
    alerts_csv  : path to generated_alerts.csv (P2 output)
    scored_csv  : optional path to a supplementary scored-transactions CSV
                  (e.g., data/evaluation/scored_transactions.csv generated
                  by score_ground_truth.py). Rows here are added to the
                  universe even if no alert was raised.

    Returns a DataFrame with columns:
        transaction_id, y_true, y_pred, y_score,
        risk_level, fraud_ring_id, synthetic_pattern_type, scenario_severity
    """
    truth_path  = Path(truth_csv)
    alerts_path = Path(str(alerts_csv))

    truth = pd.read_csv(truth_path)

    # Normalise truth column names
    gt_cols = {
        "transaction_id": "transaction_id",
        "fraud_ring_id":  "fraud_ring_id",
        "synthetic_pattern_type": "synthetic_pattern_type",
        "pattern_type":  "synthetic_pattern_type",   # legacy alias
        "scenario_severity": "scenario_severity",
    }
    for old, new in gt_cols.items():
        if old in truth.columns and old != new and new not in truth.columns:
            truth = truth.rename(columns={old: new})

    truth["y_true"] = 1

    # ── Load alerts (y_pred = 1) ──────────────────────────────────────────
    if alerts_path.exists():
        alerts = pd.read_csv(alerts_path)
    else:
        alerts = pd.DataFrame()

    if not alerts.empty:
        score_col = next(
            (c for c in ("transaction_risk_score", "risk_score", "alert_score")
             if c in alerts.columns),
            None,
        )
        level_col = next(
            (c for c in ("risk_level",) if c in alerts.columns),
            None,
        )

        keep: list[str] = ["transaction_id"]
        rename: dict[str, str] = {}
        if score_col:
            keep.append(score_col)
            rename[score_col] = "y_score"
        if level_col:
            keep.append(level_col)
            rename[level_col] = "risk_level"

        keep = [c for c in keep if c in alerts.columns]
        alerts_slim = (
            alerts[keep]
            .rename(columns=rename)
            # Keep the highest score per transaction_id in case of duplicates
            .sort_values("y_score", ascending=False) if "y_score" in rename.values()
            else alerts[keep].rename(columns=rename)
        )
        if "y_score" in alerts_slim.columns:
            alerts_slim = alerts_slim.drop_duplicates("transaction_id", keep="first")
        else:
            alerts_slim = alerts_slim.drop_duplicates("transaction_id")

        alerts_slim["y_pred"] = 1
        if "y_score" not in alerts_slim.columns:
            alerts_slim["y_score"] = 50.0
    else:
        alerts_slim = pd.DataFrame(columns=["transaction_id", "y_pred", "y_score"])

    # ── Load supplementary scored transactions (not necessarily alerted) ──
    sup_rows: list[pd.DataFrame] = []
    if scored_csv is not None:
        sc_path = Path(str(scored_csv))
        if sc_path.exists():
            sc = pd.read_csv(sc_path)
            # Expected columns: transaction_id, y_score (optionally y_pred, risk_level)
            if "y_score" not in sc.columns and "transaction_risk_score" in sc.columns:
                sc = sc.rename(columns={"transaction_risk_score": "y_score"})
            if "y_pred" not in sc.columns:
                sc["y_pred"] = 0   # not alerted unless present in alerts
            sup_rows.append(sc[["transaction_id", "y_score", "y_pred"]])

    # ── Build universe: truth union alerted union supplementary ──────────
    truth_ids = set(truth["transaction_id"])
    alert_ids = set(alerts_slim["transaction_id"]) if not alerts_slim.empty else set()

    # Start with ground truth rows
    base = truth[
        ["transaction_id", "fraud_ring_id", "synthetic_pattern_type", "scenario_severity", "y_true"]
    ].copy()

    # Merge in alert scores for ground truth transactions
    base = base.merge(
        alerts_slim.rename(columns={}),
        on="transaction_id",
        how="left",
    )
    base["y_pred"]  = base["y_pred"].fillna(0).astype(int)
    base["y_score"] = base["y_score"].fillna(0.0)

    # Supplementary scored transactions (fill y_true=1 for GT, 0 otherwise)
    for sup in sup_rows:
        # Update scores for GT transactions that appear in sup
        in_gt = sup[sup["transaction_id"].isin(truth_ids)]
        not_in_gt = sup[~sup["transaction_id"].isin(truth_ids)]

        # Update existing rows with better scores
        for _, row in in_gt.iterrows():
            mask = base["transaction_id"] == row["transaction_id"]
            if mask.any():
                if base.loc[mask, "y_score"].iloc[0] == 0.0:
                    base.loc[mask, "y_score"] = row["y_score"]
                    base.loc[mask, "y_pred"]  = int(row["y_pred"])

        # Add new normal rows (FP / TN candidates)
        if not not_in_gt.empty:
            extra = not_in_gt.copy()
            extra["y_true"] = 0
            extra["fraud_ring_id"] = None
            extra["synthetic_pattern_type"] = None
            extra["scenario_severity"] = None
            if "risk_level" not in extra.columns:
                extra["risk_level"] = None
            base = pd.concat([base, extra[base.columns.intersection(extra.columns)]], ignore_index=True)

    # Alerted normal transactions (FP) — add those not already in base
    alerted_normals = alerts_slim[~alerts_slim["transaction_id"].isin(truth_ids)]
    if not alerted_normals.empty:
        fp_rows = alerted_normals.copy()
        fp_rows["y_true"] = 0
        fp_rows["fraud_ring_id"] = None
        fp_rows["synthetic_pattern_type"] = None
        fp_rows["scenario_severity"] = None
        base = pd.concat(
            [base, fp_rows[base.columns.intersection(fp_rows.columns)]],
            ignore_index=True,
        )

    # Ensure canonical types
    base["y_true"]  = base["y_true"].fillna(0).astype(int)
    base["y_pred"]  = base["y_pred"].fillna(0).astype(int)
    base["y_score"] = base["y_score"].fillna(0.0).astype(float)
    if "risk_level" not in base.columns:
        base["risk_level"] = None

    base = base.drop_duplicates("transaction_id", keep="first").reset_index(drop=True)
    return base


def build_and_save(
    truth_csv: Path | str = "data/raw/hidden_ground_truth.csv",
    alerts_csv: Path | str = "data/generated_alerts.csv",
    scored_csv: Optional[Path | str] = "data/evaluation/scored_transactions.csv",
    out_path: Path | str = "data/evaluation/joined_evaluation_frame.parquet",
) -> pd.DataFrame:
    """Join + save joined_evaluation_frame.parquet. Returns the frame."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame = join(truth_csv, alerts_csv, scored_csv)
    frame.to_parquet(out_path, index=False)
    print(
        f"[E1] Joined frame: {len(frame)} rows "
        f"(y_true=1: {frame['y_true'].sum()}, "
        f"y_pred=1: {frame['y_pred'].sum()}) -> {out_path}"
    )
    return frame
