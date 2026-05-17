"""
Score all ground-truth fraud transactions through the live API.

Reads data/raw/hidden_ground_truth.csv, POSTs each row to
POST /stream/transaction, and writes the resulting risk scores to
data/evaluation/scored_transactions.csv.

Usage:
    python scripts/score_ground_truth.py
    python scripts/score_ground_truth.py --url http://localhost:8000 --delay 0.02
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import requests


TRUTH_CSV    = Path("data/raw/hidden_ground_truth.csv")
OUT_CSV      = Path("data/evaluation/scored_transactions.csv")
API_ENDPOINT = "/stream/transaction"

# Columns passed as the transaction payload
PAYLOAD_COLS = [
    "transaction_id", "timestamp", "sender_account", "receiver_account",
    "sender_name", "receiver_name", "sender_bank", "receiver_bank",
    "sender_country", "receiver_country", "amount", "currency",
    "transaction_type", "payment_channel", "device_id", "ip_address",
    "geo_latitude", "geo_longitude", "merchant_category",
    "transaction_status", "sender_balance_before", "sender_balance_after",
    "receiver_balance_before", "receiver_balance_after",
    "kyc_level", "is_international", "remarks",
]


def score_ground_truth(
    base_url: str = "http://localhost:8000",
    truth_csv: Path = TRUTH_CSV,
    out_csv: Path   = OUT_CSV,
    delay_s: float  = 0.0,
    max_rows: int   = 0,
) -> pd.DataFrame:

    truth_path = Path(truth_csv)
    if not truth_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {truth_path}")

    df = pd.read_csv(truth_path)
    if max_rows and max_rows < len(df):
        df = df.head(max_rows)

    print(f"[score_gt] Scoring {len(df)} ground-truth transactions via {base_url}...")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url     = base_url.rstrip("/") + API_ENDPOINT
    session = requests.Session()
    results: list[dict] = []
    errors  = 0

    for i, row in df.iterrows():
        payload = {
            col: _safe_val(row.get(col))
            for col in PAYLOAD_COLS
            if col in df.columns
        }

        try:
            resp = session.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            score = float(data.get("transaction_risk_score", 0.0))
            results.append({
                "transaction_id": str(row["transaction_id"]),
                "y_score":        score,
                "y_pred":         1 if score >= 31.0 else 0,
                "risk_level":     data.get("risk_level", "Low"),
            })
        except Exception as exc:
            errors += 1
            # Store zero score so evaluation still works
            results.append({
                "transaction_id": str(row["transaction_id"]),
                "y_score":        0.0,
                "y_pred":         0,
                "risk_level":     "Low",
            })
            if errors <= 5:
                print(f"[score_gt] Error on {row['transaction_id']}: {exc}")

        if (i + 1) % 50 == 0:
            print(f"[score_gt]   {i + 1}/{len(df)} scored  (errors so far: {errors})")

        if delay_s > 0:
            time.sleep(delay_s)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    print(f"[score_gt] Done.  {len(out_df)} rows -> {out_path}")
    print(f"[score_gt] Errors: {errors}")

    n_alerted = int(out_df["y_pred"].sum())
    mean_score = float(out_df["y_score"].mean())
    print(f"[score_gt] Alerted (score>=31): {n_alerted}  mean_score={mean_score:.1f}")
    return out_df


def _safe_val(v):
    """Convert pandas NA / numpy types to JSON-safe Python scalars."""
    if v is None:
        return None
    try:
        import numpy as np
        if isinstance(v, float) and (v != v):   # NaN
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except ImportError:
        pass
    # pandas NA
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score ground-truth transactions via API")
    parser.add_argument("--url",     default="http://localhost:8000")
    parser.add_argument("--truth",   default=str(TRUTH_CSV))
    parser.add_argument("--out",     default=str(OUT_CSV))
    parser.add_argument("--delay",   type=float, default=0.0,
                        help="Seconds to sleep between requests (default 0)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Limit rows (0 = all)")
    args = parser.parse_args()

    score_ground_truth(
        base_url=args.url,
        truth_csv=Path(args.truth),
        out_csv=Path(args.out),
        delay_s=args.delay,
        max_rows=args.max_rows,
    )
