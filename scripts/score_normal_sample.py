"""
Score a stratified sample of NORMAL transactions from stream_transactions.csv.
Writes:
  data/evaluation/normal_sample_scores.jsonl   — all scored results
  data/evaluation/generated_alerts.csv         — ALERTED transactions only (y_pred=1)

This is the correct input for e1_joiner / run_evaluator.
"""
from __future__ import annotations

import csv, json, time, random
from pathlib import Path
import pandas as pd
import requests

BASE_URL   = "http://localhost:8000"
STREAM_CSV = Path("data/raw/stream_transactions.csv")
TRUTH_CSV  = Path("data/raw/hidden_ground_truth.csv")
OUT_JSONL  = Path("data/evaluation/normal_sample_scores.jsonl")
OUT_ALERTS = Path("data/evaluation/generated_alerts.csv")
N_SAMPLE   = 3000   # normal transactions to score
SEED       = 42

PAYLOAD_COLS = [
    "transaction_id","timestamp","sender_account","receiver_account",
    "sender_name","receiver_name","sender_bank","receiver_bank",
    "sender_country","receiver_country","amount","currency",
    "transaction_type","payment_channel","device_id","ip_address",
    "geo_latitude","geo_longitude","merchant_category","transaction_status",
    "sender_balance_before","sender_balance_after",
    "receiver_balance_before","receiver_balance_after",
    "kyc_level","is_international","remarks",
]


def main():
    # Load fraud IDs to exclude
    truth_ids = set(pd.read_csv(TRUTH_CSV)["transaction_id"].astype(str))
    print(f"[normal_sample] {len(truth_ids)} fraud IDs excluded")

    # Load stream, drop fraud, cap per account to avoid hub bias, then sample
    df = pd.read_csv(STREAM_CSV)
    normal_df = df[~df["transaction_id"].astype(str).isin(truth_ids)]

    # Cap each sender to at most 10 transactions to prevent hub domination
    MAX_PER_ACCOUNT = 10
    capped_parts = []
    for _, grp in normal_df.groupby("sender_account"):
        n = min(len(grp), MAX_PER_ACCOUNT)
        capped_parts.append(grp.sample(n, random_state=SEED))
    capped_df = pd.concat(capped_parts, ignore_index=True)

    sample_df = capped_df.sample(min(N_SAMPLE, len(capped_df)), random_state=SEED)
    top3 = sample_df["sender_account"].value_counts().head(3).to_dict()
    print(f"[normal_sample] Scoring {len(sample_df)} normal transactions "
          f"(capped {MAX_PER_ACCOUNT}/account, top-3 senders: {top3})")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    n_ok = n_err = n_alerted = 0
    alert_rows = []

    with open(OUT_JSONL, "w") as jf:
        for i, (_, row) in enumerate(sample_df.iterrows()):
            payload = {}
            for col in PAYLOAD_COLS:
                if col not in df.columns:
                    continue
                v = row.get(col)
                try:
                    import numpy as np, pandas as pd_
                    if pd_.isna(v):
                        v = None
                    elif isinstance(v, (np.integer,)):
                        v = int(v)
                    elif isinstance(v, (np.floating,)):
                        v = float(v)
                    elif isinstance(v, (np.bool_,)):
                        v = bool(v)
                except Exception:
                    pass
                payload[col] = v

            try:
                resp = session.post(
                    f"{BASE_URL}/stream/transaction",
                    json=payload, timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
                n_ok += 1
                jf.write(json.dumps(data) + "\n")

                if data.get("alert_generated", False):
                    n_alerted += 1
                    alert_rows.append({
                        "transaction_id":        str(row["transaction_id"]),
                        "transaction_risk_score": float(data.get("transaction_risk_score", 0)),
                        "risk_level":             data.get("risk_level", "Low"),
                        "is_alerted":             1,
                    })

            except Exception as exc:
                n_err += 1
                if n_err <= 5:
                    print(f"  Error [{i}]: {exc}")

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(sample_df)}  alerted={n_alerted}  errors={n_err}")

    # Write alerts CSV (alerted normals only — these are FPs for evaluation)
    alerts_df = pd.DataFrame(alert_rows)
    alerts_df.to_csv(OUT_ALERTS, index=False)

    print(f"\n[normal_sample] Done: {n_ok} OK | {n_err} errors | {n_alerted} alerted (FP candidates)")
    print(f"  JSONL  -> {OUT_JSONL}")
    print(f"  Alerts -> {OUT_ALERTS}  ({len(alert_rows)} rows)")
    fpr = n_alerted / n_ok if n_ok else 0
    print(f"  Raw FPR on sample: {fpr:.3f} ({n_alerted}/{n_ok})")


if __name__ == "__main__":
    main()
