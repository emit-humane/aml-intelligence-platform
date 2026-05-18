"""Re-score all 418 fraud transactions from hidden_ground_truth.csv."""
from __future__ import annotations
import csv, json
from pathlib import Path
import pandas as pd
import requests
import numpy as np

BASE_URL   = "http://localhost:8000"
TRUTH_CSV  = Path("data/raw/hidden_ground_truth.csv")
OUT_CSV    = Path("data/evaluation/scored_transactions.csv")

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
    df = pd.read_csv(TRUTH_CSV)
    print(f"Scoring {len(df)} fraud transactions...")

    session = requests.Session()
    rows = []
    n_ok = n_err = n_alerted = 0

    for i, (_, row) in enumerate(df.iterrows()):
        payload = {}
        for col in PAYLOAD_COLS:
            if col not in df.columns: continue
            v = row.get(col)
            try:
                if pd.isna(v): v = None
                elif isinstance(v, (np.integer,)): v = int(v)
                elif isinstance(v, (np.floating,)): v = float(v)
                elif isinstance(v, (np.bool_,)): v = bool(v)
            except: pass
            payload[col] = v
        try:
            resp = session.post(f"{BASE_URL}/stream/transaction", json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            n_ok += 1
            score = float(data.get("transaction_risk_score", 0))
            alerted = data.get("alert_generated", False)
            if alerted: n_alerted += 1
            rows.append({
                "transaction_id": str(row["transaction_id"]),
                "y_score": score,
                "y_pred": int(alerted),
                "risk_level": data.get("risk_level", "Low"),
                "rule": data.get("score_breakdown", {}).get("rule", 0),
                "behavioral": data.get("score_breakdown", {}).get("behavioral", 0),
                "gnn": data.get("score_breakdown", {}).get("gnn", 0),
                "graph_boost": data.get("score_breakdown", {}).get("graph_boost", 0),
            })
        except Exception as exc:
            n_err += 1
            if n_err <= 5: print(f"  Error [{i}]: {exc}")

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)}  alerted={n_alerted}  errors={n_err}")

    result_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)

    print(f"\nFraud scoring done: {n_ok} OK | {n_err} errors | {n_alerted} alerted")
    print(f"Score distribution:")
    print(result_df["y_score"].describe().round(2))
    print(f"Alerted (>= 31): {(result_df['y_score'] >= 31).sum()} / {len(result_df)}")
    print(f"Avg component scores: rule={result_df['rule'].mean():.2f}  "
          f"behav={result_df['behavioral'].mean():.2f}  "
          f"gnn={result_df['gnn'].mean():.2f}  "
          f"boost={result_df['graph_boost'].mean():.2f}")

if __name__ == "__main__":
    main()
