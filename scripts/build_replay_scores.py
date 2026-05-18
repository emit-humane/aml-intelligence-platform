"""
Build data/evaluation/replay_scores.jsonl by merging:
  - scored_transactions.csv  (418 fraud, from hidden_ground_truth)
  - normal_sample_scores.jsonl  (1036 normal API responses)
into a single JSONL expected by run_evaluator --scores.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

FRAUD_CSV    = Path("data/evaluation/scored_transactions.csv")
NORMAL_JSONL = Path("data/evaluation/normal_sample_scores.jsonl")
OUT_JSONL    = Path("data/evaluation/replay_scores.jsonl")

fraud_df = pd.read_csv(FRAUD_CSV)
print(f"Fraud transactions: {len(fraud_df)}")

OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
n_fraud = n_normal = 0

with open(OUT_JSONL, "w") as out:
    # Write fraud transactions
    for _, row in fraud_df.iterrows():
        rec = {
            "transaction_id": str(row["transaction_id"]),
            "transaction_risk_score": float(row["y_score"]),
            "risk_level": str(row.get("risk_level", "Low")),
        }
        out.write(json.dumps(rec) + "\n")
        n_fraud += 1

    # Write normal transactions from JSONL
    with open(NORMAL_JSONL) as jf:
        for line in jf:
            line = line.strip()
            if line:
                data = json.loads(line)
                # Keep only essential fields for quick evaluator
                rec = {
                    "transaction_id": str(data.get("transaction_id", "")),
                    "transaction_risk_score": float(data.get("transaction_risk_score", 0)),
                    "risk_level": str(data.get("risk_level", "Low")),
                }
                out.write(json.dumps(rec) + "\n")
                n_normal += 1

print(f"Written {n_fraud} fraud + {n_normal} normal = {n_fraud+n_normal} total")
print(f"Output: {OUT_JSONL}")
