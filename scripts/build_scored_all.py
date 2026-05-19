"""
Build data/evaluation/scored_all.csv = ALL scored transactions
(418 fraud + 1036 normals, alerted or not) so the E1 joiner can construct
the full evaluation universe with proper True Negatives.

Columns: transaction_id, y_score, y_pred, risk_level
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

FRAUD_CSV    = Path("data/evaluation/scored_transactions.csv")
NORMAL_JSONL = Path("data/evaluation/normal_sample_scores.jsonl")
OUT_CSV      = Path("data/evaluation/scored_all.csv")
ALERT_THRESHOLD = 45.0  # v4 — aligned with p2_alert_manager._ALERT_THRESHOLD

rows = []

# Fraud (y_true is set by joiner via truth join; we just provide score/pred)
fraud = pd.read_csv(FRAUD_CSV)
for _, r in fraud.iterrows():
    score = float(r["y_score"])
    rows.append({
        "transaction_id": str(r["transaction_id"]),
        "y_score": score,
        "y_pred": int(score >= ALERT_THRESHOLD),
        "risk_level": str(r.get("risk_level", "")),
    })

# Normals (all 1036, alerted or not)
with open(NORMAL_JSONL) as jf:
    for line in jf:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        score = float(d.get("transaction_risk_score", 0))
        rows.append({
            "transaction_id": str(d.get("transaction_id", "")),
            "y_score": score,
            "y_pred": int(score >= ALERT_THRESHOLD),
            "risk_level": str(d.get("risk_level", "")),
        })

df = pd.DataFrame(rows).drop_duplicates("transaction_id", keep="first")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} rows to {OUT_CSV}")
print(f"  y_pred=1 (alerted): {int(df['y_pred'].sum())}")
print(f"  y_pred=0 (silent):  {int((df['y_pred']==0).sum())}")
