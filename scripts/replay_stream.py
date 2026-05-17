"""
Stream Replay Script.

Reads data/raw/stream_transactions.csv and POSTs each transaction
to the API in chronological order at a configurable rate.

Writes two output files:
  --out / --output  JSONL file: one API response per line (default: data/evaluation/replay_scores.jsonl)
  --alerts-csv      CSV for System 3 evaluator (transaction_id, is_alerted, risk_score)

Usage:
    python scripts/replay_stream.py
    python scripts/replay_stream.py --rate 50 --url http://localhost:8000
    python scripts/replay_stream.py --input data/raw/stream_transactions.csv \\
                                    --output data/evaluation/replay_scores.jsonl \\
                                    --rate 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_URL  = "http://localhost:8000"
DEFAULT_RATE = 10   # transactions per second


def replay(
    csv_path: str | Path,
    api_url: str,
    rate: float,
    out_path: str | Path | None = "data/evaluation/replay_scores.jsonl",
    alerts_csv: str | Path | None = "data/evaluation/generated_alerts.csv",
) -> None:
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    print(f"[replay] Replaying {n:,} transactions at {rate} tx/s → {api_url}")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if alerts_csv:
        Path(alerts_csv).parent.mkdir(parents=True, exist_ok=True)

    interval = 1.0 / rate
    out_fh     = open(out_path, "w")    if out_path    else None
    alerts_fh  = open(alerts_csv, "w") if alerts_csv  else None
    if alerts_fh:
        alerts_fh.write("transaction_id,is_alerted,risk_level,transaction_risk_score\n")

    n_ok = n_err = 0

    try:
        for i, row in enumerate(df.itertuples(index=False)):
            payload = {k: _to_json(v) for k, v in row._asdict().items()}
            t0 = time.time()
            try:
                resp = requests.post(
                    f"{api_url}/stream/transaction",
                    json=payload,
                    timeout=5.0,
                )
                resp.raise_for_status()
                result = resp.json()
                n_ok += 1

                if out_fh:
                    out_fh.write(json.dumps(result) + "\n")

                if alerts_fh:
                    tx_id      = result.get("transaction_id", payload.get("transaction_id", ""))
                    risk_score = float(result.get("transaction_risk_score", 0.0))
                    risk_level = result.get("risk_level", "Low")
                    is_alerted = 1 if result.get("alert_generated", False) else 0
                    alerts_fh.write(f"{tx_id},{is_alerted},{risk_level},{risk_score}\n")

                if (i + 1) % 500 == 0:
                    print(
                        f"[replay] {i+1:,}/{n:,}  "
                        f"risk={result.get('transaction_risk_score', 0):.0f}  "
                        f"level={result.get('risk_level', '?')}"
                    )
            except Exception as exc:
                n_err += 1
                if n_err <= 5:
                    print(f"[replay] ERROR tx {i}: {exc}")

            elapsed = time.time() - t0
            sleep_t = max(0.0, interval - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        if out_fh:
            out_fh.close()
        if alerts_fh:
            alerts_fh.close()

    print(f"[replay] Done: {n_ok:,} OK, {n_err:,} errors")
    if out_path:
        print(f"[replay] JSONL → {out_path}")
    if alerts_csv:
        print(f"[replay] Alerts CSV → {alerts_csv}")


def _to_json(v):
    """Convert numpy types and pandas NaT to JSON-serialisable."""
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay stream transactions to API")
    # Support both old and new argument names
    parser.add_argument("--csv",   "--input",  dest="csv",
                        default="data/raw/stream_transactions.csv",
                        help="Input CSV of stream transactions")
    parser.add_argument("--out",   "--output", dest="out",
                        default="data/evaluation/replay_scores.jsonl",
                        help="JSONL file for API responses")
    parser.add_argument("--alerts-csv", default="data/evaluation/generated_alerts.csv",
                        help="CSV for System 3 evaluator")
    parser.add_argument("--url",  default=DEFAULT_URL)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help="Transactions per second")
    args = parser.parse_args()

    replay(
        csv_path=args.csv,
        api_url=args.url,
        rate=args.rate,
        out_path=args.out,
        alerts_csv=args.alerts_csv,
    )


if __name__ == "__main__":
    main()
