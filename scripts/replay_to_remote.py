"""
replay_to_remote.py — Feed synthetic transaction data into any AML API endpoint.

Reads a local Parquet or CSV file and POSTs the transactions to
POST /stream/batch in parallel batches.

Usage examples
--------------
# Stream data (54K rows) → remote Render API
python scripts/replay_to_remote.py

# All historical data (550K rows) → remote API
python scripts/replay_to_remote.py --file data/raw/all_transactions.parquet

# Stream data → local API
python scripts/replay_to_remote.py --url http://localhost:8000

# Custom batch size, 6 workers, limit 10K rows
python scripts/replay_to_remote.py --batch 100 --workers 6 --limit 10000

# Dry-run (parse + print first batch, no POST)
python scripts/replay_to_remote.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_URL    = "https://aml-intelligence-platform.onrender.com"
DEFAULT_FILE   = "data/raw/stream_transactions.csv"
DEFAULT_BATCH  = 50      # transactions per POST /stream/batch request
DEFAULT_WORKERS = 4      # parallel HTTP workers
DEFAULT_LIMIT  = 0       # 0 = no limit


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path, limit: int) -> list[dict]:
    """Load CSV or Parquet file into a list of row dicts."""
    print(f"[loader] Reading {path} …")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    print(f"[loader] {len(df):,} rows × {len(df.columns)} cols loaded")

    if limit and limit < len(df):
        df = df.head(limit)
        print(f"[loader] Truncated to {limit:,} rows (--limit)")

    # Coerce common type problems before serialising to JSON
    # NaN → None (JSON null), booleans, ints
    df = df.where(pd.notna(df), other=None)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(bool)
    for col in df.select_dtypes(include="int64").columns:
        df[col] = df[col].astype(int)

    return df.to_dict(orient="records")


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    """Replace NaN/None in string fields with '' so Pydantic doesn't reject them."""
    str_fields = ("remarks", "device_id", "ip_address", "merchant_category",
                  "sender_name", "receiver_name")
    for f in str_fields:
        if f in row and (row[f] is None or (isinstance(row[f], float) and math.isnan(row[f]))):
            row[f] = ""
    return row


def _post_batch(session: requests.Session, url: str, batch: list[dict]) -> dict:
    """POST one batch; return a result summary dict."""
    cleaned = [_clean_row(dict(r)) for r in batch]
    t0 = time.perf_counter()
    resp = session.post(f"{url}/stream/batch", json=cleaned, timeout=60)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    results = resp.json()
    flagged = sum(1 for r in results if r.get("transaction_risk_score", 0) >= 31)
    return {"sent": len(cleaned), "flagged": flagged, "elapsed_s": elapsed}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Replay synthetic transactions to the AML API")
    parser.add_argument("--url",     default=DEFAULT_URL,    help="API base URL")
    parser.add_argument("--file",    default=DEFAULT_FILE,   help="Parquet or CSV file path")
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH,   help="Rows per POST request")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel HTTP workers")
    parser.add_argument("--limit",   type=int, default=DEFAULT_LIMIT,   help="Max rows (0=all)")
    parser.add_argument("--dry-run", action="store_true",  help="Parse only — no HTTP calls")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    rows = _load(file_path, args.limit)
    batches = [rows[i : i + args.batch] for i in range(0, len(rows), args.batch)]

    print(f"\n{'='*60}")
    print(f"  Target : {args.url}")
    print(f"  File   : {file_path}  ({len(rows):,} rows)")
    print(f"  Batches: {len(batches)}  ×  {args.batch} rows")
    print(f"  Workers: {args.workers}")
    print(f"  Est.   : ~{len(rows)*0.045/args.workers:.0f}s  (45ms/tx ÷ {args.workers} workers)")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[dry-run] First batch sample:")
        for row in batches[0][:2]:
            print(" ", {k: v for k, v in list(row.items())[:6]}, "…")
        return

    # Warmup / health check
    print("[init] Checking API health …")
    try:
        r = requests.get(f"{args.url}/health/ready", timeout=30)
        r.raise_for_status()
        print(f"[init] API is up: {r.json()}\n")
    except Exception as exc:
        print(f"[WARN] Health check failed ({exc}) — proceeding anyway\n")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    total_sent    = 0
    total_flagged = 0
    total_errors  = 0
    wall_start    = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_post_batch, session, args.url, b): idx
                   for idx, b in enumerate(batches)}

        done = 0
        for future in as_completed(futures):
            done += 1
            idx = futures[future]
            try:
                res = future.result()
                total_sent    += res["sent"]
                total_flagged += res["flagged"]
                elapsed_total  = time.perf_counter() - wall_start
                rate           = total_sent / elapsed_total if elapsed_total > 0 else 0
                eta_s          = (len(rows) - total_sent) / rate if rate > 0 else 0
                print(
                    f"  batch {idx+1:>5}/{len(batches)}  "
                    f"sent={total_sent:>7,}  "
                    f"flagged={total_flagged:>5,}  "
                    f"rate={rate:>5.0f}/s  "
                    f"ETA={eta_s/60:>4.1f}min"
                )
            except Exception as exc:
                total_errors += 1
                print(f"  batch {idx+1:>5} ERROR: {exc}")

    wall_total = time.perf_counter() - wall_start
    print(f"\n{'='*60}")
    print(f"DONE   {total_sent:,} sent  |  {total_flagged:,} flagged  "
          f"|  {total_errors} errors  |  {wall_total:.0f}s elapsed")
    print(f"       avg rate: {total_sent/wall_total:.0f} tx/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
