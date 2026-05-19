"""
Ordered, in-process eval replay — fixes the Level-2 harness bug.

THE BUG
-------
The old eval used two separate, non-chronological phases:
  rescore_fraud.py        POSTs all 418 fraud (CSV order) to the server
  score_normal_sample.py  POSTs ~1036 normals (sample order) afterward
The server's S5 FeatureStore and S6 GraphUpdater are STATEFUL — each call
computes rolling-window features from current state, then ingests. Feeding
fraud-then-normal in non-timestamp order means a transaction's velocity /
beneficiary / graph features at scoring time do NOT reflect the true
interleaved stream. This degraded measured behavioral AUROC to ~0.61 even
though the layer scores ~0.996 under correct streaming
(scripts/diagnose_l3_discrimination.py).

THE FIX
-------
Replay the FULL stream_transactions.csv in timestamp order through the
exact production orchestrator (init_orchestrator — same artifacts + same
pre-stream S5 seed + S6 seed as the live server). The full pipeline is run
only for the recorded subset (all 418 fraud + the same 1036-normal sample
the old harness used, so the 1454-row universe and all v3 numbers stay
comparable). EVERY other stream row still advances the stateful S5/S6 in
order (mirroring diagnose_l3_discrimination.py) so rolling-window state is
correct when a recorded transaction is scored.

Outputs (drop-in replacements — downstream pipeline unchanged):
  data/evaluation/scored_transactions.csv      (418 fraud)
  data/evaluation/normal_sample_scores.jsonl   (1036 normal, full responses)
  data/evaluation/generated_alerts.csv         (alerted normals)

Then:
  build_replay_scores -> build_scored_all -> run_evaluator --full \
     --scored data/evaluation/scored_all.csv \
     --alerts data/evaluation/generated_alerts.csv
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi.encoders import jsonable_encoder

from system2_platform.api.dependencies import init_orchestrator
from system2_platform.api.routes.stream import _serialise
from system2_platform.shared.s4_stream_ingestion import StreamIngestion

STREAM_CSV = Path("data/raw/stream_transactions.csv")
TRUTH_CSV  = Path("data/raw/hidden_ground_truth.csv")
OUT_FRAUD  = Path("data/evaluation/scored_transactions.csv")
OUT_NORMAL = Path("data/evaluation/normal_sample_scores.jsonl")
OUT_ALERTS = Path("data/evaluation/generated_alerts.csv")

# Must match score_normal_sample.py exactly so the eval universe is identical
N_SAMPLE = 3000
SEED = 42
MAX_PER_ACCOUNT = 10


def _clean(v):
    """pandas/numpy scalar -> native Python (NaN -> None) for pydantic."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    try:
        if v is None:
            return None
        if isinstance(v, float) and v != v:
            return None
    except Exception:
        pass
    return v


def _pick_normal_sample(stream_df: pd.DataFrame, fraud_ids: set[str]) -> set[str]:
    """Reproduce score_normal_sample.py's sampling EXACTLY (same seed/caps)."""
    normal_df = stream_df[~stream_df["transaction_id"].astype(str).isin(fraud_ids)]
    parts = []
    for _, grp in normal_df.groupby("sender_account"):
        k = min(len(grp), MAX_PER_ACCOUNT)
        parts.append(grp.sample(k, random_state=SEED))
    capped = pd.concat(parts, ignore_index=True)
    sample = capped.sample(min(N_SAMPLE, len(capped)), random_state=SEED)
    return set(sample["transaction_id"].astype(str))


def main() -> None:
    fraud_ids = set(pd.read_csv(TRUTH_CSV, usecols=["transaction_id"])
                    ["transaction_id"].astype(str))
    stream = pd.read_csv(STREAM_CSV)
    stream["transaction_id"] = stream["transaction_id"].astype(str)
    normal_ids = _pick_normal_sample(stream, fraud_ids)
    record_ids = fraud_ids | normal_ids
    stream_sorted = stream.sort_values("timestamp").reset_index(drop=True)
    print(f"[ordered] stream={len(stream_sorted):,}  "
          f"record={len(record_ids):,} (fraud={len(fraud_ids):,}, "
          f"normal={len(normal_ids):,})")

    print("[ordered] Building orchestrator (identical to live server)...")
    orch = init_orchestrator()                 # same seeding as server lifespan
    ingest = StreamIngestion()
    payload_cols = [c for c in stream_sorted.columns]

    fraud_rows: list[dict] = []
    normal_lines: list[str] = []
    alert_rows: list[dict] = []
    n_fr = n_nr = n_fr_alert = n_nr_alert = 0

    t0 = time.time()
    for i, row in enumerate(stream_sorted.itertuples(index=False)):
        d = row._asdict()
        tid = str(d["transaction_id"])
        payload = {c: _clean(d[c]) for c in payload_cols}
        event = ingest.parse(payload)

        if tid in record_ids:
            result = orch.process(event)       # FULL pipeline (S5+S6+L1+L3+L4+P1+P2)
            # jsonable_encoder exactly mirrors what FastAPI emits over HTTP
            # (datetime -> ISO string, numpy -> native) so the JSONL is
            # byte-compatible with the old score_normal_sample.py output.
            serial = jsonable_encoder(_serialise(result))
            sb = serial.get("score_breakdown", {})
            alerted = bool(serial.get("alert_generated", False))
            if tid in fraud_ids:
                n_fr += 1
                n_fr_alert += int(alerted)
                fraud_rows.append({
                    "transaction_id": tid,
                    "y_score": float(serial.get("transaction_risk_score", 0.0)),
                    "y_pred": int(alerted),
                    "risk_level": serial.get("risk_level", "Low"),
                    "rule": sb.get("rule", 0),
                    "behavioral": sb.get("behavioral", 0),
                    "gnn": sb.get("gnn", 0),
                    "graph_boost": sb.get("graph_boost", 0),
                })
            else:
                n_nr += 1
                normal_lines.append(json.dumps(serial))
                if alerted:
                    n_nr_alert += 1
                    alert_rows.append({
                        "transaction_id": tid,
                        "transaction_risk_score": float(
                            serial.get("transaction_risk_score", 0.0)),
                        "risk_level": serial.get("risk_level", "Low"),
                        "is_alerted": 1,
                    })
        else:
            # Not recorded: still advance stateful layers IN ORDER so rolling
            # windows / live graph are correct when we reach a recorded row.
            orch._feat._store.update(event)         # S5 ingest (cheap)
            orch._graph.update_and_extract(event)   # S6 graph mutate

        if (i + 1) % 5000 == 0:
            print(f"[ordered]   {i+1:,}/{len(stream_sorted):,}  "
                  f"fraud={n_fr} normal={n_nr}  "
                  f"({time.time()-t0:.0f}s)")

    OUT_FRAUD.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fraud_rows).to_csv(OUT_FRAUD, index=False)
    with open(OUT_NORMAL, "w") as fh:
        fh.write("\n".join(normal_lines) + "\n")
    pd.DataFrame(alert_rows).to_csv(OUT_ALERTS, index=False)

    fpr = n_nr_alert / n_nr if n_nr else 0.0
    rec = n_fr_alert / n_fr if n_fr else 0.0
    print(f"\n[ordered] done in {time.time()-t0:.0f}s")
    print(f"[ordered] fraud  : {n_fr} scored, {n_fr_alert} alerted "
          f"(recall {rec:.3f})")
    print(f"[ordered] normal : {n_nr} scored, {n_nr_alert} alerted "
          f"(raw FPR {fpr:.3f})")
    print(f"[ordered] wrote {OUT_FRAUD}, {OUT_NORMAL}, {OUT_ALERTS}")


if __name__ == "__main__":
    main()
