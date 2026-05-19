"""
Recompute the System 3 eval inputs under the v3 fusion weights WITHOUT
re-scoring through the server.

Rationale: the per-component scores (rule / behavioral / gnn / graph_boost)
are independent of the fusion weights — only their linear combination
changes. p1_risk_fusion applies exactly:

    tx_score = W_RULE*rule + W_BEHAV*behavioral + W_GNN*gnn + W_GRAPH*graph_boost

and the recorded score_breakdown already stores the post-cycle-bonus
graph_boost, so recomputing from the existing breakdowns reproduces exactly
what the server would emit with the v3 weights. This rewrites:

  data/evaluation/scored_transactions.csv     (fraud, v3 y_score/y_pred/level)
  data/evaluation/normal_sample_scores.jsonl  (normal, v3 fused + alert flag)
  data/evaluation/generated_alerts.csv        (normals alerted under v3)

Then run the standard tail of the pipeline:
  build_replay_scores -> build_scored_all -> run_evaluator --full \
      --scored data/evaluation/scored_all.csv \
      --alerts data/evaluation/generated_alerts.csv

Run:
  python -m scripts.rebuild_fused_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from system2_platform.post_detection.p1_risk_fusion import (
    _W_RULE, _W_BEHAV, _W_GNN, _W_GRAPH, _risk_level,
)
from system2_platform.post_detection.p2_alert_manager import _ALERT_THRESHOLD

FRAUD_CSV    = Path("data/evaluation/scored_transactions.csv")
NORMAL_JSONL = Path("data/evaluation/normal_sample_scores.jsonl")
OUT_ALERTS   = Path("data/evaluation/generated_alerts.csv")


def _fuse(rule: float, behav: float, gnn: float, graph: float) -> float:
    s = _W_RULE * rule + _W_BEHAV * behav + _W_GNN * gnn + _W_GRAPH * graph
    return float(min(max(s, 0.0), 100.0))


def main() -> None:
    print(f"[rebuild] v3 weights: rule={_W_RULE} behav={_W_BEHAV} "
          f"gnn={_W_GNN} graph={_W_GRAPH}  alert_thr={_ALERT_THRESHOLD}")

    # ── Fraud ────────────────────────────────────────────────────────────────
    fr = pd.read_csv(FRAUD_CSV)
    new_score, new_pred, new_level = [], [], []
    for _, r in fr.iterrows():
        s = _fuse(float(r["rule"]), float(r["behavioral"]),
                  float(r["gnn"]), float(r["graph_boost"]))
        new_score.append(round(s, 2))
        new_pred.append(int(s >= _ALERT_THRESHOLD))
        new_level.append(_risk_level(s))
    fr["y_score"] = new_score
    fr["y_pred"] = new_pred
    fr["risk_level"] = new_level
    fr.to_csv(FRAUD_CSV, index=False)
    n_fr_alert = int(sum(new_pred))
    print(f"[rebuild] fraud: {len(fr)} rows  alerted={n_fr_alert} "
          f"({n_fr_alert/len(fr):.1%} recall)")

    # ── Normal ───────────────────────────────────────────────────────────────
    lines_out = []
    alert_rows = []
    n_norm = n_norm_alert = 0
    with open(NORMAL_JSONL) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            b = d.get("score_breakdown", {})
            s = _fuse(float(b.get("rule", 0)), float(b.get("behavioral", 0)),
                      float(b.get("gnn", 0)), float(b.get("graph_boost", 0)))
            lvl = _risk_level(s)
            alerted = bool(s >= _ALERT_THRESHOLD)
            d["transaction_risk_score"] = round(s, 2)
            d["risk_level"] = lvl
            d["alert_generated"] = alerted
            lines_out.append(json.dumps(d))
            n_norm += 1
            if alerted:
                n_norm_alert += 1
                alert_rows.append({
                    "transaction_id": str(d.get("transaction_id", "")),
                    "transaction_risk_score": round(s, 2),
                    "risk_level": lvl,
                    "is_alerted": 1,
                })
    with open(NORMAL_JSONL, "w") as fh:
        fh.write("\n".join(lines_out) + "\n")
    pd.DataFrame(alert_rows).to_csv(OUT_ALERTS, index=False)
    fpr = n_norm_alert / n_norm if n_norm else 0.0
    print(f"[rebuild] normal: {n_norm} rows  alerted={n_norm_alert} "
          f"(raw FPR {fpr:.3f})")
    print(f"[rebuild] wrote {FRAUD_CSV}, {NORMAL_JSONL}, {OUT_ALERTS}")


if __name__ == "__main__":
    main()
