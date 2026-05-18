"""
CLI entry point for System 3 — Blind Evaluation.

Two evaluation modes:
  1. Quick mode (default): reads replay_scores.jsonl → P/R/F1/AUROC summary
  2. Full mode (--full): additionally runs system3_evaluator.generate_report()
     which produces evaluation_report.json with per-typology coverage,
     ring detection rate, and benchmark comparison.

Usage:
    python -m scripts.run_evaluator
    python -m scripts.run_evaluator --full
    python -m scripts.run_evaluator --scores data/evaluation/replay_scores.jsonl
                                    --alerts data/evaluation/generated_alerts.csv
                                    --truth  data/raw/hidden_ground_truth.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Quick evaluation (reads JSONL from replay)
# ─────────────────────────────────────────────────────────────────────────────

def quick_evaluate(
    scores_jsonl: str | Path,
    truth_csv: str | Path,
    threshold_medium: float = 40.0,
    threshold_high: float = 49.0,
) -> dict:
    scores_path = Path(scores_jsonl)
    if not scores_path.exists():
        print(f"[eval] Scores file not found: {scores_path}")
        print("[eval] Run 'python scripts/replay_stream.py' first.")
        return {}

    records = []
    with open(scores_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("[eval] No scored records found.")
        return {}

    scores_df = pd.DataFrame([{
        "transaction_id": r["transaction_id"],
        "risk_score":     float(r.get("transaction_risk_score", 0.0)),
        "risk_level":     r.get("risk_level", "Low"),
    } for r in records])

    # Load ground truth
    truth_path = Path(truth_csv)
    if not truth_path.exists():
        print(f"[eval] Ground truth not found: {truth_path}")
        return {}

    truth_df  = pd.read_csv(truth_path)
    truth_set = set(truth_df["transaction_id"].astype(str))

    scores_df["is_fraud_pred_medium"] = scores_df["risk_score"] >= threshold_medium
    scores_df["is_fraud_pred_high"]   = scores_df["risk_score"] >= threshold_high
    scores_df["is_fraud_true"] = scores_df["transaction_id"].astype(str).isin(truth_set)

    def prf(y_true, y_pred):
        tp = (y_true & y_pred).sum()
        fp = (~y_true & y_pred).sum()
        fn = (y_true & ~y_pred).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(prec, 4), round(rec, 4), round(f1, 4), int(tp), int(fp), int(fn)

    p_m, r_m, f_m, tp_m, fp_m, fn_m = prf(
        scores_df["is_fraud_true"], scores_df["is_fraud_pred_medium"])
    p_h, r_h, f_h, tp_h, fp_h, fn_h = prf(
        scores_df["is_fraud_true"], scores_df["is_fraud_pred_high"])

    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(
            scores_df["is_fraud_true"].astype(int), scores_df["risk_score"]))
    except Exception:
        auroc = -1.0

    n_fraud_truth  = len(truth_set)
    n_fraud_scored = int(scores_df["is_fraud_true"].sum())
    coverage = n_fraud_scored / n_fraud_truth if n_fraud_truth > 0 else 0.0

    results = {
        "n_scored":           len(scores_df),
        "n_fraud_in_truth":   n_fraud_truth,
        "n_fraud_in_scored":  n_fraud_scored,
        "coverage_pct":       round(coverage * 100, 2),
        "auroc":              round(auroc, 4),
        "threshold_medium": {
            "threshold": threshold_medium,
            "precision": p_m, "recall": r_m, "f1": f_m,
            "TP": tp_m, "FP": fp_m, "FN": fn_m,
        },
        "threshold_high": {
            "threshold": threshold_high,
            "precision": p_h, "recall": r_h, "f1": f_h,
            "TP": tp_h, "FP": fp_h, "FN": fn_h,
        },
    }

    print("\n[Evaluator] Quick Results:")
    print(f"  Scored:          {results['n_scored']:,}")
    print(f"  Fraud in truth:  {results['n_fraud_in_truth']:,}")
    print(f"  Coverage:        {results['coverage_pct']}%")
    print(f"  AUROC:           {results['auroc']}")
    print(f"\n  @Medium (>={threshold_medium}):  "
          f"P={p_m:.3f}  R={r_m:.3f}  F1={f_m:.3f}  "
          f"TP={tp_m}  FP={fp_m}  FN={fn_m}")
    print(f"  @High   (>={threshold_high}):  "
          f"P={p_h:.3f}  R={r_h:.3f}  F1={f_h:.3f}  "
          f"TP={tp_h}  FP={fp_h}  FN={fn_h}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation via System 3 pipeline
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluate(
    truth_csv: str | Path,
    alerts_csv: str | Path = "data/generated_alerts.csv",
    scored_csv: str | Path | None = "data/evaluation/scored_transactions.csv",
    out_dir: str | Path = "data/evaluation",
) -> dict:
    alerts_path = Path(alerts_csv)
    truth_path  = Path(truth_csv)

    if not alerts_path.exists():
        print(f"[eval] Alerts CSV not found: {alerts_path}")
        print("[eval] Run 'python scripts/replay_stream.py' or use POST /stream/start first.")
        return {}

    if not truth_path.exists():
        print(f"[eval] Ground truth not found: {truth_path}")
        return {}

    try:
        from system3_evaluator import generate_report
        print("\n[Evaluator] Running System 3 full evaluation pipeline...")
        report = generate_report(
            truth_csv=truth_path,
            alerts_csv=alerts_path,
            scored_csv=scored_csv,
            out_dir=out_dir,
        )
        return report
    except Exception as exc:
        print(f"[eval] System 3 evaluator failed: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AML evaluator")
    parser.add_argument("--scores",
                        default="data/evaluation/replay_scores.jsonl",
                        help="JSONL file from replay_stream.py")
    parser.add_argument("--alerts",
                        default="data/generated_alerts.csv",
                        help="Alerts CSV for System 3 evaluator")
    parser.add_argument("--truth",
                        default="data/raw/hidden_ground_truth.csv",
                        help="Hidden ground truth CSV (System 1 output)")
    parser.add_argument("--scored",
                        default="data/evaluation/scored_transactions.csv",
                        help="Supplementary scored transactions CSV (optional)")
    parser.add_argument("--out-dir",
                        default="data/evaluation",
                        help="Output directory for evaluation_report.json")
    parser.add_argument("--threshold-medium", type=float, default=40.0)
    parser.add_argument("--threshold-high",   type=float, default=49.0)
    parser.add_argument("--full",  action="store_true",
                        help="Run full System 3 evaluation (produces evaluation_report.json)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick P/R/F1 evaluation only (default)")
    args = parser.parse_args()

    # Default: run both quick + full
    run_quick = True
    run_full  = args.full or (not args.quick)

    if run_quick:
        quick_evaluate(
            scores_jsonl=args.scores,
            truth_csv=args.truth,
            threshold_medium=args.threshold_medium,
            threshold_high=args.threshold_high,
        )

    if run_full:
        full_evaluate(
            truth_csv=args.truth,
            alerts_csv=args.alerts,
            scored_csv=args.scored,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
