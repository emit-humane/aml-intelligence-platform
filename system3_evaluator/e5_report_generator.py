"""
E5 — Evaluation Report Generator.

Aggregates outputs from E1–E4 and writes:
  data/evaluation/evaluation_report.json
  data/evaluation/pr_curve.png
  data/evaluation/roc_curve.png
  data/evaluation/confusion_matrix.png
  data/evaluation/pattern_coverage.png
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .e1_joiner import join
from .e2_transaction_evaluator import evaluate_transactions
from .e3_community_evaluator import evaluate_communities
from .e4_pattern_coverage import evaluate_pattern_coverage
from .benchmarks import BENCHMARKS
from . import plotting


def generate_report(
    truth_csv: Path | str = "data/raw/hidden_ground_truth.csv",
    alerts_csv: Path | str = "data/generated_alerts.csv",
    scored_csv: Path | str | None = "data/evaluation/scored_transactions.csv",
    out_dir: Path | str = "data/evaluation",
) -> dict:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    truth_csv   : ground truth CSV
    alerts_csv  : P2 generated_alerts.csv (primary alert source)
    scored_csv  : optional supplementary scored transactions CSV
    out_dir     : directory for report JSON + PNG outputs

    Returns the report dict and writes evaluation_report.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── E1: Join ──────────────────────────────────────────────────────────────
    print("[E5] Joining alerts to ground truth...")
    joined = join(truth_csv, alerts_csv, scored_csv)
    print(f"[E5]   Universe: {len(joined)} rows  "
          f"(fraud={int(joined['y_true'].sum())}, alerted={int(joined['y_pred'].sum())})")

    # ── E2: Transaction-level metrics ─────────────────────────────────────────
    print("[E5] Transaction-level metrics...")
    tx_metrics = evaluate_transactions(joined)

    # ── E3: Community / ring detection ────────────────────────────────────────
    print("[E5] Community/ring detection...")
    community_metrics = evaluate_communities(joined)

    # ── E4: Per-pattern coverage + per-rule precision ─────────────────────────
    print("[E5] Per-pattern coverage...")
    # Pass raw alerts DF for rule-precision and latency analysis
    alerts_df = _load_alerts_df(alerts_csv)
    pattern_coverage = evaluate_pattern_coverage(joined, alerts_df)

    # ── Operational metrics (threshold 44.0, v3-calibrated) ───────────────────
    op = tx_metrics["operational_metrics"]
    best = tx_metrics["best_f1_metrics"]

    # ── Plots ─────────────────────────────────────────────────────────────────
    import numpy as np
    y_true  = joined["y_true"].values.astype(int)
    y_score = joined["y_score"].values.astype(float)
    y_pred  = joined["y_pred"].values.astype(int)

    _safe_plot(plotting.plot_pr_curve,
               y_true, y_score, out_dir / "pr_curve.png",
               title="Precision-Recall Curve — AML Platform")
    _safe_plot(plotting.plot_roc_curve,
               y_true, y_score, out_dir / "roc_curve.png",
               title="ROC Curve — AML Platform")
    _safe_plot(plotting.plot_confusion_matrix,
               y_true, y_pred, out_dir / "confusion_matrix.png",
               title=f"Confusion Matrix (threshold=44.0)")
    if pattern_coverage.get("per_typology"):
        _safe_plot(plotting.plot_pattern_coverage,
                   pattern_coverage["per_typology"],
                   out_dir / "pattern_coverage.png")

    # ── Assemble report ───────────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system":       "AML Intelligence Platform v1.0",
        "dataset": {
            "truth_path":  str(truth_csv),
            "alerts_path": str(alerts_csv),
            "n_total":     tx_metrics["n_total"],
            "n_fraud":     tx_metrics["n_fraud"],
            "n_normal":    tx_metrics["n_normal"],
            "n_alerted":   tx_metrics["n_alerted"],
        },
        "transaction_metrics": {
            "roc_auc":              tx_metrics["roc_auc"],
            "pr_auc":               tx_metrics["pr_auc"],
            "operational_threshold": 44.0,
            "operational": {
                "precision":          op["precision"],
                "recall":             op["recall"],
                "f1_fraud":           op["f1_fraud"],
                "f1_macro":           op["f1_macro"],
                "minority_class_f1":  op["minority_class_f1"],
                "false_positive_rate": op["false_positive_rate"],
                "false_negative_rate": op["false_negative_rate"],
                "TP": op["TP"], "FP": op["FP"],
                "FN": op["FN"], "TN": op["TN"],
                "total_alerts": op["total_alerts"],
            },
            "best_f1_threshold": best["threshold"],
            "best_f1_metrics": {
                "precision":          best["precision"],
                "recall":             best["recall"],
                "f1_fraud":           best["f1_fraud"],
                "f1_macro":           best["f1_macro"],
                "minority_class_f1":  best["minority_class_f1"],
                "TP": best["TP"], "FP": best["FP"],
                "FN": best["FN"], "TN": best["TN"],
            },
            "threshold_results": tx_metrics["threshold_results"],
        },
        "community_metrics":  community_metrics,
        "pattern_coverage":   pattern_coverage,
        "external_benchmarks": BENCHMARKS,
        "benchmark_comparison": _compare_to_benchmarks(tx_metrics),
        "plots": {
            "pr_curve":         str(out_dir / "pr_curve.png"),
            "roc_curve":        str(out_dir / "roc_curve.png"),
            "confusion_matrix": str(out_dir / "confusion_matrix.png"),
            "pattern_coverage": str(out_dir / "pattern_coverage.png"),
        },
    }

    out_path = out_dir / "evaluation_report.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    print(f"[E5] Report written to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[E5] ========== EVALUATION SUMMARY ==========")
    print(f"  ROC-AUC:              {tx_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:               {tx_metrics['pr_auc']:.4f}")
    print(f"  Minority-class F1     (thr=31): {op['minority_class_f1']:.4f}")
    print(f"  F1-macro              (thr=31): {op['f1_macro']:.4f}")
    print(f"  Precision             (thr=31): {op['precision']:.4f}")
    print(f"  Recall                (thr=31): {op['recall']:.4f}")
    print(f"  FPR                   (thr=31): {op['false_positive_rate']:.4f}")
    print(f"  FNR                   (thr=31): {op['false_negative_rate']:.4f}")
    print(f"  Best-F1 threshold:    {best['threshold']}  "
          f"(F1={best['minority_class_f1']:.4f})")
    print(f"  Ring detection rate:  "
          f"{community_metrics.get('fraud_ring_detection_rate', 0.0):.4f}")
    print(f"  Overall pattern det:  "
          f"{pattern_coverage.get('overall_detection_rate', 0.0):.4f}")
    print("[E5] ============================================")

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_alerts_df(alerts_csv: Path | str) -> pd.DataFrame | None:
    try:
        p = Path(alerts_csv)
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None


def _safe_plot(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        print(f"[E5] Plot skipped ({fn.__name__}): {exc}")


def _json_default(obj):
    """Handle numpy / pandas types in JSON serialization."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


def _compare_to_benchmarks(tx_metrics: dict) -> dict:
    our_roc_auc = tx_metrics["roc_auc"]
    our_pr_auc  = tx_metrics["pr_auc"]
    best_f1     = tx_metrics["best_f1_metrics"]["minority_class_f1"]
    op          = tx_metrics["operational_metrics"]
    our_prec    = op["precision"]
    our_rec     = op["recall"]

    comparison: dict = {}
    for name, bm in BENCHMARKS.items():
        delta_auroc = round(our_roc_auc - bm["auroc"], 4)
        delta_f1    = round(best_f1     - bm["f1"],    4)
        comparison[name] = {
            "our_roc_auc":   our_roc_auc,
            "bm_auroc":      bm["auroc"],
            "delta_auroc":   delta_auroc,
            "our_best_f1":   best_f1,
            "bm_f1":         bm["f1"],
            "delta_f1":      delta_f1,
            "our_precision": our_prec,
            "our_recall":    our_rec,
            "beats_auroc":   delta_auroc > 0,
            "beats_f1":      delta_f1 > 0,
        }
    return comparison
