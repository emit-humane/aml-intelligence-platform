"""
E2 — Transaction-Level Classification Metrics.

Computes the full suite required for a serious AML evaluation:
  precision, recall, f1_macro, minority_class_f1, pr_auc, roc_auc,
  false_positive_rate, false_negative_rate, total_alerts,
  TP, FP, FN, TN — at the operational threshold (44.0, v3-calibrated) and
  at the best-F1 threshold found by sweeping 0–100.

Also sweeps thresholds 0–100 (step 1) for the PR curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_transactions(
    joined: pd.DataFrame,
    score_col: str = "y_score",
    default_threshold: float = 44.0,
    thresholds: list[float] | None = None,
) -> dict:
    """
    Full transaction-level evaluation.

    Parameters
    ----------
    joined            : output of e1_joiner.build_and_save() or join()
    score_col         : continuous risk score column (default 'y_score')
    default_threshold : operational alert threshold (default 44.0, v3-calibrated)
    thresholds        : optional override; default sweeps 0-100 step 1

    Returns
    -------
    dict with keys:
      roc_auc, pr_auc, threshold_results (list), operational_metrics,
      best_f1_metrics, n_total, n_fraud, n_alerted
    """
    if thresholds is None:
        thresholds = list(range(0, 101, 1))

    y_true  = joined["y_true"].fillna(0).astype(int).values
    y_score = joined[score_col].fillna(0.0).values if score_col in joined.columns \
              else joined["y_score"].fillna(0.0).values

    n_total  = len(y_true)
    n_fraud  = int(y_true.sum())
    n_normal = n_total - n_fraud

    # ── Curve metrics ──────────────────────────────────────────────────────
    try:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            precision_recall_curve,
        )
        if n_fraud > 0 and n_normal > 0:
            roc_auc = float(roc_auc_score(y_true, y_score))
            pr_auc  = float(average_precision_score(y_true, y_score))
            pr_precisions, pr_recalls, pr_thresholds = precision_recall_curve(y_true, y_score)
            pr_curve = {
                "precisions": [round(float(p), 4) for p in pr_precisions],
                "recalls":    [round(float(r), 4) for r in pr_recalls],
                "thresholds": [round(float(t), 2) for t in pr_thresholds],
            }
        else:
            roc_auc, pr_auc, pr_curve = 0.0, 0.0, {}
    except Exception:
        roc_auc, pr_auc, pr_curve = 0.0, 0.0, {}

    # ── Threshold sweep ────────────────────────────────────────────────────
    threshold_results: list[dict] = []
    for thr in thresholds:
        row = _metrics_at(y_true, y_score, thr)
        threshold_results.append(row)

    # ── Operational threshold ──────────────────────────────────────────────
    operational = _metrics_at(y_true, y_score, default_threshold)

    # ── Best-F1 threshold ─────────────────────────────────────────────────
    best = max(threshold_results, key=lambda r: r["minority_class_f1"])

    return {
        "roc_auc":             round(roc_auc, 4),
        "pr_auc":              round(pr_auc, 4),
        "n_total":             n_total,
        "n_fraud":             n_fraud,
        "n_normal":            n_normal,
        "n_alerted":           int((y_score >= default_threshold).sum()),
        "operational_metrics": operational,
        "best_f1_metrics":     best,
        "threshold_results":   threshold_results,
        "pr_curve":            pr_curve,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _metrics_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """All metrics at a single score threshold."""
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    n_pos  = tp + fn   # total positives
    n_neg  = fp + tn   # total negatives
    n_pred = tp + fp   # total predicted positive

    precision        = tp / n_pred if n_pred > 0 else 0.0
    recall           = tp / n_pos  if n_pos  > 0 else 0.0
    f1_fraud         = _f1(precision, recall)

    # Negative class (majority)
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg   = _f1(prec_neg, rec_neg)

    f1_macro         = (f1_fraud + f1_neg) / 2
    minority_class_f1 = f1_fraud   # fraud IS the minority class

    fpr = fp / n_neg if n_neg > 0 else 0.0
    fnr = fn / n_pos if n_pos > 0 else 0.0

    return {
        "threshold":          threshold,
        "TP":                 tp,
        "FP":                 fp,
        "FN":                 fn,
        "TN":                 tn,
        "total_alerts":       tp + fp,
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "f1_fraud":           round(f1_fraud, 4),
        "f1_macro":           round(f1_macro, 4),
        "minority_class_f1":  round(minority_class_f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }


def _f1(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
