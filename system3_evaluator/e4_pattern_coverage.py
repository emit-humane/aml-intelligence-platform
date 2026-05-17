"""
E4 — Per-Typology Pattern Coverage and Rule Precision.

Metrics:
  1. Per synthetic_pattern_type detection rates
     (fraction of fraud transactions of each type that were alerted)
  2. Per triggered_rule precision
     (fraction of alerts triggering each rule that are TRUE fraud)
  3. Alert latency statistics
     (time between transaction timestamp and alert creation, if available)
"""

from __future__ import annotations

import ast
from typing import Optional

import pandas as pd


# Canonical AML typologies (from the synthetic data generator)
TYPOLOGIES = [
    "fraud_ring",
    "structuring",
    "circular_laundering",
    "fan_in",
    "fan_out",
    "velocity_burst",
    "layering_chain",
    "dormant_activation",
    "cross_border_layering",
    "round_tripping",
]


def evaluate_pattern_coverage(
    joined: pd.DataFrame,
    alerts_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Parameters
    ----------
    joined     : E1 joined frame (y_true, y_pred, y_score, synthetic_pattern_type)
    alerts_df  : optional raw alerts DataFrame (for per-rule and latency analysis)
                 Must have columns: triggered_patterns, transaction_risk_score,
                 created_at (and optionally timestamp).

    Returns
    -------
    dict with keys:
      per_typology          : {typology: {total, detected, detection_rate}}
      per_rule_precision    : {rule_id: {triggered, true_fraud, precision}}
      latency_stats         : {mean_ms, median_ms, p95_ms} or None
      overall_detection_rate: fraction of all fraud txns detected
    """
    results: dict = {}

    # ── Per-typology detection rates ──────────────────────────────────────
    fraud = joined[joined["y_true"] == 1].copy()
    pat_col = next(
        (c for c in ("synthetic_pattern_type", "pattern_type") if c in fraud.columns),
        None,
    )

    per_typ: dict[str, dict] = {}
    if not fraud.empty and pat_col:
        for typology in TYPOLOGIES:
            subset = fraud[fraud[pat_col] == typology]
            total    = len(subset)
            detected = int(subset["y_pred"].sum()) if total > 0 else 0
            rate     = detected / total if total > 0 else 0.0
            per_typ[typology] = {
                "total":          total,
                "detected":       detected,
                "detection_rate": round(rate, 4),
            }
    else:
        per_typ = {t: {"total": 0, "detected": 0, "detection_rate": 0.0} for t in TYPOLOGIES}

    overall_fraud = len(fraud)
    overall_det   = int(fraud["y_pred"].sum()) if not fraud.empty else 0
    overall_rate  = overall_det / overall_fraud if overall_fraud > 0 else 0.0

    results["per_typology"]           = per_typ
    results["overall_detection_rate"] = round(overall_rate, 4)

    # ── Per-rule precision ─────────────────────────────────────────────────
    results["per_rule_precision"] = _per_rule_precision(joined, alerts_df)

    # ── Alert latency ─────────────────────────────────────────────────────
    results["latency_stats"] = _latency_stats(alerts_df)

    return results


# ---------------------------------------------------------------------------
# Per-rule precision
# ---------------------------------------------------------------------------

def _per_rule_precision(joined: pd.DataFrame, alerts_df: Optional[pd.DataFrame]) -> dict:
    """
    For each rule that fired in alerts, compute what fraction of its alerts
    correspond to actual fraud transactions.

    Requires alerts_df with column 'triggered_patterns' (pipe-separated or list).
    """
    if alerts_df is None or alerts_df.empty:
        return {}

    if "triggered_patterns" not in alerts_df.columns:
        return {}

    # Build fraud tx set
    fraud_ids = set(joined.loc[joined["y_true"] == 1, "transaction_id"])

    rule_stats: dict[str, dict] = {}

    for _, row in alerts_df.iterrows():
        tx_id   = str(row.get("transaction_id", ""))
        is_fraud = tx_id in fraud_ids

        patterns_raw = row["triggered_patterns"]
        if pd.isna(patterns_raw):
            continue

        # Pipe-separated string or Python list
        if isinstance(patterns_raw, str) and "|" in patterns_raw:
            rules = [r.strip() for r in patterns_raw.split("|")]
        elif isinstance(patterns_raw, str) and patterns_raw.startswith("["):
            try:
                rules = ast.literal_eval(patterns_raw)
            except Exception:
                rules = [patterns_raw]
        elif isinstance(patterns_raw, list):
            rules = patterns_raw
        else:
            rules = [str(patterns_raw)]

        for rule in rules:
            if not rule:
                continue
            if rule not in rule_stats:
                rule_stats[rule] = {"triggered": 0, "true_fraud": 0}
            rule_stats[rule]["triggered"] += 1
            if is_fraud:
                rule_stats[rule]["true_fraud"] += 1

    # Compute precision per rule
    for rule, stats in rule_stats.items():
        stats["precision"] = (
            round(stats["true_fraud"] / stats["triggered"], 4)
            if stats["triggered"] > 0 else 0.0
        )

    return rule_stats


# ---------------------------------------------------------------------------
# Alert latency
# ---------------------------------------------------------------------------

def _latency_stats(alerts_df: Optional[pd.DataFrame]) -> Optional[dict]:
    """
    Latency = time between transaction timestamp and alert created_at.
    Requires 'timestamp' and 'created_at' in alerts_df.
    Returns None if data is insufficient.
    """
    if alerts_df is None or alerts_df.empty:
        return None

    if "timestamp" not in alerts_df.columns or "created_at" not in alerts_df.columns:
        return None

    try:
        import numpy as np
        ts  = pd.to_datetime(alerts_df["timestamp"],   utc=True, errors="coerce")
        cat = pd.to_datetime(alerts_df["created_at"],  utc=True, errors="coerce")
        delta_ms = (cat - ts).dt.total_seconds() * 1000
        delta_ms = delta_ms.dropna()
        if delta_ms.empty:
            return None
        return {
            "mean_ms":   round(float(delta_ms.mean()),          1),
            "median_ms": round(float(delta_ms.median()),        1),
            "p95_ms":    round(float(np.percentile(delta_ms, 95)), 1),
            "p99_ms":    round(float(np.percentile(delta_ms, 99)), 1),
            "min_ms":    round(float(delta_ms.min()),           1),
            "max_ms":    round(float(delta_ms.max()),           1),
            "n":         len(delta_ms),
        }
    except Exception as exc:
        return {"error": str(exc)}
