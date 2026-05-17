"""
E3 — Community / Fraud-Ring Detection Evaluator.

A ring is "detected" if >= 50 % of its transactions received an alert.
A ring is "partially detected" if >= 1 transaction was alerted (but < 50 %).
A ring is "fully detected" if all transactions were alerted.

Metrics produced:
  fraud_ring_detection_rate     : rings with >=50% txns alerted / total rings
  partial_ring_detection_rate   : rings with >=1 txn alerted / total rings
  full_ring_detection_rate      : rings with 100% txns alerted / total rings
  community_id_match_score      : fraction of alerted fraud txns that share
                                  the same community_id as the majority of
                                  their ring (requires community_id column)
  group_risk_precision          : fraction of alerted ring members where the
                                  alert risk_level is >= Medium
  mean_ring_coverage            : mean(alerted_txns / total_txns) across rings
  ring_stats                    : per-ring breakdown
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def evaluate_communities(
    joined: pd.DataFrame,
    community_profiles_path: Optional[Path | str] = None,
) -> dict:
    """
    Evaluate fraud-ring detection quality.

    Parameters
    ----------
    joined                   : output of e1_joiner
    community_profiles_path  : optional path to community_profiles.parquet
                               (not required; used for community_id_match_score)

    Returns
    -------
    dict with ring-level metrics.
    """
    # Only rows with a ground-truth fraud label
    fraud = joined[joined["y_true"] == 1].copy()

    if fraud.empty or "fraud_ring_id" not in fraud.columns:
        return _empty_result()

    fraud["alerted"] = (fraud["y_pred"] == 1)

    # Per-ring statistics
    ring_agg = (
        fraud.groupby("fraud_ring_id")
        .agg(
            total_txns=("transaction_id", "count"),
            alerted_txns=("alerted", "sum"),
        )
        .reset_index()
    )
    ring_agg["coverage"]    = ring_agg["alerted_txns"] / ring_agg["total_txns"]
    ring_agg["fully_det"]   = ring_agg["alerted_txns"] == ring_agg["total_txns"]
    ring_agg["det_50pct"]   = ring_agg["coverage"] >= 0.50
    ring_agg["partial_det"] = ring_agg["alerted_txns"] > 0

    total   = len(ring_agg)
    full    = int(ring_agg["fully_det"].sum())
    det50   = int(ring_agg["det_50pct"].sum())
    partial = int(ring_agg["partial_det"].sum())
    missed  = total - partial

    mean_coverage = float(ring_agg["coverage"].mean()) if total > 0 else 0.0

    # ── community_id_match_score ──────────────────────────────────────────
    # For each ring, find the modal community_id among its transactions.
    # Score = fraction of ring transactions where community_id == modal_id
    # (requires "community_id" in joined; skip if absent)
    community_match = _community_match_score(fraud)

    # ── group_risk_precision ──────────────────────────────────────────────
    # Of all alerted fraud transactions, what fraction have risk_level >= Medium?
    alerted_fraud = fraud[fraud["alerted"]]
    if not alerted_fraud.empty and "risk_level" in alerted_fraud.columns:
        medium_or_above = alerted_fraud["risk_level"].isin(
            ["Medium", "High", "Critical"]
        )
        grp_prec = float(medium_or_above.mean())
    else:
        grp_prec = 0.0

    # Per-ring detail (keep it small)
    ring_stats = ring_agg[
        ["fraud_ring_id", "total_txns", "alerted_txns", "coverage", "det_50pct", "fully_det"]
    ].copy()
    ring_stats["coverage"] = ring_stats["coverage"].round(4)
    ring_stats_list = ring_stats.to_dict(orient="records")

    return {
        "ring_ids_total":               total,
        "rings_fully_detected":         full,
        "rings_50pct_detected":         det50,
        "rings_partially_detected":     partial,
        "rings_missed":                 missed,
        "fraud_ring_detection_rate":    round(det50 / total, 4)   if total > 0 else 0.0,
        "partial_ring_detection_rate":  round(partial / total, 4) if total > 0 else 0.0,
        "full_ring_detection_rate":     round(full / total, 4)    if total > 0 else 0.0,
        "mean_ring_coverage":           round(mean_coverage, 4),
        "community_id_match_score":     round(community_match, 4),
        "group_risk_precision":         round(grp_prec, 4),
        "ring_stats":                   ring_stats_list[:50],   # cap for JSON size
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _community_match_score(fraud_df: pd.DataFrame) -> float:
    """
    How well do community assignments align with fraud rings?
    Computed as: mean over rings of (mode_community_id_count / ring_size).
    Returns 0.0 if no community_id column.
    """
    if "community_id" not in fraud_df.columns:
        return 0.0

    scores: list[float] = []
    for _, ring in fraud_df.groupby("fraud_ring_id"):
        if ring.empty:
            continue
        comm_counts = ring["community_id"].value_counts()
        modal_count = int(comm_counts.iloc[0]) if not comm_counts.empty else 0
        scores.append(modal_count / len(ring))

    return float(sum(scores) / len(scores)) if scores else 0.0


def _empty_result() -> dict:
    return {
        "ring_ids_total":              0,
        "rings_fully_detected":        0,
        "rings_50pct_detected":        0,
        "rings_partially_detected":    0,
        "rings_missed":                0,
        "fraud_ring_detection_rate":   0.0,
        "partial_ring_detection_rate": 0.0,
        "full_ring_detection_rate":    0.0,
        "mean_ring_coverage":          0.0,
        "community_id_match_score":    0.0,
        "group_risk_precision":        0.0,
        "ring_stats":                  [],
    }
