"""
Fit the P1 fused-score calibration (Platt / logistic).

PROBLEM: the raw weighted-sum fused score discriminates near-perfectly by
RANK (AUROC ~0.977) but is crushed into a ~28-51 band — the dominant GNN
component itself only spans ~25-50. So fraud (~48) and normal (~41) look
"almost the same" to an analyst/dashboard even though they are correctly
ordered.

FIX: a monotone logistic map  calibrated = 100*sigmoid(A*raw + B)  fit on the
validation set. Monotone => AUROC/ranking is exactly preserved, but the
output now uses the full 0-100 range and reads as a calibrated fraud
probability (normal ~0-20, fraud ~85-99).

This script fits (A, B) from the current ordered-eval outputs and prints the
constants to paste into p1_risk_fusion.py plus the recalibrated bands.

Run:  python -m scripts.fit_score_calibration
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FRAUD_CSV = Path("data/evaluation/scored_transactions.csv")
NORMAL_JSONL = Path("data/evaluation/normal_sample_scores.jsonl")


def _auroc(p: np.ndarray, n: np.ndarray) -> float:
    s = np.concatenate([p, n])
    y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    o = np.argsort(s, kind="mergesort")
    rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    sr = np.zeros(len(c)); np.add.at(sr, inv, rk); rk = (sr / c)[inv]
    return float((rk[y == 1].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main() -> None:
    fr = pd.read_csv(FRAUD_CSV)["y_score"].astype(float).values
    no = []
    for line in open(NORMAL_JSONL):
        line = line.strip()
        if line:
            no.append(float(json.loads(line).get("transaction_risk_score", 0.0)))
    no = np.array(no)

    X = np.concatenate([fr, no]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(fr)), np.zeros(len(no))])
    lr = LogisticRegression(C=1e6).fit(X, y)        # ~unregularised Platt
    A = float(lr.coef_[0, 0]); B = float(lr.intercept_[0])
    cal = lambda r: 100.0 / (1.0 + np.exp(-(A * np.asarray(r, float) + B)))
    cf, cn = cal(fr), cal(no)

    print(f"_CAL_A = {A:.6f}")
    print(f"_CAL_B = {B:.6f}")
    print(f"  calibrated = 100 / (1 + exp(-({A:.4f}*raw {B:+.3f})))")
    print(f"  raw decision point (calibrated=50) = raw {(-B / A):.2f}")
    print(f"AUROC raw={_auroc(fr, no):.4f}  calibrated={_auroc(cf, cn):.4f}"
          f"  (monotone -> identical)\n")
    for nm, v in (("fraud", cf), ("normal", cn)):
        print(f"  {nm:<6} p10={np.percentile(v,10):5.1f} p50={np.percentile(v,50):5.1f}"
              f" p90={np.percentile(v,90):5.1f}")

    ally = np.concatenate([np.ones(len(cf)), np.zeros(len(cn))])
    alls = np.concatenate([cf, cn])

    def metr(t):
        pr = alls >= t
        tp = int(((ally == 1) & pr).sum()); fp = int(((ally == 0) & pr).sum())
        fn = int(((ally == 1) & ~pr).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        return P, R, F

    print("\n  thr   P     R     F1")
    best = (-1, 0)
    for t in np.arange(0, 100.5, 0.5):
        P, R, F = metr(t)
        if F > best[0]:
            best = (F, t)
    for t in [40, 50, best[1], 60, 75, 90]:
        P, R, F = metr(t)
        tag = "  <- F1-opt" if abs(t - best[1]) < 1e-6 else ""
        print(f"  {t:>4.0f}  {P:.3f} {R:.3f} {F:.3f}{tag}")

    thr = round(best[1])
    print(f"\nRECOMMENDED (calibrated scale, ~= P(fraud)*100):")
    print(f"  alert threshold = {thr}")
    print(f"  bands: Low <{thr}  Medium [{thr},80)  High [80,95)  Critical >=95")


if __name__ == "__main__":
    main()
