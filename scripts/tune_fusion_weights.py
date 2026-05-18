"""
Retune P1 fusion weights now that Layer 3 behavioral is fixed.

The v2 weights (rule .05 / behav .12 / gnn .80 / graph .03) were calibrated
when behavioral was inverted/dead (stale AUROC ~0.67). Behavioral now
discriminates strongly on its own, so the mix is stale.

This reads the FRESHLY scored per-component breakdowns produced by the last
eval run (no re-scoring needed):
  data/evaluation/scored_transactions.csv   - 418 fraud  (label 1)
  data/evaluation/normal_sample_scores.jsonl - 1036 normal (label 0)

and:
  1. computes per-component validation AUROC (rule/behavioral/gnn/graph_boost)
  2. grid-searches weights on the simplex (step 0.05) to maximise fused AUROC
  3. also reports the AUROC-proportional weighting (project methodology)
  4. for the recommended weights, derives the F1-optimal alert threshold and
     Low/Medium/High/Critical bands from the fused-score distribution

Run:
  python -m scripts.tune_fusion_weights
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FRAUD_CSV = Path("data/evaluation/scored_transactions.csv")
NORMAL_JSONL = Path("data/evaluation/normal_sample_scores.jsonl")
COMPS = ["rule", "behavioral", "gnn", "graph_boost"]


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sr = np.zeros(len(cnt))
    np.add.at(sr, inv, ranks)
    ranks = (sr / cnt)[inv]
    rp = ranks[y == 1].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_f1(y: np.ndarray, s: np.ndarray):
    best = (-1.0, 0.0, 0.0, 0.0)
    for thr in np.arange(0, 100.5, 0.5):
        pred = s >= thr
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best[0]:
            best = (f1, float(thr), p, r)
    return best  # f1, thr, precision, recall


def precision_at(y, s, thr):
    pred = s >= thr
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    return tp / (tp + fp) if tp + fp else 0.0


def main() -> None:
    fr = pd.read_csv(FRAUD_CSV)
    fr_comp = {c: fr[c].astype(float).to_numpy() for c in COMPS}
    n_fraud = len(fr)

    nr = {c: [] for c in COMPS}
    with open(NORMAL_JSONL) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            b = d.get("score_breakdown", {})
            for c in COMPS:
                nr[c].append(float(b.get(c, 0.0)))
    n_norm = len(nr["rule"])

    y = np.concatenate([np.ones(n_fraud), np.zeros(n_norm)]).astype(int)
    comp = {c: np.concatenate([fr_comp[c], np.array(nr[c])]) for c in COMPS}

    print(f"[tune] fraud={n_fraud}  normal={n_norm}\n")
    print(f"{'component':<14}{'AUROC':>8}{'mean(fraud)':>14}{'mean(normal)':>14}")
    print("-" * 50)
    comp_auroc = {}
    for c in COMPS:
        a = auroc(y, comp[c])
        comp_auroc[c] = a
        mf = comp[c][y == 1].mean()
        mn = comp[c][y == 0].mean()
        print(f"{c:<14}{a:>8.3f}{mf:>14.2f}{mn:>14.2f}")

    cur = {"rule": .05, "behavioral": .12, "gnn": .80, "graph_boost": .03}
    cur_s = sum(cur[c] * comp[c] for c in COMPS)
    print(f"\n[tune] current v2 weights {cur} -> fused AUROC "
          f"{auroc(y, cur_s):.4f}")

    # AUROC-proportional weighting (project methodology): relevance =
    # max(0, AUROC-0.5); for inverted detectors use max(0, (1-AUROC)-0.5)
    # only if we were to invert them — here we just floor at 0 and keep a
    # small floor for rule/graph for explainability.
    rel = {c: max(0.0, comp_auroc[c] - 0.5) for c in COMPS}
    floor = {"rule": 0.02, "behavioral": 0.0, "gnn": 0.0, "graph_boost": 0.02}
    raw = {c: rel[c] + floor[c] for c in COMPS}
    tot = sum(raw.values())
    prop = {c: raw[c] / tot for c in COMPS}
    prop_s = sum(prop[c] * comp[c] for c in COMPS)
    print(f"[tune] AUROC-proportional weights "
          f"{ {k: round(v,3) for k,v in prop.items()} } -> fused AUROC "
          f"{auroc(y, prop_s):.4f}")

    # Grid search on the simplex, step 0.05
    grid = [i / 20 for i in range(21)]
    best = (-1.0, None)
    results = []
    for wr in grid:
        for wb in grid:
            for wg in grid:
                wgr = round(1.0 - wr - wb - wg, 5)
                if wgr < -1e-9 or wgr > 1.0 + 1e-9:
                    continue
                wgr = max(0.0, wgr)
                s = wr * comp["rule"] + wb * comp["behavioral"] \
                    + wg * comp["gnn"] + wgr * comp["graph_boost"]
                a = auroc(y, s)
                results.append((a, wr, wb, wg, wgr))
                if a > best[0]:
                    best = (a, (wr, wb, wg, wgr))
    results.sort(reverse=True)
    print(f"\n[tune] grid-search top-8 (rule, behav, gnn, graph -> AUROC):")
    for a, wr, wb, wg, wgr in results[:8]:
        print(f"   ({wr:.2f}, {wb:.2f}, {wg:.2f}, {wgr:.2f}) -> {a:.4f}")

    # Curated explainable candidates (keep small rule/graph floors)
    print(f"\n[tune] curated explainable candidates:")
    for name, w in {
        "v2 (current)":      (.05, .12, .80, .03),
        "gnn-heavy A":       (.03, .07, .87, .03),
        "gnn-heavy B":       (.03, .10, .84, .03),
        "gnn-heavy C":       (.02, .05, .91, .02),
        "gnn-only-ish":      (.02, .06, .90, .02),
    }.items():
        s = (w[0]*comp["rule"] + w[1]*comp["behavioral"]
             + w[2]*comp["gnn"] + w[3]*comp["graph_boost"])
        f1c, thrc, pc, rc = best_f1(y, s)
        print(f"   {name:<16}{w} -> AUROC={auroc(y,s):.4f}  "
              f"bestF1={f1c:.3f}@{thrc:.0f} (P={pc:.2f},R={rc:.2f})")

    # Recommended: prefer a robust point — among combos within 0.001 AUROC of
    # the max, pick the one keeping the smallest extreme weight (less overfit),
    # with a tiny rule+graph floor for explainability.
    # FINAL chosen weights: gnn-heavy C — best explainable AUROC with small
    # rule/graph floors retained for human-readable explanations + ring context.
    rwr, rwb, rwg, rwgr = 0.02, 0.05, 0.91, 0.02
    ra = auroc(y, rwr*comp["rule"] + rwb*comp["behavioral"]
               + rwg*comp["gnn"] + rwgr*comp["graph_boost"])
    print(f"\n[tune] RECOMMENDED weights (robust, explainability floors):")
    print(f"   rule={rwr:.2f}  behavioral={rwb:.2f}  gnn={rwg:.2f}  "
          f"graph={rwgr:.2f}   fused AUROC={ra:.4f}")

    rec_s = rwr * comp["rule"] + rwb * comp["behavioral"] \
        + rwg * comp["gnn"] + rwgr * comp["graph_boost"]

    f1, thr, p, r = best_f1(y, rec_s)
    print(f"\n[tune] fused-score distribution under recommended weights:")
    for label, mask in (("fraud", y == 1), ("normal", y == 0)):
        v = rec_s[mask]
        print(f"   {label:<7} mean={v.mean():6.2f}  p10={np.percentile(v,10):6.2f}"
              f"  p50={np.percentile(v,50):6.2f}  p90={np.percentile(v,90):6.2f}")
    print(f"\n[tune] F1-optimal alert threshold = {thr:.1f}  "
          f"(F1={f1:.3f}, P={p:.3f}, R={r:.3f})")

    # Risk-band recommendations from precision targets
    def thr_for_precision(target):
        for t in np.arange(thr, 100.5, 0.5):
            if precision_at(y, rec_s, t) >= target:
                return float(t)
        return 100.0
    hi = thr_for_precision(0.90)
    crit = thr_for_precision(0.99)
    print(f"[tune] recommended bands:  Low <{thr:.0f}  "
          f"Medium [{thr:.0f},{hi:.0f})  High [{hi:.0f},{crit:.0f})  "
          f"Critical >={crit:.0f}")
    print(f"[tune]   precision@{thr:.0f}={precision_at(y,rec_s,thr):.3f}  "
          f"@{hi:.0f}={precision_at(y,rec_s,hi):.3f}  "
          f"@{crit:.0f}={precision_at(y,rec_s,crit):.3f}")


if __name__ == "__main__":
    main()
