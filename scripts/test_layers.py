"""
Per-layer detection test harness.

Goal: a curated subset of REAL transactions that exercises each detection
layer separately, so you can see whether each layer fires on what it should
— and that normal transactions stay quiet across ALL layers.

Why real rows: the pipeline is stateful + graph-based. The GNN, graph_boost
and behavioral rolling windows only have signal for accounts that exist in
the seeded history/offline graph and for 2026-era timestamps. Fabricated
accounts make the strong layers blind (see send_demo_transactions.ps1
history). So every test row is sampled from the real datasets.

Layer  -> component in score_breakdown        targeted by
  L1   rule           (rule_score, 15 FATF)   real stream txns engineered to
                                               trip a specific rule (high
                                               value / high-risk country /
                                               sub-threshold) — normal account
                                               otherwise => ISOLATION test
  L2   graph_boost     (community/cycle risk)  structural fraud typologies
  L3   behavioral      (iso+lof+ae ensemble)   behavioral fraud typologies
  L4   gnn             (TGN/MegaGNN structural)structural fraud typologies
  --   NORMAL control                          random real normals: every
                                               component must stay low

"Detected by layer L" is decided data-driven: a component FIRES for a txn
if it exceeds the 90th percentile of that component over the NORMAL control
sample (so normals fire ~<=10% by construction; a working layer fires far
more often on its targeted fraud).

Usage:
  python -m scripts.test_layers
  python -m scripts.test_layers --base-url http://localhost:8000 --per-layer 6 --normal 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

STREAM_CSV = Path("data/raw/stream_transactions.csv")
TRUTH_CSV  = Path("data/raw/hidden_ground_truth.csv")

PAYLOAD_COLS = [
    "transaction_id", "timestamp", "sender_account", "receiver_account",
    "sender_name", "receiver_name", "sender_bank", "receiver_bank",
    "sender_country", "receiver_country", "amount", "currency",
    "transaction_type", "payment_channel", "device_id", "ip_address",
    "geo_latitude", "geo_longitude", "merchant_category", "transaction_status",
    "sender_balance_before", "sender_balance_after",
    "receiver_balance_before", "receiver_balance_after",
    "kyc_level", "is_international", "remarks",
]

HIGH_RISK = {"AE", "MU", "CN", "NG", "PK", "CH"}

# fraud synthetic_pattern_type -> which layer it primarily exercises
BEHAVIORAL_TYPOLOGIES = {"velocity_burst", "dormant_activation", "structuring"}
STRUCTURAL_TYPOLOGIES = {
    "fan_in", "fan_out", "circular_laundering", "fraud_ring",
    "layering_chain", "cross_border_layering", "round_tripping",
}


def _payload(row: pd.Series) -> dict:
    p = {}
    for c in PAYLOAD_COLS:
        if c not in row:
            continue
        v = row[c]
        if pd.isna(v):
            v = None
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        p[c] = v
    return p


def _score(session: requests.Session, url: str, row: pd.Series) -> dict | None:
    try:
        r = session.post(url, json=_payload(row), timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _pick(df: pd.DataFrame, mask, n: int, rng) -> pd.DataFrame:
    sub = df[mask]
    if len(sub) == 0:
        return sub
    return sub.sample(min(n, len(sub)), random_state=int(rng.integers(1 << 30)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url",
                    default="https://aml-intelligence-platform.onrender.com")
    ap.add_argument("--per-layer", type=int, default=5)
    ap.add_argument("--normal", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    url = f"{args.base_url}/stream/transaction"
    rng = np.random.default_rng(args.seed)

    stream = pd.read_csv(STREAM_CSV)
    truth = pd.read_csv(TRUTH_CSV)
    stream["transaction_id"] = stream["transaction_id"].astype(str)
    truth["transaction_id"] = truth["transaction_id"].astype(str)
    truth_ids = set(truth["transaction_id"])
    normals = stream[~stream["transaction_id"].isin(truth_ids)].copy()

    N = args.per_layer
    # (group, expect_layer, expect_rule_id_or_None, row)
    cases: list[tuple[str, str, str | None, pd.Series]] = []

    # ── L1 rule-isolation: real normal-account txns engineered to trip ONE
    # single-transaction rule. L1 "detects" = expected rule id in
    # triggered_rules (the diluted rule_score component is a separate axis).
    for _, r in _pick(normals, normals["amount"] > 1_000_000, N, rng).iterrows():
        cases.append(("L1:R01 high-value", "rule", "R01", r))
    for _, r in _pick(normals, normals["receiver_country"].isin(HIGH_RISK),
                      N, rng).iterrows():
        cases.append(("L1:R06 high-risk-ctry", "rule", "R06", r))
    rnd = (normals["amount"] % 100_000 == 0) & (normals["amount"] >= 100_000)
    for _, r in _pick(normals, rnd, N, rng).iterrows():
        cases.append(("L1:R11 round-amount", "rule", "R11", r))

    # ── L3 behavioral: behavioral fraud typologies
    bmask = truth["synthetic_pattern_type"].isin(BEHAVIORAL_TYPOLOGIES)
    for _, r in _pick(truth, bmask, N, rng).iterrows():
        cases.append((f"L3:{r['synthetic_pattern_type']}", "behavioral", None, r))

    # ── L4/L2 structural: structural fraud typologies (GNN + graph)
    smask = truth["synthetic_pattern_type"].isin(STRUCTURAL_TYPOLOGIES)
    for _, r in _pick(truth, smask, N, rng).iterrows():
        cases.append((f"L4:{r['synthetic_pattern_type']}", "gnn", None, r))

    # ── Normal control: must stay quiet on ALL layers
    norm_ctrl = normals.sample(min(args.normal, len(normals)),
                               random_state=args.seed)
    for _, r in norm_ctrl.iterrows():
        cases.append(("NORMAL", "none", None, r))

    print(f"[layers] scoring {len(cases)} transactions via {url}")
    s = requests.Session()
    rows = []
    for i, (grp, expect, exp_rule, r) in enumerate(cases):
        resp = _score(s, url, r)
        if resp is None:
            print(f"  [{i}] {grp:<24} {r['transaction_id']:<16} FAILED")
            continue
        b = resp.get("score_breakdown", {})
        trig = list(resp.get("rule_out", {}).get("triggered_rules", []))
        rows.append({
            "group": grp, "expect": expect, "exp_rule": exp_rule,
            "tid": r["transaction_id"],
            "rule": float(b.get("rule", 0)),
            "graph_boost": float(b.get("graph_boost", 0)),
            "behavioral": float(b.get("behavioral", 0)),
            "gnn": float(b.get("gnn", 0)),
            "fused": float(resp.get("transaction_risk_score", 0)),
            "rules": ",".join(trig),
            "exp_rule_hit": bool(exp_rule and exp_rule in trig),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[layers] no responses — is the server up?")
        return

    # Data-driven 'fired' threshold = normal-control p90 per component
    nc = df[df.group == "NORMAL"]
    comps = ["rule", "graph_boost", "behavioral", "gnn"]
    p90 = {c: float(np.percentile(nc[c], 90)) if len(nc) else 0.0 for c in comps}
    print("\n[layers] normal-control p90 (a component 'fires' above this): "
          + "  ".join(f"{c}={p90[c]:.1f}" for c in comps))

    print(f"\n{'group':<22}{'tid':<15}{'RULE':>6}{'GRAPH':>6}{'BEHAV':>6}"
          f"{'GNN':>6}{'fuse':>6}  fired_components / triggered_rules")
    print("-" * 100)
    for _, x in df.iterrows():
        fired = [c.split('_')[0].upper()[:5] for c in comps if x[c] > p90[c]]
        mark = ",".join(fired) if fired else "-"
        rl = x["rules"][:34] if x["rules"] else "-"
        print(f"{x.group:<22}{x.tid:<15}{x.rule:>6.1f}{x.graph_boost:>6.1f}"
              f"{x.behavioral:>6.1f}{x.gnn:>6.1f}{x.fused:>6.1f}  {mark:<22} [{rl}]")

    # ── Per-layer verdict ────────────────────────────────────────────────────
    layer_comp = {"behavioral": "behavioral", "gnn": "gnn"}
    print("\n[layers] ===== PER-LAYER VERDICT =====")
    for grp in sorted(df["group"].unique()):
        if grp == "NORMAL":
            continue
        g = df[df.group == grp]
        expect = g["expect"].iloc[0]
        if expect == "rule":
            # L1 'detects' = expected rule id appears in triggered_rules
            # (rule_score is intentionally tiny: 1/15 rules, weight 0.02)
            hit = g["exp_rule_hit"].mean()
            er = g["exp_rule"].iloc[0]
            print(f"  {grp:<22} expect rule {er:<4} in triggered_rules: "
                  f"fired on {hit*100:4.0f}% of {len(g)}  "
                  f"(rule_score median={g['rule'].median():.1f} - low by "
                  f"design: 1/15 rules, AUROC~0.28, weight 0.02)")
        else:
            comp = layer_comp[expect]
            hit = (g[comp] > p90[comp]).mean()
            print(f"  {grp:<22} expect={expect:<10} "
                  f"{comp} fired on {hit*100:4.0f}% of {len(g)}  "
                  f"(median {comp}={g[comp].median():.1f})")

    # Normal cleanliness — the real test is the FUSED score (what alerts),
    # not per-component p90 (4 components x 10% => ~25-34% union by definition).
    if len(nc):
        any_fire = (np.column_stack([nc[c] > p90[c] for c in comps])
                    .any(axis=1)).mean()
        alerted = (nc["fused"] >= 45).mean()   # v4 alert threshold
        print(f"\n  NORMAL control ({len(nc)} txns):")
        print(f"    FUSED alerted (>=45): {alerted*100:.0f}%   "
              f"fused median={nc['fused'].median():.1f} "
              f"p90={np.percentile(nc['fused'],90):.1f}  "
              f"(<- the real 'normals stay quiet' test)")
        print(f"    any single component > its own p90: {any_fire*100:.0f}% "
              f"(statistical artifact of 4x p90 union, not real alerts)")


if __name__ == "__main__":
    main()
