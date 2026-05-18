"""
Offline Training Pipeline - S1 -> S2 -> L2A -> L2B -> L3A -> L4A -> S3.save()

Orchestrates all offline modules in dependency order:

  S1  FeatureStore            - build per-account rolling feature windows
  S2  GraphBuilder            - build NetworkX MultiDiGraph from historical data
  L2A GraphFeaturePreprocessor - compute per-node graph features (PageRank, etc.)
  L2B AnalyticsEngine         - community profiles + suspicious paths
  L3A BehavioralTraining      - train IsoForest / LOF / AutoEncoder
  L4A GNNTraining             - train TGN + MegaGNN
  S3  ArtifactStore.save()    - persist everything + write manifest.json

Usage:
    python -m scripts.run_offline_pipeline
    python -m scripts.run_offline_pipeline --data data/raw/all_transactions.parquet
    python -m scripts.run_offline_pipeline --skip-l3   # skip behavioral models
    python -m scripts.run_offline_pipeline --skip-l4   # skip GNN training
    python -m scripts.run_offline_pipeline --skip-l3 --skip-l4  # graph only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(step: str, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  [{step}] {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_path:   Path,
    artifact_dir: Path,
    out_dir:     Path,
    seed:        int  = 42,
    skip_l3:     bool = False,
    skip_l4:     bool = False,
) -> dict:
    """
    Full offline pipeline.  Returns a summary dict.
    All steps are idempotent: re-running overwrites existing artifacts.
    """
    from system2_platform.shared.s3_artifact_store import ArtifactStore

    store = ArtifactStore(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    summary: dict = {}

    # ── Load data ──────────────────────────────────────────────────────────────
    _banner("DATA", f"Loading {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Historical parquet not found: {data_path}\n"
            f"Run 'make generate' first."
        )
    df = pd.read_parquet(data_path)
    n_tx  = len(df)
    n_acc = df["sender_account"].nunique()
    print(f"  {n_tx:,} transactions | {n_acc:,} unique sender accounts")
    summary["n_transactions"] = n_tx
    summary["n_accounts"]     = n_acc

    # ── S2 - Build MultiDiGraph ────────────────────────────────────────────────
    _banner("S2", "Building MultiDiGraph (GraphBuilder)")
    t0 = time.time()
    from system2_platform.shared.s2_multigraph_builder import GraphBuilder
    gb = GraphBuilder()
    G  = gb.build(df)
    print(f"  Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,} "
          f"({time.time()-t0:.1f}s)")
    store.save("multigraph", G)
    # Also save a lightweight node/edge index (faster to load than full graph)
    all_nodes  = sorted(G.nodes())
    node_index = {n: i for i, n in enumerate(all_nodes)}
    store.save("node_index", node_index)
    summary["n_nodes"] = G.number_of_nodes()
    summary["n_edges"] = G.number_of_edges()

    # ── L2A - Graph Features ───────────────────────────────────────────────────
    _banner("L2A", "Computing per-node graph features (PageRank, Betweenness, Cycles…)")
    t0 = time.time()
    from system2_platform.layer2_graph.l2a_feature_preprocessor import build_graph_features
    graph_features = build_graph_features(G, out_dir=out_dir / "cache")
    store.save("graph_features", graph_features)
    print(f"  graph_features: {graph_features.shape}  ({time.time()-t0:.1f}s)")
    summary["n_graph_feature_nodes"] = len(graph_features)

    # ── L2B - Community Analytics ──────────────────────────────────────────────
    _banner("L2B", "Running community analytics (Infomap + suspicious paths)")
    t0 = time.time()
    from system2_platform.layer2_graph.l2b_analytics_engine import run_analytics
    community_profiles, suspicious_paths = run_analytics(
        G, graph_features, out_dir=out_dir / "cache"
    )
    store.save("community_profiles", community_profiles)
    store.save("suspicious_paths",   suspicious_paths)
    print(f"  {len(community_profiles)} communities | "
          f"{len(suspicious_paths)} suspicious paths  ({time.time()-t0:.1f}s)")
    summary["n_communities"]     = len(community_profiles)
    summary["n_suspicious_paths"] = len(suspicious_paths)

    # ── Build behavioral profiles (fast groupby - for later use by S6 / API) ──
    _banner("S1", "Building per-account behavioral profiles (S1 FeatureStore groupby)")
    t0 = time.time()
    from system2_platform.layer3_behavioral.l3a_training import _build_account_profiles
    behavioral_profiles = _build_account_profiles(df)
    behavioral_profiles.to_parquet(out_dir / "behavioral_profiles.parquet", index=False)
    store.save("behavioral_profiles", behavioral_profiles)
    print(f"  {len(behavioral_profiles)} account profiles  ({time.time()-t0:.1f}s)")
    summary["n_behavioral_profiles"] = len(behavioral_profiles)

    # ── L3A - Behavioral Anomaly Models ────────────────────────────────────────
    if not skip_l3:
        _banner("L3A", "Training behavioral anomaly models (IsoForest + LOF + AutoEncoder)")
        t0 = time.time()
        from system2_platform.layer3_behavioral.l3a_training import (
            train_models_only,
        )
        # Pass the FULL transaction stream. train_models_only replays every
        # row in timestamp order to build true per-account rolling history,
        # then fits the scaler/models on a per-account-capped sample of NORMAL
        # transactions whose features were computed WITH that real history.
        # (The old code pre-capped/subsampled BEFORE the replay, which gave
        # every account near-empty history and produced an inverted L3 — see
        # scripts/diagnose_l3_discrimination.py.)
        l3_summary = train_models_only(
            history_df     = df,
            graph_features = graph_features,
            artifact_store = store,
            seed           = seed,
        )
        print(f"  L3A done in {time.time()-t0:.1f}s  | summary: {l3_summary}")
        summary.update({f"l3a_{k}": v for k, v in l3_summary.items()})
    else:
        print("\n[SKIP] L3A - behavioral models (--skip-l3)")

    # ── L4A - GNN Training ─────────────────────────────────────────────────────
    if not skip_l4:
        _banner("L4A", "Training TGN + MegaGNN (self-supervised link prediction)")
        t0 = time.time()
        from system2_platform.layer4_gnn.l4a_training import train as train_gnn
        l4_summary = train_gnn(
            historical_parquet = data_path,
            artifact_dir       = artifact_dir,
            seed               = seed,
        )
        print(f"  L4A done in {time.time()-t0:.1f}s  | summary: {l4_summary}")
        summary.update({f"l4a_{k}": v for k, v in l4_summary.items()})
    else:
        print("\n[SKIP] L4A - GNN training (--skip-l4)")

    # ── S3.save() - Write manifest ─────────────────────────────────────────────
    _banner("S3", "Writing artifact manifest")
    store.save_manifest()

    elapsed = time.time() - total_start
    summary["pipeline_elapsed_seconds"] = round(elapsed, 1)
    _banner("DONE", f"Offline pipeline complete in {elapsed:.0f}s")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AML offline pipeline: S1->S2->L2A->L2B->L3A->L4A->S3.save()"
    )
    parser.add_argument(
        "--data", default="data/raw/all_transactions.parquet",
        help="Path to historical transactions parquet"
    )
    parser.add_argument(
        "--artifacts", default="data/artifacts",
        help="Artifact output directory"
    )
    parser.add_argument(
        "--out", default="data",
        help="Parquet output directory (behavioral_profiles, cache)"
    )
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--skip-l3",  action="store_true",
                        help="Skip behavioral model training (use existing artifacts)")
    parser.add_argument("--skip-l4",  action="store_true",
                        help="Skip GNN training (use existing artifacts)")
    args = parser.parse_args()

    run_pipeline(
        data_path    = Path(args.data),
        artifact_dir = Path(args.artifacts),
        out_dir      = Path(args.out),
        seed         = args.seed,
        skip_l3      = args.skip_l3,
        skip_l4      = args.skip_l4,
    )


if __name__ == "__main__":
    main()
