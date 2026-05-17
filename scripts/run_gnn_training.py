"""
CLI entry point for Layer 4 GNN training (L4A).

Usage:
    python -m scripts.run_gnn_training
    python -m scripts.run_gnn_training --data data/raw/all_transactions.parquet --artifacts data/artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path

from system2_platform.layer4_gnn.l4a_training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="AML GNN training (L4A)")
    parser.add_argument("--data", default="data/raw/all_transactions.parquet")
    parser.add_argument("--artifacts", default="data/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = train(
        historical_parquet=Path(args.data),
        artifact_dir=Path(args.artifacts),
        seed=args.seed,
    )
    print("\n[run_gnn_training] Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
