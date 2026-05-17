"""
CLI entry point for System 1 data generation.

Usage:
    python -m scripts.run_generator
    python -m scripts.run_generator --config config/generator_config.json
    python -m scripts.run_generator --config config/generator_config.json --out-dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

from system1_generator.config import GeneratorConfig
from system1_generator.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="AML data generator (System 1)")
    parser.add_argument(
        "--config",
        default="config/generator_config.json",
        help="Path to generator_config.json",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw",
        help="Output directory for all generated files",
    )
    args = parser.parse_args()

    cfg = GeneratorConfig.from_json(args.config)
    out_dir = Path(args.out_dir)

    print(f"[run_generator] config: {args.config}")
    print(f"[run_generator] out_dir: {out_dir}")
    print(f"[run_generator] seed={cfg.seed}, accounts={cfg.num_accounts}, "
          f"hist_txns={cfg.num_historical_transactions}, stream_txns={cfg.num_stream_transactions}")

    run_pipeline(cfg, out_dir)


if __name__ == "__main__":
    main()
