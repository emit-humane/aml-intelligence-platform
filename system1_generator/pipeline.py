"""
System 1 Generator — full pipeline.

Orchestrates G1 -> G2 -> G3 -> G4 in order.
All randomness flows from a single seeded numpy Generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import GeneratorConfig
from .g1_account_builder import build_accounts
from .g2_normal_transactions import build_normal_transactions
from .g3_scenario_injector import inject_scenarios
from .g4_splitter_exporter import export_splits


def run_pipeline(cfg: GeneratorConfig, out_dir: Path) -> None:
    """
    Run the full data-generation pipeline.

    Writes to out_dir:
      accounts.parquet
      normal_transactions.parquet
      fraud_transactions.parquet
      all_transactions.parquet
      historical_transactions.csv
      stream_transactions.csv
      hidden_ground_truth.csv
    """
    rng = np.random.default_rng(cfg.seed)

    print("[Pipeline] G1: building accounts...")
    accounts = build_accounts(cfg, rng, out_dir)

    print("[Pipeline] G2: building normal transactions...")
    normal_df = build_normal_transactions(cfg, accounts, rng, out_dir)

    print("[Pipeline] G3: injecting fraud scenarios...")
    fraud_df, combined_df, all_rings = inject_scenarios(cfg, accounts, normal_df, rng, out_dir)

    print("[Pipeline] G4: splitting and exporting CSVs...")
    export_splits(cfg, combined_df, all_rings, out_dir)

    print("[Pipeline] Done.")
