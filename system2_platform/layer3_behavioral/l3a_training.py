"""
L3A — Behavioral Anomaly Model Training.

Offline pipeline:
  1. Load historical_transactions.parquet (normal + labelled fraud)
  2. Build transaction graph (S2 GraphBuilder)
  3. Compute graph features (L2A)
  4. Run L2B analytics to produce community_profiles / suspicious_paths
  5. Compute per-account behavioral profiles (groupby stats)
  6. Merge 45 behavioral features (S1 FeatureStore, sampled) + 7 graph features = 52 dims
  7. Train models on NORMAL transactions only:
       - IsolationForest(n_estimators=200, contamination=0.01)
       - LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=0.01)
       - BehavioralAutoencoder (50 epochs, lr=1e-3, batch=256)
  8. Save artifacts via ArtifactStore
  9. Write behavioral_profiles.parquet

Outputs (data/artifacts/):
  scaler_behavioral.joblib
  isolation_forest.joblib
  lof_model.joblib
  autoencoder.pt
  feature_columns.joblib   -- ordered list of 52 feature names

Outputs (data/):
  behavioral_profiles.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ..shared.s1_feature_engineering import FeatureStore, FEATURE_DIM
from ..shared.s2_multigraph_builder import GraphBuilder
from ..shared.s3_artifact_store import ArtifactStore
from ..layer2_graph.l2a_feature_preprocessor import build_graph_features
from ..layer2_graph.l2b_analytics_engine import run_analytics
from .models.autoencoder import BehavioralAutoencoder, INPUT_DIM

# 7 graph features merged per sender account onto behavioral features
_GRAPH_FEATURE_COLS = [
    "pagerank", "betweenness", "fan_in_score", "fan_out_score",
    "community_density", "benford_chi2_community", "volume_asymmetry",
]

# Maximum transactions to use for model training (sampling for speed)
_MAX_TRAIN_ROWS = 80_000
_MAX_FEATURE_ROWS = 30_000  # FeatureStore is O(n*history) — sample for S1


def _build_account_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast vectorised per-account behavioral profile using pandas groupby.
    Does NOT use the rolling FeatureStore — that's too slow for all 496K rows.
    Used for behavioral_profiles.parquet output.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].between(0, 5) | df["hour"].between(22, 23)
    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    df["is_success"] = df["transaction_status"] == "Success"

    grp = df.groupby("sender_account")

    profiles = pd.DataFrame({
        "account_id": grp["sender_account"].first().index,
        "tx_count": grp.size(),
        "total_sent": grp["amount"].sum(),
        "mean_amount": grp["amount"].mean(),
        "std_amount": grp["amount"].std().fillna(0),
        "max_amount": grp["amount"].max(),
        "min_amount": grp["amount"].min(),
        "unique_receivers": grp["receiver_account"].nunique(),
        "unique_banks": grp["receiver_bank"].nunique(),
        "unique_countries": grp["receiver_country"].nunique(),
        "night_tx_ratio": grp["is_night"].mean(),
        "weekend_tx_ratio": grp["is_weekend"].mean(),
        "success_rate": grp["is_success"].mean(),
        "sub_threshold_count": (df.groupby("sender_account")
                                .apply(lambda g: (g["amount"] < 1_000_000).sum(), include_groups=False)),
    }).reset_index(drop=True)

    # Primary channel and country
    primary_channel = (df.groupby("sender_account")["payment_channel"]
                       .agg(lambda x: x.mode()[0] if len(x) > 0 else "Unknown"))
    primary_country = (df.groupby("sender_account")["receiver_country"]
                       .agg(lambda x: x.mode()[0] if len(x) > 0 else "Unknown"))

    profiles["primary_channel"] = profiles["account_id"].map(primary_channel)
    profiles["primary_country"] = profiles["account_id"].map(primary_country)

    # Simple risk tier based on tx_count and amount
    def _tier(row):
        if row["tx_count"] > 1000 or row["max_amount"] > 5_000_000:
            return "elevated"
        if row["night_tx_ratio"] > 0.5 or row["unique_countries"] > 5:
            return "suspicious"
        return "normal"

    profiles["risk_tier"] = profiles.apply(_tier, axis=1)
    return profiles


def _extract_52_features(
    behavioral_df: pd.DataFrame,
    graph_features_df: pd.DataFrame,
    feature_store: FeatureStore,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute 45 S1 features via FeatureStore, merge 7 graph features = 52 dims.
    behavioral_df is expected to already be the desired sample (pre-filtered).
    Returns (X: ndarray shape (n, 52), feature_names: list[str]).
    """
    sample = behavioral_df.reset_index(drop=True)
    print(f"[L3A] Computing S1 features on {len(sample):,} transactions...")
    t0 = time.time()
    feature_vectors = feature_store.transform(sample)
    print(f"[L3A] S1 features done in {time.time()-t0:.1f}s")

    # Build 45-dim matrix from FeatureStore output
    behavioral_cols = [
        "tx_velocity_1h", "tx_velocity_6h", "tx_velocity_24h", "tx_velocity_7d",
        "avg_amount_7d", "std_amount_7d", "tx_gap_seconds", "night_tx_ratio",
        "weekend_tx_ratio", "hour_of_day", "day_of_week",
        "beneficiary_count_7d", "beneficiary_count_30d", "receiver_entropy",
        "amount_zscore", "avg_daily_volume_30d", "round_amount_flag",
        "sub_threshold_flag", "amount_leading_digit", "benford_chi2_score",
        "geo_distance_km", "country_switch_count_7d", "impossible_travel_flag",
        "high_risk_country_flag", "cross_border_ratio_30d",
        "device_change_count_7d", "new_device_flag", "shared_device_count",
        "ip_change_count_24h",
    ]

    # Use scaled_feature_vector (45-dim) directly from S1
    X_45 = np.array([fv.scaled_feature_vector for fv in feature_vectors], dtype=float)

    # Merge 7 graph features per sender account
    gf_indexed = graph_features_df.set_index("account_id")[_GRAPH_FEATURE_COLS]

    sender_accounts = sample["sender_account"].tolist()
    X_7 = np.zeros((len(sender_accounts), len(_GRAPH_FEATURE_COLS)), dtype=float)
    for i, acc in enumerate(sender_accounts):
        if acc in gf_indexed.index:
            X_7[i] = gf_indexed.loc[acc].values

    X = np.hstack([X_45, X_7])  # (n, 52)

    feature_names = [f"s1_{j}" for j in range(FEATURE_DIM)] + _GRAPH_FEATURE_COLS
    return X, feature_names


def _train_autoencoder(
    X_train: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> BehavioralAutoencoder:
    device = torch.device("cpu")
    model = BehavioralAutoencoder(input_dim=INPUT_DIM).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(batch)
        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / len(X_train)
            print(f"[L3A]   Epoch {epoch+1:3d}/{epochs}  loss={avg:.6f}")

    return model


def train(
    historical_parquet: Path | str,
    artifact_dir: Path | str,
    out_dir: Path | str,
    seed: int = 42,
) -> dict:
    """
    Full offline training pipeline.

    Parameters
    ----------
    historical_parquet : path to data/raw/all_transactions.parquet
    artifact_dir       : where to save models (data/artifacts)
    out_dir            : where to save parquet outputs (data/)

    Returns dict of summary metrics.
    """
    rng = np.random.default_rng(seed)
    artifact_store = ArtifactStore(artifact_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    print(f"[L3A] Loading {historical_parquet}...")
    df = pd.read_parquet(historical_parquet)
    print(f"[L3A] {len(df):,} transactions, {df['sender_account'].nunique():,} accounts")

    # ---------------------------------------------------------------
    # 2. Build transaction graph
    # ---------------------------------------------------------------
    print("[L3A] Building transaction graph...")
    gb = GraphBuilder()
    G = gb.build(df)
    print(f"[L3A] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---------------------------------------------------------------
    # 3. Graph features (L2A)
    # ---------------------------------------------------------------
    graph_features = build_graph_features(G, out_dir=out_dir / "artifacts_cache")

    # ---------------------------------------------------------------
    # 4. L2B analytics
    # ---------------------------------------------------------------
    print("[L3A] Running L2B analytics...")
    community_profiles, suspicious_paths = run_analytics(
        G, graph_features, out_dir=out_dir / "artifacts_cache"
    )

    # ---------------------------------------------------------------
    # 5. Account behavioral profiles (fast groupby)
    # ---------------------------------------------------------------
    print("[L3A] Building behavioral profiles...")
    behavioral_profiles = _build_account_profiles(df)
    behavioral_profiles.to_parquet(out_dir / "behavioral_profiles.parquet", index=False)
    print(f"[L3A] wrote behavioral_profiles.parquet ({len(behavioral_profiles)} accounts)")

    # ---------------------------------------------------------------
    # 6. Build 52-dim feature matrix (sample first to avoid 550K Python loop)
    # ---------------------------------------------------------------
    print(f"[L3A] Sampling {_MAX_FEATURE_ROWS:,} normal transactions for feature extraction...")
    normal_df = df[df["is_fraud"] == False].reset_index(drop=True)
    n_sample = min(_MAX_FEATURE_ROWS, len(normal_df))
    sample_idx = rng.choice(len(normal_df), size=n_sample, replace=False)
    sample_df = normal_df.iloc[sample_idx].sort_values("timestamp").reset_index(drop=True)
    fs = FeatureStore()

    X, feature_names = _extract_52_features(
        sample_df, graph_features, fs, len(sample_df), rng
    )
    print(f"[L3A] Feature matrix: {X.shape}")

    # Remove NaN / Inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    print(f"[L3A] After NaN/Inf filter: {X.shape}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    artifact_store.save("scaler_behavioral", scaler)
    artifact_store.save("feature_columns", feature_names)

    # ---------------------------------------------------------------
    # 7. Train models
    # ---------------------------------------------------------------
    # Sample to _MAX_TRAIN_ROWS for sklearn models (they don't scale as well)
    if len(X_scaled) > _MAX_TRAIN_ROWS:
        idx = rng.choice(len(X_scaled), size=_MAX_TRAIN_ROWS, replace=False)
        X_train = X_scaled[idx]
    else:
        X_train = X_scaled

    print(f"[L3A] Training IsolationForest on {len(X_train):,} rows...")
    t0 = time.time()
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_train)
    artifact_store.save("isolation_forest", iso)
    print(f"[L3A] IsolationForest done in {time.time()-t0:.1f}s")

    print(f"[L3A] Training LocalOutlierFactor on {len(X_train):,} rows...")
    t0 = time.time()
    lof = LocalOutlierFactor(
        n_neighbors=30,
        novelty=True,
        contamination=0.01,
        n_jobs=-1,
    )
    lof.fit(X_train)
    artifact_store.save("lof_model", lof)
    print(f"[L3A] LOF done in {time.time()-t0:.1f}s")

    print(f"[L3A] Training Autoencoder ({INPUT_DIM}->32->16->32->{INPUT_DIM}) "
          f"for 50 epochs on {len(X_train):,} rows...")
    t0 = time.time()
    autoencoder = _train_autoencoder(X_train, epochs=50, lr=1e-3, batch_size=256)
    artifact_store.save_torch("autoencoder", autoencoder.state_dict())
    print(f"[L3A] Autoencoder done in {time.time()-t0:.1f}s")

    # ---------------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------------
    # Reconstruction error threshold (95th percentile on training set)
    autoencoder.eval()
    with torch.no_grad():
        t = torch.tensor(X_train, dtype=torch.float32)
        errors = autoencoder.reconstruction_error(t).numpy()
    threshold_95 = float(np.percentile(errors, 95))
    artifact_store.save("autoencoder_threshold", threshold_95)
    print(f"[L3A] Autoencoder recon error 95th pct (normal) = {threshold_95:.6f}")

    summary = {
        "n_transactions": len(df),
        "n_accounts": df["sender_account"].nunique(),
        "n_communities": len(community_profiles),
        "n_suspicious_paths": len(suspicious_paths),
        "feature_dim": INPUT_DIM,
        "train_rows": len(X_train),
        "autoencoder_threshold_95": threshold_95,
    }
    print("[L3A] Training complete:", summary)
    return summary


# ---------------------------------------------------------------------------
# train_models_only — used by top-level run_offline_pipeline.py
# when S2/L2A/L2B have already been run externally.
# ---------------------------------------------------------------------------

def train_models_only(
    sample_df: pd.DataFrame,
    graph_features: pd.DataFrame,
    artifact_store: ArtifactStore,
    seed: int = 42,
    epochs: int = 50,
    max_train_rows: int = _MAX_TRAIN_ROWS,
) -> dict:
    """
    Train IsoForest + LOF + AutoEncoder from pre-built data.

    Parameters
    ----------
    sample_df       : sampled NORMAL transactions (pre-filtered, sorted by timestamp)
    graph_features  : DataFrame from L2A (per-account graph features, indexed by account_id)
    artifact_store  : where to persist models
    seed            : RNG seed

    Saves: scaler_behavioral, feature_columns, isolation_forest,
           lof_model, autoencoder, autoencoder_threshold
    """
    rng = np.random.default_rng(seed)
    fs  = FeatureStore()

    print(f"[L3A.models_only] Extracting features from {len(sample_df):,} normal transactions...")
    X, feature_names = _extract_52_features(
        sample_df, graph_features, fs, len(sample_df), rng
    )
    print(f"[L3A.models_only] Feature matrix: {X.shape}")

    # Remove NaN / Inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    print(f"[L3A.models_only] After NaN/Inf filter: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    artifact_store.save("scaler_behavioral", scaler)
    artifact_store.save("feature_columns", feature_names)

    if len(X_scaled) > max_train_rows:
        idx = rng.choice(len(X_scaled), size=max_train_rows, replace=False)
        X_train = X_scaled[idx]
    else:
        X_train = X_scaled

    print(f"[L3A.models_only] Training IsolationForest on {len(X_train):,} rows...")
    t0 = time.time()
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=seed, n_jobs=-1)
    iso.fit(X_train)
    artifact_store.save("isolation_forest", iso)
    print(f"[L3A.models_only] IsoForest done in {time.time()-t0:.1f}s")

    print(f"[L3A.models_only] Training LocalOutlierFactor on {len(X_train):,} rows...")
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=0.01, n_jobs=-1)
    lof.fit(X_train)
    artifact_store.save("lof_model", lof)
    print(f"[L3A.models_only] LOF done in {time.time()-t0:.1f}s")

    print(f"[L3A.models_only] Training Autoencoder for {epochs} epochs...")
    t0 = time.time()
    autoencoder = _train_autoencoder(X_train, epochs=epochs, lr=1e-3, batch_size=256)
    artifact_store.save_torch("autoencoder", autoencoder.state_dict())
    print(f"[L3A.models_only] Autoencoder done in {time.time()-t0:.1f}s")

    autoencoder.eval()
    with torch.no_grad():
        t = torch.tensor(X_train, dtype=torch.float32)
        errors = autoencoder.reconstruction_error(t).numpy()
    threshold_95 = float(np.percentile(errors, 95))
    artifact_store.save("autoencoder_threshold", threshold_95)
    print(f"[L3A.models_only] AE threshold (95th pct normal) = {threshold_95:.6f}")

    return {
        "feature_dim":             INPUT_DIM,
        "train_rows":              len(X_train),
        "autoencoder_threshold_95": threshold_95,
    }
