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
_MAX_FEATURE_ROWS = 20_000  # uniform-by-transaction sample (count is plenty;
                            # uniformity is what fixes the scaler, not size)


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


def _pick_sample_ids(
    normal_df: pd.DataFrame,
    feature_sample_n: int,
    max_per_account: int | None,
    rng: np.random.Generator,
) -> set[str]:
    """
    Choose which NORMAL transaction_ids to compute training features for.

    DEFAULT: uniform-by-transaction sample (max_per_account=None).

    This is critical for scaler correctness. The model scores EVERY live
    transaction, so the inference feature distribution is the natural
    per-transaction distribution — dominated by a few very busy accounts
    (high velocity / beneficiary counts). A per-account cap instead builds
    a sample of "a few rows from each of thousands of quiet accounts"
    (low velocity), so the StandardScaler learns a quiet-account mean/std
    and real busy traffic lands ~12σ out at inference (AE saturates at 100,
    IsoForest/LOF invert because low-velocity fraud then looks "normal").
    See scripts/probe_l3_features.py for the empirical breakdown.

    A per-account cap is therefore only offered as an explicit opt-in
    (max_per_account=int); it should normally be left None.
    """
    if max_per_account is None:
        ids = normal_df["transaction_id"].astype(str).to_numpy()
        if len(ids) > feature_sample_n:
            ids = rng.choice(ids, size=feature_sample_n, replace=False)
        return set(ids.tolist())

    parts = []
    for _, grp in normal_df.groupby("sender_account"):
        k = min(len(grp), max_per_account)
        parts.append(grp.sample(k, random_state=int(rng.integers(0, 2**31 - 1))))
    capped = pd.concat(parts, ignore_index=True)
    if len(capped) > feature_sample_n:
        idx = rng.choice(len(capped), size=feature_sample_n, replace=False)
        capped = capped.iloc[idx]
    return set(capped["transaction_id"].astype(str))


def _extract_52_features(
    history_df: pd.DataFrame,
    sample_ids: set[str],
    graph_features_df: pd.DataFrame,
    feature_store: FeatureStore,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute 45 S1 features via FeatureStore, merge 7 graph features = 52 dims.

    CRITICAL FIX (training/inference parity):
    ``history_df`` is the FULL transaction stream (normal + fraud), sorted by
    timestamp.  Every row is ingested into the FeatureStore to build true
    per-account rolling-window history, but features are only computed/recorded
    for rows whose transaction_id is in ``sample_ids`` (a per-account-capped
    sample of NORMAL transactions).

    The previous implementation pre-capped/subsampled transactions *before*
    replaying them, so during training every account had near-empty rolling
    history (velocity≈0, beneficiary_count≈0).  The scaler + IsoForest/LOF/AE
    learned that degenerate "sparse history" distribution, so at inference with
    real history every transaction looked out-of-distribution and scores
    saturated/inverted (AUROC ≈ 0.1).  Replaying full history here makes the
    training feature distribution match what S5 produces at inference once the
    FeatureStore is seeded from historical data.

    IMPORTANT — raw features, not batch-standardised:
    We call _compute_raw() / _ingest_row() directly so X_45 holds raw
    (unscaled) values, matching S5 FeatureUpdater at inference time (which
    applies the saved scaler_behavioral afterward).
    """
    print(f"[L3A] Replaying {len(history_df):,} txns for full rolling history; "
          f"computing features for {len(sample_ids):,} sampled normals...")
    t0 = time.time()

    # Reset FeatureStore state so we replay history deterministically
    feature_store._states.clear()
    feature_store._device_to_accounts.clear()

    hist_sorted = history_df.sort_values("timestamp")
    raw_matrix: list[list[float]] = []
    sender_col: list[str] = []
    n_seen = 0
    for row in hist_sorted.itertuples(index=False):
        tid = str(getattr(row, "transaction_id", ""))
        if tid in sample_ids:
            raw = feature_store._compute_raw(row)    # pre-event raw features (TRUE history)
            raw_matrix.append(raw)
            sender_col.append(str(getattr(row, "sender_account", "")))
        feature_store._ingest_row(row)               # always ingest -> builds history
        n_seen += 1
        if n_seen % 100_000 == 0:
            print(f"[L3A]   replayed {n_seen:,}/{len(hist_sorted):,} "
                  f"({len(raw_matrix):,} sampled)")

    print(f"[L3A] S1 raw features (full-history) done in {time.time()-t0:.1f}s")

    X_45 = np.array(raw_matrix, dtype=float)     # (n, 45) — raw, unscaled

    # Merge 7 graph features per sender account
    gf_indexed = graph_features_df.set_index("account_id")[_GRAPH_FEATURE_COLS]

    X_7 = np.zeros((len(sender_col), len(_GRAPH_FEATURE_COLS)), dtype=float)
    for i, acc in enumerate(sender_col):
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
    print(f"[L3A] Selecting feature sample (full-history replay)...")
    normal_df = df[df["is_fraud"] == False].reset_index(drop=True)
    sample_ids = _pick_sample_ids(normal_df, _MAX_FEATURE_ROWS, None, rng)
    fs = FeatureStore()

    # Pass the FULL df (normal + fraud) as history so rolling windows reflect
    # true per-account activity — matches S5 inference once it is seeded.
    X, feature_names = _extract_52_features(
        df, sample_ids, graph_features, fs, rng
    )
    print(f"[L3A] Feature matrix: {X.shape}")

    # Remove NaN / Inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    print(f"[L3A] After NaN/Inf filter: {X.shape}")

    # Scale ALL 52 dims (behavioral + graph) with a single StandardScaler.
    # This is essential because benford_chi2_community can reach 14 000+ and
    # causes AE gradient explosion when left unscaled.
    #
    # The 52-dim scaler is used consistently:
    #   • S5 FeatureUpdater._rescale() extracts mean_[:45] / scale_[:45] to
    #     scale the 45 behavioral dims → stored in fv.scaled_feature_vector
    #   • L3B BehavioralInferenceEngine._assemble_features() extracts mean_[45:]
    #     / scale_[45:] to scale the 7 graph dims at inference time
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)               # (n, 52) all scaled
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
    history_df: pd.DataFrame,
    graph_features: pd.DataFrame,
    artifact_store: ArtifactStore,
    seed: int = 42,
    epochs: int = 50,
    max_train_rows: int = _MAX_TRAIN_ROWS,
    feature_sample_n: int = _MAX_FEATURE_ROWS,
    max_per_account: int | None = None,
) -> dict:
    """
    Train IsoForest + LOF + AutoEncoder from the FULL transaction history.

    Parameters
    ----------
    history_df       : FULL transaction stream (normal + fraud). Replayed in
                       timestamp order to build true per-account rolling
                       history; features are fit only on a per-account-capped
                       sample of NORMAL transactions.
    graph_features   : DataFrame from L2A (per-account graph features)
    artifact_store   : where to persist models
    seed             : RNG seed
    feature_sample_n : max #normal transactions to compute fit features for
    max_per_account  : leave None (uniform-by-transaction). An int cap
                       reintroduces the scaler distribution-shift bug.

    Saves: scaler_behavioral, feature_columns, isolation_forest,
           lof_model, autoencoder, autoencoder_threshold
    """
    rng = np.random.default_rng(seed)
    fs  = FeatureStore()

    normal_df = history_df[history_df["is_fraud"] == False].reset_index(drop=True)
    sample_ids = _pick_sample_ids(normal_df, feature_sample_n, max_per_account, rng)
    print(f"[L3A.models_only] Full-history replay of {len(history_df):,} txns; "
          f"fitting on {len(sample_ids):,} sampled normals...")
    X, feature_names = _extract_52_features(
        history_df, sample_ids, graph_features, fs, rng
    )
    print(f"[L3A.models_only] Feature matrix: {X.shape}")

    # Remove NaN / Inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    print(f"[L3A.models_only] After NaN/Inf filter: {X.shape}")

    # Scale ALL 52 dims — see comment in train() above for rationale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)               # (n, 52) all scaled
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
