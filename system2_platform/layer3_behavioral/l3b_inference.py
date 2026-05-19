"""
L3B — Behavioral Anomaly Inference Engine.

Loads trained artifacts from L3A and scores individual transactions.

Input: (LiveFeatureVector, optional LiveGraphFeatureVector)
Output: BehavioralAnomalyOutput

Artifacts loaded from artifact_dir:
  scaler_behavioral.joblib
  isolation_forest.joblib
  lof_model.joblib
  autoencoder.pt
  feature_columns.joblib         — ordered list of 52 feature names
  autoencoder_threshold.joblib   — 95th-pct reconstruction error on normal data

Anomaly score mapping:
  IsolationForest: decision_function in [−0.5, 0.5] → inverted → [0, 100]
  LOF:             score_samples in [−inf, 0] → clipped to [−5, 0] → inverted → [0, 100]
  Autoencoder:     recon_error / threshold → clamped → scaled to [0, 100]
  Ensemble:        mean(iso_score, lof_score, ae_score)
"""

from __future__ import annotations

import numpy as np
import torch

from ..shared.s3_artifact_store import ArtifactStore
from ..contracts.live_feature_vector import LiveFeatureVector
from ..contracts.live_graph_feature_vector import LiveGraphFeatureVector
from ..contracts.behavioral_anomaly_output import BehavioralAnomalyOutput
from .models.autoencoder import BehavioralAutoencoder, INPUT_DIM

# 7 graph features — same order as l3a_training._GRAPH_FEATURE_COLS
_GRAPH_FEATURE_COLS = [
    "pagerank", "betweenness", "fan_in_score", "fan_out_score",
    "community_density", "benford_chi2_community", "volume_asymmetry",
]

# All 45 behavioral feature names (same order as FeatureStore._compute_raw())
# [0-10] Temporal, [11-19] Behavioral, [20-24] Geographic, [25-28] Device, [29-44] Derived
_BEHAVIORAL_COLS = [
    # Temporal (0-10)
    "tx_velocity_1h", "tx_velocity_6h", "tx_velocity_24h", "tx_velocity_7d",
    "avg_amount_7d", "std_amount_7d", "tx_gap_seconds", "night_tx_ratio",
    "weekend_tx_ratio", "hour_of_day", "day_of_week",
    # Behavioral (11-19)
    "beneficiary_count_7d", "beneficiary_count_30d", "receiver_entropy",
    "amount_zscore", "avg_daily_volume_30d", "round_amount_flag",
    "sub_threshold_flag", "amount_leading_digit", "benford_chi2_score",
    # Geographic (20-24)
    "geo_distance_km", "country_switch_count_7d", "impossible_travel_flag",
    "high_risk_country_flag", "cross_border_ratio_30d",
    # Device (25-28)
    "device_change_count_7d", "new_device_flag", "shared_device_count",
    "ip_change_count_24h",
    # Derived (29-44)
    "kyc_level", "amount_ratio_7d", "balance_utilization", "same_bank_flag",
    "is_international", "tx_type_encoded", "channel_encoded",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "day_of_month", "success_rate_7d", "max_amount_7d", "min_amount_7d",
    "amount_to_max_ratio",
]


# Behavioral ensemble weights — set from per-detector validation AUROC.
# These reflect the OLD (pre-fix) measurement: IsoForest 0.996, LOF 0.11
# (inverted), AE 0.50 (saturated). The LOF n_neighbors bump + held-out AE
# calibration (l3a_training._fit_detectors + the graded mappings in score())
# are expected to materially improve LOF/AE. They are kept iso-dominant here
# so a stale-artifact server stays safe; do NOT guess new weights.
#
# AFTER retraining: run
#   python -m scripts.diagnose_l3_discrimination --normal-sample 3000 --skip-noseed
# read the new per-detector AUROC, then set the weights proportional to
# max(0, AUROC-0.5) (the project's standard methodology) — e.g. if it comes
# back iso 0.99 / ae 0.85 / lof 0.70, use roughly _W_ISO .55 _W_AE .28
# _W_LOF .17. Leave a detector at 0 only if it stays <=0.5 / inverted.
_W_ISO = 0.90
_W_LOF = 0.00
_W_AE  = 0.10


class BehavioralInferenceEngine:
    """
    Stateless behavioral anomaly scorer.

    Usage
    -----
    engine = BehavioralInferenceEngine.load(artifact_dir)
    out = engine.score(fv, gfv)
    """

    def __init__(
        self,
        scaler,
        iso_forest,
        lof,
        autoencoder: BehavioralAutoencoder,
        ae_threshold: float,
        feature_columns: list[str],
        ae_stats: dict | None = None,
        lof_stats: dict | None = None,
    ) -> None:
        self._scaler        = scaler
        self._iso           = iso_forest
        self._lof           = lof
        self._ae            = autoencoder
        self._ae_thresh     = ae_threshold
        self._feat_cols     = feature_columns
        # Held-out-normal calibration stats (None -> legacy fallback mapping).
        # ae_stats:  {"p50","p95",...} recon error on held-out normal
        # lof_stats: {"p50","p05"} score_samples on held-out normal
        self._ae_stats      = ae_stats
        self._lof_stats     = lof_stats

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, artifact_dir) -> "BehavioralInferenceEngine":
        from pathlib import Path
        artifact_dir = Path(artifact_dir)
        store = ArtifactStore(artifact_dir)

        scaler    = store.load("scaler_behavioral")
        iso       = store.load("isolation_forest")
        lof       = store.load("lof_model")
        feat_cols = store.load("feature_columns")
        ae_thresh = float(store.load("autoencoder_threshold"))

        # New held-out-normal calibration stats (absent on pre-fix artifacts
        # -> fall back to the legacy mappings so the server still runs).
        try:
            ae_stats = dict(store.load("autoencoder_norm_stats"))
        except Exception:
            ae_stats = None
        try:
            lof_stats = dict(store.load("lof_norm_stats"))
        except Exception:
            lof_stats = None

        ae = BehavioralAutoencoder(input_dim=INPUT_DIM)
        ckpt = store.load_torch("autoencoder", map_location="cpu")
        ae.load_state_dict(ckpt)
        ae.eval()

        mode = "calibrated(held-out)" if (ae_stats and lof_stats) else "legacy-fallback"
        print(
            f"[L3B] Loaded behavioral engine: {len(feat_cols)} features, "
            f"AE thr={ae_thresh:.4f}, LOF/AE mapping={mode}"
        )
        return cls(scaler, iso, lof, ae, ae_thresh, feat_cols,
                   ae_stats=ae_stats, lof_stats=lof_stats)

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _assemble_features(
        self,
        fv: LiveFeatureVector,
        gfv: LiveGraphFeatureVector | None,
    ) -> np.ndarray:
        """
        Assemble a (1, 52) numpy feature vector matching the training schema.

        fv.scaled_feature_vector (45 dims) — already scaled by S5 using the
        first 45 components of the 52-dim scaler_behavioral.

        The 7 graph features are raw at this point; we apply the tail
        (mean_[45:] / scale_[45:]) of the same 52-dim scaler here so that
        ALL 52 dims are on a consistent z-score scale before reaching the
        IsoForest / LOF / Autoencoder models.
        """
        behav = np.array(fv.scaled_feature_vector, dtype=np.float32)
        # Pad / truncate to 45 dims if necessary
        behav = behav[:45] if len(behav) >= 45 else np.pad(behav, (0, 45 - len(behav)))

        if gfv is not None:
            graph_feat_raw = np.array([
                gfv.sender_pagerank,
                gfv.sender_betweenness,
                gfv.sender_fan_in_score,
                gfv.sender_fan_out_score,
                gfv.sender_community_density,
                gfv.sender_benford_chi2_community,
                # volume_asymmetry: live graph doesn't compute it; use 0
                0.0,
            ], dtype=np.float32)
        else:
            graph_feat_raw = np.zeros(7, dtype=np.float32)

        # Scale the 7 graph dims using the tail of the 52-dim scaler
        graph_feat = self._scale_graph(graph_feat_raw)

        x = np.concatenate([behav, graph_feat]).reshape(1, -1)   # (1, 52)
        return x

    def _scale_graph(self, graph_feat_raw: np.ndarray) -> np.ndarray:
        """Apply the graph-feature portion of the 52-dim scaler."""
        if self._scaler is None:
            return graph_feat_raw
        try:
            n_feat = getattr(self._scaler, "n_features_in_", 0)
            if n_feat >= 52:
                mean_7  = np.asarray(self._scaler.mean_[45:52],  dtype=np.float32)
                scale_7 = np.asarray(self._scaler.scale_[45:52], dtype=np.float32)
                return (graph_feat_raw - mean_7) / np.where(scale_7 > 0, scale_7, 1.0)
            # Old 45-dim scaler — no graph scaling available, return raw
            return graph_feat_raw
        except Exception:
            return graph_feat_raw

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        fv: LiveFeatureVector,
        gfv: LiveGraphFeatureVector | None = None,
    ) -> BehavioralAnomalyOutput:
        x = self._assemble_features(fv, gfv)        # (1, 52) already scaled

        # --- IsolationForest: decision_function ∈ [~-0.5, ~0.5] ---
        iso_raw = float(self._iso.decision_function(x)[0])
        # More negative → more anomalous; map to [0, 100]
        iso_score = float(np.clip((0.5 - iso_raw) / 1.0 * 100.0, 0.0, 100.0))

        # --- LOF: score_samples ∈ (-inf, 0] (more negative → more anomalous) ---
        lof_raw = float(self._lof.score_samples(x)[0])
        if self._lof_stats:
            # Graded against held-out-normal LOF distribution:
            # normal median (p50) → 0, normal 5th-pct tail (p05) → 100,
            # anything beyond → clipped. If LOF is non-discriminative on this
            # data (fraud less anomalous than normal) this correctly yields a
            # low, quiet score instead of a loud inverted one.
            p50, p05 = self._lof_stats["p50"], self._lof_stats["p05"]
            denom = (p50 - p05) or 1e-9
            lof_score = float(np.clip((p50 - lof_raw) / denom * 100.0, 0.0, 100.0))
        else:
            lof_clipped = float(np.clip(lof_raw, -5.0, 0.0))
            lof_score = float((0.0 - lof_clipped) / 5.0 * 100.0)

        # --- Autoencoder reconstruction error ---
        x_tensor = torch.from_numpy(x.astype(np.float32))
        ae_recon = float(self._ae.reconstruction_error(x_tensor)[0].item())
        if self._ae_stats:
            # Graded against held-out-normal recon distribution:
            # normal median recon (p50) → 0, normal p95 → 100, fraud (higher
            # recon) → high. This is the fix for the saturation: the old
            # threshold was the 95th pct of *training* recon (memorised data),
            # so it was tiny and every inference point pegged at 100.
            p50, p95 = self._ae_stats["p50"], self._ae_stats["p95"]
            denom = (p95 - p50) or 1e-9
            ae_score = float(np.clip((ae_recon - p50) / denom * 100.0, 0.0, 100.0))
        else:
            ae_ratio = ae_recon / (self._ae_thresh + 1e-9)
            ae_score = float(np.clip(ae_ratio * 50.0, 0.0, 100.0))  # legacy

        # --- Ensemble (AUROC-weighted, not a blind mean) ---
        # Weighted by demonstrated per-detector discriminative power (same
        # methodology as the P1 fusion weights). The OLD measurement was
        # IsoForest 0.996 / LOF 0.107 (inverted) / AE 0.500 (saturated), hence
        # the current iso-dominant _W_ISO/_W_LOF/_W_AE. The LOF/AE training
        # + mapping fixes are expected to change this — re-measure with
        # scripts/diagnose_l3_discrimination.py after retraining and reset the
        # weights (see the _W_* block above). All three sub-scores are always
        # returned so the diagnostic can re-measure them.
        ensemble = float(
            _W_ISO * iso_score + _W_LOF * lof_score + _W_AE * ae_score
        )

        # --- Anomaly drivers: top-3 raw feature dims with highest absolute value ---
        drivers = self._top_drivers(x[0])

        return BehavioralAnomalyOutput(
            transaction_id=fv.transaction_id,
            iso_forest_score=iso_score,
            lof_score=lof_score,
            autoencoder_recon_error=ae_recon,
            autoencoder_score=ae_score,
            ensemble_anomaly_score=ensemble,
            anomaly_drivers=drivers,
        )

    def _top_drivers(self, x_row: np.ndarray, top_k: int = 3) -> list[str]:
        """Return top-k feature names by absolute scaled value."""
        abs_vals = np.abs(x_row)
        top_idx  = np.argsort(abs_vals)[::-1][:top_k]
        all_cols = _BEHAVIORAL_COLS + _GRAPH_FEATURE_COLS   # 45 + 7 = 52
        names: list[str] = []
        for i in top_idx:
            name = all_cols[i] if i < len(all_cols) else f"feature_{i}"
            names.append(name)
        return names
