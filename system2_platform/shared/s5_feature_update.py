"""
S5 — Live Feature Update.

Maintains a live FeatureStore and updates it with each incoming transaction.
Converts a TransactionEvent into a LiveFeatureVector using the shared
FeatureStore (S1), which maintains per-account rolling windows.

Usage
-----
updater = FeatureUpdater(artifact_dir)
fv = updater.update_and_extract(event)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..contracts.transaction_event import TransactionEvent
from ..contracts.live_feature_vector import LiveFeatureVector
from ..shared.s1_feature_engineering import FeatureStore, FEATURE_DIM
from ..shared.s3_artifact_store import ArtifactStore


class FeatureUpdater:
    """
    Wraps FeatureStore for streaming use.

    Loads the trained scaler from L3A artifacts (scaler_behavioral)
    but uses its own internal FeatureStore for rolling window state.
    The scaler is applied ONLY to the 45 behavioral dims — it was fit
    on those dims in L3A.
    """

    def __init__(self, artifact_dir: Path | str) -> None:
        artifact_dir = Path(artifact_dir)
        self._store = FeatureStore()
        store = ArtifactStore(artifact_dir)
        try:
            self._scaler = store.load("scaler_behavioral")
        except Exception:
            self._scaler = None
            print("[S5] Warning: scaler_behavioral not found; features will be unscaled")

    def update_and_extract(self, event: TransactionEvent) -> LiveFeatureVector:
        """
        Compute features for the event, then update the rolling state.
        FeatureStore.compute() → LiveFeatureVector (uses pre-event state)
        FeatureStore.update() → None              (adds event to state)
        """
        # compute() uses current state BEFORE the new event is ingested
        fv: LiveFeatureVector = self._store.compute(event)
        # update() ingests the event so future calls see it in windows
        self._store.update(event)

        # If we have a trained scaler, re-scale the 45 behavioral dims
        if self._scaler is not None:
            scaled = self._rescale(fv.scaled_feature_vector)
            fv = fv.model_copy(update={"scaled_feature_vector": scaled})

        return fv

    def _rescale(self, feature_vector: list[float]) -> list[float]:
        """
        Apply L3A trained scaler to the 45-dim behavioral vector.

        The saved scaler_behavioral is a 52-dim StandardScaler (all behavioral
        + graph features).  We extract only the first 45 components (mean_[:45]
        and scale_[:45]) to scale the behavioral dims — graph dims are handled
        separately by L3B at inference time.
        """
        import numpy as np
        x = np.array(feature_vector[:45], dtype=np.float32)
        try:
            n_feat = getattr(self._scaler, "n_features_in_", 0)
            if n_feat == 45:
                # Backward-compat: 45-dim scaler (old artifacts)
                scaled = self._scaler.transform(x.reshape(1, -1))[0]
            else:
                # 52-dim scaler: apply only the 45 behavioral components
                mean_45  = np.asarray(self._scaler.mean_[:45],  dtype=np.float32)
                scale_45 = np.asarray(self._scaler.scale_[:45], dtype=np.float32)
                scaled = (x - mean_45) / np.where(scale_45 > 0, scale_45, 1.0)
            return scaled.tolist()
        except Exception:
            return feature_vector
