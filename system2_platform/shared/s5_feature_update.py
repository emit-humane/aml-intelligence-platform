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
        """Apply L3A trained scaler to the 45-dim behavioral vector."""
        import numpy as np
        x = np.array(feature_vector[:45], dtype=np.float32).reshape(1, -1)
        try:
            scaled = self._scaler.transform(x)[0]
            return scaled.tolist()
        except Exception:
            return feature_vector
