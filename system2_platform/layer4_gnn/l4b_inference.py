"""
L4B — GNN Inference Engine.

Loads trained GNN artifacts and computes per-transaction GNN scores at
stream inference time.

GNNInferenceOutput contract (identical regardless of TGN or GraphSAGE training):
  embedding_src   : np.ndarray (EMBEDDING_DIM,) — source node embedding
  embedding_dst   : np.ndarray (EMBEDDING_DIM,) — dest node embedding
  link_score      : float in [0, 1]  — sigmoid(dot(emb_src, emb_dst))
  gnn_anomaly_score: float in [0, 100] — scaled link score (inverted: low link = high anomaly)
  src_known       : bool — whether sender was in training graph
  dst_known       : bool — whether receiver was in training graph

Artifact files expected in artifact_dir:
  node_embeddings.npy        (N, 64) float32
  node_embedding_index.json  {account_id: row_index}
  tgn_model.pt               (may contain {"graphsage": True} if fallback was used)
  tgn_memory_state.pkl       {node_int: Tensor(64)}
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .models.mega_gnn import EMBEDDING_DIM

ZERO_EMB = np.zeros(EMBEDDING_DIM, dtype=np.float32)


@dataclass
class GNNInferenceOutput:
    embedding_src: np.ndarray        # (64,)
    embedding_dst: np.ndarray        # (64,)
    link_score: float                # sigmoid(dot(emb_src, emb_dst)) in [0,1]
    gnn_anomaly_score: float         # [0, 100] — high = unusual pair
    src_known: bool
    dst_known: bool


class GNNInferenceEngine:
    """
    Stateless (after loading) inference engine.

    Loads pre-trained node embeddings and the optional TGN memory
    for streaming updates.

    Usage
    -----
    engine = GNNInferenceEngine.load(artifact_dir)
    output = engine.score(sender_account_id, receiver_account_id)
    """

    def __init__(
        self,
        embeddings: np.ndarray,          # (N, 64)
        index: dict[str, int],           # account_id → row_index
        use_tgn: bool,
    ) -> None:
        self._emb   = embeddings
        self._index = index
        self._use_tgn = use_tgn
        self._N = len(embeddings)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, artifact_dir: Path | str) -> "GNNInferenceEngine":
        """Load all artifacts from artifact_dir."""
        artifact_dir = Path(artifact_dir)

        emb_path   = artifact_dir / "node_embeddings.npy"
        index_path = artifact_dir / "node_embedding_index.json"
        model_path = artifact_dir / "tgn_model.pt"

        if not emb_path.exists():
            raise FileNotFoundError(f"node_embeddings.npy not found in {artifact_dir}")
        if not index_path.exists():
            raise FileNotFoundError(f"node_embedding_index.json not found in {artifact_dir}")

        embeddings = np.load(str(emb_path)).astype(np.float32)  # (N, 64)
        with open(index_path) as fh:
            index: dict[str, int] = json.load(fh)

        use_tgn = True
        if model_path.exists():
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
            use_tgn = not ckpt.get("graphsage", False)

        print(
            f"[L4B] Loaded GNN engine: {len(index)} nodes, "
            f"emb shape={embeddings.shape}, backend={'TGN+MegaGNN' if use_tgn else 'GraphSAGE'}"
        )
        return cls(embeddings, index, use_tgn)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _get_embedding(self, account_id: str) -> tuple[np.ndarray, bool]:
        """
        Return (embedding, is_known).
        For unknown accounts returns a zero vector (cold-start fallback).
        """
        idx = self._index.get(account_id)
        if idx is None:
            return ZERO_EMB.copy(), False
        return self._emb[idx], True

    def score(self, sender_id: str, receiver_id: str) -> GNNInferenceOutput:
        """
        Compute GNNInferenceOutput for a (sender, receiver) pair.

        link_score      = sigmoid(dot(emb_src, emb_dst))
        gnn_anomaly_score = (1 - link_score) * 100

        Rationale: nodes that frequently transact together have high dot product
        (learned in self-supervised link prediction). Unusual pairs → low link score
        → high anomaly score.
        """
        emb_src, src_known = self._get_embedding(sender_id)
        emb_dst, dst_known = self._get_embedding(receiver_id)

        dot = float(np.dot(emb_src, emb_dst))
        link_score = float(1.0 / (1.0 + np.exp(-dot)))          # sigmoid
        gnn_anomaly_score = float((1.0 - link_score) * 100.0)

        return GNNInferenceOutput(
            embedding_src=emb_src,
            embedding_dst=emb_dst,
            link_score=link_score,
            gnn_anomaly_score=gnn_anomaly_score,
            src_known=src_known,
            dst_known=dst_known,
        )

    def batch_score(
        self,
        sender_ids: list[str],
        receiver_ids: list[str],
    ) -> list[GNNInferenceOutput]:
        """Vectorised batch scoring."""
        results = []
        src_embs = []
        dst_embs = []
        src_knowns = []
        dst_knowns = []

        for s, d in zip(sender_ids, receiver_ids):
            es, sk = self._get_embedding(s)
            ed, dk = self._get_embedding(d)
            src_embs.append(es)
            dst_embs.append(ed)
            src_knowns.append(sk)
            dst_knowns.append(dk)

        src_mat = np.array(src_embs, dtype=np.float32)  # (B, 64)
        dst_mat = np.array(dst_embs, dtype=np.float32)  # (B, 64)
        dots = (src_mat * dst_mat).sum(axis=1)           # (B,)
        link_scores = 1.0 / (1.0 + np.exp(-dots))       # (B,) sigmoid
        anomaly_scores = (1.0 - link_scores) * 100.0    # (B,)

        for i in range(len(sender_ids)):
            results.append(GNNInferenceOutput(
                embedding_src=src_embs[i],
                embedding_dst=dst_embs[i],
                link_score=float(link_scores[i]),
                gnn_anomaly_score=float(anomaly_scores[i]),
                src_known=src_knowns[i],
                dst_known=dst_knowns[i],
            ))
        return results

    # ------------------------------------------------------------------
    # Convenience: embedding lookup
    # ------------------------------------------------------------------

    def get_embedding(self, account_id: str) -> Optional[np.ndarray]:
        """Return (64,) embedding or None if account not in training graph."""
        idx = self._index.get(account_id)
        if idx is None:
            return None
        return self._emb[idx].copy()

    @property
    def num_known_nodes(self) -> int:
        return len(self._index)
