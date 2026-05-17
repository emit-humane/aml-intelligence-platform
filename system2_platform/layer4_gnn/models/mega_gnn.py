"""
MegaGNN — bidirectional neighbourhood aggregation layer.

For each node u at inference time:
  forward_agg  = mean of all TGN message embeddings on outgoing edges u→*
  backward_agg = mean of all TGN message embeddings on incoming edges *→u
  ego_emb      = ego_embedding_matrix[ego_id[u]]   (EGO_DIM = 16 dims)
  node_emb[u]  = MLP(cat(forward_agg, backward_agg, ego_emb, memory[u])) → 64

MEGA_INPUT_DIM = 64 + 64 + 16 + 64 = 208
EMBEDDING_DIM  = 64

The MLP uses: Linear(208→128) → ReLU → Linear(128→64) → ReLU → Linear(64→64)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .port_numbering import EGO_DIM, get_ego_vectors
from .tgn import TGNCore, MEMORY_DIM

MEGA_INPUT_DIM = MEMORY_DIM + MEMORY_DIM + EGO_DIM + MEMORY_DIM   # 64+64+16+64=208
EMBEDDING_DIM  = 64


class MegaGNN(nn.Module):
    """
    Bidirectional neighbourhood aggregator → per-node 64-dim embedding.

    Operates in two modes:
      train=False (default): takes pre-computed agg tensors, applies MLP
      batch_embed():         full-graph pass given TGNCore and edge stream
    """

    def __init__(
        self,
        input_dim: int = MEGA_INPUT_DIM,
        hidden: int = 128,
        out_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, MEGA_INPUT_DIM) → (N, EMBEDDING_DIM)"""
        return self.mlp(x)

    @torch.no_grad()
    def batch_embed(
        self,
        tgn: TGNCore,
        node_ids: list[str],
        node_to_int: dict[str, int],
        src_ints: Sequence[int],
        dst_ints: Sequence[int],
        messages_np: np.ndarray,            # (num_edges, MEMORY_DIM) — pre-computed
    ) -> np.ndarray:
        """
        Compute final node embeddings for all nodes.

        Parameters
        ----------
        tgn           : trained TGNCore (for memory read)
        node_ids      : ordered list of account_id strings
        node_to_int   : account_id → int index map
        src_ints      : int array of edge sources (same order as messages_np)
        dst_ints      : int array of edge destinations
        messages_np   : TGN message embeddings for all edges in time order

        Returns
        -------
        embeddings : (N, EMBEDDING_DIM) float32 numpy array
        """
        self.eval()
        N = len(node_ids)
        memory_dim = tgn.memory_dim

        # Accumulate forward and backward aggregations
        fwd_sum = np.zeros((N, memory_dim), dtype=np.float32)
        fwd_cnt = np.zeros(N, dtype=np.int32)
        bwd_sum = np.zeros((N, memory_dim), dtype=np.float32)
        bwd_cnt = np.zeros(N, dtype=np.int32)

        for i, (s, d) in enumerate(zip(src_ints, dst_ints)):
            fwd_sum[s] += messages_np[i]
            fwd_cnt[s] += 1
            bwd_sum[d] += messages_np[i]
            bwd_cnt[d] += 1

        # Mean (zero if no edges)
        fwd_agg = np.divide(fwd_sum, np.maximum(fwd_cnt[:, None], 1))
        bwd_agg = np.divide(bwd_sum, np.maximum(bwd_cnt[:, None], 1))

        # Ego vectors: (N, EGO_DIM)
        ego_vecs = get_ego_vectors(node_ids, EGO_DIM)

        # Memory: (N, memory_dim)
        all_ints = [node_to_int[acc] for acc in node_ids]
        mem_tensor = tgn.get_memory(all_ints)       # (N, memory_dim)
        mem_np = mem_tensor.cpu().numpy()

        # Assemble input: (N, 208)
        X = np.concatenate([fwd_agg, bwd_agg, ego_vecs, mem_np], axis=-1).astype(np.float32)
        X_tensor = torch.from_numpy(X)

        # Forward through MLP in chunks
        chunk = 1024
        parts = []
        for start in range(0, N, chunk):
            parts.append(self.forward(X_tensor[start:start + chunk]).numpy())
        return np.concatenate(parts, axis=0)   # (N, 64)


def build_mega_gnn() -> MegaGNN:
    return MegaGNN()
