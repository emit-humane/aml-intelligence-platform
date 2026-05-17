"""
Temporal Graph Network (TGN) core modules.

Architecture per spec:
  memory_dim      = 64   (per-node memory vector)
  time_enc_dim    = 16   (time encoder output)
  edge_feat_dim   = 3    (log_amount_norm, is_international, delta_t_norm)
  msg_input_dim   = 64+64+16+3 = 147
  msg_output_dim  = 64

Components:
  TimeEncoder      — Linear(1 → time_enc_dim) with sin activation
  MessageFunction  — MLP(msg_input_dim → 128 → msg_output_dim, ReLU)
  MemoryUpdater    — GRU(msg_output_dim → memory_dim)
  TGNCore          — orchestrates memory dict + the three modules above

Training contract:
  Memory is stored as a plain Python dict {node_int → Tensor(memory_dim)}.
  During each training batch, memory tensors are DETACHED before use
  (standard TGN training approximation — avoids unbounded BPTT).
  Gradients flow through TimeEncoder, MessageFunction, and MemoryUpdater
  parameters only.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

MEMORY_DIM    = 64
TIME_ENC_DIM  = 16
EDGE_FEAT_DIM = 3       # [log_amount_norm, is_international, delta_t_norm]
MSG_INPUT_DIM = MEMORY_DIM + MEMORY_DIM + TIME_ENC_DIM + EDGE_FEAT_DIM  # 147
MSG_OUTPUT_DIM = 64


# ---------------------------------------------------------------------------
# TimeEncoder
# ---------------------------------------------------------------------------

class TimeEncoder(nn.Module):
    """
    Time2Vec-style encoder: projects scalar Δt → TIME_ENC_DIM floats.
    Uses learnable linear projection followed by sin non-linearity (d-1 dims)
    plus a linear component (1 dim) for monotonic trend.
    """

    def __init__(self, out_dim: int = TIME_ENC_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(1, out_dim)
        self._out_dim = out_dim

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        dt : (batch,) or (batch, 1)  — normalised Δt in [0, 1]

        Returns
        -------
        (batch, out_dim)
        """
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        x = self.linear(dt)               # (B, out_dim)
        # First component: linear; rest: sin
        out = torch.cat([x[:, :1], torch.sin(x[:, 1:])], dim=-1)
        return out


# ---------------------------------------------------------------------------
# MessageFunction (MLP)
# ---------------------------------------------------------------------------

class MessageFunction(nn.Module):
    """MLP: msg_input_dim → 128 → msg_output_dim with ReLU."""

    def __init__(
        self,
        in_dim: int = MSG_INPUT_DIM,
        hidden: int = 128,
        out_dim: int = MSG_OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# TGNCore
# ---------------------------------------------------------------------------

class TGNCore(nn.Module):
    """
    Orchestrates TGN memory + message + update loop.

    Memory is stored as a Python dict (node_int → Tensor) so it can be
    serialised to tgn_memory_state.pkl independently of model weights.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int = MEMORY_DIM,
        time_enc_dim: int = TIME_ENC_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
    ) -> None:
        super().__init__()
        self.num_nodes   = num_nodes
        self.memory_dim  = memory_dim
        msg_in = memory_dim + memory_dim + time_enc_dim + edge_feat_dim

        self.time_encoder  = TimeEncoder(time_enc_dim)
        self.message_fn    = MessageFunction(msg_in, 128, memory_dim)
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)

        # Memory lives outside the autograd graph (plain dict of detached tensors)
        self._memory: dict[int, torch.Tensor] = {}
        self._device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Memory accessors
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        self._memory.clear()

    def get_memory(self, node_ids: Sequence[int]) -> torch.Tensor:
        """Return (len(node_ids), memory_dim) tensor. Zeros for unseen nodes."""
        rows = []
        for nid in node_ids:
            if nid in self._memory:
                rows.append(self._memory[nid])
            else:
                rows.append(torch.zeros(self.memory_dim, device=self._device))
        return torch.stack(rows)                    # (N, memory_dim)

    def _set_memory(
        self,
        node_ids: Sequence[int],
        new_mem: torch.Tensor,          # (N, memory_dim), detached
    ) -> None:
        for i, nid in enumerate(node_ids):
            # .clone() breaks the view: each stored tensor owns its storage
            # so pickle doesn't serialise the full batch tensor repeatedly
            self._memory[nid] = new_mem[i].clone()

    def memory_state_dict(self) -> dict[int, torch.Tensor]:
        return {k: v.cpu() for k, v in self._memory.items()}

    def load_memory_state(self, state: dict[int, torch.Tensor]) -> None:
        self._memory = {k: v.to(self._device) for k, v in state.items()}

    # ------------------------------------------------------------------
    # Forward: compute messages and update memory
    # ------------------------------------------------------------------

    def forward(
        self,
        src: Sequence[int],
        dst: Sequence[int],
        dt_norm: torch.Tensor,          # (B,) normalised Δt
        edge_feat: torch.Tensor,        # (B, edge_feat_dim)
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Compute TGN message embeddings for a batch of edges.

        Parameters
        ----------
        src, dst      : node integer indices (length B)
        dt_norm       : normalised time delta per edge (B,) in [0,1]
        edge_feat     : (B, EDGE_FEAT_DIM) edge feature tensor
        update_memory : if True, update memory for src nodes after forward

        Returns
        -------
        messages : (B, memory_dim) — updated src node embeddings
        """
        src_mem = self.get_memory(src).detach()     # (B, memory_dim) — detached!
        dst_mem = self.get_memory(dst).detach()

        time_enc = self.time_encoder(dt_norm)        # (B, time_enc_dim)
        msg_input = torch.cat([src_mem, dst_mem, time_enc, edge_feat], dim=-1)
        messages = self.message_fn(msg_input)        # (B, memory_dim)

        # GRU update: new_src_mem = GRU(message, old_src_mem)
        new_src_mem = self.memory_updater(messages, src_mem)  # (B, memory_dim)

        if update_memory:
            self._set_memory(src, new_src_mem.detach())

        return new_src_mem                           # (B, memory_dim), grad flows


def build_tgn(num_nodes: int) -> TGNCore:
    return TGNCore(num_nodes=num_nodes)
