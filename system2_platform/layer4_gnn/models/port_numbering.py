"""
Port Numbering for Temporal Graph Network.

For each node, outgoing edges are sorted by timestamp and assigned a
port_number 0, 1, ..., out_degree-1 reflecting the order of activity.
This creates a positional encoding for the edge stream.

ego_id = stable_hash(account_id) % EGO_DIM
Used as a deterministic node identity feature that survives across
training and inference.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

EGO_DIM = 16   # dimensionality of ego embedding lookup table


# ---------------------------------------------------------------------------
# Port assignment
# ---------------------------------------------------------------------------

def assign_edge_ports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'port_number' column to a transaction DataFrame.

    For each sender_account, edges are sorted by timestamp (ascending) and
    numbered 0, 1, 2, ... in that order.  Ties broken by transaction_id.

    Parameters
    ----------
    df : DataFrame with at least ['sender_account', 'timestamp', 'transaction_id']

    Returns
    -------
    DataFrame with additional 'port_number' (int32) column.
    """
    df = df.copy()
    df["_ts_num"] = pd.to_datetime(df["timestamp"], utc=True).astype(np.int64)
    df = df.sort_values(["sender_account", "_ts_num", "transaction_id"])
    df["port_number"] = df.groupby("sender_account").cumcount().astype(np.int32)
    df.drop(columns=["_ts_num"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Ego ID
# ---------------------------------------------------------------------------

def compute_ego_id(account_id: str, ego_dim: int = EGO_DIM) -> int:
    """
    Deterministic ego integer from account_id string.
    Uses SHA-256 for stability across Python interpreter restarts.
    """
    h = int(hashlib.sha256(account_id.encode()).hexdigest(), 16)
    return h % ego_dim


def compute_ego_ids(
    account_ids: list[str],
    ego_dim: int = EGO_DIM,
) -> dict[str, int]:
    """Return {account_id: ego_int} for every account in the list."""
    return {acc: compute_ego_id(acc, ego_dim) for acc in account_ids}


def ego_embedding_matrix(ego_dim: int = EGO_DIM) -> np.ndarray:
    """
    Return a learnable-style lookup table (ego_dim, ego_dim) initialised with
    a deterministic identity-like matrix (each row i is a unit vector for class i).
    Used to convert ego_id integer → ego_dim-dim float vector.
    """
    return np.eye(ego_dim, dtype=np.float32)


def get_ego_vectors(
    account_ids: list[str],
    ego_dim: int = EGO_DIM,
) -> np.ndarray:
    """
    Return (len(account_ids), ego_dim) float32 array of ego vectors.
    """
    lookup = ego_embedding_matrix(ego_dim)
    ids = [compute_ego_id(acc, ego_dim) for acc in account_ids]
    return lookup[ids]
