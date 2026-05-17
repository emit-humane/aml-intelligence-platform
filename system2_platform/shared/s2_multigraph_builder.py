"""
S2 — MultiDiGraph Builder.

Builds and maintains a networkx.MultiDiGraph where:
  - Nodes  = account IDs
  - Edges  = transactions (multi-edges allowed: same sender/receiver, different tx)

Node attributes (set on first appearance, updated on each tx):
  account_id, total_sent, total_received, tx_count_sent, tx_count_recv,
  last_seen_ts, country, bank, kyc_level, geo_latitude, geo_longitude

Edge attributes (one edge per transaction):
  transaction_id, timestamp (ISO str), amount, currency,
  transaction_type, payment_channel, is_fraud (if available), ring_id

Provides:
  - GraphBuilder.build(df)          — build from a DataFrame
  - GraphBuilder.add_transaction(tx) — incremental update (live mode)
  - GraphBuilder.get_graph()        — return the nx.MultiDiGraph
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import networkx as nx
import pandas as pd


class GraphBuilder:
    """Builds and incrementally updates a transaction MultiDiGraph."""

    def __init__(self) -> None:
        self._G: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> nx.MultiDiGraph:
        """Build graph from a full transaction DataFrame (sorted or unsorted)."""
        self._G = nx.MultiDiGraph()
        df_sorted = df.sort_values("timestamp") if "timestamp" in df.columns else df
        for row in df_sorted.itertuples(index=False):
            self._add_row(row)
        return self._G

    def add_transaction(self, tx: dict | Any) -> None:
        """Add a single transaction dict to the live graph."""
        if isinstance(tx, dict):
            self._add_dict(tx)
        else:
            self._add_row(tx)

    def get_graph(self) -> nx.MultiDiGraph:
        return self._G

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_row(self, row: Any) -> None:
        sender = str(getattr(row, "sender_account", ""))
        receiver = str(getattr(row, "receiver_account", ""))
        ts = getattr(row, "timestamp", None)
        ts_str = _ts_iso(ts)
        amount = float(getattr(row, "amount", 0.0))

        self._ensure_node(
            node_id=sender,
            country=str(getattr(row, "sender_country", "")),
            bank=str(getattr(row, "sender_bank", "")),
            kyc_level=int(getattr(row, "kyc_level", 0)),
            lat=float(getattr(row, "geo_latitude", 0.0)),
            lon=float(getattr(row, "geo_longitude", 0.0)),
            name=str(getattr(row, "sender_name", "")),
        )
        self._ensure_node(
            node_id=receiver,
            country=str(getattr(row, "receiver_country", "")),
            bank=str(getattr(row, "receiver_bank", "")),
            kyc_level=int(getattr(row, "kyc_level", 0)),
            lat=0.0,
            lon=0.0,
            name=str(getattr(row, "receiver_name", "")),
        )

        # Update node aggregate stats
        sn = self._G.nodes[sender]
        sn["total_sent"] = sn.get("total_sent", 0.0) + amount
        sn["tx_count_sent"] = sn.get("tx_count_sent", 0) + 1
        sn["last_seen_ts"] = ts_str

        rn = self._G.nodes[receiver]
        rn["total_received"] = rn.get("total_received", 0.0) + amount
        rn["tx_count_recv"] = rn.get("tx_count_recv", 0) + 1
        rn["last_seen_ts"] = ts_str

        # Add edge
        edge_attrs: dict[str, Any] = {
            "transaction_id": str(getattr(row, "transaction_id", "")),
            "timestamp": ts_str,
            "amount": amount,
            "currency": str(getattr(row, "currency", "INR")),
            "transaction_type": str(getattr(row, "transaction_type", "")),
            "payment_channel": str(getattr(row, "payment_channel", "")),
            "is_fraud": bool(getattr(row, "is_fraud", False)),
            "ring_id": str(getattr(row, "fraud_ring_id", "")),
        }
        self._G.add_edge(sender, receiver, **edge_attrs)

    def _add_dict(self, tx: dict) -> None:
        sender = tx.get("sender_account", "")
        receiver = tx.get("receiver_account", "")
        ts = tx.get("timestamp")
        ts_str = _ts_iso(ts)
        amount = float(tx.get("amount", 0.0))

        self._ensure_node(
            node_id=sender,
            country=tx.get("sender_country", ""),
            bank=tx.get("sender_bank", ""),
            kyc_level=int(tx.get("kyc_level", 0)),
            lat=float(tx.get("geo_latitude", 0.0)),
            lon=float(tx.get("geo_longitude", 0.0)),
            name=tx.get("sender_name", ""),
        )
        self._ensure_node(
            node_id=receiver,
            country=tx.get("receiver_country", ""),
            bank=tx.get("receiver_bank", ""),
            kyc_level=int(tx.get("kyc_level", 0)),
            lat=0.0, lon=0.0,
            name=tx.get("receiver_name", ""),
        )

        sn = self._G.nodes[sender]
        sn["total_sent"] = sn.get("total_sent", 0.0) + amount
        sn["tx_count_sent"] = sn.get("tx_count_sent", 0) + 1
        sn["last_seen_ts"] = ts_str

        rn = self._G.nodes[receiver]
        rn["total_received"] = rn.get("total_received", 0.0) + amount
        rn["tx_count_recv"] = rn.get("tx_count_recv", 0) + 1
        rn["last_seen_ts"] = ts_str

        self._G.add_edge(sender, receiver,
            transaction_id=tx.get("transaction_id", ""),
            timestamp=ts_str,
            amount=amount,
            currency=tx.get("currency", "INR"),
            transaction_type=tx.get("transaction_type", ""),
            payment_channel=tx.get("payment_channel", ""),
            is_fraud=bool(tx.get("is_fraud", False)),
            ring_id=tx.get("fraud_ring_id", ""),
        )

    def _ensure_node(self, node_id: str, country: str, bank: str,
                     kyc_level: int, lat: float, lon: float, name: str) -> None:
        if node_id not in self._G:
            self._G.add_node(
                node_id,
                account_id=node_id,
                account_name=name,
                country=country,
                bank=bank,
                kyc_level=kyc_level,
                geo_latitude=lat,
                geo_longitude=lon,
                total_sent=0.0,
                total_received=0.0,
                tx_count_sent=0,
                tx_count_recv=0,
                last_seen_ts="",
            )


# ---------------------------------------------------------------------------
# Graph analytics helpers (used by L2)
# ---------------------------------------------------------------------------

def two_hop_neighborhood(G: nx.MultiDiGraph, node: str) -> list[str]:
    """Return all nodes within 2 hops of node (successors + their successors)."""
    one_hop = set(G.successors(node)) | set(G.predecessors(node))
    two_hop = set(one_hop)
    for n in one_hop:
        two_hop |= set(G.successors(n)) | set(G.predecessors(n))
    two_hop.discard(node)
    return sorted(two_hop)


def two_hop_edge_list(
    G: nx.MultiDiGraph,
    node: str,
) -> list[tuple[str, str, str, float]]:
    """
    Return edges (u, v, timestamp_iso, amount) in the 2-hop neighbourhood.
    """
    neighbors = set(two_hop_neighborhood(G, node)) | {node}
    edges = []
    for u, v, data in G.edges(data=True):
        if u in neighbors or v in neighbors:
            edges.append((u, v, data.get("timestamp", ""), data.get("amount", 0.0)))
    return edges


def detect_cycle_closure(G: nx.MultiDiGraph, sender: str, receiver: str) -> tuple[bool, int]:
    """
    Check if adding edge (sender -> receiver) closes a cycle.
    Returns (creates_cycle, shortest_cycle_length_in_existing_graph).
    Uses simple path check: does a path exist from receiver -> sender?
    """
    try:
        path = nx.shortest_path(G, source=receiver, target=sender)
        return True, len(path)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False, 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ts_iso(ts: Any) -> str:
    if isinstance(ts, str):
        return ts
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    return str(ts)
