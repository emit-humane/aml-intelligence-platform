"""
S6 — Live Graph Update.

Maintains a live MultiDiGraph and updates it with each incoming transaction.
Also extracts a LiveGraphFeatureVector for the transaction pair.

Usage
-----
updater = GraphUpdater(artifact_dir)
gfv = updater.update_and_extract(event)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx

from ..contracts.transaction_event import TransactionEvent
from ..contracts.live_graph_feature_vector import LiveGraphFeatureVector
from ..shared.s2_multigraph_builder import GraphBuilder


class GraphUpdater:
    """
    Live graph updater for streaming inference.

    Maintains an in-memory MultiDiGraph seeded from historical data
    (if graph_features.parquet and accounts.parquet are available)
    and updated with each new stream event.
    """

    def __init__(self) -> None:
        self._builder = GraphBuilder()
        self._G: Optional[nx.MultiDiGraph] = None

    @property
    def graph(self) -> nx.MultiDiGraph:
        if self._G is None:
            self._G = self._builder.get_graph()
        return self._G

    def seed_from_historical(self, historical_parquet: Path | str) -> None:
        """Load historical transactions to seed the graph."""
        import pandas as pd
        df = pd.read_parquet(str(historical_parquet))
        print(f"[S6] Seeding graph from {len(df):,} historical transactions...")
        self._builder.build(df)
        self._G = self._builder.get_graph()
        print(f"[S6] Graph seeded: {self._G.number_of_nodes()} nodes, "
              f"{self._G.number_of_edges()} edges")

    def update_and_extract(self, event: TransactionEvent) -> LiveGraphFeatureVector:
        """
        Add the transaction to the graph and return graph feature vector.
        """
        G = self.graph

        # Update graph with this transaction
        tx_dict = {
            "transaction_id": event.transaction_id,
            "timestamp": event.timestamp.isoformat(),
            "amount": event.amount,
            "currency": event.currency,
            "transaction_type": event.transaction_type,
            "payment_channel": event.payment_channel,
            "is_international": event.is_international,
            "transaction_status": event.transaction_status,
        }

        # Ensure nodes exist
        if not G.has_node(event.sender_account):
            G.add_node(event.sender_account,
                       account_id=event.sender_account,
                       country=event.sender_country,
                       bank=event.sender_bank,
                       kyc_level=event.kyc_level,
                       lat=event.geo_latitude,
                       lon=event.geo_longitude,
                       total_sent=0.0, total_received=0.0,
                       tx_count_sent=0, tx_count_recv=0)
        if not G.has_node(event.receiver_account):
            G.add_node(event.receiver_account,
                       account_id=event.receiver_account,
                       country=event.receiver_country,
                       bank=event.receiver_bank,
                       kyc_level=event.kyc_level,
                       lat=0.0, lon=0.0,
                       total_sent=0.0, total_received=0.0,
                       tx_count_sent=0, tx_count_recv=0)

        G.add_edge(event.sender_account, event.receiver_account, **tx_dict)

        # Update node aggregates
        G.nodes[event.sender_account]["total_sent"] += event.amount
        G.nodes[event.sender_account]["tx_count_sent"] += 1
        G.nodes[event.receiver_account]["total_received"] += event.amount
        G.nodes[event.receiver_account]["tx_count_recv"] += 1

        # --- Extract features ---
        src = event.sender_account
        dst = event.receiver_account

        src_in_deg  = G.in_degree(src)
        src_out_deg = G.out_degree(src)
        dst_in_deg  = G.in_degree(dst)
        dst_out_deg = G.out_degree(dst)

        # Unique predecessor/successor counts
        src_in_unique  = len(list(G.predecessors(src)))
        src_out_unique = len(list(G.successors(src)))
        dst_in_unique  = len(list(G.predecessors(dst)))  # fan-in signal

        # Cycle detection (lightweight: look for src in dst's successors within 3 hops)
        creates_cycle, cycle_len = _detect_cycle(G, src, dst, max_depth=3)

        # 2-hop neighbourhood of src
        two_hop = _two_hop_neighbors(G, src, max_nodes=20)
        two_hop_edges = _two_hop_edge_list(G, src, max_edges=30)

        # Community detection skipped for live inference (too expensive per-transaction)
        # Use defaults
        return LiveGraphFeatureVector(
            transaction_id=event.transaction_id,
            sender_account=src,
            receiver_account=dst,
            sender_in_degree=src_in_deg,
            sender_out_degree=src_out_deg,
            sender_in_degree_unique=src_in_unique,
            sender_out_degree_unique=src_out_unique,
            sender_2hop_cycle_count=0,
            sender_3hop_cycle_count=0,
            sender_fan_in_score=_fan_score(src_in_deg, src_out_deg, direction="in"),
            sender_fan_out_score=_fan_score(src_in_deg, src_out_deg, direction="out"),
            sender_pagerank=0.0,
            sender_betweenness=0.0,
            sender_community_id=-1,
            sender_community_size=0,
            sender_community_density=0.0,
            sender_community_risk_score=0.0,
            sender_benford_chi2_community=0.0,
            receiver_in_degree=dst_in_deg,
            receiver_in_degree_unique=dst_in_unique,
            receiver_out_degree=dst_out_deg,
            receiver_community_id=-1,
            receiver_community_risk_score=0.0,
            edge_creates_cycle=creates_cycle,
            cycle_length=cycle_len,
            shared_community=False,
            two_hop_neighborhood=two_hop,
            two_hop_edge_list=two_hop_edges,
        )


def _fan_score(in_deg: int, out_deg: int, direction: str) -> float:
    total = in_deg + out_deg
    if total == 0:
        return 0.0
    if direction == "in":
        return in_deg / total
    return out_deg / total


def _detect_cycle(
    G: nx.MultiDiGraph,
    src: str,
    dst: str,
    max_depth: int = 3,
) -> tuple[bool, int]:
    """
    BFS from dst looking for src within max_depth hops.
    Returns (creates_cycle, hop_count). O(d^max_depth) — fast for small depth.
    """
    if src == dst:
        return True, 1
    frontier = {dst}
    for depth in range(1, max_depth + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for succ in G.successors(node):
                if succ == src:
                    return True, depth + 1
                next_frontier.add(succ)
        frontier = next_frontier
        if not frontier:
            break
    return False, 0


def _two_hop_neighbors(G: nx.MultiDiGraph, src: str, max_nodes: int = 20) -> list[str]:
    """Return up to max_nodes nodes within 2 hops of src."""
    hop1 = set(G.successors(src)) | set(G.predecessors(src))
    hop2: set[str] = set()
    for n in hop1:
        hop2 |= set(G.successors(n)) | set(G.predecessors(n))
    all_nodes = (hop1 | hop2) - {src}
    return list(all_nodes)[:max_nodes]


def _two_hop_edge_list(
    G: nx.MultiDiGraph,
    src: str,
    max_edges: int = 30,
) -> list[tuple[str, str, str, float]]:
    """Return up to max_edges edges within 1 hop of src as (u, v, ts, amount)."""
    edges = []
    for u, v, data in G.out_edges(src, data=True):
        edges.append((u, v, data.get("timestamp", ""), float(data.get("amount", 0.0))))
        if len(edges) >= max_edges:
            break
    for u, v, data in G.in_edges(src, data=True):
        edges.append((u, v, data.get("timestamp", ""), float(data.get("amount", 0.0))))
        if len(edges) >= max_edges:
            break
    return edges[:max_edges]
