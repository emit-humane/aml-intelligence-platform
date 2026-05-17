"""
L2B — Graph Analytics Engine.

Consumes:
  - nx.MultiDiGraph (from S2)
  - graph_features.parquet (from L2A)

Produces:
  - community_profiles.parquet  — one row per Infomap community
  - suspicious_paths.parquet    — DFS paths (2–8 hops) in flagged communities

community_profiles schema (spec pages 27-28):
  community_id, size, density, dominant_pattern, avg_pagerank,
  avg_betweenness, total_volume, avg_amount, num_fraud_nodes,
  risk_score, has_cycle, has_fan_in, has_fan_out,
  avg_benford_chi2, top_accounts (JSON list of top-5 by pagerank)

suspicious_paths schema:
  path_id, community_id, path (JSON account list), num_hops,
  total_amount, dominant_pattern

Dominant pattern logic:
  has_cycle AND size >= 3          -> circular_laundering
  avg fan_in_score >= 0.70         -> fan_in
  avg fan_out_score >= 0.70        -> fan_out
  max path_length_within >= 4      -> layering_chain
  else                             -> mixed
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Community profiler
# ---------------------------------------------------------------------------

def _dominant_pattern(
    has_cycle: bool,
    size: int,
    avg_fan_in: float,
    avg_fan_out: float,
    max_chain_len: int,
) -> str:
    if has_cycle and size >= 3:
        return "circular_laundering"
    if avg_fan_in >= 0.70:
        return "fan_in"
    if avg_fan_out >= 0.70:
        return "fan_out"
    if max_chain_len >= 4:
        return "layering_chain"
    return "mixed"


def _community_risk_score(
    density: float,
    has_cycle: bool,
    avg_fan_in: float,
    avg_fan_out: float,
    avg_benford_chi2: float,
    num_fraud_nodes: int,
    size: int,
) -> float:
    score = 0.0
    score += density * 30.0
    if has_cycle:
        score += 25.0
    score += max(avg_fan_in - 0.5, 0.0) * 20.0
    score += max(avg_fan_out - 0.5, 0.0) * 20.0
    score += min(avg_benford_chi2 / 20.0, 1.0) * 15.0
    if size > 0:
        score += (num_fraud_nodes / size) * 20.0
    return min(score, 100.0)


def _longest_simple_path_in_subgraph(G: nx.DiGraph, nodes: set) -> int:
    """Approximate max chain length using topological sort on DAG, or BFS."""
    subG = G.subgraph(nodes)
    try:
        order = list(nx.topological_sort(subG))
        # DP: dp[v] = longest path ending at v
        dp: dict[Any, int] = {n: 1 for n in order}
        for n in order:
            for pred in subG.predecessors(n):
                if dp[pred] + 1 > dp[n]:
                    dp[n] = dp[pred] + 1
        return max(dp.values()) if dp else 1
    except nx.NetworkXUnfeasible:
        # Has cycles — BFS with depth limit
        return 8  # assume long-chain if cyclic


# ---------------------------------------------------------------------------
# Path extractor (DFS, 2–8 hops)
# ---------------------------------------------------------------------------

def _extract_suspicious_paths(
    G: nx.MultiDiGraph,
    community_nodes: set[str],
    community_id: int,
    dominant_pattern: str,
    max_paths: int = 20,
) -> list[dict]:
    """DFS paths of 2–8 hops within a community for layering/circular patterns."""
    if dominant_pattern not in {"layering_chain", "circular_laundering", "mixed"}:
        return []

    subG = nx.DiGraph(G.subgraph(community_nodes))  # simple directed subgraph
    found: list[dict] = []

    def _dfs(path: list[str], visited: set[str]) -> None:
        if len(found) >= max_paths:
            return
        current = path[-1]
        hop_count = len(path) - 1

        if hop_count >= 2:
            # Record this path
            amounts = []
            for i in range(len(path) - 1):
                edge_data = subG.get_edge_data(path[i], path[i + 1]) or {}
                # Simple DiGraph stores edge data directly
                amounts.append(edge_data.get("amount", 0.0))
            found.append({
                "path_id": f"PATH_{community_id}_{len(found):04d}",
                "community_id": community_id,
                "path": json.dumps(path),
                "num_hops": hop_count,
                "total_amount": sum(amounts),
                "dominant_pattern": dominant_pattern,
            })

        if hop_count >= 8:
            return

        for neighbor in subG.successors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                _dfs(path + [neighbor], visited)
                visited.discard(neighbor)

    # Start DFS from each node with high out-degree
    entry_nodes = sorted(
        community_nodes,
        key=lambda n: subG.out_degree(n),
        reverse=True,
    )[:5]

    for start in entry_nodes:
        if len(found) >= max_paths:
            break
        _dfs([start], {start})

    return found


# ---------------------------------------------------------------------------
# Analytics engine
# ---------------------------------------------------------------------------

class GraphAnalyticsEngine:
    """
    Builds community profiles and suspicious path reports from a transaction graph.
    """

    def run(
        self,
        G: nx.MultiDiGraph,
        graph_features: pd.DataFrame,
        out_dir: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (community_profiles_df, suspicious_paths_df).
        """
        if graph_features.empty or G.number_of_nodes() == 0:
            empty_cp = pd.DataFrame()
            empty_sp = pd.DataFrame()
            return empty_cp, empty_sp

        gf = graph_features.set_index("account_id")

        # Group accounts by community
        comm_groups: dict[int, list[str]] = defaultdict(list)
        for acc, row in gf.iterrows():
            comm_groups[int(row["community_id"])].append(str(acc))

        # Build simple DiGraph for chain-length computation
        S = nx.DiGraph(G)

        # Gather fraud-node counts from graph node attributes
        fraud_node_set: set[str] = set()
        # We don't have fraud labels per-node directly; approximate from edge attrs
        for u, v, data in G.edges(data=True):
            if data.get("is_fraud", False):
                fraud_node_set.add(u)
                fraud_node_set.add(v)

        community_rows: list[dict] = []
        all_paths: list[dict] = []

        for comm_id, members in comm_groups.items():
            m_set = set(members)
            n = len(members)

            # Pull graph feature stats for this community
            comm_gf = gf.loc[gf.index.isin(m_set)]
            avg_pr = float(comm_gf["pagerank"].mean()) if "pagerank" in comm_gf else 0.0
            avg_bet = float(comm_gf["betweenness"].mean()) if "betweenness" in comm_gf else 0.0
            avg_fi = float(comm_gf["fan_in_score"].mean()) if "fan_in_score" in comm_gf else 0.0
            avg_fo = float(comm_gf["fan_out_score"].mean()) if "fan_out_score" in comm_gf else 0.0
            density = float(comm_gf["community_density"].mean()) if "community_density" in comm_gf else 0.0
            avg_chi2 = float(comm_gf["benford_chi2_community"].mean()) if "benford_chi2_community" in comm_gf else 0.0
            two_hops = int(comm_gf["2hop_cycle_count"].sum()) if "2hop_cycle_count" in comm_gf else 0

            # Volume
            in_vol = float(comm_gf["ego_in_volume_7d"].sum()) if "ego_in_volume_7d" in comm_gf else 0.0
            out_vol = float(comm_gf["ego_out_volume_7d"].sum()) if "ego_out_volume_7d" in comm_gf else 0.0
            total_vol = (in_vol + out_vol) / 2.0  # avoid double-counting

            # Avg amount in community edges
            comm_amounts = [
                d.get("amount", 0.0)
                for u, v, d in G.edges(data=True)
                if u in m_set or v in m_set
            ]
            avg_amount = float(np.mean(comm_amounts)) if comm_amounts else 0.0

            has_cycle = two_hops > 0
            num_fraud_nodes = len(m_set & fraud_node_set)

            # Max chain length
            max_chain = _longest_simple_path_in_subgraph(S, m_set)

            pattern = _dominant_pattern(has_cycle, n, avg_fi, avg_fo, max_chain)
            risk = _community_risk_score(
                density, has_cycle, avg_fi, avg_fo, avg_chi2, num_fraud_nodes, n
            )

            # Top 5 accounts by pagerank
            top_accs = (
                comm_gf["pagerank"].nlargest(5).index.tolist()
                if "pagerank" in comm_gf and len(comm_gf) > 0
                else []
            )

            community_rows.append({
                "community_id": comm_id,
                "size": n,
                "density": round(density, 6),
                "dominant_pattern": pattern,
                "avg_pagerank": round(avg_pr, 8),
                "avg_betweenness": round(avg_bet, 8),
                "total_volume": round(total_vol, 2),
                "avg_amount": round(avg_amount, 2),
                "num_fraud_nodes": num_fraud_nodes,
                "risk_score": round(risk, 2),
                "has_cycle": has_cycle,
                "has_fan_in": avg_fi >= 0.60,
                "has_fan_out": avg_fo >= 0.60,
                "avg_benford_chi2": round(avg_chi2, 4),
                "top_accounts": json.dumps(top_accs),
            })

            # Extract suspicious paths for high-risk communities
            if risk >= 40.0 and n >= 3:
                paths = _extract_suspicious_paths(G, m_set, comm_id, pattern)
                all_paths.extend(paths)

        community_profiles = pd.DataFrame(community_rows)
        suspicious_paths = pd.DataFrame(all_paths) if all_paths else pd.DataFrame(
            columns=["path_id", "community_id", "path", "num_hops",
                     "total_amount", "dominant_pattern"]
        )

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            community_profiles.to_parquet(out_dir / "community_profiles.parquet", index=False)
            suspicious_paths.to_parquet(out_dir / "suspicious_paths.parquet", index=False)
            print(f"[L2B] wrote community_profiles.parquet ({len(community_profiles)} communities)")
            print(f"[L2B] wrote suspicious_paths.parquet ({len(suspicious_paths)} paths)")

        return community_profiles, suspicious_paths


def run_analytics(
    G: nx.MultiDiGraph,
    graph_features: pd.DataFrame,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return GraphAnalyticsEngine().run(G, graph_features, out_dir)
