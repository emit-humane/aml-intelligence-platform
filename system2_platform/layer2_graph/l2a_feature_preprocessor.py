"""
L2A — Graph Feature Preprocessor.

Takes a nx.MultiDiGraph (from S2 GraphBuilder) and computes per-node
(per-account) graph features, writing graph_features.parquet.

Output columns (one row per account / graph node):
  account_id, in_degree, out_degree, total_degree,
  in_degree_unique, out_degree_unique,
  pagerank, betweenness,
  clustering_coefficient,        -- using undirected projection
  2hop_cycle_count, 3hop_cycle_count,
  fan_in_score, fan_out_score, scatter_gather_score,
  community_id, community_size, community_density,
  ego_in_volume_7d, ego_out_volume_7d,
  volume_asymmetry, benford_chi2_community

Community detection: directed Infomap (fallback: weakly connected components).
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Benford helper (duplicated locally to avoid circular import)
# ---------------------------------------------------------------------------

_BENFORD_P = np.log10(1.0 + 1.0 / np.arange(1, 10))
_BENFORD_P /= _BENFORD_P.sum()


def _leading_digit(x: float) -> int:
    if x <= 0:
        return 1
    s = f"{x:.2f}".lstrip("0").replace(".", "")
    return int(s[0]) if s else 1


def _benford_chi2(amounts: list[float]) -> float:
    if len(amounts) < 5:
        return 0.0
    obs = np.zeros(9)
    for a in amounts:
        obs[_leading_digit(a) - 1] += 1
    n = obs.sum()
    expected = _BENFORD_P * n
    return float(np.sum((obs - expected) ** 2 / (expected + 1e-9)))


# ---------------------------------------------------------------------------
# Infomap community detection
# ---------------------------------------------------------------------------

def _infomap_communities(G: nx.MultiDiGraph) -> dict[str, int]:
    """Directed Infomap. Falls back to weakly-connected components."""
    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}
    int_to_node = {i: n for n, i in node_to_int.items()}

    try:
        import infomap as _imp  # type: ignore
        im = _imp.Infomap("--directed --two-level --silent --num-trials 3")
        for u, v in G.edges():
            im.add_link(node_to_int[u], node_to_int[v])
        im.run()
        communities: dict[str, int] = {}
        for node_int, module_id in im.modules:
            node_name = int_to_node.get(node_int)
            if node_name is not None:
                communities[node_name] = int(module_id)
        # Assign any unassigned nodes
        for n in nodes:
            if n not in communities:
                communities[n] = 0
        return communities
    except Exception:
        pass

    # Fallback: weakly connected components as pseudo-communities
    communities = {}
    for comm_id, component in enumerate(nx.weakly_connected_components(G)):
        for node in component:
            communities[node] = comm_id
    return communities


# ---------------------------------------------------------------------------
# Cycle count helpers
# ---------------------------------------------------------------------------

def _cycle_counts(G: nx.MultiDiGraph) -> tuple[dict[str, int], dict[str, int]]:
    """
    For each node, count 2-hop and 3-hop cycles using sparse matrix multiplication.

    A²[v,v] = number of 2-hop cycles through v (v->u->v paths)
    A³[v,v] = number of 3-hop cycles through v (v->u->w->v paths)
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    # Build simple adjacency (deduplicate multi-edges)
    seen_edges: set[tuple[int, int]] = set()
    rows, cols = [], []
    for u, v in G.edges():
        ui, vi = node_idx[u], node_idx[v]
        if (ui, vi) not in seen_edges:
            seen_edges.add((ui, vi))
            rows.append(ui)
            cols.append(vi)

    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )
    A2 = A @ A
    A3 = A2 @ A

    diag2 = np.array(A2.diagonal(), dtype=int)
    diag3 = np.array(A3.diagonal(), dtype=int)

    two_hop = {nd: int(diag2[i]) for i, nd in enumerate(nodes)}
    three_hop = {nd: int(diag3[i]) for i, nd in enumerate(nodes)}
    return two_hop, three_hop


# ---------------------------------------------------------------------------
# Main preprocessor
# ---------------------------------------------------------------------------

class GraphFeaturePreprocessor:
    """
    Computes per-account graph feature rows from a MultiDiGraph.
    """

    def compute(
        self,
        G: nx.MultiDiGraph,
        out_dir: Path | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        G       : transaction MultiDiGraph from GraphBuilder
        out_dir : if given, writes graph_features.parquet there

        Returns DataFrame with one row per node (account).
        """
        if G.number_of_nodes() == 0:
            return pd.DataFrame()

        nodes = list(G.nodes())
        print(f"[L2A] Computing graph features for {len(nodes)} nodes, "
              f"{G.number_of_edges()} edges...")

        # --- Degree features ---
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())

        # Unique neighbours (ignoring parallel edges)
        S = nx.DiGraph(G)
        in_deg_unique = {n: S.in_degree(n) for n in nodes}
        out_deg_unique = {n: S.out_degree(n) for n in nodes}

        # --- PageRank ---
        print("[L2A] Computing PageRank...")
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)

        # --- Betweenness (approximate, k=min(100, n)) ---
        print("[L2A] Computing betweenness (approximate)...")
        k_bet = min(100, len(nodes))
        betweenness = nx.betweenness_centrality(G, k=k_bet, normalized=True, seed=42)

        # --- Clustering coefficient (undirected projection) ---
        U = nx.Graph(G)
        clustering = nx.clustering(U)

        # --- Cycle counts (2-hop, 3-hop) ---
        print("[L2A] Computing cycle counts...")
        two_hop_cycles, three_hop_cycles = _cycle_counts(G)

        # --- Fan-in / fan-out / scatter-gather scores ---
        # fan_in_score  = in_degree / (in_degree + out_degree + 1e-9)
        # fan_out_score = out_degree / (in_degree + out_degree + 1e-9)
        # scatter_gather: a node that both receives from many AND sends to many
        #   = harmonic mean of fan_in and fan_out normalised by log(degree+1)
        fan_in: dict[str, float] = {}
        fan_out: dict[str, float] = {}
        scatter_gather: dict[str, float] = {}
        for n in nodes:
            i = float(in_deg[n])
            o = float(out_deg[n])
            total = i + o + 1e-9
            fi = i / total
            fo = o / total
            fan_in[n] = fi
            fan_out[n] = fo
            # scatter-gather: high when both fi and fo are moderate (~0.5 each)
            sg = 1.0 - abs(fi - fo)  # max at fi==fo==0.5
            sg *= math.log(total + 1) / 5.0  # scale by activity
            scatter_gather[n] = min(sg, 1.0)

        # --- Community detection ---
        print("[L2A] Running Infomap community detection...")
        communities = _infomap_communities(G)

        comm_members: dict[int, list[str]] = defaultdict(list)
        for node, comm_id in communities.items():
            comm_members[comm_id].append(node)
        comm_size = {n: len(comm_members[communities[n]]) for n in nodes}

        # --- Community density: edges within / max possible ---
        comm_density: dict[str, float] = {}
        for comm_id, members in comm_members.items():
            m_set = set(members)
            n_m = len(members)
            if n_m < 2:
                for n in members:
                    comm_density[n] = 0.0
                continue
            internal_edges = sum(
                1 for u, v in S.edges()
                if u in m_set and v in m_set
            )
            max_edges = n_m * (n_m - 1)
            density = internal_edges / max_edges if max_edges > 0 else 0.0
            for n in members:
                comm_density[n] = density

        # --- Ego volume (in/out) — sum edge weights in ego neighbourhood ---
        ego_in_vol: dict[str, float] = {}
        ego_out_vol: dict[str, float] = {}
        for n in nodes:
            out_amounts = [d.get("amount", 0.0) for _, _, d in G.out_edges(n, data=True)]
            in_amounts = [d.get("amount", 0.0) for _, _, d in G.in_edges(n, data=True)]
            ego_out_vol[n] = sum(out_amounts)
            ego_in_vol[n] = sum(in_amounts)

        # --- Volume asymmetry: (out - in) / (out + in + 1) ---
        vol_asym: dict[str, float] = {}
        for n in nodes:
            o = ego_out_vol[n]
            i = ego_in_vol[n]
            vol_asym[n] = (o - i) / (o + i + 1.0)

        # --- Benford chi2 per community ---
        print("[L2A] Computing Benford chi2 per community...")
        comm_amounts: dict[int, list[float]] = defaultdict(list)
        for u, v, data in G.edges(data=True):
            c = communities.get(u, 0)
            comm_amounts[c].append(data.get("amount", 0.0))
        benford_comm: dict[str, float] = {
            n: _benford_chi2(comm_amounts[communities[n]])
            for n in nodes
        }

        # --- Assemble DataFrame ---
        df = pd.DataFrame({
            "account_id": nodes,
            "in_degree": [in_deg[n] for n in nodes],
            "out_degree": [out_deg[n] for n in nodes],
            "total_degree": [in_deg[n] + out_deg[n] for n in nodes],
            "in_degree_unique": [in_deg_unique[n] for n in nodes],
            "out_degree_unique": [out_deg_unique[n] for n in nodes],
            "pagerank": [pagerank.get(n, 0.0) for n in nodes],
            "betweenness": [betweenness.get(n, 0.0) for n in nodes],
            "clustering_coefficient": [clustering.get(n, 0.0) for n in nodes],
            "2hop_cycle_count": [two_hop_cycles.get(n, 0) for n in nodes],
            "3hop_cycle_count": [three_hop_cycles.get(n, 0) for n in nodes],
            "fan_in_score": [fan_in[n] for n in nodes],
            "fan_out_score": [fan_out[n] for n in nodes],
            "scatter_gather_score": [scatter_gather[n] for n in nodes],
            "community_id": [communities[n] for n in nodes],
            "community_size": [comm_size[n] for n in nodes],
            "community_density": [comm_density.get(n, 0.0) for n in nodes],
            "ego_in_volume_7d": [ego_in_vol[n] for n in nodes],
            "ego_out_volume_7d": [ego_out_vol[n] for n in nodes],
            "volume_asymmetry": [vol_asym[n] for n in nodes],
            "benford_chi2_community": [benford_comm[n] for n in nodes],
        })

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_dir / "graph_features.parquet", index=False)
            print(f"[L2A] wrote graph_features.parquet ({len(df)} rows) -> {out_dir}")

        return df


def build_graph_features(
    G: nx.MultiDiGraph,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Convenience wrapper."""
    return GraphFeaturePreprocessor().compute(G, out_dir)
