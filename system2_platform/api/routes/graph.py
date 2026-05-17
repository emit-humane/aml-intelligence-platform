"""Graph introspection routes."""

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_orchestrator
from ..orchestrator import TransactionOrchestrator

router = APIRouter(prefix="/graph", tags=["Graph"])


@router.get("/stats")
async def graph_stats(
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> dict:
    G = orch.graph_updater.graph
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
    }


@router.get("/node/{account_id}")
async def node_info(
    account_id: str,
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> dict:
    G = orch.graph_updater.graph
    if not G.has_node(account_id):
        raise HTTPException(status_code=404, detail=f"Account {account_id} not in graph")
    data = dict(G.nodes[account_id])
    data["in_degree"]  = G.in_degree(account_id)
    data["out_degree"] = G.out_degree(account_id)
    return data


@router.get("/node/{account_id}/neighbors")
async def node_neighbors(
    account_id: str,
    hops: int = Query(1, ge=1, le=2),
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> dict:
    G = orch.graph_updater.graph
    if not G.has_node(account_id):
        raise HTTPException(status_code=404, detail=f"Account {account_id} not in graph")
    successors   = list(G.successors(account_id))
    predecessors = list(G.predecessors(account_id))
    return {
        "account_id":   account_id,
        "successors":   successors[:50],
        "predecessors": predecessors[:50],
    }


@router.get("/node/{account_id}/embedding")
async def node_embedding(
    account_id: str,
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> dict:
    emb = orch._gnn.get_embedding(account_id)
    if emb is None:
        raise HTTPException(status_code=404, detail=f"No embedding for {account_id}")
    return {"account_id": account_id, "embedding": emb.tolist(), "dim": len(emb)}


@router.get("/subgraph/{account_id}")
async def subgraph(
    account_id: str,
    hops: int = Query(2, ge=1, le=2, description="BFS depth (1 or 2)"),
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> dict:
    """
    Return the N-hop ego subgraph around account_id as node + edge lists.

    Response:
      {
        "center": "<account_id>",
        "hops":   2,
        "nodes":  [{"id": "...", "in_degree": N, "out_degree": N}, ...],
        "edges":  [{"source": "...", "target": "...", "weight": N}, ...]
      }
    """
    G = orch.graph_updater.graph
    if not G.has_node(account_id):
        raise HTTPException(status_code=404, detail=f"Account {account_id} not in graph")

    # BFS to collect nodes within `hops` of account_id
    visited: set[str] = {account_id}
    frontier: set[str] = {account_id}
    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for nb in list(G.successors(node)) + list(G.predecessors(node)):
                if nb not in visited:
                    next_frontier.add(nb)
        visited |= next_frontier
        frontier = next_frontier

    # Build node list (cap at 500 nodes to avoid huge payloads)
    nodes = []
    for n in list(visited)[:500]:
        nodes.append({
            "id":         n,
            "in_degree":  G.in_degree(n),
            "out_degree": G.out_degree(n),
            "is_center":  n == account_id,
        })

    # Build edge list — only edges where BOTH endpoints are in the subgraph
    edge_dict: dict[tuple, int] = {}
    for src in visited:
        for dst in G.successors(src):
            if dst in visited:
                key = (src, dst)
                edge_dict[key] = edge_dict.get(key, 0) + 1

    edges = [
        {"source": s, "target": d, "weight": w}
        for (s, d), w in list(edge_dict.items())[:2000]
    ]

    return {
        "center": account_id,
        "hops":   hops,
        "nodes":  nodes,
        "edges":  edges,
    }
