"""Analytics and reporting routes."""

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..dependencies import get_orchestrator
from ..orchestrator import TransactionOrchestrator

router = APIRouter(prefix="/analytics", tags=["Analytics"])

_COMMUNITY_PROFILES_PATH  = Path("data/cache/community_profiles.parquet")
_SUSPICIOUS_PATHS_PATH    = Path("data/cache/suspicious_paths.parquet")
_BEHAVIORAL_PROFILES_PATH = Path("data/behavioral_profiles.parquet")


@router.get("/community/{community_id}")
async def community_profile(community_id: int) -> dict:
    if not _COMMUNITY_PROFILES_PATH.exists():
        raise HTTPException(status_code=503, detail="community_profiles not available")
    df = pd.read_parquet(_COMMUNITY_PROFILES_PATH)
    row = df[df["community_id"] == community_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Community {community_id} not found")
    return row.iloc[0].to_dict()


@router.get("/communities")
async def list_communities(
    min_risk: float = Query(0.0),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    if not _COMMUNITY_PROFILES_PATH.exists():
        raise HTTPException(status_code=503, detail="community_profiles not available")
    df = pd.read_parquet(_COMMUNITY_PROFILES_PATH)
    df = df[df["risk_score"] >= min_risk].sort_values("risk_score", ascending=False)
    return df.head(limit).to_dict(orient="records")


@router.get("/suspicious-paths")
async def suspicious_paths(
    community_id: Optional[int] = None,
    limit: int = Query(20, ge=1, le=200),
) -> list[dict]:
    if not _SUSPICIOUS_PATHS_PATH.exists():
        raise HTTPException(status_code=503, detail="suspicious_paths not available")
    df = pd.read_parquet(_SUSPICIOUS_PATHS_PATH)
    if community_id is not None:
        df = df[df["community_id"] == community_id]
    return df.head(limit).to_dict(orient="records")


@router.get("/metrics")
async def live_metrics(
    request: Request,
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> dict:
    """
    Live operational metrics combining graph state, alert counts,
    and offline artifact statistics.
    """
    G            = orch.graph_updater.graph
    alert_counts = orch.alert_manager.alert_count()

    # Offline profile counts from parquet (use cache dir)
    n_communities = 0
    n_suspicious  = 0
    n_behavioral  = 0
    if _COMMUNITY_PROFILES_PATH.exists():
        try:
            df_c = pd.read_parquet(_COMMUNITY_PROFILES_PATH)
            n_communities = len(df_c)
        except Exception:
            pass
    if _SUSPICIOUS_PATHS_PATH.exists():
        try:
            df_s = pd.read_parquet(_SUSPICIOUS_PATHS_PATH)
            n_suspicious = len(df_s)
        except Exception:
            pass
    if _BEHAVIORAL_PROFILES_PATH.exists():
        try:
            df_b = pd.read_parquet(_BEHAVIORAL_PROFILES_PATH)
            n_behavioral = len(df_b)
        except Exception:
            pass

    return {
        "graph": {
            "live_nodes": G.number_of_nodes(),
            "live_edges": G.number_of_edges(),
        },
        "alerts": alert_counts,
        "offline": {
            "n_communities":      n_communities,
            "n_suspicious_paths": n_suspicious,
            "n_behavioral_profiles": n_behavioral,
        },
        "engines_loaded": getattr(request.app.state, "engines_loaded", False),
    }


@router.get("/account/{account_id}/profile")
async def account_behavioral_profile(account_id: str) -> dict:
    if not _BEHAVIORAL_PROFILES_PATH.exists():
        raise HTTPException(status_code=503, detail="behavioral_profiles not available")
    df = pd.read_parquet(_BEHAVIORAL_PROFILES_PATH)
    row = df[df["account_id"] == account_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    return row.iloc[0].to_dict()
