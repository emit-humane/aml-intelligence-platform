"""
FastAPI dependency injection for the AML platform.

Dependencies available:
  get_orchestrator()      → TransactionOrchestrator singleton
  get_artifact_store()    → ArtifactStore (from app.state)
  get_live_graph()        → GraphUpdater (from app.state)
  get_db()               → async DB session (SQLAlchemy)

The TransactionOrchestrator and ArtifactStore are initialised once at
app startup (via the lifespan context manager in main.py) and shared
across all requests through app.state.

Usage in route handlers:
    from .dependencies import get_orchestrator, get_artifact_store, get_live_graph

    @router.post("/foo")
    async def foo(
        orch: TransactionOrchestrator = Depends(get_orchestrator),
        store: ArtifactStore          = Depends(get_artifact_store),
    ):
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, Request

from .orchestrator import TransactionOrchestrator

# ─────────────────────────────────────────────────────────────────────────────
# Paths (resolved relative to CWD at runtime)
# ─────────────────────────────────────────────────────────────────────────────

ARTIFACT_DIR    = Path("data/artifacts")
HISTORICAL_DATA = Path("data/raw/all_transactions.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (set by init_orchestrator during lifespan startup)
# ─────────────────────────────────────────────────────────────────────────────

_orchestrator: Optional[TransactionOrchestrator] = None


# ─────────────────────────────────────────────────────────────────────────────
# Initialiser — called once from lifespan
# ─────────────────────────────────────────────────────────────────────────────

def init_orchestrator(
    artifact_dir:       Path = ARTIFACT_DIR,
    historical_parquet: Optional[Path] = None,
) -> TransactionOrchestrator:
    """
    Load all ML engines and build the TransactionOrchestrator.
    Called once from the FastAPI lifespan context manager.
    """
    global _orchestrator
    hp = historical_parquet or (
        HISTORICAL_DATA if HISTORICAL_DATA.exists() else None
    )
    _orchestrator = TransactionOrchestrator.from_artifacts(
        artifact_dir=artifact_dir,
        historical_parquet=hp,
    )
    return _orchestrator


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency: Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def get_orchestrator(request: Request) -> TransactionOrchestrator:
    """
    FastAPI dependency: returns the shared TransactionOrchestrator.

    Reads from app.state.orchestrator (set during lifespan startup).
    Falls back to the module-level singleton for backward compat.
    """
    orch = getattr(request.app.state, "orchestrator", None) or _orchestrator
    if orch is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "ML engines not initialised. "
                "Check startup logs — run 'make train-offline' if artifacts are missing."
            ),
        )
    return orch


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency: ArtifactStore
# ─────────────────────────────────────────────────────────────────────────────

def get_artifact_store(request: Request):
    """
    FastAPI dependency: yields the ArtifactStore from app.state.

    The store is initialised at startup by the lifespan context manager.
    """
    store = getattr(request.app.state, "artifact_store", None)
    if store is None:
        # Fallback: create a new store (read-only access, won't have manifest)
        from ..shared.s3_artifact_store import ArtifactStore
        store = ArtifactStore(ARTIFACT_DIR)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency: Live Graph (GraphUpdater)
# ─────────────────────────────────────────────────────────────────────────────

def get_live_graph(request: Request):
    """
    FastAPI dependency: yields the live GraphUpdater from app.state.

    The graph is seeded from historical data at startup and updated
    incrementally as transactions stream in.
    """
    graph = getattr(request.app.state, "live_graph", None)
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Live graph not initialised. Check startup logs.",
        )
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency: Database session
# ─────────────────────────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator:
    """
    FastAPI dependency: yields an async SQLAlchemy session.

    In production: points to PostgreSQL via DATABASE_URL env var.
    In dev/demo: SQLite at data/aml.db (no Postgres required).
    """
    from ..db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
