"""
AML Intelligence Platform — FastAPI Application.

Startup sequence (lifespan):
  1. ArtifactStore.load_manifest()    — validate artifacts present
  2. TransactionOrchestrator.from_artifacts() — load all ML engines
  3. Store orchestrator + store + live_graph in app.state

Mount order:
  /health, /stream, /alerts, /graph, /analytics

Serve:
  uvicorn system2_platform.api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import init_orchestrator, ARTIFACT_DIR, HISTORICAL_DATA
from .routes import (
    health_router,
    stream_router,
    alerts_router,
    graph_router,
    analytics_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    On startup:
      - Load ArtifactStore and validate manifest
      - Initialise TransactionOrchestrator (loads all ML engines)
      - Expose orchestrator, artifact_store, live_graph via app.state

    On shutdown:
      - Graceful cleanup (nothing expensive for in-memory store)
    """
    print("[API] ====================================")
    print("[API] AML Intelligence Platform starting")
    print("[API] ====================================")

    # ── 1. Artifact store ───────────────────────────────────────────────────
    from ..shared.s3_artifact_store import ArtifactStore
    store = ArtifactStore(ARTIFACT_DIR)
    manifest = store.load_manifest()
    n_artifacts = len(manifest.get("artifacts", {}))
    print(f"[API] ArtifactStore: {n_artifacts} artifacts in {ARTIFACT_DIR}")
    app.state.artifact_store = store

    # ── 2. ML engines (orchestrator) ────────────────────────────────────────
    try:
        orchestrator = init_orchestrator(
            artifact_dir=ARTIFACT_DIR,
            historical_parquet=HISTORICAL_DATA if HISTORICAL_DATA.exists() else None,
        )
        app.state.orchestrator    = orchestrator
        app.state.engines_loaded  = True
        app.state.live_graph      = orchestrator.graph_updater
        print("[API] All ML engines loaded. OK")
    except Exception as exc:
        print(f"[API] WARNING: ML engine load failed: {exc}")
        print("[API] Scoring endpoints will return 503 until engines are loaded.")
        app.state.orchestrator   = None
        app.state.engines_loaded = False
        app.state.live_graph     = None

    print("[API] Ready to serve requests.")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("[API] Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AML Intelligence Platform",
    description=(
        "Real-time anti-money-laundering detection API. "
        "Four detection layers: Rule Engine → Graph Analytics → "
        "Behavioral Anomaly (IsoForest+LOF+AE) → Temporal GNN (TGN+MegaGNN)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server and production origins
import os as _os

_FRONTEND_ORIGIN = _os.getenv("FRONTEND_ORIGIN", "")
_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
if _FRONTEND_ORIGIN:
    _CORS_ORIGINS.append(_FRONTEND_ORIGIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.(vercel\.app|onrender\.com)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(stream_router)
app.include_router(alerts_router)
app.include_router(graph_router)
app.include_router(analytics_router)


@app.get("/", tags=["Root"])
async def root() -> dict:
    return {
        "name":    "AML Intelligence Platform",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }
