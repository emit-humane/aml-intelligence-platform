"""
Health-check and readiness routes.

GET /health       — lightweight liveness probe (always 200 if API is up)
GET /health/ready — readiness probe (200 only when ML engines are loaded)
GET /health/artifacts — manifest of loaded artifacts
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health(request: Request) -> dict:
    """
    Liveness probe.  Always returns 200 while the process is alive.
    Includes whether ML engines are loaded (engines_loaded field).
    """
    engines_loaded: bool = getattr(request.app.state, "engines_loaded", False)
    store = getattr(request.app.state, "artifact_store", None)

    artifact_count = 0
    if store is not None:
        try:
            manifest = store.load_manifest()
            artifact_count = len(manifest.get("artifacts", {}))
        except Exception:
            pass

    return {
        "status":          "ok",
        "artifacts_loaded": engines_loaded,
        "artifact_count":   artifact_count,
    }


@router.get("/health/ready")
async def ready(request: Request) -> dict:
    """
    Readiness probe.
    Returns 200 when engines are loaded, 503 when they are not.
    """
    from fastapi import HTTPException
    engines_loaded: bool = getattr(request.app.state, "engines_loaded", False)
    if not engines_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML engines not yet loaded. Check startup logs.",
        )
    return {"ready": True, "engines_loaded": True}


@router.get("/health/artifacts")
async def list_artifacts(request: Request) -> dict:
    """Return the artifact manifest (what's loaded in artifact_store)."""
    store = getattr(request.app.state, "artifact_store", None)
    if store is None:
        return {"artifacts": {}, "root": None}
    manifest = store.load_manifest()
    return {
        "root":      manifest.get("root"),
        "count":     len(manifest.get("artifacts", {})),
        "artifacts": manifest.get("artifacts", {}),
    }
