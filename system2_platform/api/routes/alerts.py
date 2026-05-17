"""Alert management REST routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..dependencies import get_orchestrator
from ..orchestrator import TransactionOrchestrator
from ...contracts.alert import Alert, AlertStatus

router = APIRouter(prefix="/alerts", tags=["Alerts"])


class StatusUpdate(BaseModel):
    status: AlertStatus
    assigned_to: Optional[str] = None


@router.get("/", response_model=list[Alert])
async def list_alerts(
    risk_level: Optional[str] = Query(None, description="Low | Medium | High | Critical"),
    status: Optional[str] = Query(None, description="Open | Investigating | SAR_Filed | Closed"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> list[Alert]:
    return orch.alert_manager.get_alerts(
        risk_level=risk_level, status=status, limit=limit, offset=offset
    )


@router.get("/stats")
async def alert_stats(
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> dict:
    return orch.alert_manager.alert_count()


@router.get("/{alert_id}", response_model=Alert)
async def get_alert(
    alert_id: str,
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> Alert:
    alert = orch.alert_manager.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return alert


@router.patch("/{alert_id}/status", response_model=Alert)
async def update_alert_status(
    alert_id: str,
    body: StatusUpdate,
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> Alert:
    updated = orch.alert_manager.update_status(alert_id, body.status, body.assigned_to)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return updated
