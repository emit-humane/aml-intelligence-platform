"""
Stream ingestion and WebSocket broadcasting routes.

POST /stream/transaction     -- score a single transaction; result broadcast to all WS clients
POST /stream/batch           -- score a list of transactions in order
POST /stream/start           -- begin replaying a CSV/JSONL file at a configurable rate
GET  /stream/stop            -- stop the running replay loop
WS   /stream/ws              -- subscribe to live scored-event feed (receive-only)
WS   /stream/ws/interactive  -- bidirectional: send transaction JSON, receive scored result

Architecture:
  Each POST /stream/transaction:
    1. Parses + scores via TransactionOrchestrator.async_process()
    2. Broadcasts serialised result to all connected WebSocket subscribers
    3. Returns the full scored result as HTTP response

  WebSocket /stream/ws clients connect and passively receive JSON objects
  as they are submitted via POST (replay, upstream ingestion, etc.).
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..dependencies import get_orchestrator
from ..orchestrator import TransactionOrchestrator
from ...contracts.transaction_event import TransactionEvent
from ...shared.s4_stream_ingestion import StreamIngestion

router = APIRouter(prefix="/stream", tags=["Stream"])
_ingestion = StreamIngestion()

# Replay state — one replay task at a time
_replay_task: Optional[asyncio.Task] = None


# ─────────────────────────────────────────────────────────────────────────────
# Connection manager — maintains the set of live WebSocket subscribers
# ─────────────────────────────────────────────────────────────────────────────

class _ConnectionManager:
    """Thread-safe (asyncio) broadcast manager for WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, data: dict) -> None:
        """Send data to every connected client; silently drop disconnected ones."""
        dead: list[WebSocket] = []
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self._clients)


_manager = _ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/transaction")
async def score_transaction(
    payload: dict[str, Any],
    response: Response,
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> dict:
    """
    Score a single transaction.

    - Runs the full 4-layer pipeline via async_process() (L1+L3B+L4B concurrent)
    - Broadcasts the result to all WebSocket /stream/ws subscribers
    - Returns the full scored result as JSON
    - X-Process-Time-Ms header contains server-side processing latency
    """
    t0 = time.perf_counter()
    event  = _ingestion.parse(payload)
    result = await orch.async_process(event)
    serial = _serialise(result)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)

    asyncio.create_task(_manager.broadcast(serial))
    return serial


@router.post("/batch")
async def score_batch(
    payloads: list[dict[str, Any]],
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> list[dict]:
    """Score a list of transactions in order and broadcast each one."""
    results = []
    for payload in payloads:
        event  = _ingestion.parse(payload)
        result = await orch.async_process(event)
        serial = _serialise(result)
        asyncio.create_task(_manager.broadcast(serial))
        results.append(serial)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Replay endpoints
# ─────────────────────────────────────────────────────────────────────────────

class ReplayRequest(BaseModel):
    file_path: str = "data/raw/all_transactions.parquet"
    rate_per_sec: float = 10.0
    max_rows: int = 1000


@router.post("/start")
async def start_replay(
    body: ReplayRequest,
    request: Request,
) -> dict:
    """
    Start replaying a CSV/Parquet file through the scoring pipeline.

    - Reads rows sequentially from file_path
    - Scores each transaction via async_process()
    - Broadcasts results to WebSocket subscribers
    - Limited to one active replay at a time
    """
    global _replay_task
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="Engines not loaded")

    if _replay_task and not _replay_task.done():
        raise HTTPException(status_code=409, detail="Replay already running. POST /stream/stop first.")

    src_path = Path(body.file_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {src_path}")

    async def _replay_loop() -> None:
        import pandas as pd
        delay = 1.0 / max(body.rate_per_sec, 0.1)
        try:
            if src_path.suffix == ".parquet":
                df = pd.read_parquet(src_path)
                rows = df.head(body.max_rows).to_dict(orient="records")
            else:
                rows = []
                with open(src_path, newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for i, row in enumerate(reader):
                        if i >= body.max_rows:
                            break
                        rows.append(row)

            scored = 0
            for row in rows:
                try:
                    event  = _ingestion.parse(row)
                    result = await orch.async_process(event)
                    serial = _serialise(result)
                    await _manager.broadcast(serial)
                    scored += 1
                except Exception as exc:
                    pass   # log quietly, keep replaying
                await asyncio.sleep(delay)

            await _manager.broadcast({
                "type": "replay_complete",
                "scored": scored,
                "file": str(src_path),
            })
        except asyncio.CancelledError:
            await _manager.broadcast({"type": "replay_stopped"})

    _replay_task = asyncio.create_task(_replay_loop())
    return {
        "status": "started",
        "file":   str(src_path),
        "rate_per_sec": body.rate_per_sec,
        "max_rows": body.max_rows,
    }


@router.post("/stop")
async def stop_replay() -> dict:
    """Cancel a running replay loop."""
    global _replay_task
    if _replay_task is None or _replay_task.done():
        return {"status": "no_replay_running"}
    _replay_task.cancel()
    return {"status": "stopped"}


@router.get("/connections")
async def connection_count() -> dict:
    """Return the number of active WebSocket subscribers."""
    return {
        "active_websocket_connections": _manager.connection_count,
        "replay_running": _replay_task is not None and not _replay_task.done(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket: subscribe-only broadcast receiver
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_broadcast(websocket: WebSocket) -> None:
    """
    Subscribe to the live scored-event stream.

    Clients connect and passively receive JSON objects whenever a transaction
    is submitted via POST /stream/transaction or POST /stream/batch.

    Ping/pong keepalive is handled automatically by the uvicorn WS layer.
    """
    await _manager.connect(websocket)
    try:
        # Keep the connection alive; we only receive pings / close frames
        while True:
            try:
                # Drain any incoming frames (pings, close signals)
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Optionally echo back client pings
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # No message for 30s — send a keepalive ping
                try:
                    await websocket.send_json({"type": "keepalive"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await _manager.disconnect(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket: interactive (bidirectional — send a transaction, receive scored)
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/interactive")
async def websocket_interactive(
    websocket: WebSocket,
    orch: TransactionOrchestrator = Depends(get_orchestrator),
) -> None:
    """
    Interactive WebSocket endpoint.

    Client sends JSON transaction objects; server responds immediately with
    the scored result AND broadcasts it to all /stream/ws subscribers.
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                event   = _ingestion.parse(payload)
                result  = await orch.async_process(event)
                serial  = _serialise(result)
                await websocket.send_json(serial)
                asyncio.create_task(_manager.broadcast(serial))
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _serialise(result: dict) -> dict:
    """Convert pipeline result dict to a JSON-serialisable dict."""
    alert = result.get("alert")
    fused = result["fused"]
    return {
        "transaction_id":         fused.transaction_id,
        "sender_account":         fused.sender_account,
        "transaction_risk_score": fused.transaction_risk_score,
        "risk_level":             fused.risk_level,
        "triggered_patterns":     fused.triggered_patterns,
        "explanation":            fused.explanation,
        "score_breakdown":        fused.score_breakdown,
        "alert_generated":        alert is not None,
        "rule_out": {
            "rule_score":          result["rule_out"].rule_score,
            "triggered_rules":     result["rule_out"].triggered_rules,
            "rule_explanations":   result["rule_out"].rule_explanations,
        },
        "behav_out": {
            "iso_forest_score":         result["behav_out"].iso_forest_score,
            "lof_score":                result["behav_out"].lof_score,
            "autoencoder_score":        result["behav_out"].autoencoder_score,
            "ensemble_anomaly_score":   result["behav_out"].ensemble_anomaly_score,
            "anomaly_drivers":          result["behav_out"].anomaly_drivers,
        },
        "gnn_out": {
            "link_anomaly_score":              result["gnn_out"].link_anomaly_score,
            "gnn_anomaly_score":               result["gnn_out"].gnn_anomaly_score,
            "structural_anomaly_explanations": result["gnn_out"].structural_anomaly_explanations,
            "community_anomaly_flag":          result["gnn_out"].community_anomaly_flag,
        },
        "alert": alert.model_dump() if alert else None,
    }
