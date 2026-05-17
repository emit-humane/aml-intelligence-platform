"""Diagnose HTTP latency — run the server with this one-off endpoint."""
import asyncio
import time
import sys

sys.path.insert(0, ".")

from system2_platform.api.dependencies import ARTIFACT_DIR, HISTORICAL_DATA, init_orchestrator
from system2_platform.shared.s4_stream_ingestion import StreamIngestion

orch = init_orchestrator(ARTIFACT_DIR, HISTORICAL_DATA if HISTORICAL_DATA.exists() else None)
ingestion = StreamIngestion()

PAYLOAD = {
    "transaction_id": "DBG_001",
    "sender_account": "ACC000001",
    "receiver_account": "ACC000999",
    "sender_name": "A", "receiver_name": "B",
    "sender_bank": "HDFC", "receiver_bank": "SBI",
    "sender_country": "India", "receiver_country": "UAE",
    "amount": 50000.0, "timestamp": "2024-03-15T15:30:00",
    "transaction_type": "Wire", "payment_channel": "Web",
    "device_id": "D1", "ip_address": "1.1.1.1",
    "geo_latitude": 28.6, "geo_longitude": 77.2,
    "merchant_category": "Financial", "transaction_status": "Success",
    "currency": "USD", "is_international": True,
    "sender_balance_before": 100000.0, "sender_balance_after": 50000.0,
    "receiver_balance_before": 5000.0, "receiver_balance_after": 55000.0,
    "kyc_level": 2, "remarks": "perf", "amount_leading_digit": 5,
}


async def timed_process():
    loop = asyncio.get_event_loop()
    event = ingestion.parse(PAYLOAD)

    checkpoints = {}

    t0 = time.perf_counter()
    fv  = await loop.run_in_executor(None, orch._feat.update_and_extract, event)
    checkpoints["S5"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    gfv = await loop.run_in_executor(None, orch._graph.update_and_extract, event)
    checkpoints["S6"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    rule_out, behav_out, gnn_raw = await asyncio.gather(
        loop.run_in_executor(None, orch._rules.evaluate, fv, gfv),
        loop.run_in_executor(None, orch._behav.score, fv, gfv),
        loop.run_in_executor(None, orch._gnn.score, event.sender_account, event.receiver_account),
    )
    checkpoints["L1+L3B+L4B"] = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    fused = orch._fusion.fuse(rule_out, behav_out, _convert_gnn(event, gnn_raw, gfv),
                               gfv=gfv, sender_account=event.sender_account)
    checkpoints["P1"] = (time.perf_counter() - t3) * 1000

    t4 = time.perf_counter()
    orch._alerts.maybe_create_alert(fused, rule_out, behav_out,
                                     _convert_gnn(event, gnn_raw, gfv), gfv)
    checkpoints["P2"] = (time.perf_counter() - t4) * 1000

    total = sum(checkpoints.values())
    print("\n=== Stage timings ===")
    for stage, ms in checkpoints.items():
        print(f"  {stage:<20} {ms:6.1f}ms")
    print(f"  {'TOTAL':<20} {total:6.1f}ms")


def _convert_gnn(event, raw, gfv):
    from system2_platform.contracts.gnn_inference_output import GNNInferenceOutput
    return GNNInferenceOutput(
        transaction_id=event.transaction_id,
        sender_embedding=raw.embedding_src.tolist(),
        receiver_embedding=raw.embedding_dst.tolist(),
        sender_embedding_drift=0.0,
        receiver_embedding_drift=0.0,
        link_anomaly_score=float(1.0 - raw.link_score),
        gnn_anomaly_score=raw.gnn_anomaly_score,
        structural_anomaly_explanations=[],
        community_anomaly_flag=False,
    )


for i in range(3):
    print(f"\n--- Run {i+1} ---")
    asyncio.run(timed_process())
