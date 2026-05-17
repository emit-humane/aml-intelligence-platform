"""Quick profile of async_process stage timings."""
import asyncio
import time
import sys

sys.path.insert(0, ".")
from pathlib import Path
from system2_platform.api.dependencies import ARTIFACT_DIR, HISTORICAL_DATA, init_orchestrator
from system2_platform.shared.s4_stream_ingestion import StreamIngestion

orch = init_orchestrator(ARTIFACT_DIR, HISTORICAL_DATA if HISTORICAL_DATA.exists() else None)
ingestion = StreamIngestion()

payload = {
    "transaction_id": "PROF001",
    "sender_account": "ACC000001",
    "receiver_account": "ACC000999",
    "sender_name": "A", "receiver_name": "B",
    "sender_bank": "HDFC", "receiver_bank": "SBI",
    "sender_country": "India", "receiver_country": "UAE",
    "amount": 95000.0, "timestamp": "2024-03-15T14:22:00",
    "transaction_type": "Wire", "payment_channel": "Web",
    "device_id": "DEV001", "ip_address": "1.1.1.1",
    "geo_latitude": 28.6, "geo_longitude": 77.2,
    "merchant_category": "Financial", "transaction_status": "Success",
    "currency": "USD", "is_international": True,
    "sender_balance_before": 200000.0, "sender_balance_after": 105000.0,
    "receiver_balance_before": 10000.0, "receiver_balance_after": 105000.0,
    "kyc_level": 2, "remarks": "test", "amount_leading_digit": 9,
}

event = ingestion.parse(payload)


async def profile():
    loop = asyncio.get_event_loop()

    t0 = time.perf_counter()
    fv = await loop.run_in_executor(None, orch._feat.update_and_extract, event)
    print(f"S5 FeatureUpdater:       {(time.perf_counter()-t0)*1000:.0f}ms")

    t1 = time.perf_counter()
    gfv = await loop.run_in_executor(None, orch._graph.update_and_extract, event)
    print(f"S6 GraphUpdater:         {(time.perf_counter()-t1)*1000:.0f}ms")

    t2 = time.perf_counter()
    r1, r2, r3 = await asyncio.gather(
        loop.run_in_executor(None, orch._rules.evaluate, fv, gfv),
        loop.run_in_executor(None, orch._behav.score, fv, gfv),
        loop.run_in_executor(None, orch._gnn.score, event.sender_account, event.receiver_account),
    )
    print(f"L1+L3B+L4B (concurrent): {(time.perf_counter()-t2)*1000:.0f}ms")

    # Full end-to-end
    t3 = time.perf_counter()
    for _ in range(3):
        await orch.async_process(event)
    avg = (time.perf_counter() - t3) * 1000 / 3
    print(f"async_process avg (3x):  {avg:.0f}ms")


asyncio.run(profile())
