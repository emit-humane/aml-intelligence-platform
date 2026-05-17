"""Compare latency of GET /health vs POST /stream/transaction."""
import time
import requests

BASE = "http://localhost:8000"
s = requests.Session()  # reuse connection

# Health GET (minimal)
print("GET /health (5 calls):")
for i in range(5):
    t0 = time.perf_counter()
    r = s.get(f"{BASE}/health")
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {ms:.1f}ms")

# Small POST to check JSON parsing
print("\nPOST /stream/transaction (5 calls, with session keep-alive):")
PAYLOAD = {
    "transaction_id": "T",
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
for i in range(5):
    p = {**PAYLOAD, "transaction_id": f"T{i}"}
    t0 = time.perf_counter()
    r = s.post(f"{BASE}/stream/transaction", json=p)
    ms = (time.perf_counter() - t0) * 1000
    server_ms = r.headers.get("X-Process-Time-Ms", "?")
    print(f"  total={ms:.1f}ms  server={server_ms}ms  status={r.status_code}")
