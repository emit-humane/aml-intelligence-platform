"""Check X-Process-Time-Ms header to isolate server vs client latency."""
import time
import requests

URL = "http://localhost:8000/stream/transaction"

PAYLOAD = {
    "transaction_id": "TIMING_0",
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

# Warm-up
requests.post(URL, json={**PAYLOAD, "transaction_id": "WARMUP"})

print(f"{'Call':<6} {'Total ms':>10} {'Server ms':>10}")
print("-" * 30)
for i in range(5):
    p = {**PAYLOAD, "transaction_id": f"TIMING_{i}"}
    t0 = time.perf_counter()
    r = requests.post(URL, json=p)
    total_ms = (time.perf_counter() - t0) * 1000
    server_ms = r.headers.get("X-Process-Time-Ms", "N/A")
    print(f"{i+1:<6} {total_ms:>10.0f} {server_ms:>10}")
