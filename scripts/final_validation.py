"""
Day 3 Morning final validation.
Replays 100 transactions and asserts all routes work.
"""
import sys
import time
import requests

BASE = "http://localhost:8000"
s = requests.Session()

def check(label, r, expected_status=200):
    ok = r.status_code == expected_status
    sym = "OK" if ok else "FAIL"
    print(f"  [{sym}] {label}: {r.status_code}")
    if not ok:
        print(f"       {r.text[:200]}")
    return ok

# --- Health ---
print("\n[1] Health endpoints")
check("GET /health", s.get(f"{BASE}/health"))
check("GET /health/ready", s.get(f"{BASE}/health/ready"))

# --- Score one transaction ---
print("\n[2] Single transaction scoring")
PAYLOAD = {
    "transaction_id": "FINAL_001",
    "sender_account": "ACC000001",
    "receiver_account": "ACC000999",
    "sender_name": "A", "receiver_name": "B",
    "sender_bank": "HDFC", "receiver_bank": "SBI",
    "sender_country": "India", "receiver_country": "UAE",
    "amount": 95000.0, "timestamp": "2024-03-15T14:22:00",
    "transaction_type": "Wire", "payment_channel": "Web",
    "device_id": "DEV001", "ip_address": "192.168.1.1",
    "geo_latitude": 28.6139, "geo_longitude": 77.2090,
    "merchant_category": "Financial", "transaction_status": "Success",
    "currency": "USD", "is_international": True,
    "sender_balance_before": 200000.0, "sender_balance_after": 105000.0,
    "receiver_balance_before": 10000.0, "receiver_balance_after": 105000.0,
    "kyc_level": 2, "remarks": "Test", "amount_leading_digit": 9,
}
r = s.post(f"{BASE}/stream/transaction", json=PAYLOAD)
check("POST /stream/transaction", r)
if r.status_code == 200:
    d = r.json()
    server_ms = r.headers.get("X-Process-Time-Ms", "?")
    print(f"       Score={d['transaction_risk_score']}  Level={d['risk_level']}  "
          f"Alert={d['alert_generated']}  Server={server_ms}ms")
    print(f"       Weights={d['score_breakdown']['weights']}")

# --- Replay 100 transactions ---
print("\n[3] Replay 100 transactions")
r = s.post(f"{BASE}/stream/start", json={
    "file_path": "data/raw/all_transactions.parquet",
    "rate_per_sec": 25.0,
    "max_rows": 100,
})
check("POST /stream/start", r)
time.sleep(8)  # wait ~100/25 = 4s + buffer

# --- Alerts ---
print("\n[4] Alert endpoints")
r_alerts = s.get(f"{BASE}/alerts/?limit=200")
check("GET /alerts/", r_alerts)
alerts = r_alerts.json()
print(f"       Total alerts: {len(alerts)}")

r_stats = s.get(f"{BASE}/alerts/stats")
check("GET /alerts/stats", r_stats)
print(f"       Stats: {r_stats.json()}")

if alerts:
    aid = alerts[0]["alert_id"]
    check(f"GET /alerts/<id>", s.get(f"{BASE}/alerts/{aid}"))
    check(f"PATCH /alerts/<id>/status", s.patch(f"{BASE}/alerts/{aid}/status",
          json={"status": "Investigating", "assigned_to": "validator"}))

# --- Graph ---
print("\n[5] Graph endpoints")
check("GET /graph/stats", s.get(f"{BASE}/graph/stats"))

if alerts:
    acct = alerts[0]["sender_account"]
    r_sg = s.get(f"{BASE}/graph/subgraph/{acct}")
    check(f"GET /graph/subgraph/{acct}", r_sg)
    if r_sg.status_code == 200:
        sg = r_sg.json()
        print(f"       nodes={len(sg['nodes'])} edges={len(sg['edges'])}")

# --- Analytics ---
print("\n[6] Analytics endpoints")
r_m = s.get(f"{BASE}/analytics/metrics")
check("GET /analytics/metrics", r_m)
if r_m.status_code == 200:
    m = r_m.json()
    print(f"       live_nodes={m['graph']['live_nodes']} live_edges={m['graph']['live_edges']}")
    print(f"       communities={m['offline']['n_communities']}  "
          f"behavioral={m['offline']['n_behavioral_profiles']}")

check("GET /analytics/communities", s.get(f"{BASE}/analytics/communities?limit=5"))
check("GET /analytics/suspicious-paths", s.get(f"{BASE}/analytics/suspicious-paths?limit=5"))

# --- SQLite / CSV check ---
print("\n[7] Persistence check")
import sqlite3, csv
from pathlib import Path

conn = sqlite3.connect("data/aml.db")
count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
conn.close()
sym = "OK" if count > 0 else "FAIL"
print(f"  [{sym}] SQLite alert count: {count}")

csv_rows = sum(1 for _ in open("data/generated_alerts.csv", encoding="utf-8")) - 1  # minus header
sym = "OK" if csv_rows > 0 else "FAIL"
print(f"  [{sym}] CSV alert count: {csv_rows}")

print("\n=== Validation complete ===")
