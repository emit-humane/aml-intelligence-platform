"""
S4 — Stream Ingestion.

Converts raw transaction dicts / JSON payloads into TransactionEvent objects,
validates field types, and normalises timestamps to UTC-aware datetimes.

Used by the API websocket handler and the batch replay script.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..contracts.transaction_event import TransactionEvent


class StreamIngestion:
    """
    Validates and normalises incoming transaction payloads.

    Usage
    -----
    ingestion = StreamIngestion()
    event = ingestion.parse(raw_dict)
    """

    def parse(self, payload: dict[str, Any]) -> TransactionEvent:
        """
        Parse a raw dict (e.g. from WebSocket JSON) into a TransactionEvent.

        Handles:
          - ISO-8601 timestamp strings (with or without 'Z')
          - amount as str or float
          - is_international as bool or int
          - amount_leading_digit computed if missing
        """
        payload = dict(payload)   # shallow copy

        # Normalise timestamp
        ts = payload.get("timestamp")
        if isinstance(ts, str):
            ts = ts.replace("Z", "+00:00")
            ts_dt = datetime.fromisoformat(ts)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            payload["timestamp"] = ts_dt
        elif isinstance(ts, (int, float)):
            payload["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc)

        # Normalise numeric fields
        payload["amount"] = float(payload.get("amount", 0.0))

        # Derive amount_leading_digit if not present
        if "amount_leading_digit" not in payload:
            payload["amount_leading_digit"] = _leading_digit(payload["amount"])

        # Normalise bool fields
        for bool_field in ("is_international", "round_amount_flag"):
            if bool_field in payload:
                payload[bool_field] = bool(payload[bool_field])

        # Normalise string fields — replace NaN/None with empty string
        for str_field in ("remarks", "device_id", "ip_address",
                          "merchant_category", "sender_name", "receiver_name"):
            v = payload.get(str_field)
            if v is None or (isinstance(v, float) and v != v):  # None or NaN
                payload[str_field] = ""

        # Normalise transaction_type to the valid Literal enum
        _TX_TYPE_MAP = {
            "TRANSFER":       "NEFT",
            "WIRE":           "Wire",
            "SWIFT":          "Wire",
            "SWIFT_TRANSFER": "Wire",
            "UPI":            "UPI",
            "IMPS":           "IMPS",
            "NEFT":           "NEFT",
            "RTGS":           "RTGS",
            "CARD":           "Card",
        }
        tx_type = str(payload.get("transaction_type", "NEFT")).upper().replace("-", "_")
        payload["transaction_type"] = _TX_TYPE_MAP.get(tx_type, "NEFT")

        # Normalise payment_channel to the valid Literal enum
        _CHANNEL_MAP = {
            "NEFT":    "Web",
            "RTGS":    "Web",
            "IMPS":    "Mobile",
            "UPI":     "Mobile",
            "WIRE":    "Web",
            "SWIFT":   "Web",
            "ATM":     "ATM",
            "BRANCH":  "Branch",
            "MOBILE":  "Mobile",
            "WEB":     "Web",
        }
        channel = str(payload.get("payment_channel", "Web")).upper()
        payload["payment_channel"] = _CHANNEL_MAP.get(channel, "Web")

        # Normalise transaction_status
        _STATUS_MAP = {
            "SUCCESS":  "Success",
            "FAILED":   "Failed",
            "PENDING":  "Pending",
            "FAILURE":  "Failed",
        }
        status = str(payload.get("transaction_status", "Success")).upper()
        payload["transaction_status"] = _STATUS_MAP.get(status, "Success")

        return TransactionEvent.model_validate(payload)

    def parse_json(self, raw: str | bytes) -> TransactionEvent:
        """Parse a raw JSON string."""
        return self.parse(json.loads(raw))

    def parse_batch(self, payloads: list[dict[str, Any]]) -> list[TransactionEvent]:
        """Parse a list of raw dicts."""
        return [self.parse(p) for p in payloads]


def _leading_digit(amount: float) -> int:
    """Fast integer math for Benford leading digit."""
    if amount <= 0:
        return 1
    x = amount if amount >= 1.0 else amount * 1_000_000
    d = int(x)
    if d == 0:
        return 1
    while d >= 10:
        d //= 10
    return max(1, d)
