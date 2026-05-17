"""
S1 — Feature Engineering.

FeatureStore maintains per-account rolling windows and computes the 45-feature
LiveFeatureVector for each transaction. Designed to work in both:
  - Offline batch mode: fit(df) then transform(df)
  - Online streaming mode: update(tx) after compute(tx)

Feature dimensions (FEATURE_DIM = 45):
  [0-10]  Temporal (11 features)
  [11-19] Behavioral (9 features)
  [20-24] Geographic (5 features)
  [25-28] Device (4 features)
  [29-44] Derived / encoded (16 features)

All features are StandardScaler-normalised in the scaled_feature_vector.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..contracts.live_feature_vector import LiveFeatureVector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_DIM = 45

_HIGH_RISK_COUNTRIES = {"AE", "MU", "CN", "NG", "PK", "CH"}

_TX_TYPES = ["NEFT", "RTGS", "IMPS", "Wire", "UPI", "Card",
             "TRANSFER", "WIRE", "PAYMENT"]
_CHANNELS = ["Mobile", "Web", "ATM", "Branch", "SWIFT"]

_NIGHT_HOURS = set(range(0, 6)) | set(range(22, 24))  # 22:00–05:59
_WEEKEND_DAYS = {5, 6}  # Saturday, Sunday

# Benford expected probabilities for digits 1–9
_BENFORD_P = np.log10(1.0 + 1.0 / np.arange(1, 10))
_BENFORD_P /= _BENFORD_P.sum()

_STRUCTURING_THRESHOLD = 1_000_000.0
_IMPOSSIBLE_SPEED_KMH = 900.0  # faster than subsonic flight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(max(0.0, a)))


def _leading_digit(amount: float) -> int:
    """Fast integer-math leading digit extraction — avoids string formatting."""
    if amount <= 0:
        return 1
    # Scale to an integer >= 1 by multiplying up
    x = amount if amount >= 1.0 else amount * 1_000_000
    d = int(x)
    if d == 0:
        return 1
    while d >= 10:
        d //= 10
    return d


def _benford_chi2(amounts: list[float]) -> float:
    if len(amounts) < 5:
        return 0.0
    observed = np.zeros(9)
    for a in amounts:
        d = _leading_digit(a) - 1
        observed[d] += 1
    n = observed.sum()
    if n == 0:
        return 0.0
    expected = _BENFORD_P * n
    chi2 = float(np.sum((observed - expected) ** 2 / (expected + 1e-9)))
    return chi2


def _shannon_entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total + 1e-12) for c in counts.values() if c > 0)


def _encode_tx_type(tx_type: str) -> int:
    _map = {"NEFT": 0, "RTGS": 1, "IMPS": 2, "Wire": 3, "WIRE": 3,
            "UPI": 4, "Card": 5, "TRANSFER": 6, "PAYMENT": 7}
    return _map.get(tx_type, 0)


def _encode_channel(channel: str) -> int:
    _map = {"Mobile": 0, "Web": 1, "ATM": 2, "Branch": 3, "SWIFT": 4,
            "NEFT": 5, "RTGS": 6, "IMPS": 7, "UPI": 8}
    return _map.get(channel, 0)


# ---------------------------------------------------------------------------
# Per-account rolling state
# ---------------------------------------------------------------------------

class _AccountState:
    """Maintains rolling windows for a single account."""

    __slots__ = (
        "txns",             # deque[(ts, amount, receiver, device, ip, country, lat, lon, status)]
        "device_seen",      # set of all device_ids ever seen
        "ip_seen",          # set of all ips ever seen
        "all_amounts",      # deque for Benford chi2 (last 200)
        "last_tx_ts_ever",  # most recent tx timestamp, NEVER trimmed (needed for R04 dormancy)
    )

    def __init__(self) -> None:
        self.txns: deque[tuple] = deque()
        self.device_seen: set[str] = set()
        self.ip_seen: set[str] = set()
        self.all_amounts: deque[float] = deque(maxlen=200)
        self.last_tx_ts_ever: datetime | None = None

    def add(self, ts: datetime, amount: float, receiver: str, device: str,
            ip: str, country: str, lat: float, lon: float, status: str) -> None:
        self.txns.append((ts, amount, receiver, device, ip, country, lat, lon, status))
        self.device_seen.add(device)
        self.ip_seen.add(ip)
        self.all_amounts.append(amount)
        # Always track the most-recent timestamp so dormancy gap survives the 35-day trim
        self.last_tx_ts_ever = ts

    def compute_windows(self, now: datetime) -> dict:
        """
        Single-pass over txns deque; returns pre-built window lists.
        Keys: 'w1h', 'w6h', 'w24h', 'w7d', 'w30d'
        Each entry: (ts, amount, receiver, device, ip, country, lat, lon, status)
        """
        now_ts = now.timestamp()
        c1h  = now_ts - 3_600
        c6h  = now_ts - 21_600
        c24h = now_ts - 86_400
        c7d  = now_ts - 604_800
        c30d = now_ts - 2_592_000

        w1h: list[tuple] = []
        w6h: list[tuple] = []
        w24h: list[tuple] = []
        w7d: list[tuple] = []
        w30d: list[tuple] = []

        for entry in self.txns:           # oldest first, all within 35-day trim
            ets = entry[0].timestamp()
            if ets < c30d:
                continue
            w30d.append(entry)
            if ets >= c7d:
                w7d.append(entry)
                if ets >= c24h:
                    w24h.append(entry)
                    if ets >= c6h:
                        w6h.append(entry)
                        if ets >= c1h:
                            w1h.append(entry)

        return {"w1h": w1h, "w6h": w6h, "w24h": w24h, "w7d": w7d, "w30d": w30d}

    def last_location(self) -> tuple[float, float] | None:
        if not self.txns:
            return None
        t = self.txns[-1]
        return t[6], t[7]

    def last_tx_ts(self) -> datetime | None:
        if not self.txns:
            return None
        return self.txns[-1][0]

    def shared_device_count(self, device: str) -> int:
        return len(self.device_seen)

    def last_location(self) -> tuple[float, float] | None:
        if not self.txns:
            return None
        t = self.txns[-1]
        return t[6], t[7]

    def last_tx_ts(self) -> datetime | None:
        if not self.txns:
            return None
        return self.txns[-1][0]

    def shared_device_count(self, device: str) -> int:
        """How many distinct devices has this account used historically?"""
        return len(self.device_seen)

    def trim(self, now: datetime, keep_days: int = 35) -> None:
        cutoff = now - timedelta(days=keep_days)
        while self.txns and self.txns[0][0] < cutoff:
            self.txns.popleft()


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Stateful feature computer.  Maintains per-account rolling history.

    Usage (batch):
        fs = FeatureStore()
        fs.fit(historical_df)
        feature_df = fs.transform(historical_df)

    Usage (stream):
        fs = FeatureStore()
        fs.fit(historical_df)
        for tx in stream:
            fv = fs.compute(tx)
            fs.update(tx)
    """

    def __init__(
        self,
        structuring_threshold: float = _STRUCTURING_THRESHOLD,
        high_risk_countries: set[str] | None = None,
    ) -> None:
        self._threshold = structuring_threshold
        self._hr_countries = high_risk_countries or _HIGH_RISK_COUNTRIES
        self._states: dict[str, _AccountState] = defaultdict(_AccountState)
        self._scaler: StandardScaler | None = None
        self._device_to_accounts: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureStore":
        """Replay historical transactions to warm up per-account state."""
        df_sorted = df.sort_values("timestamp")
        for row in df_sorted.itertuples(index=False):
            self._ingest_row(row)
        return self

    def transform(self, df: pd.DataFrame) -> list[LiveFeatureVector]:
        """
        Compute features for every row in df (sorted by timestamp).
        State is rebuilt from scratch so this is deterministic.
        """
        self._states.clear()
        self._device_to_accounts.clear()
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        vectors: list[LiveFeatureVector] = []
        raw_matrix: list[list[float]] = []

        for row in df_sorted.itertuples(index=False):
            raw = self._compute_raw(row)
            raw_matrix.append(raw)
            self._ingest_row(row)

        # Fit scaler on this batch and scale
        X = np.array(raw_matrix, dtype=float)
        if self._scaler is None:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = self._scaler.transform(X)

        for i, row in enumerate(df_sorted.itertuples(index=False)):
            fv = self._row_to_fv(row, raw_matrix[i], X_scaled[i].tolist())
            vectors.append(fv)

        return vectors

    def compute(self, tx: dict | Any) -> LiveFeatureVector:
        """Compute features for a single live transaction."""
        if isinstance(tx, dict):
            row = _DictRow(tx)
        else:
            row = tx
        raw = self._compute_raw(row)
        scaled = self._scale_one(raw)
        return self._row_to_fv(row, raw, scaled)

    def update(self, tx: dict | Any) -> None:
        """Add a transaction to the rolling state (call after compute)."""
        if isinstance(tx, dict):
            row = _DictRow(tx)
        else:
            row = tx
        self._ingest_row(row)

    def fit_scaler(self, feature_matrix: np.ndarray) -> None:
        self._scaler = StandardScaler()
        self._scaler.fit(feature_matrix)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ingest_row(self, row: Any) -> None:
        ts = _to_dt(getattr(row, "timestamp", None))
        sender = str(getattr(row, "sender_account", ""))
        state = self._states[sender]
        state.add(
            ts=ts,
            amount=float(getattr(row, "amount", 0.0)),
            receiver=str(getattr(row, "receiver_account", "")),
            device=str(getattr(row, "device_id", "")),
            ip=str(getattr(row, "ip_address", "")),
            country=str(getattr(row, "receiver_country", "")),
            lat=float(getattr(row, "geo_latitude", 0.0)),
            lon=float(getattr(row, "geo_longitude", 0.0)),
            status=str(getattr(row, "transaction_status", "Success")),
        )
        state.trim(ts)
        device = str(getattr(row, "device_id", ""))
        # Only track non-empty device IDs — empty string is a data-quality
        # placeholder and must not be treated as a shared device signal.
        if device:
            self._device_to_accounts[device].add(sender)

    def _compute_raw(self, row: Any) -> list[float]:
        ts = _to_dt(getattr(row, "timestamp", None))
        sender = str(getattr(row, "sender_account", ""))
        amount = float(getattr(row, "amount", 0.0))
        device = str(getattr(row, "device_id", ""))
        ip = str(getattr(row, "ip_address", ""))
        sender_country = str(getattr(row, "sender_country", ""))
        recv_country = str(getattr(row, "receiver_country", ""))
        lat = float(getattr(row, "geo_latitude", 0.0))
        lon = float(getattr(row, "geo_longitude", 0.0))
        kyc = int(getattr(row, "kyc_level", 0))
        tx_type = str(getattr(row, "transaction_type", "NEFT"))
        channel = str(getattr(row, "payment_channel", "Mobile"))
        sender_bank = str(getattr(row, "sender_bank", ""))
        recv_bank = str(getattr(row, "receiver_bank", ""))
        bal_before = float(getattr(row, "sender_balance_before", amount + 1.0))
        is_intl = bool(getattr(row, "is_international", sender_country != recv_country))

        st = self._states[sender]

        # Single-pass window computation (1 deque scan for all windows)
        win = st.compute_windows(ts)
        w1h, w6h, w24h, w7d, w30d = win["w1h"], win["w6h"], win["w24h"], win["w7d"], win["w30d"]

        # ---- Temporal [0–10] ----
        v1h  = float(len(w1h))
        v6h  = float(len(w6h))
        v24h = float(len(w24h))
        v7d  = float(len(w7d))

        amts_7d = [t[1] for t in w7d]
        avg_7d = float(np.mean(amts_7d)) if amts_7d else 0.0
        std_7d = float(np.std(amts_7d)) if len(amts_7d) > 1 else 0.0

        # Gap: time elapsed since the account's most recent previous transaction.
        # Use last_tx_ts_ever (never trimmed) so R04 dormancy detection works even when
        # the prior transaction is older than the 24h/35d rolling windows.
        if st.last_tx_ts_ever is not None:
            gap_sec = max(0.0, (ts - st.last_tx_ts_ever).total_seconds())
        else:
            gap_sec = 86_400.0  # default: 1 day for brand-new accounts

        night_r = (sum(1 for t in w7d if t[0].hour in _NIGHT_HOURS) / len(w7d)) if w7d else 0.0
        wknd_r  = (sum(1 for t in w7d if t[0].weekday() in _WEEKEND_DAYS) / len(w7d)) if w7d else 0.0
        hour = float(ts.hour)
        dow  = float(ts.weekday())

        # ---- Behavioral [11–19] ----
        ben_7d_set  = {t[2] for t in w7d}
        ben_30d_set = {t[2] for t in w30d}
        ben_7d  = float(len(ben_7d_set))
        ben_30d = float(len(ben_30d_set))

        recv_counts: dict[str, int] = defaultdict(int)
        for t in w7d:
            recv_counts[t[2]] += 1
        recv_ent = _shannon_entropy(recv_counts)

        z_score = (amount - avg_7d) / (std_7d + 1e-6) if std_7d > 0 else 0.0

        amts_30d = [t[1] for t in w30d]
        avg_daily_vol_30d = sum(amts_30d) / 30.0 if amts_30d else 0.0

        round_flag   = 1.0 if (amount % 100_000 == 0 and amount >= 100_000) else 0.0
        sub_thr_flag = 1.0 if (0.85 * self._threshold <= amount < self._threshold) else 0.0
        ld = float(_leading_digit(amount))
        benford_chi2 = _benford_chi2(list(st.all_amounts))

        # ---- Geographic [20–24] ----
        last_loc = st.last_location()
        last_ts_val = st.last_tx_ts()
        geo_dist = _haversine_km(last_loc[0], last_loc[1], lat, lon) if last_loc else 0.0

        imp_travel = 0.0
        if last_loc and last_ts_val:
            elapsed_h = (ts - last_ts_val).total_seconds() / 3600.0
            if elapsed_h > 0 and (geo_dist / elapsed_h) > _IMPOSSIBLE_SPEED_KMH:
                imp_travel = 1.0

        ctrs_7d = [t[5] for t in w7d]
        ctr_sw_7d = float(sum(1 for i in range(1, len(ctrs_7d)) if ctrs_7d[i] != ctrs_7d[i - 1]))
        hr_flag = 1.0 if recv_country in self._hr_countries or sender_country in self._hr_countries else 0.0
        intl_30d = sum(1 for t in w30d if t[5] != sender_country)
        xb_ratio = intl_30d / len(w30d) if w30d else 0.0

        # ---- Device [25–28] ----
        devs_7d = {t[3] for t in w7d}
        dev_change = float(len(devs_7d))
        new_dev    = 0.0 if device in st.device_seen else 1.0
        shared_dev = float(len(self._device_to_accounts.get(device, set()))) if device else 0.0
        ip_chg     = float(len({t[4] for t in w24h}))

        # ---- Derived / encoded [29–44] ----
        kyc_f       = float(kyc)
        amt_ratio_7d = (amount / avg_7d) if avg_7d > 0 else 1.0
        bal_util     = (amount / bal_before) if bal_before > 0 else 1.0
        same_bank    = 1.0 if sender_bank == recv_bank else 0.0
        is_intl_f    = 1.0 if is_intl else 0.0
        tx_type_enc  = float(_encode_tx_type(tx_type))
        channel_enc  = float(_encode_channel(channel))
        hour_sin = math.sin(2 * math.pi * ts.hour / 24.0)
        hour_cos = math.cos(2 * math.pi * ts.hour / 24.0)
        dow_sin  = math.sin(2 * math.pi * ts.weekday() / 7.0)
        dow_cos  = math.cos(2 * math.pi * ts.weekday() / 7.0)
        dom = float(ts.day)
        ok_7d = sum(1 for t in w7d if t[8] == "Success")
        success_rate = ok_7d / len(w7d) if w7d else 1.0
        max_7d = float(max(amts_7d)) if amts_7d else amount
        min_7d = float(min(amts_7d)) if amts_7d else amount
        amt_to_max = (amount / max_7d) if max_7d > 0 else 1.0

        return [
            v1h, v6h, v24h, v7d, avg_7d, std_7d, gap_sec, night_r, wknd_r, hour, dow,
            ben_7d, ben_30d, recv_ent, z_score, avg_daily_vol_30d,
            round_flag, sub_thr_flag, ld, benford_chi2,
            geo_dist, ctr_sw_7d, imp_travel, hr_flag, xb_ratio,
            dev_change, new_dev, shared_dev, ip_chg,
            kyc_f, amt_ratio_7d, bal_util, same_bank, is_intl_f,
            tx_type_enc, channel_enc,
            hour_sin, hour_cos, dow_sin, dow_cos,
            dom, success_rate, max_7d, min_7d, amt_to_max,
        ]  # FEATURE_DIM == 45

    def _scale_one(self, raw: list[float]) -> list[float]:
        if self._scaler is None:
            return raw
        x = np.array(raw, dtype=float).reshape(1, -1)
        return self._scaler.transform(x)[0].tolist()

    def _row_to_fv(self, row: Any, raw: list[float], scaled: list[float]) -> LiveFeatureVector:
        r = raw
        return LiveFeatureVector(
            transaction_id=str(getattr(row, "transaction_id", "")),
            sender_account=str(getattr(row, "sender_account", "")),
            # Temporal
            tx_velocity_1h=r[0],
            tx_velocity_6h=r[1],
            tx_velocity_24h=r[2],
            tx_velocity_7d=r[3],
            avg_amount_7d=r[4],
            std_amount_7d=r[5],
            tx_gap_seconds=r[6],
            night_tx_ratio=r[7],
            weekend_tx_ratio=r[8],
            hour_of_day=int(r[9]),
            day_of_week=int(r[10]),
            # Behavioral
            beneficiary_count_7d=int(r[11]),
            beneficiary_count_30d=int(r[12]),
            receiver_entropy=r[13],
            amount_zscore=r[14],
            avg_daily_volume_30d=r[15],
            round_amount_flag=bool(r[16]),
            sub_threshold_flag=bool(r[17]),
            amount_leading_digit=int(r[18]),
            benford_chi2_score=r[19],
            # Geographic
            geo_distance_km=r[20],
            country_switch_count_7d=int(r[21]),
            impossible_travel_flag=bool(r[22]),
            high_risk_country_flag=bool(r[23]),
            cross_border_ratio_30d=r[24],
            # Device
            device_change_count_7d=int(r[25]),
            new_device_flag=bool(r[26]),
            shared_device_count=int(r[27]),
            ip_change_count_24h=int(r[28]),
            # Transaction-level fields for rule engine
            amount=float(getattr(row, "amount", 0.0)),
            is_international=bool(getattr(row, "is_international", False)),
            kyc_level=int(getattr(row, "kyc_level", 0)),
            # Scaled vector (all 45 normalised)
            scaled_feature_vector=scaled,
        )


# ---------------------------------------------------------------------------
# Convenience row adaptor for dict inputs
# ---------------------------------------------------------------------------

class _DictRow:
    """Wraps a dict so attribute access works."""
    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)


def _to_dt(val: Any) -> datetime:
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val, tz=timezone.utc)
    if isinstance(val, str):
        return datetime.fromisoformat(val).replace(tzinfo=timezone.utc)
    if isinstance(val, pd.Timestamp):
        dt = val.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return datetime.now(tz=timezone.utc)
