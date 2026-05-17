"""
Reusable sampling primitives.  All functions accept a seeded numpy Generator
so the caller controls determinism — no module-level random state.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
from numpy.random import Generator

# ---------------------------------------------------------------------------
# Benford's law
# ---------------------------------------------------------------------------

# P(leading digit = d) = log10(1 + 1/d)  for d in 1..9
_BENFORD_DIGITS = np.arange(1, 10)
_BENFORD_PROBS = np.log10(1.0 + 1.0 / _BENFORD_DIGITS)
_BENFORD_PROBS /= _BENFORD_PROBS.sum()  # normalise (already ~1 but float safety)


def sample_benford_digit(rng: Generator) -> int:
    return int(rng.choice(_BENFORD_DIGITS, p=_BENFORD_PROBS))


def sample_benford_digits(rng: Generator, n: int) -> np.ndarray:
    return rng.choice(_BENFORD_DIGITS, size=n, p=_BENFORD_PROBS)


def leading_digit(amount: float) -> int:
    """Extract first significant digit (1–9) from a positive float."""
    if amount <= 0:
        return 1
    return int(str(amount).lstrip("0").replace(".", "")[0])


# ---------------------------------------------------------------------------
# Amount sampling (lognormal with Benford-consistent leading digit)
# ---------------------------------------------------------------------------

def _lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """Convert arithmetic mean/std to lognormal mu/sigma."""
    cv2 = (std / mean) ** 2
    sigma = math.sqrt(math.log(1.0 + cv2))
    mu = math.log(mean) - 0.5 * sigma**2
    return mu, sigma


def sample_amounts(
    rng: Generator,
    n: int,
    mean: float,
    std: float,
    min_amount: float,
    max_amount: float,
) -> np.ndarray:
    """
    Draw n amounts from a lognormal distribution, clipped to [min_amount, max_amount].
    """
    mu, sigma = _lognormal_params(mean, std)
    raw = rng.lognormal(mu, sigma, size=n)
    return np.clip(raw, min_amount, max_amount)


def sample_structuring_amounts(
    rng: Generator,
    n: int,
    threshold: float = 1_000_000.0,
) -> np.ndarray:
    """
    Sub-threshold amounts in [0.85*threshold, 0.999*threshold].
    Used by structuring / velocity-burst typologies.
    """
    lo, hi = 0.85 * threshold, 0.999 * threshold
    return rng.uniform(lo, hi, size=n)


# ---------------------------------------------------------------------------
# Intraday timestamp distribution
# ---------------------------------------------------------------------------

# Hourly weights — two peaks: 9-12 and 14-17, low 0-6
_HOUR_WEIGHTS = np.array([
    0.5, 0.3, 0.2, 0.2, 0.3, 0.5,   # 0–5
    1.0, 2.0, 3.5, 4.5, 5.0, 4.8,   # 6–11
    4.0, 3.5, 4.5, 5.0, 4.8, 4.0,   # 12–17
    3.0, 2.5, 2.0, 1.5, 1.0, 0.7,   # 18–23
], dtype=float)
_HOUR_WEIGHTS /= _HOUR_WEIGHTS.sum()

_HOURS = np.arange(24)


def sample_intraday_seconds(rng: Generator, n: int) -> np.ndarray:
    """
    Return n offsets (in seconds from midnight, 0..86399) drawn from the
    intraday distribution.
    """
    hours = rng.choice(_HOURS, size=n, p=_HOUR_WEIGHTS)
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)
    return (hours * 3600 + minutes * 60 + seconds).astype(np.int32)


def make_timestamps(
    rng: Generator,
    start: datetime,
    total_days: int,
    n: int,
) -> list[datetime]:
    """
    Generate n timestamps uniformly spread over [start, start+total_days),
    with an intraday bias towards business hours.
    """
    day_offsets = rng.integers(0, total_days, size=n)
    intraday = sample_intraday_seconds(rng, n)
    base = start + timedelta(days=0)  # explicit copy
    return [
        base + timedelta(days=int(d), seconds=int(s))
        for d, s in zip(day_offsets, intraday)
    ]


# ---------------------------------------------------------------------------
# Country → representative (lat, lon) with jitter
# ---------------------------------------------------------------------------

_COUNTRY_CENTRES: dict[str, tuple[float, float]] = {
    "IN": (20.6, 78.9),
    "US": (37.1, -95.7),
    "AE": (23.4, 53.8),
    "SG": (1.35, 103.82),
    "GB": (51.5, -0.12),
    "CN": (35.9, 104.2),
    "MU": (-20.3, 57.6),
    "NG": (9.1, 8.7),
    "PK": (30.4, 69.3),
    "CH": (46.8, 8.2),
}
_DEFAULT_CENTRE = (0.0, 0.0)


def country_coords(country: str, rng: Generator, jitter: float = 3.0) -> tuple[float, float]:
    lat0, lon0 = _COUNTRY_CENTRES.get(country, _DEFAULT_CENTRE)
    lat = float(lat0 + rng.uniform(-jitter, jitter))
    lon = float(lon0 + rng.uniform(-jitter, jitter))
    return round(lat, 4), round(lon, 4)


def country_coords_bulk(countries: list[str], rng: Generator, jitter: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    lats, lons = [], []
    for c in countries:
        lat0, lon0 = _COUNTRY_CENTRES.get(c, _DEFAULT_CENTRE)
        lats.append(lat0 + rng.uniform(-jitter, jitter))
        lons.append(lon0 + rng.uniform(-jitter, jitter))
    return np.round(lats, 4), np.round(lons, 4)
