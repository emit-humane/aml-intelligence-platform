"""
Tests for the Layer 1 Rule Engine — R01 through R13 plus scoring formula.

Each rule has a trigger test (condition exactly at/above threshold) and a
no-trigger test (condition just below threshold).

The final test verifies the scoring formula directly.
"""
from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path so imports work without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from system2_platform.contracts.live_feature_vector import LiveFeatureVector
from system2_platform.contracts.live_graph_feature_vector import LiveGraphFeatureVector
from system2_platform.layer1_rules.engine import RuleEngine, MAX_POSSIBLE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCALED = [0.0] * 45  # dummy scaled vector


def _fv(**kwargs) -> LiveFeatureVector:
    """Build a minimal LiveFeatureVector with sensible defaults and override via kwargs."""
    defaults = dict(
        transaction_id="TX_TEST",
        sender_account="ACC_TEST",
        tx_velocity_1h=0.0,
        tx_velocity_6h=0.0,
        tx_velocity_24h=0.0,
        tx_velocity_7d=0.0,
        avg_amount_7d=0.0,
        std_amount_7d=0.0,
        tx_gap_seconds=0.0,
        night_tx_ratio=0.0,
        weekend_tx_ratio=0.0,
        hour_of_day=12,
        day_of_week=1,
        beneficiary_count_7d=0,
        beneficiary_count_30d=0,
        receiver_entropy=0.0,
        amount_zscore=0.0,
        avg_daily_volume_30d=0.0,
        round_amount_flag=False,
        sub_threshold_flag=False,
        amount_leading_digit=1,
        benford_chi2_score=0.0,
        geo_distance_km=0.0,
        country_switch_count_7d=0,
        impossible_travel_flag=False,
        high_risk_country_flag=False,
        cross_border_ratio_30d=0.0,
        device_change_count_7d=0,
        new_device_flag=False,
        shared_device_count=0,
        ip_change_count_24h=0,
        amount=0.0,
        is_international=False,
        kyc_level=1,
        scaled_feature_vector=_SCALED,
    )
    defaults.update(kwargs)
    return LiveFeatureVector(**defaults)


def _gfv(**kwargs) -> LiveGraphFeatureVector:
    """Build a minimal LiveGraphFeatureVector with sensible defaults."""
    defaults = dict(
        transaction_id="TX_TEST",
        sender_account="ACC_TEST",
        receiver_account="ACC_RCV",
        sender_in_degree=0,
        sender_out_degree=1,
        sender_in_degree_unique=0,
        sender_out_degree_unique=1,
        sender_2hop_cycle_count=0,
        sender_3hop_cycle_count=0,
        sender_fan_in_score=0.0,
        sender_fan_out_score=1.0,
        sender_pagerank=0.01,
        sender_betweenness=0.0,
        sender_community_id=1,
        sender_community_size=10,
        sender_community_density=0.1,
        sender_community_risk_score=0.0,
        sender_benford_chi2_community=0.0,
        receiver_in_degree=1,
        receiver_in_degree_unique=1,
        receiver_out_degree=0,
        receiver_community_id=1,
        receiver_community_risk_score=0.0,
        edge_creates_cycle=False,
        cycle_length=0,
        shared_community=True,
        two_hop_neighborhood=[],
        two_hop_edge_list=[],
    )
    defaults.update(kwargs)
    return LiveGraphFeatureVector(**defaults)


engine = RuleEngine()

# ---------------------------------------------------------------------------
# R01 — High-Value Single Transaction
# Spec: amount > 1_000_000, weight 1.0
# ---------------------------------------------------------------------------

def test_R01_trigger():
    fv = _fv(amount=1_000_001.0)
    result = engine.evaluate(fv)
    assert "R01" in result.triggered_rules


def test_R01_no_trigger():
    fv = _fv(amount=1_000_000.0)  # not strictly greater than
    result = engine.evaluate(fv)
    assert "R01" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R02 — Structuring
# Spec: sub_threshold_flag == True AND tx_velocity_1h >= 3, weight 1.0
# ---------------------------------------------------------------------------

def test_R02_trigger():
    fv = _fv(sub_threshold_flag=True, tx_velocity_1h=3.0)
    result = engine.evaluate(fv)
    assert "R02" in result.triggered_rules


def test_R02_no_trigger():
    # sub_threshold_flag True but velocity too low
    fv = _fv(sub_threshold_flag=True, tx_velocity_1h=2.0)
    result = engine.evaluate(fv)
    assert "R02" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R03 — Velocity Spike
# Spec: tx_velocity_1h > 10 OR tx_velocity_24h > 50, weight 0.8
# ---------------------------------------------------------------------------

def test_R03_trigger():
    fv = _fv(tx_velocity_1h=11.0)
    result = engine.evaluate(fv)
    assert "R03" in result.triggered_rules


def test_R03_no_trigger():
    fv = _fv(tx_velocity_1h=10.0, tx_velocity_24h=50.0)
    result = engine.evaluate(fv)
    assert "R03" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R04 — Dormant Account Activation
# Spec: tx_gap_seconds > 15_552_000 AND amount > 100_000, weight 0.9
# ---------------------------------------------------------------------------

def test_R04_trigger():
    fv = _fv(tx_gap_seconds=15_552_001.0, amount=100_001.0)
    result = engine.evaluate(fv)
    assert "R04" in result.triggered_rules


def test_R04_no_trigger():
    # gap exactly at threshold (not strictly greater)
    fv = _fv(tx_gap_seconds=15_552_000.0, amount=100_001.0)
    result = engine.evaluate(fv)
    assert "R04" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R05 — Impossible Travel
# Spec: impossible_travel_flag == True, weight 0.9
# ---------------------------------------------------------------------------

def test_R05_trigger():
    fv = _fv(impossible_travel_flag=True)
    result = engine.evaluate(fv)
    assert "R05" in result.triggered_rules


def test_R05_no_trigger():
    fv = _fv(impossible_travel_flag=False)
    result = engine.evaluate(fv)
    assert "R05" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R06 — High-Risk Jurisdiction
# Spec: high_risk_country_flag == True AND is_international == True, weight 0.7
# ---------------------------------------------------------------------------

def test_R06_trigger():
    fv = _fv(high_risk_country_flag=True, is_international=True)
    result = engine.evaluate(fv)
    assert "R06" in result.triggered_rules


def test_R06_no_trigger():
    # high risk but NOT international
    fv = _fv(high_risk_country_flag=True, is_international=False)
    result = engine.evaluate(fv)
    assert "R06" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R07 — Device Anomaly
# Spec: new_device_flag == True AND amount > 200_000, weight 0.6
# ---------------------------------------------------------------------------

def test_R07_trigger():
    fv = _fv(new_device_flag=True, amount=200_001.0)
    result = engine.evaluate(fv)
    assert "R07" in result.triggered_rules


def test_R07_no_trigger():
    # new device but amount exactly at threshold (not strictly greater)
    fv = _fv(new_device_flag=True, amount=200_000.0)
    result = engine.evaluate(fv)
    assert "R07" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R08 — Shared Device
# Spec: shared_device_count > 5, weight 0.7
# ---------------------------------------------------------------------------

def test_R08_trigger():
    fv = _fv(shared_device_count=6)
    result = engine.evaluate(fv)
    assert "R08" in result.triggered_rules


def test_R08_no_trigger():
    fv = _fv(shared_device_count=5)  # not strictly greater than 5
    result = engine.evaluate(fv)
    assert "R08" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R09 — Excessive Beneficiaries
# Spec: beneficiary_count_7d > 20, weight 0.8
# ---------------------------------------------------------------------------

def test_R09_trigger():
    fv = _fv(beneficiary_count_7d=21)
    result = engine.evaluate(fv)
    assert "R09" in result.triggered_rules


def test_R09_no_trigger():
    fv = _fv(beneficiary_count_7d=20)  # not strictly greater than 20
    result = engine.evaluate(fv)
    assert "R09" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R10 — Cycle Closure
# Spec: edge_creates_cycle == True, weight 1.0
# ---------------------------------------------------------------------------

def test_R10_trigger():
    fv = _fv()
    gfv = _gfv(edge_creates_cycle=True, cycle_length=3)
    result = engine.evaluate(fv, gfv)
    assert "R10" in result.triggered_rules


def test_R10_no_trigger():
    fv = _fv()
    gfv = _gfv(edge_creates_cycle=False)
    result = engine.evaluate(fv, gfv)
    assert "R10" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R11 — Round Amount Structuring
# Spec: round_amount_flag == True AND tx_velocity_24h > 5, weight 0.6
# ---------------------------------------------------------------------------

def test_R11_trigger():
    fv = _fv(round_amount_flag=True, tx_velocity_24h=6.0)
    result = engine.evaluate(fv)
    assert "R11" in result.triggered_rules


def test_R11_no_trigger():
    # round_amount_flag True but velocity exactly at threshold (not strictly greater)
    fv = _fv(round_amount_flag=True, tx_velocity_24h=5.0)
    result = engine.evaluate(fv)
    assert "R11" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R12 — KYC Mismatch
# Spec: kyc_level == 0 AND amount > 500_000, weight 0.8
# ---------------------------------------------------------------------------

def test_R12_trigger():
    fv = _fv(kyc_level=0, amount=500_001.0)
    result = engine.evaluate(fv)
    assert "R12" in result.triggered_rules


def test_R12_no_trigger():
    # kyc_level 0 but amount exactly at threshold (not strictly greater)
    fv = _fv(kyc_level=0, amount=500_000.0)
    result = engine.evaluate(fv)
    assert "R12" not in result.triggered_rules


# ---------------------------------------------------------------------------
# R13 — Benford Anomaly
# Spec: benford_chi2_score > 3.84, weight 0.7
# ---------------------------------------------------------------------------

def test_R13_trigger():
    fv = _fv(benford_chi2_score=3.85)
    result = engine.evaluate(fv)
    assert "R13" in result.triggered_rules


def test_R13_no_trigger():
    fv = _fv(benford_chi2_score=3.84)  # not strictly greater than
    result = engine.evaluate(fv)
    assert "R13" not in result.triggered_rules


# ---------------------------------------------------------------------------
# Scoring formula test
# Fire only R01 (weight 1.0): score = round(1.0 / 12.2 * 100, 2)
# ---------------------------------------------------------------------------

def test_scoring_formula_single_rule():
    fv = _fv(amount=1_000_001.0)  # triggers R01 only (all other conditions off)
    result = engine.evaluate(fv)
    assert "R01" in result.triggered_rules
    expected_score = round(1.0 / MAX_POSSIBLE * 100, 2)
    assert result.rule_score == pytest.approx(expected_score, abs=0.01)
