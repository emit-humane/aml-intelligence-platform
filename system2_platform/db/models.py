"""
SQLAlchemy models — stub for hackathon (in-memory AlertManager used instead).
Defined for completeness; not wired to a live DB in v1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class AlertModel(Base):
    __tablename__ = "alerts"

    alert_id             = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    transaction_id       = Column(String(64), nullable=False, index=True)
    sender_account       = Column(String(64), nullable=False, index=True)
    community_id         = Column(Integer, nullable=False, default=-1)
    transaction_risk_score = Column(Float, nullable=False)
    group_risk_score     = Column(Float, nullable=False)
    risk_level           = Column(String(16), nullable=False)
    risk_level_group     = Column(String(16), nullable=False)
    triggered_patterns   = Column(JSON, nullable=False, default=list)
    rule_explanations    = Column(JSON, nullable=False, default=list)
    anomaly_drivers      = Column(JSON, nullable=False, default=list)
    structural_anomaly_explanations = Column(JSON, nullable=False, default=list)
    score_breakdown      = Column(JSON, nullable=False, default=dict)
    explanation          = Column(Text, nullable=False, default="")
    alert_status         = Column(String(32), nullable=False, default="Open")
    assigned_to          = Column(String(128), nullable=True)
    created_at           = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at           = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                                  onupdate=lambda: datetime.now(timezone.utc))
