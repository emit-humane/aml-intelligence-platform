from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .fused_risk_output import RiskLevel

AlertStatus = Literal["Open", "Investigating", "SAR_Filed", "Closed"]


class Alert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    transaction_id: str
    sender_account: str
    community_id: int
    transaction_risk_score: float = Field(ge=0.0, le=100.0)
    group_risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel
    risk_level_group: RiskLevel
    triggered_patterns: list[str]
    rule_explanations: list[str]
    anomaly_drivers: list[str]
    structural_anomaly_explanations: list[str]
    score_breakdown: dict[str, Any]
    explanation: str
    alert_status: AlertStatus = "Open"
    assigned_to: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
