from typing import Any, Literal

from pydantic import BaseModel, Field

# Thresholds: 0–30 → Low | 31–60 → Medium | 61–80 → High | 81–100 → Critical
RiskLevel = Literal["Low", "Medium", "High", "Critical"]


class FusedRiskOutput(BaseModel):
    transaction_id: str
    sender_account: str
    transaction_risk_score: float = Field(ge=0.0, le=100.0)
    group_risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel
    risk_level_group: RiskLevel
    # Expected keys: rule, graph, anomaly, gnn, weights
    score_breakdown: dict[str, Any]
    triggered_patterns: list[str]   # consolidated from all layer outputs
    explanation: str                # one-sentence human-readable summary
    fusion_mode: Literal["static_weights", "meta_learner"]
