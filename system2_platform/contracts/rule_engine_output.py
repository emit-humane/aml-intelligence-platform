from pydantic import BaseModel, Field


class RuleEngineOutput(BaseModel):
    transaction_id: str
    rule_score: float = Field(ge=0.0, le=100.0)
    triggered_rules: list[str]    # e.g. ["R02", "R10"]
    rule_explanations: list[str]  # one human-readable string per triggered rule
    rule_count: int
