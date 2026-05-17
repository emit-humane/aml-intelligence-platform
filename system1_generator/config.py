import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class AmountDistribution(BaseModel):
    normal_mean: float
    normal_std: float
    min: float
    max: float


class GeneratorConfig(BaseModel):
    seed: int = 42
    num_accounts: int = 5000
    num_historical_transactions: int = 500_000
    num_stream_transactions: int = 50_000
    fraud_ratio: float = Field(default=0.008, ge=0.0, le=1.0)
    history_days: int = 90
    stream_days: int = 10
    banks: list[str]
    countries: list[str]
    high_risk_countries: list[str]
    transaction_types: list[str]
    payment_channels: list[str]
    amount_distribution: AmountDistribution
    structuring_threshold: float = 1_000_000
    scenario_probabilities: dict[str, float]

    @model_validator(mode="after")
    def _probs_sum_to_one(self) -> "GeneratorConfig":
        total = sum(self.scenario_probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scenario_probabilities must sum to 1.0, got {total:.6f}")
        return self

    @classmethod
    def from_json(cls, path: Path | str) -> "GeneratorConfig":
        with open(path) as fh:
            return cls(**json.load(fh))
