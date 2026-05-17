from pydantic import BaseModel, Field


class BehavioralAnomalyOutput(BaseModel):
    transaction_id: str
    iso_forest_score: float = Field(ge=0.0, le=100.0)
    lof_score: float = Field(ge=0.0, le=100.0)
    autoencoder_recon_error: float        # raw MSE; unbounded
    autoencoder_score: float = Field(ge=0.0, le=100.0)
    ensemble_anomaly_score: float = Field(ge=0.0, le=100.0)  # mean of three scores
    anomaly_drivers: list[str]            # top-3 feature names driving the anomaly
