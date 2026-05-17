from pydantic import BaseModel, Field


class GNNInferenceOutput(BaseModel):
    transaction_id: str
    sender_embedding: list[float]    # shape (64,) — TGN node embedding
    receiver_embedding: list[float]  # shape (64,)
    sender_embedding_drift: float = Field(ge=0.0, le=1.0)    # cosine dist from baseline
    receiver_embedding_drift: float = Field(ge=0.0, le=1.0)
    link_anomaly_score: float = Field(ge=0.0, le=1.0)   # 1 − link-prediction probability
    gnn_anomaly_score: float = Field(ge=0.0, le=100.0)  # aggregated, normalised 0–100
    structural_anomaly_explanations: list[str]   # e.g. ["cycle closure", "embedding drift"]
    community_anomaly_flag: bool                 # community Benford or density anomaly
