from pydantic import BaseModel

# TODO(ambiguous): spec comment says 45 features but L3A autoencoder is 52→32→16→32→52.
# scaled_feature_vector length must match FEATURE_DIM constant in
# layer3_behavioral/models/autoencoder.py (currently set to 52).


class LiveFeatureVector(BaseModel):
    transaction_id: str
    sender_account: str

    # --- Temporal ---
    tx_velocity_1h: float
    tx_velocity_6h: float
    tx_velocity_24h: float
    tx_velocity_7d: float
    avg_amount_7d: float
    std_amount_7d: float
    tx_gap_seconds: float
    night_tx_ratio: float
    weekend_tx_ratio: float
    hour_of_day: int
    day_of_week: int

    # --- Behavioral ---
    beneficiary_count_7d: int
    beneficiary_count_30d: int
    receiver_entropy: float
    amount_zscore: float
    avg_daily_volume_30d: float
    round_amount_flag: bool
    sub_threshold_flag: bool
    amount_leading_digit: int
    benford_chi2_score: float

    # --- Geographic ---
    geo_distance_km: float
    country_switch_count_7d: int
    impossible_travel_flag: bool
    high_risk_country_flag: bool
    cross_border_ratio_30d: float

    # --- Device ---
    device_change_count_7d: int
    new_device_flag: bool
    shared_device_count: int
    ip_change_count_24h: int

    # --- Transaction-level fields (needed by rule engine) ---
    amount: float = 0.0
    is_international: bool = False
    kyc_level: int = 0

    # StandardScaler output fed directly to L3 models
    scaled_feature_vector: list[float]
