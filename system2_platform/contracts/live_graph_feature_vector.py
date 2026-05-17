from pydantic import BaseModel


class LiveGraphFeatureVector(BaseModel):
    transaction_id: str
    sender_account: str
    receiver_account: str

    # --- Sender node metrics (live MultiDiGraph) ---
    sender_in_degree: int
    sender_out_degree: int
    sender_in_degree_unique: int
    sender_out_degree_unique: int
    sender_2hop_cycle_count: int
    sender_3hop_cycle_count: int
    sender_fan_in_score: float
    sender_fan_out_score: float
    sender_pagerank: float
    sender_betweenness: float
    sender_community_id: int
    sender_community_size: int
    sender_community_density: float
    sender_community_risk_score: float
    sender_benford_chi2_community: float

    # --- Receiver node metrics ---
    receiver_in_degree: int
    receiver_in_degree_unique: int   # unique predecessor accounts (fan-in signal)
    receiver_out_degree: int
    receiver_community_id: int
    receiver_community_risk_score: float

    # --- Edge-level structural features ---
    edge_creates_cycle: bool
    cycle_length: int        # hop count of closed cycle; 0 if no cycle
    shared_community: bool   # sender and receiver in same Infomap community

    # --- Neighbourhood context for GNN message passing ---
    two_hop_neighborhood: list[str]               # account IDs within 2 hops of sender
    two_hop_edge_list: list[tuple[str, str, str, float]]  # (u, v, timestamp_iso, amount)
