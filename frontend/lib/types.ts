// Types mirroring backend Pydantic models

export type RiskLevel = "Low" | "Medium" | "High" | "Critical";
export type AlertStatus = "Open" | "Investigating" | "SAR_Filed" | "Closed";

export interface RuleEngineOutput {
  transaction_id: string;
  rule_score: number;
  triggered_rules: string[];
  rule_explanations: string[];
  rule_count: number;
}

export interface BehavioralAnomalyOutput {
  transaction_id: string;
  iso_forest_score: number;
  lof_score: number;
  autoencoder_recon_error: number;
  autoencoder_score: number;
  ensemble_anomaly_score: number;
  anomaly_drivers: string[];
}

export interface GNNInferenceOutput {
  transaction_id: string;
  sender_embedding: number[];
  receiver_embedding: number[];
  sender_embedding_drift: number;
  receiver_embedding_drift: number;
  link_anomaly_score: number;
  gnn_anomaly_score: number;
  structural_anomaly_explanations: string[];
  community_anomaly_flag: boolean;
}

export interface FusedRiskOutput {
  transaction_id: string;
  sender_account: string;
  transaction_risk_score: number;
  group_risk_score: number;
  risk_level: RiskLevel;
  risk_level_group: RiskLevel;
  score_breakdown: {
    rule: number;
    behavioral: number;
    gnn: number;
    graph_boost: number;
    weights: Record<string, number>;
  };
  triggered_patterns: string[];
  explanation: string;
  fusion_mode: string;
}

export interface TransactionDetails {
  transaction_id: string;
  timestamp: string;
  sender_account: string;
  receiver_account: string;
  sender_name: string;
  receiver_name: string;
  sender_bank: string;
  receiver_bank: string;
  sender_country: string;
  receiver_country: string;
  amount: number;
  currency: string;
  transaction_type: string;
  payment_channel: string;
  device_id: string;
  ip_address: string;
  geo_latitude: number;
  geo_longitude: number;
  merchant_category: string;
  transaction_status: string;
  sender_balance_before: number;
  sender_balance_after: number;
  receiver_balance_before: number;
  receiver_balance_after: number;
  kyc_level: number;
  is_international: boolean;
  remarks: string;
  amount_leading_digit: number;
}

export interface Alert {
  alert_id: string;
  transaction_id: string;
  sender_account: string;
  community_id: number;
  transaction_risk_score: number;
  group_risk_score: number;
  risk_level: RiskLevel;
  risk_level_group: RiskLevel;
  triggered_patterns: string[];
  rule_explanations: string[];
  anomaly_drivers: string[];
  structural_anomaly_explanations: string[];
  score_breakdown: Record<string, unknown>;
  explanation: string;
  transaction_details: TransactionDetails | null;
  alert_status: AlertStatus;
  assigned_to: string | null;
  created_at: string;
  updated_at: string;
}

export interface ScoredEvent {
  transaction_id: string;
  sender_account: string;
  transaction_risk_score: number;
  risk_level: RiskLevel;
  triggered_patterns: string[];
  explanation: string;
  score_breakdown: {
    rule: number;
    behavioral: number;
    gnn: number;
    graph_boost: number;
  };
  rule_out: Pick<RuleEngineOutput, "rule_score" | "triggered_rules" | "rule_explanations">;
  behav_out: Pick<BehavioralAnomalyOutput, "iso_forest_score" | "lof_score" | "autoencoder_score" | "ensemble_anomaly_score" | "anomaly_drivers">;
  gnn_out: Pick<GNNInferenceOutput, "link_anomaly_score" | "gnn_anomaly_score" | "structural_anomaly_explanations" | "community_anomaly_flag">;
  alert: Alert | null;
}

export interface CommunityProfile {
  community_id: number;
  size: number;
  risk_score: number;
  dominant_pattern: string;
  volume_total: number;
  benford_chi2: number;
}

export interface GraphStats {
  num_nodes: number;
  num_edges: number;
}

export interface NodeInfo {
  account_id: string;
  country: string;
  bank: string;
  kyc_level: number;
  total_sent: number;
  total_received: number;
  in_degree: number;
  out_degree: number;
}
