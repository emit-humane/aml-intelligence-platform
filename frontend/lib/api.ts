import axios from "axios";
import type { Alert, ScoredEvent, CommunityProfile, GraphStats, NodeInfo } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 10_000,
});

// --------------------------------------------------------------------------
// Alerts
// --------------------------------------------------------------------------

export async function fetchAlerts(params?: {
  risk_level?: string;
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<Alert[]> {
  const { data } = await client.get<Alert[]>("/alerts/", { params });
  return data;
}

export async function fetchAlert(alertId: string): Promise<Alert> {
  const { data } = await client.get<Alert>(`/alerts/${alertId}`);
  return data;
}

export async function fetchAlertStats(): Promise<Record<string, number>> {
  const { data } = await client.get<Record<string, number>>("/alerts/stats");
  return data;
}

export async function updateAlertStatus(
  alertId: string,
  status: string,
  assigned_to?: string
): Promise<Alert> {
  const { data } = await client.patch<Alert>(`/alerts/${alertId}/status`, {
    status,
    assigned_to,
  });
  return data;
}

// --------------------------------------------------------------------------
// Stream (single transaction scoring)
// --------------------------------------------------------------------------

export async function scoreTransaction(payload: Record<string, unknown>): Promise<ScoredEvent> {
  const { data } = await client.post<ScoredEvent>("/stream/transaction", payload);
  return data;
}

// --------------------------------------------------------------------------
// Graph
// --------------------------------------------------------------------------

export async function fetchGraphStats(): Promise<GraphStats> {
  const { data } = await client.get<GraphStats>("/graph/stats");
  return data;
}

export async function fetchNodeInfo(accountId: string): Promise<NodeInfo> {
  const { data } = await client.get<NodeInfo>(`/graph/node/${accountId}`);
  return data;
}

export async function fetchNodeNeighbors(accountId: string, hops = 1) {
  const { data } = await client.get(`/graph/node/${accountId}/neighbors`, {
    params: { hops },
  });
  return data;
}

export interface SubgraphData {
  center: string;
  hops: number;
  nodes: Array<{ id: string; in_degree: number; out_degree: number; is_center: boolean }>;
  edges: Array<{ source: string; target: string; weight: number }>;
}

export async function fetchSubgraph(accountId: string, hops = 2): Promise<SubgraphData> {
  const { data } = await client.get<SubgraphData>(`/graph/subgraph/${accountId}`, {
    params: { hops },
  });
  return data;
}

export interface Metrics {
  graph: { live_nodes: number; live_edges: number };
  alerts: Record<string, number>;
  offline: { n_communities: number; n_suspicious_paths: number; n_behavioral_profiles: number };
  engines_loaded: boolean;
}

export async function fetchMetrics(): Promise<Metrics> {
  const { data } = await client.get<Metrics>("/analytics/metrics");
  return data;
}

// --------------------------------------------------------------------------
// Analytics
// --------------------------------------------------------------------------

export async function fetchCommunities(min_risk = 0, limit = 50): Promise<CommunityProfile[]> {
  const { data } = await client.get<CommunityProfile[]>("/analytics/communities", {
    params: { min_risk, limit },
  });
  return data;
}

export async function fetchCommunityProfile(communityId: number): Promise<CommunityProfile> {
  const { data } = await client.get<CommunityProfile>(`/analytics/community/${communityId}`);
  return data;
}

export async function fetchSuspiciousPaths(communityId?: number, limit = 20) {
  const { data } = await client.get("/analytics/suspicious-paths", {
    params: { community_id: communityId, limit },
  });
  return data;
}

// --------------------------------------------------------------------------
// Health
// --------------------------------------------------------------------------

export async function fetchHealth() {
  const { data } = await client.get("/health/ready");
  return data;
}
