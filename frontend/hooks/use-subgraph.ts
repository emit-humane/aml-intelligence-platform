"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchNodeInfo, fetchNodeNeighbors, fetchSubgraph } from "@/lib/api";
import type { SubgraphData } from "@/lib/api";

export function useNodeInfo(accountId: string | null) {
  return useQuery({
    queryKey: ["node", accountId],
    queryFn: () => fetchNodeInfo(accountId!),
    enabled: !!accountId,
  });
}

export function useSubgraph(accountId: string | null, hops = 2) {
  return useQuery<SubgraphData>({
    queryKey: ["subgraph", accountId, hops],
    queryFn: () => fetchSubgraph(accountId!, hops),
    enabled: !!accountId,
    staleTime: 30_000,
  });
}

export function useNeighbors(accountId: string | null, hops = 1) {
  return useQuery({
    queryKey: ["neighbors", accountId, hops],
    queryFn: () => fetchNodeNeighbors(accountId!, hops),
    enabled: !!accountId,
  });
}
