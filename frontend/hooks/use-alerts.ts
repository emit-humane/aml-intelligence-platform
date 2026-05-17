"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchAlerts, fetchAlert, fetchAlertStats, updateAlertStatus } from "@/lib/api";
import type { Alert } from "@/lib/types";

export function useAlerts(params?: {
  risk_level?: string;
  status?: string;
  limit?: number;
}) {
  return useQuery({
    queryKey: ["alerts", params],
    queryFn: () => fetchAlerts(params),
    refetchInterval: 5000,   // poll every 5s for live updates
  });
}

export function useAlert(alertId: string) {
  return useQuery({
    queryKey: ["alert", alertId],
    queryFn: () => fetchAlert(alertId),
    enabled: !!alertId,
  });
}

export function useAlertStats() {
  return useQuery({
    queryKey: ["alert-stats"],
    queryFn: fetchAlertStats,
    refetchInterval: 5000,
  });
}

export function useUpdateAlertStatus() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ alertId, status, assigned_to }: {
      alertId: string; status: string; assigned_to?: string;
    }) => updateAlertStatus(alertId, status, assigned_to),
    onSuccess: (updated: Alert) => {
      qc.invalidateQueries({ queryKey: ["alerts"] });
      qc.setQueryData(["alert", updated.alert_id], updated);
    },
  });
}
