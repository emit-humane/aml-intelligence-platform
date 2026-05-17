"use client";

import { useState } from "react";
import { useAlerts, useAlertStats } from "@/hooks/use-alerts";
import { AlertTable } from "@/components/alerts/alert-table";

const RISK_LEVELS = ["", "Low", "Medium", "High", "Critical"];
const STATUSES = ["", "Open", "Investigating", "SAR_Filed", "Closed"];

export default function AlertsPage() {
  const [riskLevel, setRiskLevel] = useState("");
  const [status, setStatus] = useState("");

  const { data: alerts = [], isLoading } = useAlerts({
    risk_level: riskLevel || undefined,
    status: status || undefined,
    limit: 100,
  });
  const { data: stats } = useAlertStats();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Alert Queue</h1>
        <div className="text-sm text-gray-500">{alerts.length} alerts shown</div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="flex gap-4 text-sm">
          {["Open", "Investigating", "SAR_Filed", "Closed"].map((s) => (
            <div key={s} className="rounded-lg bg-gray-900 border border-gray-800 px-4 py-2">
              <span className="text-gray-500">{s}: </span>
              <span className="font-semibold text-gray-200">{stats[s] ?? 0}</span>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-3">
        <select
          value={riskLevel}
          onChange={(e) => setRiskLevel(e.target.value)}
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-blue-500"
        >
          <option value="">All Risk Levels</option>
          {RISK_LEVELS.filter(Boolean).map((r) => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-blue-500"
        >
          <option value="">All Statuses</option>
          {STATUSES.filter(Boolean).map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      <AlertTable alerts={alerts} loading={isLoading} />
    </div>
  );
}
