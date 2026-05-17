"use client";

import Link from "next/link";
import { useStream } from "@/hooks/use-stream";
import { useAlerts, useAlertStats } from "@/hooks/use-alerts";
import { useQuery } from "@tanstack/react-query";
import { fetchMetrics } from "@/lib/api";
import { LiveEventFeed } from "@/components/stream/live-event-feed";
import { StreamControls } from "@/components/stream/stream-controls";
import { RiskTrend } from "@/components/charts/risk-trend";
import { AlertTable } from "@/components/alerts/alert-table";

export default function DashboardPage() {
  const { events, connected, paused, togglePause, clearEvents } = useStream();
  const { data: stats } = useAlertStats();
  const { data: alerts = [], isLoading: alertsLoading } = useAlerts({ limit: 10 });
  const { data: metrics } = useQuery({
    queryKey: ["metrics"],
    queryFn: fetchMetrics,
    refetchInterval: 10_000,
  });

  const total    = stats?.total    ?? 0;
  const critical = stats?.Critical ?? 0;
  const high     = stats?.High     ?? 0;
  const medium   = stats?.Medium   ?? 0;
  // False-positive rate placeholder: alerts that were Closed / total
  const closed   = stats?.Closed   ?? 0;
  const fpr      = total > 0 ? ((closed / total) * 100).toFixed(1) : "0.0";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Live Dashboard</h1>
          {metrics && (
            <p className="text-xs text-gray-500 mt-0.5">
              {metrics.graph.live_nodes.toLocaleString()} nodes ·{" "}
              {metrics.graph.live_edges.toLocaleString()} edges ·{" "}
              {metrics.offline.n_communities} communities
            </p>
          )}
        </div>
        <StreamControls
          connected={connected}
          paused={paused}
          eventCount={events.length}
          onTogglePause={togglePause}
          onClear={clearEvents}
        />
      </div>

      {/* ── Top row: 4 stat cards ── */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Total Alerts"   value={total}    color="text-blue-400"   sub={`${metrics?.alerts?.Open ?? 0} open`} />
        <StatCard label="Critical"       value={critical} color="text-red-400"    sub="highest priority" />
        <StatCard label="High"           value={high}     color="text-orange-400" sub="investigate now" />
        <StatCard label="Closed / FPR"   value={`${fpr}%`} color="text-green-400"  sub={`${closed} closed`} />
      </div>

      {/* ── Middle row: trend + live feed side by side ── */}
      <div className="grid grid-cols-5 gap-4">
        {/* Risk trend (wider) */}
        <div className="col-span-3 rounded-xl bg-gray-900 border border-gray-800 p-4">
          <h2 className="text-sm font-semibold text-gray-400 mb-3">
            Risk Score Trend — last {Math.min(events.length, 60)} events
          </h2>
          <RiskTrend events={events} maxPoints={60} />
        </div>

        {/* Live event feed (narrower) */}
        <div className="col-span-2 rounded-xl bg-gray-900 border border-gray-800 p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-gray-400">Live Event Stream</h2>
            <span className="text-xs text-gray-600">{events.length} buffered</span>
          </div>
          <div className="flex-1 overflow-hidden">
            <LiveEventFeed events={events.slice(0, 20)} />
          </div>
        </div>
      </div>

      {/* ── Bottom row: recent alerts ── */}
      <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-400">Recent Alerts</h2>
          <Link
            href="/alerts"
            className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
          >
            View all {total} alerts →
          </Link>
        </div>
        <AlertTable alerts={alerts} loading={alertsLoading} />
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  color,
  sub,
}: {
  label: string;
  value: number | string;
  color: string;
  sub?: string;
}) {
  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-5">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className={`text-3xl font-bold mt-1 ${color}`}>{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-1">{sub}</div>}
    </div>
  );
}
