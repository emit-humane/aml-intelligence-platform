"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchCommunities, fetchSuspiciousPaths } from "@/lib/api";
import { riskTextColor } from "@/lib/utils";
import type { CommunityProfile } from "@/lib/types";

export default function AnalyticsPage() {
  const { data: communities = [], isLoading } = useQuery({
    queryKey: ["communities"],
    queryFn: () => fetchCommunities(0, 50),
  });

  const { data: paths = [] } = useQuery({
    queryKey: ["suspicious-paths"],
    queryFn: () => fetchSuspiciousPaths(undefined, 20),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Analytics</h1>

      {/* Community risk table */}
      <div className="rounded-xl bg-gray-900 border border-gray-800 p-5">
        <h2 className="text-sm font-semibold text-gray-400 mb-4">
          Community Risk Profiles ({communities.length})
        </h2>
        {isLoading ? (
          <div className="text-gray-500 text-sm">Loading…</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-gray-500 uppercase">
                <tr>
                  <th className="py-2 px-3 text-left">Community</th>
                  <th className="py-2 px-3 text-left">Size</th>
                  <th className="py-2 px-3 text-left">Risk Score</th>
                  <th className="py-2 px-3 text-left">Pattern</th>
                  <th className="py-2 px-3 text-left">Volume</th>
                </tr>
              </thead>
              <tbody>
                {communities.map((c: CommunityProfile) => (
                  <tr key={c.community_id} className="border-t border-gray-800">
                    <td className="py-2 px-3 font-mono text-gray-400">#{c.community_id}</td>
                    <td className="py-2 px-3 text-gray-300">{c.size}</td>
                    <td className={`py-2 px-3 font-semibold ${riskTextColor(
                      c.risk_score >= 81 ? "Critical" : c.risk_score >= 61 ? "High" : c.risk_score >= 31 ? "Medium" : "Low"
                    )}`}>
                      {c.risk_score?.toFixed(1) ?? "—"}
                    </td>
                    <td className="py-2 px-3 text-gray-400 text-xs">{c.dominant_pattern ?? "—"}</td>
                    <td className="py-2 px-3 text-gray-500 text-xs">
                      {c.volume_total != null ? `₹${(c.volume_total / 1e6).toFixed(1)}M` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Suspicious paths */}
      <div className="rounded-xl bg-gray-900 border border-gray-800 p-5">
        <h2 className="text-sm font-semibold text-gray-400 mb-4">
          Suspicious Paths ({paths.length})
        </h2>
        <div className="space-y-2">
          {paths.map((p: Record<string, unknown>, i: number) => (
            <div key={i} className="text-xs text-gray-400 font-mono bg-gray-800 rounded p-2">
              {JSON.stringify(p)}
            </div>
          ))}
          {paths.length === 0 && (
            <div className="text-gray-600 text-sm">No suspicious paths found.</div>
          )}
        </div>
      </div>
    </div>
  );
}
