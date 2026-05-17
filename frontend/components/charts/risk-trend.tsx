"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { ScoredEvent } from "@/lib/types";

interface Props {
  events: ScoredEvent[];
  maxPoints?: number;
}

export function RiskTrend({ events, maxPoints = 60 }: Props) {
  const data = events
    .slice(0, maxPoints)
    .reverse()
    .map((e, i) => ({
      i,
      score: e.transaction_risk_score,
      id: e.transaction_id.slice(0, 8),
    }));

  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ left: 0, right: 8, top: 4, bottom: 4 }}>
        <XAxis dataKey="i" hide />
        <YAxis domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 10 }} width={28} />
        <Tooltip
          formatter={(v: number) => [`${v.toFixed(1)}`, "Risk score"]}
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 6 }}
          labelStyle={{ color: "#6b7280", fontSize: 10 }}
          labelFormatter={() => ""}
        />
        <ReferenceLine y={31} stroke="#facc15" strokeDasharray="3 3" label={{ value: "Med", fill: "#facc15", fontSize: 9, position: "insideTopLeft" }} />
        <ReferenceLine y={61} stroke="#f97316" strokeDasharray="3 3" label={{ value: "High", fill: "#f97316", fontSize: 9, position: "insideTopLeft" }} />
        <ReferenceLine y={81} stroke="#ef4444" strokeDasharray="3 3" label={{ value: "Crit", fill: "#ef4444", fontSize: 9, position: "insideTopLeft" }} />
        <Line
          type="monotone"
          dataKey="score"
          stroke="#60a5fa"
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3, fill: "#60a5fa" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
