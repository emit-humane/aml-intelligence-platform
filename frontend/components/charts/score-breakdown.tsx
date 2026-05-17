"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface ScoreBreakdownData {
  rule: number;
  behavioral: number;
  gnn: number;
  graph_boost: number;
}

interface Props {
  breakdown: ScoreBreakdownData;
}

const COLORS = {
  rule:        "#ef4444",
  behavioral:  "#f97316",
  gnn:         "#8b5cf6",
  graph_boost: "#06b6d4",
};

export function ScoreBreakdown({ breakdown }: Props) {
  const data = [
    { name: "Rules",     value: breakdown.rule,        color: COLORS.rule },
    { name: "Behavioral",value: breakdown.behavioral,  color: COLORS.behavioral },
    { name: "GNN",       value: breakdown.gnn,         color: COLORS.gnn },
    { name: "Graph",     value: breakdown.graph_boost, color: COLORS.graph_boost },
  ];

  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
        <XAxis type="number" domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 10 }} />
        <YAxis type="category" dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }} width={72} />
        <Tooltip
          formatter={(v: number) => [`${v.toFixed(1)}`, "Score"]}
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 6 }}
          labelStyle={{ color: "#e5e7eb" }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {data.map((entry) => (
            <Cell key={entry.name} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
