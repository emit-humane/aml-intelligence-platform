"use client";

// Pattern frequency heatmap — recharts simple bar chart
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const PATTERN_COLORS: Record<string, string> = {
  structuring:           "#ef4444",
  circular_laundering:   "#f97316",
  layering_chain:        "#eab308",
  fan_in:                "#84cc16",
  fan_out:               "#06b6d4",
  fraud_ring:            "#8b5cf6",
  dormant_activation:    "#ec4899",
  velocity_burst:        "#14b8a6",
  cross_border_layering: "#f59e0b",
  round_tripping:        "#6366f1",
};

interface Props {
  patterns: Record<string, number>;
}

export function PatternHeatmap({ patterns }: Props) {
  const data = Object.entries(patterns)
    .map(([name, count]) => ({ name: name.replace(/_/g, " "), count, key: name }))
    .sort((a, b) => b.count - a.count);

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ left: 8, right: 8, top: 4, bottom: 40 }}>
        <XAxis
          dataKey="name"
          tick={{ fill: "#6b7280", fontSize: 9 }}
          angle={-35}
          textAnchor="end"
        />
        <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} />
        <Tooltip
          formatter={(v: number) => [v, "Count"]}
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 6 }}
        />
        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.key} fill={PATTERN_COLORS[d.key] ?? "#6b7280"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
