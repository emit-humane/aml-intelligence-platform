"use client";

import Link from "next/link";
import { RiskBadge } from "./risk-badge";
import { formatTimestamp, truncateId } from "@/lib/utils";
import type { Alert } from "@/lib/types";

interface Props {
  alert: Alert;
}

export function AlertRow({ alert }: Props) {
  return (
    <tr className="border-b border-gray-800 hover:bg-gray-900 transition-colors">
      <td className="py-3 px-4">
        <Link
          href={`/alerts/${alert.alert_id}`}
          className="font-mono text-blue-400 hover:underline text-sm"
        >
          {truncateId(alert.alert_id)}
        </Link>
      </td>
      <td className="py-3 px-4 font-mono text-xs text-gray-300">
        {truncateId(alert.sender_account)}
      </td>
      <td className="py-3 px-4">
        <RiskBadge level={alert.risk_level} score={alert.transaction_risk_score} />
      </td>
      <td className="py-3 px-4 text-xs text-gray-400 max-w-xs truncate">
        {alert.triggered_patterns.slice(0, 3).join(", ")}
      </td>
      <td className="py-3 px-4">
        <span className={`text-xs px-2 py-1 rounded font-medium ${statusClass(alert.alert_status)}`}>
          {alert.alert_status}
        </span>
      </td>
      <td className="py-3 px-4 text-xs text-gray-500">
        {formatTimestamp(alert.created_at)}
      </td>
    </tr>
  );
}

function statusClass(status: string): string {
  switch (status) {
    case "Open":          return "bg-blue-900 text-blue-300";
    case "Investigating": return "bg-yellow-900 text-yellow-300";
    case "SAR_Filed":     return "bg-purple-900 text-purple-300";
    case "Closed":        return "bg-gray-800 text-gray-400";
    default:              return "bg-gray-800 text-gray-400";
  }
}
