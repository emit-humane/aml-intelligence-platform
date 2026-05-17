"use client";

import { AlertRow } from "./alert-row";
import type { Alert } from "@/lib/types";

interface Props {
  alerts: Alert[];
  loading?: boolean;
}

export function AlertTable({ alerts, loading }: Props) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-500">
        Loading alerts…
      </div>
    );
  }

  if (alerts.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-600">
        No alerts yet. Start the stream to see live detections.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-800">
      <table className="w-full text-sm">
        <thead className="bg-gray-900 text-gray-400 uppercase text-xs">
          <tr>
            <th className="py-3 px-4 text-left">Alert ID</th>
            <th className="py-3 px-4 text-left">Sender</th>
            <th className="py-3 px-4 text-left">Risk</th>
            <th className="py-3 px-4 text-left">Patterns</th>
            <th className="py-3 px-4 text-left">Status</th>
            <th className="py-3 px-4 text-left">Created</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((a) => (
            <AlertRow key={a.alert_id} alert={a} />
          ))}
        </tbody>
      </table>
    </div>
  );
}
