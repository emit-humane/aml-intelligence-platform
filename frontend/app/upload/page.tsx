"use client";

import { useState } from "react";
import { scoreTransaction } from "@/lib/api";
import { RiskBadge } from "@/components/alerts/risk-badge";
import { ScoreBreakdown } from "@/components/charts/score-breakdown";
import type { ScoredEvent } from "@/lib/types";

const SAMPLE = {
  transaction_id: "TXN_DEMO_001",
  timestamp: new Date().toISOString(),
  sender_account: "ACC_DEMO_SENDER",
  receiver_account: "ACC_DEMO_RECV",
  sender_name: "Demo Sender",
  receiver_name: "Demo Receiver",
  sender_bank: "HDFC",
  receiver_bank: "SBI",
  sender_country: "IN",
  receiver_country: "AE",
  amount: 950000,
  currency: "INR",
  transaction_type: "Wire",
  payment_channel: "Web",
  device_id: "DEV_001",
  ip_address: "192.168.1.1",
  geo_latitude: 28.6139,
  geo_longitude: 77.209,
  merchant_category: "Finance",
  transaction_status: "Success",
  sender_balance_before: 2000000,
  sender_balance_after: 1050000,
  receiver_balance_before: 100000,
  receiver_balance_after: 1050000,
  kyc_level: 1,
  is_international: true,
  remarks: "Wire transfer",
  amount_leading_digit: 9,
};

export default function UploadPage() {
  const [json, setJson] = useState(JSON.stringify(SAMPLE, null, 2));
  const [result, setResult] = useState<ScoredEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
    setError(null);
    try {
      const payload = JSON.parse(json);
      const res = await scoreTransaction(payload);
      setResult(res);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  const bd = result?.score_breakdown;

  return (
    <div className="max-w-2xl space-y-6">
      <h1 className="text-2xl font-bold text-white">Score Transaction</h1>

      <div className="rounded-xl bg-gray-900 border border-gray-800 p-5 space-y-3">
        <h2 className="text-sm font-semibold text-gray-400">Transaction JSON</h2>
        <textarea
          value={json}
          onChange={(e) => setJson(e.target.value)}
          rows={18}
          className="w-full font-mono text-xs bg-gray-950 border border-gray-700 rounded-lg p-3 text-green-300 focus:outline-none focus:border-blue-500 resize-none"
        />
        <button
          onClick={submit}
          disabled={loading}
          className="w-full py-2 bg-blue-700 hover:bg-blue-600 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50"
        >
          {loading ? "Scoring…" : "Score Transaction"}
        </button>
        {error && <div className="text-red-400 text-xs">{error}</div>}
      </div>

      {result && (
        <div className="rounded-xl bg-gray-900 border border-gray-800 p-5 space-y-4">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-gray-400">Result</h2>
            <RiskBadge level={result.risk_level} score={result.transaction_risk_score} />
          </div>
          <p className="text-sm text-gray-300">{result.explanation}</p>
          {bd && (
            <ScoreBreakdown breakdown={{
              rule:        (bd as Record<string, number>).rule        ?? 0,
              behavioral:  (bd as Record<string, number>).behavioral  ?? 0,
              gnn:         (bd as Record<string, number>).gnn         ?? 0,
              graph_boost: (bd as Record<string, number>).graph_boost ?? 0,
            }} />
          )}
          <details className="text-xs text-gray-500">
            <summary className="cursor-pointer hover:text-gray-400">Raw JSON</summary>
            <pre className="mt-2 bg-gray-950 p-3 rounded overflow-x-auto text-green-400">
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}
