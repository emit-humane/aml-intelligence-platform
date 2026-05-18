"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { useAlert, useUpdateAlertStatus } from "@/hooks/use-alerts";
import { RiskBadge } from "@/components/alerts/risk-badge";
import { ScoreBreakdown } from "@/components/charts/score-breakdown";
import { formatTimestamp, formatCurrency } from "@/lib/utils";
import type { TransactionDetails } from "@/lib/types";

export default function AlertDetailPage() {
  const { alertId } = useParams<{ alertId: string }>();
  const { data: alert, isLoading } = useAlert(alertId);
  const { mutate: updateStatus, isPending } = useUpdateAlertStatus();

  if (isLoading) return <div className="text-gray-500">Loading alert…</div>;
  if (!alert)   return <div className="text-gray-500">Alert not found.</div>;

  const breakdown = alert.score_breakdown as Record<string, number>;
  const tx = alert.transaction_details;

  return (
    <div className="max-w-4xl space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/alerts" className="text-gray-500 hover:text-gray-300 text-sm">← Alerts</Link>
        <h1 className="text-xl font-bold text-white">Alert Detail</h1>
        <RiskBadge level={alert.risk_level} score={alert.transaction_risk_score} />
      </div>

      {/* ── Alert Header ───────────────────────────────────────────── */}
      <div className="rounded-xl bg-gray-900 border border-gray-800 p-5 space-y-3">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Alert Info</h2>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <Field label="Alert ID"       value={alert.alert_id} mono />
          <Field label="Transaction ID" value={alert.transaction_id} mono />
          <Field label="Sender"         value={alert.sender_account} mono />
          <Field label="Community"      value={`#${alert.community_id}`} />
          <Field label="Risk Score"     value={`${alert.transaction_risk_score.toFixed(1)}`} />
          <Field label="Group Risk"     value={`${alert.group_risk_score.toFixed(1)}`} />
          <Field label="Created"        value={formatTimestamp(alert.created_at)} />
          <Field label="Status"         value={alert.alert_status} />
        </div>
        <p className="text-sm text-gray-300 bg-gray-800 rounded p-3">{alert.explanation}</p>
      </div>

      {/* ── Transaction Details ────────────────────────────────────── */}
      {tx ? (
        <TransactionDetailsCard tx={tx} />
      ) : (
        <div className="rounded-xl bg-gray-900 border border-gray-800 p-5">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Transaction Details
          </h2>
          <p className="text-xs text-gray-600">
            Not available — alert was created before transaction embedding was added.
          </p>
        </div>
      )}

      {/* ── Score breakdown ────────────────────────────────────────── */}
      <div className="rounded-xl bg-gray-900 border border-gray-800 p-5">
        <h2 className="text-sm font-semibold text-gray-400 mb-3">Score Breakdown</h2>
        <ScoreBreakdown breakdown={{
          rule:        breakdown.rule        ?? 0,
          behavioral:  breakdown.behavioral  ?? 0,
          gnn:         breakdown.gnn         ?? 0,
          graph_boost: breakdown.graph_boost ?? 0,
        }} />
      </div>

      {/* ── Patterns / drivers ─────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4">
        <ListCard title="Triggered Patterns" items={alert.triggered_patterns} color="text-red-400" />
        <ListCard title="Anomaly Drivers"    items={alert.anomaly_drivers}    color="text-orange-400" />
      </div>

      {/* ── Rule explanations ──────────────────────────────────────── */}
      {alert.rule_explanations.length > 0 && (
        <ListCard title="Rule Explanations" items={alert.rule_explanations} color="text-yellow-300" />
      )}

      {/* ── Structural explanations ────────────────────────────────── */}
      {alert.structural_anomaly_explanations.length > 0 && (
        <ListCard
          title="Structural Anomalies"
          items={alert.structural_anomaly_explanations}
          color="text-purple-400"
        />
      )}

      {/* ── Status update ──────────────────────────────────────────── */}
      <div className="rounded-xl bg-gray-900 border border-gray-800 p-5">
        <h2 className="text-sm font-semibold text-gray-400 mb-3">Update Status</h2>
        <div className="flex gap-2 flex-wrap">
          {["Open", "Investigating", "SAR_Filed", "Closed"].map((s) => (
            <button
              key={s}
              disabled={isPending || alert.alert_status === s}
              onClick={() => updateStatus({ alertId: alert.alert_id, status: s })}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
                ${alert.alert_status === s
                  ? "bg-blue-800 text-blue-200 cursor-default"
                  : "bg-gray-800 text-gray-300 hover:bg-gray-700"}`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Transaction Details Card ──────────────────────────────────────────────────

function TransactionDetailsCard({ tx }: { tx: TransactionDetails }) {
  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-5 space-y-4">
      <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Transaction Details
      </h2>

      {/* Money movement */}
      <div className="flex items-center gap-3 bg-gray-800 rounded-lg p-3">
        <div className="flex-1 text-center">
          <div className="text-xs text-gray-500">Sender</div>
          <div className="font-mono text-sm text-gray-200 mt-0.5">{tx.sender_account}</div>
          <div className="text-xs text-gray-500">{tx.sender_name}</div>
          <div className="text-xs text-gray-600">{tx.sender_bank} · {tx.sender_country}</div>
        </div>
        <div className="text-center px-4">
          <div className="text-lg font-bold text-green-400">
            {formatCurrency(tx.amount)}
          </div>
          <div className="text-xs text-gray-500">{tx.currency}</div>
          <div className="text-xs text-blue-400 mt-0.5">{tx.transaction_type}</div>
          <div className="text-gray-600 text-xs mt-1">→</div>
        </div>
        <div className="flex-1 text-center">
          <div className="text-xs text-gray-500">Receiver</div>
          <div className="font-mono text-sm text-gray-200 mt-0.5">{tx.receiver_account}</div>
          <div className="text-xs text-gray-500">{tx.receiver_name}</div>
          <div className="text-xs text-gray-600">{tx.receiver_bank} · {tx.receiver_country}</div>
        </div>
      </div>

      {/* Two-column detail grid */}
      <div className="grid grid-cols-3 gap-x-6 gap-y-3 text-sm">
        <Field label="Timestamp"          value={formatTimestamp(tx.timestamp)} />
        <Field label="Payment Channel"    value={tx.payment_channel} />
        <Field label="Status"             value={tx.transaction_status} />

        <Field label="Device ID"          value={tx.device_id} mono />
        <Field label="IP Address"         value={tx.ip_address} mono />
        <Field label="Merchant Category"  value={tx.merchant_category || "—"} />

        <Field label="Geo"
               value={`${tx.geo_latitude.toFixed(4)}, ${tx.geo_longitude.toFixed(4)}`}
               mono />
        <Field label="International"      value={tx.is_international ? "Yes" : "No"} />
        <Field label="KYC Level"          value={String(tx.kyc_level)} />

        <Field label="Sender Balance Before"   value={formatCurrency(tx.sender_balance_before)} />
        <Field label="Sender Balance After"    value={formatCurrency(tx.sender_balance_after)} />
        <Field label="Balance Change"
               value={formatCurrency(tx.sender_balance_after - tx.sender_balance_before)}
        />

        <Field label="Receiver Balance Before" value={formatCurrency(tx.receiver_balance_before)} />
        <Field label="Receiver Balance After"  value={formatCurrency(tx.receiver_balance_after)} />
        <Field label="Leading Digit (Benford)" value={String(tx.amount_leading_digit)} />
      </div>

      {tx.remarks && (
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Remarks</div>
          <p className="text-xs text-gray-400 bg-gray-800 rounded p-2">{tx.remarks}</p>
        </div>
      )}
    </div>
  );
}

// ── Shared helpers ────────────────────────────────────────────────────────────

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className={`mt-0.5 text-gray-200 ${mono ? "font-mono text-xs" : "text-sm"}`}>
        {value}
      </div>
    </div>
  );
}

function ListCard({ title, items, color }: { title: string; items: string[]; color: string }) {
  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
      <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">{title}</h2>
      <ul className="space-y-1">
        {items.length === 0 ? (
          <li className="text-xs text-gray-600">None</li>
        ) : (
          items.map((item, i) => (
            <li key={i} className={`text-xs ${color}`}>• {item}</li>
          ))
        )}
      </ul>
    </div>
  );
}
