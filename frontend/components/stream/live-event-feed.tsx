"use client";

import { AnimatePresence, motion } from "framer-motion";
import { RiskBadge } from "@/components/alerts/risk-badge";
import { truncateId, formatTimestamp } from "@/lib/utils";
import type { ScoredEvent } from "@/lib/types";

interface Props {
  events: ScoredEvent[];
}

export function LiveEventFeed({ events }: Props) {
  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-600 text-sm">
        No events yet — connect the stream.
      </div>
    );
  }

  return (
    <div className="space-y-1 max-h-[420px] overflow-y-auto pr-1">
      <AnimatePresence initial={false}>
        {events.map((evt) => (
          <motion.div
            key={evt.transaction_id}
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-3 rounded-lg bg-gray-900 px-3 py-2 border border-gray-800"
          >
            <RiskBadge level={evt.risk_level} score={evt.transaction_risk_score} className="shrink-0" />
            <span className="font-mono text-xs text-gray-400 shrink-0">
              {truncateId(evt.sender_account, 10)}
            </span>
            <span className="text-xs text-gray-500 truncate flex-1">
              {evt.explanation}
            </span>
            {evt.alert && (
              <span className="text-xs font-semibold text-red-400 shrink-0">⚠ ALERT</span>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
