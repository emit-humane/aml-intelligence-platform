"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { AMLWebSocket } from "@/lib/ws";
import type { ScoredEvent } from "@/lib/types";

const MAX_EVENTS = 200;  // keep last N events in memory

export function useStream() {
  const [events, setEvents] = useState<ScoredEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const wsRef = useRef<AMLWebSocket | null>(null);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    const ws = new AMLWebSocket({
      onMessage: (evt) => {
        if (pausedRef.current) return;
        setEvents((prev) => [evt, ...prev].slice(0, MAX_EVENTS));
      },
      onOpen:  () => setConnected(true),
      onClose: () => setConnected(false),
      onError: () => setConnected(false),
    });
    wsRef.current = ws;
    ws.connect();

    return () => {
      ws.disconnect();
    };
  }, []);

  const togglePause = useCallback(() => setPaused((p) => !p), []);
  const clearEvents = useCallback(() => setEvents([]), []);

  // Expose send so stream-controls can inject a transaction manually
  const sendTransaction = useCallback((payload: Record<string, unknown>) => {
    wsRef.current?.send(payload);
  }, []);

  return { events, connected, paused, togglePause, clearEvents, sendTransaction };
}
