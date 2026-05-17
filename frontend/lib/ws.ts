// WebSocket client with auto-reconnect for live event feed

import type { ScoredEvent } from "./types";

// WebSocket must use the full wss:// URL (can't proxy through Next.js rewrites).
// NEXT_PUBLIC_WS_URL is set in .env.production / Vercel env vars.
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";

export type WSMessage = ScoredEvent;

export interface WSOptions {
  onMessage: (event: WSMessage) => void;
  onError?: (err: Event) => void;
  onOpen?: () => void;
  onClose?: () => void;
  reconnectDelayMs?: number;
}

export class AMLWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  private opts: WSOptions;

  constructor(opts: WSOptions) {
    this.opts = opts;
  }

  connect(): void {
    this.stopped = false;
    this._open();
  }

  disconnect(): void {
    this.stopped = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  send(payload: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  private _open(): void {
    try {
      this.ws = new WebSocket(`${WS_BASE}/stream/ws`);
    } catch {
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.opts.onOpen?.();
    };

    this.ws.onmessage = (e) => {
      try {
        const msg: WSMessage = JSON.parse(e.data as string);
        this.opts.onMessage(msg);
      } catch {
        // ignore parse errors
      }
    };

    this.ws.onerror = (err) => {
      this.opts.onError?.(err);
    };

    this.ws.onclose = () => {
      this.opts.onClose?.();
      if (!this.stopped) {
        this._scheduleReconnect();
      }
    };
  }

  private _scheduleReconnect(): void {
    const delay = this.opts.reconnectDelayMs ?? 3000;
    this.reconnectTimer = setTimeout(() => {
      if (!this.stopped) {
        this._open();
      }
    }, delay);
  }
}
