"use client";

interface Props {
  connected: boolean;
  paused: boolean;
  eventCount: number;
  onTogglePause: () => void;
  onClear: () => void;
}

export function StreamControls({ connected, paused, eventCount, onTogglePause, onClear }: Props) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2 text-sm">
        <div className={`h-2 w-2 rounded-full ${connected ? "bg-green-400 animate-pulse" : "bg-red-500"}`} />
        <span className="text-gray-400">
          {connected ? "Live" : "Disconnected"}
        </span>
      </div>
      <span className="text-gray-600 text-sm">{eventCount} events</span>
      <button
        onClick={onTogglePause}
        disabled={!connected}
        className="px-3 py-1 rounded text-sm font-medium bg-gray-800 hover:bg-gray-700 text-gray-300 disabled:opacity-50 transition-colors"
      >
        {paused ? "▶ Resume" : "⏸ Pause"}
      </button>
      <button
        onClick={onClear}
        className="px-3 py-1 rounded text-sm font-medium bg-gray-800 hover:bg-gray-700 text-gray-400 transition-colors"
      >
        Clear
      </button>
    </div>
  );
}
