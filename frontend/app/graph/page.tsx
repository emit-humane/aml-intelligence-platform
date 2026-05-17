"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchGraphStats } from "@/lib/api";
import { useSubgraph, useNodeInfo } from "@/hooks/use-subgraph";
import { CytoscapeGraph } from "@/components/graph/cytoscape-graph";

export default function GraphPage() {
  const [inputValue, setInputValue] = useState("");
  const [accountId, setAccountId] = useState<string | null>(null);
  const [hops, setHops] = useState<1 | 2>(2);

  const { data: stats } = useQuery({
    queryKey: ["graph-stats"],
    queryFn: fetchGraphStats,
    refetchInterval: 15_000,
  });

  const { data: subgraph, isLoading: sgLoading, error: sgError } = useSubgraph(accountId, hops);
  const { data: nodeInfo } = useNodeInfo(accountId);

  function handleExplore() {
    const id = inputValue.trim();
    if (id) setAccountId(id);
  }

  function handleNodeClick(id: string) {
    setInputValue(id);
    setAccountId(id);
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Graph Explorer</h1>
        {stats && (
          <div className="text-sm text-gray-500">
            {stats.num_nodes.toLocaleString()} nodes ·{" "}
            {stats.num_edges.toLocaleString()} edges
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex gap-3 items-center">
        <input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleExplore()}
          placeholder="Enter account ID (e.g. ACC000001)…"
          className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-sm text-gray-200
                     focus:outline-none focus:border-blue-500 font-mono"
        />
        <div className="flex items-center gap-1 bg-gray-900 border border-gray-700 rounded-lg px-1 py-1">
          {([1, 2] as const).map((h) => (
            <button
              key={h}
              onClick={() => setHops(h)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                hops === h
                  ? "bg-blue-700 text-white"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {h}-hop
            </button>
          ))}
        </div>
        <button
          onClick={handleExplore}
          disabled={!inputValue.trim()}
          className="px-5 py-2 bg-blue-700 hover:bg-blue-600 disabled:opacity-40
                     text-white rounded-lg text-sm font-medium transition-colors"
        >
          Explore
        </button>
        {accountId && (
          <button
            onClick={() => { setAccountId(null); setInputValue(""); }}
            className="px-3 py-2 bg-gray-800 hover:bg-gray-700 text-gray-400
                       rounded-lg text-sm transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {/* Loading / error */}
      {sgLoading && (
        <div className="text-gray-500 text-sm animate-pulse">Loading subgraph…</div>
      )}
      {sgError && (
        <div className="text-red-400 text-sm">
          Account not found or graph unavailable.
        </div>
      )}

      {/* Main graph */}
      <CytoscapeGraph
        data={subgraph}
        onNodeClick={handleNodeClick}
        height={520}
      />

      {/* Node detail panel */}
      {nodeInfo && accountId && (
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2 rounded-xl bg-gray-900 border border-gray-800 p-5">
            <h2 className="text-sm font-semibold text-gray-400 mb-3">
              Node: <span className="font-mono text-white">{accountId}</span>
            </h2>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
              {Object.entries(nodeInfo as unknown as Record<string, unknown>)
                .filter(([, v]) => v !== null && v !== undefined)
                .map(([k, v]) => (
                  <div key={k} className="flex gap-2">
                    <dt className="text-gray-500 w-36 shrink-0 text-xs uppercase">{k}</dt>
                    <dd className="text-gray-300 font-mono text-xs">{String(v)}</dd>
                  </div>
                ))}
            </dl>
          </div>

          {subgraph && (
            <div className="rounded-xl bg-gray-900 border border-gray-800 p-5">
              <h2 className="text-sm font-semibold text-gray-400 mb-3">Subgraph Summary</h2>
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-gray-500 text-xs">Nodes in view</dt>
                  <dd className="text-gray-300">{subgraph.nodes.length}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500 text-xs">Edges in view</dt>
                  <dd className="text-gray-300">{subgraph.edges.length}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500 text-xs">Hop depth</dt>
                  <dd className="text-gray-300">{subgraph.hops}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500 text-xs">Center</dt>
                  <dd className="text-gray-300 font-mono text-xs truncate">{subgraph.center}</dd>
                </div>
              </dl>
            </div>
          )}
        </div>
      )}

      {!accountId && (
        <div className="text-center py-12 text-gray-600 text-sm">
          Enter an account ID above and press Explore.
          <br />
          Click any node in the graph to navigate to it.
        </div>
      )}
    </div>
  );
}
