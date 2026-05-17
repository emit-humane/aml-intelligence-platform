"use client";

import { useEffect, useRef, useCallback } from "react";
import type { SubgraphData } from "@/lib/api";

interface Props {
  data: SubgraphData | null | undefined;
  /** Called when user clicks a non-center node */
  onNodeClick?: (accountId: string) => void;
  height?: number;
  /** Set of account IDs that form suspicious/cycle paths — highlighted red */
  suspiciousNodes?: Set<string>;
}

// Risk color mapping based on degree centrality (higher degree = more suspicious)
function nodeColor(node: SubgraphData["nodes"][0], isCenter: boolean, isSuspicious: boolean): string {
  if (isSuspicious) return "#ef4444";          // red — known suspicious
  if (isCenter)     return "#f97316";          // orange — selected account
  const degree = node.in_degree + node.out_degree;
  if (degree > 100)  return "#dc2626";         // red — high-degree hub
  if (degree > 40)   return "#f97316";         // orange — elevated
  if (degree > 15)   return "#facc15";         // yellow — moderate
  return "#3b82f6";                            // blue — normal
}

function nodeBorder(isCenter: boolean, isSuspicious: boolean): string {
  if (isSuspicious) return "#fca5a5";
  if (isCenter)     return "#fdba74";
  return "transparent";
}

export function CytoscapeGraph({
  data,
  onNodeClick,
  height = 520,
  suspiciousNodes = new Set(),
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef        = useRef<unknown>(null);

  const buildGraph = useCallback(async () => {
    if (!containerRef.current || !data || data.nodes.length === 0) return;

    const [cytoscapeModule, fcoseModule] = await Promise.all([
      import("cytoscape"),
      import("cytoscape-fcose"),
    ]);
    const cytoscape = cytoscapeModule.default;
    const fcose     = fcoseModule.default;

    // Register fcose only once
    try { cytoscape.use(fcose); } catch { /* already registered */ }

    // Destroy old instance
    if (cyRef.current) {
      (cyRef.current as { destroy: () => void }).destroy();
    }

    const maxWeight = Math.max(...data.edges.map((e) => e.weight), 1);

    const cy = cytoscape({
      container: containerRef.current,
      elements: {
        nodes: data.nodes.map((n) => ({
          data: {
            id:         n.id,
            label:      n.id.length > 10 ? n.id.slice(0, 10) : n.id,
            in_degree:  n.in_degree,
            out_degree: n.out_degree,
            is_center:  n.is_center,
            color:      nodeColor(n, n.is_center, suspiciousNodes.has(n.id)),
            border:     nodeBorder(n.is_center, suspiciousNodes.has(n.id)),
          },
        })),
        edges: data.edges.map((e, i) => ({
          data: {
            id:           `e${i}`,
            source:       e.source,
            target:       e.target,
            weight:       e.weight,
            // Normalised width 1–6
            edgeWidth:    1 + Math.round((e.weight / maxWeight) * 5),
          },
        })),
      },
      style: [
        {
          selector: "node",
          style: {
            label:              "data(label)",
            "font-size":        8,
            color:              "#d1d5db",
            "text-valign":      "bottom",
            "text-margin-y":    4,
            "background-color": "data(color)",
            "border-width":     2,
            "border-color":     "data(border)",
            width:              24,
            height:             24,
          },
        },
        {
          selector: "node[?is_center]",
          style: {
            width:  38,
            height: 38,
            "font-size": 10,
            "font-weight": "bold",
          },
        },
        {
          selector: "edge",
          style: {
            width:                "data(edgeWidth)",
            "line-color":         "#374151",
            "target-arrow-color": "#374151",
            "target-arrow-shape": "triangle",
            "curve-style":        "bezier",
            opacity:              0.7,
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 3,
            "border-color": "#60a5fa",
          },
        },
        {
          selector: "node.highlighted",
          style: {
            "background-color": "#ef4444",
            "border-color":     "#fca5a5",
            "border-width":     3,
          },
        },
      ],
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      layout: {
        name:                    "fcose",
        quality:                 "default",
        randomize:               true,
        animate:                 false,
        idealEdgeLength:         60,
        edgeElasticity:          0.45,
        nodeRepulsion:           4500,
        numIter:                 2500,
        tile:                    true,
        tilingPaddingVertical:   10,
        tilingPaddingHorizontal: 10,
      } as unknown as import("cytoscape").LayoutOptions,
      userZoomingEnabled:  true,
      userPanningEnabled:  true,
      minZoom: 0.1,
      maxZoom: 4,
    });

    // Click handler
    cy.on("tap", "node", (evt: { target: { id: () => string; data: (key: string) => unknown } }) => {
      const nodeId  = evt.target.id();
      const isCenter = evt.target.data("is_center");
      if (!isCenter && onNodeClick) {
        onNodeClick(nodeId);
      }
    });

    // Tooltip on hover
    cy.on("mouseover", "node", (evt: { target: { data: (key: string) => unknown; style: (key: string, val: string) => void } }) => {
      evt.target.style("opacity", "0.8");
      if (containerRef.current) {
        containerRef.current.title = `${evt.target.data("id")} (in=${evt.target.data("in_degree")} out=${evt.target.data("out_degree")})`;
      }
    });
    cy.on("mouseout", "node", (evt: { target: { style: (key: string, val: string) => void } }) => {
      evt.target.style("opacity", "1");
    });

    cyRef.current = cy;
  }, [data, onNodeClick, suspiciousNodes]);

  useEffect(() => {
    buildGraph();
    return () => {
      if (cyRef.current) {
        (cyRef.current as { destroy: () => void }).destroy();
        cyRef.current = null;
      }
    };
  }, [buildGraph]);

  if (!data || data.nodes.length === 0) {
    return (
      <div
        style={{ height, background: "#111827", borderRadius: 8 }}
        className="border border-gray-800 flex items-center justify-center"
      >
        <p className="text-gray-600 text-sm">
          Enter an account ID and click Explore to visualise the subgraph.
        </p>
      </div>
    );
  }

  return (
    <div className="relative">
      <div
        ref={containerRef}
        style={{ height, background: "#111827", borderRadius: 8 }}
        className="border border-gray-800"
      />
      {/* Legend */}
      <div className="absolute bottom-3 left-3 flex gap-3 bg-gray-900/90 border border-gray-800 rounded-lg px-3 py-2">
        {[
          { color: "#f97316", label: "Selected" },
          { color: "#dc2626", label: "High-degree hub" },
          { color: "#ef4444", label: "Suspicious" },
          { color: "#facc15", label: "Moderate" },
          { color: "#3b82f6", label: "Normal" },
        ].map((l) => (
          <div key={l.label} className="flex items-center gap-1.5">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ background: l.color }}
            />
            <span className="text-gray-400 text-xs">{l.label}</span>
          </div>
        ))}
      </div>
      {/* Node + edge count */}
      <div className="absolute top-3 right-3 text-xs text-gray-500 bg-gray-900/90 border border-gray-800 rounded px-2 py-1">
        {data.nodes.length} nodes · {data.edges.length} edges
      </div>
    </div>
  );
}
