import type { CommunityProfile } from "@/lib/types";
import { riskColor } from "@/lib/utils";

interface Props {
  communities: CommunityProfile[];
}

export function CommunityLegend({ communities }: Props) {
  const top = communities.slice(0, 8);
  return (
    <div className="space-y-1.5">
      {top.map((c) => {
        const level = c.risk_score >= 81 ? "Critical" : c.risk_score >= 61 ? "High" : c.risk_score >= 31 ? "Medium" : "Low";
        return (
          <div key={c.community_id} className="flex items-center gap-2 text-xs">
            <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-semibold ${riskColor(level)}`}>
              #{c.community_id}
            </span>
            <span className="text-gray-400 flex-1">{c.dominant_pattern ?? "—"}</span>
            <span className="text-gray-600">{c.size} nodes</span>
          </div>
        );
      })}
    </div>
  );
}
