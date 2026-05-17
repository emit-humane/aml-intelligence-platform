import { cn, riskColor } from "@/lib/utils";
import type { RiskLevel } from "@/lib/types";

interface Props {
  level: RiskLevel;
  score?: number;
  className?: string;
}

export function RiskBadge({ level, score, className }: Props) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold",
        riskColor(level),
        className
      )}
    >
      {level}
      {score !== undefined && (
        <span className="opacity-80">({score.toFixed(0)})</span>
      )}
    </span>
  );
}
