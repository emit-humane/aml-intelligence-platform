import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { RiskLevel } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function riskColor(level: RiskLevel): string {
  switch (level) {
    case "Critical": return "bg-red-600 text-white";
    case "High":     return "bg-orange-500 text-white";
    case "Medium":   return "bg-yellow-500 text-black";
    case "Low":      return "bg-green-500 text-white";
    default:         return "bg-gray-400 text-white";
  }
}

export function riskTextColor(level: RiskLevel): string {
  switch (level) {
    case "Critical": return "text-red-600";
    case "High":     return "text-orange-500";
    case "Medium":   return "text-yellow-600";
    case "Low":      return "text-green-600";
    default:         return "text-gray-500";
  }
}

export function scoreToRiskLevel(score: number): RiskLevel {
  if (score >= 81) return "Critical";
  if (score >= 61) return "High";
  if (score >= 31) return "Medium";
  return "Low";
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(amount);
}

export function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString("en-IN", {
    dateStyle: "short",
    timeStyle: "medium",
  });
}

export function truncateId(id: string, n = 12): string {
  return id.length > n ? id.slice(0, n) + "…" : id;
}
