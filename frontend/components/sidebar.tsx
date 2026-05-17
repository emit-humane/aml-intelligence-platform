"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/",          label: "Dashboard",  icon: "🏠" },
  { href: "/alerts",    label: "Alerts",     icon: "🔔" },
  { href: "/graph",     label: "Graph",      icon: "🕸" },
  { href: "/analytics", label: "Analytics",  icon: "📊" },
  { href: "/upload",    label: "Upload",     icon: "📤" },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="fixed left-0 top-0 h-screen w-56 bg-gray-900 border-r border-gray-800 flex flex-col">
      <div className="px-4 py-5 border-b border-gray-800">
        <div className="text-sm font-bold text-blue-400">AML Intelligence</div>
        <div className="text-xs text-gray-500 mt-0.5">Platform v1.0</div>
      </div>
      <nav className="flex-1 px-2 py-4 space-y-1">
        {NAV.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
              pathname === item.href
                ? "bg-blue-900/50 text-blue-300 font-medium"
                : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
            )}
          >
            <span>{item.icon}</span>
            {item.label}
          </Link>
        ))}
      </nav>
      <div className="px-4 py-3 border-t border-gray-800">
        <div className="text-xs text-gray-600">iDEA 2.0 Hackathon</div>
        <div className="text-xs text-gray-700">PS3 Fund Flow Tracking</div>
      </div>
    </aside>
  );
}
