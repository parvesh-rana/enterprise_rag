import type { ReactNode } from "react";
import HealthBadge from "./HealthBadge";
import { Database } from "lucide-react";

interface Props {
  children: ReactNode;
}

export default function Layout({ children }: Props) {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="sticky top-0 z-40 bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database size={22} className="text-blue-600" />
            <h1 className="text-lg font-semibold text-gray-900">
              Enterprise RAG
            </h1>
            <span className="text-xs text-gray-400 ml-2 hidden sm:inline">
              SEC 10-K Filing Intelligence
            </span>
          </div>
          <HealthBadge />
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-4 py-6">{children}</main>
    </div>
  );
}
