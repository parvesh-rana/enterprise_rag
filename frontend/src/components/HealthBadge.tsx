import { useEffect, useState } from "react";
import { getHealth } from "../api/client";
import type { HealthResponse } from "../api/types";
import { Activity } from "lucide-react";

export default function HealthBadge() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      try {
        const data = await getHealth();
        if (mounted) {
          setHealth(data);
          setError(false);
        }
      } catch {
        if (mounted) setError(true);
      }
    };
    poll();
    const id = setInterval(poll, 30_000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  const color = error
    ? "text-red-500"
    : health?.status === "ok"
    ? "text-green-500"
    : "text-yellow-500";

  const label = error
    ? "Backend unreachable"
    : health
    ? `${health.status === "ok" ? "All systems operational" : "Degraded"} · ${health.llm_provider} · ${health.embedding_model}`
    : "Checking...";

  return (
    <div className="relative group flex items-center gap-1.5 cursor-default">
      <Activity size={16} className={color} />
      <span className={`text-xs font-medium ${color}`}>
        {health?.status ?? (error ? "error" : "...")}
      </span>
      <div className="absolute top-full right-0 mt-2 w-72 bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-xs text-gray-700 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
        <p className="font-semibold mb-1">{label}</p>
        {health && (
          <ul className="space-y-0.5">
            <li>Qdrant: {health.qdrant ? "✓" : "✗"}</li>
            <li>BM25: {health.bm25 ? "✓" : "✗"}</li>
            <li>Provider: {health.llm_provider}</li>
            <li>Embedding: {health.embedding_model}</li>
          </ul>
        )}
      </div>
    </div>
  );
}
