import type { QueryRequest } from "../api/types";
import { History, X, Trash2 } from "lucide-react";

const STORAGE_KEY = "rag-query-history";
const MAX_HISTORY = 20;

export interface HistoryEntry {
  question: string;
  filters: Omit<QueryRequest, "question">;
  timestamp: number;
}

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveToHistory(req: QueryRequest) {
  const history = loadHistory();
  const { question, ...filters } = req;
  const entry: HistoryEntry = { question, filters, timestamp: Date.now() };
  // Remove duplicate questions
  const filtered = history.filter((h) => h.question !== question);
  const updated = [entry, ...filtered].slice(0, MAX_HISTORY);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
}

export function clearHistory() {
  localStorage.removeItem(STORAGE_KEY);
}

interface Props {
  onSelect: (req: QueryRequest) => void;
}

export default function QueryHistory({ onSelect }: Props) {
  const history = loadHistory();

  if (history.length === 0) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
          <History size={16} className="text-gray-400" />
          Recent Queries
        </h3>
        <button
          onClick={() => {
            clearHistory();
            window.dispatchEvent(new Event("history-updated"));
          }}
          className="text-xs text-gray-400 hover:text-red-500 flex items-center gap-1 transition-colors"
          title="Clear history"
        >
          <Trash2 size={12} />
          Clear
        </button>
      </div>
      <div className="space-y-1.5 max-h-48 overflow-y-auto">
        {history.map((entry, idx) => {
          const filterTags = [
            entry.filters.company,
            entry.filters.year?.toString(),
            entry.filters.item ? `Item ${entry.filters.item}` : null,
          ].filter(Boolean);

          return (
            <button
              key={idx}
              onClick={() => onSelect({ question: entry.question, ...entry.filters })}
              className="w-full text-left px-3 py-2 rounded-md hover:bg-gray-50 transition-colors group flex items-start justify-between"
            >
              <div className="min-w-0">
                <p className="text-sm text-gray-700 truncate">{entry.question}</p>
                {filterTags.length > 0 && (
                  <div className="flex gap-1 mt-0.5">
                    {filterTags.map((tag, j) => (
                      <span key={j} className="text-[10px] text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <X size={14} className="text-gray-300 group-hover:text-gray-400 shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
          );
        })}
      </div>
    </div>
  );
}
