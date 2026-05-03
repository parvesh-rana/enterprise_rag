import { useState } from "react";
import { ITEM_KEYS } from "../api/types";
import type { QueryRequest } from "../api/types";
import { Search, SlidersHorizontal, Loader2 } from "lucide-react";

interface Props {
  onSubmit: (req: QueryRequest) => void;
  isLoading: boolean;
}

export default function QueryInput({ onSubmit, isLoading }: Props) {
  const [question, setQuestion] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [company, setCompany] = useState("");
  const [year, setYear] = useState("");
  const [item, setItem] = useState("");
  const [topK, setTopK] = useState(5);
  const [useReranker, setUseReranker] = useState(true);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;
    const req: QueryRequest = {
      question: question.trim(),
      top_k: topK,
      use_reranker: useReranker,
    };
    if (company.trim()) req.company = company.trim().toUpperCase();
    if (year) req.year = parseInt(year);
    if (item) req.item = item;
    onSubmit(req);
  };

  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: currentYear - 1990 + 1 }, (_, i) => currentYear - i);

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about SEC 10-K filings... e.g. What are Apple's main risk factors?"
            className="w-full pl-10 pr-4 py-2.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          />
        </div>
        <button
          type="submit"
          disabled={!question.trim() || isLoading}
          className="px-4 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
        >
          {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
          {isLoading ? "Searching..." : "Search"}
        </button>
      </div>

      <div className="mt-2">
        <button
          type="button"
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
        >
          <SlidersHorizontal size={14} />
          {showFilters ? "Hide" : "Show"} Filters
        </button>

        {showFilters && (
          <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-3">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Company (ticker)</label>
              <input
                type="text"
                value={company}
                onChange={(e) => setCompany(e.target.value)}
                placeholder="e.g. AAPL"
                className="w-full px-2.5 py-1.5 border border-gray-200 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Year</label>
              <select
                value={year}
                onChange={(e) => setYear(e.target.value)}
                className="w-full px-2.5 py-1.5 border border-gray-200 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Any</option>
                {years.map((y) => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">10-K Item</label>
              <select
                value={item}
                onChange={(e) => setItem(e.target.value)}
                className="w-full px-2.5 py-1.5 border border-gray-200 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Any</option>
                {ITEM_KEYS.map((k) => (
                  <option key={k} value={k}>Item {k}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Top K: {topK}</label>
              <input
                type="range"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-full h-1.5 mt-2 accent-blue-600"
              />
            </div>
            <div className="flex items-end">
              <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useReranker}
                  onChange={(e) => setUseReranker(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                Reranker
              </label>
            </div>
          </div>
        )}
      </div>
    </form>
  );
}
