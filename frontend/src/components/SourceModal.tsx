import { useEffect, useState } from "react";
import { getSource } from "../api/client";
import type { SourceResponse } from "../api/types";
import { X, Loader2, ExternalLink } from "lucide-react";

interface Props {
  chunkId: string;
  onClose: () => void;
}

export default function SourceModal({ chunkId, onClose }: Props) {
  const [source, setSource] = useState<SourceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    getSource(chunkId)
      .then((data) => {
        if (mounted) setSource(data);
      })
      .catch((err) => {
        if (mounted) setError(err.message || "Failed to load source");
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [chunkId]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative bg-white rounded-xl shadow-xl w-full max-w-3xl max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div>
            <h2 className="text-sm font-semibold text-gray-800">Source Document</h2>
            <p className="text-xs text-gray-400 mt-0.5 font-mono">{chunkId}</p>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded-md transition-colors">
            <X size={18} className="text-gray-500" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {loading && (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={24} className="animate-spin text-blue-500" />
            </div>
          )}
          {error && (
            <div className="text-center py-12">
              <p className="text-sm text-red-500">{error}</p>
            </div>
          )}
          {source && (
            <div>
              <div className="flex flex-wrap gap-2 mb-4">
                <span className="text-xs font-medium bg-blue-50 text-blue-700 px-2 py-1 rounded">
                  {source.company} — {source.company_name}
                </span>
                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                  {source.year}
                </span>
                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                  Item {source.item}
                </span>
                {source.section_title && (
                  <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                    {source.section_title}
                  </span>
                )}
              </div>
              <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap bg-gray-50 rounded-lg p-4 border border-gray-100">
                {source.text}
              </div>
              {source.source_url && (
                <a
                  href={source.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 mt-3 text-xs text-blue-600 hover:text-blue-700"
                >
                  <ExternalLink size={12} />
                  View original SEC filing
                </a>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
