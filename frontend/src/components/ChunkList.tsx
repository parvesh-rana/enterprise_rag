import type { RetrievedChunkOut } from "../api/types";
import { FileText, ExternalLink } from "lucide-react";

interface Props {
  chunks: RetrievedChunkOut[];
  onChunkClick: (chunkId: string) => void;
  highlightId?: string | null;
}

export default function ChunkList({ chunks, onChunkClick, highlightId }: Props) {
  if (chunks.length === 0) return null;

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
        <FileText size={16} className="text-gray-400" />
        Retrieved Sources ({chunks.length})
      </h3>
      <div className="grid gap-2">
        {chunks.map((chunk, idx) => (
          <div
            key={chunk.chunk_id}
            id={`chunk-${chunk.chunk_id}`}
            onClick={() => onChunkClick(chunk.chunk_id)}
            className={`bg-white rounded-lg border p-3 cursor-pointer hover:border-blue-300 hover:shadow-sm transition-all ${
              highlightId === chunk.chunk_id
                ? "border-blue-400 ring-2 ring-blue-100"
                : "border-gray-200"
            }`}
          >
            <div className="flex items-start justify-between gap-2 mb-1.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-bold bg-blue-100 text-blue-700 rounded-full">
                  {idx + 1}
                </span>
                <span className="text-xs font-medium bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded">
                  {chunk.company}
                </span>
                <span className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">
                  {chunk.year}
                </span>
                <span className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">
                  Item {chunk.item}
                </span>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span className="text-xs font-mono text-gray-400">
                  {chunk.score.toFixed(3)}
                </span>
                {chunk.source_url && (
                  <a
                    href={chunk.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="text-gray-400 hover:text-blue-500"
                    title="Open SEC filing"
                  >
                    <ExternalLink size={14} />
                  </a>
                )}
              </div>
            </div>
            {chunk.section_title && (
              <p className="text-xs font-medium text-gray-600 mb-1">
                {chunk.section_title}
              </p>
            )}
            <p className="text-xs text-gray-500 leading-relaxed line-clamp-3">
              {chunk.text_preview}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
