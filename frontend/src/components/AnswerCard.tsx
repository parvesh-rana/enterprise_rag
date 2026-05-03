import type { QueryResponse, Citation } from "../api/types";
import { Bot, Clock, Copy, Check } from "lucide-react";
import { useState } from "react";

interface Props {
  data: QueryResponse;
  onCitationClick: (chunkId: string) => void;
}

function renderAnswer(text: string, citations: Citation[], onCitationClick: (id: string) => void) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const idx = parseInt(match[1]) - 1;
      const citation = citations[idx];
      if (citation) {
        return (
          <button
            key={i}
            onClick={() => onCitationClick(citation.chunk_id)}
            className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-bold bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors mx-0.5 align-super"
            title={`Source: ${citation.chunk_id} (score: ${citation.score.toFixed(3)})`}
          >
            {match[1]}
          </button>
        );
      }
    }
    return <span key={i}>{part}</span>;
  });
}

export default function AnswerCard({ data, onCitationClick }: Props) {
  const [copied, setCopied] = useState(false);

  const copyAnswer = async () => {
    await navigator.clipboard.writeText(data.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Bot size={18} className="text-blue-600" />
          <span className="text-sm font-semibold text-gray-800">Answer</span>
          <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">
            {data.model}
          </span>
        </div>
        <button
          onClick={copyAnswer}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition-colors"
          title="Copy answer"
        >
          {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
        {renderAnswer(data.answer, data.citations, onCitationClick)}
      </div>

      {Object.keys(data.timings_ms).length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-100 flex flex-wrap gap-3">
          <Clock size={14} className="text-gray-400 mt-0.5" />
          {Object.entries(data.timings_ms).map(([stage, ms]) => (
            <span key={stage} className="text-xs text-gray-400">
              {stage}: <span className="font-medium text-gray-600">{ms.toFixed(0)}ms</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
