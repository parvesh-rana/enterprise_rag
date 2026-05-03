import { useState, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { postQuery } from "../api/client";
import type { QueryRequest, QueryResponse } from "../api/types";
import QueryInput from "../components/QueryInput";
import AnswerCard from "../components/AnswerCard";
import ChunkList from "../components/ChunkList";
import SourceModal from "../components/SourceModal";
import QueryHistory, { saveToHistory } from "../components/QueryHistory";
import { AlertTriangle, FileSearch } from "lucide-react";

const EXAMPLE_QUERIES: QueryRequest[] = [
  { question: "What are Apple's main risk factors?" },
  { question: "Describe Microsoft's revenue recognition policies", company: "MSFT" },
  { question: "What legal proceedings is Tesla facing?", company: "TSLA", item: "3" },
  { question: "Summarize Amazon's business overview", company: "AMZN", item: "1" },
];

export default function QueryPage() {
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [sourceModal, setSourceModal] = useState<string | null>(null);
  const [highlightChunk, setHighlightChunk] = useState<string | null>(null);
  const [, setHistoryTick] = useState(0);

  const mutation = useMutation({
    mutationFn: postQuery,
    onSuccess: (data) => {
      setResult(data);
      setHighlightChunk(null);
    },
  });

  const handleSubmit = useCallback(
    (req: QueryRequest) => {
      saveToHistory(req);
      setHistoryTick((t) => t + 1);
      mutation.mutate(req);
    },
    [mutation]
  );

  const handleCitationClick = useCallback((chunkId: string) => {
    setHighlightChunk(chunkId);
    const el = document.getElementById(`chunk-${chunkId}`);
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  const handleChunkClick = useCallback((chunkId: string) => {
    setSourceModal(chunkId);
  }, []);

  return (
    <div className="space-y-4">
      <QueryInput onSubmit={handleSubmit} isLoading={mutation.isPending} />

      {!result && !mutation.isPending && !mutation.isError && (
        <div className="space-y-4">
          <QueryHistory onSelect={handleSubmit} />

          <div className="text-center py-12">
            <FileSearch size={48} className="mx-auto text-gray-300 mb-4" />
            <h2 className="text-lg font-medium text-gray-600 mb-2">
              Ask about SEC 10-K Filings
            </h2>
            <p className="text-sm text-gray-400 mb-6 max-w-md mx-auto">
              Query financial filings using natural language. The system retrieves relevant
              passages and generates grounded answers with citations.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {EXAMPLE_QUERIES.map((eq, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSubmit(eq)}
                  className="text-xs bg-white border border-gray-200 text-gray-600 px-3 py-1.5 rounded-full hover:border-blue-300 hover:text-blue-600 transition-colors"
                >
                  {eq.question}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {mutation.isPending && (
        <div className="space-y-3">
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 animate-pulse">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-5 h-5 bg-gray-200 rounded-full" />
              <div className="w-24 h-4 bg-gray-200 rounded" />
            </div>
            <div className="space-y-2">
              <div className="h-3 bg-gray-100 rounded w-full" />
              <div className="h-3 bg-gray-100 rounded w-5/6" />
              <div className="h-3 bg-gray-100 rounded w-4/6" />
              <div className="h-3 bg-gray-100 rounded w-full" />
              <div className="h-3 bg-gray-100 rounded w-3/6" />
            </div>
          </div>
          <div className="grid gap-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-white rounded-lg border border-gray-200 p-3 animate-pulse">
                <div className="flex gap-2 mb-2">
                  <div className="w-5 h-5 bg-gray-200 rounded-full" />
                  <div className="w-12 h-4 bg-gray-200 rounded" />
                  <div className="w-8 h-4 bg-gray-200 rounded" />
                </div>
                <div className="h-2.5 bg-gray-100 rounded w-full" />
                <div className="h-2.5 bg-gray-100 rounded w-3/4 mt-1.5" />
              </div>
            ))}
          </div>
        </div>
      )}

      {mutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle size={18} className="text-red-500 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-red-800">Query failed</p>
            <p className="text-sm text-red-600 mt-0.5">
              {(mutation.error as Error)?.message || "An unexpected error occurred"}
            </p>
          </div>
        </div>
      )}

      {result && !mutation.isPending && (
        <div className="space-y-4">
          <AnswerCard data={result} onCitationClick={handleCitationClick} />
          <ChunkList
            chunks={result.retrieved}
            onChunkClick={handleChunkClick}
            highlightId={highlightChunk}
          />
        </div>
      )}

      {sourceModal && (
        <SourceModal chunkId={sourceModal} onClose={() => setSourceModal(null)} />
      )}
    </div>
  );
}
