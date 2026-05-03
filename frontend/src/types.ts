export type HealthResponse = {
  status: "ok" | "degraded";
  qdrant: boolean;
  bm25: boolean;
  llm_provider: string;
  embedding_model: string;
};

export type QueryRequest = {
  question: string;
  company?: string | null;
  year?: number | null;
  item?: string | null;
  top_k?: number | null;
  use_reranker: boolean;
};

export type Citation = {
  chunk_id: string;
  score: number;
  quote?: string | null;
};

export type RetrievedChunk = {
  chunk_id: string;
  score: number;
  company: string;
  year: number;
  item: string;
  section_title: string;
  text_preview: string;
  source_url: string;
};

export type QueryResponse = {
  request_id: string;
  answer: string;
  model: string;
  citations: Citation[];
  retrieved: RetrievedChunk[];
  timings_ms: Record<string, number>;
};

export type SourceResponse = {
  chunk_id: string;
  company: string;
  company_name: string;
  year: number;
  item: string;
  section_title: string;
  text: string;
  source_url: string;
};

export class ApiError extends Error {
  status: number;
  requestId?: string;

  constructor(message: string, status: number, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.requestId = requestId;
  }
}
