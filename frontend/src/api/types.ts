export interface QueryRequest {
  question: string;
  company?: string;
  year?: number;
  item?: string;
  top_k?: number;
  use_reranker?: boolean;
}

export interface Citation {
  chunk_id: string;
  score: number;
}

export interface RetrievedChunkOut {
  chunk_id: string;
  score: number;
  company: string;
  year: number;
  item: string;
  section_title: string;
  text_preview: string;
  source_url: string;
}

export interface QueryResponse {
  request_id: string;
  answer: string;
  model: string;
  citations: Citation[];
  retrieved: RetrievedChunkOut[];
  timings_ms: Record<string, number>;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  qdrant: boolean;
  bm25: boolean;
  llm_provider: string;
  embedding_model: string;
}

export interface SourceResponse {
  chunk_id: string;
  company: string;
  company_name: string;
  year: number;
  item: string;
  section_title: string;
  text: string;
  source_url: string;
}

export interface ErrorResponse {
  request_id: string;
  detail: string;
}

export const ITEM_KEYS = [
  "1", "1A", "1B", "1C", "2", "3", "4",
  "5", "6", "7", "7A", "8", "9", "9A", "9B",
  "10", "11", "12", "13", "14", "15",
] as const;

export type ItemKey = (typeof ITEM_KEYS)[number];
