import type { QueryRequest, QueryResponse, HealthResponse, SourceResponse } from "./types";

const BASE = "";

class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore parse errors
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export async function postQuery(req: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return handleResponse<QueryResponse>(res);
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  return handleResponse<HealthResponse>(res);
}

export async function getSource(chunkId: string): Promise<SourceResponse> {
  const res = await fetch(`${BASE}/sources/${encodeURIComponent(chunkId)}`);
  return handleResponse<SourceResponse>(res);
}

export { ApiError };
