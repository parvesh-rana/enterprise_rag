import type { HealthResponse, QueryRequest, QueryResponse, SourceResponse } from "./types";
import { ApiError } from "./types";

const API_BASE = "/api";

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};

  if (!response.ok) {
    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : `Request failed with status ${response.status}`;
    throw new ApiError(detail, response.status, payload.request_id);
  }

  return payload as T;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, { signal });
  return readJson<HealthResponse>(response);
}

export async function queryRag(
  body: QueryRequest,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return readJson<QueryResponse>(response);
}

export async function getSource(chunkId: string, signal?: AbortSignal): Promise<SourceResponse> {
  const response = await fetch(`${API_BASE}/sources/${encodeURIComponent(chunkId)}`, { signal });
  return readJson<SourceResponse>(response);
}
