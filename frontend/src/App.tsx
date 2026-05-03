import {
  Activity,
  BookOpenText,
  CheckCircle2,
  Clock3,
  Database,
  ExternalLink,
  FileSearch,
  Gauge,
  Loader2,
  Search,
  Server,
  ShieldAlert,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getHealth, getSource, queryRag } from "./api";
import styles from "./App.module.css";
import type { HealthResponse, QueryRequest, QueryResponse, RetrievedChunk, SourceResponse } from "./types";
import { ApiError } from "./types";

const COMPANIES = ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA"];
const ITEMS = ["1", "1A", "1B", "1C", "2", "3", "5", "7", "7A", "8", "9A", "15"];

const SAMPLE_QUESTIONS = [
  "What does the company describe as its primary product lines?",
  "What supply chain risks are discussed in the filing?",
  "How does management describe research and development priorities?",
];

type SourceState =
  | { status: "idle"; data: null; error: "" }
  | { status: "loading"; data: null; error: "" }
  | { status: "ready"; data: SourceResponse; error: "" }
  | { status: "error"; data: null; error: string };

function describeError(error: unknown): string {
  if (error instanceof DOMException && error.name === "AbortError") {
    return "";
  }
  if (error instanceof ApiError) {
    if (error.status === 429) {
      return "The API rate limit was reached. Wait a minute and try again.";
    }
    if (error.status === 503) {
      return error.message || "A required backend service is unavailable.";
    }
    return error.message;
  }
  if (error instanceof TypeError) {
    return "Could not reach the API. Start the FastAPI backend on port 8000.";
  }
  return "Something went wrong while contacting the API.";
}

function formatScore(score: number): string {
  if (!Number.isFinite(score)) {
    return "0.000";
  }
  return score.toFixed(score >= 1 ? 2 : 3);
}

function normalizeAnswerText(text: string): string {
  return text.trim().replace(/\s+-\s+\*\*/g, "\n- **");
}

function renderInlineText(text: string) {
  const boldParts = text.split(/(\*\*[^*]+\*\*)/g);

  return boldParts.map((part, partIndex) => {
    const isBold = part.startsWith("**") && part.endsWith("**");
    const cleanPart = isBold ? part.slice(2, -2) : part;
    const citationParts = cleanPart.split(/(\[[^\]]+\])/g).filter(Boolean);
    const content = citationParts.map((piece, pieceIndex) => {
      if (piece.startsWith("[") && piece.endsWith("]")) {
        return (
          <span key={`${partIndex}-${pieceIndex}`} className={styles.inlineCitation}>
            {piece}
          </span>
        );
      }
      return <span key={`${partIndex}-${pieceIndex}`}>{piece}</span>;
    });

    if (isBold) {
      return <strong key={partIndex}>{content}</strong>;
    }
    return <span key={partIndex}>{content}</span>;
  });
}

function App() {
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [company, setCompany] = useState("");
  const [year, setYear] = useState("");
  const [item, setItem] = useState("");
  const [topK, setTopK] = useState(5);
  const [useReranker, setUseReranker] = useState(true);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState("");
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [queryError, setQueryError] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const [sourceState, setSourceState] = useState<SourceState>({
    status: "idle",
    data: null,
    error: "",
  });
  const [selectedChunk, setSelectedChunk] = useState<RetrievedChunk | null>(null);

  const queryAbortRef = useRef<AbortController | null>(null);
  const sourceAbortRef = useRef<AbortController | null>(null);

  const refreshHealth = useCallback(async () => {
    const controller = new AbortController();
    try {
      const data = await getHealth(controller.signal);
      setHealth(data);
      setHealthError("");
    } catch (error) {
      const message = describeError(error);
      if (message) {
        setHealthError(message);
      }
    }
    return () => controller.abort();
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  const requestBody = useMemo<QueryRequest>(
    () => ({
      question: question.trim(),
      company: company || null,
      year: year ? Number(year) : null,
      item: item || null,
      top_k: topK,
      use_reranker: useReranker,
    }),
    [company, item, question, topK, useReranker, year],
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (trimmed.length < 3) {
      setQueryError("Questions must be at least 3 characters.");
      return;
    }

    queryAbortRef.current?.abort();
    const controller = new AbortController();
    queryAbortRef.current = controller;
    setIsQuerying(true);
    setQueryError("");
    setQueryResult(null);
    setSelectedChunk(null);
    setSourceState({ status: "idle", data: null, error: "" });

    try {
      const data = await queryRag({ ...requestBody, question: trimmed }, controller.signal);
      setQueryResult(data);
    } catch (error) {
      const message = describeError(error);
      if (message) {
        setQueryError(message);
      }
    } finally {
      if (queryAbortRef.current === controller) {
        queryAbortRef.current = null;
      }
      setIsQuerying(false);
    }
  }

  async function openSource(chunk: RetrievedChunk) {
    sourceAbortRef.current?.abort();
    const controller = new AbortController();
    sourceAbortRef.current = controller;
    setSelectedChunk(chunk);
    setSourceState({ status: "loading", data: null, error: "" });

    try {
      const data = await getSource(chunk.chunk_id, controller.signal);
      setSourceState({ status: "ready", data, error: "" });
    } catch (error) {
      const message = describeError(error);
      if (message) {
        setSourceState({ status: "error", data: null, error: message });
      }
    } finally {
      if (sourceAbortRef.current === controller) {
        sourceAbortRef.current = null;
      }
    }
  }

  function closeSource() {
    sourceAbortRef.current?.abort();
    sourceAbortRef.current = null;
    setSelectedChunk(null);
    setSourceState({ status: "idle", data: null, error: "" });
  }

  const citationIds = new Set(queryResult?.citations.map((citation) => citation.chunk_id) ?? []);
  const statusLabel = health?.status ?? (healthError ? "offline" : "checking");

  return (
    <main className={styles.shell}>
      <section className={styles.header}>
        <div>
          <p className={styles.eyebrow}>SEC 10-K retrieval workspace</p>
          <h1>Enterprise RAG Analyst Console</h1>
          <p className={styles.subtitle}>
            Ask grounded filing questions, inspect retrieved evidence, and trace each answer back
            to its source chunk.
          </p>
        </div>
        <div className={styles.statusCard}>
          <div className={styles.statusTopline}>
            <span className={`${styles.statusDot} ${health?.status === "ok" ? styles.ok : ""}`} />
            <strong>{statusLabel}</strong>
            <button type="button" className={styles.iconButton} onClick={() => void refreshHealth()}>
              <Activity size={16} />
              <span>Refresh health</span>
            </button>
          </div>
          <div className={styles.healthGrid}>
            <HealthPill icon={<Database size={15} />} label="Qdrant" active={health?.qdrant} />
            <HealthPill icon={<FileSearch size={15} />} label="BM25" active={health?.bm25} />
            <HealthPill icon={<Server size={15} />} label={health?.llm_provider ?? "LLM"} active />
            <HealthPill icon={<Gauge size={15} />} label={health?.embedding_model ?? "Embedding"} active />
          </div>
          {healthError && <p className={styles.inlineError}>{healthError}</p>}
        </div>
      </section>

      <section className={styles.workspace}>
        <form className={styles.queryPanel} onSubmit={handleSubmit}>
          <div className={styles.panelTitle}>
            <Search size={18} />
            <span>Query</span>
          </div>
          <label className={styles.questionLabel} htmlFor="question">
            Question
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            className={styles.questionInput}
            rows={5}
            minLength={3}
            maxLength={2000}
          />
          <div className={styles.sampleRow}>
            {SAMPLE_QUESTIONS.map((sample) => (
              <button
                key={sample}
                type="button"
                className={styles.sampleButton}
                onClick={() => setQuestion(sample)}
              >
                {sample}
              </button>
            ))}
          </div>

          <div className={styles.filterHeader}>
            <SlidersHorizontal size={17} />
            <span>Filters</span>
          </div>
          <div className={styles.filters}>
            <label>
              Company
              <select value={company} onChange={(event) => setCompany(event.target.value)}>
                <option value="">All companies</option>
                {COMPANIES.map((ticker) => (
                  <option key={ticker} value={ticker}>
                    {ticker}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Filing year
              <input
                type="number"
                min="1990"
                max="2100"
                placeholder="Any"
                value={year}
                onChange={(event) => setYear(event.target.value)}
              />
            </label>
            <label>
              10-K item
              <select value={item} onChange={(event) => setItem(event.target.value)}>
                <option value="">All items</option>
                {ITEMS.map((itemKey) => (
                  <option key={itemKey} value={itemKey}>
                    Item {itemKey}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Top K
              <input
                type="number"
                min="1"
                max="20"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </label>
          </div>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={useReranker}
              onChange={(event) => setUseReranker(event.target.checked)}
            />
            <span>Cross-encoder reranker</span>
          </label>
          {queryError && (
            <div className={styles.errorBanner}>
              <ShieldAlert size={18} />
              <span>{queryError}</span>
            </div>
          )}
          <button type="submit" className={styles.primaryButton} disabled={isQuerying}>
            {isQuerying ? <Loader2 size={18} className={styles.spin} /> : <Search size={18} />}
            <span>{isQuerying ? "Retrieving evidence" : "Ask filings"}</span>
          </button>
        </form>

        <section className={styles.resultsPanel}>
          {!queryResult && !isQuerying && !queryError && <EmptyState />}
          {isQuerying && <LoadingState />}
          {queryResult && (
            <>
              <div className={styles.answerCard}>
                <div className={styles.answerMeta}>
                  <span>{queryResult.model}</span>
                  <span>Request {queryResult.request_id}</span>
                  <span>{queryResult.timings_ms.total?.toFixed(0) ?? "0"} ms total</span>
                </div>
                <h2>Answer</h2>
                <AnswerText text={queryResult.answer} />
                <div className={styles.citationRow}>
                  {queryResult.citations.length === 0 ? (
                    <span className={styles.muted}>No validated citations returned.</span>
                  ) : (
                    queryResult.citations.map((citation) => (
                      <span key={citation.chunk_id} className={styles.citationChip}>
                        {citation.chunk_id}
                      </span>
                    ))
                  )}
                </div>
              </div>

              <div className={styles.evidenceHeader}>
                <div>
                  <h2>Retrieved Evidence</h2>
                  <p>{queryResult.retrieved.length} chunks returned from hybrid retrieval.</p>
                </div>
                <div className={styles.timingGrid}>
                  <Metric label="Retrieval" value={`${queryResult.timings_ms.retrieval ?? 0} ms`} />
                  <Metric label="Generation" value={`${queryResult.timings_ms.generation ?? 0} ms`} />
                </div>
              </div>

              <div className={styles.evidenceList}>
                {queryResult.retrieved.map((chunk) => (
                  <article
                    key={chunk.chunk_id}
                    className={`${styles.evidenceCard} ${
                      citationIds.has(chunk.chunk_id) ? styles.citedEvidence : ""
                    }`}
                  >
                    <div className={styles.evidenceTopline}>
                      <div>
                        <span className={styles.chunkId}>{chunk.chunk_id}</span>
                        <span className={styles.chunkMeta}>
                          {chunk.company} / {chunk.year || "Unknown"} / Item {chunk.item || "N/A"}
                        </span>
                      </div>
                      <span className={styles.score}>Score {formatScore(chunk.score)}</span>
                    </div>
                    <h3>{chunk.section_title || "Untitled section"}</h3>
                    <p>{chunk.text_preview}</p>
                    <div className={styles.cardActions}>
                      <button type="button" onClick={() => void openSource(chunk)}>
                        <BookOpenText size={16} />
                        <span>Open source</span>
                      </button>
                      {chunk.source_url && (
                        <a href={chunk.source_url} target="_blank" rel="noreferrer">
                          <ExternalLink size={16} />
                          <span>SEC filing</span>
                        </a>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            </>
          )}
        </section>
      </section>

      {selectedChunk && (
        <aside className={styles.drawer} aria-label="Source detail">
          <div className={styles.drawerScrim} onClick={closeSource} />
          <div className={styles.drawerPanel}>
            <div className={styles.drawerHeader}>
              <div>
                <p className={styles.eyebrow}>Source chunk</p>
                <h2>{selectedChunk.chunk_id}</h2>
              </div>
              <button type="button" className={styles.iconOnlyButton} onClick={closeSource}>
                <X size={18} />
                <span>Close source drawer</span>
              </button>
            </div>
            {sourceState.status === "loading" && (
              <div className={styles.drawerState}>
                <Loader2 size={22} className={styles.spin} />
                <span>Loading source text</span>
              </div>
            )}
            {sourceState.status === "error" && (
              <div className={styles.errorBanner}>
                <ShieldAlert size={18} />
                <span>{sourceState.error}</span>
              </div>
            )}
            {sourceState.status === "ready" && (
              <>
                <div className={styles.sourceMeta}>
                  <span>{sourceState.data.company_name || sourceState.data.company}</span>
                  <span>{sourceState.data.year}</span>
                  <span>Item {sourceState.data.item}</span>
                </div>
                <h3>{sourceState.data.section_title}</h3>
                <p className={styles.sourceText}>{sourceState.data.text}</p>
                {sourceState.data.source_url && (
                  <a className={styles.sourceLink} href={sourceState.data.source_url} target="_blank" rel="noreferrer">
                    <ExternalLink size={16} />
                    <span>Open original filing</span>
                  </a>
                )}
              </>
            )}
          </div>
        </aside>
      )}
    </main>
  );
}

function HealthPill({
  icon,
  label,
  active,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
}) {
  return (
    <span className={`${styles.healthPill} ${active ? styles.healthActive : ""}`}>
      {icon}
      <span>{label}</span>
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className={styles.metric}>
      <Clock3 size={14} />
      <span>{label}</span>
      <strong>{value}</strong>
    </span>
  );
}

function AnswerText({ text }: { text: string }) {
  const blocks = normalizeAnswerText(text)
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);

  const allBullets = blocks.length > 1 && blocks.every((line) => line.startsWith("- "));
  if (allBullets) {
    return (
      <ul className={styles.answerList}>
        {blocks.map((line, index) => (
          <li key={`${line}-${index}`}>{renderInlineText(line.slice(2).trim())}</li>
        ))}
      </ul>
    );
  }

  return (
    <div className={styles.answerBody}>
      {blocks.map((line, index) => {
        if (line.startsWith("- ")) {
          return <p key={`${line}-${index}`}>{renderInlineText(line.slice(2).trim())}</p>;
        }
        return <p key={`${line}-${index}`}>{renderInlineText(line)}</p>;
      })}
    </div>
  );
}

function EmptyState() {
  return (
    <div className={styles.emptyState}>
      <FileSearch size={38} />
      <h2>No query run yet</h2>
      <p>
        Choose filters, ask a question, and the console will show the answer beside the evidence
        that supported it.
      </p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className={styles.emptyState}>
      <Loader2 size={38} className={styles.spin} />
      <h2>Searching filings</h2>
      <p>Running dense retrieval, BM25 search, fusion, reranking, and grounded generation.</p>
    </div>
  );
}

export default App;
