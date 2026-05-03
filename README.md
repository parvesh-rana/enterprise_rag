# Enterprise RAG — SEC 10-K Question Answering

> Production-shaped retrieval pipeline over the most recent 10-K filings of Apple,
> Microsoft, Amazon, Tesla, and Nvidia. Hybrid retrieval, cross-encoder reranking,
> grounded answers with citations, an HTTP API, and an evaluation harness.

A portfolio project demonstrating the components a production RAG system needs —
without a framework hiding the moving parts. **No LangChain, no LlamaIndex.**
Each layer is wired directly so the responsibility of every dependency is visible.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION (offline)                                        │
│                                                                                     │
│   SEC EDGAR ──► HTML Parser ──► Semantic Chunker                                    │
│   (10-K filings,    (Item-level       (Item-aware splits,                           │
│    last N years)     metadata)         configurable overlap)                         │
└────────────────────────────────────┬────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           INDEXING (offline)                                         │
│                                                                                     │
│              ┌──► Embeddings (BAAI/bge-small-en-v1.5) ──► Qdrant (vector DB)        │
│   Chunks ───┤                                                                       │
│              └──► BM25 Tokenizer ──► Sparse Index (rank_bm25, pickled)              │
│                                                                                     │
└──────────────────────────┬──────────────────────────────┬───────────────────────────┘
                           │                              │
                           ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SERVING (online)                                           │
│                                                                                     │
│   POST /query                                                                       │
│       │                                                                             │
│       ▼                                                                             │
│   ┌─────────────────────────────────────────┐                                       │
│   │  Hybrid Retrieval                       │                                       │
│   │  Dense (Qdrant) + Sparse (BM25) → RRF  │                                       │
│   └─────────────────┬───────────────────────┘                                       │
│                     ▼                                                                │
│   ┌─────────────────────────────────────────┐                                       │
│   │  Cross-Encoder Reranker                 │                                       │
│   │  top-20 → top-5                         │                                       │
│   └─────────────────┬───────────────────────┘                                       │
│                     ▼                                                                │
│   ┌─────────────────────────────────────────┐                                       │
│   │  LLM Generation (NVIDIA NIM)            │                                       │
│   │  Citation-enforced prompt               │                                       │
│   └─────────────────┬───────────────────────┘                                       │
│                     ▼                                                                │
│   ┌─────────────────────────────────────────┐                                       │
│   │  Citation Parser                        │                                       │
│   │  Validates chunk IDs, drops hallucinated│                                       │
│   └─────────────────┬───────────────────────┘                                       │
│                     ▼                                                                │
│             Answer + Citations ──► React UI                                          │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Detailed per-stage data model and sequence diagrams are in [docs/architecture.md](docs/architecture.md).

---

## How to run

### Prerequisites

**Qdrant** (vector database) — pick one:

| Option | Setup |
|--------|-------|
| **Qdrant Cloud** (easiest, no install) | Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io), then set in `.env`:<br/>`QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333`<br/>`QDRANT_API_KEY=your-api-key` |
| **Docker** | `docker compose up -d qdrant` |
| **Standalone binary** (no Docker) | Download from [github.com/qdrant/qdrant/releases](https://github.com/qdrant/qdrant/releases), then run:<br/>`./qdrant` (Linux/macOS) or `.\qdrant.exe` (Windows) |

Qdrant must be reachable at the URL configured in `.env` before indexing or serving.

---

### 5-minute path (sample corpus, no SEC access required)

```bash
cp .env.example .env                          # add NVIDIA_API_KEY
pip install -e .                              # install dependencies
# Start Qdrant (Docker OR binary — see above)
python -m ingestion --sample-only             # parse + chunk the bundled filing
python -m index --recreate                    # build dense + BM25 indexes
python -m uvicorn api.main:app --port 8000    # FastAPI on :8000
```

Then `curl -X POST localhost:8000/query -H "content-type: application/json" \
-d '{"question":"What does the company describe as its primary product lines?"}'`.

### Full path (multi-year 10-Ks with React UI)

```bash
cp .env.example .env                          # add NVIDIA_API_KEY
pip install -e .
# Start Qdrant (Docker OR binary — see above)
python -m ingestion --tickers AAPL MSFT AMZN TSLA NVDA --years 4   # last 4 years
python -m index --recreate                    # embed + upsert to Qdrant + BM25
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal — start the frontend
cd frontend && npm install && npm run dev
```

The React UI will be available at `http://localhost:5173` and proxies API calls to `:8000`.

`make help` lists every Makefile target.

---

## Eval results

The eval harness runs against [evaluation/qa_set.jsonl](evaluation/qa_set.jsonl)
— 40 hand-written questions across the five filings, split into four kinds:

| Kind          | Count | What it tests                                                  |
|---------------|------:|----------------------------------------------------------------|
| factoid       | 25    | Single-passage lookups (segments, risks, products, R&D, etc.)  |
| comparative   | 5     | "Which of A vs B…" — requires hits in two filings              |
| multi_hop     | 5     | Synthesis across Items within a filing or across filings       |
| unanswerable  | 5     | Abstention probes — the system should refuse, not fabricate    |

**Metrics:** `Recall@5`, `Recall@10`, `MRR` for retrieval, plus
LLM-as-judge `Faithfulness` against the cited chunks
(SUPPORTED=1.0, PARTIAL=0.5, UNSUPPORTED=0.0; ABSTAINED is rewarded only when
the gold example is `unanswerable`). Run `make eval` to produce a fresh report
under [evaluation/runs/](evaluation/runs/).

| Run         | Model                              | Recall@5 | Recall@10 |   MRR | Faithfulness |
|-------------|------------------------------------|---------:|----------:|------:|-------------:|
| _(pending)_ | nvidia / deepseek-ai/deepseek-v3.1 |        — |         — |     — |            — |

> Populated after the first `make ingest && make index && make eval` run on
> real EDGAR data. The numbers stay `—` until then.

Ablations are first-class: `make eval ARGS="--no-reranker"` runs the same set
without the cross-encoder, and `--no-judge` skips the faithfulness pass for a
fast retrieval-only iteration.

---

## Design decisions

**Why hybrid retrieval (dense + BM25 + RRF).** 10-K queries mix paraphrase
("how does the company think about supply chain risk") with exact-match terms
("Item 1A", "AWS", numerical metrics). Dense embeddings excel at the former,
BM25 at the latter. Reciprocal Rank Fusion combines the rankings without
needing to calibrate score scales — RRF is rank-based, not score-based, so it
sidesteps the cosine-vs-BM25 distribution mismatch that breaks naive
`alpha * dense + (1-alpha) * sparse` blending.

**Why a cross-encoder reranker.** Bi-encoder embeddings (used in dense
retrieval) score query and document independently, so they trade precision
for speed and miss subtle relevance. Cross-encoders score them jointly. The
cost is quadratic in candidates, so we only run the cross-encoder on the
fused top-20 → top-5. The price is ~150 ms; the win shows up clearly in
ablations on the multi-hop questions.

**Why this chunking strategy.** 10-K text is not flat prose — it's an Item
structure with strong semantic boundaries (Item 1A. Risk Factors is a
different topic than Item 7. MD&A). The chunker respects those boundaries
first, paragraph-packs within each Item up to a token budget, falls back to
sentence-then-window splits when a paragraph alone is oversized, and bleeds a
configurable overlap across chunks but never across Items. The result is
chunks that retain enough local context to be answerable while still being
addressable by a clean `Company-Year-Item-Ordinal` ID for citation.

**Why NVIDIA NIM as the default LLM.** NIM exposes an OpenAI-compatible API
and offers free-tier reasoning-capable models suitable for portfolio use. The
provider abstraction in [generation/llm.py](generation/llm.py) makes
Anthropic and Ollama drop-in alternatives — the `OpenAI` SDK happens to also
work against Ollama and other compatible servers via `base_url`, so
`NvidiaClient` and `OllamaClient` share an SDK without sharing config.

**Citation contract enforced at parse time, not at decode time.** The system
prompt asks the model to cite every factual claim in `[chunk-id]` form. The
parser then validates each id against the actually-retrieved set and drops
anything hallucinated. So the structured `Answer.citations` list is always
clean even when the prose contains a bad bracket. Decode-time grammar
constraints would be stronger; that's listed under "what next" below.

**What the eval set actually measures.** Retrieval `Recall@k` / `MRR` are
computed against either gold chunk IDs or — preferred — gold *substrings*
that must appear in any retrieved chunk. The substring path keeps the eval
set robust to re-chunking. Faithfulness is scored by the same LLM provider
against only the cited chunks; the judge is told to ignore real-world truth
and grade groundedness only.

**Honest limitations.**

- **Eval set is small and single-author.** 40 questions written by one
  person can't capture the diversity of real user queries; expect
  overfitting to my framing of the topics.
- **No numerical reasoning step.** Hybrid retrieval surfaces relevant
  passages, but financial questions often require arithmetic across them
  (year-over-year deltas, segment-mix shifts). The current generation
  prompt does not invoke a calculator or structured-output schema.
- **English-only.** The cross-encoder, the embedding model, and the QA set
  are all English. Non-English filings (or non-English question languages)
  would degrade silently.
- **Re-chunking changes IDs.** Citation `chunk-id` values are deterministic
  given a chunker config, but a config change re-numbers them. The eval
  set's `gold_substrings` fallback exists for this reason; downstream
  consumers of citations would need a stable corpus version pinned in
  `pyproject.toml`.
- **No streaming.** `/query` is request/response only. Streaming was
  considered for Phase 5 and intentionally skipped to keep the LLM
  abstraction one-method.

---

## What I would do next

1. **Multi-vector / late-interaction retrieval (ColBERT-style)** for
   numerically dense passages where token-level matching matters. The
   payload-shape fits Qdrant's multi-vector mode without an architectural
   change.
2. **Query rewriting and decomposition** for multi-hop questions like
   "compare Tesla's and Nvidia's R&D spend over the last two years". Today
   the retriever sees the original question; a rewriter could split it into
   sub-queries, retrieve per-company, and the generator can stitch.
3. **Decode-time citation validation.** Move from regex post-hoc parsing to
   structured-output schemas (response-format JSON or constrained decoding)
   so the model can't emit a malformed citation in the first place.
4. **Eval-on-PR.** A small subset of the QA set runs on every PR via GitHub
   Actions, with regressions in `Recall@k` or `Faithfulness` failing the
   build. The full set runs nightly with a markdown report committed to
   `evaluation/runs/main/`.
5. **Corpus expansion to 10-Q + 8-K** with temporal filters. "Latest" or
   "most recent quarter" queries currently can't route to the freshest
   filing — adding a `filing_date` filter and a query-time clock would fix
   it.

---

## Project layout

```
api/          FastAPI service, routes, schemas, middleware, metrics
core/         shared config (pydantic-settings), logging (structlog), types
ingestion/    SEC EDGAR downloader, HTML→text, Item-aware semantic chunker
index/        embedding + Qdrant + BM25 build pipeline
retrieval/    dense, sparse, RRF fusion, metadata filter, cross-encoder rerank
generation/   LLM client abstraction (NVIDIA NIM, Anthropic, Ollama), prompts
evaluation/   metrics, LLM-as-judge, hand-written QA set, run reports
frontend/     Vite + React + TypeScript UI (Tailwind, @tanstack/react-query)
tests/        pytest: chunker, fusion, BM25, prompts, reranker, eval, API integration
data/         sample/ (committed), raw/ chunks/ bm25/ (gitignored)
docs/         architecture.md (sequence diagram, data model, observability)
```

---

## Reproducibility notes

- Python ≥ 3.11. `uv` or `pip install -e .` for dep management; `uv.lock` is committed.
- Every dependency in [pyproject.toml](pyproject.toml) carries a one-line
  comment justifying its presence.
- Configuration is read from `.env` via `pydantic-settings`. **No secrets
  are read from anywhere else, ever.**
- The pipeline writes nothing outside `data/` and `evaluation/runs/`.
- CI runs ruff, black `--check`, mypy (strict on `api/`, `retrieval/`),
  and pytest on every push.

---

## License

MIT — see [LICENSE](LICENSE).
