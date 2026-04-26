# Enterprise RAG — convenience targets.
# All real work lives in module __main__ entrypoints; Make is a thin wrapper.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PY := uv run python
ARGS ?=

.PHONY: help install ingest ingest-sample index index-recreate eval eval-fast \
        serve test test-fast lint fmt typecheck check build up down logs clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

install: ## Sync Python deps with uv (creates .venv, writes uv.lock)
	uv sync

ingest: ## Download SEC 10-K filings and parse to chunks
	$(PY) -m ingestion $(ARGS)

ingest-sample: ## Skip EDGAR; ingest only files under data/sample/
	$(PY) -m ingestion --sample-only

index: ## Build Qdrant dense index + on-disk BM25 index from data/chunks
	$(PY) -m index $(ARGS)

index-recreate: ## Drop and recreate the Qdrant collection before upserting
	$(PY) -m index --recreate

eval: ## Run the full eval harness (retrieval + LLM-as-judge faithfulness)
	$(PY) -m evaluation $(ARGS)

eval-fast: ## Eval without LLM-as-judge (retrieval metrics only; no LLM cost)
	$(PY) -m evaluation --no-judge

serve: ## Run the FastAPI service locally on $$API_PORT (default 8000)
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port $${API_PORT:-8000} --reload

test: ## Run pytest with coverage on critical modules
	uv run pytest --cov=api --cov=retrieval --cov=generation --cov-report=term-missing

test-fast: ## Run pytest with no coverage, fail-fast
	uv run pytest -x -q

lint: ## ruff + black --check (CI-equivalent)
	uv run ruff check .
	uv run black --check .

fmt: ## Apply ruff --fix and black (mutates files)
	uv run ruff check --fix .
	uv run black .

typecheck: ## mypy (strict on api + retrieval; loose elsewhere)
	uv run mypy api retrieval generation core ingestion index evaluation

check: lint typecheck test ## Everything CI runs, in order

build: ## Build the API Docker image
	docker build -t enterprise-rag-api:dev .

up: ## docker compose up -d (Qdrant + API)
	docker compose up -d

down: ## docker compose down
	docker compose down

logs: ## Tail compose logs
	docker compose logs -f --tail=200

clean: ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov build dist *.egg-info
