"""`python -m index` (a.k.a. `make index`).

Reads chunk JSONLs from data/chunks/, embeds them, upserts to Qdrant, and
builds + persists the BM25 index to data/bm25/index.pkl.

Both indexes are built from the same chunks, in the same order, with the same
ids — that's the contract the hybrid retriever depends on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from core.config import get_settings
from core.logging import configure_logging, get_logger
from core.types import Chunk
from index.bm25 import BM25Index
from index.embeddings import embed_documents
from index.vector_store import ensure_collection, upsert_chunks

log = get_logger(__name__)


def _load_chunks(chunks_dir: Path) -> list[Chunk]:
    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No chunk files in {chunks_dir}. Run `make ingest` first.")
    chunks: list[Chunk] = []
    for f in files:
        with f.open("rb") as fh:
            for line in fh:
                chunks.append(Chunk.model_validate_json(line))
    return chunks


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="index")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before upserting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    args = parser.parse_args()

    settings = get_settings()
    chunks = _load_chunks(settings.chunks_dir)
    log.info("index.loaded_chunks", n=len(chunks))

    # --- Dense ---
    ensure_collection(recreate=args.recreate)

    vectors_chunks: list[Chunk] = []
    all_vectors: list[np.ndarray] = []
    for i in tqdm(range(0, len(chunks), args.batch_size), desc="embed"):
        batch = chunks[i : i + args.batch_size]
        vecs = embed_documents([c.text for c in batch], batch_size=args.batch_size)
        vectors_chunks.extend(batch)
        all_vectors.append(vecs)

    matrix = np.vstack(all_vectors) if all_vectors else np.zeros((0, settings.embedding_dim))
    upsert_chunks(vectors_chunks, matrix)

    # --- Sparse ---
    bm25 = BM25Index.build(chunks)
    bm25_path = settings.bm25_dir / "index.pkl"
    bm25.save(bm25_path)
    log.info("index.bm25_saved", path=str(bm25_path), n=len(chunks))

    print(
        json.dumps(
            {
                "chunks_indexed": len(chunks),
                "bm25_path": str(bm25_path),
                "chroma_collection": settings.chroma_collection,
            }
        )
    )


if __name__ == "__main__":
    main()
