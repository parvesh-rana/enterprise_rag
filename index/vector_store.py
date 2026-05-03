"""ChromaDB wrapper. Owns collection lifecycle and upsert/search APIs."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import chromadb
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from core.types import Chunk

log = get_logger(__name__)


def _client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def _point_id(chunk_id: str) -> str:
    """Derive a stable ID string for a chunk."""
    return str(uuid.UUID(hashlib.md5(chunk_id.encode()).hexdigest()))


@dataclass(frozen=True)
class DenseHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


def ensure_collection(*, recreate: bool = False) -> None:
    settings = get_settings()
    client = _client()
    if recreate:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass  # collection didn't exist
    client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("chroma.collection_ready", name=settings.chroma_collection)


def upsert_chunks(chunks: Sequence[Chunk], vectors: np.ndarray, batch_size: int = 256) -> int:
    if len(chunks) != vectors.shape[0]:
        raise ValueError("chunks and vectors length mismatch")
    settings = get_settings()
    client = _client()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vecs = vectors[i : i + batch_size]
        ids = [_point_id(c.id) for c in batch]
        embeddings = vecs.tolist()
        documents = [c.text for c in batch]
        metadatas = [
            {
                "chunk_id": c.id,
                "company": c.company,
                "company_name": c.company_name,
                "year": c.year,
                "item": c.item,
                "section_title": c.section_title,
                "source_url": c.source_url,
            }
            for c in batch
        ]
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        total += len(batch)
    log.info("chroma.upsert", count=total)
    return total


def search(
    *, query_vector: np.ndarray, top_k: int, where: dict[str, Any] | None = None
) -> list[DenseHit]:
    settings = get_settings()
    client = _client()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_vector.tolist()],
        "n_results": top_k,
        "include": ["metadatas", "documents", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits: list[DenseHit] = []
    if results["ids"] and results["ids"][0]:
        for idx, _id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][idx] if results["metadatas"] else {}
            doc = results["documents"][0][idx] if results["documents"] else ""
            distance = results["distances"][0][idx] if results["distances"] else 0.0
            # ChromaDB returns distance (lower = better for cosine); convert to similarity
            score = 1.0 - distance
            payload = dict(meta)
            payload["text"] = doc
            hits.append(
                DenseHit(
                    chunk_id=meta.get("chunk_id", ""),
                    score=score,
                    payload=payload,
                )
            )
    return hits


def collection_count() -> int:
    """Return the number of documents in the collection, or 0 if not ready."""
    try:
        settings = get_settings()
        client = _client()
        collection = client.get_collection(name=settings.chroma_collection)
        return collection.count()
    except Exception:
        return 0
