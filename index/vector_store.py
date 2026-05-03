"""Qdrant wrapper. Owns collection lifecycle and upsert/search APIs."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from core.config import get_settings
from core.logging import get_logger
from core.types import Chunk

log = get_logger(__name__)


def _client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_url.startswith("local:"):
        path = settings.qdrant_url.removeprefix("local:").strip()
        return QdrantClient(path=str(path if path else settings.data_dir / "qdrant_local"))
    return QdrantClient(url=settings.qdrant_url, prefer_grpc=False, timeout=30)


def _point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be UUID or unsigned int; derive a stable UUIDv5."""
    return str(uuid.UUID(hashlib.md5(chunk_id.encode()).hexdigest()))


@dataclass(frozen=True)
class DenseHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


def ensure_collection(*, recreate: bool = False) -> None:
    settings = get_settings()
    client = _client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection in existing and not recreate:
        return
    if settings.qdrant_collection in existing and recreate:
        client.delete_collection(settings.qdrant_collection)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(size=settings.embedding_dim, distance=qm.Distance.COSINE),
    )

    # Indexes for the metadata fields we filter on.
    for field, schema in (
        ("company", qm.PayloadSchemaType.KEYWORD),
        ("year", qm.PayloadSchemaType.INTEGER),
        ("item", qm.PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name=field,
            field_schema=schema,
        )
    log.info("qdrant.collection_created", name=settings.qdrant_collection)


def upsert_chunks(chunks: Sequence[Chunk], vectors: np.ndarray, batch_size: int = 256) -> int:
    if len(chunks) != vectors.shape[0]:
        raise ValueError("chunks and vectors length mismatch")
    settings = get_settings()
    client = _client()

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vecs = vectors[i : i + batch_size]
        points = [
            qm.PointStruct(
                id=_point_id(c.id),
                vector=vec.tolist(),
                payload={
                    "chunk_id": c.id,
                    "company": c.company,
                    "company_name": c.company_name,
                    "year": c.year,
                    "item": c.item,
                    "section_title": c.section_title,
                    "text": c.text,
                    "source_url": c.source_url,
                },
            )
            for c, vec in zip(batch, vecs, strict=True)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
        total += len(points)
    log.info("qdrant.upsert", count=total)
    return total


def search(
    *, query_vector: np.ndarray, top_k: int, qdrant_filter: qm.Filter | None = None
) -> list[DenseHit]:
    client = _client()
    settings = get_settings()
    # `query_points` is the supported API in qdrant-client >=1.10. The older
    # `client.search(...)` method was removed.
    res = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector.tolist(),
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )
    return [
        DenseHit(
            chunk_id=p.payload["chunk_id"],
            score=float(p.score),
            payload=dict(p.payload),
        )
        for p in res.points
        if p.payload is not None
    ]
