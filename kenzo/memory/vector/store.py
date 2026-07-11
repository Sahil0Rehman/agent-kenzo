"""Vector store abstraction: `VectorStore` is the interface every backend
implements; `QdrantStore` is the first (and currently only) concrete
implementation, per the Phase 5 requirement to "support first: Qdrant."

`QdrantStore` defaults to Qdrant's embedded ":memory:" client mode -- no
external server process required, verified to work fully offline (create
collection, upsert, similarity search, delete, and payload-filtered search
all confirmed working against `QdrantClient(":memory:")` directly). This is
what makes semantic search usable out of the box for a *local* coding agent
and deterministic in tests, while still supporting a real persistent Qdrant
server via `kenzo.config.settings.VectorStoreConfig.url` for anyone who wants
one.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from kenzo.memory.vector.models import CodeChunk


class VectorStore(ABC):
    """Interface every vector store backend implements."""

    @abstractmethod
    def add(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Insert (or overwrite, if a chunk with the same `id` already
        exists) each chunk with its corresponding embedding."""

    @abstractmethod
    def update(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Re-embed/replace existing chunks. Semantically identical to `add`
        for backends (like Qdrant) whose native write op is an upsert --
        kept as a distinct method because callers (e.g. incremental
        re-indexing after an edit) want to express INTENT, and a future
        backend without native upsert semantics may need to implement this
        differently (e.g. delete-then-insert)."""

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by their `CodeChunk.id`. Silently no-ops for any id
        that isn't present."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[CodeChunk, float]]:
        """Return up to `top_k` (CodeChunk, similarity_score) pairs, highest
        score first. `filters` is an optional exact-match filter over
        `CodeChunk` fields/metadata (e.g. `{"language": "java"}` or
        `{"chunk_type": "method"}`) -- backends that can't filter natively
        should still support this by filtering client-side."""


class QdrantStore(VectorStore):
    """`VectorStore` backed by a Qdrant collection (embedded `:memory:` mode
    by default -- see module docstring)."""

    def __init__(self, collection_name: str, dimensions: int, url: str = ":memory:") -> None:
        from qdrant_client import QdrantClient

        self._collection_name = collection_name
        self._dimensions = dimensions
        self._client = QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._dimensions, distance=Distance.COSINE),
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Qdrant point IDs must be an unsigned integer or a UUID (a plain
        arbitrary string raises `ValueError: ... is not a valid UUID`) --
        deterministically derive one from the chunk's real, stable string id
        via UUID5, so re-adding the same chunk id always maps to the same
        point (true upsert semantics) while the original id is preserved
        verbatim in the payload for round-tripping.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def add(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        if len(chunks) != len(embeddings):
            raise ValueError(f"got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return

        points = [
            PointStruct(
                id=self._point_id(chunk.id),
                vector=embedding,
                payload={"chunk": chunk.model_dump(), "chunk_id": chunk.id},
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def update(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        # Qdrant's upsert already IS "insert or replace by id" -- add() and
        # update() are the same operation here, see the class docstring.
        self.add(chunks, embeddings)

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        point_ids = [self._point_id(cid) for cid in chunk_ids]
        self._client.delete(collection_name=self._collection_name, points_selector=point_ids)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[CodeChunk, float]]:
        query_filter = self._build_filter(filters) if filters else None
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        return [(CodeChunk.model_validate(point.payload["chunk"]), point.score) for point in results.points]

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = [FieldCondition(key=f"chunk.{key}", match=MatchValue(value=value)) for key, value in filters.items()]
        return Filter(must=conditions)

    def count(self) -> int:
        """Number of chunks currently stored -- mainly for tests/diagnostics."""
        return self._client.count(collection_name=self._collection_name).count
