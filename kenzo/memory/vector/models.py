"""Shared data contract for the Phase 5 semantic memory layer: `CodeChunk` is
what `SemanticIndexer` produces and what `VectorStore` stores/retrieves --
every embedding provider and vector store implementation speaks this same
shape, mirroring how `kenzo.intelligence.models.FileNode` is the shared
contract every Phase 4 parser speaks.

Kept dependency-free of the rest of `kenzo.memory.vector` (no imports from
`embeddings.py`/`store.py`) so both can import this without cycles.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CodeChunk(BaseModel):
    """One semantically-meaningful unit of indexed code -- a class, a method,
    a standalone function, or (rarely, as a fallback) a whole module. NEVER a
    blind fixed-size slice of a file; see `kenzo.memory.vector.semantic_indexer`
    for how these are produced from an already-parsed `SymbolIndex`.

    `id` is a stable, deterministic identifier (e.g. "path/to/File.java::
    ClassName.methodName") so re-indexing the same symbol after an edit
    updates its vector-store entry in place instead of accumulating
    duplicates -- see `VectorStore.update`.
    """

    id: str
    file: str
    symbol: str  # e.g. "AuthService.terminateSession", "AuthService", or "" for a module-level chunk
    content: str  # the rendered text actually embedded: signature + docstring/comments + body + light surrounding context
    language: str
    chunk_type: str  # "class" | "method" | "function" | "module"
    metadata: dict[str, Any] = Field(default_factory=dict)
