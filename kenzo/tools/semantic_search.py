"""Agent-facing hybrid search tool, backed by Phase 5's `CodeSearchEngine`.

Unlike `kenzo/tools/code_intelligence.py`'s tools (which load their data
fresh from `.kenzo/*.json` on every call, since that data is just files on
disk), this tool MUST be bound to the live, in-process
`kenzo.core.indexing_service.IndexingService` rather than reconstructing a
`VectorStore` per call -- Qdrant's default `:memory:` mode is a purely
in-process structure with no on-disk representation, so a freshly-constructed
`QdrantClient(":memory:")` on every tool call would always be empty. Reading
`indexing_service.symbol_index`/`.reference_graph`/`.dependency_graph`/
`.embedding_router`/`.vector_store` at CALL TIME (not captured once at
registration) also keeps this correct while a background full rebuild is
still in flight, exactly like `kenzo.core.reindex.ReindexingToolRegistry`.
"""

from __future__ import annotations

import functools
from typing import Any

from kenzo.core.code_search import CodeSearchEngine
from kenzo.core.indexing_service import IndexingService
from kenzo.core.schemas import ToolResult, ToolSpec

_NOT_READY_ERROR = (
    "No code intelligence/semantic index is available for this repository yet "
    "(background indexing may still be running -- try again shortly, or run "
    "`kenzo index` for a synchronous build)."
)


def semantic_search(query: str, top_k: int = 10, *, indexing_service: IndexingService) -> ToolResult:
    """Hybrid search combining exact symbol match, dependency-graph
    centrality, vector similarity, and keyword overlap -- see
    `kenzo.core.code_search.CodeSearchEngine` for the exact ranking formula.
    Finds relevant code even when the query's words don't literally appear
    in it (e.g. "fix sign out problem" -> LogoutController/AuthService).
    """
    symbol_index = indexing_service.symbol_index
    if symbol_index is None or not symbol_index.files:
        return ToolResult(tool_name="semantic_search", success=False, error=_NOT_READY_ERROR)

    engine = CodeSearchEngine(
        symbol_index,
        indexing_service.reference_graph,
        indexing_service.dependency_graph,
        indexing_service.embedding_router,
        indexing_service.vector_store,
    )
    results = engine.search(query, top_k=top_k)

    output = (
        f"{len(results)} result(s) for '{query}'."
        if results
        else f"No relevant code found for '{query}'."
    )
    return ToolResult(
        tool_name="semantic_search",
        success=True,
        output=output,
        data={"results": [r.model_dump() for r in results]},
    )


def get_tool_specs(repo_root: str, permissions: Any, indexing_service: IndexingService | None) -> list[ToolSpec]:
    """Build the `semantic_search` ToolSpec, bound to the live
    `indexing_service`. Returns `[]` (no tool registered at all) if
    `indexing_service` is `None` -- there is nothing this tool could search
    without it, unlike the Phase 4 code-intelligence tools which can at least
    report a clear "not indexed yet" error from a stateless, per-call disk
    load.
    """
    del repo_root, permissions  # unused: this tool is scoped entirely by indexing_service
    if indexing_service is None:
        return []

    return [
        ToolSpec(
            name="semantic_search",
            description=(
                "Search the repository for code relevant to a natural-language query, combining "
                "exact symbol matching, call-graph centrality, semantic similarity, and keyword "
                "overlap. Use this when you don't know the exact class/method name -- e.g. "
                "'fix the sign out problem' or 'where do we validate the OIDC token'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A natural-language description of what you're looking for."},
                    "top_k": {"type": "integer", "description": "Maximum number of results to return.", "default": 10},
                },
                "required": ["query"],
            },
            func=functools.partial(semantic_search, indexing_service=indexing_service),
        )
    ]
