"""Semantic-aware context retrieval: an ADDITIVE companion to
`ContextManager.retrieve_relevant`'s plain keyword-overlap search and to
`kenzo.core.graph_context`'s structural-graph expansion.

`ContextManager` itself is untouched -- same schema, same tests, same
behavior. `retrieve_semantic_relevant` is a free function that sits
alongside it (and alongside `retrieve_graph_relevant`), using Phase 5's
`CodeSearchEngine` -- which itself blends exact-symbol, dependency-graph,
vector-semantic, and keyword signals into one ranked list -- to answer "what
code is actually relevant to this query," including cases where the query's
own words never appear verbatim anywhere in the codebase. For example, "OIDC
logout not called" should be able to surface OIDC config, `SecurityConfig`,
logout handlers, and their tests via the vector-semantic signal's conceptual
similarity, not only through literal keyword overlap -- though note that with
the default offline `HashingEmbeddingProvider` (see
`kenzo.memory.vector.embeddings`), that similarity is genuinely lexical/
fuzzy-substring, not learned synonym-level understanding; a real semantic
provider is needed for e.g. "sign out" to strongly relate to "logout" on
meaning alone.

Mirrors `kenzo.core.graph_context`'s exact shape: a `retrieve_*_relevant`
function returning short fact strings, and a `render_*_relevant` convenience
wrapper for prompt-building, both gracefully returning `[]`/`""` (never
raising) when no code intelligence/semantic index has been built for this
repo yet.
"""

from __future__ import annotations

from typing import Optional

from kenzo.core.code_search import CodeSearchEngine, SearchResult

# A candidate that clears `CodeSearchEngine`'s own "at least one of
# symbol_match/semantic/keyword is positive" filter can still carry a
# vanishingly small combined score (e.g. one matched token out of a long,
# noisy query). This floor keeps `retrieve_semantic_relevant` from padding
# its output with results too weak to be worth an agent's attention -- it is
# deliberately low relative to the weights in `kenzo.core.code_search`
# (which sum to 1.0) so a single strong signal (e.g. a name match on one
# token) still clears it.
_MIN_RELEVANCE_SCORE = 0.05


def _render_fact(result: SearchResult) -> str:
    """Render one `SearchResult` into a short, informative fact string --
    enough for a human/model reading it to immediately know WHAT matched and
    WHERE, e.g. "SecurityConfig.configure (method, score=0.82) in
    src/main/java/.../SecurityConfig.java"."""
    return f"{result.symbol} ({result.chunk_type}, score={result.score:.2f}) in {result.file}"


def retrieve_semantic_relevant(
    query: str,
    code_search_engine: Optional[CodeSearchEngine],
    top_k: int = 8,
) -> list[str]:
    """Return up to `top_k` short fact strings naming the symbols
    `CodeSearchEngine`'s hybrid (symbol+graph+vector+keyword) ranking finds
    most relevant to `query`.

    Returns `[]` (never raises) if `code_search_engine` is `None` (no code
    intelligence/semantic index has been built for this repo yet), if
    `query` is empty/whitespace, if `top_k <= 0`, if nothing scores above a
    reasonable relevance floor, or if the search itself fails for any reason
    (e.g. a network error from a remotely-hosted embedding provider) --
    callers should treat every one of these exactly like
    `ContextManager.retrieve_relevant`/`retrieve_graph_relevant` returning
    `[]`.
    """
    if code_search_engine is None:
        return []
    if not query or not query.strip():
        return []
    if top_k <= 0:
        return []

    try:
        results = code_search_engine.search(query, top_k=top_k)
    except Exception:  # noqa: BLE001 - a best-effort context signal must never break the agent loop
        return []

    return [_render_fact(result) for result in results if result.score > _MIN_RELEVANCE_SCORE]


def render_semantic_relevant(
    query: str,
    code_search_engine: Optional[CodeSearchEngine],
    top_k: int = 8,
) -> str:
    """Convenience wrapper mirroring `ContextManager.render_relevant`'s and
    `graph_context.render_graph_relevant`'s exact rendering style, for the
    semantic-search fact list above. Returns `""` if there's nothing to
    show."""
    facts = retrieve_semantic_relevant(query, code_search_engine, top_k)
    if not facts:
        return ""
    lines = [f"- {fact}" for fact in facts]
    return "Relevant code (from semantic/hybrid search):\n" + "\n".join(lines)
