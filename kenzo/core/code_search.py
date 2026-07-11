"""Hybrid code search: an ADDITIVE companion that combines Phase 4's
structural intelligence (`SymbolIndex`/`ReferenceGraph`/`DependencyGraph`)
with Phase 5's semantic memory (`EmbeddingRouter`/`VectorStore`) into one
ranked result list.

Where `kenzo.core.context_manager.ContextManager.retrieve_relevant` only ever
sees an agent run's own short-term notes, and `kenzo.core.graph_context` only
ever expands a query into structural FACTS, `CodeSearchEngine` answers a
different question -- "where in this codebase is X" -- by combining FOUR
independent signals over the whole indexed repository:

1. **symbol_match** -- does the query's text literally match a class/method/
   function's (qualified) name? An exact (case-insensitive) match of the
   whole query scores 1.0; otherwise the fraction of the query's tokens found
   as a substring of the name scores proportionally. See `_name_match_score`.
2. **semantic** -- cosine similarity between the query's embedding and a
   `CodeChunk`'s embedding, from `VectorStore.search` (embedded via the SAME
   `EmbeddingRouter` that built the semantic index), clamped into [0, 1] --
   see `_normalize_semantic`.
3. **dependency_importance** -- how "central" a symbol is: a class many
   others depend on (`DependencyGraph.get_dependents`), or a method many
   others call (`ReferenceGraph.callers`). Normalized via log-scaling
   (`log1p(count) / log1p(max_count)`) ACROSS THE CURRENT RESULT SET (not
   globally) so one very popular class doesn't uniformly dominate every
   query's ranking -- see `_dependency_raw_count`.
4. **keyword** -- token-overlap ratio between the query and a `CodeChunk`'s
   rendered content, using the same tokenization convention as
   `ContextManager._tokenize` (lowercase alphanumeric, length > 2,
   stopword-filtered), reused directly rather than reimplemented. Computed
   over the SAME canonical per-symbol text `SemanticIndexer` embeds (via its
   pure, side-effect-free `chunk_file`), so this signal has genuine coverage
   of the whole corpus independent of what the vector store happens to
   return -- not merely a byproduct of the semantic search's own result set.

Final score, computed once every signal is independently normalized to
roughly 0..1, is a plain WEIGHTED SUM -- never a product, so a symbol that
scores 0 on one signal (e.g. `semantic`, because nothing was ever embedded)
still surfaces reasonably on the strength of the others:

    score = 0.35 * symbol_match
          + 0.30 * semantic
          + 0.20 * keyword
          + 0.15 * dependency_importance

The weights sum to 1.0 and are deliberately biased towards the two signals
most likely to directly answer "where is X" -- a name match or a real
conceptual match -- with `keyword` as a cheaper literal-text signal and
`dependency_importance` kept smallest since it is a CENTRALITY BOOST, not
independent evidence of relevance to the query: a candidate is only ever
included in results if `symbol_match`, `semantic`, or `keyword` found it in
the first place (see `search`'s filtering step) -- popularity alone never
earns a place in the results. Tune the four `_*_WEIGHT` constants below to
change this balance; they are the one place the formula lives.

Performance note: `_collect_keyword_candidates` re-reads and re-chunks every
indexed file's text on every `search()` call (one full corpus pass), which is
fine for a locally-run coding agent calling `search()` a handful of times per
turn, but a caller doing this in a tight loop should cache/memoize a
`CodeSearchEngine` per unchanged index rather than reconstruct the corpus
scan on every call.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel

from kenzo.core.context_manager import _tokenize
from kenzo.intelligence.dependency_graph import DependencyGraph
from kenzo.intelligence.reference_graph import ReferenceGraph
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.memory.vector.embeddings import EmbeddingRouter
from kenzo.memory.vector.semantic_indexer import SemanticIndexer
from kenzo.memory.vector.store import VectorStore

_SYMBOL_WEIGHT = 0.35
_SEMANTIC_WEIGHT = 0.30
_KEYWORD_WEIGHT = 0.20
_DEPENDENCY_WEIGHT = 0.15

# How many (CodeChunk, score) pairs to pull from the vector store per query --
# deliberately wider than `top_k` so a genuinely relevant symbol that ranks
# just outside a narrow top_k has a chance to be found and re-ranked upward
# once the other three signals are folded in.
_CANDIDATE_POOL_MULTIPLIER = 5
_MIN_CANDIDATE_POOL = 40


class SearchResult(BaseModel):
    """One ranked hit from `CodeSearchEngine.search`.

    `signals` always has all four keys ("symbol_match", "semantic",
    "dependency_importance", "keyword"), each independently in roughly
    [0, 1], even when a given signal contributed nothing (0.0) -- so callers/
    tests can always inspect the full breakdown behind `score`, not just the
    combined number.
    """

    symbol: str
    file: str
    chunk_type: str
    score: float
    signals: dict[str, float]


def _name_match_score(
    simple_name: str, qualified_name: str, normalized_query: str, query_tokens: list[str]
) -> float:
    """1.0 for an exact (case-insensitive) match of the whole query against
    either the simple or qualified name; otherwise the fraction of the
    query's tokens that appear as a substring of "simple_name
    qualified_name" (lowercased). 0.0 if the query has no usable tokens.
    """
    if not query_tokens:
        return 0.0
    if normalized_query and normalized_query in (simple_name.lower(), qualified_name.lower()):
        return 1.0
    haystack = f"{simple_name} {qualified_name}".lower()
    matched = sum(1 for tok in query_tokens if tok in haystack)
    return matched / len(query_tokens)


def _keyword_score(content: str, query_tokens: list[str]) -> float:
    """Fraction of the query's tokens present anywhere in `content`'s own
    tokenization -- plain token-overlap, no positional/order weighting."""
    if not query_tokens:
        return 0.0
    content_tokens = set(_tokenize(content))
    matched = sum(1 for tok in query_tokens if tok in content_tokens)
    return matched / len(query_tokens)


def _normalize_semantic(raw_score: float) -> float:
    """Clamp a raw cosine similarity into [0, 1].

    Cosine similarity over the hashing provider's signed (+1/-1) features can
    legitimately go negative for unrelated text -- that is treated as "no
    signal" (0.0) rather than rescaled towards 0.5 via e.g. `(score + 1) / 2`,
    since an anti-correlated embedding should not read as "somewhat
    relevant."
    """
    return max(0.0, min(1.0, float(raw_score)))


def _dependency_raw_count(
    reference_graph: ReferenceGraph, dependency_graph: DependencyGraph, chunk_type: str, symbol: str
) -> int:
    """Raw (un-normalized) centrality count for one candidate: dependents for
    a class, callers for a method.

    Top-level functions (and any "module"-level chunk) always get 0 here --
    `ReferenceGraph`'s call-resolution algorithm only ever resolves a callee
    through a class's declared dependency variables (see
    `kenzo.intelligence.reference_graph`'s module docstring), so a top-level
    function is never a resolvable callee target under that algorithm. This
    is an inherited, documented limitation of Phase 4, not something
    introduced here.
    """
    if chunk_type == "class":
        return len(dependency_graph.get_dependents(symbol))
    if chunk_type == "method":
        return len(reference_graph.callers.get(symbol, []))
    return 0


class CodeSearchEngine:
    """Combines exact-symbol, dependency-graph, vector-semantic, and keyword
    signals into one ranked `list[SearchResult]`. See the module docstring
    for the exact scoring formula.

    Reads three already-built Phase 4 structures (`SymbolIndex`,
    `ReferenceGraph`, `DependencyGraph`) and two already-populated Phase 5
    structures (`EmbeddingRouter`, `VectorStore`) -- this class never builds
    or mutates any of them; it is a pure query-time reader, safe to construct
    fresh for every search or to keep around across many searches (it holds
    no query-specific state between calls).
    """

    def __init__(
        self,
        symbol_index: SymbolIndex,
        reference_graph: ReferenceGraph,
        dependency_graph: DependencyGraph,
        embedding_router: EmbeddingRouter,
        vector_store: VectorStore,
    ) -> None:
        self.symbol_index = symbol_index
        self.reference_graph = reference_graph
        self.dependency_graph = dependency_graph
        self.embedding_router = embedding_router
        self.vector_store = vector_store
        # Used ONLY for its pure, side-effect-free `chunk_file` -- never
        # `index_file`/`index_all` -- so this never touches `vector_store` or
        # any on-disk bookkeeping. See `_collect_keyword_candidates`.
        self._chunker = SemanticIndexer(
            str(symbol_index.repo_root), symbol_index, embedding_router, vector_store
        )

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Rank every candidate symbol found by name match, keyword overlap,
        or vector similarity against `query`, and return the top `top_k`.

        Returns `[]` for `top_k <= 0`, or if nothing scores positively on any
        of `symbol_match`/`semantic`/`keyword` -- an empty result is the
        correct answer when nothing in the indexed repo is actually relevant,
        never padded with irrelevant "central" symbols.
        """
        if top_k <= 0:
            return []

        normalized_query = query.strip().lower()
        query_tokens = _tokenize(query)

        candidates = self._collect_symbol_match_candidates(normalized_query, query_tokens)
        self._collect_keyword_candidates(query_tokens, candidates)
        self._merge_semantic_candidates(query, top_k, candidates)

        raw_dependency = {
            key: _dependency_raw_count(self.reference_graph, self.dependency_graph, c["chunk_type"], c["symbol"])
            for key, c in candidates.items()
        }
        max_raw = max(raw_dependency.values(), default=0)

        results: list[SearchResult] = []
        for key, c in candidates.items():
            if c["symbol_match"] <= 0.0 and c["semantic"] <= 0.0 and c["keyword"] <= 0.0:
                # No independent evidence this candidate is relevant to the
                # query at all -- dependency importance alone never earns a
                # place in the results (see module docstring).
                continue

            dependency_importance = (
                math.log1p(raw_dependency[key]) / math.log1p(max_raw) if max_raw > 0 else 0.0
            )
            signals = {
                "symbol_match": c["symbol_match"],
                "semantic": c["semantic"],
                "dependency_importance": dependency_importance,
                "keyword": c["keyword"],
            }
            score = (
                _SYMBOL_WEIGHT * signals["symbol_match"]
                + _SEMANTIC_WEIGHT * signals["semantic"]
                + _DEPENDENCY_WEIGHT * signals["dependency_importance"]
                + _KEYWORD_WEIGHT * signals["keyword"]
            )
            results.append(
                SearchResult(symbol=c["symbol"], file=c["file"], chunk_type=c["chunk_type"], score=score, signals=signals)
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Candidate collection (each independent of the others -- merged by the
    # same "file::symbol" / "file::Class.method" id scheme `SemanticIndexer`
    # already uses for `CodeChunk.id`, so a hit found by more than one signal
    # merges into a single candidate rather than appearing twice).
    # ------------------------------------------------------------------

    def _collect_symbol_match_candidates(
        self, normalized_query: str, query_tokens: list[str]
    ) -> dict[str, dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        if not query_tokens:
            return candidates

        for file_path, cls in self.symbol_index.all_classes():
            score = _name_match_score(cls.name, cls.qualified_name or cls.name, normalized_query, query_tokens)
            if score > 0:
                key = f"{file_path}::{cls.name}"
                candidates[key] = self._new_candidate(cls.name, file_path, "class", symbol_match=score)

            for method in cls.methods:
                qualified = method.qualified_name or f"{cls.name}.{method.name}"
                m_score = _name_match_score(method.name, qualified, normalized_query, query_tokens)
                if m_score > 0:
                    key = f"{file_path}::{cls.name}.{method.name}"
                    candidates[key] = self._new_candidate(
                        f"{cls.name}.{method.name}", file_path, "method", symbol_match=m_score
                    )

        for file_path, fn in self.symbol_index.all_functions():
            qualified = fn.qualified_name or fn.name
            f_score = _name_match_score(fn.name, qualified, normalized_query, query_tokens)
            if f_score > 0:
                key = f"{file_path}::{fn.name}"
                candidates[key] = self._new_candidate(fn.name, file_path, "function", symbol_match=f_score)

        return candidates

    def _collect_keyword_candidates(self, query_tokens: list[str], candidates: dict[str, dict[str, Any]]) -> None:
        """Score every symbol's canonical chunk text (the same text
        `SemanticIndexer` embeds) against the query's tokens, independent of
        whether that symbol was ever actually embedded/stored.
        """
        if not query_tokens:
            return

        for file_path in sorted(self.symbol_index.files.keys()):
            for chunk in self._chunker.chunk_file(file_path):
                score = _keyword_score(chunk.content, query_tokens)
                if score <= 0:
                    continue
                entry = candidates.get(chunk.id)
                if entry is None:
                    entry = self._new_candidate(chunk.symbol, chunk.file, chunk.chunk_type)
                    candidates[chunk.id] = entry
                entry["keyword"] = max(entry["keyword"], score)

    def _merge_semantic_candidates(self, query: str, top_k: int, candidates: dict[str, dict[str, Any]]) -> None:
        pool_size = max(top_k * _CANDIDATE_POOL_MULTIPLIER, _MIN_CANDIDATE_POOL)
        query_embedding = self.embedding_router.embed(query)
        hits = self.vector_store.search(query_embedding, top_k=pool_size)

        for chunk, raw_score in hits:
            entry = candidates.get(chunk.id)
            if entry is None:
                entry = self._new_candidate(chunk.symbol, chunk.file, chunk.chunk_type)
                candidates[chunk.id] = entry
            entry["semantic"] = max(entry["semantic"], _normalize_semantic(raw_score))

    @staticmethod
    def _new_candidate(symbol: str, file: str, chunk_type: str, symbol_match: float = 0.0) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "file": file,
            "chunk_type": chunk_type,
            "symbol_match": symbol_match,
            "semantic": 0.0,
            "keyword": 0.0,
        }
