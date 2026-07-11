"""Graph-aware context retrieval: an ADDITIVE companion to
`ContextManager.retrieve_relevant`'s plain keyword-overlap search.

`ContextManager` itself is untouched -- same schema, same tests, same
behavior. `retrieve_graph_relevant` is a free function that sits alongside
it, using Phase 4's `SymbolIndex`/`ReferenceGraph`/`DependencyGraph` to expand
a query like "Fix logout" into a working set of ACTUALLY-CONNECTED facts: the
classes/methods whose name or HTTP endpoint textually matches the query, plus
whatever the reference graph says calls into or out of them, plus whatever
the dependency graph says depends on (or is depended on by) them, plus the
tests `find_related_tests` says cover them -- not just files that happen to
share a keyword, the way `ContextManager.retrieve_relevant` alone would find.
"""

from __future__ import annotations

from typing import Optional

from kenzo.core.context_manager import _tokenize
from kenzo.intelligence.dependency_graph import DependencyGraph
from kenzo.intelligence.reference_graph import ReferenceGraph
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.intelligence.test_discovery import find_related_tests

_MAX_SEED_CLASSES = 5
_MAX_EDGES_PER_METHOD = 3
_MAX_DEPS_PER_CLASS = 5
_MAX_TESTS_PER_FILE = 3


def _matches_query(name: str, endpoint: Optional[str], query_tokens: list[str]) -> bool:
    haystack = f"{name} {endpoint or ''}".lower()
    return any(tok in haystack for tok in query_tokens)


def retrieve_graph_relevant(
    query: str,
    symbol_index: Optional[SymbolIndex],
    reference_graph: Optional[ReferenceGraph],
    dependency_graph: Optional[DependencyGraph],
    top_k: int = 8,
) -> list[str]:
    """Return up to `top_k` short fact strings connecting `query` to real,
    AST-derived structure -- seed classes/methods matched by name or
    endpoint, then one hop out via the reference graph (callers/callees),
    the dependency graph (class deps/dependents), and related tests.

    Returns `[]` (never raises) if no code intelligence has been built yet,
    or if nothing matches -- callers should treat this exactly like
    `ContextManager.retrieve_relevant` returning `[]`.
    """
    if symbol_index is None or reference_graph is None or dependency_graph is None:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    seed_classes = [
        (file_path, cls)
        for file_path, cls in symbol_index.all_classes()
        if _matches_query(cls.name, None, query_tokens)
        or any(_matches_query(m.name, m.endpoint, query_tokens) for m in cls.methods)
    ][:_MAX_SEED_CLASSES]

    facts: list[str] = []

    for file_path, cls in seed_classes:
        role = (cls.stereotype or "class").capitalize()
        facts.append(f"{role} '{cls.name}' in {file_path} matches the query.")

        for method in cls.methods:
            if method.endpoint and _matches_query(method.name, method.endpoint, query_tokens):
                facts.append(f"Endpoint {method.endpoint} is served by {cls.name}.{method.name} in {file_path}.")

        for method in cls.methods:
            key = method.qualified_name or f"{cls.name}.{method.name}"
            for edge in reference_graph.callees.get(key, [])[:_MAX_EDGES_PER_METHOD]:
                facts.append(f"{key} calls {edge.get('callee')} in {edge.get('callee_file', '')}.")
            for edge in reference_graph.callers.get(key, [])[:_MAX_EDGES_PER_METHOD]:
                facts.append(f"{edge.get('caller')} calls {key}.")

        for dep in dependency_graph.get_class_dependencies(cls.name)[:_MAX_DEPS_PER_CLASS]:
            facts.append(f"{cls.name} depends on {dep}.")
        for dependent in dependency_graph.get_dependents(cls.name)[:_MAX_DEPS_PER_CLASS]:
            facts.append(f"{dependent} depends on {cls.name}.")

        for test_path in find_related_tests(symbol_index, file_path)[:_MAX_TESTS_PER_FILE]:
            facts.append(f"{test_path} is a related test for {file_path}.")

    unique_facts = list(dict.fromkeys(facts))  # de-dup, preserve discovery order
    return unique_facts[:top_k]


def render_graph_relevant(
    query: str,
    symbol_index: Optional[SymbolIndex],
    reference_graph: Optional[ReferenceGraph],
    dependency_graph: Optional[DependencyGraph],
    top_k: int = 8,
) -> str:
    """Convenience wrapper mirroring `ContextManager.render_relevant`'s
    prompt-ready rendering, for the graph-aware fact list above."""
    facts = retrieve_graph_relevant(query, symbol_index, reference_graph, dependency_graph, top_k)
    if not facts:
        return ""
    lines = [f"- {fact}" for fact in facts]
    return "Relevant code structure (from the indexed reference/dependency graph):\n" + "\n".join(lines)
