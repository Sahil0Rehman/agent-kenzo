"""Incremental code-intelligence re-indexing: keeps the symbol index,
reference/dependency graphs, and (Phase 5) semantic index fresh as
`AgentController` edits files during a run, without ever re-walking the
whole repository mid-session.

Delegates entirely to `kenzo.core.indexing_service.IndexingService.reindex_file`
-- that method already does the real work (re-parse exactly the changed
file, rebuild the graphs in memory, re-embed the file's chunks, persist all
of it, all under its own lock) and always operates on whichever objects are
CURRENT on the service, even if a background full rebuild is still in
flight. This wrapper's only job is noticing "a write/edit just succeeded, on
this path" and forwarding that fact.
"""

from __future__ import annotations

from typing import Any

from kenzo.core.indexing_service import IndexingService


class ReindexingToolRegistry:
    """Wraps a tool registry so a successful `write_file`/`edit_file` call
    triggers `IndexingService.reindex_file` for the changed path, so the next
    `code_intelligence`/semantic-search call sees the change immediately.
    """

    _MUTATING = {"write_file", "edit_file"}

    def __init__(self, wrapped: Any, indexing_service: IndexingService) -> None:
        self._wrapped = wrapped
        self._indexing_service = indexing_service

    def execute(self, name: str, arguments: dict) -> Any:
        result = self._wrapped.execute(name, arguments)
        if name in self._MUTATING and bool(getattr(result, "success", False)):
            data = getattr(result, "data", None) or {}
            path = arguments.get("path") or data.get("path")
            if path:
                self._indexing_service.reindex_file(path)
        return result

    # ------------------------------------------------------------------
    # Pass-through delegation, so this class duck-types as a ToolRegistry.
    # ------------------------------------------------------------------

    def to_openai_tools(self) -> list[dict]:
        return self._wrapped.to_openai_tools()

    def names(self) -> list[str]:
        return self._wrapped.names()

    def get(self, name: str):
        return self._wrapped.get(name)

    def list_specs(self):
        return self._wrapped.list_specs()

    def register(self, spec) -> None:
        self._wrapped.register(spec)

    def register_many(self, specs) -> None:
        self._wrapped.register_many(specs)

    def subset(self, names: list[str]) -> "ReindexingToolRegistry":
        return ReindexingToolRegistry(self._wrapped.subset(names), self._indexing_service)
