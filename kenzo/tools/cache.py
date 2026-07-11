"""Caching wrapper around a `ToolRegistry`-like object.

`CachedToolRegistry` sits between the agent loop and a real `ToolRegistry`
(see `kenzo.tools.base.ToolRegistry`) -- or another `CachedToolRegistry`, it
duck-types either way -- so that within a single long-running agent run, an
identical read-only tool call (same tool name + same arguments) is served
from an in-memory cache instead of re-executed, while a successful edit
correctly invalidates whatever cached reads/searches it could have made
stale.

This is purely an in-process, single-run optimization: the cache lives on
`self._cache` for the lifetime of this object and is never persisted to
disk (unlike `kenzo.memory.repository.RepositoryMemory`'s `.kenzo/` files).
"""

from __future__ import annotations

from typing import Any

from kenzo.core.schemas import ToolResult


class CachedToolRegistry:
    """Caches pure-read tool calls and invalidates them on successful edits.

    Drop-in substitute anywhere a `ToolRegistry` is expected: every method a
    caller could reach on the wrapped registry (`register`, `register_many`,
    `get`, `list_specs`, `names`, `subset`, `to_openai_tools`, `execute`) is
    implemented here too, either by delegating straight through or -- for
    `execute` -- by adding the caching/invalidation behavior on top.
    """

    # Pure reads with no side effects: safe to serve repeat calls from cache.
    _CACHEABLE: frozenset[str] = frozenset(
        {
            "read_file",
            "search_files",
            "find_symbol",
            "repository_tree",
            "map_flow",
            # Phase 4 code intelligence: also pure reads over `.kenzo/`
            # artifacts, same caching/invalidation treatment as find_symbol/
            # map_flow above.
            "trace_flow",
            "find_references",
            "find_callers",
            "find_usages",
            "explain_architecture",
            "find_related_tests",
        }
    )

    # Tools whose successful execution can change what's on disk, and so must
    # invalidate any cached reads/searches that could now be stale.
    _MUTATING: frozenset[str] = frozenset({"write_file", "edit_file"})

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        # cache_key -> ToolResult
        self._cache: dict[Any, ToolResult] = {}
        # cache_key -> the original arguments dict, kept alongside self._cache
        # so _invalidate_for_edit can inspect a cached call's arguments (e.g.
        # to check "was this read_file for this exact path?") without trying
        # to reconstruct a dict from the tuple-of-pairs form of the key.
        self._cache_args: dict[Any, dict] = {}

    # ------------------------------------------------------------------
    # Core caching/invalidation behavior
    # ------------------------------------------------------------------

    def execute(self, name: str, arguments: dict) -> ToolResult:
        if name in self._MUTATING:
            result = self._wrapped.execute(name, arguments)
            if result.success is True:
                path = arguments.get("path") or (result.data or {}).get("path")
                self._invalidate_for_edit(path)
            return result

        if name not in self._CACHEABLE:
            # Side-effecting (run_command) or cheap/likely-to-change tools
            # (status, diff, branch, history, list_directory) -- always pass
            # straight through, never cached.
            return self._wrapped.execute(name, arguments)

        key = self._cache_key(name, arguments)
        if key in self._cache:
            return self._cache[key]

        result = self._wrapped.execute(name, arguments)
        self._cache[key] = result
        self._cache_args[key] = arguments
        return result

    def _cache_key(self, name: str, arguments: dict) -> tuple:
        """Hashable, order-independent key for (tool name, arguments).

        Note: constructing `tuple(sorted(arguments.items()))` does NOT itself
        raise `TypeError` when a value is unhashable (e.g. a list/dict) -- a
        tuple happily *contains* an unhashable item; the `TypeError` only
        fires when that tuple is later *hashed* (e.g. used as a dict key).
        So we force the hash to be evaluated here, inside the try, rather
        than only wrapping the tuple() call -- otherwise the exception would
        surface later in `execute()`'s `key in self._cache` lookup instead
        of being caught here.
        """
        try:
            key = (name, tuple(sorted(arguments.items())))
            hash(key)
            return key
        except TypeError:
            # An argument value wasn't hashable (e.g. a list/dict) -- fall
            # back to a string-based key so this degrades to "always treat
            # as a fresh call" rather than crashing.
            return (name, repr(sorted(arguments.items())))

    def _invalidate_for_edit(self, path: str | None) -> None:
        if not path:
            return

        for key in list(self._cache):
            cached_name = key[0]
            if cached_name == "read_file":
                cached_args = self._cache_args.get(key, {})
                if cached_args.get("path") == path:
                    del self._cache[key]
                    self._cache_args.pop(key, None)
            elif cached_name in (
                "search_files",
                "find_symbol",
                "repository_tree",
                "map_flow",
                "trace_flow",
                "find_references",
                "find_callers",
                "find_usages",
                "explain_architecture",
                "find_related_tests",
            ):
                # Repo-wide/relationship queries whose results could plausibly
                # change from any file edit -- drop all of them, regardless
                # of their specific arguments.
                del self._cache[key]
                self._cache_args.pop(key, None)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_args.clear()

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
        return self._wrapped.register(spec)

    def register_many(self, specs) -> None:
        return self._wrapped.register_many(specs)

    def subset(self, names: list[str]) -> "CachedToolRegistry":
        """Return a fresh, independently-cached `CachedToolRegistry` wrapping
        `self._wrapped.subset(names)`. The subset does not share `self._cache`
        with the parent -- simpler and safer than trying to share/filter a
        cache across differently-scoped views."""
        return CachedToolRegistry(self._wrapped.subset(names))
