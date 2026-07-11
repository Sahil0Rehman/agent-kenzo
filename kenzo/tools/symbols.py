"""Read-only Java symbol lookup tools, backed by `kenzo.memory.java_analyzer`'s
output as persisted by `RepositoryMemory` (`.kenzo/symbols.json`).

Two tools live here:

- `find_symbol`: look up a class/interface or method by name (exact match
  preferred, substring fallback).
- `map_flow`: given an HTTP endpoint, find the controller that serves it and
  walk the naive `calls` graph the analyzer extracted (controller -> service
  -> repository, etc.) as far as it can be traced.

Both are read-only and require nothing beyond an already-indexed repo (see
`RepositoryMemory.load_symbols`), so -- like `kenzo/tools/git.py` -- no
permissions gating is needed here.
"""

from __future__ import annotations

import functools
from typing import Any

from kenzo.config.settings import PermissionsConfig
from kenzo.core.schemas import ToolResult, ToolSpec
from kenzo.memory.repository import RepositoryMemory

_NO_SYMBOLS_ERROR = (
    "No symbols indexed yet for this repository. Run `kenzo index` first "
    "(Java repositories only, for now)."
)

_MAX_CHAIN_LEN = 6


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _load_symbols(repo_root: str) -> list[dict[str, Any]] | None:
    """Returns the flat list of symbol dicts, or None if nothing is indexed
    yet (no `.kenzo/symbols.json`, or it's unreadable/malformed --
    `RepositoryMemory.load_symbols()` already handles that gracefully)."""
    symbols_data = RepositoryMemory(repo_root).load_symbols()
    if symbols_data is None:
        return None
    return symbols_data.get("symbols") or []


def _normalize_path(path: str | None) -> str | None:
    """Strip a single trailing slash (but never collapse "/" itself)."""
    if not path:
        return path
    if path.endswith("/") and len(path) > 1:
        return path[:-1]
    return path


# --------------------------------------------------------------------------
# find_symbol
# --------------------------------------------------------------------------


def _method_match_entry(symbol: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "method",
        "name": method.get("name"),
        "class": symbol.get("name"),
        "file": symbol.get("file"),
        "type": symbol.get("type"),
        "annotations": method.get("annotations"),
        "endpoint": method.get("endpoint"),
    }


def _collect_matches(
    symbols: list[dict[str, Any]], name_lower: str, *, substring: bool
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for symbol in symbols:
        symbol_name_lower = (symbol.get("name") or "").lower()
        class_hit = (name_lower in symbol_name_lower) if substring else (symbol_name_lower == name_lower)
        if class_hit:
            matches.append(symbol)
        for method in symbol.get("methods") or []:
            method_name_lower = (method.get("name") or "").lower()
            method_hit = (
                (name_lower in method_name_lower) if substring else (method_name_lower == name_lower)
            )
            if method_hit:
                matches.append(_method_match_entry(symbol, method))
    return matches


def _summarize_find(name: str, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return f"No symbol matching '{name}' found."

    class_entries = [m for m in matches if m.get("kind") != "method"]
    method_entries = [m for m in matches if m.get("kind") == "method"]

    parts: list[str] = []
    if class_entries:
        if len(class_entries) == 1:
            entry = class_entries[0]
            kind_word = entry.get("kind") or "class"
            method_count = len(entry.get("methods") or [])
            parts.append(
                f"Found {kind_word} '{entry.get('name')}' ({entry.get('type') or 'unknown'}) "
                f"in {entry.get('file')} with {method_count} method(s)."
            )
        else:
            names = ", ".join(entry.get("name") or "?" for entry in class_entries)
            parts.append(
                f"Found {len(class_entries)} class(es)/interface(s) matching '{name}': {names}."
            )
    if method_entries:
        seen_classes: list[str] = []
        for entry in method_entries:
            cls = entry.get("class")
            if cls not in seen_classes:
                seen_classes.append(cls)
        parts.append(
            f"Found {len(method_entries)} method(s) named '{name}' across {', '.join(seen_classes)}."
        )
    return " ".join(parts)


def find_symbol(name: str, repo_root: str) -> ToolResult:
    """Look up a class/interface or method by name across the indexed
    Java symbols. Exact (case-insensitive) matches win when any exist;
    otherwise falls back to a forgiving substring match.
    """
    symbols = _load_symbols(repo_root)
    if symbols is None:
        return ToolResult(tool_name="find_symbol", success=False, error=_NO_SYMBOLS_ERROR)

    name_lower = name.lower()
    matches = _collect_matches(symbols, name_lower, substring=False)
    if not matches:
        matches = _collect_matches(symbols, name_lower, substring=True)

    data = {"matches": matches, "symbol": matches[0] if matches else None}
    output = _summarize_find(name, matches)
    return ToolResult(tool_name="find_symbol", success=True, output=output, data=data)


# --------------------------------------------------------------------------
# map_flow
# --------------------------------------------------------------------------


def _find_endpoint_symbol(
    symbols: list[dict[str, Any]], endpoint: str
) -> dict[str, Any] | None:
    """Find the best symbol whose `endpoints` list matches `endpoint`,
    preferring type=="controller" among equally-good candidates, and
    preferring exact > normalized (trailing-slash-insensitive) > substring
    matches overall."""
    target_norm = _normalize_path(endpoint)
    endpoint_lower = endpoint.lower()

    tiers: list[list[dict[str, Any]]] = [
        [s for s in symbols if endpoint in (s.get("endpoints") or [])],
        [
            s
            for s in symbols
            if any(_normalize_path(e) == target_norm for e in (s.get("endpoints") or []))
        ],
        [
            s
            for s in symbols
            if any(endpoint_lower in (e or "").lower() for e in (s.get("endpoints") or []))
        ],
    ]

    for candidates in tiers:
        if not candidates:
            continue
        for candidate in candidates:
            if candidate.get("type") == "controller":
                return candidate
        return candidates[0]
    return None


def _find_method_for_endpoint(symbol: dict[str, Any], endpoint: str) -> dict[str, Any] | None:
    """Find the specific method on `symbol` whose own endpoint matches, using
    the same exact -> normalized -> substring tiering as the class-level
    search. Returns None if no individual method can be pinned down (e.g. the
    match only came from the class-level base_path)."""
    target_norm = _normalize_path(endpoint)
    endpoint_lower = endpoint.lower()
    methods = symbol.get("methods") or []

    for method in methods:
        if method.get("endpoint") == endpoint:
            return method
    for method in methods:
        method_endpoint = method.get("endpoint")
        if method_endpoint and _normalize_path(method_endpoint) == target_norm:
            return method
    for method in methods:
        method_endpoint = method.get("endpoint")
        if method_endpoint and endpoint_lower in method_endpoint.lower():
            return method
    return None


def _lookup_symbol_by_name(
    symbols: list[dict[str, Any]], target_name: str | None
) -> dict[str, Any] | None:
    target_lower = (target_name or "").lower()
    if not target_lower:
        return None
    for symbol in symbols:
        if (symbol.get("name") or "").lower() == target_lower:
            return symbol
    for symbol in symbols:
        if target_lower in (symbol.get("name") or "").lower():
            return symbol
    return None


def _lookup_method_by_name(
    symbol: dict[str, Any], method_name: str | None
) -> dict[str, Any] | None:
    method_lower = (method_name or "").lower()
    if not method_lower:
        return None
    methods = symbol.get("methods") or []
    for method in methods:
        if (method.get("name") or "").lower() == method_lower:
            return method
    for method in methods:
        if method_lower in (method.get("name") or "").lower():
            return method
    return None


def _render_chain(chain: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, hop in enumerate(chain):
        role = (hop.get("role") or "unknown").capitalize()
        cls = hop.get("class")
        method = hop.get("method")
        label = f"{role}: {cls}.{method}()" if method else f"{role}: {cls}"
        lines.append(label if i == 0 else f"  -> {label}")
    if len(chain) == 1:
        lines.append("(could not trace further -- no resolvable calls found)")
    return "\n".join(lines)


def map_flow(endpoint: str, repo_root: str) -> ToolResult:
    """Trace an HTTP endpoint through its controller -> service -> repository
    (etc.) call chain, using the analyzer's naive `calls` graph. Succeeds as
    soon as a starting controller/endpoint match is found, even if the
    resulting chain can only be traced one hop deep.
    """
    symbols = _load_symbols(repo_root)
    if symbols is None:
        return ToolResult(tool_name="map_flow", success=False, error=_NO_SYMBOLS_ERROR)

    start_symbol = _find_endpoint_symbol(symbols, endpoint)
    if start_symbol is None:
        return ToolResult(
            tool_name="map_flow",
            success=False,
            error=f"No controller found for endpoint '{endpoint}' in the indexed symbols.",
        )

    starting_method = _find_method_for_endpoint(start_symbol, endpoint)

    chain: list[dict[str, Any]] = [
        {
            "role": start_symbol.get("type") or "unknown",
            "class": start_symbol.get("name"),
            "method": starting_method.get("name") if starting_method else None,
            "file": start_symbol.get("file"),
        }
    ]

    current_method = starting_method
    while current_method is not None and len(chain) < _MAX_CHAIN_LEN:
        calls = current_method.get("calls") or []
        if not calls:
            break
        first_call = calls[0]
        target_type = first_call.get("target_type")
        call_method_name = first_call.get("method")
        if not target_type:
            break

        target_symbol = _lookup_symbol_by_name(symbols, target_type)
        if target_symbol is None:
            break

        next_method = _lookup_method_by_name(target_symbol, call_method_name)
        chain.append(
            {
                "role": target_symbol.get("type") or "unknown",
                "class": target_symbol.get("name"),
                "method": next_method.get("name") if next_method else None,
                "file": target_symbol.get("file"),
            }
        )
        if next_method is None:
            break
        current_method = next_method

    data = {"chain": chain}
    output = _render_chain(chain)
    return ToolResult(tool_name="map_flow", success=True, output=output, data=data)


# --------------------------------------------------------------------------
# ToolSpec registration
# --------------------------------------------------------------------------


def get_tool_specs(repo_root: str, permissions: PermissionsConfig) -> list[ToolSpec]:
    """Build the read-only symbol-lookup ToolSpecs, bound to this repo_root.

    `permissions` is accepted for interface symmetry with the other tool
    modules -- both tools here are read-only lookups over an already-built
    `.kenzo/symbols.json`, so there's nothing to gate.
    """
    del permissions  # unused: both operations in this module are read-only
    return [
        ToolSpec(
            name="find_symbol",
            description=(
                "Find a Java class/interface or method by name in the indexed repository "
                "symbols (e.g. find_symbol('logout') finds every method literally named "
                "'logout', across all classes)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Class/interface name or method name to look up (case-insensitive).",
                    }
                },
                "required": ["name"],
            },
            func=functools.partial(find_symbol, repo_root=repo_root),
        ),
        ToolSpec(
            name="map_flow",
            description=(
                "Trace an HTTP endpoint (e.g. '/api/auth/logout') through its "
                "controller -> service -> repository call chain, using the indexed "
                "repository symbols."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "The HTTP endpoint path to trace, e.g. '/api/auth/logout'.",
                    }
                },
                "required": ["endpoint"],
            },
            func=functools.partial(map_flow, repo_root=repo_root),
        ),
    ]
