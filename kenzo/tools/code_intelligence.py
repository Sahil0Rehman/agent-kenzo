"""Phase 4 code-intelligence tools: real AST-derived navigation, replacing
regex/keyword guessing with the `kenzo.intelligence` package's parsed symbol
index, reference graph, and dependency graph.

Additive to Phase 2/3's `find_symbol`/`map_flow` (kenzo.tools.symbols) --
those are untouched and keep working exactly as before. These tools are a
strictly better alternative built on genuinely parsed code, registered
alongside the old ones so nothing that worked before stops working.

Every tool here is read-only and loads its data fresh from `.kenzo/` on each
call (mirroring kenzo.tools.symbols' pattern), so no live Python object needs
to cross the tool-registration boundary.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

from kenzo.core.schemas import ToolResult, ToolSpec
from kenzo.intelligence.dependency_graph import DependencyGraph
from kenzo.intelligence.models import ClassSymbol, FunctionSymbol
from kenzo.intelligence.reference_graph import ReferenceGraph
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.intelligence.test_discovery import find_related_tests as _find_related_tests
from kenzo.memory.repository import RepositoryMemory

_NO_INDEX_ERROR = (
    "No code intelligence index found for this repository. Run `kenzo index` first "
    "(or wait for the agent runtime to build it automatically at startup)."
)

_MAX_CHAIN_LEN = 8


def _load_intelligence(
    repo_root: str,
) -> tuple[Optional[SymbolIndex], Optional[ReferenceGraph], Optional[DependencyGraph]]:
    memory = RepositoryMemory(repo_root)
    symbol_index = SymbolIndex.load(repo_root, str(memory.code_index_path))
    if symbol_index is None:
        return None, None, None
    reference_graph = ReferenceGraph.load(symbol_index, str(memory.references_path))
    dependency_graph = DependencyGraph.load(symbol_index, str(memory.dependencies_path))
    return symbol_index, reference_graph, dependency_graph


def _method_lookup(symbol_index: SymbolIndex) -> dict[str, dict[str, Any]]:
    """qualified_name ("Class.method") -> {"class", "file", "stereotype"}."""
    lookup: dict[str, dict[str, Any]] = {}
    for file_path, cls in symbol_index.all_classes():
        for method in cls.methods:
            key = method.qualified_name or f"{cls.name}.{method.name}"
            lookup[key] = {"class": cls.name, "file": file_path, "stereotype": cls.stereotype}
    return lookup


def _normalize_endpoint(path: str) -> str:
    if len(path) > 1 and path.endswith("/"):
        return path[:-1]
    return path


def _find_endpoint_start(
    symbol_index: SymbolIndex, endpoint: str
) -> Optional[tuple[str, ClassSymbol, FunctionSymbol]]:
    """Tiered match (exact -> normalized -> substring) mirroring
    kenzo.tools.symbols.map_flow's forgiving philosophy, but returning the
    specific (file, class, method) hit rather than just a class."""
    target_norm = _normalize_endpoint(endpoint)
    endpoint_lower = endpoint.lower()

    tiers: list[list[tuple[str, ClassSymbol, FunctionSymbol]]] = [[], [], []]
    for file_path, cls in symbol_index.all_classes():
        for method in cls.methods:
            if not method.endpoint:
                continue
            if method.endpoint == endpoint:
                tiers[0].append((file_path, cls, method))
            elif _normalize_endpoint(method.endpoint) == target_norm:
                tiers[1].append((file_path, cls, method))
            elif endpoint_lower in method.endpoint.lower():
                tiers[2].append((file_path, cls, method))

    for tier in tiers:
        if not tier:
            continue
        for hit in tier:
            if hit[1].stereotype == "controller":
                return hit
        return tier[0]
    return None


def _find_method_start(
    symbol_index: SymbolIndex, target: str
) -> Optional[tuple[str, Optional[ClassSymbol], FunctionSymbol]]:
    """Resolve a "ClassName.method" or bare method/function name to a
    starting (file, class_or_None, method) triple."""
    if "." in target:
        class_part, _, method_part = target.rpartition(".")
        for file_path, cls in symbol_index.find_class(class_part):
            for method in cls.methods:
                if method.name.lower() == method_part.lower():
                    return file_path, cls, method

    hits = symbol_index.find_function_or_method(target.rpartition(".")[-1] if "." in target else target)
    if hits:
        hit = hits[0]
        cls = None
        if hit["class"]:
            class_matches = symbol_index.find_class(hit["class"])
            cls = next((c for p, c in class_matches if p == hit["file"]), None)
        return hit["file"], cls, hit["function"]
    return None


def trace_flow(target: str, repo_root: str) -> ToolResult:
    """Trace `target` (an HTTP endpoint like "/logout", or a qualified
    method reference like "UserService.createUser") forward through its
    real, AST-resolved call chain (e.g. Controller -> Service ->
    Repository), using the reference graph instead of naive text matching.
    """
    symbol_index, reference_graph, _ = _load_intelligence(repo_root)
    if symbol_index is None or reference_graph is None:
        return ToolResult(tool_name="trace_flow", success=False, error=_NO_INDEX_ERROR)

    looks_like_endpoint = target.startswith("/") or ("." not in target and "/" not in target and target.islower())
    start = None
    is_endpoint = False
    if target.startswith("/"):
        start = _find_endpoint_start(symbol_index, target)
        is_endpoint = start is not None
    if start is None:
        start = _find_method_start(symbol_index, target)
    if start is None and not target.startswith("/"):
        start = _find_endpoint_start(symbol_index, target)
        is_endpoint = start is not None

    if start is None:
        return ToolResult(
            tool_name="trace_flow",
            success=False,
            error=f"No endpoint or method found matching '{target}' in the indexed code.",
        )

    file_path, cls, method = start
    method_lookup = _method_lookup(symbol_index)

    chain: list[dict[str, Any]] = [
        {
            "role": (cls.stereotype if cls else None) or "unknown",
            "class": cls.name if cls else None,
            "method": method.name,
            "file": file_path,
        }
    ]

    current_key = method.qualified_name or (f"{cls.name}.{method.name}" if cls else method.name)
    visited = {current_key}
    while len(chain) < _MAX_CHAIN_LEN:
        edges = reference_graph.callees.get(current_key, [])
        if not edges:
            break
        next_edge = edges[0]
        callee_key = next_edge["callee"]
        if callee_key in visited:
            break
        visited.add(callee_key)
        info = method_lookup.get(callee_key, {})
        callee_class, callee_method = callee_key.rsplit(".", 1) if "." in callee_key else (None, callee_key)
        chain.append(
            {
                "role": info.get("stereotype") or "unknown",
                "class": info.get("class", callee_class),
                "method": callee_method,
                "file": info.get("file", next_edge.get("callee_file")),
            }
        )
        current_key = callee_key

    # ASCII-only rendering (no unicode arrows): matches kenzo.tools.symbols'
    # map_flow._render_chain style, and avoids a real crash risk -- a
    # non-UTF8 terminal code page (the Windows default, cp1252) cannot
    # encode "->"-style unicode arrows and raises UnicodeEncodeError on print.
    lines: list[str] = []
    if is_endpoint:
        lines.append(f"HTTP Endpoint {target}")
    for i, hop in enumerate(chain):
        role = (hop["role"] or "unknown").capitalize()
        label = f"{hop['class']}.{hop['method']}()" if hop["class"] else f"{hop['method']}()"
        rendered = f"{role}: {label}"
        lines.append(rendered if (i == 0 and not is_endpoint) else f"  -> {rendered}")
    if len(chain) == 1:
        lines.append("(could not trace further -- no resolvable calls found)")

    return ToolResult(
        tool_name="trace_flow",
        success=True,
        output="\n".join(lines),
        data={"chain": chain, "target": target, "is_endpoint": is_endpoint},
    )


def find_references(symbol: str, repo_root: str) -> ToolResult:
    """Find every real, AST-resolved call site referencing `symbol` (a bare
    method/function name, or "ClassName.method")."""
    symbol_index, reference_graph, _ = _load_intelligence(repo_root)
    if symbol_index is None or reference_graph is None:
        return ToolResult(tool_name="find_references", success=False, error=_NO_INDEX_ERROR)

    refs = reference_graph.find_references(symbol)
    output = f"{len(refs)} reference(s) to '{symbol}'." if refs else f"No references found to '{symbol}'."
    return ToolResult(tool_name="find_references", success=True, output=output, data={"references": refs})


def find_callers(method: str, repo_root: str) -> ToolResult:
    """Find every real caller of `method` ("ClassName.method" or bare name)."""
    symbol_index, reference_graph, _ = _load_intelligence(repo_root)
    if symbol_index is None or reference_graph is None:
        return ToolResult(tool_name="find_callers", success=False, error=_NO_INDEX_ERROR)

    callers = reference_graph.find_callers(method)
    output = f"{len(callers)} caller(s) of '{method}'." if callers else f"No callers found for '{method}'."
    return ToolResult(tool_name="find_callers", success=True, output=output, data={"callers": callers})


def find_usages(class_name: str, repo_root: str) -> ToolResult:
    """Find every place `class_name` is used: as a dependency, a field/
    parameter type, an extends/implements target, or a resolved call target."""
    symbol_index, reference_graph, _ = _load_intelligence(repo_root)
    if symbol_index is None or reference_graph is None:
        return ToolResult(tool_name="find_usages", success=False, error=_NO_INDEX_ERROR)

    usages = reference_graph.find_usages(class_name)
    output = f"{len(usages)} usage(s) of '{class_name}'." if usages else f"No usages found for '{class_name}'."
    return ToolResult(tool_name="find_usages", success=True, output=output, data={"usages": usages})


def explain_architecture(module: str = "", repo_root: str = "") -> ToolResult:
    """Summarize the repository's architecture: classes grouped by module
    and stereotype (controllers/services/repositories/entities/...)."""
    symbol_index, _, dependency_graph = _load_intelligence(repo_root)
    if symbol_index is None or dependency_graph is None:
        return ToolResult(tool_name="explain_architecture", success=False, error=_NO_INDEX_ERROR)

    result = dependency_graph.explain_architecture(module=module or None)
    summary = result.get("summary") or "(no classes found for this module)"
    return ToolResult(tool_name="explain_architecture", success=True, output=summary, data=result)


def find_related_tests_tool(file_path: str, repo_root: str) -> ToolResult:
    """Find test files likely covering `file_path` (naming convention +
    import-reference heuristics)."""
    symbol_index, _, _ = _load_intelligence(repo_root)
    if symbol_index is None:
        return ToolResult(tool_name="find_related_tests", success=False, error=_NO_INDEX_ERROR)

    tests = _find_related_tests(symbol_index, file_path)
    output = (
        f"Found {len(tests)} related test file(s) for '{file_path}': {', '.join(tests)}"
        if tests
        else f"No related test files found for '{file_path}'."
    )
    return ToolResult(tool_name="find_related_tests", success=True, output=output, data={"tests": tests})


# --------------------------------------------------------------------------
# ToolSpec registration
# --------------------------------------------------------------------------


def get_tool_specs(repo_root: str, permissions: Any) -> list[ToolSpec]:
    """Build the code-intelligence ToolSpecs, bound to this repo_root.

    `permissions` is accepted for interface symmetry with the other tool
    modules -- every tool here is a read-only lookup over already-built
    `.kenzo/` intelligence artifacts, so there's nothing to gate.
    """
    del permissions  # unused: all operations in this module are read-only
    return [
        ToolSpec(
            name="trace_flow",
            description=(
                "Trace an HTTP endpoint (e.g. '/api/auth/logout') or a qualified method "
                "(e.g. 'UserService.createUser') forward through its real, AST-resolved "
                "call chain -- e.g. Controller -> Service -> Repository. More accurate "
                "than map_flow, which uses a naive regex-based index."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "An HTTP endpoint path or a 'ClassName.method' reference.",
                    }
                },
                "required": ["target"],
            },
            func=functools.partial(trace_flow, repo_root=repo_root),
        ),
        ToolSpec(
            name="find_references",
            description="Find every real, AST-resolved call site referencing a method or function by name.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "A bare method/function name, or 'ClassName.method'.",
                    }
                },
                "required": ["symbol"],
            },
            func=functools.partial(find_references, repo_root=repo_root),
        ),
        ToolSpec(
            name="find_callers",
            description="Find every real caller of a specific method ('ClassName.method' or bare name).",
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "'ClassName.method' or a bare method name.",
                    }
                },
                "required": ["method"],
            },
            func=functools.partial(find_callers, repo_root=repo_root),
        ),
        ToolSpec(
            name="find_usages",
            description=(
                "Find every place a class is used: as a dependency, a field/parameter type, "
                "an extends/implements target, or a resolved call target."
            ),
            parameters={
                "type": "object",
                "properties": {"class_name": {"type": "string", "description": "The class or interface name."}},
                "required": ["class_name"],
            },
            func=functools.partial(find_usages, repo_root=repo_root),
        ),
        ToolSpec(
            name="explain_architecture",
            description=(
                "Summarize the repository's architecture: classes grouped by module and "
                "stereotype (controllers/services/repositories/entities/components/"
                "configurations). Optionally restrict to one module."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": "Optional module name to restrict to (default: all modules).",
                        "default": "",
                    }
                },
                "required": [],
            },
            func=functools.partial(explain_architecture, repo_root=repo_root),
        ),
        ToolSpec(
            name="find_related_tests",
            description=(
                "Find test files likely covering a given source file, using naming "
                "conventions (FooTest.java, test_foo.py, foo.spec.ts, ...) and import "
                "references. Use this before running tests to know what to run."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Repository-relative path to the source file to find tests for.",
                    }
                },
                "required": ["file_path"],
            },
            func=functools.partial(find_related_tests_tool, repo_root=repo_root),
        ),
    ]
