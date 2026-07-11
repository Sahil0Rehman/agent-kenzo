"""Pure-Python structural analysis over the SAME `SymbolIndex`/
`DependencyGraph` Phase 4 already builds -- no LLM call in this module.

Mirrors the established pattern of Phase 4/5's graph-aware/semantic
retrieval: pre-compute structural FACTS deterministically in Python, then
hand them to an LLM (here, the three Phase 10 worker types) to INTERPRET,
rather than asking the model to rediscover "is this file too large" or "is
there a dependency cycle" via unreliable tool exploration. This makes
`kenzo improve --analyze` fast, deterministic, and testable without any
model call at all -- the worker layer only adds judgment (severity,
phrasing, prioritization) on top of facts that are already known to be true.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from kenzo.intelligence.test_discovery import find_related_tests

_DEFAULT_LARGE_FILE_THRESHOLD = 400
_DEFAULT_MIN_DUPLICATE_LINES = 4
_DEFAULT_MAX_CYCLES = 20

_TEST_PATH_MARKERS = ("test", "spec", "__tests__")
_ENTRY_POINT_BASENAMES = {"__init__.py", "__main__.py", "app.py", "main.py", "index.ts", "extension.ts"}


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return any(marker in lowered for marker in _TEST_PATH_MARKERS)


def find_large_files(symbol_index: Any, repo_root: str, threshold_lines: int = _DEFAULT_LARGE_FILE_THRESHOLD) -> list[dict]:
    """Every indexed file whose real line count exceeds `threshold_lines`,
    largest first. Reads the actual file (not a symbol-count proxy) since a
    file can be "large" in ways the AST-derived symbol table doesn't fully
    capture (long comments, generated code, data literals)."""
    root = Path(repo_root)
    results: list[dict] = []
    for path in sorted(symbol_index.files.keys()):
        full = root / path
        try:
            line_count = len(full.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
        if line_count > threshold_lines:
            results.append({"file": path, "lines": line_count})
    results.sort(key=lambda r: r["lines"], reverse=True)
    return results


def find_dependency_cycles(dependency_graph: Any, max_cycles: int = _DEFAULT_MAX_CYCLES) -> list[list[str]]:
    """Simple cycles in the FILE-level dependency graph (`dependency_graph
    .file_deps`, a plain adjacency list built from real resolved imports --
    see `kenzo.intelligence.dependency_graph`), found via a straightforward
    DFS with a recursion-stack color marking. Capped at `max_cycles` so a
    pathologically tangled repo can't make this run unbounded; cycles found
    first (by sorted file-path traversal order, for determinism) are kept.
    """
    file_deps: dict[str, list[str]] = dependency_graph.file_deps or {}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {node: WHITE for node in file_deps}
    cycles: list[list[str]] = []
    path_stack: list[str] = []

    def visit(node: str) -> None:
        if len(cycles) >= max_cycles:
            return
        color[node] = GRAY
        path_stack.append(node)
        for neighbor in file_deps.get(node, []):
            if len(cycles) >= max_cycles:
                break
            if neighbor not in color:
                continue  # a dep resolved to something outside the indexed file set
            if color[neighbor] == GRAY:
                # Found a back-edge -- the cycle is the stack from neighbor's
                # first occurrence through the current node.
                start = path_stack.index(neighbor)
                cycle = path_stack[start:] + [neighbor]
                if cycle not in cycles:
                    cycles.append(cycle)
            elif color[neighbor] == WHITE:
                visit(neighbor)
        path_stack.pop()
        color[node] = BLACK

    for node in sorted(file_deps.keys()):
        if len(cycles) >= max_cycles:
            break
        if color.get(node) == WHITE:
            visit(node)

    return cycles


def find_unused_modules(symbol_index: Any, dependency_graph: Any) -> list[str]:
    """Indexed files with zero file-level dependents (nothing imports them)
    AND zero class-level dependents, excluding conventional entry points and
    test files. HEURISTIC, not proof of dead code -- a dynamically-imported
    plugin, a CLI entry point this heuristic doesn't recognize, or a
    file loaded by string/reflection would all false-positive here; treat
    the result as "worth a human/LLM second look", not "safe to delete".
    """
    file_deps: dict[str, list[str]] = dependency_graph.file_deps or {}
    file_dependents: dict[str, int] = {path: 0 for path in file_deps}
    for _source, targets in file_deps.items():
        for target in targets:
            if target in file_dependents:
                file_dependents[target] += 1

    class_dependents = dependency_graph.class_dependents or {}
    classes_by_file: dict[str, list[str]] = {}
    for file_path, cls in symbol_index.all_classes():
        classes_by_file.setdefault(file_path, []).append(cls.name)

    unused: list[str] = []
    for path in sorted(file_deps.keys()):
        if _is_test_path(path):
            continue
        if Path(path).name in _ENTRY_POINT_BASENAMES:
            continue
        if file_dependents.get(path, 0) > 0:
            continue
        has_class_dependent = any(class_dependents.get(name) for name in classes_by_file.get(path, []))
        if has_class_dependent:
            continue
        unused.append(path)
    return unused


def _normalize_body(lines: list[str]) -> str:
    normalized = [line.strip() for line in lines]
    return "\n".join(line for line in normalized if line)


def find_duplicate_functions(
    symbol_index: Any, repo_root: str, min_lines: int = _DEFAULT_MIN_DUPLICATE_LINES
) -> list[dict]:
    """Functions/methods whose NORMALIZED body text (whitespace-per-line
    stripped, blank lines dropped) is byte-identical to another function/
    method's body somewhere else in the repo. Deliberately exact-match, not
    fuzzy -- a real duplicate a self-improvement pass should consolidate, not
    a coincidental structural similarity. `min_lines` (of non-blank
    normalized content) filters out trivially short bodies (`return None`,
    `pass`, a one-line getter) that would otherwise dominate the results
    with noise.
    """
    root = Path(repo_root)
    file_cache: dict[str, list[str]] = {}
    by_hash: dict[str, list[dict]] = {}

    def _lines_for(path: str) -> Optional[list[str]]:
        if path not in file_cache:
            try:
                file_cache[path] = (root / path).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                file_cache[path] = None  # type: ignore[assignment]
        return file_cache[path]

    def _consider(path: str, qualified_name: str, location: Any) -> None:
        lines = _lines_for(path)
        if not lines:
            return
        # location.start_line (1-indexed) points at the `def ...:` line
        # itself -- skip it so two functions with identical BODIES but
        # different NAMES/signatures still hash the same. (Does not handle
        # a signature that wraps across multiple lines -- an accepted
        # limitation of this heuristic, not a correctness requirement.)
        start = min(len(lines), location.start_line)
        end = min(len(lines), location.end_line)
        if start >= end:
            return
        body = _normalize_body(lines[start:end])
        if len([line for line in body.splitlines() if line]) < min_lines:
            return
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()
        by_hash.setdefault(digest, []).append({"file": path, "name": qualified_name, "start_line": location.start_line, "end_line": location.end_line})

    for path, func in symbol_index.all_functions():
        _consider(path, func.qualified_name or func.name, func.location)
    for path, cls in symbol_index.all_classes():
        for method in cls.methods:
            _consider(path, method.qualified_name or f"{cls.name}.{method.name}", method.location)

    duplicates: list[dict] = []
    for members in by_hash.values():
        # Only a genuine duplicate if it appears at 2+ DISTINCT (file, name)
        # locations -- guards against one function being visited twice by
        # some future caller of this function, and against a class method
        # trivially "duplicating" itself.
        distinct = {(m["file"], m["name"]) for m in members}
        if len(distinct) >= 2:
            duplicates.append({"locations": members, "line_count": members[0]["end_line"] - members[0]["start_line"] + 1})

    duplicates.sort(key=lambda d: d["line_count"], reverse=True)
    return duplicates


def find_test_gaps(symbol_index: Any) -> list[dict]:
    """Non-test source files with at least one class/function but ZERO
    related test files found by `kenzo.intelligence.test_discovery
    .find_related_tests` -- the exact same naming-convention+reference
    heuristic the `find_related_tests` tool already uses, just run across
    every indexed file instead of one at a time."""
    gaps: list[dict] = []
    for path, node in sorted(symbol_index.files.items()):
        if _is_test_path(path):
            continue
        if not node.classes and not node.functions:
            continue
        tests = find_related_tests(symbol_index, path)
        if not tests:
            gaps.append({"file": path, "classes": [c.name for c in node.classes], "functions": [f.name for f in node.functions if not f.is_method]})
    return gaps


def run_full_analysis(
    symbol_index: Any,
    dependency_graph: Any,
    repo_root: str,
    *,
    large_file_threshold: int = _DEFAULT_LARGE_FILE_THRESHOLD,
    min_duplicate_lines: int = _DEFAULT_MIN_DUPLICATE_LINES,
) -> dict:
    """Everything above, combined into one structured report -- what
    `kenzo improve --analyze` renders, and what `SelfImprovementAgent`
    hands to the three Phase 10 workers as pre-computed context."""
    return {
        "large_files": find_large_files(symbol_index, repo_root, large_file_threshold),
        "dependency_cycles": find_dependency_cycles(dependency_graph),
        "unused_modules": find_unused_modules(symbol_index, dependency_graph),
        "duplicate_functions": find_duplicate_functions(symbol_index, repo_root, min_duplicate_lines),
        "test_gaps": find_test_gaps(symbol_index),
    }


def render_issue_summary(facts: dict) -> str:
    """Plain-text "Issues found: 1. ... 2. ... 3. ..." rendering -- the
    `kenzo improve --analyze` spec example's exact shape."""
    lines: list[str] = []
    n = 0
    for cycle in facts.get("dependency_cycles", []):
        n += 1
        lines.append(f"{n}. Dependency cycle: {' -> '.join(cycle)}")
    for dup in facts.get("duplicate_functions", []):
        n += 1
        locations = ", ".join(f"{loc['file']}:{loc['name']}" for loc in dup["locations"])
        lines.append(f"{n}. Duplicate logic ({dup['line_count']} lines) in {locations}")
    for large in facts.get("large_files", []):
        n += 1
        lines.append(f"{n}. Large file: {large['file']} ({large['lines']} lines)")
    for unused in facts.get("unused_modules", []):
        n += 1
        lines.append(f"{n}. Possibly unused module: {unused}")
    for gap in facts.get("test_gaps", []):
        n += 1
        lines.append(f"{n}. Missing tests for {gap['file']}")
    if not lines:
        return "Issues found: none -- analysis found no large files, dependency cycles, duplicate functions, unused modules, or test gaps under the current thresholds."
    return "Issues found:\n" + "\n".join(lines)
