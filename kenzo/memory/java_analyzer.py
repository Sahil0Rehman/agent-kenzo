"""Lightweight, regex/text-based Java symbol analyzer.

This is deliberately NOT a real Java parser (no tree-sitter, no JDK/javac
dependency). It's a "useful approximation" extractor of classes, interfaces,
methods, annotations, fields, and naive call/dependency relationships from
ordinary, conventionally-formatted Spring-Boot-style `.java` source, meant to
back two tools built on top of it later: `find_symbol` and `map_flow`.

Mirrors the repo-walking pattern established in `kenzo.memory.indexer`
(os.walk with `dirnames` pruned in place against a configurable
`ignored_dirs` set, deterministic sorted+capped file collection) but adds
Java-specific text analysis on top.

Regex-based analysis will misparse some edge cases (generics-heavy class
declarations, multi-line annotations, annotation arguments spanning multiple
lines, etc). That's an accepted tradeoff: this module prioritizes handling
ordinary Spring Boot code correctly and degrading gracefully (skipping what
it can't confidently parse, one file at a time) over handling every Java
syntax edge case.
"""

from __future__ import annotations

import bisect
import os
import re
from pathlib import Path
from typing import Any

from kenzo.config.settings import DEFAULT_IGNORED_DIRS

# --------------------------------------------------------------------------
# Regex patterns
# --------------------------------------------------------------------------

_PACKAGE_RE = re.compile(r"package\s+([\w.]+)\s*;")

_CLASS_RE = re.compile(
    r"(?:public\s+|private\s+|protected\s+)?"
    r"(?:static\s+)?"
    r"(?:final\s+|abstract\s+)?"
    r"(class|interface)\s+(\w+)"
    r"(?:\s+extends\s+(\w+))?"
    r"(?:\s+implements\s+([\w,\s]+))?"
)

# NOTE: the spec's field regex is `(?:private|protected|public)\s+(?:final\s+)?...`.
# We additionally allow an optional `static` modifier between visibility and
# `final` (e.g. `public static final String X`) since that ordering is
# extremely common in real Spring Boot code and this is a strict superset of
# what the spec pattern matches -- it doesn't change behavior for any field
# the original pattern already handled.
_FIELD_RE = re.compile(
    r"(?:private|protected|public)\s+(?:static\s+)?(?:final\s+)?"
    r"([\w<>\[\],.\s]+?)\s+(\w+)\s*;"
)

_METHOD_RE = re.compile(
    r"(?:public|protected|private)\s+(?:static\s+)?(?:final\s+)?"
    r"[\w<>\[\],\s]+?\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\s*\{"
)

_CALL_RE = re.compile(r"(\w+)\.(\w+)\s*\(")

# Non-greedy on the parenthesized args so two annotations sharing a line
# (e.g. `@Valid @NotNull`) don't get their argument spans merged together.
_ANNOTATION_RE = re.compile(r"@(\w+)(?:\((.*?)\))?")

_QUOTED_RE = re.compile(r'"([^"]*)"')

_MAPPING_ANNOTATIONS = {
    "GetMapping",
    "PostMapping",
    "PutMapping",
    "DeleteMapping",
    "RequestMapping",
}

# Stereotype priority: first matching annotation set wins.
_STEREOTYPE_PRIORITY: list[tuple[set[str], str]] = [
    ({"RestController", "Controller"}, "controller"),
    ({"Service"}, "service"),
    ({"Repository"}, "repository"),
    ({"Entity"}, "entity"),
    ({"Component"}, "component"),
]


# --------------------------------------------------------------------------
# Small text-scanning helpers (string/char-literal- and comment-aware)
# --------------------------------------------------------------------------


def _iter_code_chars(text: str, start: int = 0):
    """Yield (index, char) for every character in text[start:] that is NOT
    inside a string literal, char literal, line comment, or block comment.

    Not a real Java lexer -- just enough to keep brace-matching from getting
    thrown off by a stray `{`/`}`/`(`/`)` inside an ordinary string, char
    literal, or comment.
    """
    i = start
    n = len(text)
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    while i < n:
        c = text[i]
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if in_char:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == "'":
                in_char = False
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            in_line_comment = True
            i += 2
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            in_block_comment = True
            i += 2
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "'":
            in_char = True
            i += 1
            continue
        yield i, c
        i += 1


def _find_next_char(text: str, start: int, target: str) -> int | None:
    """Find the next occurrence of `target` char in "real code" (skipping
    string/char literals and comments), starting at `start`."""
    for idx, c in _iter_code_chars(text, start):
        if c == target:
            return idx
    return None


def _find_matching_brace(text: str, open_index: int) -> int | None:
    """text[open_index] must be '{'. Returns the index of the matching '}',
    or None if unmatched (unbalanced/truncated source)."""
    depth = 0
    for idx, c in _iter_code_chars(text, open_index):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _line_start_offsets(text: str) -> list[int]:
    """Character offset at which each line (as split by `str.splitlines()`)
    begins, so a match's absolute offset can be mapped back to a line index
    via bisect."""
    offsets = [0]
    for i, c in enumerate(text):
        if c == "\n":
            offsets.append(i + 1)
    return offsets


def _line_of_offset(line_offsets: list[int], offset: int) -> int:
    idx = bisect.bisect_right(line_offsets, offset) - 1
    return max(0, idx)


def _split_top_level_commas(params: str) -> list[str]:
    """Split a parameter list on commas, ignoring commas nested inside
    <...>, (...), or [...] (so generic types like `Map<String, Object>`
    don't get split in half)."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params:
        if ch in "<([":
            depth += 1
            current.append(ch)
        elif ch in ">)]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _parse_param(part: str) -> tuple[str, str] | None:
    """Parse a single "Type paramName" constructor parameter (optionally
    prefixed with `final` and/or a parameter annotation) into (type, name).
    Returns None if it doesn't look like a type+name pair."""
    part = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", part.strip()).strip()
    if not part:
        return None
    match = re.match(r"^(?:final\s+)?(.+?)\s+(\w+)$", part)
    if not match:
        return None
    param_type = re.sub(r"\s+", " ", match.group(1).strip())
    param_name = match.group(2)
    return param_type, param_name


def _join_path(base_path: str | None, sub_path: str | None) -> str:
    """Naive endpoint composition: base + sub, collapsing a doubled slash at
    the join point. Not full path normalization -- just enough to avoid the
    most common `/api/auth` + `/logout` -> `/api/auth//logout` glitch."""
    base = base_path or ""
    sub = sub_path or ""
    if base.endswith("/") and sub.startswith("/"):
        return base + sub[1:]
    return base + sub


# --------------------------------------------------------------------------
# Annotation-block collection
# --------------------------------------------------------------------------


def _collect_preceding_annotations(lines: list[str], decl_line_idx: int) -> list[str]:
    """Walk upward from the line before `decl_line_idx`, collecting raw
    annotation lines (stripped, each starting with "@") while permitting
    blank lines and comment lines (//, /*, *, */) to pass through without
    breaking the walk. Stops at the first line that is neither blank,
    a comment, nor an annotation.

    Comment lines (including javadoc `* @param ...` tag lines) are used only
    to keep walking past them -- they are never themselves added to the
    returned block, so javadoc @param/@return/@throws tags never get
    misread as real annotations.
    """
    block: list[str] = []
    i = decl_line_idx - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped == "" or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            i -= 1
            continue
        if stripped.startswith("@"):
            block.append(stripped)
            i -= 1
            continue
        break
    block.reverse()
    return block


def _extract_annotations(raw_lines: list[str]) -> list[tuple[str, str | None]]:
    """Parse every `@Name` / `@Name(args)` occurrence across `raw_lines`
    into (name, first_quoted_argument_or_None), in encounter order."""
    results: list[tuple[str, str | None]] = []
    for raw in raw_lines:
        for match in _ANNOTATION_RE.finditer(raw):
            name = match.group(1)
            args = match.group(2)
            first_quoted = None
            if args:
                quoted = _QUOTED_RE.search(args)
                if quoted:
                    first_quoted = quoted.group(1)
            results.append((name, first_quoted))
    return results


def _derive_stereotype(annotation_names: list[str]) -> str | None:
    names = set(annotation_names)
    for keys, stereotype in _STEREOTYPE_PRIORITY:
        if names & keys:
            return stereotype
    return None


# --------------------------------------------------------------------------
# Per-class extraction
# --------------------------------------------------------------------------


def _build_symbol(
    class_match: re.Match[str],
    text: str,
    lines: list[str],
    line_offsets: list[int],
    package: str,
    rel_posix: str,
) -> dict[str, Any]:
    kind = class_match.group(1)
    name = class_match.group(2)
    extends_ = class_match.group(3)
    implements_raw = class_match.group(4)
    implements_list = (
        [p.strip() for p in implements_raw.split(",") if p.strip()] if implements_raw else []
    )

    decl_line_idx = _line_of_offset(line_offsets, class_match.start())
    class_raw_block = _collect_preceding_annotations(lines, decl_line_idx)
    class_annotations = _extract_annotations(class_raw_block)
    annotation_names = [n for n, _ in class_annotations]

    base_path: str | None = None
    for ann_name, first_quoted in class_annotations:
        if ann_name == "RequestMapping" and first_quoted is not None:
            base_path = first_quoted
            break

    stereotype = _derive_stereotype(annotation_names)

    fields: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    known_deps: dict[str, str] = {}

    open_brace = _find_next_char(text, class_match.end(), "{")
    if open_brace is not None:
        close_brace = _find_matching_brace(text, open_brace)
        if close_brace is None:
            close_brace = len(text)
        body_start = open_brace + 1
        body_text = text[body_start:close_brace]

        # (a) Fields
        for field_match in _FIELD_RE.finditer(body_text):
            field_type = re.sub(r"\s+", " ", field_match.group(1).strip())
            field_name = field_match.group(2)
            abs_offset = body_start + field_match.start()
            field_line_idx = _line_of_offset(line_offsets, abs_offset)

            preceding = _collect_preceding_annotations(lines, field_line_idx)
            own_line = [lines[field_line_idx]] if 0 <= field_line_idx < len(lines) else []
            field_annotations = _extract_annotations(preceding + own_line)
            field_annotation_names = [n for n, _ in field_annotations]

            fields.append(
                {"name": field_name, "type": field_type, "annotations": field_annotation_names}
            )
            if "Autowired" in field_annotation_names:
                known_deps[field_name] = field_type

        # (b) Constructor (constructor-injection dependencies)
        ctor_re = re.compile(r"(?:public\s+)?" + re.escape(name) + r"\s*\(([^)]*)\)\s*\{")
        ctor_match = ctor_re.search(body_text)
        if ctor_match:
            for part in _split_top_level_commas(ctor_match.group(1)):
                parsed = _parse_param(part)
                if parsed:
                    param_type, param_name = parsed
                    known_deps.setdefault(param_name, param_type)

        # (c) Methods
        for method_match in _METHOD_RE.finditer(body_text):
            method_name = method_match.group(1)
            if method_name == name:
                # Looks like the constructor itself -- skip (spec: don't
                # double-count it as a regular method).
                continue

            abs_offset = body_start + method_match.start()
            method_line_idx = _line_of_offset(line_offsets, abs_offset)
            preceding = _collect_preceding_annotations(lines, method_line_idx)
            method_annotations = _extract_annotations(preceding)
            method_annotation_names = [n for n, _ in method_annotations]

            endpoint_sub: str | None = None
            has_mapping = False
            for ann_name, first_quoted in method_annotations:
                if ann_name in _MAPPING_ANNOTATIONS:
                    has_mapping = True
                    if endpoint_sub is None and first_quoted is not None:
                        endpoint_sub = first_quoted

            endpoint = _join_path(base_path, endpoint_sub) if has_mapping else None

            # (d) Calls within this method's own body span.
            calls: list[dict[str, str]] = []
            method_open_local = method_match.end() - 1
            if 0 <= method_open_local < len(body_text) and body_text[method_open_local] == "{":
                method_close_local = _find_matching_brace(body_text, method_open_local)
                if method_close_local is not None:
                    method_body = body_text[method_open_local + 1 : method_close_local]
                    for call_match in _CALL_RE.finditer(method_body):
                        var_name, called_method = call_match.group(1), call_match.group(2)
                        if var_name in known_deps:
                            calls.append(
                                {
                                    "target_field": var_name,
                                    "target_type": known_deps[var_name],
                                    "method": called_method,
                                }
                            )

            methods.append(
                {
                    "name": method_name,
                    "annotations": method_annotation_names,
                    "endpoint": endpoint,
                    "calls": calls,
                }
            )

    endpoints: list[str] = []
    seen_endpoints: set[str] = set()
    for method in methods:
        endpoint = method["endpoint"]
        if endpoint is not None and endpoint not in seen_endpoints:
            endpoints.append(endpoint)
            seen_endpoints.add(endpoint)
    if not endpoints and base_path:
        endpoints.append(base_path)

    dependencies: list[str] = []
    seen_types: set[str] = set()
    for dep_type in known_deps.values():
        if dep_type not in seen_types:
            dependencies.append(dep_type)
            seen_types.add(dep_type)

    return {
        "kind": kind,
        "name": name,
        "type": stereotype,
        "file": rel_posix,
        "package": package,
        "annotations": annotation_names,
        "extends": extends_,
        "implements": implements_list,
        "dependencies": dependencies,
        "endpoints": endpoints,
        "fields": fields,
        "methods": methods,
    }


def _analyze_java_file(abs_path: Path, rel_posix: str) -> list[dict[str, Any]]:
    """Read and analyze a single .java file, returning one symbol dict per
    class/interface declaration found in it. Raises on I/O errors so the
    caller's per-file try/except can decide to skip the file."""
    raw = abs_path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    line_offsets = _line_start_offsets(text)

    package_match = _PACKAGE_RE.search(text)
    package = package_match.group(1) if package_match else ""

    symbols: list[dict[str, Any]] = []
    for class_match in _CLASS_RE.finditer(text):
        try:
            symbol = _build_symbol(class_match, text, lines, line_offsets, package, rel_posix)
        except Exception:  # noqa: BLE001 - one malformed declaration must not sink the whole file
            continue
        symbols.append(symbol)
    return symbols


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def analyze_java_repository(
    repo_root: str,
    ignored_dirs: set[str] | list[str] | None = None,
    max_files: int = 3000,
) -> dict[str, Any]:
    """Walk `repo_root`, analyze every `.java` file found, and return a
    JSON-serializable summary of the classes/interfaces (and their
    annotations, dependencies, endpoints, fields, and methods) discovered.

    Mirrors `kenzo.memory.indexer.index_repository`'s walking pattern:
    os.walk with `dirnames` pruned in place against `ignored_dirs` (default
    `kenzo.config.settings.DEFAULT_IGNORED_DIRS`), deterministic sorted file
    collection, capped at `max_files`.

    A single malformed or unusual .java file is skipped (still counted
    toward `file_count`) rather than aborting the whole scan.
    """
    root = Path(repo_root).resolve()
    skip_dirs = set(ignored_dirs) if ignored_dirs is not None else set(DEFAULT_IGNORED_DIRS)

    collected_paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        current = Path(dirpath)
        for filename in filenames:
            if not filename.endswith(".java"):
                continue
            file_path = current / filename
            collected_paths.append(file_path.relative_to(root).as_posix())

    collected_paths.sort()
    truncated = len(collected_paths) > max_files
    included_paths = collected_paths[:max_files]

    symbols: list[dict[str, Any]] = []
    for rel_posix in included_paths:
        abs_path = root / rel_posix
        try:
            symbols.extend(_analyze_java_file(abs_path, rel_posix))
        except Exception:  # noqa: BLE001 - a single bad file must not abort the whole scan
            continue

    return {
        "language": "java",
        "file_count": len(included_paths),
        "truncated": truncated,
        "symbols": symbols,
    }
