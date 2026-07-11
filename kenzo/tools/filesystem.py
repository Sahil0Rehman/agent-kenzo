"""Filesystem tools: read/write/edit/search/list, all sandboxed to repo_root.

Every function here is directly importable and callable on its own (for unit
tests or programmatic use) and is also exposed to the LLM tool-calling loop
via `get_tool_specs`, which binds `repo_root` / `permissions` into a
`functools.partial` closure so the model only ever supplies the "real"
arguments (path, content, pattern, ...).

Security: every path argument is resolved relative to `repo_root` and then
verified to still live inside `repo_root` before any filesystem access
happens. This blocks both `../..` traversal and absolute paths that point
outside the sandboxed repository root.
"""

from __future__ import annotations

import difflib
import fnmatch
import functools
import re
from pathlib import Path
from typing import Any

from kenzo.config.settings import DEFAULT_IGNORED_DIRS, IndexingConfig, PermissionsConfig
from kenzo.core.schemas import ToolResult, ToolSpec

# Directories we never walk into for search_files (VCS internals, caches,
# dependency trees, build output -- never useful to a code-editing agent and
# often huge). Overridable per-call via `ignored_dirs` (see IndexingConfig);
# this is just the fallback when the caller doesn't supply its own.
_SKIP_DIRS = set(DEFAULT_IGNORED_DIRS)

# Cap the number of matches returned by search_files so a too-broad pattern
# over a large repo can't produce an unbounded response.
_MAX_SEARCH_MATCHES = 500

# Cap the number of bytes sniffed to decide whether a file is binary.
_BINARY_SNIFF_BYTES = 8192


class PathTraversalError(ValueError):
    """Raised internally when a requested path would escape repo_root."""


def _resolve_within_repo(repo_root: str, path: str) -> Path:
    """Resolve `path` (relative or absolute) against `repo_root` and verify
    the result is still inside `repo_root`.

    Raises PathTraversalError if the resolved path escapes repo_root.
    """
    root = Path(repo_root).resolve()
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve()
    if not (resolved == root or resolved.is_relative_to(root)):
        raise PathTraversalError(
            f"path '{path}' resolves to '{resolved}', which is outside repo_root '{root}'"
        )
    return resolved


def _is_binary(data: bytes) -> bool:
    """Cheap binary sniff: a NUL byte in the first chunk is a strong signal
    the file isn't text."""
    return b"\x00" in data[:_BINARY_SNIFF_BYTES]


def _decode_text_or_result(raw: bytes, path: str, tool_name: str) -> tuple[str | None, ToolResult | None]:
    """Strictly decode `raw` as UTF-8, returning (text, None) on success or
    (None, ToolResult) with a clear refusal error otherwise.

    We refuse rather than lossily decode (errors="replace") because silently
    replacing invalid bytes with U+FFFD and then writing the result back
    permanently corrupts non-UTF-8 source files (e.g. legacy cp1252/Latin-1
    files with curly quotes or non-ASCII author names) -- even when the edit
    itself targets a completely unrelated part of the file. Refusing leaves
    the file byte-for-byte untouched instead.
    """
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, ToolResult(
            tool_name=tool_name,
            success=False,
            error=(
                f"'{path}' is not valid UTF-8 text ({exc}). Kenzo only reads and edits "
                "UTF-8 files and refuses to touch this one, to avoid silently corrupting "
                "non-UTF-8 bytes on a lossy decode/re-encode round trip. Leave it as-is or "
                "convert it to UTF-8 outside of Kenzo first."
            ),
        )


def _unified_diff(old_text: str, new_text: str, path: str) -> str:
    """Render a unified diff between `old_text` and `new_text` for display/audit."""
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )


# --------------------------------------------------------------------------
# read_file
# --------------------------------------------------------------------------


def read_file(path: str, repo_root: str) -> ToolResult:
    """Read the full contents of a text file located inside repo_root."""
    try:
        resolved = _resolve_within_repo(repo_root, path)
    except PathTraversalError as exc:
        return ToolResult(tool_name="read_file", success=False, error=str(exc))

    if not resolved.exists():
        return ToolResult(tool_name="read_file", success=False, error=f"file not found: {path}")
    if resolved.is_dir():
        return ToolResult(
            tool_name="read_file", success=False, error=f"'{path}' is a directory, not a file"
        )

    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        return ToolResult(tool_name="read_file", success=False, error=f"could not read '{path}': {exc}")

    if _is_binary(raw):
        return ToolResult(
            tool_name="read_file",
            success=False,
            error=f"'{path}' looks like a binary file and cannot be read as text",
        )

    content, decode_error = _decode_text_or_result(raw, path, "read_file")
    if decode_error is not None:
        return decode_error
    return ToolResult(tool_name="read_file", success=True, output=content, data={"content": content})


# --------------------------------------------------------------------------
# write_file
# --------------------------------------------------------------------------


def write_file(
    path: str, content: str, repo_root: str, permissions: PermissionsConfig
) -> ToolResult:
    """Overwrite (or create) a file inside repo_root with `content`.

    Gated by permissions.allow_write; parent directories are created as
    needed so new files in new subdirectories can be written in one call.
    """
    if not permissions.allow_write:
        return ToolResult(
            tool_name="write_file",
            success=False,
            error=(
                "write_file is disabled by config (permissions.allow_write=false). "
                "Enable it with `kenzo config set permissions.allow_write true`."
            ),
        )

    try:
        resolved = _resolve_within_repo(repo_root, path)
    except PathTraversalError as exc:
        return ToolResult(tool_name="write_file", success=False, error=str(exc))

    if resolved.exists() and resolved.is_dir():
        return ToolResult(
            tool_name="write_file", success=False, error=f"'{path}' is a directory, not a file"
        )

    # Capture the pre-write content (best-effort) so the caller gets a real
    # diff/old_content, not just a bare "file written" confirmation. This is
    # a full create-or-overwrite tool, so we don't refuse the write if the
    # existing file can't be decoded as UTF-8 (unlike edit_file, which does a
    # partial round trip and must refuse) -- we just can't show a diff for it.
    old_content: str | None = None
    created = not resolved.exists()
    if not created:
        try:
            old_raw = resolved.read_bytes()
            if not _is_binary(old_raw):
                old_content, _decode_error = _decode_text_or_result(old_raw, path, "write_file")
        except OSError:
            old_content = None

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # newline="" disables platform newline translation so the bytes we
        # write match `content` exactly -- important since read_file reads
        # raw bytes without translation too; without this, Windows would
        # silently turn '\n' into '\r\n' on write and break the round trip.
        resolved.write_text(content, encoding="utf-8", newline="")
    except OSError as exc:
        return ToolResult(tool_name="write_file", success=False, error=f"could not write '{path}': {exc}")

    diff_text = _unified_diff(old_content or "", content, path)

    return ToolResult(
        tool_name="write_file",
        success=True,
        output=f"wrote {len(content)} bytes to {path}",
        data={
            "path": path,
            "bytes_written": len(content),
            "created": created,
            "old_content": old_content,
            "new_content": content,
            "diff": diff_text,
        },
    )


# --------------------------------------------------------------------------
# edit_file
# --------------------------------------------------------------------------


def edit_file(
    path: str, old_text: str, new_text: str, repo_root: str, permissions: PermissionsConfig
) -> ToolResult:
    """Exact-match search/replace within a file.

    Gated by permissions.allow_write, exactly like write_file -- this is a
    mutating tool and must not be reachable when the config says the agent
    should be read-only.

    Errors (rather than guessing) if `old_text` doesn't appear in the file,
    or appears more than once, since either case makes the intended edit
    ambiguous.
    """
    if not permissions.allow_write:
        return ToolResult(
            tool_name="edit_file",
            success=False,
            error=(
                "edit_file is disabled by config (permissions.allow_write=false). "
                "Enable it with `kenzo config set permissions.allow_write true`."
            ),
        )

    try:
        resolved = _resolve_within_repo(repo_root, path)
    except PathTraversalError as exc:
        return ToolResult(tool_name="edit_file", success=False, error=str(exc))

    if not resolved.exists():
        return ToolResult(tool_name="edit_file", success=False, error=f"file not found: {path}")
    if resolved.is_dir():
        return ToolResult(
            tool_name="edit_file", success=False, error=f"'{path}' is a directory, not a file"
        )

    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        return ToolResult(tool_name="edit_file", success=False, error=f"could not read '{path}': {exc}")

    if _is_binary(raw):
        return ToolResult(
            tool_name="edit_file",
            success=False,
            error=f"'{path}' looks like a binary file and cannot be edited as text",
        )

    text, decode_error = _decode_text_or_result(raw, path, "edit_file")
    if decode_error is not None:
        return decode_error

    occurrences = text.count(old_text)

    if occurrences == 0:
        return ToolResult(
            tool_name="edit_file",
            success=False,
            error=f"old_text not found in '{path}'; no edit was made",
        )
    if occurrences > 1:
        return ToolResult(
            tool_name="edit_file",
            success=False,
            error=(
                f"old_text found {occurrences} times in '{path}'; edit_file requires an "
                "exact, unique match. Include more surrounding context to disambiguate."
            ),
        )

    new_content = text.replace(old_text, new_text, 1)
    try:
        resolved.write_text(new_content, encoding="utf-8", newline="")
    except OSError as exc:
        return ToolResult(tool_name="edit_file", success=False, error=f"could not write '{path}': {exc}")

    diff_text = _unified_diff(text, new_content, path)

    return ToolResult(
        tool_name="edit_file",
        success=True,
        output=f"applied 1 edit to {path}",
        data={
            "path": path,
            "old_content": text,
            "new_content": new_content,
            "diff": diff_text,
        },
    )


# --------------------------------------------------------------------------
# search_files
# --------------------------------------------------------------------------


def search_files(
    pattern: str, repo_root: str, glob: str | None = None, ignored_dirs: set[str] | list[str] | None = None
) -> ToolResult:
    """Regex search across text files under repo_root.

    Skips well-known noise directories (see `ignored_dirs`, defaulting to
    `kenzo.config.settings.DEFAULT_IGNORED_DIRS`) and binary files. `glob`,
    if given, filters candidate files by filename or repo-relative path
    (e.g. "*.py").
    """
    root = Path(repo_root).resolve()
    if not root.exists():
        return ToolResult(tool_name="search_files", success=False, error=f"repo_root does not exist: {repo_root}")

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult(tool_name="search_files", success=False, error=f"invalid regex pattern: {exc}")

    matches: list[dict[str, Any]] = []
    truncated = False
    skip_dirs = set(ignored_dirs) if ignored_dirs is not None else _SKIP_DIRS

    for dirpath, dirnames, filenames in _walk_skip_dirs(root, skip_dirs):
        for filename in filenames:
            file_path = dirpath / filename
            rel_path = file_path.relative_to(root).as_posix()

            if glob and not (fnmatch.fnmatch(filename, glob) or fnmatch.fnmatch(rel_path, glob)):
                continue

            try:
                raw = file_path.read_bytes()
            except OSError:
                continue

            if _is_binary(raw):
                continue

            text = raw.decode("utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append({"path": rel_path, "line": line_no, "text": line})
                    if len(matches) >= _MAX_SEARCH_MATCHES:
                        truncated = True
                        break
            if truncated:
                break
        if truncated:
            break

    summary = f"{len(matches)} match(es)" + (" (truncated)" if truncated else "")
    return ToolResult(
        tool_name="search_files",
        success=True,
        output=summary,
        data={"matches": matches, "truncated": truncated},
    )


def _walk_skip_dirs(root: Path, skip_dirs: set[str] | None = None):
    """os.walk-alike over pathlib, pruning `skip_dirs` (default _SKIP_DIRS) in place."""
    import os

    skip = skip_dirs if skip_dirs is not None else _SKIP_DIRS
    for dirpath_str, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        yield Path(dirpath_str), dirnames, filenames


# --------------------------------------------------------------------------
# list_directory
# --------------------------------------------------------------------------


def list_directory(path: str, repo_root: str) -> ToolResult:
    """Non-recursive listing of a directory's immediate entries."""
    try:
        resolved = _resolve_within_repo(repo_root, path)
    except PathTraversalError as exc:
        return ToolResult(tool_name="list_directory", success=False, error=str(exc))

    if not resolved.exists():
        return ToolResult(tool_name="list_directory", success=False, error=f"directory not found: {path}")
    if not resolved.is_dir():
        return ToolResult(tool_name="list_directory", success=False, error=f"'{path}' is not a directory")

    try:
        entries = sorted(resolved.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        return ToolResult(
            tool_name="list_directory", success=False, error=f"could not list '{path}': {exc}"
        )

    entry_dicts = [{"name": entry.name, "is_dir": entry.is_dir()} for entry in entries]
    return ToolResult(
        tool_name="list_directory",
        success=True,
        output=f"{len(entry_dicts)} entries",
        data={"entries": entry_dicts},
    )


# --------------------------------------------------------------------------
# ToolSpec registration
# --------------------------------------------------------------------------


def get_tool_specs(
    repo_root: str, permissions: PermissionsConfig, indexing: IndexingConfig | None = None
) -> list[ToolSpec]:
    """Build the filesystem ToolSpecs, bound to this repo_root/permissions.

    `indexing` (defaults to a stock IndexingConfig if omitted) supplies the
    configurable ignored-directories list search_files uses, so exclusions
    stay in sync with the indexer and repository_tree instead of each tool
    keeping its own separate hardcoded list.
    """
    ignored_dirs = (indexing or IndexingConfig()).ignored_dirs
    return [
        ToolSpec(
            name="read_file",
            description="Read the full text contents of a file inside the repository.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the repository root.",
                    }
                },
                "required": ["path"],
            },
            func=functools.partial(read_file, repo_root=repo_root),
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Create or overwrite a file inside the repository with the given content. "
                "Creates parent directories as needed. May be disabled by config."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the repository root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full new contents of the file.",
                    },
                },
                "required": ["path", "content"],
            },
            func=functools.partial(write_file, repo_root=repo_root, permissions=permissions),
        ),
        ToolSpec(
            name="edit_file",
            description=(
                "Apply a targeted edit to an existing file by replacing an exact, unique "
                "block of text (old_text) with new_text. Fails if old_text is not found "
                "exactly once, to avoid ambiguous edits. May be disabled by config."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the repository root.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find; must appear exactly once in the file.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to replace old_text with.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
            func=functools.partial(edit_file, repo_root=repo_root, permissions=permissions),
        ),
        ToolSpec(
            name="search_files",
            description=(
                "Search text files in the repository for a regex pattern, returning matching "
                "lines with file path and line number. Skips VCS/build/dependency directories "
                "and binary files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for, applied per line.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional filename glob to restrict which files are searched, e.g. '*.py'.",
                    },
                },
                "required": ["pattern"],
            },
            func=functools.partial(search_files, repo_root=repo_root, ignored_dirs=ignored_dirs),
        ),
        ToolSpec(
            name="list_directory",
            description="List the immediate files and subdirectories of a directory in the repository.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory, relative to the repository root ('.' for the root).",
                    }
                },
                "required": ["path"],
            },
            func=functools.partial(list_directory, repo_root=repo_root),
        ),
    ]
