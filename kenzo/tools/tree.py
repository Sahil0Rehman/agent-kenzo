"""Repository tree tool: a recursive, depth-and-size-bounded directory tree
listing that replaces laboriously walking one level at a time with
`list_directory` (see `kenzo.tools.filesystem.list_directory`).

Sandboxed to repo_root via `kenzo.tools.filesystem._resolve_within_repo`, the
same security boundary every other filesystem tool uses -- no path-traversal
logic is reimplemented here.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from kenzo.config.settings import DEFAULT_IGNORED_DIRS, IndexingConfig, PermissionsConfig
from kenzo.core.schemas import ToolResult, ToolSpec
from kenzo.tools.filesystem import PathTraversalError, _resolve_within_repo

# Box-drawing connectors used to render the ASCII tree, matching the visual
# style of a conventional `tree` command.
_BRANCH = "├── "
_LAST_BRANCH = "└── "
_VERTICAL = "│   "
_BLANK = "    "


def _sorted_children(dir_path: Path, skip_dirs: set[str]) -> list[Path]:
    """Immediate children of `dir_path`, directories first then files, each
    group sorted case-insensitively by name. Directories named in `skip_dirs`
    are excluded entirely (never listed, never descended into).

    Unreadable directories (permission errors, races) yield no children
    rather than raising, so one bad subdirectory doesn't abort the whole walk.
    """
    try:
        children = list(dir_path.iterdir())
    except OSError:
        return []
    dirs = sorted(
        (c for c in children if c.is_dir() and c.name not in skip_dirs),
        key=lambda p: p.name.lower(),
    )
    files = sorted(
        (c for c in children if not c.is_dir()),
        key=lambda p: p.name.lower(),
    )
    return dirs + files


def repository_tree(
    path: str,
    repo_root: str,
    ignored_dirs: set[str] | list[str] | None = None,
    max_depth: int = 4,
    max_entries: int = 400,
) -> ToolResult:
    """Recursively render a depth- and size-bounded ASCII tree of a directory.

    The starting directory itself is depth 0; its immediate children are
    depth 1; and so on. Descent stops once depth would exceed `max_depth` --
    entries at depth strictly greater than `max_depth` are excluded entirely,
    not merely not-descended-into.

    Skips any directory whose *name* (not full path) is in `ignored_dirs`
    (defaulting to `kenzo.config.settings.DEFAULT_IGNORED_DIRS`). Entries
    within each directory are sorted alphabetically, directories before
    files, so walk order -- and therefore which entries survive truncation --
    is deterministic.

    Stops adding new entries once the total files+directories collected
    reaches `max_entries`; `data["truncated"]` (and a trailing line in
    `output`) reflect whether that cap was hit.
    """
    try:
        resolved = _resolve_within_repo(repo_root, path)
    except PathTraversalError as exc:
        return ToolResult(tool_name="repository_tree", success=False, error=str(exc))

    if not resolved.exists():
        return ToolResult(
            tool_name="repository_tree", success=False, error=f"directory not found: {path}"
        )
    if not resolved.is_dir():
        return ToolResult(
            tool_name="repository_tree", success=False, error=f"'{path}' is not a directory"
        )

    skip_dirs = set(ignored_dirs) if ignored_dirs is not None else set(DEFAULT_IGNORED_DIRS)
    root = Path(repo_root).resolve()

    entries: list[dict[str, Any]] = []
    lines: list[str] = [f"{resolved.name or resolved.as_posix()}/"]
    truncated = False

    def _walk(dir_path: Path, depth: int, prefix: str) -> None:
        nonlocal truncated
        if truncated or depth > max_depth:
            return

        children = _sorted_children(dir_path, skip_dirs)
        for index, child in enumerate(children):
            if len(entries) >= max_entries:
                truncated = True
                return

            is_last = index == len(children) - 1
            connector = _LAST_BRANCH if is_last else _BRANCH
            is_dir = child.is_dir()
            display_name = child.name + ("/" if is_dir else "")
            lines.append(f"{prefix}{connector}{display_name}")
            entries.append(
                {"path": child.relative_to(root).as_posix(), "is_dir": is_dir, "depth": depth}
            )

            if is_dir:
                extension = _BLANK if is_last else _VERTICAL
                _walk(child, depth + 1, prefix + extension)
                if truncated:
                    return

    _walk(resolved, 1, "")

    if truncated:
        lines.append(f"(truncated at {max_entries} entries)")

    return ToolResult(
        tool_name="repository_tree",
        success=True,
        output="\n".join(lines),
        data={"entries": entries, "truncated": truncated},
    )


# --------------------------------------------------------------------------
# ToolSpec registration
# --------------------------------------------------------------------------


def get_tool_specs(
    repo_root: str, permissions: PermissionsConfig, indexing: IndexingConfig | None = None
) -> list[ToolSpec]:
    """Build the `repository_tree` ToolSpec, bound to this repo_root/ignored_dirs.

    `permissions` is accepted for interface symmetry with the other tool
    modules' `get_tool_specs` factories (see `kenzo.tools.git`, which does the
    same for its own read-only operations), but this tool never mutates the
    filesystem and does not need to gate on it.
    """
    del permissions  # unused: this tool is read-only
    ignored_dirs = indexing.ignored_dirs if indexing is not None else None
    return [
        ToolSpec(
            name="repository_tree",
            description=(
                "Render a recursive ASCII tree of a directory's contents, bounded by "
                "depth and total entry count. Prefer this over repeated list_directory "
                "calls when orienting in a repository or exploring a subtree."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the directory, relative to the repository root. "
                            "Defaults to the repository root itself."
                        ),
                        "default": ".",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to descend (the starting directory is depth 0).",
                        "default": 4,
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum total files+directories to collect before truncating.",
                        "default": 400,
                    },
                },
                "required": [],
            },
            func=functools.partial(repository_tree, repo_root=repo_root, ignored_dirs=ignored_dirs),
        ),
    ]
