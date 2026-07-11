"""Cross-repo tools: `read_other_repo_file`/`search_other_repo_files`/
`list_other_repo_directory` (always registered when a workspace has any
`--see`/`--also-edit` repo) and `write_other_repo_file`/`edit_other_repo_file`
(only usable against a repo named in `--also-edit`).

Every function here is a thin `repo` alias -> filesystem-root lookup in
front of the EXACT SAME, already-hardened functions in `kenzo.tools
.filesystem` (`read_file`/`write_file`/`edit_file`/`search_files`/
`list_directory`) -- this file adds no new path-traversal logic of its own;
it reuses `_resolve_within_repo`'s guarantee by simply calling those
functions with a DIFFERENT `repo_root` (the target workspace repo's root
instead of the primary repo's), which is the only genuinely new behavior
here. A write/edit attempt against an alias that isn't in the workspace's
writable set is refused before ever reaching `filesystem.write_file`/
`edit_file`, so it fails closed the same way an unknown tool name does.
"""

from __future__ import annotations

from typing import Any

from kenzo.config.settings import IndexingConfig, PermissionsConfig
from kenzo.core.schemas import ToolResult, ToolSpec
from kenzo.core.workspace import Workspace
from kenzo.tools import filesystem

# Shared with kenzo.core.controller (ACTION_TOOLS) and kenzo.workers
# .worker_agent (WORKER_TOOL_NAMES) so both offer the exact same tool names
# without duplicating the list -- read tools are safe to offer broadly
# (mirrors read_file/search_files/list_directory's own reach); the write
# tools are additionally gated per-call against the workspace's writable
# set (see write_other_repo_file/edit_other_repo_file below), so listing
# their names somewhere doesn't by itself grant write access to anything.
READ_TOOL_NAMES = ["read_other_repo_file", "list_other_repo_directory", "search_other_repo_files"]
WRITE_TOOL_NAMES = ["write_other_repo_file", "edit_other_repo_file"]


def _unknown_repo_error(tool_name: str, repo: str, workspace: Workspace) -> ToolResult:
    known = ", ".join(workspace.readable_aliases()) or "(none configured)"
    return ToolResult(
        tool_name=tool_name,
        success=False,
        error=f"unknown workspace repo '{repo}'. Known repos in this workspace: {known}",
    )


def _not_writable_error(tool_name: str, repo: str, workspace: Workspace) -> ToolResult:
    writable = ", ".join(workspace.writable_aliases()) or "(none -- every other repo in this workspace is read-only)"
    return ToolResult(
        tool_name=tool_name,
        success=False,
        error=(
            f"'{repo}' is not writable in this workspace -- it was given via --see (read-only), "
            f"not --also-edit. Writable repos: {writable}. If you genuinely need to edit '{repo}', "
            "the user must re-run with --also-edit for it."
        ),
    )


def read_other_repo_file(repo: str, path: str, workspace: Workspace) -> ToolResult:
    entry = workspace.find(repo)
    if entry is None:
        return _unknown_repo_error("read_other_repo_file", repo, workspace)
    return filesystem.read_file(path, entry.path)


def list_other_repo_directory(repo: str, path: str, workspace: Workspace) -> ToolResult:
    entry = workspace.find(repo)
    if entry is None:
        return _unknown_repo_error("list_other_repo_directory", repo, workspace)
    return filesystem.list_directory(path, entry.path)


def search_other_repo_files(
    repo: str, pattern: str, workspace: Workspace, glob: str | None = None, ignored_dirs: Any = None
) -> ToolResult:
    entry = workspace.find(repo)
    if entry is None:
        return _unknown_repo_error("search_other_repo_files", repo, workspace)
    return filesystem.search_files(pattern, entry.path, glob=glob, ignored_dirs=ignored_dirs)


def write_other_repo_file(
    repo: str, path: str, content: str, workspace: Workspace, permissions: PermissionsConfig
) -> ToolResult:
    entry = workspace.find(repo)
    if entry is None:
        return _unknown_repo_error("write_other_repo_file", repo, workspace)
    if not entry.writable:
        return _not_writable_error("write_other_repo_file", repo, workspace)
    return filesystem.write_file(path, content, entry.path, permissions)


def edit_other_repo_file(
    repo: str, path: str, old_text: str, new_text: str, workspace: Workspace, permissions: PermissionsConfig
) -> ToolResult:
    entry = workspace.find(repo)
    if entry is None:
        return _unknown_repo_error("edit_other_repo_file", repo, workspace)
    if not entry.writable:
        return _not_writable_error("edit_other_repo_file", repo, workspace)
    return filesystem.edit_file(path, old_text, new_text, entry.path, permissions)


# --------------------------------------------------------------------------
# ToolSpec registration
# --------------------------------------------------------------------------


def get_tool_specs(
    workspace: Workspace, permissions: PermissionsConfig, indexing: IndexingConfig | None = None
) -> list[ToolSpec]:
    """Cross-repo ToolSpecs for `workspace`. Returns an empty list if the
    workspace has no other repos at all -- callers can unconditionally call
    this and register the (possibly empty) result, matching how MCP tools
    are registered unconditionally and simply contribute nothing when no
    server is configured.
    """
    if not workspace.has_any_other_repo():
        return []

    ignored_dirs = (indexing or IndexingConfig()).ignored_dirs
    known_repos_desc = ", ".join(f"'{r.alias}'" + (" (writable)" if r.writable else " (read-only)") for r in workspace.others)

    repo_param = {
        "type": "string",
        "description": f"Which workspace repo to use. Known: {known_repos_desc}.",
    }

    specs = [
        ToolSpec(
            name="read_other_repo_file",
            description=(
                "Read the full text contents of a file in ANOTHER repo in this workspace "
                "(not the primary repo you're editing) -- e.g. checking how a legacy branch "
                "implemented something before fixing the current one."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "repo": repo_param,
                    "path": {"type": "string", "description": "Path to the file, relative to that repo's root."},
                },
                "required": ["repo", "path"],
            },
            func=lambda repo, path: read_other_repo_file(repo, path, workspace),
        ),
        ToolSpec(
            name="list_other_repo_directory",
            description="List the immediate files and subdirectories of a directory in ANOTHER repo in this workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "repo": repo_param,
                    "path": {"type": "string", "description": "Path to the directory, relative to that repo's root."},
                },
                "required": ["repo", "path"],
            },
            func=lambda repo, path: list_other_repo_directory(repo, path, workspace),
        ),
        ToolSpec(
            name="search_other_repo_files",
            description="Search text files in ANOTHER repo in this workspace for a regex pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "repo": repo_param,
                    "pattern": {"type": "string", "description": "Regular expression to search for, applied per line."},
                    "glob": {"type": "string", "description": "Optional filename glob to restrict which files are searched, e.g. '*.java'."},
                },
                "required": ["repo", "pattern"],
            },
            func=lambda repo, pattern, glob=None: search_other_repo_files(
                repo, pattern, workspace, glob=glob, ignored_dirs=ignored_dirs
            ),
        ),
    ]

    if workspace.has_writable_other_repo():
        specs.append(
            ToolSpec(
                name="write_other_repo_file",
                description=(
                    "Create or overwrite a file in ANOTHER repo in this workspace that the user "
                    "explicitly marked editable (--also-edit) -- e.g. porting the same fix into a "
                    "second branch. Refuses if the named repo isn't in the writable set."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "repo": repo_param,
                        "path": {"type": "string", "description": "Path to the file, relative to that repo's root."},
                        "content": {"type": "string", "description": "Full new contents of the file."},
                    },
                    "required": ["repo", "path", "content"],
                },
                func=lambda repo, path, content: write_other_repo_file(repo, path, content, workspace, permissions),
            )
        )
        specs.append(
            ToolSpec(
                name="edit_other_repo_file",
                description=(
                    "Apply a targeted edit to an existing file in ANOTHER repo in this workspace "
                    "that the user explicitly marked editable (--also-edit), by replacing an exact, "
                    "unique block of text. Refuses if the named repo isn't in the writable set."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "repo": repo_param,
                        "path": {"type": "string", "description": "Path to the file, relative to that repo's root."},
                        "old_text": {"type": "string", "description": "Exact text to find; must appear exactly once in the file."},
                        "new_text": {"type": "string", "description": "Text to replace old_text with."},
                    },
                    "required": ["repo", "path", "old_text", "new_text"],
                },
                func=lambda repo, path, old_text, new_text: edit_other_repo_file(
                    repo, path, old_text, new_text, workspace, permissions
                ),
            )
        )

    return specs
