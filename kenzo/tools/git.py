"""Read-only git introspection tools, backed by GitPython.

Only read-only operations are implemented here on purpose (status, diff,
branch, history) -- no commit/add/push/checkout. Mutating git operations are
out of scope for this module; if they're ever needed they should be gated
behind `permissions.allow_git_write` in a dedicated module.
"""

from __future__ import annotations

import functools
import os

# GitPython raises ImportError at import time if it can't locate a `git`
# executable on PATH. That's a reasonable thing for a git *library* to do,
# but it's too aggressive for a tool module that should degrade gracefully:
# we'd rather let `git.Repo(...)` fail per-call (caught below) than take
# down the entire tool registry (filesystem/terminal tools included) just
# because git isn't on PATH in this environment. Silence that startup
# check unless the caller already opted into a stricter mode.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import git  # noqa: E402 - must follow the GIT_PYTHON_REFRESH env var above

from kenzo.config.settings import PermissionsConfig
from kenzo.core.schemas import ToolResult, ToolSpec


def _open_repo(repo_root: str) -> git.Repo:
    return git.Repo(repo_root, search_parent_directories=False)


def _not_a_repo_result(tool_name: str) -> ToolResult:
    return ToolResult(tool_name=tool_name, success=False, error="not a git repository")


def _git_unavailable_result(tool_name: str, exc: Exception) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        success=False,
        error=f"git is not usable in this environment: {exc}",
    )


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def status(repo_root: str) -> ToolResult:
    """Current branch, changed (tracked) files, and untracked files."""
    try:
        repo = _open_repo(repo_root)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return _not_a_repo_result("status")
    except Exception as exc:  # noqa: BLE001 - e.g. git executable missing entirely
        return _git_unavailable_result("status", exc)

    try:
        if repo.head.is_valid() and not repo.head.is_detached:
            branch_name = repo.active_branch.name
        elif repo.head.is_valid():
            branch_name = f"DETACHED@{repo.head.commit.hexsha[:8]}"
        else:
            branch_name = "(unborn)"
    except Exception as exc:  # noqa: BLE001 - unusual repo states (no commits, etc.)
        return ToolResult(tool_name="status", success=False, error=f"could not determine branch: {exc}")

    changed: set[str] = set()
    try:
        # staged changes: index vs HEAD
        if repo.head.is_valid():
            for diff_item in repo.index.diff("HEAD"):
                if diff_item.a_path:
                    changed.add(diff_item.a_path)
                if diff_item.b_path:
                    changed.add(diff_item.b_path)
        # unstaged changes: working tree vs index
        for diff_item in repo.index.diff(None):
            if diff_item.a_path:
                changed.add(diff_item.a_path)
            if diff_item.b_path:
                changed.add(diff_item.b_path)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(tool_name="status", success=False, error=f"could not compute changed files: {exc}")

    try:
        untracked = list(repo.untracked_files)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(tool_name="status", success=False, error=f"could not list untracked files: {exc}")

    data = {
        "branch": branch_name,
        "changed_files": sorted(changed),
        "untracked_files": sorted(untracked),
    }
    summary = (
        f"branch={branch_name} changed={len(data['changed_files'])} "
        f"untracked={len(data['untracked_files'])}"
    )
    return ToolResult(tool_name="status", success=True, output=summary, data=data)


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------


def diff(repo_root: str, staged: bool = False) -> ToolResult:
    """Unified diff text: staged (index vs HEAD) or unstaged (working tree vs index)."""
    try:
        repo = _open_repo(repo_root)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return _not_a_repo_result("diff")
    except Exception as exc:  # noqa: BLE001
        return _git_unavailable_result("diff", exc)

    try:
        diff_text = repo.git.diff("--cached") if staged else repo.git.diff()
    except git.GitCommandError as exc:
        return ToolResult(tool_name="diff", success=False, error=f"git diff failed: {exc}")

    return ToolResult(tool_name="diff", success=True, output=diff_text, data={"diff": diff_text})


# --------------------------------------------------------------------------
# branch
# --------------------------------------------------------------------------


def branch(repo_root: str) -> ToolResult:
    """Current branch name (or a detached-HEAD marker) and all local branch names."""
    try:
        repo = _open_repo(repo_root)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return _not_a_repo_result("branch")
    except Exception as exc:  # noqa: BLE001
        return _git_unavailable_result("branch", exc)

    try:
        if repo.head.is_valid() and not repo.head.is_detached:
            current = repo.active_branch.name
        elif repo.head.is_valid():
            current = f"DETACHED@{repo.head.commit.hexsha[:8]}"
        else:
            current = "(unborn)"
        all_branches = [head.name for head in repo.heads]
    except Exception as exc:  # noqa: BLE001
        return ToolResult(tool_name="branch", success=False, error=f"could not read branches: {exc}")

    return ToolResult(
        tool_name="branch",
        success=True,
        output=f"current={current} ({len(all_branches)} local branches)",
        data={"current": current, "all": all_branches},
    )


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------


def history(repo_root: str, max_count: int = 20) -> ToolResult:
    """The most recent `max_count` commits reachable from HEAD."""
    try:
        repo = _open_repo(repo_root)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return _not_a_repo_result("history")
    except Exception as exc:  # noqa: BLE001
        return _git_unavailable_result("history", exc)

    try:
        commits = [
            {
                "hexsha": commit.hexsha,
                "message": commit.message.strip()
                if isinstance(commit.message, str)
                else commit.message.decode("utf-8", errors="replace").strip(),
                "author": str(commit.author),
                "date": commit.committed_datetime.isoformat(),
            }
            for commit in repo.iter_commits(max_count=max_count)
        ]
    except ValueError:
        # Empty repo: no commits yet on the current branch.
        commits = []
    except git.GitCommandError as exc:
        return ToolResult(tool_name="history", success=False, error=f"git log failed: {exc}")

    return ToolResult(
        tool_name="history",
        success=True,
        output=f"{len(commits)} commit(s)",
        data={"commits": commits},
    )


# --------------------------------------------------------------------------
# ToolSpec registration
# --------------------------------------------------------------------------


def get_tool_specs(repo_root: str, permissions: PermissionsConfig) -> list[ToolSpec]:
    """Build the read-only git ToolSpecs, bound to this repo_root.

    `permissions` is accepted for interface symmetry with the other tool
    modules (and so a future write-capable git tool can gate on
    `allow_git_write`), but every operation here is read-only regardless.
    """
    del permissions  # unused: all operations in this module are read-only
    return [
        ToolSpec(
            name="status",
            description="Show the current git branch, changed tracked files, and untracked files.",
            parameters={"type": "object", "properties": {}, "required": []},
            func=functools.partial(status, repo_root=repo_root),
        ),
        ToolSpec(
            name="diff",
            description="Show the current git diff (unstaged by default, or staged/--cached).",
            parameters={
                "type": "object",
                "properties": {
                    "staged": {
                        "type": "boolean",
                        "description": "If true, show the staged diff (index vs HEAD) instead of the working-tree diff.",
                        "default": False,
                    }
                },
                "required": [],
            },
            func=functools.partial(diff, repo_root=repo_root),
        ),
        ToolSpec(
            name="branch",
            description="Show the current git branch and the list of all local branches.",
            parameters={"type": "object", "properties": {}, "required": []},
            func=functools.partial(branch, repo_root=repo_root),
        ),
        ToolSpec(
            name="history",
            description="Show recent git commit history (hash, message, author, date).",
            parameters={
                "type": "object",
                "properties": {
                    "max_count": {
                        "type": "integer",
                        "description": "Maximum number of commits to return.",
                        "default": 20,
                    }
                },
                "required": [],
            },
            func=functools.partial(history, repo_root=repo_root),
        ),
    ]
