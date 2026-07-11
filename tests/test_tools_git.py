"""Tests for kenzo.tools.git: read-only status/diff/branch/history against a
real tmp git repo, plus the "not a git repository" failure path.
"""

from __future__ import annotations

import pytest

from kenzo.tools import git as git_tools
from tests.conftest import run_git

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_on_freshly_committed_repo(repo_root):
    result = git_tools.status(repo_root=str(repo_root))

    assert result.success is True
    assert result.data["branch"] == "main"
    assert result.data["changed_files"] == []
    assert result.data["untracked_files"] == []


def test_status_reports_changed_and_untracked_files(repo_root):
    (repo_root / "main.py").write_text("def main():\n    return 100\n", encoding="utf-8")
    (repo_root / "scratch.txt").write_text("temp\n", encoding="utf-8")

    result = git_tools.status(repo_root=str(repo_root))

    assert result.success is True
    assert "main.py" in result.data["changed_files"]
    assert "scratch.txt" in result.data["untracked_files"]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def test_diff_unstaged_shows_working_tree_changes(repo_root):
    (repo_root / "main.py").write_text("def main():\n    return 999\n", encoding="utf-8")

    result = git_tools.diff(repo_root=str(repo_root))

    assert result.success is True
    assert "999" in result.output
    assert result.data["diff"] == result.output


def test_diff_staged_shows_index_changes(repo_root):
    (repo_root / "main.py").write_text("def main():\n    return 999\n", encoding="utf-8")
    run_git(["add", "-A"], cwd=repo_root)

    result = git_tools.diff(repo_root=str(repo_root), staged=True)

    assert result.success is True
    assert "999" in result.output


def test_diff_empty_when_no_changes(repo_root):
    result = git_tools.diff(repo_root=str(repo_root))

    assert result.success is True
    assert result.output == ""


# ---------------------------------------------------------------------------
# branch
# ---------------------------------------------------------------------------


def test_branch_reports_current_and_all(repo_root):
    result = git_tools.branch(repo_root=str(repo_root))

    assert result.success is True
    assert result.data["current"] == "main"
    assert result.data["all"] == ["main"]


def test_branch_lists_multiple_local_branches(repo_root):
    run_git(["branch", "feature/foo"], cwd=repo_root)

    result = git_tools.branch(repo_root=str(repo_root))

    assert result.success is True
    assert result.data["current"] == "main"
    assert set(result.data["all"]) == {"main", "feature/foo"}


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def test_history_returns_single_initial_commit(repo_root):
    result = git_tools.history(repo_root=str(repo_root))

    assert result.success is True
    assert len(result.data["commits"]) == 1
    commit = result.data["commits"][0]
    assert commit["message"] == "Initial commit"
    assert commit["hexsha"]
    assert commit["author"]
    assert commit["date"]


def test_history_returns_commits_most_recent_first(repo_root):
    (repo_root / "README.md").write_text("# Sample repo\n\nUpdated.\n", encoding="utf-8")
    run_git(["add", "-A"], cwd=repo_root)
    run_git(["commit", "-m", "Second commit"], cwd=repo_root)

    result = git_tools.history(repo_root=str(repo_root))

    assert result.success is True
    messages = [c["message"] for c in result.data["commits"]]
    assert messages == ["Second commit", "Initial commit"]


def test_history_respects_max_count(repo_root):
    (repo_root / "README.md").write_text("# Sample repo\n\nUpdated again.\n", encoding="utf-8")
    run_git(["add", "-A"], cwd=repo_root)
    run_git(["commit", "-m", "Second commit"], cwd=repo_root)

    result = git_tools.history(repo_root=str(repo_root), max_count=1)

    assert result.success is True
    assert len(result.data["commits"]) == 1
    assert result.data["commits"][0]["message"] == "Second commit"


# ---------------------------------------------------------------------------
# non-git directory: must fail gracefully, never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func_name", ["status", "diff", "branch", "history"])
def test_non_git_directory_returns_failure_not_exception(tmp_path, func_name):
    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()

    func = getattr(git_tools, func_name)
    result = func(repo_root=str(non_repo))

    assert result.success is False
    assert result.error is not None
    assert "not a git repository" in result.error
