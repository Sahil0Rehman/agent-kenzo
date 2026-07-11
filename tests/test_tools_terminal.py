"""Tests for kenzo.tools.terminal: no-shell command execution, the
metacharacter guard, the command allowlist, and the allow_shell_exec gate.
"""

from __future__ import annotations

import pytest

from kenzo.config.settings import PermissionsConfig
from kenzo.tools.terminal import get_tool_specs, run_command

_FORBIDDEN_EXAMPLES = [
    "git status && echo hi",
    "git status || echo hi",
    "git status; echo hi",
    "git status | cat",
    "echo hi > out.txt",
    "echo hi < in.txt",
    "echo `whoami`",
    "echo $(whoami)",
]


@pytest.fixture
def shell_enabled():
    """Default allowlist, shell exec turned on -- the baseline permissive
    config most of these tests need."""
    return PermissionsConfig(allow_shell_exec=True)


@pytest.mark.parametrize("command", _FORBIDDEN_EXAMPLES)
def test_shell_metacharacters_are_rejected(repo_root, shell_enabled, command):
    result = run_command(command, repo_root=str(repo_root), permissions=shell_enabled)

    assert result.success is False
    assert "disallowed token" in result.error


def test_non_allowlisted_command_is_rejected(repo_root, shell_enabled):
    # "whoami" is not in the default command_allowlist.
    result = run_command("whoami", repo_root=str(repo_root), permissions=shell_enabled)

    assert result.success is False
    assert "not in the allowlist" in result.error


def test_allow_shell_exec_false_blocks_everything(repo_root):
    permissions = PermissionsConfig(allow_shell_exec=False)

    result = run_command("python --version", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "allow_shell_exec" in result.error


def test_allow_shell_exec_false_blocks_even_allowlisted_commands(repo_root):
    permissions = PermissionsConfig(allow_shell_exec=False, command_allowlist=["python"])

    result = run_command("python --version", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "allow_shell_exec" in result.error


def test_allowlisted_command_succeeds_when_enabled(repo_root, shell_enabled):
    # "python" is in the default command_allowlist.
    result = run_command("python --version", repo_root=str(repo_root), permissions=shell_enabled)

    assert result.success is True
    assert "Python" in result.output
    assert result.data["returncode"] == 0


def test_empty_command_is_rejected(repo_root, shell_enabled):
    result = run_command("   ", repo_root=str(repo_root), permissions=shell_enabled)

    assert result.success is False
    assert "empty command" in result.error


def test_nonexistent_allowlisted_program_reports_error_not_exception(repo_root):
    # "dir" is allowlisted by default but isn't a real standalone executable
    # on most systems (it's a shell builtin) -- exercises the
    # FileNotFoundError/OSError branch instead of a successful run.
    permissions = PermissionsConfig(allow_shell_exec=True, command_allowlist=["totally-not-a-real-binary"])

    result = run_command(
        "totally-not-a-real-binary --flag", repo_root=str(repo_root), permissions=permissions
    )

    assert result.success is False
    assert "could not execute" in result.error


def test_get_tool_specs_binds_repo_root_and_permissions(repo_root, shell_enabled):
    specs = get_tool_specs(str(repo_root), shell_enabled)

    assert [spec.name for spec in specs] == ["run_command"]

    result = specs[0].func(command="python --version")
    assert result.success is True


# ---------------------------------------------------------------------------
# git command classification (allow_git_write gate)
# ---------------------------------------------------------------------------

_GIT_MUTATING_EXAMPLES = [
    "git commit -m test",
    "git push",
    "git reset --hard",
    "git checkout -- .",
    "git merge other-branch",
    "git rebase main",
]


@pytest.mark.parametrize("command", _GIT_MUTATING_EXAMPLES)
def test_git_mutating_subcommands_blocked_without_allow_git_write(repo_root, command):
    permissions = PermissionsConfig(allow_shell_exec=True, allow_git_write=False)

    result = run_command(command, repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "allow_git_write" in result.error


@pytest.mark.parametrize(
    "command",
    ["git status", "git status --short", "git diff", "git diff --stat", "git log -1", "git branch"],
)
def test_git_readonly_subcommands_allowed_without_allow_git_write(repo_root, command):
    permissions = PermissionsConfig(allow_shell_exec=True, allow_git_write=False)

    result = run_command(command, repo_root=str(repo_root), permissions=permissions)

    assert result.success is True


def test_git_unlisted_subcommand_blocked_by_default(repo_root):
    """Subcommands not explicitly named safe (e.g. stash) must default-deny,
    not default-allow, since git has many mutating subcommands beyond the
    handful anyone would think to blocklist up front."""
    permissions = PermissionsConfig(allow_shell_exec=True, allow_git_write=False)

    result = run_command("git stash list", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "allow_git_write" in result.error


def test_git_commit_succeeds_when_allow_git_write_true(repo_root):
    (repo_root / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    permissions = PermissionsConfig(allow_shell_exec=True, allow_git_write=True)

    add_result = run_command("git add main.py", repo_root=str(repo_root), permissions=permissions)
    commit_result = run_command(
        'git commit -m "hardened commit"', repo_root=str(repo_root), permissions=permissions
    )

    assert add_result.success is True
    assert commit_result.success is True

    log = run_command("git log -1", repo_root=str(repo_root), permissions=permissions)
    assert "hardened commit" in log.output


def test_git_status_still_requires_base_allowlist(repo_root):
    """The allow_git_write gate is layered on top of, not instead of, the
    base command_allowlist -- removing 'git' from the allowlist still blocks
    even a read-only subcommand."""
    permissions = PermissionsConfig(allow_shell_exec=True, command_allowlist=["python"])

    result = run_command("git status", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "not in the allowlist" in result.error
