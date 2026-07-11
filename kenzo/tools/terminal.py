"""Terminal tool: run a single, safe (non-shell) command inside the repo.

Deliberately restrictive: no shell is ever spawned (`shell=False`), chaining
or redirection metacharacters are rejected outright, and the resolved
executable's basename must appear in `permissions.command_allowlist`. This
is not a sandbox against a determined attacker with local code execution
elsewhere in the pipeline, but it does stop the common failure mode of an
LLM-issued command silently chaining into something destructive.
"""

from __future__ import annotations

import functools
import os
import shlex
import subprocess

from kenzo.config.settings import PermissionsConfig
from kenzo.core.schemas import ToolResult, ToolSpec

# Tokens that imply shell chaining/redirection/substitution. Since we exec
# argv directly with shell=False these would either be inert (harmless but
# confusing) or -- worse -- misleadingly appear to do nothing while the user
# assumes shell semantics. Reject up front instead.
_FORBIDDEN_TOKENS = ("&&", "||", ";", "|", ">", "<", "`", "$(")

# `git` is in the default command_allowlist so agents can run read-only
# introspection (status/diff/log/branch) via run_command. But the allowlist
# only ever checks the executable's basename, not the subcommand -- so
# without a separate check here, allowlisting "git" for that reason would
# also silently grant `git commit`/`git push`/`git reset`/etc, completely
# bypassing `permissions.allow_git_write`. To close that gap, any git
# subcommand other than these explicitly-safe, read-only ones requires
# `permissions.allow_git_write=true`. This is a default-deny list (unknown
# subcommands are treated as mutating) rather than a blocklist of just the
# obviously-dangerous ones, since git has many mutating subcommands
# (add, stash, tag, clean, apply, fetch, pull, ...) beyond the handful
# anyone would think to blocklist up front.
_GIT_SAFE_READONLY_SUBCOMMANDS = {"status", "diff", "log", "branch"}


def _git_subcommand(argv: list[str]) -> str | None:
    """Return the first non-flag token after argv[0] (the git subcommand),
    skipping leading global flags like `-C <path>` or `--no-pager`."""
    for token in argv[1:]:
        if not token.startswith("-"):
            return token.lower()
    return None


def run_command(
    command: str, repo_root: str, permissions: PermissionsConfig, timeout: int = 300
) -> ToolResult:
    """Execute a single allowlisted command with no shell, inside repo_root."""
    if not permissions.allow_shell_exec:
        return ToolResult(
            tool_name="run_command",
            success=False,
            error=(
                "run_command is disabled by config (permissions.allow_shell_exec=false). "
                "Enable it with `kenzo config set permissions.allow_shell_exec true`."
            ),
        )

    for token in _FORBIDDEN_TOKENS:
        if token in command:
            return ToolResult(
                tool_name="run_command",
                success=False,
                error=(
                    f"command rejected: contains disallowed token '{token}'. "
                    "Chaining, piping, redirection, and substitution are not permitted; "
                    "run one plain command at a time."
                ),
            )

    try:
        argv: list[str] = shlex.split(command, posix=(os.name != "nt"))
    except ValueError as exc:
        return ToolResult(tool_name="run_command", success=False, error=f"could not parse command: {exc}")

    if not argv:
        return ToolResult(tool_name="run_command", success=False, error="empty command")

    program = argv[0]
    basename = os.path.basename(program)
    basename_no_ext = os.path.splitext(basename)[0].lower()
    allowlist = {entry.lower() for entry in permissions.command_allowlist}

    if basename_no_ext not in allowlist and basename.lower() not in allowlist:
        return ToolResult(
            tool_name="run_command",
            success=False,
            error=(
                f"command '{program}' is not in the allowlist "
                f"({sorted(permissions.command_allowlist)}); refusing to run it."
            ),
        )

    if basename_no_ext == "git":
        subcommand = _git_subcommand(argv)
        if subcommand not in _GIT_SAFE_READONLY_SUBCOMMANDS and not permissions.allow_git_write:
            return ToolResult(
                tool_name="run_command",
                success=False,
                error=(
                    f"git subcommand '{subcommand}' requires permissions.allow_git_write=true. "
                    f"Read-only subcommands allowed without it: {sorted(_GIT_SAFE_READONLY_SUBCOMMANDS)}. "
                    "Enable it with `kenzo config set permissions.allow_git_write true` if you "
                    "trust this agent to modify git history, the index, or remotes."
                ),
            )

    try:
        completed = subprocess.run(
            argv,
            shell=False,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="run_command",
            success=False,
            error=f"command timed out after {timeout}s: {command}",
        )
    except FileNotFoundError as exc:
        return ToolResult(
            tool_name="run_command",
            success=False,
            error=f"could not execute '{program}': {exc}",
        )
    except OSError as exc:
        return ToolResult(
            tool_name="run_command",
            success=False,
            error=f"could not execute '{program}': {exc}",
        )

    return ToolResult(
        tool_name="run_command",
        success=(completed.returncode == 0),
        output=completed.stdout + completed.stderr,
        data={"returncode": completed.returncode},
    )


def get_tool_specs(repo_root: str, permissions: PermissionsConfig) -> list[ToolSpec]:
    """Build the terminal ToolSpec, bound to this repo_root/permissions."""
    return [
        ToolSpec(
            name="run_command",
            description=(
                "Run a single, safe shell command (no chaining/piping/redirection) inside "
                "the repository, subject to a configured command allowlist. May be disabled "
                "by config."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command and its arguments, e.g. 'pytest -q' or 'git status'.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Maximum number of seconds to allow the command to run. Defaults to "
                            "300s, enough headroom for a typical unit test run (including Java "
                            "builds via Maven/Gradle); pass a larger value for slower commands."
                        ),
                        "default": 300,
                    },
                },
                "required": ["command"],
            },
            func=functools.partial(run_command, repo_root=repo_root, permissions=permissions),
        )
    ]
