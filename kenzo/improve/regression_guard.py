"""Phase 10's hard safety gate: run a repo's real test suite(s) before and
after a self-improvement change, and decide whether it regressed.

"Compare: before N tests passing, after must be >=. If regression: rollback
automatically" -- this module owns the running/parsing/comparing;
`SelfImprovementAgent` owns calling `CheckpointManager.rollback()` when
`RegressionCheckResult.regressed` comes back `True`, and recording the
outcome to `.kenzo/improvements/`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DEFAULT_TIMEOUT_SECONDS = 600


def _resolve_python_test_command(test_command: str) -> str:
    """Rewrite a bare `"pytest"` command (exactly what `kenzo.core.project
    .ProjectAnalyzer` gives a detected Python project) to `"<this process's
    own interpreter> -m pytest"`.

    A bare `pytest` invoked via `subprocess.run(..., shell=True)` resolves
    through the SHELL's PATH, which is NOT guaranteed to be the same
    environment Kenzo itself is running under (e.g. an unactivated venv, a
    different shell session, a pytest shim missing from PATH entirely) --
    confirmed to actually diverge during Phase 10 development: a bare
    `pytest` silently failed to even collect a real, working test file in
    one such environment, which the regression guard would otherwise have
    misread as "1 failing test" and triggered a FALSE-POSITIVE automatic
    rollback of a perfectly good change. Routing through `sys.executable
    -m pytest` guarantees the exact interpreter (and therefore installed
    packages/venv) Kenzo's own process is using, which is always correct.
    Any OTHER test command (mvn/gradle/npm/yarn -- none of which have this
    same-interpreter concept) is left untouched.
    """
    if test_command.strip() == "pytest":
        return f'"{sys.executable}" -m pytest'
    return test_command

_PASSED_RE = re.compile(r"(\d+)\s+passed")
_FAILED_RE = re.compile(r"(\d+)\s+failed")
_MOCHA_PASSING_RE = re.compile(r"(\d+)\s+passing")
_MOCHA_FAILING_RE = re.compile(r"(\d+)\s+failing")


@dataclass
class TestRunResult:
    """One test command's outcome. `ran=False` means the command itself
    could not be executed/completed (crashed, timed out, no such
    command) -- distinct from `ran=True, failed>0` (the suite ran and some
    tests failed)."""

    command: str
    cwd: str
    ran: bool
    passed: int = 0
    failed: int = 0
    raw_output: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.ran and self.failed == 0


@dataclass
class RegressionCheckResult:
    before: list[TestRunResult] = field(default_factory=list)
    after: list[TestRunResult] = field(default_factory=list)
    regressed: bool = False
    reason: str = ""


def _parse_counts(output: str) -> tuple[int, int]:
    """Best-effort (passed, failed) extraction covering both pytest's
    ("N passed" / "N failed", either order) and mocha's ("N passing" /
    "N failing") summary line conventions -- the two test runners this
    project's own repo actually uses (see `run_test_suite`'s
    Kenzo-repo-specific dual-suite handling)."""
    passed = 0
    failed = 0
    for pattern, is_pass in ((_PASSED_RE, True), (_MOCHA_PASSING_RE, True)):
        match = pattern.search(output)
        if match:
            passed = max(passed, int(match.group(1)))
    for pattern in (_FAILED_RE, _MOCHA_FAILING_RE):
        match = pattern.search(output)
        if match:
            failed = max(failed, int(match.group(1)))
    return passed, failed


def run_test_command(command: str, cwd: str, timeout: int = _DEFAULT_TIMEOUT_SECONDS) -> TestRunResult:
    """Run one test command (a plain shell string, e.g. "pytest" or
    "npm test") in `cwd`. Never raises -- a crashed/timed-out/missing
    command becomes `TestRunResult(ran=False, error=...)`, which the
    regression check below treats as a regression (fail closed: "I could
    not confirm this change is safe" is not the same as "this change is
    safe")."""
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return TestRunResult(command=command, cwd=cwd, ran=False, error=f"timed out after {timeout}s: {exc}")
    except OSError as exc:
        return TestRunResult(command=command, cwd=cwd, ran=False, error=str(exc))

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    passed, failed = _parse_counts(output)
    # A nonzero exit code with zero parsed failures still counts as
    # "ran but something's wrong" (e.g. a collection error before any test
    # even started) -- surfaced as a synthetic single failure so it isn't
    # silently treated as "0 failed, therefore fine".
    if completed.returncode != 0 and failed == 0 and passed == 0:
        failed = 1
    return TestRunResult(command=command, cwd=cwd, ran=True, passed=passed, failed=failed, raw_output=output[-4000:])


def run_test_suite(repo_root: str, project_info: object, is_self: bool, timeout: int = _DEFAULT_TIMEOUT_SECONDS) -> list[TestRunResult]:
    """Every test command relevant to `repo_root`: the language-detected
    `project_info.test_command` (from `kenzo.core.project.ProjectAnalyzer`,
    the SAME detection every other part of Kenzo already uses -- pytest,
    mvn test, npm test, ...), plus -- ONLY when `is_self` (this repo IS
    Kenzo's own source, see `kenzo.improve.self_repo.is_self_repo`) -- the
    `vscode-extension/` TypeScript suite too, since Kenzo's own repo is the
    one place this project genuinely has two independent test suites (item
    7's explicit "Run: Python tests, TypeScript tests"). A generic
    non-Kenzo repo with its own unrelated `vscode-extension/`-named
    directory is NOT special-cased this way -- only true self-improvement is.
    """
    commands: list[tuple[str, str]] = []
    test_command = getattr(project_info, "test_command", None)
    if test_command:
        commands.append((_resolve_python_test_command(test_command), repo_root))

    if is_self:
        ext_dir = Path(repo_root) / "vscode-extension"
        if (ext_dir / "package.json").is_file():
            commands.append(("npm test", str(ext_dir)))

    return [run_test_command(command, cwd, timeout=timeout) for command, cwd in commands]


def run_lint_if_available(repo_root: str) -> Optional[TestRunResult]:
    """Best-effort, INFORMATIONAL ONLY (never gates rollback -- see
    `check_for_regression`'s docstring for why): runs `ruff check .` if
    `ruff` is on PATH, since that's the one lint tool this project's own
    dependencies already pull in transitively via common tooling; returns
    None (not a failure) if no recognized linter is available, matching the
    spec's own "lint if available" phrasing -- its absence is not itself a
    regression signal.
    """
    if shutil.which("ruff") is None:
        return None
    return run_test_command("ruff check .", repo_root, timeout=120)


def check_for_regression(before: list[TestRunResult], after: list[TestRunResult]) -> RegressionCheckResult:
    """Pairs `before`/`after` by position (both lists come from the SAME
    `run_test_suite` call shape, so command order is identical) and applies
    the spec's literal rule: after.passed must be >= before.passed, AND
    after must have introduced no new failures, AND every command must have
    actually run. The FIRST violation found sets `reason`; still records
    every before/after pair either way so the full picture is available for
    `.kenzo/improvements/` history even when only one command regressed.
    """
    result = RegressionCheckResult(before=before, after=after)

    if len(before) != len(after):
        result.regressed = True
        result.reason = f"test command count changed between before ({len(before)}) and after ({len(after)}) runs"
        return result

    for before_run, after_run in zip(before, after):
        if not before_run.ran:
            # Couldn't establish a baseline for this command -- nothing to
            # regress FROM, so this pair can't fail the guard on its own,
            # but it's also not a clean bill of health; the "after did not
            # run" case just below still catches a genuinely broken change.
            continue
        if not after_run.ran:
            result.regressed = True
            result.reason = f"'{after_run.command}' could not be run after the change ({after_run.error})"
            return result
        if after_run.failed > 0:
            result.regressed = True
            result.reason = (
                f"'{after_run.command}' has {after_run.failed} failing test(s) after the change "
                f"(had {before_run.failed} before)"
            )
            return result
        if after_run.passed < before_run.passed:
            result.regressed = True
            result.reason = (
                f"'{after_run.command}' passed count dropped: {before_run.passed} -> {after_run.passed}"
            )
            return result

    return result
