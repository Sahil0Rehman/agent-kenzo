"""Tests for Phase 10 "Self Improvement System" (`kenzo/improve/*.py` +
`kenzo/cli/commands/improve.py`'s underlying machinery + the 3 new
Phase 10 worker types in `kenzo/workers/*.py`).

Sections:
  1. `kenzo.improve.self_repo.is_self_repo`
  2. `kenzo.improve.self_repo.is_protected_path`
  3. `kenzo.improve.self_repo.SelfModificationGuardRegistry`
  4. `kenzo.improve.analysis` (find_large_files/find_dependency_cycles/
     find_unused_modules/find_duplicate_functions/find_test_gaps/
     render_issue_summary/run_full_analysis) against real synthetic repos
     built with a real `SymbolIndex`/`DependencyGraph`.
  5. `kenzo.improve.regression_guard` (run_test_command against real tiny
     pytest projects, `_resolve_python_test_command`, `check_for_regression`).
  6. `kenzo.improve.history` (ImprovementHistory/ImprovementRecord).
  7. `SelfImprovementAgent._pick_goal`'s priority branches.
  8. The 3 new WorkerType tool-name tables never include a mutating tool.
  9. Full integration scenario: intentional duplicate function -> auto-picked
     goal -> real AgentController EDIT consolidates it -> real pytest suite
     stays green -> recorded to `.kenzo/improvements/`.
  10. Full integration scenario: a bad EDIT breaks the real pytest suite ->
      regression detected -> automatic rollback -> file restored byte-for-byte.

Faking approach for 9/10, mirroring `tests/test_controller_scenarios.py` and
`tests/test_workers.py`'s section 9: only the model/router layer is faked.
`AgentController`, the real tool registry (`build_default_registry`), real
`RepositoryMemory`, real `KenzoConfig`, a real `SymbolIndex`/`DependencyGraph`
pair, and the real `CheckpointManager`/regression-guard subprocess pytest
runs are all genuine -- nothing about safety-relevant behavior is mocked.
`SelfImprovementAgent` only needs a duck-typed `runtime` exposing
`.repo_path`/`.controller`/`.indexing_service`/`.project_info` (see its own
`__init__`), so a `types.SimpleNamespace` standing in for the full
`AgentRuntime` wrapper class is used here -- every component IT actually
touches (controller/tools/memory/config/symbol index/dependency graph/
checkpoint) is the real thing; only the heavier `AgentRuntime.__init__`
plumbing (background indexing thread, embeddings, MCP client) that
`SelfImprovementAgent` never touches is skipped, for a deterministic,
non-flaky test.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from kenzo.config.settings import KenzoConfig, PermissionsConfig
from kenzo.core.controller import AgentController, ControllerAction
from kenzo.core.project import ProjectAnalyzer
from kenzo.core.schemas import ToolResult
from kenzo.improve.analysis import (
    find_dependency_cycles,
    find_duplicate_functions,
    find_large_files,
    find_test_gaps,
    find_unused_modules,
    render_issue_summary,
    run_full_analysis,
)
from kenzo.improve.history import ImprovementHistory, ImprovementRecord
from kenzo.improve.regression_guard import TestRunResult as _TestRunResult  # aliased: pytest's default
# class-discovery pattern would otherwise try (and fail) to collect this as a test class,
# since its unaliased name starts with "Test" and it has an __init__ (mirrors
# tests/test_workers.py's identical TestWorker aliasing).
from kenzo.improve.regression_guard import (
    _resolve_python_test_command,
    check_for_regression,
    run_test_command,
)
from kenzo.improve.self_improvement_agent import SelfImprovementAgent
from kenzo.improve.self_repo import (
    SelfModificationGuardRegistry,
    is_protected_path,
    is_self_repo,
)
from kenzo.intelligence.dependency_graph import DependencyGraph
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.registry import build_default_registry
from kenzo.workers.models import WorkerType
from kenzo.workers.worker_agent import WORKER_TOOL_NAMES
from tests.conftest import run_git

_KENZO_REPO_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# 1. is_self_repo
# ===========================================================================


def test_is_self_repo_true_for_the_real_kenzo_checkout():
    assert (_KENZO_REPO_ROOT / "pyproject.toml").is_file(), "sanity check: this really is the Kenzo checkout"
    assert is_self_repo(str(_KENZO_REPO_ROOT)) is True


def test_is_self_repo_false_for_tmp_path_with_no_pyproject(tmp_path):
    assert is_self_repo(str(tmp_path)) is False


def test_is_self_repo_false_for_pyproject_naming_a_different_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "some-other-project"\n', encoding="utf-8")
    (tmp_path / "kenzo" / "core").mkdir(parents=True)
    (tmp_path / "kenzo" / "core" / "controller.py").write_text("", encoding="utf-8")
    assert is_self_repo(str(tmp_path)) is False


def test_is_self_repo_false_when_name_matches_but_controller_py_is_missing(tmp_path):
    # Both signals (pyproject name AND kenzo/core/controller.py) must agree --
    # a name match alone is not enough.
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "agent-kenzo"\n', encoding="utf-8")
    assert is_self_repo(str(tmp_path)) is False


# ===========================================================================
# 2. is_protected_path
# ===========================================================================


@pytest.mark.parametrize(
    "path",
    [".kenzo/memory.md", ".kenzo/improvements/x.json", "config/secret_key.txt", "path/.env"],
)
def test_is_protected_path_true_for_protected_paths(path):
    assert is_protected_path(path) is True


@pytest.mark.parametrize("path", ["kenzo/core/controller.py", "tests/test_foo.py"])
def test_is_protected_path_false_for_ordinary_source_paths(path):
    assert is_protected_path(path) is False


# ===========================================================================
# 3. SelfModificationGuardRegistry
# ===========================================================================


class _FakeWrappedRegistry:
    """A fake wrapped tool registry: `execute()` either raises (to PROVE the
    guard never reaches it for a protected path) or records the call and
    returns a real `ToolResult` (to prove a non-protected call passes
    through unchanged)."""

    def __init__(self, raise_on_execute: bool = False) -> None:
        self.raise_on_execute = raise_on_execute
        self.execute_calls: list[tuple[str, dict]] = []
        self.register_calls: list[object] = []
        self.register_many_calls: list[object] = []
        self.subset_calls: list[list[str]] = []

    def execute(self, name, arguments):
        if self.raise_on_execute:
            raise AssertionError(
                f"wrapped registry.execute() must NEVER be called for a protected path "
                f"(got name={name!r} arguments={arguments!r})"
            )
        self.execute_calls.append((name, dict(arguments)))
        return ToolResult(tool_name=name, success=True, data={"ok": True})

    def to_openai_tools(self):
        return [{"type": "function", "function": {"name": "fake_tool"}}]

    def names(self):
        return ["fake_tool"]

    def get(self, name):
        return f"spec-for-{name}"

    def list_specs(self):
        return ["spec1", "spec2"]

    def register(self, spec):
        self.register_calls.append(spec)

    def register_many(self, specs):
        self.register_many_calls.append(specs)

    def subset(self, names):
        self.subset_calls.append(list(names))
        return f"subset-of-{names}"


def test_guard_registry_refuses_protected_write_file_without_reaching_wrapped():
    wrapped = _FakeWrappedRegistry(raise_on_execute=True)
    guard = SelfModificationGuardRegistry(wrapped)

    result = guard.execute("write_file", {"path": ".kenzo/memory.md", "content": "hacked"})

    assert result.success is False
    assert result.error
    assert ".kenzo/memory.md" in result.error


def test_guard_registry_refuses_protected_edit_file_without_reaching_wrapped():
    wrapped = _FakeWrappedRegistry(raise_on_execute=True)
    guard = SelfModificationGuardRegistry(wrapped)

    result = guard.execute("edit_file", {"path": "config/secret_key.txt", "old_text": "a", "new_text": "b"})

    assert result.success is False
    assert "secret_key.txt" in result.error


def test_guard_registry_passes_through_write_to_non_protected_path():
    wrapped = _FakeWrappedRegistry()
    guard = SelfModificationGuardRegistry(wrapped)

    result = guard.execute("write_file", {"path": "kenzo/core/controller.py", "content": "x"})

    assert result.success is True
    assert wrapped.execute_calls == [("write_file", {"path": "kenzo/core/controller.py", "content": "x"})]


def test_guard_registry_passes_through_edit_to_non_protected_path():
    wrapped = _FakeWrappedRegistry()
    guard = SelfModificationGuardRegistry(wrapped)

    result = guard.execute("edit_file", {"path": "tests/test_foo.py", "old_text": "a", "new_text": "b"})

    assert result.success is True
    assert wrapped.execute_calls == [("edit_file", {"path": "tests/test_foo.py", "old_text": "a", "new_text": "b"})]


def test_guard_registry_passes_through_non_mutating_calls_even_for_a_protected_path():
    wrapped = _FakeWrappedRegistry()
    guard = SelfModificationGuardRegistry(wrapped)

    # read_file is not in _MUTATING -- the guard only ever gates write_file/edit_file.
    result = guard.execute("read_file", {"path": ".kenzo/memory.md"})

    assert result.success is True
    assert wrapped.execute_calls == [("read_file", {"path": ".kenzo/memory.md"})]


def test_guard_registry_delegates_every_other_method_to_wrapped():
    wrapped = _FakeWrappedRegistry()
    guard = SelfModificationGuardRegistry(wrapped)

    assert guard.to_openai_tools() == wrapped.to_openai_tools()
    assert guard.names() == ["fake_tool"]
    assert guard.get("fake_tool") == "spec-for-fake_tool"
    assert guard.list_specs() == ["spec1", "spec2"]

    guard.register("spec-x")
    assert wrapped.register_calls == ["spec-x"]

    guard.register_many(["spec-a", "spec-b"])
    assert wrapped.register_many_calls == [["spec-a", "spec-b"]]

    subset = guard.subset(["read_file"])
    assert isinstance(subset, SelfModificationGuardRegistry)
    assert wrapped.subset_calls == [["read_file"]]


# ===========================================================================
# 4. kenzo.improve.analysis -- real SymbolIndex/DependencyGraph
# ===========================================================================


def _graphs(repo_root: Path) -> tuple[SymbolIndex, DependencyGraph]:
    symbol_index = SymbolIndex(str(repo_root))
    symbol_index.build()
    dependency_graph = DependencyGraph(symbol_index)
    dependency_graph.build()
    return symbol_index, dependency_graph


# -- find_large_files -------------------------------------------------------


def _write_n_lines(path: Path, n: int) -> None:
    lines = [f"x{i} = {i}" for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def test_find_large_files_flags_file_over_threshold_and_skips_file_under(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_n_lines(repo / "big.py", 20)
    _write_n_lines(repo / "small.py", 3)

    symbol_index, _ = _graphs(repo)
    results = find_large_files(symbol_index, str(repo), threshold_lines=10)

    files = {r["file"] for r in results}
    assert "big.py" in files
    assert "small.py" not in files
    big_entry = next(r for r in results if r["file"] == "big.py")
    assert big_entry["lines"] == 20


# -- find_dependency_cycles --------------------------------------------------


def test_find_dependency_cycles_detects_a_real_circular_import(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("import b\n\n\ndef fn_a():\n    return b.fn_b()\n", encoding="utf-8", newline="")
    (repo / "b.py").write_text("import a\n\n\ndef fn_b():\n    return a.fn_a()\n", encoding="utf-8", newline="")

    _symbol_index, dependency_graph = _graphs(repo)
    # Sanity check the resolved file-level import edges really are circular
    # before trusting find_dependency_cycles' verdict on top of them.
    assert dependency_graph.file_deps.get("a.py") == ["b.py"]
    assert dependency_graph.file_deps.get("b.py") == ["a.py"]

    cycles = find_dependency_cycles(dependency_graph)

    assert len(cycles) == 1
    cycle = cycles[0]
    assert cycle[0] == cycle[-1]
    assert set(cycle) == {"a.py", "b.py"}


def test_find_dependency_cycles_finds_none_for_acyclic_imports(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "c.py").write_text("def fn_c():\n    return 1\n", encoding="utf-8", newline="")
    (repo / "d.py").write_text("import c\n\n\ndef fn_d():\n    return c.fn_c()\n", encoding="utf-8", newline="")

    _symbol_index, dependency_graph = _graphs(repo)
    assert dependency_graph.file_deps.get("d.py") == ["c.py"]

    assert find_dependency_cycles(dependency_graph) == []


# -- find_unused_modules ------------------------------------------------------


def test_find_unused_modules_flags_orphan_but_not_imported_file_or_entry_point(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "used.py").write_text("def helper():\n    return 1\n", encoding="utf-8", newline="")
    (repo / "importer.py").write_text("import used\n\n\ndef call_helper():\n    return used.helper()\n", encoding="utf-8", newline="")
    (repo / "unused.py").write_text("def orphan():\n    return 2\n", encoding="utf-8", newline="")
    (repo / "main.py").write_text("def run():\n    return 0\n", encoding="utf-8", newline="")

    symbol_index, dependency_graph = _graphs(repo)
    unused = find_unused_modules(symbol_index, dependency_graph)

    assert "unused.py" in unused
    assert "used.py" not in unused
    # main.py is a conventional entry point -- excluded even though nothing
    # imports it either.
    assert "main.py" not in unused


# -- find_duplicate_functions -------------------------------------------------

_DUPLICATE_FUNCTIONS_REPO = {
    # The off-by-one regression case: DIFFERENT names, IDENTICAL (4-line)
    # bodies -- must be detected as a duplicate.
    "dup_functions.py": (
        "def calculate_total(data):\n"
        "    total = 0\n"
        "    for value in data:\n"
        "        total = total + value\n"
        "    return total\n"
        "\n\n"
        "def sum_all(data):\n"
        "    total = 0\n"
        "    for value in data:\n"
        "        total = total + value\n"
        "    return total\n"
    ),
    # Two functions with genuinely different (4-line each) bodies -- must
    # NOT be flagged.
    "diff_bodies.py": (
        "def alpha():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    z = x + y\n"
        "    return z\n"
        "\n\n"
        "def beta():\n"
        "    a = 10\n"
        "    b = 20\n"
        "    c = a - b\n"
        "    return c\n"
    ),
    # Two functions with an IDENTICAL but trivially short (1-line) body --
    # duplicated, but must be filtered out by min_lines.
    "short_dup.py": (
        "def short_one():\n"
        "    return 42\n"
        "\n\n"
        "def short_two():\n"
        "    return 42\n"
    ),
}


@pytest.fixture
def duplicate_functions_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for name, content in _DUPLICATE_FUNCTIONS_REPO.items():
        (repo / name).write_text(content, encoding="utf-8", newline="")
    return repo


def test_find_duplicate_functions_detects_identical_bodies_with_different_names(duplicate_functions_repo):
    symbol_index, _ = _graphs(duplicate_functions_repo)

    duplicates = find_duplicate_functions(symbol_index, str(duplicate_functions_repo))

    matching = [
        d
        for d in duplicates
        if {(loc["file"], loc["name"]) for loc in d["locations"]} == {("dup_functions.py", "calculate_total"), ("dup_functions.py", "sum_all")}
    ]
    assert len(matching) == 1, f"expected the calculate_total/sum_all duplicate pair, got {duplicates}"


def test_find_duplicate_functions_does_not_flag_genuinely_different_bodies(duplicate_functions_repo):
    symbol_index, _ = _graphs(duplicate_functions_repo)

    duplicates = find_duplicate_functions(symbol_index, str(duplicate_functions_repo))

    flagged_names = {loc["name"] for d in duplicates for loc in d["locations"]}
    assert "alpha" not in flagged_names
    assert "beta" not in flagged_names


def test_find_duplicate_functions_filters_out_bodies_under_min_lines(duplicate_functions_repo):
    symbol_index, _ = _graphs(duplicate_functions_repo)

    duplicates = find_duplicate_functions(symbol_index, str(duplicate_functions_repo), min_lines=4)

    flagged_names = {loc["name"] for d in duplicates for loc in d["locations"]}
    assert "short_one" not in flagged_names
    assert "short_two" not in flagged_names


# -- find_test_gaps ------------------------------------------------------------


def test_find_test_gaps_flags_module_with_no_test_but_not_one_that_has_one(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "documented.py").write_text("def documented_fn():\n    return 1\n", encoding="utf-8", newline="")
    (repo / "test_documented.py").write_text(
        "from documented import documented_fn\n\n\ndef test_documented_fn():\n    assert documented_fn() == 1\n",
        encoding="utf-8",
        newline="",
    )
    (repo / "orphan_module.py").write_text("def orphan_fn():\n    return 2\n", encoding="utf-8", newline="")

    symbol_index, _ = _graphs(repo)
    gaps = find_test_gaps(symbol_index)

    gap_files = {g["file"] for g in gaps}
    assert "orphan_module.py" in gap_files
    assert "documented.py" not in gap_files


# -- render_issue_summary -------------------------------------------------------


def test_render_issue_summary_empty_facts_returns_a_clean_message_not_a_crash():
    result = render_issue_summary({})
    assert result == (
        "Issues found: none -- analysis found no large files, dependency cycles, "
        "duplicate functions, unused modules, or test gaps under the current thresholds."
    )


def test_render_issue_summary_renders_a_numbered_list_for_populated_facts():
    facts = {
        "dependency_cycles": [["a.py", "b.py", "a.py"]],
        "duplicate_functions": [
            {"locations": [{"file": "dup.py", "name": "f1"}, {"file": "dup.py", "name": "f2"}], "line_count": 5}
        ],
        "large_files": [{"file": "big.py", "lines": 500}],
        "unused_modules": ["orphan.py"],
        "test_gaps": [{"file": "gapped.py", "classes": [], "functions": ["fn"]}],
    }

    result = render_issue_summary(facts)

    assert result.startswith("Issues found:\n1. Dependency cycle: a.py -> b.py -> a.py")
    assert "2. Duplicate logic (5 lines) in dup.py:f1, dup.py:f2" in result
    assert "3. Large file: big.py (500 lines)" in result
    assert "4. Possibly unused module: orphan.py" in result
    assert "5. Missing tests for gapped.py" in result


def test_run_full_analysis_combines_all_five_checks_into_one_report(duplicate_functions_repo):
    symbol_index, dependency_graph = _graphs(duplicate_functions_repo)

    report = run_full_analysis(symbol_index, dependency_graph, str(duplicate_functions_repo))

    assert set(report.keys()) == {"large_files", "dependency_cycles", "unused_modules", "duplicate_functions", "test_gaps"}
    assert report["duplicate_functions"]  # the calculate_total/sum_all pair, real from disk


# ===========================================================================
# 5. kenzo.improve.regression_guard
# ===========================================================================

_PASSING_TEST_PY = "def test_one():\n    assert 1 + 1 == 2\n\n\ndef test_two():\n    assert \"a\" + \"b\" == \"ab\"\n"
_FAILING_TEST_PY = "def test_ok():\n    assert 1 == 1\n\n\ndef test_broken():\n    assert 1 == 2\n"


def _minimal_pyproject(name: str) -> str:
    return f'[project]\nname = "{name}"\n'


def test_run_test_command_parses_a_real_all_passing_pytest_project(tmp_path):
    repo = tmp_path / "passing_repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(_minimal_pyproject("tmp-passing"), encoding="utf-8", newline="")
    (repo / "test_sample.py").write_text(_PASSING_TEST_PY, encoding="utf-8", newline="")

    result = run_test_command(f'"{sys.executable}" -m pytest', str(repo))

    assert result.ran is True
    assert result.passed == 2
    assert result.failed == 0
    assert result.ok is True


def test_run_test_command_parses_a_real_failing_pytest_project(tmp_path):
    repo = tmp_path / "failing_repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(_minimal_pyproject("tmp-failing"), encoding="utf-8", newline="")
    (repo / "test_sample.py").write_text(_FAILING_TEST_PY, encoding="utf-8", newline="")

    result = run_test_command(f'"{sys.executable}" -m pytest', str(repo))

    assert result.ran is True
    assert result.passed == 1
    assert result.failed == 1
    assert result.ok is False


def test_resolve_python_test_command_rewrites_bare_pytest_to_use_sys_executable():
    resolved = _resolve_python_test_command("pytest")
    assert sys.executable in resolved
    assert "-m pytest" in resolved


def test_resolve_python_test_command_rewrites_bare_pytest_with_surrounding_whitespace():
    resolved = _resolve_python_test_command("  pytest  ")
    assert sys.executable in resolved
    assert resolved.endswith("-m pytest")


@pytest.mark.parametrize("command", ["npm test", "mvn test", "pytest -v", "./gradlew test"])
def test_resolve_python_test_command_leaves_non_bare_pytest_commands_untouched(command):
    assert _resolve_python_test_command(command) == command


def _run(command="pytest", cwd=".", ran=True, passed=0, failed=0):
    return _TestRunResult(command=command, cwd=cwd, ran=ran, passed=passed, failed=failed)


def test_check_for_regression_no_regression_when_passed_holds_and_nothing_new_fails():
    result = check_for_regression([_run(passed=5, failed=0)], [_run(passed=5, failed=0)])
    assert result.regressed is False
    assert result.reason == ""


def test_check_for_regression_no_regression_when_passed_count_improves():
    result = check_for_regression([_run(passed=5, failed=0)], [_run(passed=7, failed=0)])
    assert result.regressed is False


def test_check_for_regression_detects_dropped_passed_count():
    result = check_for_regression([_run(passed=5, failed=0)], [_run(passed=4, failed=0)])
    assert result.regressed is True
    assert "passed count dropped" in result.reason


def test_check_for_regression_detects_a_new_failure():
    result = check_for_regression([_run(passed=5, failed=0)], [_run(passed=5, failed=1)])
    assert result.regressed is True
    assert "failing test" in result.reason


def test_check_for_regression_detects_command_that_did_not_run_after():
    before = [_run(passed=5, failed=0, ran=True)]
    after = [_run(passed=0, failed=0, ran=False)]
    result = check_for_regression(before, after)
    assert result.regressed is True
    assert "could not be run" in result.reason


def test_check_for_regression_detects_command_count_mismatch():
    before = [_run(passed=5, failed=0)]
    after = [_run(passed=5, failed=0), _run(command="npm test", passed=3, failed=0)]
    result = check_for_regression(before, after)
    assert result.regressed is True
    assert "command count changed" in result.reason


def test_check_for_regression_a_before_run_that_never_ran_is_not_itself_a_regression():
    before = [_run(passed=0, failed=0, ran=False)]
    after = [_run(passed=0, failed=0, ran=True)]
    result = check_for_regression(before, after)
    assert result.regressed is False


# ===========================================================================
# 6. kenzo.improve.history
# ===========================================================================


def test_improvement_history_record_writes_a_valid_reloadable_json_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    history = ImprovementHistory(str(repo))
    record = ImprovementRecord(
        timestamp="2026-07-06T00:00:00",
        problem="fix X",
        solution="did Y",
        files_changed=["a.py"],
        success=True,
    )

    written_path = history.record(record)

    assert written_path.exists()
    assert written_path.parent == repo / ".kenzo" / "improvements"
    data = json.loads(written_path.read_text(encoding="utf-8"))
    assert data["problem"] == "fix X"
    assert data["solution"] == "did Y"
    assert data["files_changed"] == ["a.py"]
    assert data["success"] is True


def test_improvement_history_list_all_orders_oldest_first_and_skips_a_corrupt_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    history = ImprovementHistory(str(repo))

    history.record(ImprovementRecord(timestamp="t1", problem="first"))
    history.record(ImprovementRecord(timestamp="t2", problem="second"))

    # A corrupted file, deliberately named to sort BEFORE both real records
    # (so a naive "just take list order" implementation without exception
    # handling would trip over it first).
    corrupt_path = history.dir / "00000000-000000-000000.json"
    corrupt_path.write_text("{not valid json at all", encoding="utf-8")

    entries = history.list_all()

    assert [e["problem"] for e in entries] == ["first", "second"]


def test_improvement_history_list_all_returns_empty_list_when_nothing_recorded_yet(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    history = ImprovementHistory(str(repo))

    assert history.list_all() == []


# ===========================================================================
# 7. SelfImprovementAgent._pick_goal priority branches
# ===========================================================================


class _MinimalRuntime:
    """The minimal duck-typed surface `SelfImprovementAgent.__init__` needs
    when `force_self=False` (no `.controller` access at all in that branch)
    -- enough to unit-test `_pick_goal`, which is a pure function of the
    `analysis` dict and never touches `self.runtime`."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path


def _agent(tmp_path) -> SelfImprovementAgent:
    return SelfImprovementAgent(_MinimalRuntime(str(tmp_path)), force_self=False)


def test_pick_goal_prefers_a_high_severity_worker_recommendation(tmp_path):
    agent = _agent(tmp_path)
    analysis = {
        "worker_results": [
            {"severity": "high", "recommendations": ["Fix the race condition in the session cache"], "findings": "x" * 400},
        ],
        "facts": {
            "duplicate_functions": [{"locations": [{"file": "a.py", "name": "f1"}, {"file": "a.py", "name": "f2"}], "line_count": 99}]
        },
    }

    goal = agent._pick_goal(analysis)

    assert goal.startswith("Fix the race condition in the session cache (context:")
    assert len(goal) < len("Fix the race condition in the session cache (context: ") + 300 + 2


def test_pick_goal_prefers_critical_severity_too():
    agent = _agent(Path("."))
    analysis = {
        "worker_results": [{"severity": "critical", "recommendations": ["Patch the SQL injection in login"], "findings": "details"}],
        "facts": {},
    }

    goal = agent._pick_goal(analysis)

    assert goal.startswith("Patch the SQL injection in login (context: details)")


def test_pick_goal_falls_back_to_the_largest_duplicate_function_finding():
    agent = _agent(Path("."))
    analysis = {
        "worker_results": [{"severity": None, "recommendations": []}, {"severity": "low", "recommendations": ["minor nit"]}],
        "facts": {
            "duplicate_functions": [
                {"locations": [{"file": "a.py", "name": "f1"}, {"file": "a.py", "name": "f2"}], "line_count": 12},
            ]
        },
    }

    goal = agent._pick_goal(analysis)

    assert "Remove the duplicate logic found at a.py:f1, a.py:f2" in goal
    assert "12-line" in goal
    assert "consolidating" in goal


def test_pick_goal_falls_back_to_any_worker_recommendation_when_no_duplicates():
    agent = _agent(Path("."))
    analysis = {
        "worker_results": [
            {"severity": "low", "recommendations": []},
            {"severity": None, "recommendations": ["Add missing tests for the payment module"]},
        ],
        "facts": {},
    }

    goal = agent._pick_goal(analysis)

    assert goal == "Add missing tests for the payment module"


def test_pick_goal_falls_back_to_a_large_file_finding():
    agent = _agent(Path("."))
    analysis = {
        "worker_results": [{"severity": None, "recommendations": []}],
        "facts": {"large_files": [{"file": "huge.py", "lines": 900}]},
    }

    goal = agent._pick_goal(analysis)

    assert "huge.py" in goal
    assert "900 lines" in goal


def test_pick_goal_ultimate_fallback_is_never_empty():
    agent = _agent(Path("."))
    analysis = {"worker_results": [], "facts": {}}

    goal = agent._pick_goal(analysis)

    assert goal == "Investigate the codebase for one safe, well-scoped improvement and make it, verifying with tests."
    assert goal.strip() != ""


# ===========================================================================
# 8. The 3 new Phase 10 WorkerType tool-name tables
# ===========================================================================

_NEW_WORKER_TYPES = [WorkerType.ARCHITECTURE_CRITIC, WorkerType.TEST_GAP, WorkerType.PERFORMANCE]
_FORBIDDEN_NATIVE_TOOLS = {"write_file", "edit_file", "run_command"}


@pytest.mark.parametrize("worker_type", _NEW_WORKER_TYPES)
def test_new_phase10_worker_types_never_get_a_write_or_shell_tool(worker_type):
    names = set(WORKER_TOOL_NAMES.get(worker_type, []))
    assert not (names & _FORBIDDEN_NATIVE_TOOLS), f"{worker_type} unexpectedly has a mutating tool: {names}"


@pytest.mark.parametrize("worker_type", _NEW_WORKER_TYPES)
def test_new_phase10_worker_types_have_at_least_one_read_tool(worker_type):
    assert WORKER_TOOL_NAMES.get(worker_type), f"{worker_type} has no tools configured at all"


def test_performance_worker_specifically_gets_no_run_command_despite_analyzing_slow_paths():
    # Explicit regression guard for the module's own callout: PERFORMANCE is
    # pure static analysis, never allowed to actually execute anything.
    perf_tools = set(WORKER_TOOL_NAMES[WorkerType.PERFORMANCE])
    assert "run_command" not in perf_tools
    assert "write_file" not in perf_tools
    assert "edit_file" not in perf_tools


# ===========================================================================
# 9/10. Full integration scenarios (real AgentController/tools/memory/config,
# only the router faked)
# ===========================================================================


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeModelResponse:
    def __init__(self, content, tool_calls=None):
        self.choices = [_FakeChoice(_FakeMessage(content, tool_calls))]


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _FakeFunction(name, arguments)


def _tool_call_response(call_id: str, tool_name: str, arguments: dict):
    return _FakeModelResponse(None, tool_calls=[_FakeToolCall(call_id, tool_name, json.dumps(arguments))])


def _final_text(text: str):
    return _FakeModelResponse(text)


def _is_decide_call(tool_choice) -> bool:
    return isinstance(tool_choice, dict) and tool_choice.get("function", {}).get("name") == "decide_next_action"


def _decision(action: ControllerAction, rationale: str, target: str = ""):
    call = _FakeToolCall(
        f"decide-{action.value}",
        "decide_next_action",
        json.dumps({"action": action.value, "rationale": rationale, "target": target}),
    )
    return _FakeModelResponse(None, tool_calls=[call])


# -- fixture repo: an intentional, real duplicate between two functions -----

_MATH_UTILS_PYPROJECT = '[project]\nname = "math-utils-demo"\n'

_MATH_UTILS_PY = (
    "def calculate_total(data):\n"
    "    total = 0\n"
    "    for value in data:\n"
    "        total = total + value\n"
    "    return total\n"
    "\n\n"
    "def sum_all(data):\n"
    "    total = 0\n"
    "    for value in data:\n"
    "        total = total + value\n"
    "    return total\n"
)

_TEST_MATH_UTILS_PY = (
    "from math_utils import calculate_total, sum_all\n"
    "\n\n"
    "def test_calculate_total():\n"
    "    assert calculate_total([1, 2, 3]) == 6\n"
    "\n\n"
    "def test_sum_all():\n"
    "    assert sum_all([4, 5]) == 9\n"
)

_OLD_SUM_ALL = (
    "def sum_all(data):\n"
    "    total = 0\n"
    "    for value in data:\n"
    "        total = total + value\n"
    "    return total\n"
)
_NEW_SUM_ALL_CONSOLIDATED = "def sum_all(data):\n    return calculate_total(data)\n"
_NEW_SUM_ALL_BROKEN = (
    "def sum_all(data):\n"
    "    total = 0\n"
    "    for value in data:\n"
    "        total = total + value\n"
    "    return total + 999\n"
)


def _build_improve_repo(tmp_path: Path) -> Path:
    # newline="" avoids Windows' platform newline translation -- edit_file's
    # old_text below must match the file's actual on-disk bytes exactly (see
    # this project's own established fix, e.g. tests/test_workers.py).
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(_MATH_UTILS_PYPROJECT, encoding="utf-8", newline="")
    (repo / "math_utils.py").write_text(_MATH_UTILS_PY, encoding="utf-8", newline="")
    (repo / "test_math_utils.py").write_text(_TEST_MATH_UTILS_PY, encoding="utf-8", newline="")

    run_git(["init"], cwd=repo)
    run_git(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=repo)
    run_git(["config", "user.email", "kenzo-tests@example.com"], cwd=repo)
    run_git(["config", "user.name", "Kenzo Tests"], cwd=repo)
    # Force LF in/out regardless of the host machine's global core.autocrlf
    # setting -- otherwise a later `git checkout <ref> -- .` (rollback) can
    # re-normalize LF -> CRLF on checkout even though the committed blob
    # itself is untouched, which would make the byte-for-byte restore
    # assertion below flaky depending on which machine runs the suite.
    run_git(["config", "core.autocrlf", "false"], cwd=repo)
    run_git(["add", "-A"], cwd=repo)
    run_git(["commit", "-m", "Add math_utils with a duplicated summation function"], cwd=repo)
    return repo


def _build_fake_runtime(repo: Path, router) -> SimpleNamespace:
    memory = RepositoryMemory(str(repo))
    tools = build_default_registry(str(repo), PermissionsConfig(allow_write=True))
    controller = AgentController(router, tools, memory, KenzoConfig())

    symbol_index = SymbolIndex(str(repo))
    symbol_index.build()
    dependency_graph = DependencyGraph(symbol_index)
    dependency_graph.build()

    project_info = ProjectAnalyzer().analyze(str(repo))
    assert project_info.language == "python"
    assert project_info.test_command == "pytest"

    return SimpleNamespace(
        repo_path=str(repo),
        controller=controller,
        indexing_service=SimpleNamespace(symbol_index=symbol_index, dependency_graph=dependency_graph),
        project_info=project_info,
    )


# Markers matching the FIXED WorkerTask.description text `_build_worker_tasks`
# in kenzo.improve.self_improvement_agent always uses -- present in the
# worker's own "Investigation task: <description>" prompt line regardless of
# what pre-computed facts happen to be attached.
_ARCH_MARKER = "bad abstractions"
_TESTGAP_MARKER = "untested modules"
_PERF_MARKER = "expensive indexing"

_WORKER_RESULT_ARCH = json.dumps(
    {
        "findings": "No dependency cycles or unused modules found; the only concern is the duplicated "
        "summation logic between calculate_total and sum_all in math_utils.py.",
        "confidence": 0.6,
        "files": ["math_utils.py"],
        "recommendations": [],
    }
)
_WORKER_RESULT_TESTGAP = json.dumps(
    {
        "findings": "math_utils.py is already covered by test_math_utils.py; no coverage gap found.",
        "confidence": 0.7,
        "files": ["test_math_utils.py"],
        "recommendations": [],
    }
)
_WORKER_RESULT_PERF = json.dumps(
    {
        "findings": "Both functions are simple O(n) linear scans over small collections; no performance concern.",
        "confidence": 0.5,
        "files": [],
        "recommendations": [],
    }
)


class _ImproveScenarioRouter:
    """Fake router for the full `SelfImprovementAgent.improve()` integration
    scenario. Handles two distinct call styles, thread-safely (see this
    module's own docstring and the Phase 9 task brief's threading gotcha):

    (a) `tool_choice` names decide_next_action -> popped off a strict,
        main-thread-only ordered `decisions` queue (used by
        `AgentController.run()`'s own decision loop).
    (b) Every other `complete(...)` call is either one of the 3 CONCURRENT
        `analyze()` worker calls (identified unambiguously by the
        "Investigation task:" prefix persisting anywhere in that worker's
        OWN message history -- see this module's docstring -- then
        disambiguated by a marker substring unique to each worker's fixed
        task description) or one of the main-thread action sub-loop calls
        (READ/EDIT/TEST), served in strict order off a flat `actions` queue.
    """

    def __init__(self, decisions, actions):
        self._lock = threading.Lock()
        self._decisions = list(decisions)
        self._actions = list(actions)
        self.decision_calls: list[list[dict]] = []
        self.worker_call_counts: dict[str, int] = {}

    def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
        raise AssertionError("this scenario's goal is auto-picked/pre-scripted; complete_text should never be called")

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        snapshot = [dict(m) for m in messages]

        if _is_decide_call(tool_choice):
            with self._lock:
                self.decision_calls.append(snapshot)
                if not self._decisions:
                    raise AssertionError("ran out of scripted decisions")
                return self._decisions.pop(0)

        content = json.dumps(snapshot)
        is_worker_call = any("Investigation task:" in (m.get("content") or "") for m in messages)

        if is_worker_call:
            if _ARCH_MARKER in content:
                key, response_text = "worker-arch", _WORKER_RESULT_ARCH
            elif _TESTGAP_MARKER in content:
                key, response_text = "worker-testgap", _WORKER_RESULT_TESTGAP
            elif _PERF_MARKER in content:
                key, response_text = "worker-perf", _WORKER_RESULT_PERF
            else:
                raise AssertionError(f"unrecognized worker call content: {content[:400]}")
            with self._lock:
                self.worker_call_counts[key] = self.worker_call_counts.get(key, 0) + 1
            return _final_text(response_text)

        if not self._actions:
            raise AssertionError("ran out of scripted action responses")
        return self._actions.pop(0)


def _scripted_decisions() -> list:
    return [
        _decision(ControllerAction.READ, "Confirm the duplicate logic in math_utils.py before touching it.", "math_utils.py"),
        _decision(ControllerAction.EDIT, "Consolidate sum_all to delegate to calculate_total.", "math_utils.py"),
        _decision(ControllerAction.TEST, "Validate the change."),
        _decision(ControllerAction.FINISH, "Consolidated the duplicate summation logic; tests still pass."),
    ]


def test_full_scenario_duplicate_function_detected_patched_and_tests_pass(tmp_path):
    repo = _build_improve_repo(tmp_path)
    original_math_utils = (repo / "math_utils.py").read_bytes()

    actions = [
        _final_text("Confirmed: calculate_total and sum_all have identical bodies in math_utils.py."),
        _tool_call_response("call-edit", "edit_file", {"path": "math_utils.py", "old_text": _OLD_SUM_ALL, "new_text": _NEW_SUM_ALL_CONSOLIDATED}),
        _final_text("Consolidated sum_all to delegate to calculate_total."),
        _final_text(json.dumps({"passed": True, "summary": "both tests still pass", "command": "", "raw_output": ""})),
    ]
    router = _ImproveScenarioRouter(_scripted_decisions(), actions)
    runtime = _build_fake_runtime(repo, router)

    agent = SelfImprovementAgent(runtime, force_self=False)
    assert agent.is_self is False

    result = agent.improve()  # request=None -- goal is auto-picked via _pick_goal

    # -- the 3 analyze() workers really ran (concurrently, via the real
    # WorkerManager/ThreadPoolExecutor).
    assert set(router.worker_call_counts.keys()) == {"worker-arch", "worker-testgap", "worker-perf"}

    # -- the auto-picked goal really was the duplicate-function branch.
    assert "calculate_total" in result["goal"]
    assert "sum_all" in result["goal"]
    assert "math_utils.py" in result["goal"]

    # -- the real fix landed on disk, and the duplicate is genuinely gone.
    fixed_content = (repo / "math_utils.py").read_text(encoding="utf-8")
    assert _OLD_SUM_ALL not in fixed_content
    assert "return calculate_total(data)" in fixed_content
    assert fixed_content != original_math_utils.decode("utf-8")

    # -- the regression guard ran the REAL pytest suite, not a synthesized count.
    before_runs = result["regression_check"]["before"]
    after_runs = result["regression_check"]["after"]
    assert len(before_runs) == 1
    assert len(after_runs) == 1
    assert before_runs[0]["ran"] is True
    assert before_runs[0]["passed"] == 2
    assert before_runs[0]["failed"] == 0
    assert after_runs[0]["ran"] is True
    assert after_runs[0]["passed"] == 2
    assert after_runs[0]["failed"] == 0

    assert result["regression_check"]["regressed"] is False
    assert result["rolled_back"] is False
    assert result["success"] is True

    # -- a real, readable improvement record was written.
    assert Path(result["history_path"]).exists()
    history_entries = ImprovementHistory(str(repo)).list_all()
    assert len(history_entries) == 1
    assert history_entries[0]["success"] is True
    assert history_entries[0]["rolled_back"] is False


def test_full_scenario_bad_edit_regresses_and_is_rolled_back_byte_for_byte(tmp_path):
    repo = _build_improve_repo(tmp_path)
    original_math_utils = (repo / "math_utils.py").read_bytes()

    actions = [
        _final_text("Confirmed: calculate_total and sum_all have identical bodies in math_utils.py."),
        _tool_call_response("call-edit", "edit_file", {"path": "math_utils.py", "old_text": _OLD_SUM_ALL, "new_text": _NEW_SUM_ALL_BROKEN}),
        _final_text("Updated sum_all."),
        # The controller's own naive TEST step claims success -- exactly the
        # scenario the INDEPENDENT real-subprocess regression guard exists
        # to catch (it does not trust this claim at all).
        _final_text(json.dumps({"passed": True, "summary": "looks fine", "command": "", "raw_output": ""})),
    ]
    router = _ImproveScenarioRouter(_scripted_decisions(), actions)
    runtime = _build_fake_runtime(repo, router)

    agent = SelfImprovementAgent(runtime, force_self=False)

    result = agent.improve()

    # -- the real pytest suite genuinely failed after the bad edit.
    before_runs = result["regression_check"]["before"]
    after_runs = result["regression_check"]["after"]
    assert before_runs[0]["passed"] == 2
    assert before_runs[0]["failed"] == 0
    assert after_runs[0]["failed"] == 1

    assert result["regression_check"]["regressed"] is True
    reason = result["regression_check"]["reason"]
    assert reason
    assert "failing test" in reason

    assert result["rolled_back"] is True
    assert result["success"] is False

    # -- the strongest claim: re-read from disk AFTER improve() returns and
    # confirm the rollback genuinely restored the original bytes, not just
    # that the returned dict claims success.
    restored_math_utils = (repo / "math_utils.py").read_bytes()
    assert restored_math_utils == original_math_utils

    # -- still recorded, with the rollback reason preserved.
    history_entries = ImprovementHistory(str(repo)).list_all()
    assert len(history_entries) == 1
    assert history_entries[0]["success"] is False
    assert history_entries[0]["rolled_back"] is True
    assert history_entries[0]["rollback_reason"]
