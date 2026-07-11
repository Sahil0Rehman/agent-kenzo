"""Tests for the Phase 9 "Parallel Agent Workers" system: `kenzo.workers.*`
(models, the generic `WorkerAgent` execution engine, the five specialized
worker wrappers, `WorkerResultMerger`, and `WorkerManager`'s real concurrent
execution), plus its wiring into `kenzo.core.controller.AgentController` via
the new `ControllerAction.DELEGATE`.

Faking approach, mirroring `tests/test_controller_scenarios.py`: only the
model/router layer is ever faked. Tools (`build_default_registry`), memory
(`RepositoryMemory`), and config (`KenzoConfig`) are real, running against
real tmp repositories (the `repo_root` fixture from `tests/conftest.py`).

Sections:
  1. WorkerTask / WorkerResult / WorkerType data contracts.
  2. WORKER_TOOL_NAMES / tool_names_for (the read-only access-control table).
  3. WorkerAgent.run() lifecycle: success, graceful-degrade, crash handling,
     observed-files merging, and the scoped-registry access-control boundary.
  4. WorkerManager.run_parallel: real parallelism, max_parallel truncation,
     enabled=False fail-closed, per-worker timeout, isolated crash handling,
     result ordering, and emitted events.
  5. WorkerResultMerger.
  6. The five SpecializedWorker subclasses + build_worker factory.
  7. WorkersConfig defaults and KenzoConfig wiring.
  8. AgentController DELEGATE wiring (unit-level): ControllerAction.DELEGATE,
     _propose_worker_tasks parsing/fallback, _execute_delegate's synthetic
     tool_transcript shape, and _apply_updates merging it correctly.
  9. Full end-to-end integration scenario (real AgentController, real tools,
     real repo, only the router faked): an "OIDC logout broken" investigation
     that delegates to 3 concurrent workers, then uses their findings to
     drive a real EDIT + TEST + FINISH to a verified fix on disk.

Threading-safety note (see the Phase 9 task brief): every fake router used
below that can be called from more than one worker thread protects its
mutable state (call counters, timestamp lists) with a `threading.Lock` --
a plain unprotected `list.pop(0)`/dict increment is a real race condition
here, not a theoretical one, since `WorkerManager.run_parallel` genuinely
runs workers on separate OS threads via `concurrent.futures.ThreadPoolExecutor`.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from kenzo.config.settings import KenzoConfig, PermissionsConfig, WorkersConfig
from kenzo.core.controller import ACTION_TOOLS, AgentController, ControllerAction
from kenzo.core.schemas import ActionDecision, TaskState, TaskType
from kenzo.mcp.permissions import ToolPermission
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.registry import build_default_registry
from kenzo.workers.manager import WorkerManager
from kenzo.workers.merger import WorkerResultMerger
from kenzo.workers.models import WorkerResult, WorkerTask, WorkerType
from kenzo.workers.specialized import (
    WORKER_CLASSES,
    BugAnalysisWorker,
    CodeResearchWorker,
    DocumentationWorker,
    GitWorker,
    build_worker,
)
from kenzo.workers.specialized import TestWorker as _TestWorker  # aliased: pytest's default
# class-discovery pattern would otherwise try (and fail) to collect this as a test class,
# since its unaliased name starts with "Test" and it has an __init__.
from kenzo.workers.worker_agent import WORKER_TOOL_NAMES, WorkerAgent, tool_names_for
from tests.conftest import run_git

# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding (same pattern as
# tests/test_controller_scenarios.py).
# ---------------------------------------------------------------------------


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
    return (
        isinstance(tool_choice, dict)
        and tool_choice.get("function", {}).get("name") == "decide_next_action"
    )


def _decision(action: ControllerAction, rationale: str, target: str = ""):
    call = _FakeToolCall(
        f"decide-{action.value}",
        "decide_next_action",
        json.dumps({"action": action.value, "rationale": rationale, "target": target}),
    )
    return _FakeModelResponse(None, tool_calls=[call])


def _real_tools(repo_root, permissions=None):
    return build_default_registry(str(repo_root), permissions or PermissionsConfig(allow_write=True))


_WORKER_FINDINGS_JSON = json.dumps(
    {
        "findings": "did not find anything conclusive",
        "confidence": 0.5,
        "files": [],
        "recommendations": [],
    }
)


# ===========================================================================
# 1. WorkerTask / WorkerResult / WorkerType
# ===========================================================================


def test_worker_type_has_expected_members():
    assert {member.value for member in WorkerType} == {
        "code_research",
        "bug_analysis",
        "test",
        "documentation",
        "git",
        "architecture_critic",
        "test_gap",
        "performance",
    }


def test_worker_task_defaults():
    task = WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description="find the bug")
    assert task.worker_type == WorkerType.CODE_RESEARCH
    assert task.description == "find the bug"
    assert task.context == ""
    assert task.worker_id == ""


def test_worker_task_full_construction():
    task = WorkerTask(
        worker_type=WorkerType.GIT, description="check history", context="extra context", worker_id="worker-7"
    )
    assert task.context == "extra context"
    assert task.worker_id == "worker-7"


def test_worker_result_defaults():
    result = WorkerResult(worker_type=WorkerType.BUG_ANALYSIS)
    assert result.worker_id == ""
    assert result.task_description == ""
    assert result.success is True
    assert result.findings == ""
    assert result.confidence == 0.0
    assert result.files == []
    assert result.recommendations == []
    assert result.error is None
    assert result.duration_seconds == 0.0
    assert result.tool_calls_made == 0


def test_worker_result_full_construction():
    result = WorkerResult(
        worker_id="w1",
        worker_type=WorkerType.TEST,
        task_description="find tests",
        success=False,
        findings="some findings",
        confidence=0.9,
        files=["a.py"],
        recommendations=["do X"],
        error="boom",
        duration_seconds=1.5,
        tool_calls_made=3,
    )
    assert result.success is False
    assert result.error == "boom"
    assert result.duration_seconds == 1.5
    assert result.tool_calls_made == 3


# ===========================================================================
# 2. WORKER_TOOL_NAMES / tool_names_for
# ===========================================================================

_FORBIDDEN_NATIVE_TOOLS = {"write_file", "edit_file", "run_command"}


@pytest.mark.parametrize("worker_type", list(WorkerType))
def test_no_worker_type_ever_gets_a_write_or_shell_tool(worker_type):
    names = set(WORKER_TOOL_NAMES.get(worker_type, []))
    assert not (names & _FORBIDDEN_NATIVE_TOOLS), f"{worker_type} unexpectedly has a mutating tool: {names}"


@pytest.mark.parametrize("worker_type", list(WorkerType))
def test_every_worker_type_has_at_least_one_read_tool(worker_type):
    assert WORKER_TOOL_NAMES.get(worker_type), f"{worker_type} has no tools configured at all"


@pytest.mark.parametrize("worker_type", list(WorkerType))
def test_tool_names_for_without_mcp_client_matches_static_table(worker_type):
    assert tool_names_for(worker_type) == list(WORKER_TOOL_NAMES.get(worker_type, []))
    assert tool_names_for(worker_type, mcp_client=None) == list(WORKER_TOOL_NAMES.get(worker_type, []))


class _FakeMcpClient:
    """Records exactly which `ToolPermission` values it was asked for, and --
    deliberately, to prove the caller never asks -- would happily return
    WRITE/DESTRUCTIVE tool names too if asked, so the test can catch a
    regression where a worker's tool list leaks a mutating MCP tool."""

    def __init__(self):
        self.requested_permissions: list[ToolPermission] = []

    def tool_names_by_permission(self, permission: ToolPermission) -> list[str]:
        self.requested_permissions.append(permission)
        return {
            ToolPermission.READ: ["mcp_search_docs", "mcp_get_issue"],
            ToolPermission.WRITE: ["mcp_create_pr"],
            ToolPermission.DESTRUCTIVE: ["mcp_drop_table"],
        }[permission]


def test_documentation_worker_gains_mcp_read_tools_only():
    client = _FakeMcpClient()
    names = tool_names_for(WorkerType.DOCUMENTATION, client)

    assert "mcp_search_docs" in names
    assert "mcp_get_issue" in names
    assert "mcp_create_pr" not in names
    assert "mcp_drop_table" not in names
    # The static local fallback tools are still present too.
    assert "read_file" in names

    # The actual CALL kenzo's code makes only ever asks for READ -- never
    # WRITE/DESTRUCTIVE, regardless of what the client would hand back.
    assert client.requested_permissions == [ToolPermission.READ]


def test_code_research_worker_also_gains_mcp_read_tools_only():
    client = _FakeMcpClient()
    names = tool_names_for(WorkerType.CODE_RESEARCH, client)

    assert "mcp_search_docs" in names
    assert "mcp_create_pr" not in names
    assert "mcp_drop_table" not in names
    assert client.requested_permissions == [ToolPermission.READ]


@pytest.mark.parametrize("worker_type", list(WorkerType))
def test_every_worker_type_only_ever_requests_read_permission_mcp_tools(worker_type):
    client = _FakeMcpClient()
    names = tool_names_for(worker_type, client)
    assert client.requested_permissions == [ToolPermission.READ]
    assert not (set(names) & {"mcp_create_pr", "mcp_drop_table"})


# ===========================================================================
# 3. WorkerAgent.run()
# ===========================================================================


class _WorkerScriptedRouter:
    """Ordered-queue fake router for a single, single-threaded
    `WorkerAgent.run()` call (see module docstring)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        snapshot = [dict(m) for m in messages]
        self.calls.append({"task_type": task_type, "messages": snapshot, "tools": tools, "tool_choice": tool_choice})
        if not self._responses:
            raise AssertionError("ran out of scripted worker responses")
        return self._responses.pop(0)


def test_worker_agent_run_success_with_tool_call_and_json_findings(repo_root):
    tools = _real_tools(repo_root)
    findings_json = json.dumps(
        {
            "findings": "main.py defines main(), which currently returns 42.",
            "confidence": 0.85,
            "files": [],  # deliberately does NOT list main.py itself
            "recommendations": ["Check callers of main()."],
        }
    )
    router = _WorkerScriptedRouter(
        [
            _tool_call_response("call-1", "read_file", {"path": "main.py"}),
            _final_text(findings_json),
        ]
    )
    agent = WorkerAgent(router, tools)
    task = WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description="Understand main.py", worker_id="w1")

    result = agent.run(task)

    assert result.success is True
    assert result.worker_id == "w1"
    assert result.worker_type == WorkerType.CODE_RESEARCH
    assert result.task_description == "Understand main.py"
    assert result.confidence == 0.85
    assert result.recommendations == ["Check callers of main()."]
    assert result.tool_calls_made == 1
    assert result.duration_seconds >= 0.0

    # The worker's OWN read_file call surfaced main.py into files, even
    # though the model's JSON "files" list was empty.
    assert "main.py" in result.files


def test_worker_agent_run_degrades_gracefully_on_non_json_final_answer(repo_root):
    tools = _real_tools(repo_root)
    router = _WorkerScriptedRouter([_final_text("I looked around but found nothing conclusive.")])
    agent = WorkerAgent(router, tools)
    task = WorkerTask(worker_type=WorkerType.BUG_ANALYSIS, description="investigate a crash")

    result = agent.run(task)

    # Never raises; degrades to a low-confidence result carrying the raw text.
    assert result.success is True
    assert result.confidence == 0.2
    assert "found nothing conclusive" in result.findings
    assert result.worker_type == WorkerType.BUG_ANALYSIS


def test_worker_agent_run_empty_final_answer_degrades_without_crashing(repo_root):
    tools = _real_tools(repo_root)
    router = _WorkerScriptedRouter([_final_text("")])
    agent = WorkerAgent(router, tools)
    result = agent.run(WorkerTask(worker_type=WorkerType.GIT, description="check history"))

    assert result.success is True
    assert result.confidence == 0.0
    assert result.findings  # a placeholder string, never empty/crashing


def test_worker_agent_run_router_exception_returns_failed_result_not_raise(repo_root):
    tools = _real_tools(repo_root)

    class _RaisingRouter:
        def complete(self, *args, **kwargs):
            raise RuntimeError("provider unreachable")

    agent = WorkerAgent(_RaisingRouter(), tools)
    task = WorkerTask(worker_type=WorkerType.GIT, description="check recent commits", worker_id="w9")

    result = agent.run(task)  # must not raise

    assert result.success is False
    assert result.worker_id == "w9"
    assert result.worker_type == WorkerType.GIT
    assert result.task_description == "check recent commits"
    assert "provider unreachable" in result.error
    assert "RuntimeError" in result.error


def test_worker_agent_max_iterations_default_and_override(repo_root):
    tools = _real_tools(repo_root)
    default_agent = WorkerAgent(_WorkerScriptedRouter([]), tools)
    assert default_agent.max_iterations == 6

    custom_agent = WorkerAgent(_WorkerScriptedRouter([]), tools, max_iterations=2)
    assert custom_agent.max_iterations == 2


def test_scoped_registry_rejects_hallucinated_write_call(repo_root):
    """Not a model-call test -- directly exercises the exact access-control
    boundary `WorkerAgent.run()` relies on (`tools.subset(tool_names_for(...))`)."""
    tools = _real_tools(repo_root)
    scoped = tools.subset(tool_names_for(WorkerType.CODE_RESEARCH))

    assert "write_file" not in scoped.names()

    result = scoped.execute("write_file", {"path": "hack.py", "content": "evil"})

    assert result.success is False
    assert "no tool named" in (result.error or "").lower()
    assert not (repo_root / "hack.py").exists()


def test_scoped_registry_rejects_run_command_for_every_worker_type(repo_root):
    tools = _real_tools(repo_root)
    for worker_type in WorkerType:
        scoped = tools.subset(tool_names_for(worker_type))
        result = scoped.execute("run_command", {"command": "echo hi"})
        assert result.success is False, f"{worker_type} unexpectedly allowed run_command"


# ===========================================================================
# 4. WorkerManager.run_parallel
# ===========================================================================


class _TimingWorkerRouter:
    """Every call records a `time.monotonic()` timestamp (lock-protected --
    called concurrently from multiple worker threads), sleeps
    `sleep_seconds`, then returns a canned final-text JSON answer with no
    tool calls. Used to prove real OS-thread concurrency."""

    def __init__(self, sleep_seconds: float = 0.3, response_content: str | None = None):
        self._lock = threading.Lock()
        self.start_times: list[float] = []
        self.sleep_seconds = sleep_seconds
        self.response_content = response_content or _WORKER_FINDINGS_JSON

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        with self._lock:
            self.start_times.append(time.monotonic())
        time.sleep(self.sleep_seconds)
        return _final_text(self.response_content)


def _make_emit_recorder():
    events: list[dict] = []

    def emit(stage, message, event_type="thinking", data=None):
        events.append({"stage": stage, "message": message, "event_type": event_type, "data": data or {}})

    return emit, events


def test_run_parallel_achieves_real_concurrency_not_serial_execution(repo_root):
    tools = _real_tools(repo_root)
    sleep_seconds = 0.35
    n_tasks = 4
    router = _TimingWorkerRouter(sleep_seconds=sleep_seconds)
    config = WorkersConfig(enabled=True, max_parallel=5, timeout_seconds=30)
    manager = WorkerManager(router, tools, config)
    tasks = [WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description=f"task {i}") for i in range(n_tasks)]

    start = time.monotonic()
    results = manager.run_parallel(tasks)
    elapsed = time.monotonic() - start

    assert len(results) == n_tasks
    assert all(r.success for r in results)

    # All workers started at ~the same time (near-zero spread), not one
    # after another (which would spread them ~sleep_seconds apart each).
    spread = max(router.start_times) - min(router.start_times)
    assert spread < sleep_seconds, f"workers did not start concurrently: spread={spread}"

    # Total wall-clock is close to 1x sleep_seconds, nowhere near
    # n_tasks * sleep_seconds (what fully serial execution would take).
    assert elapsed < n_tasks * sleep_seconds * 0.75, f"run_parallel took too long: {elapsed}s"


def test_run_parallel_truncates_to_max_parallel(repo_root):
    tools = _real_tools(repo_root)
    router = _TimingWorkerRouter(sleep_seconds=0.02)
    config = WorkersConfig(enabled=True, max_parallel=2, timeout_seconds=30)
    manager = WorkerManager(router, tools, config)
    tasks = [WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description=f"task {i}") for i in range(5)]

    results = manager.run_parallel(tasks)

    assert len(results) == 2
    assert len(router.start_times) == 2  # the other 3 were never even attempted


def test_run_parallel_disabled_fails_closed_without_executing_anything(repo_root):
    tools = _real_tools(repo_root)
    router = _TimingWorkerRouter(sleep_seconds=0.02)
    config = WorkersConfig(enabled=False)
    manager = WorkerManager(router, tools, config)
    tasks = [
        WorkerTask(worker_type=WorkerType.GIT, description="check history"),
        WorkerTask(worker_type=WorkerType.TEST, description="find tests"),
    ]

    results = manager.run_parallel(tasks)

    assert len(results) == 2
    assert all(r.success is False for r in results)
    assert all("disabled" in (r.error or "").lower() for r in results)
    assert router.start_times == []  # no worker ever actually ran


def test_run_parallel_empty_task_list_returns_empty(repo_root):
    tools = _real_tools(repo_root)
    manager = WorkerManager(_TimingWorkerRouter(), tools, WorkersConfig())
    assert manager.run_parallel([]) == []


class _SlowRouter:
    """Always sleeps `sleep_seconds` before returning -- used to simulate a
    worker that outlives its configured timeout."""

    def __init__(self, sleep_seconds: float):
        self.sleep_seconds = sleep_seconds

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        time.sleep(self.sleep_seconds)
        return _final_text(_WORKER_FINDINGS_JSON)


def test_run_parallel_timeout_returns_promptly_not_after_the_full_slow_duration(repo_root):
    tools = _real_tools(repo_root)
    # WorkersConfig.timeout_seconds is an int and the manager does
    # max(1, int(...)) -- the smallest real timeout configurable is 1s, so
    # the worker must sleep clearly longer than that.
    router = _SlowRouter(sleep_seconds=2.0)
    config = WorkersConfig(enabled=True, max_parallel=5, timeout_seconds=1)
    manager = WorkerManager(router, tools, config)
    tasks = [WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description="a very slow investigation")]

    start = time.monotonic()
    results = manager.run_parallel(tasks)
    elapsed = time.monotonic() - start

    # Must return well before the 2s worker call would actually finish --
    # proving run_parallel does NOT block on pool.shutdown(wait=True).
    assert elapsed < 2.0, f"run_parallel blocked for the full slow-worker duration: {elapsed}s"
    assert len(results) == 1
    assert results[0].success is False
    assert results[0].error == "timed out after 1s"


class _MixedRouter:
    """Thread-safe: a call whose messages mention 'CRASH_ME' raises; every
    other call returns a canned success response."""

    def __init__(self):
        self._lock = threading.Lock()
        self.call_count = 0

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        with self._lock:
            self.call_count += 1
        content = json.dumps(messages)
        if "CRASH_ME" in content:
            raise RuntimeError("boom")
        return _final_text(_WORKER_FINDINGS_JSON)


def test_run_parallel_one_worker_crashing_does_not_lose_or_corrupt_others(repo_root):
    tools = _real_tools(repo_root)
    router = _MixedRouter()
    config = WorkersConfig(enabled=True, max_parallel=5, timeout_seconds=30)
    manager = WorkerManager(router, tools, config)
    tasks = [
        WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description="task A"),
        WorkerTask(worker_type=WorkerType.BUG_ANALYSIS, description="CRASH_ME please"),
        WorkerTask(worker_type=WorkerType.GIT, description="task C"),
    ]

    results = manager.run_parallel(tasks)

    assert len(results) == 3
    # Order matches TASKS order, not completion order.
    assert results[0].worker_type == WorkerType.CODE_RESEARCH
    assert results[0].success is True
    assert results[1].worker_type == WorkerType.BUG_ANALYSIS
    assert results[1].success is False
    assert "boom" in results[1].error
    assert results[2].worker_type == WorkerType.GIT
    assert results[2].success is True
    assert router.call_count == 3


def test_run_parallel_returns_results_in_task_order_not_completion_order(repo_root):
    tools = _real_tools(repo_root)

    class _VariableSleepRouter:
        def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
            content = json.dumps(messages)
            if "SLOW_ONE" in content:
                time.sleep(0.3)
                findings = "slow-finished"
            else:
                findings = "fast-finished"
            return _final_text(json.dumps({"findings": findings, "confidence": 0.5, "files": [], "recommendations": []}))

    router = _VariableSleepRouter()
    config = WorkersConfig(enabled=True, max_parallel=5, timeout_seconds=30)
    manager = WorkerManager(router, tools, config)
    tasks = [
        WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description="SLOW_ONE task, submitted first"),
        WorkerTask(worker_type=WorkerType.GIT, description="fast task, submitted second"),
    ]

    results = manager.run_parallel(tasks)

    # Despite the fast task finishing first in wall-clock time, results
    # preserve the ORIGINAL tasks order.
    assert results[0].findings == "slow-finished"
    assert results[1].findings == "fast-finished"


def test_run_parallel_emits_starting_per_worker_and_completed_events(repo_root):
    tools = _real_tools(repo_root)
    router = _TimingWorkerRouter(sleep_seconds=0.02)
    config = WorkersConfig(enabled=True, max_parallel=5, timeout_seconds=30)
    emit, events = _make_emit_recorder()
    manager = WorkerManager(router, tools, config, emit=emit)
    tasks = [
        WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description="alpha investigation"),
        WorkerTask(worker_type=WorkerType.GIT, description="beta investigation"),
    ]

    manager.run_parallel(tasks)

    assert events, "expected at least one emitted event"
    assert all(e["stage"] == "delegate" for e in events)

    starting = [e for e in events if e["message"].startswith("Starting")]
    assert len(starting) == 1
    assert "2 worker(s)" in starting[0]["message"]
    assert "alpha investigation" in starting[0]["message"]
    assert "beta investigation" in starting[0]["message"]

    completed = [e for e in events if e["message"] == "Completed."]
    assert len(completed) == 1

    per_worker_results = [e for e in events if e is not starting[0] and e is not completed[0]]
    assert len(per_worker_results) == 2


def test_run_parallel_disabled_still_emits_no_worker_start_events(repo_root):
    tools = _real_tools(repo_root)
    router = _TimingWorkerRouter()
    config = WorkersConfig(enabled=False)
    emit, events = _make_emit_recorder()
    manager = WorkerManager(router, tools, config, emit=emit)

    manager.run_parallel([WorkerTask(worker_type=WorkerType.GIT, description="x")])

    # enabled=False short-circuits before the "Starting N worker(s)" emit.
    assert not any(e["message"].startswith("Starting") for e in events)


# ===========================================================================
# 5. WorkerResultMerger
# ===========================================================================


def test_merge_empty_list_returns_nonempty_string():
    text = WorkerResultMerger.merge([])
    assert isinstance(text, str) and text


def test_merge_all_success_renders_each_worker():
    results = [
        WorkerResult(
            worker_type=WorkerType.CODE_RESEARCH,
            task_description="t1",
            success=True,
            findings="found the root cause in X",
            confidence=0.7,
            files=["a.py"],
            recommendations=["do Y"],
        ),
        WorkerResult(
            worker_type=WorkerType.GIT,
            task_description="t2",
            success=True,
            findings="commit abc123 is related",
            confidence=0.4,
            files=["b.py"],
            recommendations=[],
        ),
    ]
    text = WorkerResultMerger.merge(results, goal="fix the bug")

    assert "fix the bug" in text
    assert "found the root cause in X" in text
    assert "commit abc123 is related" in text
    assert "a.py" in text and "b.py" in text
    assert "do Y" in text
    assert "FAILED" not in text


def test_merge_some_failed_rendered_distinctly_not_dropped():
    results = [
        WorkerResult(worker_type=WorkerType.CODE_RESEARCH, task_description="t1", success=True, findings="ok", confidence=0.5),
        WorkerResult(worker_type=WorkerType.TEST, task_description="t2", success=False, error="timed out after 5s"),
    ]
    text = WorkerResultMerger.merge(results)

    assert "FAILED" in text
    assert "timed out after 5s" in text
    assert "1/2 worker(s) failed" in text


def test_merge_failed_worker_with_no_error_detail_still_renders():
    results = [WorkerResult(worker_type=WorkerType.GIT, success=False, error=None)]
    text = WorkerResultMerger.merge(results)
    assert "FAILED" in text
    assert "no error detail" in text


def test_collect_files_dedups_across_workers_and_skips_failed():
    results = [
        WorkerResult(worker_type=WorkerType.CODE_RESEARCH, success=True, files=["a.py", "b.py"]),
        WorkerResult(worker_type=WorkerType.GIT, success=True, files=["b.py", "c.py"]),
        WorkerResult(worker_type=WorkerType.TEST, success=False, files=["d.py"]),
    ]
    assert WorkerResultMerger.collect_files(results) == ["a.py", "b.py", "c.py"]


def test_collect_recommendations_dedups_across_workers_and_skips_failed():
    results = [
        WorkerResult(worker_type=WorkerType.CODE_RESEARCH, success=True, recommendations=["do X", "do Y"]),
        WorkerResult(worker_type=WorkerType.GIT, success=True, recommendations=["do Y", "do Z"]),
        WorkerResult(worker_type=WorkerType.TEST, success=False, recommendations=["do W"]),
    ]
    assert WorkerResultMerger.collect_recommendations(results) == ["do X", "do Y", "do Z"]


def test_collect_files_and_recommendations_empty_for_all_failed():
    results = [WorkerResult(worker_type=WorkerType.GIT, success=False, files=["x.py"], recommendations=["y"])]
    assert WorkerResultMerger.collect_files(results) == []
    assert WorkerResultMerger.collect_recommendations(results) == []


# ===========================================================================
# 6. Specialized worker classes + build_worker factory
# ===========================================================================


@pytest.mark.parametrize(
    "worker_cls,expected_type",
    [
        (CodeResearchWorker, WorkerType.CODE_RESEARCH),
        (BugAnalysisWorker, WorkerType.BUG_ANALYSIS),
        (_TestWorker, WorkerType.TEST),
        (DocumentationWorker, WorkerType.DOCUMENTATION),
        (GitWorker, WorkerType.GIT),
    ],
)
def test_each_specialized_worker_class_has_correct_worker_type(repo_root, worker_cls, expected_type):
    tools = _real_tools(repo_root)
    worker = worker_cls(router=object(), tools=tools)
    assert worker.worker_type == expected_type
    assert isinstance(worker, WorkerAgent)


@pytest.mark.parametrize("worker_type,worker_cls", list(WORKER_CLASSES.items()))
def test_build_worker_factory_constructs_the_right_class(repo_root, worker_type, worker_cls):
    tools = _real_tools(repo_root)
    worker = build_worker(worker_type, router=object(), tools=tools)
    assert isinstance(worker, worker_cls)
    assert worker.worker_type == worker_type


def test_build_worker_factory_covers_every_worker_type():
    assert set(WORKER_CLASSES.keys()) == set(WorkerType)


def test_specialized_worker_run_task_is_a_working_convenience_wrapper(repo_root):
    tools = _real_tools(repo_root)
    router = _WorkerScriptedRouter(
        [_final_text(json.dumps({"findings": "recent history is clean", "confidence": 0.9, "files": [], "recommendations": []}))]
    )
    worker = GitWorker(router, tools)

    result = worker.run_task("Check recent commits", context="look at the last 5 commits")

    assert result.success is True
    assert result.worker_type == WorkerType.GIT
    assert result.task_description == "Check recent commits"
    assert result.confidence == 0.9

    # The context string actually reached the model's prompt.
    sent_messages = router.calls[0]["messages"]
    user_content = next(m["content"] for m in sent_messages if m["role"] == "user")
    assert "look at the last 5 commits" in user_content


def test_specialized_worker_make_task_builds_correct_worker_task(repo_root):
    tools = _real_tools(repo_root)
    worker = _TestWorker(router=object(), tools=tools)
    task = worker.make_task("find failing tests", context="ctx", worker_id="w5")
    assert task.worker_type == WorkerType.TEST
    assert task.description == "find failing tests"
    assert task.context == "ctx"
    assert task.worker_id == "w5"


# ===========================================================================
# 7. WorkersConfig defaults + KenzoConfig wiring
# ===========================================================================


def test_workers_config_defaults():
    config = WorkersConfig()
    assert config.enabled is True
    assert config.max_parallel == 5
    assert config.timeout_seconds == 120


def test_workers_config_overrides():
    config = WorkersConfig(enabled=False, max_parallel=1, timeout_seconds=5)
    assert config.enabled is False
    assert config.max_parallel == 1
    assert config.timeout_seconds == 5


def test_kenzo_config_has_workers_field_with_correct_defaults():
    config = KenzoConfig()
    assert isinstance(config.workers, WorkersConfig)
    assert config.workers.enabled is True
    assert config.workers.max_parallel == 5
    assert config.workers.timeout_seconds == 120


# ===========================================================================
# 8. AgentController DELEGATE wiring (unit-level)
# ===========================================================================


def test_controller_action_delegate_member_exists():
    assert ControllerAction.DELEGATE.value == "delegate"
    assert ControllerAction("delegate") is ControllerAction.DELEGATE


def test_action_decision_round_trips_delegate_action():
    decision = ActionDecision(action=ControllerAction.DELEGATE, rationale="split investigation", target="logout")
    restored = ActionDecision.model_validate(decision.model_dump())
    assert restored.action == ControllerAction.DELEGATE
    assert restored.rationale == "split investigation"
    assert restored.target == "logout"


def test_action_tools_delegate_is_empty_the_main_agent_calls_no_tools_directly():
    assert ACTION_TOOLS[ControllerAction.DELEGATE] == []


def _controller_for_delegate_unit_tests(repo_root, router):
    tools = _real_tools(repo_root)
    memory = RepositoryMemory(str(repo_root))
    return AgentController(router, tools, memory, KenzoConfig())


def test_propose_worker_tasks_parses_a_valid_response(repo_root):
    tasks_json = json.dumps(
        {
            "tasks": [
                {"worker_type": "code_research", "description": "Explore the controller"},
                {"worker_type": "test", "description": "Find relevant tests"},
            ]
        }
    )

    class _PlanningRouter:
        def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
            assert task_type == TaskType.PLANNING
            return tasks_json

    controller = _controller_for_delegate_unit_tests(repo_root, _PlanningRouter())
    decision = ActionDecision(action=ControllerAction.DELEGATE, rationale="need parallel investigation", target="logout")
    task_state = TaskState(task_id="t1", goal="fix logout")

    tasks = controller._propose_worker_tasks(decision, task_state)

    assert len(tasks) == 2
    assert tasks[0].worker_type == WorkerType.CODE_RESEARCH
    assert tasks[0].description == "Explore the controller"
    assert tasks[1].worker_type == WorkerType.TEST
    assert tasks[1].description == "Find relevant tests"


def test_propose_worker_tasks_ignores_malformed_entries_but_keeps_valid_ones(repo_root):
    tasks_json = json.dumps(
        {
            "tasks": [
                {"worker_type": "not_a_real_type", "description": "should be skipped"},
                "not even a dict",
                {"worker_type": "git", "description": ""},  # empty description -- skipped
                {"worker_type": "git", "description": "check history"},
            ]
        }
    )

    class _PlanningRouter:
        def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
            return tasks_json

    controller = _controller_for_delegate_unit_tests(repo_root, _PlanningRouter())
    decision = ActionDecision(action=ControllerAction.DELEGATE, rationale="investigate")
    task_state = TaskState(task_id="t1", goal="fix bug")

    tasks = controller._propose_worker_tasks(decision, task_state)

    assert len(tasks) == 1
    assert tasks[0].worker_type == WorkerType.GIT
    assert tasks[0].description == "check history"


def test_propose_worker_tasks_falls_back_to_one_code_research_task_on_unparseable_response(repo_root):
    class _GarbageRouter:
        def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
            return "not json at all, sorry"

    controller = _controller_for_delegate_unit_tests(repo_root, _GarbageRouter())
    decision = ActionDecision(action=ControllerAction.DELEGATE, rationale="investigate", target="logout bug")
    task_state = TaskState(task_id="t1", goal="fix logout bug")

    tasks = controller._propose_worker_tasks(decision, task_state)

    assert len(tasks) == 1
    assert tasks[0].worker_type == WorkerType.CODE_RESEARCH
    assert tasks[0].description == "logout bug"  # decision.target preferred over goal


def test_propose_worker_tasks_falls_back_when_tasks_list_is_empty(repo_root):
    class _EmptyTasksRouter:
        def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
            return json.dumps({"tasks": []})

    controller = _controller_for_delegate_unit_tests(repo_root, _EmptyTasksRouter())
    decision = ActionDecision(action=ControllerAction.DELEGATE, rationale="investigate", target="")
    task_state = TaskState(task_id="t1", goal="fix the widget")

    tasks = controller._propose_worker_tasks(decision, task_state)

    assert len(tasks) == 1
    assert tasks[0].worker_type == WorkerType.CODE_RESEARCH
    assert tasks[0].description == "fix the widget"  # falls back to goal when no target


def test_propose_worker_tasks_never_zero_even_when_router_raises(repo_root):
    class _RaisingRouter:
        def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
            raise RuntimeError("network down")

    controller = _controller_for_delegate_unit_tests(repo_root, _RaisingRouter())
    decision = ActionDecision(action=ControllerAction.DELEGATE, rationale="investigate", target="widget.py")
    task_state = TaskState(task_id="t1", goal="fix the widget")

    tasks = controller._propose_worker_tasks(decision, task_state)  # must not raise

    assert len(tasks) == 1
    assert tasks[0].worker_type == WorkerType.CODE_RESEARCH


def test_execute_delegate_tool_transcript_shape_and_apply_updates_merging(repo_root):
    tasks_json = json.dumps(
        {
            "tasks": [
                {"worker_type": "code_research", "description": "Explore module A"},
                {"worker_type": "git", "description": "Check recent history"},
            ]
        }
    )
    success_json = json.dumps(
        {
            "findings": "module A does X",
            "confidence": 0.8,
            "files": ["a.py"],
            "recommendations": ["refactor a.py"],
        }
    )

    class _DelegateRouter:
        def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
            return tasks_json

        def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
            content = json.dumps(messages)
            if "Check recent history" in content:
                raise RuntimeError("git worker exploded")
            return _final_text(success_json)

    controller = _controller_for_delegate_unit_tests(repo_root, _DelegateRouter())
    decision = ActionDecision(action=ControllerAction.DELEGATE, rationale="investigate two angles", target="module A")
    task_state = TaskState(task_id="t1", goal="fix module A")

    final_text, tool_transcript = controller._execute_delegate(decision, task_state)

    assert len(tool_transcript) == 2
    for entry in tool_transcript:
        assert entry["tool"] == "delegate_worker"
        assert set(entry["arguments"]) == {"worker_type", "target", "task"}
        assert set(entry["result"]) == {"success", "output", "error", "data"}
        assert set(entry["result"]["data"]) == {"files", "confidence", "recommendations"}

    success_entry = next(e for e in tool_transcript if e["result"]["success"])
    failed_entry = next(e for e in tool_transcript if not e["result"]["success"])
    assert success_entry["arguments"]["worker_type"] == "code_research"
    assert success_entry["result"]["data"]["files"] == ["a.py"]
    assert failed_entry["arguments"]["worker_type"] == "git"
    assert "git worker exploded" in failed_entry["result"]["error"]

    assert "module A does X" in final_text

    # _apply_updates is the EXISTING, unmodified generic tool_transcript loop
    # -- proving delegate_worker slots into it with zero special-casing
    # beyond the one new branch.
    controller._apply_updates(task_state, decision, final_text, tool_transcript)

    assert "a.py" in task_state.files_seen
    failed_attempt = next(fa for fa in task_state.failed_attempts if fa.action == "delegate")
    assert "Check recent history" in failed_attempt.target
    assert "git worker exploded" in failed_attempt.error


# ===========================================================================
# 9. Full integration scenario: "OIDC logout broken" (real AgentController,
#    real tools, real repo -- only the router faked)
# ===========================================================================

_LOGOUT_CONTROLLER_PY = """\
from auth_service import AuthService


class LogoutController:
    def __init__(self):
        self.auth_service = AuthService()

    def logout(self, session_id):
        self.auth_service.clear_local_session(session_id)
        return "logged out"
"""

_AUTH_SERVICE_PY_BEFORE = """\
class AuthService:
    def __init__(self):
        self.invalidated_sessions = []

    def clear_local_session(self, session_id):
        pass

    def invalidate_session(self, session_id):
        self.invalidated_sessions.append(session_id)
"""

_TEST_SESSION_PY = """\
from logout_controller import LogoutController


def test_logout_returns_confirmation():
    controller = LogoutController()
    result = controller.logout("session-123")
    assert result == "logged out"
"""

_OIDC_CONFIG_PY = """\
OIDC_LOGOUT_REDIRECT = "/login"
"""

_DELEGATE_GOAL = "Fix OIDC logout so it actually invalidates the session server-side"

_DELEGATE_TASKS_JSON = json.dumps(
    {
        "tasks": [
            {
                "worker_type": "code_research",
                "description": "Explore logout",
            },
            {
                "worker_type": "test",
                "description": "Find tests related to logout/session behavior and note what they verify.",
            },
            {
                "worker_type": "git",
                "description": "Check recent commits/history for session config changes related to logout.",
            },
        ]
    }
)

# Distinctive marker placed at the START of the code_research worker's
# findings so it reliably survives `_apply_updates`'s 200-char truncation
# of the merged delegate summary when logging task_state.decisions (see the
# character-position calibration in this module's own test below).
_CANARY_PHRASE = "INVALSESSFLAG"

_CODE_RESEARCH_RESULT_JSON = json.dumps(
    {
        "findings": (
            f"{_CANARY_PHRASE}: LogoutController.logout() calls AuthService.clear_local_session(), "
            "which currently does nothing -- it never calls invalidate_session(), so the session is "
            "never actually invalidated server-side."
        ),
        "confidence": 0.85,
        "files": ["logout_controller.py", "auth_service.py"],
        "recommendations": ["Make clear_local_session call invalidate_session so the session is really invalidated."],
    }
)

_TEST_WORKER_RESULT_JSON = json.dumps(
    {
        "findings": "test_session.py exercises LogoutController.logout() but never asserts server-side invalidation.",
        "confidence": 0.7,
        "files": ["test_session.py"],
        "recommendations": ["Add an assertion that invalidate_session was actually called."],
    }
)

_GIT_WORKER_RESULT_JSON = json.dumps(
    {
        "findings": "No recent commit specifically touches session invalidation -- this looks like a pre-existing gap.",
        "confidence": 0.5,
        "files": [],
        "recommendations": [],
    }
)

_OLD_CLEAR_LOCAL_SESSION = "    def clear_local_session(self, session_id):\n        pass\n"
_NEW_CLEAR_LOCAL_SESSION = "    def clear_local_session(self, session_id):\n        self.invalidate_session(session_id)\n"


class _DelegateScenarioRouter:
    """Fake router for the full DELEGATE integration scenario. Handles
    THREE distinct call styles safely and thread-safely (see module
    docstring and the Phase 9 task brief's threading gotcha):

    (a) `tool_choice` names decide_next_action -> popped off a strict,
        main-thread-only ordered queue.
    (b) `complete_text(...)` (no `tools` argument at all) -> the single
        `_propose_worker_tasks` planning call.
    (c) `complete(task_type=..., tools=[...])` that is NOT a decide call ->
        either the EDIT/TEST action's own `run_tool_loop` (main thread,
        identified by `task_type` -- EDIT uses CODING, TEST uses DEBUGGING,
        neither of which any worker in this scenario uses) or one of the 3
        concurrent workers' `run_tool_loop` calls (background threads, all
        using `TaskType.RESEARCH` -- disambiguated by matching a unique
        substring of each worker's own task description, which is safe
        because each worker thread's `messages` list is a private, freshly
        built list -- see kenzo.core.tool_loop.run_tool_loop).
    """

    def __init__(self, decisions):
        self._lock = threading.Lock()
        self._decisions = list(decisions)
        self._call_counts: dict[str, int] = {}
        self.decision_calls: list[list[dict]] = []
        self.planning_calls = 0

    def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
        assert task_type == TaskType.PLANNING
        with self._lock:
            self.planning_calls += 1
        return _DELEGATE_TASKS_JSON

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        snapshot = [dict(m) for m in messages]

        if _is_decide_call(tool_choice):
            with self._lock:
                self.decision_calls.append(snapshot)
                if not self._decisions:
                    raise AssertionError("ran out of scripted decisions")
                return self._decisions.pop(0)

        content = json.dumps(snapshot)

        if task_type == TaskType.CODING:
            key = "edit"
        elif task_type == TaskType.DEBUGGING:
            key = "test"
        elif task_type == TaskType.RESEARCH:
            if "Explore logout" in content:
                key = "worker-code"
            elif "Find tests related" in content:
                key = "worker-test"
            elif "recent commits" in content:
                key = "worker-git"
            else:
                raise AssertionError(f"unrecognized RESEARCH-task call content: {content[:400]}")
        else:
            raise AssertionError(f"unexpected task_type for a non-decide call: {task_type}")

        with self._lock:
            count = self._call_counts.get(key, 0)
            self._call_counts[key] = count + 1

        return self._response_for(key, count)

    @staticmethod
    def _response_for(key: str, count: int):
        if key == "edit":
            if count == 0:
                return _tool_call_response(
                    "call-edit-fix",
                    "edit_file",
                    {"path": "auth_service.py", "old_text": _OLD_CLEAR_LOCAL_SESSION, "new_text": _NEW_CLEAR_LOCAL_SESSION},
                )
            return _final_text("Fixed AuthService.clear_local_session to actually invalidate the session.")
        if key == "test":
            return _final_text(
                json.dumps(
                    {
                        "passed": True,
                        "summary": "logout now invalidates the session server-side",
                        "command": "",
                        "raw_output": "",
                    }
                )
            )
        if key == "worker-code":
            return _final_text(_CODE_RESEARCH_RESULT_JSON)
        if key == "worker-test":
            return _final_text(_TEST_WORKER_RESULT_JSON)
        if key == "worker-git":
            return _final_text(_GIT_WORKER_RESULT_JSON)
        raise AssertionError(f"no scripted response for key={key!r} count={count}")


def test_canary_phrase_survives_the_200_char_decision_log_truncation():
    """Calibration check: proves the merged delegate summary this scenario
    produces really does carry `_CANARY_PHRASE` within the first 200 chars
    -- i.e. within what `AgentController._apply_updates` keeps of
    `final_text` in `task_state.decisions`' log line -- BEFORE relying on
    that fact in the full scenario test below."""
    from kenzo.workers.models import WorkerResult as _WR

    results = [
        _WR(worker_id="w1", worker_type=WorkerType.CODE_RESEARCH, task_description="Explore logout", success=True, **json.loads(_CODE_RESEARCH_RESULT_JSON)),
        _WR(worker_id="w2", worker_type=WorkerType.TEST, task_description="Find tests", success=True, **json.loads(_TEST_WORKER_RESULT_JSON)),
        _WR(worker_id="w3", worker_type=WorkerType.GIT, task_description="Check history", success=True, **json.loads(_GIT_WORKER_RESULT_JSON)),
    ]
    merged = WorkerResultMerger.merge(results, goal=_DELEGATE_GOAL)
    assert _CANARY_PHRASE in merged[:200]


def test_full_scenario_delegate_investigates_then_controller_fixes_and_verifies(repo_root):
    # -- seed a small fake repo: a LogoutController that calls an
    # AuthService that never actually invalidates the session, a test that
    # doesn't yet catch the gap, and a config file for GIT-worker flavor.
    # newline="" avoids Windows' platform newline translation (\n -> \r\n)
    # on write -- edit_file's old_text/new_text below embed literal "\n"s
    # and must match the file's actual on-disk bytes exactly (see
    # kenzo.tools.filesystem.write_file's own docstring on exactly this).
    (repo_root / "logout_controller.py").write_text(_LOGOUT_CONTROLLER_PY, encoding="utf-8", newline="")
    (repo_root / "auth_service.py").write_text(_AUTH_SERVICE_PY_BEFORE, encoding="utf-8", newline="")
    (repo_root / "test_session.py").write_text(_TEST_SESSION_PY, encoding="utf-8", newline="")
    (repo_root / "oidc_config.py").write_text(_OIDC_CONFIG_PY, encoding="utf-8", newline="")
    run_git(["add", "-A"], cwd=repo_root)
    run_git(["commit", "-m", "Add logout controller/auth service/session test"], cwd=repo_root)

    tools = _real_tools(repo_root)
    memory = RepositoryMemory(str(repo_root))

    decisions = [
        _decision(
            ControllerAction.DELEGATE,
            "Multiple plausible causes -- the controller, the auth service, and whether a recent "
            "commit changed logout behavior -- investigate independently in parallel.",
            "logout",
        ),
        _decision(
            ControllerAction.EDIT,
            "Delegated findings show clear_local_session never calls invalidate_session -- fix that.",
            "auth_service.py",
        ),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(ControllerAction.FINISH, "Logout now actually invalidates the session, verified by a passing test."),
    ]

    router = _DelegateScenarioRouter(decisions)
    controller = AgentController(router, tools, memory, KenzoConfig())

    result = controller.run(_DELEGATE_GOAL)

    # -- the real fix landed on disk.
    fixed_content = (repo_root / "auth_service.py").read_text(encoding="utf-8")
    assert "self.invalidate_session(session_id)" in fixed_content
    assert _OLD_CLEAR_LOCAL_SESSION not in fixed_content

    assert result["success"] is True
    assert result["finished"] is True
    assert "auth_service.py" in result["files_modified"]

    task_state = result["task_state"]

    # -- the delegate step is logged, and worker-discovered files reached
    # files_seen BEFORE the edit step (which is later in the same list).
    delegate_logged = any(d.startswith("[iter 0] delegate:") for d in task_state["decisions"])
    assert delegate_logged, task_state["decisions"]
    assert "logout_controller.py" in task_state["files_seen"]
    assert "test_session.py" in task_state["files_seen"]

    # -- stronger claim: the DELEGATE step's merged findings actually
    # reached a LATER decision prompt (the EDIT decision's own call),
    # proving delegation feeds back into the main agent's own reasoning,
    # not just into the final report.
    assert len(router.decision_calls) >= 2
    edit_decision_prompt = json.dumps(router.decision_calls[1])
    assert _CANARY_PHRASE in edit_decision_prompt

    # -- all 3 workers really ran concurrently (real ThreadPoolExecutor).
    assert router.planning_calls == 1
    assert {"worker-code", "worker-test", "worker-git"} <= set(router._call_counts.keys())

    # -- the merged findings and per-worker detail reached the FINAL report
    # too (not just an intermediate prompt).
    assert result["final_response"] is not None
