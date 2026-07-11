"""Tests for `kenzo.core.graph`: per-role tool scoping (`ROLE_TOOLS`), the
review/retry conditional edge (`_review_router`), and the fully compiled
planner -> researcher -> coder -> tester -> reviewer LangGraph pipeline
built by `build_graph`.

Only the model/router layer is faked (a `_StageScriptedRouter` keyed by
`TaskType`, scripting responses shaped like real litellm `ModelResponse`
objects -- see test_router.py's pattern, extended with tool_calls support).
Tools and memory are the REAL `kenzo.tools.registry.build_default_registry`
/ `kenzo.memory.repository.RepositoryMemory` running against the real tmp
git repo from the `repo_root` fixture, so the coder's `edit_file` calls
really touch disk and produce real diffs.

This file proves the tool-scoping hardening fix (`ToolRegistry.subset` via
`ROLE_TOOLS`): Planner/Researcher/Reviewer cannot reach `write_file`/
`edit_file`, Coder cannot reach `run_command`, and Tester cannot reach
`write_file`/`edit_file`. It also exercises the full pipeline end-to-end,
including the reviewer's retry loop and its termination at
`MAX_REVIEW_RETRIES`.
"""

from __future__ import annotations

import json
from collections import defaultdict

from kenzo.config.settings import PermissionsConfig
from kenzo.core.graph import ROLE_TOOLS, _review_router, build_graph
from kenzo.core.schemas import MAX_REVIEW_RETRIES, TaskType, new_state
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.registry import build_default_registry


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding (see test_router.py's pattern).
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeModelResponse:
    """Minimal stand-in shaped like a real litellm ModelResponse."""

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


class _StageScriptedRouter:
    """Fake router keyed by TaskType: each pipeline stage pops its next
    scripted response off its own queue, independent of the other stages."""

    def __init__(self, scripts: dict):
        self._scripts = {task_type: list(responses) for task_type, responses in scripts.items()}
        self.call_counts: dict = defaultdict(int)

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        self.call_counts[task_type] += 1
        queue = self._scripts.get(task_type)
        if not queue:
            raise AssertionError(f"_StageScriptedRouter ran out of responses for {task_type}")
        return queue.pop(0)


def _plan_response(goal="Fix simple bug"):
    return _FakeModelResponse(
        json.dumps(
            {
                "goal": goal,
                "steps": [{"id": 1, "description": "fix main.py", "rationale": "bug"}],
                "context_notes": "",
            }
        )
    )


def _research_response():
    return _FakeModelResponse(
        json.dumps(
            {
                "summary": "main.py returns the wrong value",
                "relevant_files": ["main.py"],
                "snippets": {},
                "open_questions": [],
            }
        )
    )


def _coder_edit_responses(old_text: str, new_text: str, call_id: str):
    edit_call = _FakeToolCall(
        call_id, "edit_file", json.dumps({"path": "main.py", "old_text": old_text, "new_text": new_text})
    )
    return [
        _FakeModelResponse(None, tool_calls=[edit_call]),
        _FakeModelResponse(f"Changed main.py: {old_text!r} -> {new_text!r}."),
    ]


def _tester_pass_response():
    return _FakeModelResponse(
        json.dumps({"passed": True, "summary": "tests pass", "command": "", "raw_output": ""})
    )


def _reviewer_response(approved: bool, issues=None, summary="reviewed"):
    return _FakeModelResponse(
        json.dumps({"approved": approved, "summary": summary, "issues": issues or []})
    )


# ---------------------------------------------------------------------------
# (a) ROLE_TOOLS scoping: each role only gets the tools its job requires
# ---------------------------------------------------------------------------


def test_role_tools_scoping_matches_expected_permissions(repo_root):
    permissions = PermissionsConfig()
    full_registry = build_default_registry(str(repo_root), permissions)

    # Phase 2 adds repository_tree/find_symbol/map_flow to the read-only
    # exploration roles, and gives Coder back search_files/repository_tree
    # (Phase 1.5 had left Coder with no search capability at all). Phase 4
    # adds the AST-backed code intelligence tools (trace_flow, find_references,
    # find_callers, find_usages, explain_architecture) to the same roles --
    # find_related_tests goes to Tester only (see expected below).
    expected_read_only = {
        "read_file",
        "list_directory",
        "search_files",
        "repository_tree",
        "find_symbol",
        "map_flow",
        "trace_flow",
        "find_references",
        "find_callers",
        "find_usages",
        "explain_architecture",
        "status",
        "diff",
        "branch",
        "history",
    }

    planner_tools = full_registry.subset(ROLE_TOOLS["planner"])
    researcher_tools = full_registry.subset(ROLE_TOOLS["researcher"])
    reviewer_tools = full_registry.subset(ROLE_TOOLS["reviewer"])
    coder_tools = full_registry.subset(ROLE_TOOLS["coder"])
    tester_tools = full_registry.subset(ROLE_TOOLS["tester"])

    assert set(planner_tools.names()) == expected_read_only
    assert set(researcher_tools.names()) == expected_read_only
    assert set(reviewer_tools.names()) == expected_read_only

    # Planner/Researcher/Reviewer must never reach write/edit tools.
    for role_tools in (planner_tools, researcher_tools, reviewer_tools):
        assert "write_file" not in role_tools.names()
        assert "edit_file" not in role_tools.names()

    # Coder has write/edit/search/tree but no terminal access, and cannot
    # recover from a bad Research result blindly anymore.
    assert set(coder_tools.names()) == {
        "read_file",
        "list_directory",
        "search_files",
        "repository_tree",
        "write_file",
        "edit_file",
    }
    assert "run_command" not in coder_tools.names()

    # Tester can run commands and read files, and (Phase 4) find_related_tests
    # to target the right test file for a changed source file, but cannot
    # modify anything.
    assert set(tester_tools.names()) == {"read_file", "run_command", "find_related_tests"}
    assert "write_file" not in tester_tools.names()
    assert "edit_file" not in tester_tools.names()


# ---------------------------------------------------------------------------
# (b) Full graph happy path: "Fix simple bug"
# ---------------------------------------------------------------------------


def test_full_graph_happy_path_fix_simple_bug(repo_root):
    tools = build_default_registry(str(repo_root), PermissionsConfig())
    memory = RepositoryMemory(str(repo_root))

    scripts = {
        TaskType.PLANNING: [_plan_response()],
        TaskType.RESEARCH: [_research_response()],
        TaskType.CODING: _coder_edit_responses("return 42", "return 43", "call_1"),
        TaskType.DEBUGGING: [_tester_pass_response()],
        TaskType.REVIEW: [_reviewer_response(True)],
    }
    router = _StageScriptedRouter(scripts)

    graph = build_graph(router, tools, memory)
    final_state = graph.invoke(new_state("Fix simple bug", str(repo_root)), config={"recursion_limit": 25})

    # All five stages ran and populated their corresponding state keys.
    assert final_state["plan"] is not None
    assert final_state["research"] is not None
    assert len(final_state["changes"]) == 1
    change = final_state["changes"][0]
    assert change["file_path"] == "main.py"
    assert change["diff"]  # non-empty real diff
    assert final_state["test_result"]["passed"] is True
    assert final_state["review"]["approved"] is True
    assert final_state["final_response"] is not None

    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 43\n"


# ---------------------------------------------------------------------------
# (c) Reviewer rejects once, then approves: retry loop actually retries
# ---------------------------------------------------------------------------


def test_full_graph_retry_loop_on_reviewer_rejection_then_approval(repo_root):
    tools = build_default_registry(str(repo_root), PermissionsConfig())
    memory = RepositoryMemory(str(repo_root))

    coder_responses = _coder_edit_responses("return 42", "return 43", "call_1") + _coder_edit_responses(
        "return 43", "return 44", "call_2"
    )
    scripts = {
        TaskType.PLANNING: [_plan_response()],
        TaskType.RESEARCH: [_research_response()],
        TaskType.CODING: coder_responses,
        TaskType.DEBUGGING: [_tester_pass_response(), _tester_pass_response()],
        TaskType.REVIEW: [
            _reviewer_response(False, issues=["please also update the docstring"]),
            _reviewer_response(True),
        ],
    }
    router = _StageScriptedRouter(scripts)

    stage_starts: dict = defaultdict(int)

    def on_event(event):
        if event.message.endswith("starting"):
            stage_starts[event.stage] += 1

    graph = build_graph(router, tools, memory, on_event=on_event)
    final_state = graph.invoke(new_state("Fix simple bug", str(repo_root)), config={"recursion_limit": 25})

    assert final_state["review_retries"] == 1
    assert final_state["review"]["approved"] is True

    # planner/researcher run once; coder/tester/reviewer run twice (retry loop
    # only re-enters at the coder, it never goes back through planner/researcher).
    assert stage_starts["planner"] == 1
    assert stage_starts["researcher"] == 1
    assert stage_starts["coder"] == 2
    assert stage_starts["tester"] == 2
    assert stage_starts["reviewer"] == 2

    # Cross-check via the router's own call counts.
    assert router.call_counts[TaskType.CODING] == 4  # 2 passes x (tool call + final answer)
    assert router.call_counts[TaskType.REVIEW] == 2

    # Changes accumulated from BOTH coder passes, not just the last one.
    assert len(final_state["changes"]) == 2
    assert final_state["changes"][0]["file_path"] == "main.py"
    assert final_state["changes"][1]["file_path"] == "main.py"

    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 44\n"


# ---------------------------------------------------------------------------
# (d) Reviewer always rejects: the graph still terminates at MAX_REVIEW_RETRIES
# ---------------------------------------------------------------------------


def test_full_graph_terminates_when_reviewer_always_rejects(repo_root):
    tools = build_default_registry(str(repo_root), PermissionsConfig())
    memory = RepositoryMemory(str(repo_root))

    coder_responses = _coder_edit_responses("return 42", "return 43", "call_1") + _coder_edit_responses(
        "return 43", "return 44", "call_2"
    )
    scripts = {
        TaskType.PLANNING: [_plan_response()],
        TaskType.RESEARCH: [_research_response()],
        TaskType.CODING: coder_responses,
        TaskType.DEBUGGING: [_tester_pass_response(), _tester_pass_response()],
        TaskType.REVIEW: [
            _reviewer_response(False, issues=["issue 1"]),
            _reviewer_response(False, issues=["issue 2"]),
        ],
    }
    router = _StageScriptedRouter(scripts)

    graph = build_graph(router, tools, memory)
    # If the retry loop failed to terminate, this would exceed the
    # recursion_limit and raise (or exhaust the scripted router and raise) --
    # either way the test would fail loudly rather than hang.
    final_state = graph.invoke(new_state("Fix simple bug", str(repo_root)), config={"recursion_limit": 25})

    assert final_state["review_retries"] == MAX_REVIEW_RETRIES
    assert final_state["review"]["approved"] is False
    assert final_state["final_response"] is not None


# ---------------------------------------------------------------------------
# _review_router unit tests
# ---------------------------------------------------------------------------


def test_review_router_returns_retry_when_not_approved_and_under_max():
    state = {"review": {"approved": False}, "review_retries": 1}
    assert _review_router(state) == "retry"


def test_review_router_returns_done_when_approved():
    state = {"review": {"approved": True}, "review_retries": 0}
    assert _review_router(state) == "done"


def test_review_router_returns_done_when_max_retries_reached():
    state = {"review": {"approved": False}, "review_retries": MAX_REVIEW_RETRIES}
    assert _review_router(state) == "done"
