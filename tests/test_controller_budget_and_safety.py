"""Tests for `kenzo.core.controller.AgentController`'s safety nets:
- iteration-budget enforcement (graceful stop, never a crash),
- the Phase 3 equivalent of the old hard test-gate (a FINISH decision can
  never silently read as an unqualified success if the last known test
  result failed),
- and role-boundary (`ACTION_TOOLS`) preservation, both declared AND
  enforced at runtime through the exact scoping mechanism the controller
  itself uses (`self.tools.subset(...)`).
"""

from __future__ import annotations

import json

from kenzo.config.settings import KenzoConfig, PermissionsConfig
from kenzo.core.controller import ACTION_TOOLS, AgentController, ControllerAction
from kenzo.core.schemas import AgentBudgetConfig
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.registry import build_default_registry


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding (see test_controller_scenarios.py).
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


def _is_decide_call(tool_choice) -> bool:
    return (
        isinstance(tool_choice, dict)
        and tool_choice.get("function", {}).get("name") == "decide_next_action"
    )


class _RepeatingDecisionRouter:
    """Always decides the SAME action (never finish/ask_user), and every
    action execution resolves in a single final-text turn (no tool calls) --
    used to prove the iteration budget stops the loop gracefully even when
    the model itself never decides to finish."""

    def __init__(self, action: ControllerAction, rationale: str, final_text: str):
        self._action = action
        self._rationale = rationale
        self._final_text = final_text
        self.decision_call_count = 0
        self.action_call_count = 0

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        if _is_decide_call(tool_choice):
            self.decision_call_count += 1
            call = _FakeToolCall(
                f"decide-{self.decision_call_count}",
                "decide_next_action",
                json.dumps({"action": self._action.value, "rationale": self._rationale, "target": ""}),
            )
            return _FakeModelResponse(None, tool_calls=[call])

        self.action_call_count += 1
        return _FakeModelResponse(self._final_text)


class _ScriptedRouter:
    """Ordered decisions queue + flat ordered actions queue -- same pattern
    as test_controller_scenarios.py's _ControllerScriptedRouter."""

    def __init__(self, decisions, actions):
        self._decisions = list(decisions)
        self._actions = list(actions)

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        if _is_decide_call(tool_choice):
            if not self._decisions:
                raise AssertionError("ran out of scripted decisions")
            return self._decisions.pop(0)
        if not self._actions:
            raise AssertionError("ran out of scripted action responses")
        return self._actions.pop(0)


def _decision(action: ControllerAction, rationale: str, target: str = ""):
    call = _FakeToolCall(
        f"decide-{action.value}",
        "decide_next_action",
        json.dumps({"action": action.value, "rationale": rationale, "target": target}),
    )
    return _FakeModelResponse(None, tool_calls=[call])


def _final_text(text: str):
    return _FakeModelResponse(text)


def _real_tools(repo_root, permissions=None):
    return build_default_registry(str(repo_root), permissions or PermissionsConfig(allow_write=True))


def _real_memory(repo_root):
    return RepositoryMemory(str(repo_root))


# ---------------------------------------------------------------------------
# (a) Iteration budget: graceful stop, never a crash, never relies on FINISH
# ---------------------------------------------------------------------------


def test_loop_stops_gracefully_when_max_iterations_reached_without_ever_finishing(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    config = KenzoConfig(agent=AgentBudgetConfig(max_iterations=3, max_tool_calls=1000, max_runtime_minutes=30))

    router = _RepeatingDecisionRouter(
        ControllerAction.SEARCH, "Still exploring, haven't found it yet.", "Nothing conclusive yet."
    )
    controller = AgentController(router, tools, memory, config)

    result = controller.run("An impossible task that never gets solved")

    assert result["finished"] is True
    assert result["needs_user_input"] is False
    assert "reached the maximum iteration budget" in result["final_response"]
    assert result["iteration"] == 3

    # Exactly 3 full decide+execute cycles happened -- no 4th decision call.
    assert router.decision_call_count == 3
    assert router.action_call_count == 3


def test_loop_stops_gracefully_when_max_tool_calls_reached(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    config = KenzoConfig(agent=AgentBudgetConfig(max_iterations=1000, max_tool_calls=0, max_runtime_minutes=30))

    router = _RepeatingDecisionRouter(
        ControllerAction.SEARCH, "Still exploring.", "Nothing conclusive yet."
    )
    controller = AgentController(router, tools, memory, config)

    result = controller.run("Another task")

    assert result["finished"] is True
    assert "reached the maximum tool-call budget" in result["final_response"]
    # The budget was already exhausted before the very first iteration.
    assert result["iteration"] == 0


# ---------------------------------------------------------------------------
# (b) Test-gate: FINISH while the last known test result is FAILING
# ---------------------------------------------------------------------------


def test_finish_while_last_test_result_failing_produces_warning_and_unsuccessful_result(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    config = KenzoConfig()

    decisions = [
        _decision(ControllerAction.TEST, "Check whether the current code passes."),
        _decision(ControllerAction.FINISH, "I'm out of ideas, finishing anyway despite the failure."),
    ]
    actions = [
        _final_text(json.dumps({"passed": False, "summary": "2 tests failed", "command": "", "raw_output": ""})),
    ]

    router = _ScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, config)

    result = controller.run("Fix something, but the model gives up prematurely")

    assert result["finished"] is True
    assert "WARNING" in result["final_response"]
    assert "FAILING" in result["final_response"]
    assert result["success"] is False


def test_finish_after_passing_test_has_no_warning_and_is_successful(repo_root):
    """Contrast case: proves the WARNING is conditional on a real failure,
    not always injected on every finish."""
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    config = KenzoConfig()

    decisions = [
        _decision(ControllerAction.TEST, "Check whether the current code passes."),
        _decision(ControllerAction.FINISH, "Verified working, done."),
    ]
    actions = [
        _final_text(json.dumps({"passed": True, "summary": "all tests pass", "command": "", "raw_output": ""})),
    ]

    router = _ScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, config)

    result = controller.run("Fix something and confirm it")

    assert result["finished"] is True
    assert "WARNING" not in result["final_response"]
    assert result["success"] is True


# ---------------------------------------------------------------------------
# (c) Role-boundary preservation: declared (ACTION_TOOLS) AND enforced at runtime
# ---------------------------------------------------------------------------


def test_action_tools_declares_expected_role_boundaries():
    assert "write_file" not in ACTION_TOOLS[ControllerAction.SEARCH]
    assert "write_file" not in ACTION_TOOLS[ControllerAction.REVIEW]
    assert "run_command" not in ACTION_TOOLS[ControllerAction.EDIT]
    assert "write_file" in ACTION_TOOLS[ControllerAction.EDIT]
    assert "run_command" in ACTION_TOOLS[ControllerAction.TEST]

    # A few more boundaries worth locking in explicitly, since these are the
    # crux of "never let a read-only action mutate the repo."
    assert "edit_file" not in ACTION_TOOLS[ControllerAction.SEARCH]
    assert "edit_file" not in ACTION_TOOLS[ControllerAction.READ]
    assert "edit_file" not in ACTION_TOOLS[ControllerAction.ANALYZE]
    assert "run_command" not in ACTION_TOOLS[ControllerAction.SEARCH]
    assert "run_command" not in ACTION_TOOLS[ControllerAction.READ]
    assert "run_command" not in ACTION_TOOLS[ControllerAction.ANALYZE]
    assert "run_command" not in ACTION_TOOLS[ControllerAction.REVIEW]


def test_action_tools_role_boundary_is_enforced_at_runtime_not_just_declared(repo_root):
    """Build a real AgentController and, via the exact same `.subset(...)`
    mechanism `_execute_action` uses internally, obtain the SEARCH-scoped
    tool view and prove a hallucinated write_file call fails closed instead
    of silently writing to disk."""
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    config = KenzoConfig()

    router = _RepeatingDecisionRouter(ControllerAction.SEARCH, "irrelevant", "irrelevant")
    controller = AgentController(router, tools, memory, config)

    search_scoped = controller.tools.subset(ACTION_TOOLS[ControllerAction.SEARCH])
    assert "write_file" not in search_scoped.names()

    target_path = repo_root / "should_never_be_written.txt"
    assert not target_path.exists()

    result = search_scoped.execute("write_file", {"path": "should_never_be_written.txt", "content": "malicious"})

    assert result.success is False
    assert "no tool named" in (result.error or "").lower()
    assert not target_path.exists()


def test_edit_scoped_view_can_still_write_for_contrast(repo_root):
    """Contrast case: the EDIT action's scoped view legitimately CAN write,
    proving the SEARCH-scope failure above is really about scoping, not some
    unrelated global misconfiguration."""
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    config = KenzoConfig()

    router = _RepeatingDecisionRouter(ControllerAction.EDIT, "irrelevant", "irrelevant")
    controller = AgentController(router, tools, memory, config)

    edit_scoped = controller.tools.subset(ACTION_TOOLS[ControllerAction.EDIT])
    assert "write_file" in edit_scoped.names()

    result = edit_scoped.execute("write_file", {"path": "allowed_to_write.txt", "content": "hello"})
    assert result.success is True
    assert (repo_root / "allowed_to_write.txt").read_text(encoding="utf-8") == "hello"
