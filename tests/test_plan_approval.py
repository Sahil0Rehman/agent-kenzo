"""Tests for Phase 12's "pause after every PLAN for human approval" feature:

- `AgentController.plan_approval_handler` / `set_plan_approval_handler` and
  the pause block in `AgentController.run()` right after a PLAN action's
  `_apply_updates` call (`kenzo/core/controller.py`).
- `kenzo.cli.commands.session._render_plan_for_approval`, the CLI's concrete
  Rich-panel-plus-`console.input` renderer, and `start_interactive_session`'s
  `plan_first` wiring.
- `kenzo/cli/app.py`'s `--plan-first` flag.

Controller-level faking approach mirrors `tests/test_controller_scenarios.py`
EXACTLY: only the model/router layer is faked (`_ControllerScriptedRouter`,
`_decision`/`_tool_call_response`/`_final_text`); tools
(`build_default_registry`), memory (`RepositoryMemory`), and config
(`KenzoConfig()`) are all real, running against a real tmp git repository
(the shared `repo_root` fixture from `tests/conftest.py`).

CLI-level tests mock `kenzo.cli.commands.session.console` directly (no
`CliRunner`/console-mocking precedent exists elsewhere in this project, per
a grep across `tests/` before writing these).
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

import kenzo.cli.commands.session as session
from kenzo.config.settings import KenzoConfig, PermissionsConfig
from kenzo.core.controller import AgentController, ControllerAction
from kenzo.core.runtime import AgentRuntime
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.registry import build_default_registry


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding (copied verbatim from
# tests/test_controller_scenarios.py's established conventions).
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


class _ControllerScriptedRouter:
    """Fake router for `AgentController`: a dedicated ordered queue for the
    forced `decide_next_action` decision calls, and a separate flat ordered
    queue for every action-execution call (see
    `test_controller_scenarios.py`'s module docstring for why this split is
    safe/unambiguous)."""

    def __init__(self, decisions, actions):
        self._decisions = list(decisions)
        self._actions = list(actions)
        self.decision_calls: list[dict] = []
        self.action_calls: list[dict] = []

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        snapshot = [dict(m) for m in messages]
        if _is_decide_call(tool_choice):
            self.decision_calls.append({"task_type": task_type, "messages": snapshot})
            if not self._decisions:
                raise AssertionError("ran out of scripted decisions")
            return self._decisions.pop(0)

        self.action_calls.append({"task_type": task_type, "messages": snapshot})
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


def _tool_call_response(call_id: str, tool_name: str, arguments: dict):
    return _FakeModelResponse(None, tool_calls=[_FakeToolCall(call_id, tool_name, json.dumps(arguments))])


def _final_text(text: str):
    return _FakeModelResponse(text)


def _real_tools(repo_root, permissions=None):
    return build_default_registry(str(repo_root), permissions or PermissionsConfig(allow_write=True))


def _real_memory(repo_root):
    return RepositoryMemory(str(repo_root))


def _real_config():
    return KenzoConfig()


def _plan_json(goal: str, steps: list[str]) -> str:
    return json.dumps(
        {
            "goal": goal,
            "steps": [{"id": i + 1, "description": step, "rationale": "because"} for i, step in enumerate(steps)],
            "context_notes": "",
        }
    )


# ---------------------------------------------------------------------------
# Controller-level: 1. default behavior (no handler set) is a pure no-op.
# ---------------------------------------------------------------------------


def test_no_handler_by_default_and_full_run_unaffected(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    decisions = [
        _decision(ControllerAction.PLAN, "Form a plan before touching anything."),
        _decision(ControllerAction.EDIT, "Apply the fix.", "main.py"),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(ControllerAction.FINISH, "The goal is achieved and verified by a passing test."),
    ]
    actions = [
        _final_text(_plan_json("Make main() return 100", ["fix main.py"])),
        _tool_call_response(
            "call-edit", "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 100"}
        ),
        _final_text("Changed main.py to return 100."),
        _final_text(json.dumps({"passed": True, "summary": "returns 100", "command": "", "raw_output": ""})),
    ]
    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    # Most important regression-safety assertion in this file: constructing
    # an AgentController never sets a pause handler by itself.
    assert controller.plan_approval_handler is None

    result = controller.run("Make main() return 100")

    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 100\n"
    assert result["success"] is True
    assert result["finished"] is True
    assert "not approved" not in (result["final_response"] or "")


# ---------------------------------------------------------------------------
# Controller-level: 2. the setter stores exactly what's passed and clears.
# ---------------------------------------------------------------------------


def test_set_plan_approval_handler_stores_and_clears(repo_root):
    controller = AgentController(
        _ControllerScriptedRouter([], []), _real_tools(repo_root), _real_memory(repo_root), _real_config()
    )

    calls: list[tuple] = []

    def handler(task_state, rationale):
        calls.append((task_state, rationale))
        return True

    controller.set_plan_approval_handler(handler)
    assert controller.plan_approval_handler is handler

    controller.set_plan_approval_handler(None)
    assert controller.plan_approval_handler is None
    # Never actually invoked -- this test is purely about the attribute.
    assert calls == []


# ---------------------------------------------------------------------------
# Controller-level: 3. an APPROVED plan continues normally; handler called
# exactly once with the right goal/remaining_work/rationale.
# ---------------------------------------------------------------------------


def test_approved_plan_continues_and_handler_called_once_with_correct_args(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    goal = "Make main() return 100"
    plan_rationale = "Form a plan before touching anything."
    decisions = [
        _decision(ControllerAction.PLAN, plan_rationale),
        _decision(ControllerAction.EDIT, "Apply the fix.", "main.py"),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(ControllerAction.FINISH, "The goal is achieved and verified by a passing test."),
    ]
    actions = [
        _final_text(_plan_json(goal, ["fix main.py"])),
        _tool_call_response(
            "call-edit", "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 100"}
        ),
        _final_text("Changed main.py to return 100."),
        _final_text(json.dumps({"passed": True, "summary": "returns 100", "command": "", "raw_output": ""})),
    ]
    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    calls: list[tuple] = []

    def handler(task_state, rationale):
        # Capture (don't just blindly return True) -- goal/remaining_work
        # must reflect the JUST-COMPLETED plan's own parsed steps.
        calls.append((task_state.goal, list(task_state.remaining_work), rationale))
        return True

    controller.set_plan_approval_handler(handler)

    result = controller.run(goal)

    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 100\n"
    assert result["success"] is True
    assert result["finished"] is True

    assert len(calls) == 1
    called_goal, called_remaining_work, called_rationale = calls[0]
    assert called_goal == goal
    assert called_remaining_work == ["fix main.py"]
    assert called_rationale == plan_rationale


# ---------------------------------------------------------------------------
# Controller-level: 4. a REJECTED plan stops immediately, nothing else runs.
# ---------------------------------------------------------------------------


def test_rejected_plan_stops_immediately_no_further_actions_reached(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    original_content = (repo_root / "main.py").read_text(encoding="utf-8")

    plan_rationale = "Form a plan before touching anything."
    decisions = [
        _decision(ControllerAction.PLAN, plan_rationale),
        # Deliberately NOTHING else scripted for EDIT/TEST/FINISH -- if the
        # loop asked for even one more decision or action-execution call
        # after the rejection, the fake router would raise "ran out of
        # scripted decisions"/"ran out of scripted action responses" and
        # this test would fail with THAT AssertionError, not the ones below.
        # Getting past `controller.run(...)` at all is itself proof the
        # loop never asked again.
    ]
    actions = [
        _final_text(_plan_json("Make main() return 100", ["fix main.py"])),
    ]
    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    controller.set_plan_approval_handler(lambda task_state, rationale: False)

    result = controller.run("Make main() return 100")

    assert result["finished"] is True
    assert result["success"] is False
    assert "not approved" in (result["final_response"] or "")
    assert result["changes"] == []

    # Re-read from disk (not just trusting the empty `changes` list) to
    # prove nothing was actually written.
    assert (repo_root / "main.py").read_text(encoding="utf-8") == original_content


# ---------------------------------------------------------------------------
# Controller-level: 5. two PLAN actions in one task -> handler called twice.
# ---------------------------------------------------------------------------


def test_second_plan_in_same_task_triggers_handler_again_with_revised_steps(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    first_plan_rationale = "Form an initial plan."
    second_plan_rationale = "Revise the plan after searching."
    decisions = [
        _decision(ControllerAction.PLAN, first_plan_rationale),
        _decision(ControllerAction.SEARCH, "Look for where the wrong value is returned.", "return 42"),
        _decision(ControllerAction.PLAN, second_plan_rationale),
        _decision(ControllerAction.EDIT, "Apply the revised fix.", "main.py"),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(ControllerAction.FINISH, "The goal is achieved and verified by a passing test."),
    ]
    actions = [
        # PLAN #1
        _final_text(_plan_json("Make main() return 100", ["initial step"])),
        # SEARCH
        _tool_call_response("call-search", "search_files", {"pattern": "return 42"}),
        _final_text("Found 'return 42' in main.py."),
        # PLAN #2 -- OVERWRITES remaining_work with the revised steps.
        _final_text(_plan_json("Make main() return 100", ["revised step A", "revised step B"])),
        # EDIT
        _tool_call_response(
            "call-edit", "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 100"}
        ),
        _final_text("Changed main.py to return 100."),
        # TEST
        _final_text(json.dumps({"passed": True, "summary": "returns 100", "command": "", "raw_output": ""})),
    ]
    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    calls: list[tuple] = []

    def handler(task_state, rationale):
        calls.append((list(task_state.remaining_work), rationale))
        return True

    controller.set_plan_approval_handler(handler)

    result = controller.run("Make main() return 100")

    assert result["success"] is True
    assert result["finished"] is True

    assert len(calls) == 2
    assert calls[0] == (["initial step"], first_plan_rationale)
    assert calls[1] == (["revised step A", "revised step B"], second_plan_rationale)


# ---------------------------------------------------------------------------
# Controller-level: 6. malformed plan JSON still triggers the handler.
# ---------------------------------------------------------------------------


def test_plan_with_unparseable_json_still_triggers_handler(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    plan_rationale = "Attempt to plan, but this answer will not be valid JSON."
    decisions = [
        _decision(ControllerAction.PLAN, plan_rationale),
    ]
    actions = [
        # No '{' at all -- extract_json returns None immediately, so
        # _apply_updates' PLAN branch never touches task_state.remaining_work
        # (it stays the fresh-task default, []).
        _final_text("I could not form a structured plan for this task right now."),
    ]
    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    calls: list[tuple] = []

    def handler(task_state, rationale):
        calls.append((task_state.goal, list(task_state.remaining_work), rationale))
        return False  # reject to stop cleanly -- proving the CALL happened is the point here

    controller.set_plan_approval_handler(handler)

    result = controller.run("Make main() return 100")

    assert len(calls) == 1
    called_goal, called_remaining_work, called_rationale = calls[0]
    assert called_goal == "Make main() return 100"
    assert called_remaining_work == []
    assert called_rationale == plan_rationale

    assert result["finished"] is True
    assert "not approved" in (result["final_response"] or "")


# ---------------------------------------------------------------------------
# Controller-level: 7. the handler runs synchronously; a raised exception
# propagates out of controller.run() rather than being swallowed.
# ---------------------------------------------------------------------------


def test_handler_exception_propagates_out_of_run(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    decisions = [
        _decision(ControllerAction.PLAN, "Form a plan before touching anything."),
    ]
    actions = [
        _final_text(_plan_json("Make main() return 100", ["fix main.py"])),
    ]
    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    def handler(task_state, rationale):
        raise RuntimeError("boom: console.input() hit EOF")

    controller.set_plan_approval_handler(handler)

    with pytest.raises(RuntimeError, match="boom"):
        controller.run("Make main() return 100")


# ---------------------------------------------------------------------------
# CLI-level: a minimal fake Rich console -- only `.print`/`.input` are used
# by `_render_plan_for_approval`/`start_interactive_session`.
# ---------------------------------------------------------------------------


class _FakeConsole:
    def __init__(self, input_value=""):
        self._input_value = input_value
        self.print_calls: list = []
        self.input_calls: list[str] = []

    def print(self, *args, **kwargs):
        self.print_calls.append(args[0] if args else None)

    def input(self, prompt: str = ""):
        self.input_calls.append(prompt)
        return self._input_value


# ---------------------------------------------------------------------------
# CLI-level: 8/9. _render_plan_for_approval's exact y/n matching logic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["", "y", "Y", "yes", "YES please", "sure", "  ", "not no"])
def test_render_plan_for_approval_approves_everything_except_n_no(monkeypatch, answer):
    fake_console = _FakeConsole(answer)
    monkeypatch.setattr(session, "console", fake_console)

    task_state = SimpleNamespace(goal="Do the thing", remaining_work=["step one"])
    result = session._render_plan_for_approval(task_state, "because reasons")

    assert result is True


@pytest.mark.parametrize("answer", ["n", "N", "no", "NO", "No", "  n  ", "  no  "])
def test_render_plan_for_approval_rejects_only_n_no(monkeypatch, answer):
    fake_console = _FakeConsole(answer)
    monkeypatch.setattr(session, "console", fake_console)

    task_state = SimpleNamespace(goal="Do the thing", remaining_work=["step one"])
    result = session._render_plan_for_approval(task_state, "because reasons")

    assert result is False


# ---------------------------------------------------------------------------
# CLI-level: 10. empty remaining_work renders the placeholder, doesn't crash.
# ---------------------------------------------------------------------------


def test_render_plan_for_approval_empty_steps_uses_placeholder(monkeypatch):
    fake_console = _FakeConsole("")
    monkeypatch.setattr(session, "console", fake_console)

    task_state = SimpleNamespace(goal="Do the thing", remaining_work=[])
    result = session._render_plan_for_approval(task_state, "because reasons")

    assert result is True
    assert fake_console.print_calls, "expected at least one console.print call"
    panel = fake_console.print_calls[0]
    assert "(no explicit steps -- see rationale)" in panel.renderable


# ---------------------------------------------------------------------------
# CLI-level: 11. start_interactive_session's plan_first wiring.
# ---------------------------------------------------------------------------


def _eof_on_input(prompt: str = ""):
    raise EOFError()


def test_start_interactive_session_plan_first_true_wires_real_handler(repo_root, kenzo_home, monkeypatch):
    captured: dict = {}

    class _CapturingAgentRuntime(AgentRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["runtime"] = self

    monkeypatch.setattr(session, "AgentRuntime", _CapturingAgentRuntime)

    fake_console = _FakeConsole()
    fake_console.input = _eof_on_input
    monkeypatch.setattr(session, "console", fake_console)

    # The interactive while-True loop exits as soon as console.input() first
    # raises EOFError -- already caught by start_interactive_session's own
    # try/except as a clean exit, so this returns instead of hanging.
    session.start_interactive_session(repo_path=str(repo_root), plan_first=True)

    runtime = captured["runtime"]
    assert runtime.controller.plan_approval_handler is session._render_plan_for_approval


def test_start_interactive_session_plan_first_false_leaves_handler_none(repo_root, kenzo_home, monkeypatch):
    captured: dict = {}

    class _CapturingAgentRuntime(AgentRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["runtime"] = self

    monkeypatch.setattr(session, "AgentRuntime", _CapturingAgentRuntime)

    fake_console = _FakeConsole()
    fake_console.input = _eof_on_input
    monkeypatch.setattr(session, "console", fake_console)

    # plan_first omitted -- must default to False, exactly like every
    # pre-Phase-12 caller.
    session.start_interactive_session(repo_path=str(repo_root))

    runtime = captured["runtime"]
    assert runtime.controller.plan_approval_handler is None


# ---------------------------------------------------------------------------
# kenzo/cli/app.py: --plan-first flag exists and defaults to False.
#
# No CliRunner/CLI-argument-parsing precedent exists elsewhere in tests/ (per
# a grep across the suite before writing this) -- so per the task's own
# guidance, a light `inspect.signature` check is enough here; this isn't the
# core of the feature.
# ---------------------------------------------------------------------------


def test_app_plan_first_flag_exists_and_defaults_to_false():
    from kenzo.cli.app import main_callback

    sig = inspect.signature(main_callback)
    assert "plan_first" in sig.parameters

    param = sig.parameters["plan_first"]
    # `plan_first: bool = typer.Option(False, "--plan-first", ...)` -- the
    # parameter's default is a `typer.models.OptionInfo` instance (Typer
    # reads its own `.default` at CLI-parse time), so unwrap that rather
    # than assuming the raw signature default is already a plain bool.
    resolved_default = getattr(param.default, "default", param.default)
    assert resolved_default is False
