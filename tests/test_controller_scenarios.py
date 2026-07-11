"""End-to-end scenario tests for `kenzo.core.controller.AgentController` --
the Phase 3 dynamic reasoning loop. These prove genuinely non-linear,
recoverable, and symbol-aware behavior that the old fixed pipeline could
never exhibit: the controller revisits phases in a model-chosen order, can
recover from an incorrect edit attempt without any special-casing, and can
navigate a large-ish repository via Phase 2's symbol/flow tools instead of
plain grep.

Faking approach: only the model/router layer is faked. Tools
(`build_default_registry`), memory (`RepositoryMemory`), and (for Scenario 3)
the full `AgentRuntime` construction are all real, running against real tmp
repositories (`repo_root` / `spring_boot_repo` fixtures).

The fake router (`_ControllerScriptedRouter`) branches on `tool_choice`:
- A call whose `tool_choice` names `decide_next_action` is a *decision* call
  (`AgentController._decide_next_action`) -- served from a dedicated,
  strictly-ordered `decisions` queue.
- Every other call is part of some action's own execution sub-loop
  (`run_tool_loop`, invoked from `AgentController._execute_action`). Each
  `_execute_action` call starts a BRAND NEW message list from scratch (see
  `kenzo.core.tool_loop.run_tool_loop`), so these calls are safely served
  from one flat, strictly-ordered `actions` queue shared across every
  action's sub-loop -- there is never any ambiguity about which response
  belongs to which turn, since decisions and action-executions always
  alternate in lockstep with the controller's own `while` loop.

Every script below is deliberately built so each action's sub-loop resolves
in exactly the number of turns intended (either a single final-text answer,
or exactly one tool call followed by a final-text answer) -- never leaving
the fake router in a state where the sub-loop would keep re-issuing calls
against an exhausted queue.
"""

from __future__ import annotations

import json

from kenzo.config.settings import KenzoConfig, PermissionsConfig, ProviderConfig
from kenzo.core.controller import AgentController, ControllerAction
from kenzo.core.runtime import AgentRuntime
from kenzo.memory.repository import RepositoryMemory
from kenzo.models.router import ModelRouter
from kenzo.tools.registry import build_default_registry


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding (same pattern as test_agents.py /
# test_graph.py / test_runtime.py).
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
    queue for every action-execution call (see module docstring)."""

    def __init__(self, decisions, actions):
        self._decisions = list(decisions)
        self._actions = list(actions)
        self.decision_calls: list[dict] = []
        self.action_calls: list[dict] = []

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        # Snapshot (copy) messages rather than keeping the live list reference --
        # `run_tool_loop` mutates `messages` in place across turns, so storing
        # the bare reference would make every recorded call alias the SAME
        # final list once the loop finishes.
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


# ---------------------------------------------------------------------------
# Scenario 1: search -> read -> edit -> test(fail) -> edit -> test(pass) -> finish
# ---------------------------------------------------------------------------


def test_scenario_search_read_edit_test_fail_edit_test_pass_finish(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    decisions = [
        _decision(ControllerAction.PLAN, "Form a plan before touching anything."),
        _decision(ControllerAction.SEARCH, "Look for where the wrong value is returned.", "return 42"),
        _decision(ControllerAction.READ, "Read main.py to confirm its exact contents.", "main.py"),
        _decision(ControllerAction.EDIT, "Apply a first fix attempt.", "main.py"),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(ControllerAction.EDIT, "First fix was incomplete -- apply the real fix.", "main.py"),
        _decision(ControllerAction.TEST, "Validate the corrected fix."),
        _decision(ControllerAction.FINISH, "The goal is achieved and verified by a passing test."),
    ]

    plan_json = json.dumps(
        {"goal": "Make main() return 100", "steps": [{"id": 1, "description": "fix main.py", "rationale": "bug"}], "context_notes": ""}
    )

    actions = [
        # PLAN: a single, final JSON answer, no tool calls needed.
        _final_text(plan_json),
        # SEARCH: one search_files call, then a final summary.
        _tool_call_response("call-search", "search_files", {"pattern": "return 42"}),
        _final_text("Found 'return 42' in main.py."),
        # READ: one read_file call, then a final summary.
        _tool_call_response("call-read", "read_file", {"path": "main.py"}),
        _final_text("main.py currently returns 42."),
        # EDIT #1: succeeds, but produces an INCOMPLETE fix (42 -> 43).
        _tool_call_response(
            "call-edit-1", "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 43"}
        ),
        _final_text("Changed main.py to return 43."),
        # TEST #1: fails -- 43 isn't the expected value.
        _final_text(json.dumps({"passed": False, "summary": "expected 100, got 43", "command": "", "raw_output": ""})),
        # EDIT #2: succeeds, this time with the CORRECT fix (43 -> 100).
        _tool_call_response(
            "call-edit-2", "edit_file", {"path": "main.py", "old_text": "return 43", "new_text": "return 100"}
        ),
        _final_text("Changed main.py to return 100."),
        # TEST #2: passes.
        _final_text(json.dumps({"passed": True, "summary": "returns 100 as expected", "command": "", "raw_output": ""})),
    ]

    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    result = controller.run("Make main() return 100")

    # The final file content reflects the SECOND edit's correct fix.
    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 100\n"

    assert result["success"] is True
    assert result["finished"] is True

    test_results = result["test_results"]
    assert len(test_results) >= 2
    assert test_results[0]["passed"] is False
    assert test_results[-1]["passed"] is True

    # No failed TOOL calls happened in this scenario -- both edits succeeded;
    # only the intermediate TEST verdict was a failure, which is tracked in
    # test_results, not failed_attempts.
    assert result["task_state"]["failed_attempts"] == []


# ---------------------------------------------------------------------------
# Scenario 2: wrong first hypothesis (bad old_text) -> recover via read -> edit
# ---------------------------------------------------------------------------


def test_scenario_recovers_from_wrong_edit_hypothesis(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    decisions = [
        _decision(ControllerAction.EDIT, "Try the fix I believe is right.", "main.py"),
        _decision(ControllerAction.READ, "That failed -- re-examine the actual file contents.", "main.py"),
        _decision(ControllerAction.EDIT, "Now apply the fix using the real text.", "main.py"),
        _decision(ControllerAction.TEST, "Confirm the fix works."),
        _decision(ControllerAction.FINISH, "Fixed and verified."),
    ]

    actions = [
        # EDIT #1: a plausible but WRONG old_text that does not exist in main.py.
        _tool_call_response(
            "call-bad-edit",
            "edit_file",
            {"path": "main.py", "old_text": "return 41", "new_text": "return 100"},
        ),
        _final_text("That edit failed; I'll re-read the file to see what's actually there."),
        # READ: re-examine the real contents.
        _tool_call_response("call-read", "read_file", {"path": "main.py"}),
        _final_text("main.py actually contains 'return 42', not 'return 41'."),
        # EDIT #2: the CORRECT old_text this time.
        _tool_call_response(
            "call-good-edit",
            "edit_file",
            {"path": "main.py", "old_text": "return 42", "new_text": "return 100"},
        ),
        _final_text("Changed main.py to return 100."),
        # TEST: passes.
        _final_text(json.dumps({"passed": True, "summary": "returns 100", "command": "", "raw_output": ""})),
    ]

    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    result = controller.run("Fix main() to return 100")

    # The final file content reflects the CORRECT fix.
    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 100\n"

    failed_attempts = result["task_state"]["failed_attempts"]
    assert len(failed_attempts) >= 1
    matching = [fa for fa in failed_attempts if fa["action"] == "edit"]
    assert matching, f"expected a failed edit attempt, got: {failed_attempts}"
    assert "not found" in matching[0]["error"].lower()

    # Only the SUCCESSFUL edit becomes a recorded change -- the failed
    # attempt never produced a change entry.
    changes = result["changes"]
    matching_changes = [c for c in changes if c["file_path"] == "main.py"]
    assert len(matching_changes) == 1
    assert matching_changes[0]["diff"]  # non-empty real diff
    assert "return 100" in matching_changes[0]["diff"]

    assert result["success"] is True


# ---------------------------------------------------------------------------
# Scenario 3: large-ish Spring project navigation via find_symbol/map_flow
# ---------------------------------------------------------------------------


def _spring_config():
    return KenzoConfig(
        providers={"nemotron": ProviderConfig(api_key="fake-key", default_model="fake-model")},
        permissions=PermissionsConfig(allow_write=True, allow_shell_exec=False),
    )


def test_scenario_controller_uses_find_symbol_and_map_flow_to_trace_logout_flow(spring_boot_repo, monkeypatch):
    decisions = [
        _decision(
            ControllerAction.ANALYZE,
            "Trace how the logout endpoint flows through the codebase before changing anything.",
            "/api/auth/logout",
        ),
        _decision(ControllerAction.FINISH, "Traced the full logout call chain successfully."),
    ]

    actions = [
        # ANALYZE: find_symbol, then map_flow, then a final summary.
        _tool_call_response("call-find-symbol", "find_symbol", {"name": "LogoutController"}),
        _tool_call_response("call-map-flow", "map_flow", {"endpoint": "/api/auth/logout"}),
        _final_text("LogoutController.logout() calls AuthService.terminateSession(), which calls UserRepository."),
    ]

    router = _ControllerScriptedRouter(decisions, actions)

    def _scripted_complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        return router.complete(task_type, messages, tools=tools, tool_choice=tool_choice, **kwargs)

    monkeypatch.setattr(ModelRouter, "complete", _scripted_complete)

    config = _spring_config()
    runtime = AgentRuntime(repo_path=str(spring_boot_repo), config=config)

    # AgentRuntime construction (real Java symbol analysis) really did build
    # .kenzo/symbols.json, entirely offline.
    assert (spring_boot_repo / ".kenzo" / "symbols.json").exists()

    result = runtime.controller.run("Trace the logout flow end to end")

    assert result["finished"] is True

    # The find_symbol call really ran against the real indexed symbols and
    # surfaced LogoutController's real file path into files_seen.
    files_seen = result["files_seen"]
    assert any("LogoutController.java" in f for f in files_seen)

    # Prove find_symbol/map_flow were actually EXECUTED (not just requested) --
    # by inspecting the AgentController's own ContextManager, which is only
    # ever populated by `_apply_updates` processing a REAL, successful tool
    # result (see kenzo.core.controller._apply_updates' find_symbol/map_flow
    # branches). This is real proof of symbol-based navigation, not just that
    # the fake router happened to be asked for these tools.
    assert "LogoutController" in runtime.controller.context.symbols_seen

    mapped_flow_notes = [n for n in runtime.controller.context.notes if n.startswith("mapped flow for")]
    assert mapped_flow_notes, f"expected a map_flow note, got notes={runtime.controller.context.notes}"
    chain_note = mapped_flow_notes[0]
    assert "LogoutController" in chain_note
    assert "AuthService" in chain_note
