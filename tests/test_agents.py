"""Tests for the five concrete pipeline agents (Planner, Researcher, Coder,
Tester, Reviewer) and their `build_*_node` LangGraph factories, exercised in
isolation (constructed directly, not via `kenzo.core.graph.build_graph`).

Every test fakes only the model/router layer -- a scripted `_ScriptedRouter`
whose `.complete()` returns pre-built responses shaped like real litellm
`ModelResponse` objects (see `_FakeMessage`/`_FakeChoice`/`_FakeModelResponse`
and `_FakeToolCall`/`_FakeFunction` below, following the same pattern used in
test_router.py). Tools and memory are always the REAL
`kenzo.tools.registry.build_default_registry` / `kenzo.memory.repository.RepositoryMemory`
running against the real tmp git repo provided by the `repo_root` fixture, so
tool execution, real file writes, and real diff computation are exercised.

This file specifically proves several bug fixes described in the hardening
pass:

- `BaseAgent._tool_loop` now round-trips the FULL structured `ToolResult`
  (success/output/data/error) back to the model as JSON, not just a bare
  string -- proven by inspecting the raw `messages` list passed into a
  subsequent `.complete()` call.
- `CoderAgent._changes_from_transcript` now captures a real, non-empty
  unified diff from `edit_file`'s tool result data.
- `ReviewerAgent.run()` now hard-overrides an LLM's own `approved: true`
  verdict to `False` whenever `test_result.passed` is `False`, and leaves
  the LLM's verdict alone otherwise.
"""

from __future__ import annotations

import json

from kenzo.agents.coder import CoderAgent, build_coder_node
from kenzo.agents.planner import PlannerAgent, build_planner_node
from kenzo.agents.researcher import ResearcherAgent, build_researcher_node
from kenzo.agents.reviewer import ReviewerAgent, build_reviewer_node
from kenzo.agents.tester import build_tester_node
from kenzo.config.settings import PermissionsConfig
from kenzo.core.schemas import new_state
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.registry import build_default_registry


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding (mirrors test_router.py's pattern,
# extended with tool_calls support).
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


class _ScriptedRouter:
    """Fake router: returns pre-scripted `_FakeModelResponse`s in order.

    Records every call's (task_type, messages, tools) as a snapshot so tests
    can assert on exactly what was sent to the "model" at each step of a
    multi-turn tool-calling loop.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        self.calls.append(
            {
                "task_type": task_type,
                "messages": [dict(m) for m in messages],
                "tools": tools,
            }
        )
        if not self._responses:
            raise AssertionError(
                f"_ScriptedRouter ran out of scripted responses (task_type={task_type})"
            )
        return self._responses.pop(0)


def _real_tools(repo_root, permissions=None):
    return build_default_registry(str(repo_root), permissions or PermissionsConfig())


def _real_memory(repo_root):
    return RepositoryMemory(str(repo_root))


# ---------------------------------------------------------------------------
# (a) Tool-observation-format fix: multi-round-trip tool loop + JSON content
# ---------------------------------------------------------------------------


def test_tool_loop_round_trips_multiple_tool_calls_with_json_observations(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    list_dir_call = _FakeToolCall("call_1", "list_directory", json.dumps({"path": "."}))
    read_file_call = _FakeToolCall("call_2", "read_file", json.dumps({"path": "main.py"}))
    plan_json = json.dumps(
        {
            "goal": "Inspect the repo",
            "steps": [{"id": 1, "description": "do it", "rationale": "because"}],
            "context_notes": "",
        }
    )
    router = _ScriptedRouter(
        [
            _FakeModelResponse(None, tool_calls=[list_dir_call]),
            _FakeModelResponse(None, tool_calls=[read_file_call]),
            _FakeModelResponse(plan_json),
        ]
    )

    agent = PlannerAgent(router, tools, memory)
    state = new_state("Inspect the repo", str(repo_root))

    result = agent.run(state)

    # Three model calls: two tool-call turns plus the final answer.
    assert len(router.calls) == 3
    assert result["plan"]["goal"] == "Inspect the repo"

    # The SECOND call's messages must contain a role="tool" message whose
    # content is valid JSON with success/output/data/error keys, carrying
    # list_directory's real structured payload -- not just a bare string.
    second_call_messages = router.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    observation = json.loads(tool_messages[0]["content"])
    assert set(observation.keys()) == {"success", "output", "data", "error"}
    assert observation["success"] is True
    assert isinstance(observation["data"], dict)
    assert "entries" in observation["data"]  # list_directory's real data payload

    # The THIRD call must carry both tool observations accumulated so far.
    third_call_messages = router.calls[2]["messages"]
    tool_messages_3 = [m for m in third_call_messages if m["role"] == "tool"]
    assert len(tool_messages_3) == 2
    second_observation = json.loads(tool_messages_3[1]["content"])
    assert second_observation["success"] is True
    assert "content" in second_observation["data"]  # read_file's real data payload


# ---------------------------------------------------------------------------
# (b) Diff-capture fix: CoderAgent.run() records a real unified diff
# ---------------------------------------------------------------------------


def test_coder_agent_captures_real_diff_from_edit_file(repo_root):
    tools = _real_tools(repo_root)  # default PermissionsConfig has allow_write=True
    memory = _real_memory(repo_root)

    edit_call = _FakeToolCall(
        "call_1",
        "edit_file",
        json.dumps({"path": "main.py", "old_text": "return 42", "new_text": "return 43"}),
    )
    router = _ScriptedRouter(
        [
            _FakeModelResponse(None, tool_calls=[edit_call]),
            _FakeModelResponse("Changed main() to return 43 instead of 42."),
        ]
    )

    agent = CoderAgent(router, tools, memory)
    state = new_state("Fix the return value", str(repo_root))
    state["plan"] = {"goal": "Fix the return value", "steps": []}
    state["research"] = {"summary": "main.py has the bug"}

    result = agent.run(state)

    assert len(result["changes"]) == 1
    change = result["changes"][0]
    assert change["file_path"] == "main.py"
    assert change["action"] == "edit"
    assert change["applied"] is True

    diff_text = change["diff"]
    assert diff_text  # non-empty string
    lines = diff_text.splitlines()
    assert any(line.startswith("-") and "return 42" in line for line in lines)
    assert any(line.startswith("+") and "return 43" in line for line in lines)

    # The edit was really applied to the real file on disk.
    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 43\n"
    assert result["iteration"] == 1


# ---------------------------------------------------------------------------
# (c) / (d) Hard test-gate fix: ReviewerAgent.run()
# ---------------------------------------------------------------------------


def test_reviewer_hard_gate_overrides_approval_when_tests_failed(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    verdict_json = json.dumps({"approved": True, "summary": "looks fine", "issues": []})
    router = _ScriptedRouter([_FakeModelResponse(verdict_json)])

    agent = ReviewerAgent(router, tools, memory)
    state = new_state("Fix simple bug", str(repo_root))
    state["plan"] = {"goal": "Fix simple bug"}
    state["changes"] = [{"file_path": "main.py", "action": "edit", "diff": "...", "applied": True}]
    state["test_result"] = {"passed": False, "summary": "2 tests failed"}

    result = agent.run(state)

    assert result["review"]["approved"] is False
    assert any("overridden" in issue.lower() for issue in result["review"]["issues"])
    assert result["review_retries"] == 1


def test_reviewer_does_not_override_when_tests_passed(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    verdict_json = json.dumps({"approved": True, "summary": "all good", "issues": []})
    router = _ScriptedRouter([_FakeModelResponse(verdict_json)])

    agent = ReviewerAgent(router, tools, memory)
    state = new_state("Fix simple bug", str(repo_root))
    state["plan"] = {"goal": "Fix simple bug"}
    state["changes"] = []
    state["test_result"] = {"passed": True, "summary": "all tests passed"}

    result = agent.run(state)

    assert result["review"]["approved"] is True
    assert result["review"]["issues"] == []
    assert result["review_retries"] == 0
    assert result["final_response"] is not None  # approved -> final_response always built


# ---------------------------------------------------------------------------
# (e) PlannerAgent fallback plan when the model's answer isn't JSON
# ---------------------------------------------------------------------------


def test_planner_falls_back_to_one_step_plan_on_unparseable_json(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    router = _ScriptedRouter(
        [_FakeModelResponse("I looked around but here is just prose, no JSON.")]
    )

    agent = PlannerAgent(router, tools, memory)
    state = new_state("Fix the typo in README", str(repo_root))

    result = agent.run(state)

    plan = result["plan"]
    assert plan["goal"] == "Fix the typo in README"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["id"] == 1
    assert "Fallback plan" in plan["steps"][0]["rationale"]


# ---------------------------------------------------------------------------
# (f) build_*_node factories delegate to the corresponding Agent.run()
# ---------------------------------------------------------------------------


def test_build_planner_node_delegates_to_planner_agent(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    plan_json = json.dumps({"goal": "g", "steps": [], "context_notes": ""})
    router = _ScriptedRouter([_FakeModelResponse(plan_json)])

    node = build_planner_node(router, tools, memory)
    state = new_state("g", str(repo_root))
    result = node(state)

    assert callable(node)
    assert result["plan"]["goal"] == "g"
    assert len(router.calls) == 1


def test_build_researcher_node_delegates_to_researcher_agent(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    research_json = json.dumps(
        {"summary": "s", "relevant_files": [], "snippets": {}, "open_questions": []}
    )
    router = _ScriptedRouter([_FakeModelResponse(research_json)])

    node = build_researcher_node(router, tools, memory)
    state = new_state("r", str(repo_root))
    result = node(state)

    assert result["research"]["summary"] == "s"


def test_build_coder_node_delegates_to_coder_agent(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    router = _ScriptedRouter([_FakeModelResponse("done, no changes needed")])

    node = build_coder_node(router, tools, memory)
    state = new_state("c", str(repo_root))
    result = node(state)

    assert result["changes"] == []
    assert result["iteration"] == 1


def test_build_tester_node_delegates_to_tester_agent(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    test_json = json.dumps({"passed": True, "summary": "ok", "command": "", "raw_output": ""})
    router = _ScriptedRouter([_FakeModelResponse(test_json)])

    node = build_tester_node(router, tools, memory)
    state = new_state("t", str(repo_root))
    result = node(state)

    assert result["test_result"]["passed"] is True


def test_build_reviewer_node_delegates_to_reviewer_agent(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    verdict_json = json.dumps({"approved": True, "summary": "ok", "issues": []})
    router = _ScriptedRouter([_FakeModelResponse(verdict_json)])

    node = build_reviewer_node(router, tools, memory)
    state = new_state("rv", str(repo_root))
    result = node(state)

    assert result["review"]["approved"] is True
