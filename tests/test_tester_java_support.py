"""Tests for Phase 2's Java-awareness additions around the Tester agent:
the widened default command_allowlist, run_command's raised default timeout,
and TesterAgent actually surfacing the detected project's test command in
its prompt (rather than defaulting to a Python-flavored assumption).
"""

from __future__ import annotations

import inspect
import json

from kenzo.agents import tester as tester_module
from kenzo.agents.tester import build_tester_node
from kenzo.config.settings import PermissionsConfig
from kenzo.core.schemas import new_state
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.registry import build_default_registry
from kenzo.tools.terminal import run_command

# `tester_module.TesterAgent` is referenced via the module (not imported by
# bare name) so pytest's default `Test*` class-collection heuristic doesn't
# try to collect it as a test class (it has an __init__, which would warn) --
# mirrors why tests/test_agents.py imports only `build_tester_node`, never
# `TesterAgent` itself, from this module.


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


class _CapturingRouter:
    """Fake router that returns one scripted final answer and records the
    exact messages it was called with, so the test can inspect the rendered
    prompt directly (no tool-calling round trip needed for this assertion)."""

    def __init__(self, response):
        self._response = response
        self.captured_messages: list[dict] | None = None
        self.call_count = 0

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        self.call_count += 1
        self.captured_messages = messages
        return self._response


# ---------------------------------------------------------------------------
# (a) command_allowlist widened for Java build tools
# ---------------------------------------------------------------------------


def test_default_command_allowlist_includes_java_build_tools():
    allowlist = PermissionsConfig().command_allowlist

    for tool in ("mvn", "mvnw", "gradle", "gradlew", "java"):
        assert tool in allowlist


# ---------------------------------------------------------------------------
# (b) run_command's default timeout raised to 300s
# ---------------------------------------------------------------------------


def test_run_command_default_timeout_is_300_seconds():
    default_timeout = inspect.signature(run_command).parameters["timeout"].default
    assert default_timeout == 300


# ---------------------------------------------------------------------------
# (c) TesterAgent surfaces the detected Java test command in its prompt
# ---------------------------------------------------------------------------


def test_tester_agent_prompt_surfaces_detected_maven_test_command(repo_root):
    memory = RepositoryMemory(str(repo_root))
    memory.save_project_info(
        {"language": "java", "framework": "spring-boot", "build": "maven", "test_command": "mvn test"}
    )

    tools = build_default_registry(str(repo_root), PermissionsConfig())

    test_json = json.dumps(
        {"passed": True, "summary": "ran mvn test successfully", "command": "mvn test", "raw_output": ""}
    )
    router = _CapturingRouter(_FakeModelResponse(test_json))

    state = new_state("Fix the login bug", str(repo_root))
    state["plan"] = {"goal": "Fix the login bug", "steps": []}
    state["changes"] = [{"file_path": "LoginController.java", "action": "edit", "diff": "...", "applied": True}]

    agent = tester_module.TesterAgent(router, tools, memory)
    result = agent.run(state)

    assert router.call_count == 1
    assert router.captured_messages is not None
    user_messages = [m["content"] for m in router.captured_messages if m["role"] == "user"]
    assert any("mvn test" in content for content in user_messages)
    assert any("language=java" in content for content in user_messages)
    assert any("build=maven" in content for content in user_messages)

    assert result["test_result"]["passed"] is True


def test_tester_agent_prompt_falls_back_to_unknown_when_no_project_info(repo_root):
    """Sanity check for the "before" state: with no .kenzo/project.json at
    all, TesterAgent must not silently invent a command -- it should say
    '(unknown)' rather than defaulting to a Python-flavored assumption."""
    memory = RepositoryMemory(str(repo_root))
    tools = build_default_registry(str(repo_root), PermissionsConfig())

    test_json = json.dumps({"passed": True, "summary": "ok", "command": "", "raw_output": ""})
    router = _CapturingRouter(_FakeModelResponse(test_json))

    state = new_state("Fix something", str(repo_root))
    state["plan"] = {"goal": "Fix something", "steps": []}
    state["changes"] = []

    agent = tester_module.TesterAgent(router, tools, memory)
    agent.run(state)

    user_messages = [m["content"] for m in router.captured_messages if m["role"] == "user"]
    assert any("recommended test command=(unknown)" in content for content in user_messages)


def test_build_tester_node_still_delegates_to_tester_agent(repo_root):
    memory = RepositoryMemory(str(repo_root))
    tools = build_default_registry(str(repo_root), PermissionsConfig())
    test_json = json.dumps({"passed": True, "summary": "ok", "command": "", "raw_output": ""})
    router = _CapturingRouter(_FakeModelResponse(test_json))

    node = build_tester_node(router, tools, memory)
    state = new_state("t", str(repo_root))
    result = node(state)

    assert result["test_result"]["passed"] is True
