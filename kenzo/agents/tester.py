"""Tester agent: verifies that the coder's changes actually work."""

from __future__ import annotations

from typing import Any

from kenzo.agents.base import BaseAgent
from kenzo.core.schemas import TaskType

SYSTEM_PROMPT = """\
You are the Testing agent for Kenzo, a local repository-aware coding agent.

You are given a list of code changes that were just applied to the
repository. Your job is to verify that these changes actually work.

Kenzo supports Java (Maven: `mvn test` / `./mvnw test`, or Gradle:
`gradle test` / `./gradlew test`), Python (`pytest`), and Node
(`npm test` / `yarn test`) projects. The user prompt below tells you the
detected project's language, build system, and the specific test command to
prefer for THIS repository -- use that command rather than guessing; only
fall back to a different one if it's missing or clearly wrong. Preferably,
use the run_command tool to execute the project's real test suite and
observe the real output; pass an explicit `timeout` (Java/Maven/Gradle test
runs can legitimately take several minutes) rather than relying on the
default. If run_command reports that shell execution is disabled or the
command is not allowed, you MUST explicitly say so in your final answer's
summary, and instead perform a careful static read-through of the changed
files via read_file to reason about correctness as best you can.

Before running the whole suite, call find_related_tests(file_path) for each
changed file (if any) to identify the specific test file(s) most likely to
cover it (e.g. UserService.java -> UserServiceTest.java) -- use that to
target your run/read-through, rather than assuming the whole suite or
guessing a test file's name from the source file's name alone.

Once you have verified the changes (by running tests or by static
read-through), respond with your final answer. Your final answer MUST be
ONLY a single JSON object (no prose, no markdown fences) matching this exact
shape:

{
  "passed": true or false,
  "summary": "<what you did and what you found>",
  "command": "<the command you ran, or empty string if none was run>",
  "raw_output": "<raw output from the command, or empty string if none was run>"
}
"""


class TesterAgent(BaseAgent):
    name = "tester"
    task_type = TaskType.DEBUGGING

    def run(self, state: dict) -> dict:
        changes = state.get("changes", [])
        plan = state.get("plan") or {}
        working_context = state.get("working_context") or {"files_discovered": [], "notes": []}

        project_info = self.memory.load_project_info() or {}
        project_line = (
            f"Detected project: language={project_info.get('language', 'unknown')}, "
            f"framework={project_info.get('framework') or '(none)'}, "
            f"build={project_info.get('build') or '(none)'}, "
            f"recommended test command={project_info.get('test_command') or '(unknown)'}"
        )

        context_note = self._render_working_context(working_context)
        user_prompt = (
            (context_note + "\n\n" if context_note else "")
            + f"{project_line}\n\n"
            f"Plan goal:\n{plan.get('goal', '')}\n\n"
            f"Changes that were applied to the repository:\n{changes}\n\n"
            "Verify these changes now (preferably by running the recommended test "
            "command above via run_command with an explicit generous timeout; fall "
            "back to a static read-through via read_file if shell execution is "
            "disabled), then produce the final test result JSON described in your "
            "instructions."
        )

        final_text, tool_transcript = self._tool_loop(SYSTEM_PROMPT, user_prompt)

        test_result = self._extract_json(final_text)
        if not test_result or "passed" not in test_result:
            test_result = self._fallback_test_result(final_text)

        # Normalize in place: downstream consumers (reviewer's prompt, the
        # CLI renderer) read `test_result["passed"]` directly, so a model
        # that emits "passed": "false" as a string must not survive as a
        # truthy value once stored in state.
        test_result["passed"] = self._coerce_bool(test_result.get("passed", False))

        transcript_entry: dict[str, Any] = {
            "role": "assistant",
            "content": f"[tester] {test_result.get('summary', '')}",
            "name": self.name,
        }

        return {
            "test_result": test_result,
            "messages": state.get("messages", []) + [transcript_entry],
            "working_context": self._update_working_context(working_context, tool_transcript),
        }

    def _fallback_test_result(self, raw_text: str) -> dict:
        """Conservative fallback used when the model's JSON couldn't be parsed."""
        summary = raw_text.strip() or "Tester output could not be parsed as JSON."
        return {
            "passed": False,
            "summary": summary,
            "command": "",
            "raw_output": raw_text,
        }


def build_tester_node(router: Any, tools: Any, memory: Any):
    """LangGraph node factory for the Tester agent."""

    def node(state: dict) -> dict:
        return TesterAgent(router, tools, memory).run(state)

    return node
