"""Planner agent: inspects the repository and produces a structured Plan."""

from __future__ import annotations

from typing import Any

from kenzo.agents.base import BaseAgent
from kenzo.core.schemas import TaskType

SYSTEM_PROMPT = """\
You are the Planning agent for Kenzo, a local repository-aware coding agent.

Your job is to turn a user's request into a concrete, actionable plan for the
downstream agents (Researcher, Coder, Tester, Reviewer) to execute.

You MUST use the available tools (list_directory, read_file, search_files,
and the git status/history tools) to inspect the ACTUAL repository on disk
BEFORE proposing any plan. Never answer immediately from assumptions alone --
look at the real directory structure, read relevant files, and check git
status/history first so your plan is grounded in what actually exists in the
repo.

Once you have inspected the repository enough to plan confidently, respond
with your final answer. Your final answer MUST be ONLY a single JSON object
(no prose, no markdown fences) matching this exact shape:

{
  "goal": "<one sentence describing the overall goal>",
  "steps": [
    {"id": 1, "description": "<what to do>", "rationale": "<why>"},
    {"id": 2, "description": "...", "rationale": "..."}
  ],
  "context_notes": "<any useful context for the other agents>"
}
"""


class PlannerAgent(BaseAgent):
    name = "planner"
    task_type = TaskType.PLANNING

    def run(self, state: dict) -> dict:
        request = state.get("request", "")
        repo_path = state.get("repo_path", "")
        working_context = state.get("working_context") or {"files_discovered": [], "notes": []}

        context_note = self._render_working_context(working_context)
        user_prompt = (
            (context_note + "\n\n" if context_note else "")
            + f"User request:\n{request}\n\n"
            f"Repository path: {repo_path}\n\n"
            "Inspect this repository using your tools, then produce the final "
            "plan JSON described in your instructions."
        )

        final_text, tool_transcript = self._tool_loop(SYSTEM_PROMPT, user_prompt)

        plan = self._extract_json(final_text)
        if not plan or "goal" not in plan or "steps" not in plan:
            plan = self._fallback_plan(request, final_text)

        transcript_entry: dict[str, Any] = {
            "role": "assistant",
            "content": f"[planner] Produced plan: {plan.get('goal', '')}",
            "name": self.name,
        }

        return {
            "plan": plan,
            "messages": state.get("messages", []) + [transcript_entry],
            "working_context": self._update_working_context(working_context, tool_transcript),
        }

    def _fallback_plan(self, request: str, raw_text: str) -> dict:
        """Minimal one-step plan used when the model's JSON couldn't be parsed."""
        description = raw_text.strip() or request.strip() or "Address the user's request."
        return {
            "goal": request.strip() or "Address the user's request.",
            "steps": [
                {
                    "id": 1,
                    "description": description,
                    "rationale": "Fallback plan: planner output could not be parsed as JSON.",
                }
            ],
            "context_notes": "This plan was auto-generated as a fallback because the "
            "planner's model response did not contain valid, parseable JSON.",
        }


def build_planner_node(router: Any, tools: Any, memory: Any):
    """LangGraph node factory for the Planner agent."""

    def node(state: dict) -> dict:
        return PlannerAgent(router, tools, memory).run(state)

    return node
