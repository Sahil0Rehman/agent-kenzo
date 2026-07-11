"""Researcher agent: gathers concrete evidence from the repo for each plan step."""

from __future__ import annotations

from typing import Any

from kenzo.agents.base import BaseAgent
from kenzo.core.schemas import TaskType

SYSTEM_PROMPT = """\
You are the Research agent for Kenzo, a local repository-aware coding agent.

You are given a plan (a goal plus a list of steps) that a Coder agent will
later implement. Your job is to gather concrete evidence from the ACTUAL
repository that will make the Coder's job easier and more accurate.

Use the available tools (search_files, read_file, list_directory, and the
git status/history tools) to investigate the repository for EACH step of the
plan: find the relevant files, read the code that will need to change, and
note anything the Coder needs to know (existing patterns, related code,
naming conventions, potential pitfalls, open questions that still need
clarification).

Once you have gathered enough evidence, respond with your final answer. Your
final answer MUST be ONLY a single JSON object (no prose, no markdown fences)
matching this exact shape:

{
  "summary": "<concise summary of findings relevant to the plan>",
  "relevant_files": ["<path1>", "<path2>"],
  "snippets": {"<path>": "<short excerpt from that file>"},
  "open_questions": ["<anything still unclear>"]
}
"""


class ResearcherAgent(BaseAgent):
    name = "researcher"
    task_type = TaskType.RESEARCH

    def run(self, state: dict) -> dict:
        plan = state.get("plan") or {}
        repo_path = state.get("repo_path", "")
        working_context = state.get("working_context") or {"files_discovered": [], "notes": []}

        context_note = self._render_working_context(working_context)
        user_prompt = (
            (context_note + "\n\n" if context_note else "")
            + f"Repository path: {repo_path}\n\n"
            f"Plan to research:\n{plan}\n\n"
            "Investigate the repository using your tools for each step above, "
            "then produce the final research JSON described in your instructions."
        )

        final_text, tool_transcript = self._tool_loop(SYSTEM_PROMPT, user_prompt)

        research = self._extract_json(final_text)
        if not research or "summary" not in research:
            research = self._fallback_research(final_text)

        transcript_entry: dict[str, Any] = {
            "role": "assistant",
            "content": f"[researcher] {research.get('summary', '')}",
            "name": self.name,
        }

        return {
            "research": research,
            "messages": state.get("messages", []) + [transcript_entry],
            "working_context": self._update_working_context(working_context, tool_transcript),
        }

    def _fallback_research(self, raw_text: str) -> dict:
        """Minimal finding used when the model's JSON couldn't be parsed."""
        summary = raw_text.strip() or "No structured research findings were produced."
        return {
            "summary": summary,
            "relevant_files": [],
            "snippets": {},
            "open_questions": [
                "Researcher output could not be parsed as JSON; raw text was used as a fallback summary."
            ],
        }


def build_researcher_node(router: Any, tools: Any, memory: Any):
    """LangGraph node factory for the Researcher agent."""

    def node(state: dict) -> dict:
        return ResearcherAgent(router, tools, memory).run(state)

    return node
