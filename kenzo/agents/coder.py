"""Coder agent: applies the plan/research by actually writing code to disk."""

from __future__ import annotations

from typing import Any

from kenzo.agents.base import BaseAgent
from kenzo.core.schemas import TaskType

SYSTEM_PROMPT = """\
You are the Coding agent for Kenzo, a local repository-aware coding agent.

You are given a plan and research findings from earlier agents. Your job is
to actually implement the described changes in the repository -- you MUST
use the write_file and edit_file tools to make real changes on disk. Never
just describe changes in prose; if a change needs to happen, make it happen
via a tool call.

Before editing any file, use list_directory and/or read_file to see its
current contents so your edits are precise and don't clobber unrelated code.

If review feedback from a previous attempt is provided (a list of issues to
fix), prioritize fixing those issues first.

Work through the plan step by step, making all necessary write_file/edit_file
calls. When you are done making changes, respond with a final short natural
language summary of what you changed and why (this is NOT required to be
JSON -- plain text is fine here).
"""


class CoderAgent(BaseAgent):
    name = "coder"
    task_type = TaskType.CODING

    def run(self, state: dict) -> dict:
        plan = state.get("plan") or {}
        research = state.get("research") or {}
        review = state.get("review") or {}
        issues = review.get("issues") or []
        working_context = state.get("working_context") or {"files_discovered": [], "notes": []}

        prompt_parts = []
        context_note = self._render_working_context(working_context)
        if context_note:
            prompt_parts.append(context_note)
        prompt_parts.append(f"Plan:\n{plan}")
        prompt_parts.append(f"Research findings:\n{research}")
        if issues:
            prompt_parts.append(
                "The previous attempt was reviewed and REJECTED. You MUST fix "
                f"these issues:\n{issues}"
            )
        prompt_parts.append(
            "Implement the plan now using write_file/edit_file (reading current "
            "file contents first where relevant), then summarize your changes."
        )
        user_prompt = "\n\n".join(prompt_parts)

        final_text, tool_transcript = self._tool_loop(SYSTEM_PROMPT, user_prompt)

        changes = self._changes_from_transcript(tool_transcript)

        transcript_entry: dict[str, Any] = {
            "role": "assistant",
            "content": final_text or "[coder] Completed changes.",
            "name": self.name,
        }

        return {
            "changes": state.get("changes", []) + changes,
            "iteration": state.get("iteration", 0) + 1,
            "messages": state.get("messages", []) + [transcript_entry],
            "working_context": self._update_working_context(working_context, tool_transcript),
        }

    def _changes_from_transcript(self, tool_transcript: list[dict]) -> list[dict]:
        changes: list[dict] = []
        for entry in tool_transcript:
            tool_name = entry.get("tool")
            if tool_name not in ("write_file", "edit_file"):
                continue
            result = entry.get("result") or {}
            if not result.get("success"):
                continue

            data = result.get("data") or {}
            arguments = entry.get("arguments") or {}
            action = "create" if tool_name == "write_file" else "edit"
            changes.append(
                {
                    "file_path": arguments.get("path", data.get("path", "")),
                    "action": action,
                    "diff": data.get("diff", ""),
                    "content": arguments.get("content") if action == "create" else None,
                    "applied": True,
                }
            )
        return changes


def build_coder_node(router: Any, tools: Any, memory: Any):
    """LangGraph node factory for the Coder agent."""

    def node(state: dict) -> dict:
        return CoderAgent(router, tools, memory).run(state)

    return node
