"""Reviewer agent: final quality gate over plan + changes + test results."""

from __future__ import annotations

from typing import Any

from kenzo.agents.base import BaseAgent
from kenzo.core.schemas import MAX_REVIEW_RETRIES, TaskType

SYSTEM_PROMPT = """\
You are the Reviewer agent for Kenzo, a local repository-aware coding agent.

You are given the original plan, the code changes that were applied, and the
tester's verification result. Your job is to critically review whether the
work actually satisfies the plan's goal, is correct, and is safe to ship.

You may use read_file and git diff/status/history tools to double-check the
actual state of the repository rather than trusting the summaries alone.

Once you've reviewed everything, respond with your final answer. Your final
answer MUST be ONLY a single JSON object (no prose, no markdown fences)
matching this exact shape:

{
  "approved": true or false,
  "summary": "<your overall assessment>",
  "issues": ["<specific issue that must be fixed>", "..."]
}

If you approve the work, "issues" should be an empty list. If you do not
approve, list concrete, actionable issues the Coder agent must fix.
"""


class ReviewerAgent(BaseAgent):
    name = "reviewer"
    task_type = TaskType.REVIEW

    def run(self, state: dict) -> dict:
        plan = state.get("plan") or {}
        changes = state.get("changes", [])
        test_result = state.get("test_result") or {}
        working_context = state.get("working_context") or {"files_discovered": [], "notes": []}

        context_note = self._render_working_context(working_context)
        user_prompt = (
            (context_note + "\n\n" if context_note else "")
            + f"Plan:\n{plan}\n\n"
            f"Changes applied:\n{changes}\n\n"
            f"Test result:\n{test_result}\n\n"
            "Review this work using your tools if needed, then produce the "
            "final review JSON described in your instructions."
        )

        final_text, tool_transcript = self._tool_loop(SYSTEM_PROMPT, user_prompt)

        verdict = self._extract_json(final_text)
        if not verdict or "approved" not in verdict:
            verdict = self._fallback_verdict(final_text)

        # Normalize in place: `_review_router` and the CLI renderers read
        # `state["review"]["approved"]` directly, so the stored dict -- not
        # just a local variable -- must carry a real bool.
        approved = self._coerce_bool(verdict.get("approved", False))

        # Hard gate: the reviewer LLM's own opinion is never allowed to
        # override a known test failure. Without this, a reviewer that
        # misreads or glosses over a failing test_result could approve
        # demonstrably broken work with nothing in the code stopping it.
        if test_result.get("passed") is False and approved:
            approved = False
            verdict["issues"] = list(verdict.get("issues") or []) + [
                "Approval overridden to NOT APPROVED: test_result.passed is False "
                f"(tester summary: {test_result.get('summary', '(no summary)')}). Kenzo never "
                "allows approval when tests are known to be failing, regardless of the "
                "reviewer's own judgment."
            ]

        verdict["approved"] = approved
        review_retries = state.get("review_retries", 0) + (0 if approved else 1)

        transcript_entry: dict[str, Any] = {
            "role": "assistant",
            "content": f"[reviewer] approved={approved}: {verdict.get('summary', '')}",
            "name": self.name,
        }

        result: dict[str, Any] = {
            "review": verdict,
            "review_retries": review_retries,
            "messages": state.get("messages", []) + [transcript_entry],
            "working_context": self._update_working_context(working_context, tool_transcript),
        }

        if approved or review_retries >= MAX_REVIEW_RETRIES:
            result["final_response"] = self._build_final_response(
                plan, changes, test_result, verdict
            )

        return result

    def _fallback_verdict(self, raw_text: str) -> dict:
        """Conservative fallback used when the model's JSON couldn't be parsed."""
        summary = raw_text.strip() or "Reviewer output could not be parsed as JSON."
        return {
            "approved": False,
            "summary": summary,
            "issues": [
                "Reviewer output could not be parsed as JSON; treating as not approved."
            ],
        }

    def _build_final_response(
        self,
        plan: dict,
        changes: list[dict],
        test_result: dict,
        verdict: dict,
    ) -> str:
        goal = plan.get("goal", "(no goal recorded)")
        changed_files = [c.get("file_path", "") for c in changes if c.get("file_path")]
        files_line = ", ".join(changed_files) if changed_files else "(no files changed)"
        test_summary = test_result.get("summary", "(no test summary)")
        review_summary = verdict.get("summary", "(no review summary)")

        lines = [
            f"Goal: {goal}",
            f"Files changed: {files_line}",
            f"Tests: {test_summary}",
            f"Review: {review_summary}",
        ]
        return "\n".join(lines)


def build_reviewer_node(router: Any, tools: Any, memory: Any):
    """LangGraph node factory for the Reviewer agent."""

    def node(state: dict) -> dict:
        return ReviewerAgent(router, tools, memory).run(state)

    return node
