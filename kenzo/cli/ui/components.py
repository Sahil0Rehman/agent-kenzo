"""Rich rendering helpers shared by the interactive session and CLI commands.

Every renderer here takes plain dicts (the JSON-shaped values straight out of
`KenzoState`) rather than pydantic model instances, and handles `None`/empty
input gracefully -- e.g. `plan` is `None` until the planner node has run.

The one exception is `render_event`, which renders a live
`kenzo.core.events.AgentEvent` instance streamed through `on_event` while the
graph is executing.
"""

from __future__ import annotations

from typing import Any, Optional

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from kenzo.core.events import AgentEvent
from kenzo.cli.ui.console import console


def banner() -> None:
    """Print the Agent Kenzo startup banner."""
    body = Text("Agent Kenzo", style="kenzo.title", justify="center")
    console.print(
        Panel(
            body,
            subtitle="[kenzo.tagline]Your local, repository-aware coding agent[/kenzo.tagline]",
            expand=False,
            border_style="cyan",
        )
    )


def render_event(event: AgentEvent) -> None:
    """Print a dim status line for a single agent-pipeline progress event."""
    console.print(f"[kenzo.event]• [{event.stage}] {event.message}[/kenzo.event]")


def render_plan(plan: Optional[dict]) -> None:
    """Render the plan's goal plus a numbered table of its steps.

    `plan` is `None` before the planner node has run.
    """
    if not plan:
        console.print("[kenzo.event]No plan yet.[/kenzo.event]")
        return

    goal = plan.get("goal", "")
    console.print(Panel(goal or "(no goal set)", title="Goal", border_style="magenta"))

    steps = plan.get("steps") or []
    if not steps:
        console.print("[kenzo.event]Plan has no steps.[/kenzo.event]")
        return

    table = Table(title="Plan Steps", show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Description")
    table.add_column("Rationale")
    table.add_column("Status", justify="center")

    for step in steps:
        status = str(step.get("status", "pending"))
        status_style = {
            "done": "kenzo.success",
            "failed": "kenzo.failure",
            "in_progress": "kenzo.warning",
        }.get(status, "dim")
        table.add_row(
            str(step.get("id", "")),
            step.get("description", ""),
            step.get("rationale", "") or "-",
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(table)


def render_changes(changes: Optional[list[dict]]) -> None:
    """Render a table of proposed/applied file changes, followed by the real
    unified diff for each change that has one -- not just "file edited".
    """
    changes = changes or []
    if not changes:
        console.print("[kenzo.event]No file changes.[/kenzo.event]")
        return

    table = Table(title="Changes")
    table.add_column("File Path")
    table.add_column("Action", justify="center")
    table.add_column("Applied", justify="center")

    for change in changes:
        applied = bool(change.get("applied", False))
        applied_str = "[kenzo.success]yes[/kenzo.success]" if applied else "[kenzo.warning]no[/kenzo.warning]"
        table.add_row(
            change.get("file_path", ""),
            change.get("action", ""),
            applied_str,
        )

    console.print(table)

    for change in changes:
        diff_text = (change.get("diff") or "").strip()
        if not diff_text:
            continue
        console.print(
            Panel(
                Syntax(diff_text, "diff", background_color="default", word_wrap=True),
                title=f"Diff: {change.get('file_path', '')}",
                border_style="cyan",
            )
        )


def render_test_result(test_result: Optional[dict]) -> None:
    """Render pass/fail plus a summary for the most recent test run."""
    if not test_result:
        console.print("[kenzo.event]No test results yet.[/kenzo.event]")
        return

    passed = bool(test_result.get("passed", False))
    style = "kenzo.success" if passed else "kenzo.failure"
    label = "PASSED" if passed else "FAILED"
    command = test_result.get("command", "")
    summary = test_result.get("summary", "")

    lines = [f"[{style}]{label}[/{style}]"]
    if command:
        lines.append(f"Command: {command}")
    if summary:
        lines.append(summary)

    console.print(Panel("\n".join(lines), title="Test Result", border_style="green" if passed else "red"))


def render_review(review: Optional[dict]) -> None:
    """Render approved/not-approved plus summary and any issues raised."""
    if not review:
        console.print("[kenzo.event]No review yet.[/kenzo.event]")
        return

    approved = bool(review.get("approved", False))
    style = "kenzo.success" if approved else "kenzo.failure"
    label = "APPROVED" if approved else "NOT APPROVED"
    summary = review.get("summary", "")
    issues = review.get("issues") or []

    lines = [f"[{style}]{label}[/{style}]"]
    if summary:
        lines.append(summary)
    if issues:
        lines.append("")
        lines.append("Issues:")
        for issue in issues:
            lines.append(f"  • {issue}")

    console.print(Panel("\n".join(lines), title="Review", border_style="green" if approved else "red"))


def render_controller_result(result: dict) -> None:
    """Render the result of `AgentController.run` (the Phase 3 dynamic
    action loop) -- a differently-shaped result than `KenzoState`: no single
    `plan`/`test_result`/`review`, but `changes`, a `test_results` history,
    a `decisions` log, and `remaining_work`.
    """
    iteration = result.get("iteration", 0)
    console.print(f"[kenzo.event]Kenzo took {iteration} step(s) to reach this point.[/kenzo.event]")

    decisions = result.get("decisions") or []
    if decisions:
        table = Table(title="Reasoning Steps")
        table.add_column("#", justify="right", style="bold")
        table.add_column("Step")
        for i, decision in enumerate(decisions, start=1):
            table.add_row(str(i), decision)
        console.print(table)

    render_changes(result.get("changes", []))

    test_results = result.get("test_results") or []
    if test_results:
        render_test_result(test_results[-1])

    remaining_work = result.get("remaining_work") or []
    if remaining_work:
        lines = ["Remaining work Kenzo did not finish:"]
        lines.extend(f"  • {item}" for item in remaining_work)
        console.print(Panel("\n".join(lines), title="Remaining Work", border_style="yellow"))

    if result.get("needs_user_input"):
        console.print(
            Panel(
                result.get("final_response") or "Kenzo needs more information to continue.",
                title="Kenzo Needs Your Input",
                border_style="yellow",
            )
        )
        return

    success = result.get("success")
    render_final(result.get("final_response") or "Kenzo finished without producing a final response.")
    if not success and test_results:
        console.print(
            "[kenzo.warning]Note: the most recent test result did not pass -- review the changes above "
            "before trusting this as complete.[/kenzo.warning]"
        )


def render_crash_warning(state: dict) -> None:
    """Render a prominent warning that the agent pipeline stopped mid-run due
    to an unhandled error, and -- critically -- which files (if any) it had
    already modified before failing. Those changes are NOT reverted.
    """
    error = state.get("error") or "(no error message captured)"
    completed = state.get("completed_steps") or []
    changed_files = [c.get("file_path", "") for c in (state.get("changes") or []) if c.get("file_path")]

    lines = [f"Error: {error}"]
    if completed:
        lines.append(f"Completed steps before failure: {', '.join(completed)}")

    if changed_files:
        lines.append("")
        lines.append("[kenzo.warning]Agent stopped after modifying files:[/kenzo.warning]")
        for path in changed_files:
            lines.append(f"  • {path}")
        lines.append("")
        lines.append(
            "These changes were NOT reverted and remain on disk. Review them "
            "(e.g. with your own `git diff`) before trusting or committing them."
        )
    else:
        lines.append("")
        lines.append("No files were modified before the failure.")

    console.print(Panel("\n".join(lines), title="Kenzo Stopped Unexpectedly", border_style="red"))


def render_final(final_response: Optional[str]) -> None:
    """Print the final agent response in a highlighted panel."""
    text = final_response or "(no final response)"
    console.print(Panel(text, title="Kenzo", border_style="cyan", padding=(1, 2)))
