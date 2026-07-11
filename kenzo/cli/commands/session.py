"""The bare `kenzo` interactive session -- drives every request through the
Phase 3 dynamic Agent Controller loop (`AgentRuntime.run_autonomous`), which
decides its own sequence of plan/search/read/analyze/edit/test/review steps
rather than following a fixed pipeline order. Unlike `kenzo chat`, this never
shortcuts straight to an answer -- and unlike the older fixed pipeline
(still available as `AgentRuntime.run`, unchanged), it isn't limited to one
pass through each stage.
"""

from __future__ import annotations

from typing import Optional

from rich.panel import Panel

from kenzo.cli.ui.components import banner, render_controller_result, render_event
from kenzo.cli.ui.console import console
from kenzo.core.runtime import AgentRuntime
from kenzo.core.workspace import WorkspaceError, build_workspace


def _render_plan_for_approval(task_state, rationale: str) -> bool:
    """Phase 12's CLI plan-approval renderer: print the plan Rich-formatted
    and block on `console.input()` for a yes/no -- a plain synchronous call
    is enough here (unlike VS Code's edit-approval, which needs a
    thread+queue bridge because it's driven from an async WebSocket handler)
    since the bare `kenzo` session is already single-threaded and blocking
    end to end."""
    steps = task_state.remaining_work or ["(no explicit steps -- see rationale)"]
    steps_text = "\n".join(f"  {i}. {step}" for i, step in enumerate(steps, start=1))
    console.print(
        Panel(
            f"Goal: {task_state.goal}\n\nRationale: {rationale}\n\nSteps:\n{steps_text}",
            title="Proposed Plan -- Approve?",
            border_style="cyan",
        )
    )
    answer = console.input("[bold cyan]Proceed with this plan? [Y/n]:[/bold cyan] ")
    return answer.strip().lower() not in ("n", "no")


def start_interactive_session(
    repo_path: str = ".",
    see: Optional[list[str]] = None,
    also_edit: Optional[list[str]] = None,
    plan_first: bool = False,
) -> None:
    """Run the interactive Kenzo session against `repo_path` until the user
    exits. `see`/`also_edit` (Phase 11), if given, put Kenzo in multi-repo
    workspace mode -- see `kenzo.core.workspace`. `plan_first` (Phase 12),
    if set, pauses for your approval after every PLAN step before Kenzo
    acts on it -- see `AgentController.set_plan_approval_handler`."""
    banner()

    # Phase 11: a Workspace is ALWAYS constructed (even with zero --see/
    # --also-edit) so mid-session discovery (add_reference_repo) is always
    # available -- you don't have to know up front that a request will
    # mention another repo's path; Kenzo can register one the moment your
    # own message names it. --see/--also-edit just pre-populate it for
    # repos you already know you'll need (and are the ONLY way to grant
    # write access -- add_reference_repo is always read-only).
    try:
        workspace = build_workspace(repo_path, see=see, also_edit=also_edit)
    except WorkspaceError as exc:
        console.print(Panel(str(exc), title="Kenzo Error", border_style="red"))
        return
    if workspace.has_any_other_repo():
        console.print(
            f"[dim]Workspace: {len(workspace.others)} other repo(s) -- "
            f"read-only: {[r.alias for r in workspace.others if not r.writable]}, "
            f"writable: {workspace.writable_aliases()}[/dim]"
        )

    runtime = AgentRuntime(repo_path, on_event=render_event, workspace=workspace)
    if plan_first:
        runtime.controller.set_plan_approval_handler(_render_plan_for_approval)
        console.print("[dim]--plan-first: Kenzo will pause for your approval after every plan.[/dim]")

    try:
        while True:
            try:
                request = console.input("[bold cyan]kenzo>[/bold cyan] ")
            except (KeyboardInterrupt, EOFError):
                console.print()
                break

            stripped = request.strip()
            if not stripped:
                continue
            if stripped.lower() in ("exit", "quit"):
                break

            try:
                result = runtime.run_autonomous(stripped)
            except Exception as exc:  # noqa: BLE001 - last-resort net; run_autonomous() itself never raises
                console.print(
                    Panel(
                        str(exc),
                        title="Kenzo Error",
                        border_style="red",
                    )
                )
                continue

            render_controller_result(result)
    finally:
        runtime.close()
