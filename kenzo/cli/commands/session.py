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


def start_interactive_session(
    repo_path: str = ".",
    see: Optional[list[str]] = None,
    also_edit: Optional[list[str]] = None,
) -> None:
    """Run the interactive Kenzo session against `repo_path` until the user
    exits. `see`/`also_edit` (Phase 11), if given, put Kenzo in multi-repo
    workspace mode -- see `kenzo.core.workspace`."""
    banner()

    workspace = None
    if see or also_edit:
        try:
            workspace = build_workspace(repo_path, see=see, also_edit=also_edit)
        except WorkspaceError as exc:
            console.print(Panel(str(exc), title="Kenzo Error", border_style="red"))
            return
        console.print(
            f"[dim]Workspace: {len(workspace.others)} other repo(s) -- "
            f"read-only: {[r.alias for r in workspace.others if not r.writable]}, "
            f"writable: {workspace.writable_aliases()}[/dim]"
        )

    runtime = AgentRuntime(repo_path, on_event=render_event, workspace=workspace)

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
