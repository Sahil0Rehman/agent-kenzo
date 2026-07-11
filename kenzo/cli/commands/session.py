"""The bare `kenzo` interactive session -- drives every request through the
Phase 3 dynamic Agent Controller loop (`AgentRuntime.run_autonomous`), which
decides its own sequence of plan/search/read/analyze/edit/test/review steps
rather than following a fixed pipeline order. Unlike `kenzo chat`, this never
shortcuts straight to an answer -- and unlike the older fixed pipeline
(still available as `AgentRuntime.run`, unchanged), it isn't limited to one
pass through each stage.
"""

from __future__ import annotations

from rich.panel import Panel

from kenzo.cli.ui.components import banner, render_controller_result, render_event
from kenzo.cli.ui.console import console
from kenzo.core.runtime import AgentRuntime


def start_interactive_session(repo_path: str = ".") -> None:
    """Run the interactive Kenzo session against `repo_path` until the user exits."""
    banner()

    runtime = AgentRuntime(repo_path, on_event=render_event)

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
