"""`kenzo image <path>` -- the Phase 8 vision debugging entry point:
analyze a screenshot, then drive the SAME autonomous `AgentController` loop
(`AgentRuntime.run_autonomous`) every other Kenzo entry point uses -- vision
is a new INPUT modality feeding the existing agent, not a separate pipeline.

    Image -> Vision Model -> Structured Understanding -> AgentController

matches the Phase 8 spec's flow exactly: this command performs all four
steps in one run (analyze, then act), not just analysis alone.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.panel import Panel

from kenzo.cli.ui.components import render_controller_result, render_event
from kenzo.cli.ui.console import console
from kenzo.config.settings import load_config
from kenzo.core.runtime import AgentRuntime
from kenzo.vision.image_loader import UnsupportedImageError


def run(
    image_path: str = typer.Argument(..., help="Path to a screenshot (png/jpg/jpeg/webp)."),
    request: Optional[str] = typer.Argument(
        None, help="Optional instructions to go with the screenshot (e.g. 'this button is misaligned')."
    ),
    repo: str = typer.Option(".", "--repo", "-r", help="Repository to fix the issue in."),
) -> None:
    """Analyze a screenshot and fix the issue it shows, end to end."""
    config = load_config()
    runtime = AgentRuntime(repo, config=config, on_event=render_event)

    console.print(f"[kenzo.title]Kenzo Vision[/kenzo.title] -- analyzing {image_path}...")

    try:
        analysis = runtime.router.analyze_image(image_path, prompt=request or "")
    except UnsupportedImageError as exc:
        console.print(Panel(str(exc), title="Kenzo Error", border_style="red"))
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001 - surface any provider/config error to the user, don't crash the CLI
        console.print(Panel(f"Could not analyze image: {exc}", title="Kenzo Error", border_style="red"))
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Detected:[/bold] {analysis.image_type.value}")
    if analysis.primary_error:
        location = f" ({analysis.primary_file}:{analysis.primary_line})" if analysis.primary_file else ""
        console.print(f"[bold]Primary error:[/bold] {analysis.primary_error}{location}")
    if analysis.possible_causes:
        console.print("[bold]Possible causes:[/bold] " + "; ".join(analysis.possible_causes[:3]))

    derived_request = analysis.derive_request(request or "")
    console.print(f"\n[dim]Proceeding with:[/dim] {derived_request}\n")

    runtime.controller.set_vision_context(analysis.to_context_dict())
    try:
        result = runtime.run_autonomous(derived_request)
    finally:
        runtime.controller.set_vision_context(None)
        runtime.close()

    render_controller_result(result)
