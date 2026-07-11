"""`kenzo improve` -- Phase 10 self-improvement entry point.

    kenzo improve --analyze     analysis only, no edits (item 1's spec)
    kenzo improve ["request"]   the full workflow: analyze -> pick a
                                 weakness (or use the given request) -> one
                                 real AgentController.run() -> regression
                                 guard -> automatic rollback if it regressed
    kenzo improve history       read back `.kenzo/improvements/*.json`
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from kenzo.cli.ui.components import render_controller_result, render_event
from kenzo.cli.ui.console import console
from kenzo.config.settings import load_config
from kenzo.core.runtime import AgentRuntime
from kenzo.improve.self_improvement_agent import SelfImprovementAgent
from kenzo.improve.self_repo import is_self_repo

app = typer.Typer(help="Kenzo self-improvement: analyze and safely improve a codebase (including its own).")


def _build_runtime(repo: str) -> AgentRuntime:
    config = load_config()
    runtime = AgentRuntime(repo, config=config, on_event=render_event)
    # kenzo improve needs a DETERMINISTIC, fully-current index before
    # analysis (not whatever the constructor's background build happens to
    # have finished by now) -- background=False mirrors how `kenzo index`
    # already does a synchronous rebuild for the same reason.
    runtime.indexing_service.start(background=False)
    return runtime


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    repo: str = typer.Option(".", "--repo", "-r", help="Repository to analyze/improve."),
    analyze: bool = typer.Option(False, "--analyze", help="Analysis only -- print an improvement plan, make NO edits."),
    request: Optional[str] = typer.Option(
        None,
        "--request",
        "-m",
        help="Optional specific improvement to make; omitted means Kenzo picks the top finding itself.",
    ),
) -> None:
    # `request` is deliberately an OPTION, not a positional argument: `improve`
    # has a `history` subcommand, and Click/Typer resolve a positional
    # argument on the group's own callback BEFORE considering whether the
    # first bare word is actually a subcommand name -- `kenzo improve
    # history` would silently be parsed as request="history" and run the
    # FULL improve workflow instead of the history subcommand (confirmed by
    # an actual accidental live run during Phase 10 development). An option
    # has no such ambiguity.
    if ctx.invoked_subcommand is not None:
        return

    runtime = _build_runtime(repo)
    self_targeting = is_self_repo(runtime.repo_path)
    agent = SelfImprovementAgent(runtime, force_self=self_targeting)

    console.print(
        f"[kenzo.title]Kenzo Improve[/kenzo.title] -- analyzing {runtime.repo_path}"
        + (" [bold yellow](self-improvement mode)[/bold yellow]" if self_targeting else "")
    )

    try:
        if analyze:
            analysis = agent.analyze()
            console.print(Panel(analysis["plan"], title="Improvement Plan (no edits made)", border_style="cyan"))
            return

        outcome = agent.improve(request)
    finally:
        runtime.close()

    console.print(Panel(outcome["goal"], title="Chosen Improvement", border_style="cyan"))
    render_controller_result(outcome["agent_result"])

    check = outcome["regression_check"]
    _render_regression_summary(check, outcome["rolled_back"], outcome.get("rollback_message", ""))

    console.print(f"[dim]Recorded to {outcome['history_path']}[/dim]")
    if not outcome["success"]:
        raise typer.Exit(code=1)


def _render_regression_summary(check: dict, rolled_back: bool, rollback_message: str) -> None:
    table = Table(title="Regression Guard")
    table.add_column("Command")
    table.add_column("Before")
    table.add_column("After")
    for before, after in zip(check["before"], check["after"]):
        before_label = f"{before['passed']} passed, {before['failed']} failed" if before["ran"] else f"ERROR: {before['error']}"
        after_label = f"{after['passed']} passed, {after['failed']} failed" if after["ran"] else f"ERROR: {after['error']}"
        table.add_row(before["command"], before_label, after_label)
    console.print(table)

    if check["regressed"]:
        message = f"Regression detected: {check['reason']}"
        if rolled_back:
            message += "\n\nAutomatic rollback SUCCEEDED -- the repository has been restored to its pre-change state."
        else:
            message += f"\n\nAutomatic rollback ATTEMPTED but did not fully succeed: {rollback_message or 'unknown error'}. Check the repository manually."
        console.print(Panel(message, title="Regression Guard: FAILED", border_style="red"))
    else:
        console.print(Panel("No regression detected -- the change is kept.", title="Regression Guard: PASSED", border_style="green"))


@app.command("history")
def history_cmd(
    repo: str = typer.Option(".", "--repo", "-r", help="Repository whose improvement history to show."),
    limit: int = typer.Option(20, "--limit", help="Show at most this many, most recent first."),
) -> None:
    """Show past `kenzo improve` runs recorded under `.kenzo/improvements/`."""
    from kenzo.improve.history import ImprovementHistory

    entries = ImprovementHistory(repo).list_all()
    if not entries:
        console.print("[dim]No improvement history recorded yet -- run `kenzo improve` first.[/dim]")
        return

    table = Table(title=f"Improvement History ({len(entries)} total)")
    table.add_column("Timestamp")
    table.add_column("Problem")
    table.add_column("Files Changed")
    table.add_column("Result")

    for entry in entries[-limit:][::-1]:
        if entry.get("rolled_back"):
            result = "[red]ROLLED BACK[/red]"
        elif entry.get("success"):
            result = "[green]SUCCESS[/green]"
        else:
            result = "[yellow]INCOMPLETE[/yellow]"
        table.add_row(
            entry.get("timestamp", ""),
            (entry.get("problem", "") or "")[:80],
            ", ".join(entry.get("files_changed", []) or [])[:60],
            result,
        )

    console.print(table)
