"""`kenzo rollback` -- restore the repository to its state before Kenzo's
last checkpoint (created automatically before the agent's first edit)."""

from __future__ import annotations

import typer
from rich.panel import Panel

from kenzo.cli.ui.console import console
from kenzo.core.checkpoint import CheckpointManager


def run(path: str = typer.Argument(".", help="Repository path to roll back")) -> None:
    """Restore the repository to its last recorded checkpoint."""
    manager = CheckpointManager(path)
    result = manager.rollback()

    if not result["success"]:
        console.print(Panel(result["message"], title="Rollback Failed", border_style="red"))
        raise typer.Exit(code=1)

    lines = [result["message"]]
    if result.get("restored_files"):
        lines.append("")
        lines.append("Restored:")
        for f in result["restored_files"]:
            lines.append(f"  • {f}")
    if result.get("deleted_files"):
        lines.append("")
        lines.append("Removed (created after the checkpoint):")
        for f in result["deleted_files"]:
            lines.append(f"  • {f}")

    console.print(Panel("\n".join(lines), title="Rollback Complete", border_style="green"))
