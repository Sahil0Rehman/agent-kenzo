"""`kenzo init` -- bootstrap the global config and a repository's `.kenzo/`
directory so the rest of the CLI has somewhere to read/write context from.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel

from kenzo.cli.ui.console import console
from kenzo.config.settings import ensure_config_exists
from kenzo.memory.repository import RepositoryMemory


def run(path: str = typer.Argument(".", help="Repository path to initialize")) -> None:
    """Initialize Kenzo's global config and this repository's `.kenzo/` directory."""
    repo_path = Path(path).resolve()

    config_file = ensure_config_exists()

    memory = RepositoryMemory(str(repo_path))
    kenzo_dir = memory.ensure_kenzo_dir()

    created_files = [
        str(config_file),
        str(kenzo_dir / "architecture.md"),
        str(kenzo_dir / "memory.md"),
        str(kenzo_dir / "rules.md"),
    ]

    body_lines = [f"Repository: {repo_path}", f"Kenzo directory: {kenzo_dir}", "", "Files ready:"]
    for file_path in created_files:
        body_lines.append(f"  • {file_path}")

    console.print(
        Panel(
            "\n".join(body_lines),
            title="Kenzo Initialized",
            border_style="cyan",
        )
    )
