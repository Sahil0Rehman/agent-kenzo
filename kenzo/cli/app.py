"""Assembles the `kenzo` Typer application.

Entry point referenced by `pyproject.toml`'s `[project.scripts]`:
`kenzo = "kenzo.cli.app:main"`.
"""

from __future__ import annotations

from typing import List, Optional

import typer

from kenzo.cli.commands import benchmark, chat, config_cmd, image, improve, init, rollback, server
from kenzo.cli.commands import index as index_cmd
from kenzo.cli.commands import session

app = typer.Typer(
    name="kenzo",
    help="Agent Kenzo: a local, repository-aware AI coding agent.",
    add_completion=False,
)

app.add_typer(config_cmd.app, name="config")
app.command("init")(init.run)
app.command("index")(index_cmd.run)
app.command("chat")(chat.run)
app.command("rollback")(rollback.run)
app.command("server")(server.run)
app.command("image")(image.run)
app.add_typer(improve.app, name="improve")
app.add_typer(benchmark.app, name="benchmark")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    repo: str = typer.Option(".", "--repo", "-r", help="Repository to work on (writable)."),
    see: Optional[List[str]] = typer.Option(
        None,
        "--see",
        help=(
            "Another repo Kenzo can READ/search for context but never edit, e.g. "
            "--see ../legacy-branch/MyService or --see myalias=../legacy-branch/MyService. "
            "Repeatable."
        ),
    ),
    also_edit: Optional[List[str]] = typer.Option(
        None,
        "--also-edit",
        help="Another repo Kenzo may also EDIT this run (e.g. porting a fix into a second branch). Repeatable.",
    ),
) -> None:
    """Launch the interactive Kenzo session when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        session.start_interactive_session(repo_path=repo, see=see, also_edit=also_edit)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
