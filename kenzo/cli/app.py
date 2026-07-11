"""Assembles the `kenzo` Typer application.

Entry point referenced by `pyproject.toml`'s `[project.scripts]`:
`kenzo = "kenzo.cli.app:main"`.
"""

from __future__ import annotations

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
def main_callback(ctx: typer.Context) -> None:
    """Launch the interactive Kenzo session when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        session.start_interactive_session()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
