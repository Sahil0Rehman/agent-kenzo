"""`kenzo server` -- run the local FastAPI + WebSocket agent server (Phase 7),
the backend the `vscode-extension/` package (or any other WebSocket client
speaking `kenzo.server.protocol`) connects to.
"""

from __future__ import annotations

import typer

from kenzo.cli.ui.console import console
from kenzo.config.settings import load_config


def run(
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to."),
    port: int = typer.Option(8756, help="Port to bind the server to."),
) -> None:
    """Start the Kenzo agent server (FastAPI + WebSocket) for IDE clients."""
    import uvicorn

    from kenzo.server.app import create_app

    config = load_config()
    app = create_app(config)

    console.print(f"[bold cyan]Kenzo server[/bold cyan] listening on ws://{host}:{port}/ws")
    console.print("Connect the Kenzo VS Code extension, or any client speaking kenzo.server.protocol.")

    uvicorn.run(app, host=host, port=port, log_level="info")
