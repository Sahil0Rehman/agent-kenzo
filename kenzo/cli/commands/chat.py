"""`kenzo chat` -- a lightweight, repository-aware chat REPL.

This is deliberately NOT the full autonomous pipeline: it talks directly to
the `ModelRouter` with no `AgentRuntime`/graph involved, so there is no
planner -> researcher -> coder -> tester -> reviewer sequence here, just
quick Q&A with the repository's `.kenzo/` context bundle as the system
prompt. For the full pipeline, use the bare `kenzo` interactive session.
"""

from __future__ import annotations

import typer

from kenzo.cli.ui.console import console
from kenzo.config.settings import load_config
from kenzo.core.schemas import TaskType
from kenzo.memory.repository import RepositoryMemory
from kenzo.models.router import ModelRouter


def run(
    repo: str = typer.Option(".", "--repo", "-r", help="Repository to use as context"),
) -> None:
    """Start a lightweight, direct chat REPL grounded in the repository's context."""
    config = load_config()
    router = ModelRouter(config)
    memory = RepositoryMemory(repo)
    memory.ensure_kenzo_dir()

    system_prompt = memory.build_context_bundle()
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    console.print("[kenzo.title]Kenzo Chat[/kenzo.title] -- lightweight repo-aware Q&A. Type 'exit' to quit.")

    while True:
        try:
            user_input = console.input("[bold cyan]you>[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            console.print()
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": stripped})

        try:
            response = router.complete(TaskType.GENERAL, messages)
            reply = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - surface any provider/config error to the user
            console.print(f"[kenzo.failure]Error: {exc}[/kenzo.failure]")
            messages.pop()  # don't let a failed turn pollute the running context
            continue

        console.print(f"[bold green]kenzo>[/bold green] {reply}")
        messages.append({"role": "assistant", "content": reply})
