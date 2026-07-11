"""Single shared Rich console instance used across the entire CLI layer.

Every other CLI module imports `console` from here rather than constructing
its own `rich.console.Console`, so output stays consistent (theme, width
detection, etc.) no matter which command or session path is running.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

_KENZO_THEME = Theme(
    {
        "kenzo.title": "bold cyan",
        "kenzo.tagline": "italic dim",
        "kenzo.event": "dim",
        "kenzo.success": "bold green",
        "kenzo.failure": "bold red",
        "kenzo.warning": "bold yellow",
        "kenzo.heading": "bold magenta",
    }
)

console = Console(theme=_KENZO_THEME)
