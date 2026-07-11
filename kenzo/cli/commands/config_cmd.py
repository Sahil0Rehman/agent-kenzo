"""`kenzo config` -- inspect and edit `~/.kenzo/config.yaml`."""

from __future__ import annotations

from typing import Any

import typer
import yaml
from rich.syntax import Syntax

from kenzo.cli.ui.console import console
from kenzo.config.settings import config_path, load_config, set_value

app = typer.Typer(name="config", help="Manage Kenzo configuration")


def _mask_secret(value: str) -> str:
    if not value:
        return value
    return f"{value[:4]}..."


def _masked_config_dict() -> dict[str, Any]:
    config = load_config()
    data = config.model_dump(mode="json")
    for provider in (data.get("providers") or {}).values():
        if isinstance(provider, dict) and provider.get("api_key"):
            provider["api_key"] = _mask_secret(provider["api_key"])
    return data


@app.command("show")
def show() -> None:
    """Print the resolved config with provider API keys masked."""
    data = _masked_config_dict()
    yaml_text = yaml.safe_dump(data, sort_keys=False)
    console.print(Syntax(yaml_text, "yaml", theme="ansi_dark", word_wrap=True))


@app.command("path")
def path() -> None:
    """Print the path to the global config file."""
    console.print(str(config_path()))


@app.command("set")
def set_cmd(
    key: str = typer.Argument(..., help="Dotted config key, e.g. routing.coding"),
    value: str = typer.Argument(..., help="New value for the key"),
) -> None:
    """Set a single dotted config key and confirm the new resolved value."""
    updated = set_value(key, value)
    data = updated.model_dump(mode="json")

    cursor: Any = data
    for part in key.split("."):
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
        else:
            cursor = None
            break

    if isinstance(cursor, str) and "api_key" in key:
        cursor = _mask_secret(cursor)

    console.print(f"[kenzo.success]Set[/kenzo.success] {key} = {cursor}")
