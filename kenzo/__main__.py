"""Allows `python -m kenzo` as an alternate entry point to the CLI."""

from __future__ import annotations

from kenzo.cli.app import main

if __name__ == "__main__":
    main()
