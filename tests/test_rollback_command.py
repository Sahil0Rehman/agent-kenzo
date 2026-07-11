"""Tests for `kenzo.cli.commands.rollback.run`: the `kenzo rollback` CLI
command. Calls `run()` directly (bypassing typer's own argument-parsing
layer, which is not under test here) against a real tmp git repo with a
real `CheckpointManager`-backed checkpoint already in place.
"""

from __future__ import annotations

import typer

from kenzo.cli.commands import rollback
from kenzo.core.checkpoint import CheckpointManager


def test_rollback_command_restores_file_and_prints_success_panel(repo_root, capsys):
    original = (repo_root / "main.py").read_text(encoding="utf-8")

    manager = CheckpointManager(str(repo_root))
    manager.create_checkpoint(reason="before rollback CLI test")
    manager.before_edit("main.py")
    (repo_root / "main.py").write_text("def main():\n    return -1\n", encoding="utf-8")

    rollback.run(str(repo_root))

    assert (repo_root / "main.py").read_text(encoding="utf-8") == original

    captured = capsys.readouterr()
    assert "Rollback Complete" in captured.out


def test_rollback_command_with_no_checkpoint_exits_nonzero_and_prints_failure_panel(tmp_path, capsys):
    empty_repo = tmp_path / "no_checkpoint_repo"
    empty_repo.mkdir()

    try:
        rollback.run(str(empty_repo))
        raised = False
        exit_code = None
    except typer.Exit as exc:
        raised = True
        exit_code = exc.exit_code

    assert raised is True
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Rollback Failed" in captured.out
