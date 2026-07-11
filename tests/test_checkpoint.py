"""Tests for `kenzo.core.checkpoint`: `CheckpointManager` (git-backed and
file-snapshot-backed) and `CheckpointingToolRegistry`.

Two backend groups:
- git: the real tmp git repository from the `repo_root` fixture (real `git`
  subprocess calls -- fast/local/offline, no network).
- non-git: a plain tmp directory with no `.git` at all.
"""

from __future__ import annotations

from kenzo.config.settings import PermissionsConfig
from kenzo.core.checkpoint import CheckpointingToolRegistry, CheckpointManager
from kenzo.tools.registry import build_default_registry


# ---------------------------------------------------------------------------
# create_checkpoint idempotency (both backends)
# ---------------------------------------------------------------------------


def test_create_checkpoint_idempotent_within_a_run_git_backend(repo_root):
    manager = CheckpointManager(str(repo_root))

    first = manager.create_checkpoint(reason="first reason")
    second = manager.create_checkpoint(reason="second reason")

    assert first is second
    assert second["reason"] == "first reason"  # not overwritten by the 2nd call
    assert first["method"] == "git"
    assert first["git_ref"]  # a real stash-create hash (or HEAD fallback)


def test_create_checkpoint_idempotent_within_a_run_non_git_backend(tmp_path):
    root = tmp_path / "plain_repo"
    root.mkdir()
    (root / "file.txt").write_text("hello", encoding="utf-8")

    manager = CheckpointManager(str(root))
    first = manager.create_checkpoint(reason="first reason")
    second = manager.create_checkpoint(reason="second reason")

    assert first is second
    assert second["reason"] == "first reason"
    assert first["method"] == "files"

    # Only ONE checkpoint directory was ever created, not two.
    checkpoint_dirs = list((root / ".kenzo" / "checkpoints").glob("checkpoint-*"))
    assert len(checkpoint_dirs) == 1


# ---------------------------------------------------------------------------
# git backend: before_edit needs no per-file snapshot; stash-create ref alone
# is enough to correctly restore tracked file content on rollback.
# ---------------------------------------------------------------------------


def test_git_backend_rollback_restores_tracked_file_without_before_edit_snapshot(repo_root):
    original = (repo_root / "main.py").read_text(encoding="utf-8")

    manager = CheckpointManager(str(repo_root))
    manager.create_checkpoint(reason="before edit")

    # before_edit on an existing tracked file: the git backend does NOT
    # snapshot its content (only records new-file bookkeeping, which is a
    # no-op here since the file already exists).
    manager.before_edit("main.py")

    (repo_root / "main.py").write_text("def main():\n    return 999\n", encoding="utf-8")
    assert (repo_root / "main.py").read_text(encoding="utf-8") != original

    result = manager.rollback()

    assert result["success"] is True
    assert (repo_root / "main.py").read_text(encoding="utf-8") == original


def test_git_backend_rollback_deletes_newly_created_file(repo_root):
    manager = CheckpointManager(str(repo_root))
    manager.create_checkpoint()

    manager.before_edit("brand_new.py")  # doesn't exist yet -> tracked as "created"
    (repo_root / "brand_new.py").write_text("new stuff\n", encoding="utf-8")
    assert (repo_root / "brand_new.py").exists()

    result = manager.rollback()

    assert result["success"] is True
    assert not (repo_root / "brand_new.py").exists()
    assert "brand_new.py" in result["deleted_files"]


# ---------------------------------------------------------------------------
# non-git backend: before_edit snapshots ORIGINAL content on FIRST touch only
# ---------------------------------------------------------------------------


def test_non_git_backend_snapshots_original_content_on_first_touch_only(tmp_path):
    root = tmp_path / "plain_repo"
    root.mkdir()
    (root / "file.txt").write_text("original", encoding="utf-8")

    manager = CheckpointManager(str(root))
    manager.create_checkpoint()

    manager.before_edit("file.txt")  # first touch: snapshots "original"
    (root / "file.txt").write_text("edit1", encoding="utf-8")

    manager.before_edit("file.txt")  # second touch: must NOT re-snapshot "edit1"
    (root / "file.txt").write_text("edit2", encoding="utf-8")

    result = manager.rollback()

    assert result["success"] is True
    # Restored to the state BEFORE THE FIRST edit, not the intermediate "edit1".
    assert (root / "file.txt").read_text(encoding="utf-8") == "original"
    assert "file.txt" in result["restored_files"]


def test_non_git_backend_rollback_deletes_newly_created_file(tmp_path):
    root = tmp_path / "plain_repo"
    root.mkdir()
    (root / "existing.txt").write_text("keep me", encoding="utf-8")

    manager = CheckpointManager(str(root))
    manager.create_checkpoint()

    manager.before_edit("new_file.txt")  # doesn't exist yet
    (root / "new_file.txt").write_text("brand new content", encoding="utf-8")

    result = manager.rollback()

    assert result["success"] is True
    assert not (root / "new_file.txt").exists()
    assert "new_file.txt" in result["deleted_files"]
    # Untouched files survive rollback unharmed.
    assert (root / "existing.txt").read_text(encoding="utf-8") == "keep me"


# ---------------------------------------------------------------------------
# rollback with no checkpoint at all
# ---------------------------------------------------------------------------


def test_rollback_with_no_checkpoint_returns_failure_without_raising(tmp_path):
    root = tmp_path / "never_touched"
    root.mkdir()

    manager = CheckpointManager(str(root))
    result = manager.rollback()

    assert result == {
        "success": False,
        "message": "No checkpoint found to roll back to.",
        "restored_files": [],
        "deleted_files": [],
    }


def test_rollback_with_no_checkpoint_git_backend_also_fails_gracefully(repo_root):
    manager = CheckpointManager(str(repo_root))
    result = manager.rollback()

    assert result["success"] is False
    assert result["restored_files"] == []
    assert result["deleted_files"] == []


# ---------------------------------------------------------------------------
# CheckpointingToolRegistry: automatic checkpoint creation on first mutating call
# ---------------------------------------------------------------------------


def test_checkpointing_tool_registry_triggers_checkpoint_on_first_edit_and_rollback_restores(repo_root):
    permissions = PermissionsConfig(allow_write=True)
    raw_tools = build_default_registry(str(repo_root), permissions)
    checkpoint_manager = CheckpointManager(str(repo_root))
    wrapped = CheckpointingToolRegistry(raw_tools, checkpoint_manager)

    original_content = (repo_root / "main.py").read_text(encoding="utf-8")
    assert CheckpointManager(str(repo_root)).load_latest() is None

    result = wrapped.execute(
        "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 12345"}
    )
    assert result.success is True
    assert (repo_root / "main.py").read_text(encoding="utf-8") != original_content

    # A fresh CheckpointManager instance (simulating `kenzo rollback` running
    # in a separate process) can find the persisted checkpoint.
    fresh_view = CheckpointManager(str(repo_root))
    latest = fresh_view.load_latest()
    assert latest is not None
    assert latest["method"] == "git"

    rollback_result = fresh_view.rollback()
    assert rollback_result["success"] is True
    assert (repo_root / "main.py").read_text(encoding="utf-8") == original_content


def test_checkpointing_tool_registry_creates_only_one_checkpoint_across_multiple_edits(repo_root):
    permissions = PermissionsConfig(allow_write=True)
    raw_tools = build_default_registry(str(repo_root), permissions)
    checkpoint_manager = CheckpointManager(str(repo_root))
    wrapped = CheckpointingToolRegistry(raw_tools, checkpoint_manager)

    wrapped.execute("edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 1"})
    first_record = checkpoint_manager.load_latest()

    wrapped.execute("edit_file", {"path": "main.py", "old_text": "return 1", "new_text": "return 2"})
    second_record = checkpoint_manager.load_latest()

    # The checkpoint's git_ref must be the SAME across both edits -- it was
    # only ever created once, before the FIRST edit.
    assert first_record["git_ref"] == second_record["git_ref"]


def test_checkpointing_tool_registry_subset_preserves_checkpoint_behavior(repo_root):
    permissions = PermissionsConfig(allow_write=True)
    raw_tools = build_default_registry(str(repo_root), permissions)
    checkpoint_manager = CheckpointManager(str(repo_root))
    wrapped = CheckpointingToolRegistry(raw_tools, checkpoint_manager)

    edit_only = wrapped.subset(["edit_file", "read_file"])
    assert isinstance(edit_only, CheckpointingToolRegistry)
    assert set(edit_only.names()) == {"edit_file", "read_file"}

    original_content = (repo_root / "main.py").read_text(encoding="utf-8")
    result = edit_only.execute("edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 7"})
    assert result.success is True

    assert checkpoint_manager.load_latest() is not None

    rollback_result = checkpoint_manager.rollback()
    assert rollback_result["success"] is True
    assert (repo_root / "main.py").read_text(encoding="utf-8") == original_content
