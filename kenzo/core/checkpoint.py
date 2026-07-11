"""Checkpoint + rollback: a safety net so an agent run that edits files can
always be undone.

This closes a gap flagged repeatedly across earlier phases' audits: Kenzo
could write real files to disk with no way to revert them if a run crashed,
got rejected, or simply went wrong. Two backends:

- Git repos: `git stash create` captures a snapshot commit-ish of the
  current tracked-file state with ZERO side effects (it does not touch the
  working tree, the index, or the stash list) -- a safe, non-destructive way
  to remember "everything as of right now" regardless of whether the tree
  was clean or already had unrelated uncommitted changes. Rollback restores
  tracked files via `git checkout <ref> -- .` and additionally deletes any
  files the run itself created (tracked separately, since `git stash
  create` does not capture untracked files).
- Non-git repos: a file-content snapshot, captured incrementally the first
  time EACH file is touched this run (not all at once -- we don't know in
  advance which files a dynamic agent loop will decide to edit), stored
  under `.kenzo/checkpoints/<checkpoint_id>/`.

Either way the resulting checkpoint record is persisted to
`.kenzo/checkpoints/latest.json` so `kenzo rollback` can find it without the
caller needing to remember a checkpoint id.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Creates (lazily, before the first edit) and restores checkpoints for
    one repository."""

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.checkpoints_dir = self.repo_root / ".kenzo" / "checkpoints"
        self._active: dict[str, Any] | None = None

    # -- git helpers ---------------------------------------------------

    def _is_git_repo(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _git_stash_create(self) -> str | None:
        """A commit-ish hash capturing the current tracked-file state, with
        zero side effects (does not touch the working tree, index, or stash
        list -- unlike `git stash push`). Returns None if there's nothing to
        snapshot this way (e.g. no commits yet) or git errors out."""
        try:
            result = subprocess.run(
                ["git", "stash", "create"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            stash_hash = result.stdout.strip()
            return stash_hash or None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def _git_head_hash(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    # -- lifecycle -------------------------------------------------------

    def has_active_checkpoint(self) -> bool:
        return self._active is not None

    def reset(self) -> None:
        """Clear the in-memory "already checkpointed this task" flag so the
        NEXT `create_checkpoint()` call creates a fresh checkpoint instead of
        returning a stale one.

        Callers that reuse one `CheckpointManager` across multiple separate
        tasks (e.g. `AgentController` is reused across turns of one
        interactive session, for `ContextManager` continuity) must call this
        at the start of each new task. Without it, `create_checkpoint`'s
        idempotency -- correct WITHIN one task, so a task's own retries don't
        each stomp the checkpoint -- would incorrectly persist ACROSS tasks:
        only the very first task in the session would ever get a real
        pre-edit snapshot, and `kenzo rollback` would silently revert every
        subsequent task's edits too, not just the most recent one.
        """
        self._active = None

    def create_checkpoint(self, reason: str = "") -> dict:
        """Create a checkpoint capturing repo state before Kenzo's first
        edit. Idempotent within a run: only the first call actually creates
        anything; later calls return the already-active checkpoint.
        """
        if self._active is not None:
            return self._active

        created_at = datetime.now().isoformat()
        if self._is_git_repo():
            ref = self._git_stash_create() or self._git_head_hash()
            record: dict[str, Any] = {
                "method": "git",
                "git_ref": ref,
                "created_at": created_at,
                "reason": reason,
                "created_files": [],
            }
        else:
            checkpoint_id = "checkpoint-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            checkpoint_dir = self.checkpoints_dir / checkpoint_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "method": "files",
                "checkpoint_dir": str(checkpoint_dir),
                "created_at": created_at,
                "reason": reason,
                "files": {},
                "created_files": [],
            }

        self._active = record
        self._save_latest(record)
        return record

    def before_edit(self, path: str) -> None:
        """Call BEFORE a write_file/edit_file call actually executes for
        `path`. Snapshots the file's pre-edit content on its first touch
        this run (non-git backend only -- a single `git stash create` at
        checkpoint time already covers every tracked file) and records
        brand-new paths (both backends) so rollback can delete them.
        No-op if no checkpoint is active yet (create_checkpoint() must be
        called first -- callers should ensure that).
        """
        if self._active is None:
            return

        resolved = self.repo_root / path
        is_new = not resolved.exists()

        if self._active["method"] == "files" and path not in self._active["files"]:
            if is_new:
                self._active["files"][path] = None
            else:
                checkpoint_dir = Path(self._active["checkpoint_dir"])
                snapshot_name = path.replace("/", "__").replace("\\", "__")
                try:
                    shutil.copy2(resolved, checkpoint_dir / snapshot_name)
                    self._active["files"][path] = snapshot_name
                except OSError:
                    self._active["files"][path] = None

        if is_new and path not in self._active["created_files"]:
            self._active["created_files"].append(path)

        self._save_latest(self._active)

    def _save_latest(self, record: dict) -> None:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoints_dir / "latest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    def load_latest(self) -> dict | None:
        path = self.checkpoints_dir / "latest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # -- rollback ------------------------------------------------------

    def rollback(self, record: dict | None = None) -> dict:
        """Restore the repository to the state captured by `record` (or the
        persisted latest checkpoint). Returns
        {"success": bool, "message": str, "restored_files": [...], "deleted_files": [...]}.
        """
        record = record or self.load_latest()
        if record is None:
            return {
                "success": False,
                "message": "No checkpoint found to roll back to.",
                "restored_files": [],
                "deleted_files": [],
            }

        if record.get("method") == "git":
            return self._rollback_git(record)
        return self._rollback_files(record)

    def _rollback_git(self, record: dict) -> dict:
        ref = record.get("git_ref")
        if not ref:
            return {
                "success": False,
                "message": "Checkpoint has no git reference to restore.",
                "restored_files": [],
                "deleted_files": [],
            }
        try:
            checkout = subprocess.run(
                ["git", "checkout", ref, "--", "."],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "success": False,
                "message": f"git checkout failed: {exc}",
                "restored_files": [],
                "deleted_files": [],
            }

        if checkout.returncode != 0:
            return {
                "success": False,
                "message": f"git checkout reported an error: {checkout.stderr.strip()}",
                "restored_files": [],
                "deleted_files": [],
            }

        deleted: list[str] = []
        for rel_path in record.get("created_files", []):
            target = self.repo_root / rel_path
            if target.exists():
                try:
                    target.unlink()
                    deleted.append(rel_path)
                except OSError:
                    pass

        return {
            "success": True,
            "message": f"Restored tracked files to checkpoint {ref[:8]} and removed {len(deleted)} newly created file(s).",
            "restored_files": [],
            "deleted_files": deleted,
        }

    def _rollback_files(self, record: dict) -> dict:
        checkpoint_dir = Path(record.get("checkpoint_dir", ""))
        restored: list[str] = []
        deleted: list[str] = []
        for rel_path, snapshot_name in (record.get("files") or {}).items():
            target = self.repo_root / rel_path
            if snapshot_name is None:
                if target.exists():
                    try:
                        target.unlink()
                        deleted.append(rel_path)
                    except OSError:
                        pass
                continue
            snapshot_path = checkpoint_dir / snapshot_name
            if snapshot_path.exists():
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(snapshot_path, target)
                    restored.append(rel_path)
                except OSError:
                    pass

        return {
            "success": True,
            "message": f"Restored {len(restored)} file(s) and removed {len(deleted)} newly created file(s) from checkpoint.",
            "restored_files": restored,
            "deleted_files": deleted,
        }


class CheckpointingToolRegistry:
    """Wraps a tool registry so `write_file`/`edit_file` calls snapshot the
    target file (via `CheckpointManager.before_edit`) immediately before
    they execute -- lazily creating the checkpoint itself on the very first
    mutating call, wherever in the dynamic action sequence that happens to
    occur. This is what makes `kenzo rollback` possible for the
    AgentController's non-linear execution order.
    """

    _MUTATING = {"write_file", "edit_file"}

    def __init__(self, wrapped: Any, checkpoint: CheckpointManager) -> None:
        self._wrapped = wrapped
        self._checkpoint = checkpoint

    def execute(self, name: str, arguments: dict) -> Any:
        if name in self._MUTATING:
            path = arguments.get("path")
            if path:
                if not self._checkpoint.has_active_checkpoint():
                    self._checkpoint.create_checkpoint(reason=f"before first edit ({name} {path})")
                self._checkpoint.before_edit(path)
        return self._wrapped.execute(name, arguments)

    def to_openai_tools(self) -> list[dict]:
        return self._wrapped.to_openai_tools()

    def names(self) -> list[str]:
        return self._wrapped.names()

    def get(self, name: str):
        return self._wrapped.get(name)

    def list_specs(self):
        return self._wrapped.list_specs()

    def register(self, spec) -> None:
        self._wrapped.register(spec)

    def register_many(self, specs) -> None:
        self._wrapped.register_many(specs)

    def subset(self, names: list[str]) -> "CheckpointingToolRegistry":
        return CheckpointingToolRegistry(self._wrapped.subset(names), self._checkpoint)
