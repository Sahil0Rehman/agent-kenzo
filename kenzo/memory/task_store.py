"""Task store: persistence for `AgentController`'s `TaskState` to
`.kenzo/tasks/<task_id>.json` inside the TARGET repo Kenzo is operating on.

Mirrors the read/write/graceful-None pattern established by
`kenzo.memory.repository.RepositoryMemory` (see `load_index`/`load_symbols`),
so a long-running task's progress (steps completed, files touched, reasoning
summary) survives process restarts and can be inspected -- or resumed --
between iterations of the controller loop.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

_MAX_TASK_ID_LEN = 100
_MAX_SLUG_LEN = 40
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")
_SLUG_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


class TaskStore:
    """Owns the `.kenzo/tasks/` directory inside a target repository."""

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.tasks_dir = self.repo_root / ".kenzo" / "tasks"

    @staticmethod
    def _sanitize_task_id(task_id: str) -> str:
        """Make a caller-controlled task id safe to use as a filename.

        Replaces any character that isn't alphanumeric, "-", or "_" with "_"
        (so path separators, "..", etc. can never escape `tasks_dir`) and
        truncates to `_MAX_TASK_ID_LEN` characters. Applied identically in
        `save` and `load` so a given raw `task_id` always maps to the same
        file.
        """
        sanitized = _SANITIZE_RE.sub("_", task_id)
        sanitized = sanitized[:_MAX_TASK_ID_LEN]
        if not sanitized:
            sanitized = "_"
        return sanitized

    def _path_for(self, task_id: str) -> Path:
        return self.tasks_dir / f"{self._sanitize_task_id(task_id)}.json"

    def save(self, task_id: str, task_state: dict) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(task_id)
        path.write_text(json.dumps(task_state, indent=2), encoding="utf-8")

    def load(self, task_id: str) -> dict | None:
        path = self._path_for(task_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list_tasks(self) -> list[dict[str, Any]]:
        if not self.tasks_dir.exists():
            return []

        summaries: list[dict[str, Any]] = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            summaries.append(
                {
                    "task_id": path.stem,
                    "goal": data.get("goal", ""),
                    "finished": data.get("finished", None),
                    "iteration": data.get("iteration", None),
                }
            )
        return summaries

    def delete(self, task_id: str) -> bool:
        path = self._path_for(task_id)
        try:
            path.unlink()
            return True
        except OSError:
            return False

    def generate_task_id(self, seed_text: str) -> str:
        """Derive a reasonably unique, filesystem-safe task id from
        `seed_text` (e.g. the user's request string) plus a short uuid4
        suffix, so repeated similar requests in the same repo don't collide.
        """
        slug = _SLUG_SANITIZE_RE.sub("-", seed_text.strip().lower()).strip("-")
        slug = slug[:_MAX_SLUG_LEN].strip("-")
        if not slug:
            slug = "task"
        uuid_suffix = uuid.uuid4().hex[:8]
        return f"{slug}-{uuid_suffix}"
