"""`.kenzo/improvements/<timestamp>.json` -- one file per `kenzo improve`
run, and `kenzo improve history`'s data source. Mirrors
`kenzo.core.checkpoint.CheckpointManager`'s "one JSON file per record under
a `.kenzo/` subdirectory" convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class ImprovementRecord:
    """One `kenzo improve` run's full outcome -- problem found, solution
    attempted, files touched, test results before/after, and whether an
    automatic rollback happened."""

    timestamp: str
    problem: str
    solution: str = ""
    files_changed: list[str] = field(default_factory=list)
    tests_before: dict[str, Any] = field(default_factory=dict)
    tests_after: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    rolled_back: bool = False
    rollback_reason: str = ""
    checkpoint_ref: Optional[str] = None
    task_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "problem": self.problem,
            "solution": self.solution,
            "files_changed": self.files_changed,
            "tests_before": self.tests_before,
            "tests_after": self.tests_after,
            "success": self.success,
            "rolled_back": self.rolled_back,
            "rollback_reason": self.rollback_reason,
            "checkpoint_ref": self.checkpoint_ref,
            "task_id": self.task_id,
        }


class ImprovementHistory:
    """One instance per repository, mirroring `CheckpointManager`'s
    per-repo-root construction convention."""

    def __init__(self, repo_root: str) -> None:
        self.dir = Path(repo_root) / ".kenzo" / "improvements"

    def record(self, entry: ImprovementRecord) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self.dir / f"{stamp}.json"
        path.write_text(json.dumps(entry.to_dict(), indent=2), encoding="utf-8")
        return path

    def list_all(self) -> list[dict[str, Any]]:
        """Every recorded improvement, oldest first (filename sort order --
        the timestamp-prefixed name IS the chronological key)."""
        if not self.dir.is_dir():
            return []
        entries: list[dict[str, Any]] = []
        for file_path in sorted(self.dir.glob("*.json")):
            try:
                entries.append(json.loads(file_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return entries
