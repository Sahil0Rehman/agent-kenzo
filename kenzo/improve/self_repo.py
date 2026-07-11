"""Self-repository detection, and the extra tool guard Phase 10 requires
when Kenzo is asked to improve ITS OWN source (`kenzo improve` run inside
-- or pointed at -- a checkout of Kenzo itself).

This is a SEPARATE, ADDITIONAL safety layer on top of everything Phase 3
already provides (`CheckpointingToolRegistry`, `kenzo rollback`) and Phase 7
provides (`EditApprovalToolRegistry`) -- it exists because self-modification
has one extra failure mode neither of those guards against: Kenzo could
"improve" its own memory/checkpoint/task state or a secret-bearing config
file right out from under itself, which a plain checkpoint/rollback (scoped
to CODE changes, restorable after the fact) doesn't prevent from happening
in the first place. `SelfModificationGuardRegistry` fails a write CLOSED --
refuses it outright -- rather than allowing it and relying on rollback to
undo it, since some of what it protects (a live task's own `.kenzo/tasks/`
state, mid-run) can't be meaningfully "rolled back" without corrupting the
very run that's doing the rolling back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kenzo.core.schemas import ToolResult

# Kenzo's own pyproject.toml declares this exact project name -- the
# cheapest reliable signal that a given repo_path IS a checkout of Kenzo's
# own source, not merely a project that happens to depend on it (a
# dependent project's OWN pyproject.toml would name ITSELF, not
# "agent-kenzo", in [project].name).
_SELF_PROJECT_NAME = "agent-kenzo"


def is_self_repo(repo_path: str) -> bool:
    """True if `repo_path` is a checkout (or a copy) of Kenzo's own source.

    Deliberately a cheap string check rather than a full TOML parse (no new
    dependency for one field lookup) plus a structural confirmation (the
    `kenzo/` package with `core/controller.py` actually exists alongside
    it) -- both signals must agree, so a repo that merely mentions
    "agent-kenzo" in prose somewhere doesn't false-positive.
    """
    root = Path(repo_path)
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    if f'name = "{_SELF_PROJECT_NAME}"' not in text and f"name = '{_SELF_PROJECT_NAME}'" not in text:
        return False
    return (root / "kenzo").is_dir() and (root / "kenzo" / "core" / "controller.py").is_file()


# Paths NEVER writable during self-modification, relative to repo_root,
# matched case-insensitively after normalizing to forward slashes. Prefix
# match for `.kenzo/` (Kenzo's own operational data: memory.md,
# architecture.md, checkpoints/, tasks/, improvements/ -- none of it is
# "source code" a self-improvement pass should ever be rewriting); substring
# match for anything secret-looking anywhere in the path, since a config
# file holding provider API keys could plausibly live at more than one
# location depending on how a repo is laid out.
_PROTECTED_PREFIXES = (".kenzo/",)
_PROTECTED_SUBSTRINGS = ("secret", "credential", ".env")


def is_protected_path(path: str) -> bool:
    """Would writing to `path` (relative to repo_root) touch something
    self-modification must never touch?"""
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if any(normalized.startswith(prefix) for prefix in _PROTECTED_PREFIXES):
        return True
    lowered = normalized.lower()
    return any(marker in lowered for marker in _PROTECTED_SUBSTRINGS)


class SelfModificationGuardRegistry:
    """Wraps a tool registry (mirrors `CheckpointingToolRegistry`/
    `EditApprovalToolRegistry`'s wrapping interface exactly, so it composes
    with either) so a `write_file`/`edit_file` call targeting a protected
    path (see `is_protected_path`) fails closed with a clear error instead
    of executing. Installed OUTERMOST by `SelfImprovementAgent` when
    `is_self_repo(repo_path)` is true, so it sees -- and can refuse -- every
    mutating call before checkpointing/edit-approval ever run.
    """

    _MUTATING = {"write_file", "edit_file"}

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def execute(self, name: str, arguments: dict) -> ToolResult:
        if name in self._MUTATING:
            path = arguments.get("path") or ""
            if is_protected_path(path):
                return ToolResult(
                    tool_name=name,
                    success=False,
                    error=(
                        f"Refused: '{path}' is a protected path during self-modification -- "
                        "Kenzo's own .kenzo/ operational data and anything secret/credential-"
                        "looking are never editable by `kenzo improve`."
                    ),
                )
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

    def subset(self, names: list[str]) -> "SelfModificationGuardRegistry":
        return SelfModificationGuardRegistry(self._wrapped.subset(names))
