"""Conversation-level memory across multiple tasks within ONE running Kenzo
process/session.

`SessionMemory` sits at a different altitude than everything else touching
"memory" in this codebase:

- `kenzo.core.context_manager.ContextManager` is per-TASK -- built fresh (or
  reset) for every new task, never outliving it.
- `kenzo.memory.architecture_memory.ArchitectureMemory` is cross-SESSION --
  it survives process restarts and accumulates durable, structured project
  knowledge indefinitely.
- `SessionMemory` is in between: it remembers what happened across the last
  few tasks *within this running session*, so a later, vaguer request in the
  SAME session (e.g. "fix the same issue in admin login", after an earlier
  task fixed a login bug elsewhere) can be answered by surfacing that recent
  history back into the model's prompt. Kenzo does not resolve "the same
  issue" itself -- exactly like `kenzo.core.graph_context`'s graph-derived
  facts, this module hands the model short plain-text context and lets it
  infer the reference.

Persistence is best-effort: `SessionMemory` is fully usable purely in-memory,
but also mirrors itself to `RepositoryMemory(repo_root).session_memory_path`
so a Kenzo process restarted shortly after (crash, or the CLI re-invoked) can
still recall recent session context. Any read/write failure degrades to "no
session memory" rather than raising -- this module must never be the reason
a task fails.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from kenzo.core.context_manager import _tokenize
from kenzo.memory.importance import is_worth_remembering
from kenzo.memory.repository import RepositoryMemory


class SessionEntry(BaseModel):
    """One task's request + outcome, as recalled later in the same session.

    `timestamp` is caller-supplied (ISO string) -- this module never calls
    `datetime.now()` itself, matching `ArchitectureMemory.MemoryEntry`'s
    determinism convention.
    """

    request: str
    task_id: str = ""
    succeeded: bool = True
    summary: str = ""
    files_discovered: list[str] = Field(default_factory=list)
    timestamp: str = ""


class SessionMemory:
    """Read/write facade for one session's recent task history.

    Hygiene vs. "always have something recent" tradeoff (see `record()`):
    every recorded entry is checked against
    `kenzo.memory.importance.is_worth_remembering` before being kept
    permanently. An entry that fails that check is still appended -- so
    `render_recent()`/`find_related()` are never empty right after a real
    task just ran -- but it only ever occupies a single trailing "fallback
    slot": the NEXT call to `record()` evicts it before appending anything
    new, whether or not the new entry itself passes the gate. An entry that
    DOES pass the gate is never evicted this way -- it is only ever trimmed
    later by the plain `max_entries` cap (oldest first, once genuinely full).
    Net effect: a long run of trivial/failed noise is pinned to at most one
    throwaway slot instead of growing the persisted file without bound,
    while at least one recent entry is always available to render.

    `record()` calls `is_worth_remembering(kind="task_outcome", ...)`.
    `"task_outcome"` is one of `kenzo.memory.importance`'s recognized
    success-sensitive high-value kinds (scores 0.85 when `succeeded=True`,
    0.3 when `succeeded=False` -- see that module), so a genuinely successful
    task outcome clears `IMPORTANCE_THRESHOLD` and is kept permanently; the
    fallback-slot mechanism above only ever ends up doing real work for
    failed/low-value outcomes, exactly as intended.
    """

    def __init__(self, repo_root: str, max_entries: int = 50) -> None:
        self.repo_root = repo_root
        self.max_entries = max_entries
        self._memory = RepositoryMemory(repo_root)
        self._entries: list[SessionEntry] = self._load()
        # True if the current last element of `_entries` is present ONLY
        # because of the "always keep the most recent" carve-out (i.e. it
        # failed the hygiene gate on its own) -- see class docstring. Reset
        # to False on load: entries already on disk are treated as settled
        # history from a prior process, not provisional within this one.
        self._last_is_fallback_only = False

    def _load(self) -> list[SessionEntry]:
        path = self._memory.session_memory_path
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []

        entries: list[SessionEntry] = []
        for item in raw:
            try:
                entries.append(SessionEntry.model_validate(item))
            except Exception:  # noqa: BLE001 - one malformed entry must not sink the whole store
                continue
        return entries[-self.max_entries :]

    def _persist(self) -> None:
        path = self._memory.session_memory_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = [entry.model_dump() for entry in self._entries]
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass  # best-effort: an unwritable .kenzo/ dir must not fail the caller's task

    def record(self, entry: SessionEntry) -> None:
        """Persist `entry`, subject to the hygiene-vs-recency tradeoff
        documented on the class.

        Always evaluates `is_worth_remembering(kind="task_outcome",
        content=entry.request + " " + entry.summary, succeeded=entry.succeeded)`
        per this module's spec (see class docstring for the caveat this
        currently implies in practice).
        """
        worth = is_worth_remembering(
            kind="task_outcome",
            content=f"{entry.request} {entry.summary}",
            succeeded=entry.succeeded,
        )

        if self._last_is_fallback_only and self._entries:
            self._entries.pop()

        self._entries.append(entry)
        self._last_is_fallback_only = not worth

        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

        self._persist()

    def render_recent(self, top_k: int = 3) -> str:
        """Render the most recent `top_k` entries as a short text block
        suitable for prepending to a prompt. Returns "" if there are no
        entries at all (or `top_k <= 0`)."""
        if top_k <= 0 or not self._entries:
            return ""

        recent = list(reversed(self._entries[-top_k:]))
        lines: list[str] = []
        for entry in recent:
            text = entry.summary.strip() or entry.request.strip()
            if entry.files_discovered:
                text += f" (files: {', '.join(entry.files_discovered)})"
            lines.append(f"- {text}")
        return "Recent session activity:\n" + "\n".join(lines)

    def find_related(self, query: str, top_k: int = 3) -> list[SessionEntry]:
        """Keyword-overlap ranking over `request`+`summary`+`files_discovered`,
        same tokenization convention as `kenzo.core.context_manager._tokenize`
        (lowercase alphanumeric, length > 2, stopword-filtered). Most relevant
        first; ties broken by recency (most-recently-recorded first)."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[int, int, SessionEntry]] = []
        for seq, entry in enumerate(self._entries):
            haystack = " ".join(
                [entry.request, entry.summary, " ".join(entry.files_discovered)]
            ).lower()
            score = sum(1 for tok in query_tokens if tok in haystack)
            if score > 0:
                scored.append((score, seq, entry))

        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [entry for _, _, entry in scored[: top_k]]
