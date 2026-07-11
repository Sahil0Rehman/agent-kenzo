"""Structured long-term project memory: a queryable upgrade sitting alongside
the free-text `.kenzo/memory.md` that `RepositoryMemory.append_memory`/
`read_memory` already own.

`ArchitectureMemory` is a NEW, separate, additive artifact persisted to
`RepositoryMemory(repo_root).architecture_memory_path` (a JSON file). It never
reads or writes memory.md -- that free-text log and this structured store are
independent, parallel ways of recording what an agent has learned about a
repo. Ranking uses the same plain keyword-overlap approach as
`kenzo.core.context_manager._tokenize`, deliberately no embeddings here
either: this module is about STRUCTURE (typed fields, queryable by keyword),
not semantic search -- that's `kenzo.memory.vector`'s job.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from kenzo.core.context_manager import _tokenize
from kenzo.memory.repository import RepositoryMemory


class MemoryEntry(BaseModel):
    """One structured fact an agent has learned about a repository.

    `timestamp` is a caller-supplied ISO string -- this module never calls
    `datetime.now()` itself, keeping `ArchitectureMemory` deterministic and
    testable; the caller (whoever observed the event) stamps the time.
    """

    kind: str  # "bug_fix" | "decision" | "pattern" | "module_note"
    issue: str = ""
    root_cause: str = ""
    files: list[str] = Field(default_factory=list)
    solution: str = ""
    notes: str = ""
    timestamp: str = ""


class ArchitectureMemory:
    """Read/write facade for a repository's structured memory entries.

    Persists eagerly: `record()` appends to `.all_entries()` in-memory AND
    rewrites the JSON file immediately, so a crash right after `record()`
    returns never loses the entry (mirrors `RepositoryMemory.append_memory`'s
    write-through-on-every-call discipline, not a buffered/flush-on-close
    one).
    """

    def __init__(self, repo_root: str) -> None:
        self._memory = RepositoryMemory(repo_root)
        self._entries: list[MemoryEntry] = self._load()

    def _load(self) -> list[MemoryEntry]:
        path = self._memory.architecture_memory_path
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []

        entries: list[MemoryEntry] = []
        for item in raw:
            try:
                entries.append(MemoryEntry.model_validate(item))
            except Exception:  # noqa: BLE001 - one malformed entry must not sink the whole store
                continue
        return entries

    def _persist(self) -> None:
        path = self._memory.architecture_memory_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [entry.model_dump() for entry in self._entries]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        self._persist()

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def find_related(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Rank recorded entries against `query` via keyword-overlap, same
        tokenization as `kenzo.core.context_manager._tokenize`.

        Scores each entry against the concatenation of `issue`, `root_cause`,
        `solution`, `notes`, and its `files` (space-joined). Zero-scoring
        entries are excluded. Ties are broken by recency, most-recently
        -recorded first (i.e. later entries in `self._entries` win ties).
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[int, int, MemoryEntry]] = []
        for seq, entry in enumerate(self._entries):
            haystack = " ".join(
                [entry.issue, entry.root_cause, entry.solution, entry.notes, " ".join(entry.files)]
            ).lower()
            score = sum(1 for tok in query_tokens if tok in haystack)
            if score > 0:
                scored.append((score, seq, entry))

        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [entry for _, _, entry in scored[:top_k]]
