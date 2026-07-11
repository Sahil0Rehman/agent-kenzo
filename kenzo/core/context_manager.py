"""Context manager: keeps a long-running `AgentController` loop (up to 50+
iterations) from blowing up its prompt budget.

Stores only compact facts about what has already been discovered -- file
summaries (never full file contents), small symbol-fact dicts, and short
free-text notes -- and supports pulling back a relevance-filtered SUBSET of
that state instead of always dumping everything into the prompt.

Deliberately no embeddings / vector store: `retrieve_relevant` is a plain
keyword/substring-overlap heuristic over the stored text, nothing else.
"""

from __future__ import annotations

import json
import re
from typing import Any

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "to",
    "of",
    "in",
    "for",
    "on",
    "and",
    "or",
    "at",
    "by",
    "with",
    "as",
    "it",
    "be",
    "this",
    "that",
    "are",
    "was",
    "were",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization, dropping short/stopword tokens."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


class ContextManager:
    """Compact, retrieval-friendly memory for a single `AgentController` run.

    Holds three kinds of state:
      - `files_seen`: path -> short summary (never full file contents). This
        is also the enforcement mechanism for "never re-read the same file"
        -- callers check `has_seen_file()` before issuing a real read.
      - `symbols_seen`: symbol/class name -> a small opaque fact blob about it
        (whatever the caller passes in -- a dict or a string summary).
      - `notes`: free-text short observations, append-only but subject to
        compression (oldest dropped first once over budget).

    `files_seen` and `symbols_seen` represent durable structural discoveries
    and are NEVER dropped by `compress_context()`, no matter how large the
    run gets -- only `notes` is ever trimmed.
    """

    def __init__(self, max_notes: int = 200, max_total_chars: int = 20000) -> None:
        self.max_notes = max_notes
        self.max_total_chars = max_total_chars
        self.files_seen: dict[str, str] = {}
        self.symbols_seen: dict[str, Any] = {}
        self.notes: list[str] = []

        # Internal-only recency bookkeeping (not part of the public schema,
        # not serialized). Plain dict/list insertion order is not enough on
        # its own: overwriting an existing files_seen/symbols_seen key does
        # NOT move it to the end of the dict, so we track a monotonically
        # increasing sequence number per entry to get correct
        # "most-recently-touched-first" tie-breaking in retrieve_relevant.
        self._seq_counter = 0
        self._file_order: dict[str, int] = {}
        self._symbol_order: dict[str, int] = {}
        self._note_order: list[int] = []

    def _next_seq(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add_context(self, kind: str, key: str, content: Any) -> None:
        """Generic entry point for recording a discovery.

        - kind == "file": store/overwrite `files_seen[key]` -- re-adding the
          same path UPDATES its summary rather than duplicating.
        - kind == "symbol": store/overwrite `symbols_seen[key]` verbatim
          (content may be a dict-like structure or a string -- accepted as-is,
          never force-parsed).
        - anything else (including "note"): append `content` to `notes`, then
          run compression.

        Caller-supplied strings are assumed already short; this method never
        truncates an individual entry's content itself -- only
        `compress_context()` manages overall growth.
        """
        if kind == "file":
            self.files_seen[key] = content
            self._file_order[key] = self._next_seq()
        elif kind == "symbol":
            self.symbols_seen[key] = content
            self._symbol_order[key] = self._next_seq()
        else:
            self.notes.append(content)
            self._note_order.append(self._next_seq())
            self.compress_context()

    def has_seen_file(self, path: str) -> bool:
        """True if `path` already has a stored summary.

        Callers should check this BEFORE issuing a real read_file and reuse
        `get_file_summary()` instead of re-reading an already-seen file.
        """
        return path in self.files_seen

    def get_file_summary(self, path: str) -> str | None:
        return self.files_seen.get(path)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------
    def compress_context(self) -> None:
        """Enforce notes-count and total-character budgets.

        Only `notes` is ever trimmed (oldest first). `files_seen` and
        `symbols_seen` are durable discoveries and are never removed here,
        regardless of how far over budget the run gets.
        """
        if len(self.notes) > self.max_notes:
            overflow = len(self.notes) - self.max_notes
            self.notes = self.notes[overflow:]
            self._note_order = self._note_order[overflow:]

        while self.notes and self._total_size() > self.max_total_chars:
            self.notes.pop(0)
            self._note_order.pop(0)

    def _total_size(self) -> int:
        return len(
            json.dumps(
                {
                    "files_seen": self.files_seen,
                    "symbols_seen": self.symbols_seen,
                    "notes": self.notes,
                },
                default=str,
            )
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_relevant(self, query: str, top_k: int = 8) -> list[str]:
        """Rank stored items against `query` via keyword/substring overlap.

        No embeddings, no vector DB -- just counting how many of the query's
        (lowercased, stopword-filtered) tokens appear in each rendered item's
        text. Zero-scoring items are excluded entirely. Ties are broken by
        recency (most-recently-added first).

        If `query` is empty, or nothing scores above zero, falls back to the
        most recent `min(top_k, len(notes))` notes so this never mysteriously
        returns nothing when there IS context available -- except when the
        query is non-empty and genuinely matches nothing, in which case an
        empty list is the correct (non-padded) answer.
        """
        items = self._rendered_items()  # list of (seq, text), any order

        query_tokens = _tokenize(query)
        if not query_tokens:
            return self._recent_notes(top_k)

        scored: list[tuple[int, int, str]] = []
        for seq, text in items:
            lowered = text.lower()
            score = sum(1 for tok in query_tokens if tok in lowered)
            if score > 0:
                scored.append((score, seq, text))

        if not scored:
            return []

        # Highest score first; among equal scores, most-recently-touched
        # first (higher sequence number = more recent add/update).
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [text for _, _, text in scored[:top_k]]

    def _recent_notes(self, top_k: int) -> list[str]:
        n = min(top_k, len(self.notes))
        if n <= 0:
            return []
        return list(reversed(self.notes[-n:]))

    def _rendered_items(self) -> list[tuple[int, str]]:
        """All stored items rendered as (recency_seq, text) pairs.

        The sequence number reflects true chronological add/update order
        across files_seen, symbols_seen, and notes alike (a plain dict/list
        position would not: overwriting an existing files_seen/symbols_seen
        key updates it in place without moving it to the end).
        """
        items: list[tuple[int, str]] = []
        for path, summary in self.files_seen.items():
            seq = self._file_order.get(path, 0)
            items.append((seq, f"FILE {path}: {summary}"))
        for key, content in self.symbols_seen.items():
            seq = self._symbol_order.get(key, 0)
            items.append((seq, f"SYMBOL {key}: {content}"))
        for i, note in enumerate(self.notes):
            seq = self._note_order[i] if i < len(self._note_order) else 0
            items.append((seq, note))
        return items

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_all(self) -> str:
        """Render everything currently stored into one readable text block."""
        sections: list[str] = []

        files_lines = [f"- {path}: {summary}" for path, summary in self.files_seen.items()]
        sections.append(
            "Files already examined:\n" + ("\n".join(files_lines) if files_lines else "(none)")
        )

        symbol_lines = [f"- {key}: {content}" for key, content in self.symbols_seen.items()]
        sections.append(
            "Symbols already examined:\n"
            + ("\n".join(symbol_lines) if symbol_lines else "(none)")
        )

        note_lines = [f"- {note}" for note in self.notes]
        sections.append(
            "Observations:\n" + ("\n".join(note_lines) if note_lines else "(none)")
        )

        return "\n\n".join(sections)

    def render_relevant(self, query: str, top_k: int = 8) -> str:
        """Convenience wrapper around `retrieve_relevant` for prompt-building."""
        relevant = self.retrieve_relevant(query, top_k)
        if not relevant:
            return ""
        lines = [f"- {item}" for item in relevant]
        return "Relevant context so far:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Full serialization of durable state. Internal recency-sequence
        bookkeeping is deliberately not part of this schema -- it is
        rebuilt (as a same-order approximation) in `from_dict`."""
        return {
            "max_notes": self.max_notes,
            "max_total_chars": self.max_total_chars,
            "files_seen": dict(self.files_seen),
            "symbols_seen": dict(self.symbols_seen),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextManager":
        instance = cls(
            max_notes=data.get("max_notes", 200),
            max_total_chars=data.get("max_total_chars", 20000),
        )
        instance.files_seen = dict(data.get("files_seen", {}))
        instance.symbols_seen = dict(data.get("symbols_seen", {}))
        instance.notes = list(data.get("notes", []))

        # Rebuild recency-sequence bookkeeping in stored order (files, then
        # symbols, then notes) so retrieve_relevant's tie-breaking keeps
        # working post-reload, even though this ordering isn't itself
        # persisted.
        for path in instance.files_seen:
            instance._file_order[path] = instance._next_seq()
        for key in instance.symbols_seen:
            instance._symbol_order[key] = instance._next_seq()
        for _ in instance.notes:
            instance._note_order.append(instance._next_seq())

        return instance
