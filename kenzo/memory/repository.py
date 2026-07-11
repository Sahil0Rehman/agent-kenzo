"""Per-repository memory: the `.kenzo/` folder living inside a TARGET repo
that Kenzo is operating on (architecture.md, memory.md, rules.md, index.json).

`RepositoryMemory` is the read/write facade every agent node uses to load
durable context about the repo it's working in, and to persist new learnings
back to disk between runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from kenzo.config.defaults import (
    DEFAULT_ARCHITECTURE_MD,
    DEFAULT_MEMORY_MD,
    DEFAULT_RULES_MD,
)

_MEMORY_TAIL_CHARS = 4000


class RepositoryMemory:
    """Owns the `.kenzo/` directory inside a target repository."""

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.kenzo_dir = self.repo_root / ".kenzo"

    @property
    def _architecture_path(self) -> Path:
        return self.kenzo_dir / "architecture.md"

    @property
    def _memory_path(self) -> Path:
        return self.kenzo_dir / "memory.md"

    @property
    def _rules_path(self) -> Path:
        return self.kenzo_dir / "rules.md"

    @property
    def _index_path(self) -> Path:
        return self.kenzo_dir / "index.json"

    @property
    def _symbols_path(self) -> Path:
        return self.kenzo_dir / "symbols.json"

    @property
    def _project_path(self) -> Path:
        return self.kenzo_dir / "project.json"

    # -- Phase 4 code intelligence artifacts ----------------------------
    # These are plain path conventions only -- kenzo.intelligence.symbol_index
    # .SymbolIndex / .reference_graph.ReferenceGraph / .dependency_graph
    # .DependencyGraph each already implement their own save()/load()
    # (de)serialization; RepositoryMemory just hands back where those files
    # canonically live inside `.kenzo/`, exactly like it does for
    # index.json/symbols.json/project.json above.

    @property
    def code_index_path(self) -> Path:
        return self.kenzo_dir / "code_index.json"

    @property
    def references_path(self) -> Path:
        return self.kenzo_dir / "references.json"

    @property
    def dependencies_path(self) -> Path:
        return self.kenzo_dir / "dependencies.json"

    # -- Phase 5 semantic memory artifacts -------------------------------
    # Same plain-path-convention role as the Phase 4 properties above.

    @property
    def architecture_memory_path(self) -> Path:
        return self.kenzo_dir / "architecture_memory.json"

    @property
    def session_memory_path(self) -> Path:
        return self.kenzo_dir / "session_memory.json"

    @property
    def semantic_chunk_ids_path(self) -> Path:
        return self.kenzo_dir / "semantic_chunk_ids.json"

    @property
    def vector_collection_name(self) -> str:
        """A collection name unique per repository, so one shared Qdrant
        server (persistent `url`, not the default `:memory:` mode) can safely
        hold multiple indexed repositories without their chunks colliding."""
        import hashlib

        digest = hashlib.sha1(str(self.repo_root).encode("utf-8")).hexdigest()[:12]
        return f"kenzo_{digest}"

    def ensure_kenzo_dir(self) -> Path:
        """Create `.kenzo/` and seed its default files if they don't exist yet.

        Never overwrites a file that already exists — user/agent edits to
        architecture.md, memory.md, or rules.md are always preserved.
        """
        self.kenzo_dir.mkdir(parents=True, exist_ok=True)

        defaults = (
            (self._architecture_path, DEFAULT_ARCHITECTURE_MD),
            (self._memory_path, DEFAULT_MEMORY_MD),
            (self._rules_path, DEFAULT_RULES_MD),
        )
        for path, default_content in defaults:
            if not path.exists():
                path.write_text(default_content, encoding="utf-8")

        return self.kenzo_dir

    def read_architecture(self) -> str:
        return self._read_text(self._architecture_path)

    def read_memory(self) -> str:
        return self._read_text(self._memory_path)

    def read_rules(self) -> str:
        return self._read_text(self._rules_path)

    def _read_text(self, path: Path) -> str:
        if not self.kenzo_dir.exists():
            self.ensure_kenzo_dir()
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def append_memory(self, entry: str, heading: str = "") -> None:
        """Append a new dated section to memory.md.

        `heading`, if given, is rendered as a `###` subheading above the
        entry text; otherwise the timestamp alone introduces the entry.
        """
        self.ensure_kenzo_dir()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        section_lines = [f"\n## {timestamp}"]
        if heading:
            section_lines.append(f"### {heading}")
        section_lines.append(entry.rstrip())
        section_lines.append("")  # trailing newline

        with self._memory_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(section_lines))

    def save_index(self, index: dict[str, Any]) -> None:
        self.ensure_kenzo_dir()
        self._index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def load_index(self) -> dict[str, Any] | None:
        if not self._index_path.exists():
            return None
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def save_symbols(self, symbols: dict[str, Any]) -> None:
        """Persist the Java analyzer's output (see kenzo.memory.java_analyzer)."""
        self.ensure_kenzo_dir()
        self._symbols_path.write_text(json.dumps(symbols, indent=2), encoding="utf-8")

    def load_symbols(self) -> dict[str, Any] | None:
        if not self._symbols_path.exists():
            return None
        try:
            return json.loads(self._symbols_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def save_project_info(self, project_info: dict[str, Any]) -> None:
        """Persist `kenzo.core.project.ProjectAnalyzer`'s result."""
        self.ensure_kenzo_dir()
        self._project_path.write_text(json.dumps(project_info, indent=2), encoding="utf-8")

    def load_project_info(self) -> dict[str, Any] | None:
        if not self._project_path.exists():
            return None
        try:
            return json.loads(self._project_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def build_context_bundle(self) -> str:
        """Concatenate project info/rules/architecture/memory (+ index
        summary, if any) into the single context string every agent node
        prepends to prompts.
        """
        rules = self.read_rules().strip()
        architecture = self.read_architecture().strip()
        memory = self.read_memory().strip()

        if len(memory) > _MEMORY_TAIL_CHARS:
            memory = "...(earlier entries truncated)...\n" + memory[-_MEMORY_TAIL_CHARS:]

        sections: list[str] = []

        project_info = self.load_project_info()
        if project_info:
            project_lines = [
                f"Language: {project_info.get('language', 'unknown')}",
                f"Framework: {project_info.get('framework') or '(none detected)'}",
                f"Build system: {project_info.get('build') or '(none detected)'}",
                f"Test command: {project_info.get('test_command') or '(unknown -- ask before guessing)'}",
            ]
            sections.append("# Project\n\n" + "\n".join(project_lines))

        sections.append("# Repository Rules\n\n" + rules)
        sections.append("# Repository Architecture\n\n" + architecture)
        sections.append("# Repository Memory\n\n" + memory)

        index = self.load_index()
        if index:
            languages = index.get("languages", {}) or {}
            top_langs = sorted(languages.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_langs_str = ", ".join(f"{ext} ({count})" for ext, count in top_langs) or "(none)"
            indexed_files = index.get("indexed_files", index.get("file_count", 0))
            total_files = index.get("total_files", indexed_files)
            index_lines = [
                f"Indexed files: {indexed_files}",
                f"Top languages: {top_langs_str}",
                f"Summary: {index.get('summary', '')}",
            ]
            if index.get("truncated"):
                index_lines.insert(
                    0,
                    "WARNING:\nRepository index incomplete.\n"
                    f"Indexed {indexed_files} of {total_files} files.\n"
                    "Files outside this subset were NOT indexed and may still be relevant --"
                    " use search_files/repository_tree to look beyond the index if the answer"
                    " isn't in it.",
                )
            sections.append("# Repository Index\n\n" + "\n".join(index_lines))

        symbols = self.load_symbols()
        if symbols and symbols.get("symbols"):
            symbol_count = len(symbols["symbols"])
            sections.append(
                "# Java Symbols\n\n"
                f"{symbol_count} class(es)/interface(s) indexed. Use find_symbol(name) to look "
                "one up, or map_flow(endpoint) to trace an HTTP endpoint through its "
                "controller/service/repository chain, instead of grepping for class names."
            )

        return "\n\n".join(sections).strip() + "\n"
