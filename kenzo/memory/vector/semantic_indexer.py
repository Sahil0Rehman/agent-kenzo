"""`SemanticIndexer` turns an already-built `SymbolIndex` (Phase 4 -- no
re-parsing happens here) into embedded `CodeChunk`s in a `VectorStore`.

Chunking is intentional, not blind: one chunk per class, one chunk per
method, one chunk per top-level function -- NEVER a fixed-size character
slice. Each chunk's embedded text is synthesized from real structure (kind,
stereotype, annotations, endpoint, dependencies, enclosing class) plus the
real source text for that symbol's line range (which naturally includes a
Python docstring, since that's the first statement in the body) plus a
best-effort backward scan for a Java/JSDoc-style leading comment block
immediately above the declaration -- see `_leading_comment_block`'s docstring
for its honest limitation (heuristic, not real comment-AST attachment).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from kenzo.intelligence.models import ClassSymbol, FileNode, FunctionSymbol
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.memory.vector.embeddings import EmbeddingRouter
from kenzo.memory.vector.models import CodeChunk
from kenzo.memory.vector.store import VectorStore

_MAX_LEADING_COMMENT_LINES = 10
_EMBED_BATCH_SIZE = 64


def _read_lines(repo_root: Path, rel_path: str) -> list[str]:
    try:
        text = (repo_root / rel_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()


def _leading_comment_block(lines: list[str], start_line: int) -> str:
    """Best-effort scan backward from `start_line` (1-indexed, the symbol's
    own first line) for an immediately-preceding comment block (Javadoc/
    JSDoc `/** ... */`, `//` lines, or Python `#` lines).

    Heuristic only: stops at the first non-comment/non-blank line, or after
    `_MAX_LEADING_COMMENT_LINES`, or at the top of the file. Does not
    understand real comment-to-declaration attachment (e.g. an annotation
    line like `@Deprecated` between the comment and the class is NOT
    special-cased, so a comment separated from its symbol by annotations
    will not be captured) -- a real docstring inside the body (Python) is
    captured separately, via the raw source slice itself, not by this scan.
    """
    collected: list[str] = []
    idx = start_line - 2  # 0-indexed line immediately above start_line
    scanned = 0
    while idx >= 0 and scanned < _MAX_LEADING_COMMENT_LINES:
        line = lines[idx].strip()
        if not line:
            break
        if line.startswith(("//", "#", "*", "/*", "/**")) or line.endswith("*/"):
            collected.append(lines[idx])
            idx -= 1
            scanned += 1
            continue
        break
    collected.reverse()
    return "\n".join(collected).strip()


def _source_slice(lines: list[str], start_line: int, end_line: int) -> str:
    if not lines:
        return ""
    start = max(start_line - 1, 0)
    end = min(end_line, len(lines))
    return "\n".join(lines[start:end])


class SemanticIndexer:
    """Chunks + embeds + stores an already-built `SymbolIndex` into a
    `VectorStore`. Never re-parses source files itself for structure --
    Phase 4's `SymbolIndex` already did that -- it only re-reads raw source
    text (for the literal code + any leading comment) needed to render a
    chunk's embedded content.
    """

    def __init__(
        self,
        repo_root: str,
        symbol_index: SymbolIndex,
        embedding_router: EmbeddingRouter,
        vector_store: VectorStore,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.symbol_index = symbol_index
        self.embedding_router = embedding_router
        self.vector_store = vector_store
        # file -> set of chunk ids currently stored for it, so re-indexing a
        # changed file can delete chunks for symbols that no longer exist
        # (e.g. a renamed/removed method) instead of only ever adding.
        self._chunk_ids_by_file: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Chunk construction (pure, no embedding/storage side effects)
    # ------------------------------------------------------------------

    def _render_chunk_content(
        self,
        header_lines: list[str],
        lines: list[str],
        start_line: int,
        end_line: int,
    ) -> str:
        comment = _leading_comment_block(lines, start_line)
        source = _source_slice(lines, start_line, end_line)
        sections = ["\n".join(header_lines)]
        if comment:
            sections.append(comment)
        sections.append(source)
        return "\n\n".join(section for section in sections if section)

    def _class_chunk(self, file_path: str, cls: ClassSymbol, lines: list[str]) -> CodeChunk:
        header = [
            f"{cls.kind} {cls.qualified_name or cls.name}",
            f"Stereotype: {cls.stereotype or 'none'}",
            f"Annotations: {', '.join(cls.annotations) or 'none'}",
            f"Extends: {cls.extends or 'none'}",
            f"Implements: {', '.join(cls.implements) or 'none'}",
            f"Dependencies: {', '.join(cls.dependencies) or 'none'}",
            f"Endpoints: {', '.join(cls.endpoints) or 'none'}",
        ]
        content = self._render_chunk_content(header, lines, cls.location.start_line, cls.location.end_line)
        return CodeChunk(
            id=f"{file_path}::{cls.name}",
            file=file_path,
            symbol=cls.name,
            content=content,
            language="",  # filled in by caller, which knows the FileNode's language
            chunk_type="class",
            metadata={
                "stereotype": cls.stereotype,
                "kind": cls.kind,
                "dependencies": list(cls.dependencies),
                "endpoints": list(cls.endpoints),
                "start_line": cls.location.start_line,
                "end_line": cls.location.end_line,
            },
        )

    def _method_chunk(
        self, file_path: str, cls: ClassSymbol, method: FunctionSymbol, lines: list[str]
    ) -> CodeChunk:
        header = [
            f"method {cls.name}.{method.name}",
            f"Enclosing class: {cls.name} ({cls.stereotype or 'no stereotype'})",
            f"Annotations: {', '.join(method.annotations) or 'none'}",
            f"Endpoint: {method.endpoint or 'none'}",
            f"Parameters: {', '.join(p.name + ': ' + p.type if p.type else p.name for p in method.parameters) or 'none'}",
            f"Return type: {method.return_type or 'none'}",
        ]
        content = self._render_chunk_content(header, lines, method.location.start_line, method.location.end_line)
        return CodeChunk(
            id=f"{file_path}::{cls.name}.{method.name}",
            file=file_path,
            symbol=f"{cls.name}.{method.name}",
            content=content,
            language="",
            chunk_type="method",
            metadata={
                "enclosing_class": cls.name,
                "stereotype": cls.stereotype,
                "endpoint": method.endpoint,
                "start_line": method.location.start_line,
                "end_line": method.location.end_line,
            },
        )

    def _function_chunk(self, file_path: str, fn: FunctionSymbol, lines: list[str]) -> CodeChunk:
        header = [
            f"function {fn.qualified_name or fn.name}",
            f"Annotations: {', '.join(fn.annotations) or 'none'}",
            f"Endpoint: {fn.endpoint or 'none'}",
            f"Parameters: {', '.join(p.name + ': ' + p.type if p.type else p.name for p in fn.parameters) or 'none'}",
            f"Return type: {fn.return_type or 'none'}",
        ]
        content = self._render_chunk_content(header, lines, fn.location.start_line, fn.location.end_line)
        return CodeChunk(
            id=f"{file_path}::{fn.name}",
            file=file_path,
            symbol=fn.name,
            content=content,
            language="",
            chunk_type="function",
            metadata={
                "endpoint": fn.endpoint,
                "start_line": fn.location.start_line,
                "end_line": fn.location.end_line,
            },
        )

    def chunk_file(self, file_path: str, node: Optional[FileNode] = None) -> list[CodeChunk]:
        """Build (but do not embed/store) every chunk for one file. Exposed
        separately from `index_file` so callers/tests can inspect chunking
        decisions without needing a real embedding provider or vector store.
        """
        node = node if node is not None else self.symbol_index.get_file(file_path)
        if node is None:
            return []

        lines = _read_lines(self.repo_root, file_path)
        chunks: list[CodeChunk] = []

        for cls in node.classes:
            class_chunk = self._class_chunk(file_path, cls, lines)
            class_chunk.language = node.language
            chunks.append(class_chunk)
            for method in cls.methods:
                method_chunk = self._method_chunk(file_path, cls, method, lines)
                method_chunk.language = node.language
                chunks.append(method_chunk)

        for fn in node.functions:
            function_chunk = self._function_chunk(file_path, fn, lines)
            function_chunk.language = node.language
            chunks.append(function_chunk)

        return chunks

    # ------------------------------------------------------------------
    # Embedding + storage
    # ------------------------------------------------------------------

    def index_file(self, file_path: str) -> int:
        """Chunk, embed, and store every symbol in `file_path`, removing any
        previously-stored chunks for this file that no longer correspond to
        a current chunk id (e.g. a renamed/removed method). Returns the
        number of chunks written. No-ops (after cleanup) for a file with no
        indexed symbols, or one no longer present in `symbol_index`.
        """
        chunks = self.chunk_file(file_path)
        new_ids = {c.id for c in chunks}
        stale_ids = self._chunk_ids_by_file.get(file_path, set()) - new_ids
        if stale_ids:
            self.vector_store.delete(list(stale_ids))

        if chunks:
            contents = [c.content for c in chunks]
            embeddings: list[list[float]] = []
            for start in range(0, len(contents), _EMBED_BATCH_SIZE):
                embeddings.extend(self.embedding_router.embed_batch(contents[start : start + _EMBED_BATCH_SIZE]))
            self.vector_store.update(chunks, embeddings)

        if new_ids:
            self._chunk_ids_by_file[file_path] = new_ids
        else:
            self._chunk_ids_by_file.pop(file_path, None)

        return len(chunks)

    def remove_file(self, file_path: str) -> None:
        """Delete every chunk belonging to `file_path` (file deleted, or no
        longer a recognized/indexed source file)."""
        stale_ids = self._chunk_ids_by_file.pop(file_path, set())
        if stale_ids:
            self.vector_store.delete(list(stale_ids))

    def index_all(self) -> int:
        """Chunk, embed, and store every file currently in
        `symbol_index.files`. Returns the total chunk count written."""
        total = 0
        for file_path in sorted(self.symbol_index.files.keys()):
            total += self.index_file(file_path)
        return total

    # ------------------------------------------------------------------
    # Bookkeeping persistence (file -> chunk ids), so incremental cleanup
    # survives a process restart, not just the lifetime of one SemanticIndexer.
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, list[str]]:
        return {file_path: sorted(ids) for file_path, ids in self._chunk_ids_by_file.items()}

    def load_bookkeeping(self, path: str) -> None:
        in_path = Path(path)
        if not in_path.exists():
            return
        try:
            data = json.loads(in_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            self._chunk_ids_by_file = {k: set(v) for k, v in data.items() if isinstance(v, list)}

    def save_bookkeeping(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
