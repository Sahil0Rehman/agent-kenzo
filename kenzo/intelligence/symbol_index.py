"""Repository-wide symbol index: walks a repo, parses every supported source
file into a `FileNode` (via `kenzo.intelligence.parser.parse_file`), and
exposes a queryable, persistable index across all of them.

Follows the same repo-walking discipline as `kenzo.memory.indexer`
(`os.walk` with in-place `dirnames` pruning against a configurable
`ignored_dirs` set defaulting to `kenzo.config.settings.DEFAULT_IGNORED_DIRS`,
deterministic sorted output, a `max_files` cap reporting both how many files
were indexed and how many actually exist) and the graceful-None persistence
style of `kenzo.memory.repository.RepositoryMemory.load_index` (a missing or
malformed file on disk is not an error -- it just means "no index yet").

`parse_file` already degrades gracefully per-file (a stub/unimplemented
language parser, a syntax error, an unreadable file, etc. all resolve to a
best-effort `FileNode` or `None`, never an exception escaping `parse_file`
under normal operation) -- the try/except in `build()`/`reindex_file()`
around each individual `parse_file` call is therefore a defensive outer
boundary, not the primary error-handling mechanism, consistent with how
`kenzo.memory.indexer`/`java_analyzer` treat one bad file as non-fatal to the
whole scan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from kenzo.config.settings import DEFAULT_IGNORED_DIRS
from kenzo.intelligence.models import ClassSymbol, FileNode, FunctionSymbol
from kenzo.intelligence.parser import language_for_extension, parse_file


def _round_robin_by_language(paths_by_language: dict[str, list[str]], max_files: int) -> list[str]:
    """Interleave each language's (already alphabetically-sorted) path list
    one file at a time, in a stable language order (sorted by name), until
    `max_files` total is reached or every list is exhausted. See
    `SymbolIndex.build`'s docstring for why this replaces a single
    alphabetical sort over every path combined.
    """
    if max_files <= 0:
        return []
    ordered_languages = sorted(paths_by_language.keys())
    cursors = {language: 0 for language in ordered_languages}
    result: list[str] = []
    while len(result) < max_files:
        made_progress = False
        for language in ordered_languages:
            if len(result) >= max_files:
                break
            paths = paths_by_language[language]
            cursor = cursors[language]
            if cursor < len(paths):
                result.append(paths[cursor])
                cursors[language] = cursor + 1
                made_progress = True
        if not made_progress:
            break
    return result


class SymbolIndex:
    """A queryable, persistable index of every parsed source file in a repo."""

    def __init__(self, repo_root: str) -> None:
        self.repo_root: Path = Path(repo_root).resolve()
        self.files: dict[str, FileNode] = {}  # rel_path (posix) -> FileNode
        self.truncated: bool = False
        self.total_files_found: int = 0
        # Lazy, invalidate-on-mutation cache for find_class(): without it,
        # every call re-sorts and re-walks the ENTIRE index via all_classes()
        # (see find_class below) -- fine for one-off lookups, but
        # ReferenceGraph.build() calls find_class() once per call site across
        # the whole repo, which turned an O(files) scan into an
        # effectively-quadratic-or-worse one at scale (empirically measured
        # ~files^2.8 in kenzo's own indexing benchmark). None means "rebuild
        # on next find_class() call" -- set here and by build()/reindex_file()
        # whenever self.files changes.
        self._class_by_name: dict[str, list[tuple[str, ClassSymbol]]] | None = None

    # ----------------------------------------------------------------
    # Full scan
    # ----------------------------------------------------------------

    def build(self, ignored_dirs: set[str] | list[str] | None = None, max_files: int = 5000) -> None:
        """Walk `self.repo_root`, parse every recognized source file, and
        (re)populate `self.files`/`self.truncated`/`self.total_files_found`.

        Mirrors `kenzo.memory.indexer.index_repository`'s walk discipline:
        `ignored_dirs` are pruned from `os.walk`'s `dirnames` in place (so we
        never even descend into them), and `max_files` caps how many are
        actually parsed while still reporting the true total found.

        Selection when truncated is ROUND-ROBIN ACROSS LANGUAGES (grouped by
        `language_for_extension`, each group sorted alphabetically for
        determinism), not a single alphabetical sort of every path -- a pure
        alphabetical sort was found to starve whole languages out of the
        index on real repos: a large `WebContent/` (JS/TS-heavy, common in
        JSP-based Java web apps) sorts before `src/` (where the Java actually
        lives), so with a low-enough cap the JS/TS files alone could consume
        the entire budget before the walk ever reached the Java source the
        cap was meant to leave room for. Round-robin guarantees every
        language present gets a proportional share of `max_files` regardless
        of which directory happens to sort first.
        """
        skip_dirs = set(ignored_dirs) if ignored_dirs is not None else set(DEFAULT_IGNORED_DIRS)

        paths_by_language: dict[str, list[str]] = {}
        total_found = 0
        try:
            for dirpath, dirnames, filenames in os.walk(self.repo_root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]

                current = Path(dirpath)
                for filename in filenames:
                    file_path = current / filename
                    language = language_for_extension(file_path.suffix)
                    if language is None:
                        continue
                    try:
                        rel_posix = file_path.relative_to(self.repo_root).as_posix()
                    except ValueError:
                        continue
                    paths_by_language.setdefault(language, []).append(rel_posix)
                    total_found += 1
        except OSError:
            # A directory disappearing/permission error mid-walk should not
            # sink the whole index build -- fall back to whatever we already
            # collected before the failure.
            pass

        for language_paths in paths_by_language.values():
            language_paths.sort()

        self.total_files_found = total_found
        self.truncated = total_found > max_files
        included_paths = _round_robin_by_language(paths_by_language, max_files)

        new_files: dict[str, FileNode] = {}
        for rel_posix in included_paths:
            try:
                node = parse_file(str(self.repo_root), rel_posix)
            except Exception:  # noqa: BLE001 - defensive outer boundary; one bad file must not sink the scan
                continue
            if node is not None:
                new_files[rel_posix] = node

        self.files = new_files
        self._class_by_name = None

    # ----------------------------------------------------------------
    # Incremental update
    # ----------------------------------------------------------------

    def reindex_file(self, rel_path: str) -> None:
        """Re-parse exactly one file and update `self.files` accordingly.

        O(1 file) -- does not re-walk the repo and does not touch any other
        entry in `self.files`. If the file no longer exists on disk, its
        entry (if any) is removed instead of re-parsed.
        """
        rel_posix = Path(rel_path).as_posix()
        abs_path = self.repo_root / rel_path

        if not abs_path.exists():
            self.files.pop(rel_posix, None)
            self._class_by_name = None
            return

        try:
            node = parse_file(str(self.repo_root), rel_path)
        except Exception:  # noqa: BLE001 - defensive outer boundary, mirrors build()
            return

        if node is None:
            # No longer a recognized/supported extension -- drop any stale entry.
            self.files.pop(rel_posix, None)
            self._class_by_name = None
            return

        self.files[rel_posix] = node
        self._class_by_name = None

    # ----------------------------------------------------------------
    # Lookups
    # ----------------------------------------------------------------

    def get_file(self, rel_path: str) -> Optional[FileNode]:
        return self.files.get(Path(rel_path).as_posix())

    def all_classes(self) -> list[tuple[str, ClassSymbol]]:
        """Every (file_path, ClassSymbol) pair across every indexed file,
        sorted by file path then by declaration order within the file.
        """
        result: list[tuple[str, ClassSymbol]] = []
        for file_path in sorted(self.files.keys()):
            for cls in self.files[file_path].classes:
                result.append((file_path, cls))
        return result

    def all_functions(self) -> list[tuple[str, FunctionSymbol]]:
        """Every (file_path, top-level FunctionSymbol) pair -- methods are
        reachable via `all_classes()[*][1].methods`, not here.
        """
        result: list[tuple[str, FunctionSymbol]] = []
        for file_path in sorted(self.files.keys()):
            for fn in self.files[file_path].functions:
                result.append((file_path, fn))
        return result

    def find_class(self, name: str) -> list[tuple[str, ClassSymbol]]:
        """Case-insensitive exact match on `ClassSymbol.name` (not
        `qualified_name`) across every indexed file. There can legitimately be
        more than one class with the same simple name in different
        packages/files, so every match is returned.

        Backed by a lazily-built, invalidate-on-mutation name index (see
        `self._class_by_name`) rather than an `all_classes()` scan per call --
        a caller that calls this once per call site across a whole repo (e.g.
        `ReferenceGraph.build()`) would otherwise turn an O(files) index build
        into an O(call sites * files log files) one, which measured as
        catastrophically non-linear at scale before this cache was added.
        """
        if self._class_by_name is None:
            index: dict[str, list[tuple[str, ClassSymbol]]] = {}
            for path, cls in self.all_classes():
                index.setdefault(cls.name.lower(), []).append((path, cls))
            self._class_by_name = index
        return list(self._class_by_name.get(name.lower(), []))

    def find_function_or_method(self, name: str) -> list[dict[str, Any]]:
        """Case-insensitive exact match on a plain function name OR a method
        name, returning `{"file": str, "class": str | None, "function":
        FunctionSymbol}` so callers can distinguish a bare-function hit
        (`class` is None) from a method hit (`class` is the owning class's
        name) and know which class it belongs to.
        """
        target = name.lower()
        results: list[dict[str, Any]] = []

        for file_path, fn in self.all_functions():
            if fn.name.lower() == target:
                results.append({"file": file_path, "class": None, "function": fn})

        for file_path, cls in self.all_classes():
            for method in cls.methods:
                if method.name.lower() == target:
                    results.append({"file": file_path, "class": cls.name, "function": method})

        return results

    def stats(self) -> dict[str, Any]:
        by_language: dict[str, int] = {}
        class_count = 0
        function_count = 0
        for node in self.files.values():
            by_language[node.language] = by_language.get(node.language, 0) + 1
            class_count += len(node.classes)
            function_count += len(node.functions)

        return {
            "file_count": len(self.files),
            "total_files_found": self.total_files_found,
            "truncated": self.truncated,
            "class_count": class_count,
            "function_count": function_count,
            "by_language": by_language,
        }

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "truncated": self.truncated,
            "total_files_found": self.total_files_found,
            "files": {rel_path: node.model_dump() for rel_path, node in self.files.items()},
        }

    @classmethod
    def from_dict(cls, repo_root: str, data: dict[str, Any]) -> "SymbolIndex":
        index = cls(repo_root)
        index.truncated = bool(data.get("truncated", False))
        index.total_files_found = int(data.get("total_files_found", 0))
        index.files = {
            rel_path: FileNode.model_validate(node_data)
            for rel_path, node_data in (data.get("files") or {}).items()
        }
        return index

    def save(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, repo_root: str, path: str) -> Optional["SymbolIndex"]:
        in_path = Path(path)
        if not in_path.exists():
            return None
        try:
            data = json.loads(in_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return cls.from_dict(repo_root, data)
        except Exception:  # noqa: BLE001 - malformed persisted data should degrade to None, not crash the caller
            return None
