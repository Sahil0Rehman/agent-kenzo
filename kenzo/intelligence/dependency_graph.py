"""Class-to-class dependency graph + repository-architecture summary.

Built entirely on top of a already-populated `SymbolIndex` (see
`kenzo.intelligence.symbol_index`): every `ClassSymbol.dependencies` entry
(constructor/field DI, already computed by the parser layer -- see
`kenzo.intelligence.parser`) plus `extends`/`implements` becomes a directed
"depends on" edge between two class names, and every resolvable import
becomes a file-to-file edge. On top of that, classes are grouped into a
"module" (Java: last package segment; Python/JS/TS: containing directory
name) so `explain_architecture()` can render a human-readable summary of a
repo's controllers/services/repositories/etc. per module -- the same style
of high-level answer `kenzo.memory.repository.RepositoryMemory` gives for
plain-text project memory, but derived from real parsed structure instead.

Persistence follows the same graceful-None-on-missing-or-malformed style as
`RepositoryMemory.load_index`/`SymbolIndex.load`: a missing or corrupt file
on disk just means "no graph yet", never an exception escaping to the caller.
"""

from __future__ import annotations

import json
import posixpath
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from kenzo.intelligence.models import FileNode, ImportSymbol
from kenzo.intelligence.symbol_index import SymbolIndex

# --------------------------------------------------------------------------
# Stereotype -> summary-bucket mapping, shared by `explain_architecture()`'s
# both the returned dict and the rendered summary string. Order here is also
# the rendering order of the summary (controllers first, then services,
# etc.), matching the style shown in the module's own docstring example.
# --------------------------------------------------------------------------

_STEREOTYPE_TO_BUCKET: dict[str, str] = {
    "controller": "controllers",
    "service": "services",
    "repository": "repositories",
    "entity": "entities",
    "component": "components",
    "configuration": "configurations",
}

_BUCKET_ORDER: list[str] = [
    "controllers",
    "services",
    "repositories",
    "entities",
    "components",
    "configurations",
    "other",
]

_BUCKET_HEADING: dict[str, str] = {
    "controllers": "Controllers",
    "services": "Services",
    "repositories": "Repositories",
    "entities": "Entities",
    "components": "Components",
    "configurations": "Configurations",
    "other": "Other",
}


# --------------------------------------------------------------------------
# Import-resolution helpers (module-level, pure functions -- no state)
# --------------------------------------------------------------------------


def _dotted_suffix_match(a: str, b: str) -> bool:
    """True if two dotted paths are the same, or one is a dotted-segment
    suffix of the other (e.g. "pkg.sub.mod" vs "sub.mod") -- used for
    Python's best-effort partial/suffix import matching, since Python
    imports are frequently relative or only partially qualified.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    return a.endswith("." + b) or b.endswith("." + a)


def _python_dotted_guess(file_path: str) -> str:
    """Best-effort dotted-module guess for a Python file from its own
    repo-relative path, e.g. "pkg/sub/mod.py" -> "pkg.sub.mod" and
    "pkg/sub/__init__.py" -> "pkg.sub" (the conventional package name).
    """
    parts = list(PurePosixPath(file_path).parts)
    if not parts:
        return ""
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _resolve_python_import(
    importing_file: str, imp: ImportSymbol, dotted_guesses: dict[str, str]
) -> Optional[str]:
    """Best-effort resolution of one Python `ImportSymbol` against every
    other indexed Python file's dotted-module guess. Handles both absolute
    imports ("import pkg.sub.mod" / "from pkg.sub import mod") and relative
    ones ("from . import utils" / "from .sub import utils", where `module`
    is rendered as a leading-dot string like "." or ".sub" per the parser),
    walking up one directory per extra leading dot the way a real relative
    import would.
    """
    module = imp.module or ""
    dot_count = len(module) - len(module.lstrip("."))
    rest = module[dot_count:]  # part after any leading dots, e.g. "sub" in ".sub"

    candidates: set[str] = set()

    if dot_count > 0:
        importing_parts = list(PurePosixPath(importing_file).parent.parts)
        levels_up = max(dot_count - 1, 0)
        base_parts = importing_parts[: len(importing_parts) - levels_up] if levels_up else importing_parts
        if rest:
            base_parts = base_parts + rest.split(".")
        base = ".".join(base_parts)
        if base:
            candidates.add(base)
        for name in imp.imported_names:
            joined = base_parts + [name]
            candidates.add(".".join(joined))
    else:
        if module:
            candidates.add(module)
        for name in imp.imported_names:
            candidates.add(f"{module}.{name}" if module else name)

    if not candidates:
        return None

    for file_path, guess in dotted_guesses.items():
        if file_path == importing_file or not guess:
            continue
        if any(_dotted_suffix_match(guess, cand) for cand in candidates):
            return file_path
    return None


_JSTS_RESOLUTION_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx")


def _resolve_jsts_relative(importing_file: str, module: str, indexed_paths: set[str]) -> Optional[str]:
    """Best-effort Node/TS-style relative-import resolution: only handles
    `./`/`../`-prefixed specifiers (bare specifiers like "react" or
    "@nestjs/common" are external packages, correctly left unresolved).
    Tries the joined path as-is, then with each of .ts/.tsx/.js/.jsx
    appended, then as a directory index file (`<path>/index.<ext>`) -- a
    reasonable subset of real Node/TS module resolution, not a full
    reimplementation of it.
    """
    if not (module.startswith("./") or module.startswith("../")):
        return None

    importing_dir = posixpath.dirname(importing_file)
    joined = posixpath.normpath(posixpath.join(importing_dir, module))

    candidates = [joined]
    candidates.extend(joined + ext for ext in _JSTS_RESOLUTION_EXTENSIONS)
    candidates.extend(posixpath.join(joined, "index" + ext) for ext in _JSTS_RESOLUTION_EXTENSIONS)

    for cand in candidates:
        if cand in indexed_paths:
            return cand
    return None


class DependencyGraph:
    """Class-level and file-level dependency edges across an entire indexed
    repo, plus a module grouping used to explain repo architecture.
    """

    def __init__(self, symbol_index: SymbolIndex) -> None:
        self.symbol_index: SymbolIndex = symbol_index
        self.class_deps: dict[str, list[str]] = {}
        self.class_dependents: dict[str, list[str]] = {}
        self.file_deps: dict[str, list[str]] = {}
        self.classes_by_module: dict[str, list[tuple[str, str, str]]] = {}

    # ----------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------

    def build(self) -> None:
        self._build_class_deps()
        self._build_file_deps()
        self._build_classes_by_module()

    def _build_class_deps(self) -> None:
        class_deps: dict[str, list[str]] = {}

        for _file_path, cls in self.symbol_index.all_classes():
            deps: list[str] = []
            seen: set[str] = set()
            for dep in cls.dependencies:
                if dep not in seen:
                    deps.append(dep)
                    seen.add(dep)
            if cls.extends and cls.extends not in seen:
                deps.append(cls.extends)
                seen.add(cls.extends)
            for impl in cls.implements:
                if impl not in seen:
                    deps.append(impl)
                    seen.add(impl)
            class_deps[cls.name] = deps

        class_dependents: dict[str, list[str]] = {}
        for class_name, deps in class_deps.items():
            for dep in deps:
                bucket = class_dependents.setdefault(dep, [])
                if class_name not in bucket:
                    bucket.append(class_name)

        self.class_deps = class_deps
        self.class_dependents = class_dependents

    def _build_file_deps(self) -> None:
        files = self.symbol_index.files  # rel_path -> FileNode (public attribute)
        indexed_paths = set(files.keys())

        java_qualified_to_file: dict[str, str] = {}
        python_dotted_guess: dict[str, str] = {}

        for file_path, node in files.items():
            if node.language == "java":
                for cls in node.classes:
                    if cls.qualified_name:
                        java_qualified_to_file.setdefault(cls.qualified_name, file_path)
            elif node.language == "python":
                python_dotted_guess[file_path] = _python_dotted_guess(file_path)

        file_deps: dict[str, list[str]] = {}

        for file_path, node in files.items():
            resolved: list[str] = []
            seen: set[str] = set()

            for imp in node.imports:
                target: Optional[str] = None
                if node.language == "java":
                    target = java_qualified_to_file.get(imp.module)
                elif node.language == "python":
                    target = _resolve_python_import(file_path, imp, python_dotted_guess)
                elif node.language in ("javascript", "typescript"):
                    target = _resolve_jsts_relative(file_path, imp.module, indexed_paths)

                if target and target != file_path and target not in seen:
                    resolved.append(target)
                    seen.add(target)

            file_deps[file_path] = resolved

        self.file_deps = file_deps

    def _build_classes_by_module(self) -> None:
        classes_by_module: dict[str, list[tuple[str, str, str]]] = {}

        for file_path in sorted(self.symbol_index.files.keys()):
            node = self.symbol_index.files[file_path]
            module_label = self._module_label(file_path, node)
            for cls in node.classes:
                stereotype = cls.stereotype if cls.stereotype else "other"
                classes_by_module.setdefault(module_label, []).append((file_path, cls.name, stereotype))

        self.classes_by_module = classes_by_module

    @staticmethod
    def _module_label(file_path: str, node: FileNode) -> str:
        if node.language == "java":
            package = node.package
            if not package:
                return "(default)"
            return package.rsplit(".", 1)[-1]

        parent = PurePosixPath(file_path).parent
        parent_str = parent.as_posix()
        if parent_str in ("", "."):
            return "(root)"
        return parent.name

    # ----------------------------------------------------------------
    # Lookups
    # ----------------------------------------------------------------

    def get_class_dependencies(self, class_name: str) -> list[str]:
        target = class_name.lower()
        for key, deps in self.class_deps.items():
            if key.lower() == target:
                return deps
        return []

    def get_dependents(self, class_name: str) -> list[str]:
        target = class_name.lower()
        for key, dependents in self.class_dependents.items():
            if key.lower() == target:
                return dependents
        return []

    # ----------------------------------------------------------------
    # Architecture summary
    # ----------------------------------------------------------------

    def explain_architecture(self, module: str | None = None) -> dict[str, Any]:
        if module is not None:
            target = module.lower()
            module_items = [
                (label, entries) for label, entries in self.classes_by_module.items() if label.lower() == target
            ]
        else:
            module_items = list(self.classes_by_module.items())

        modules_out: dict[str, dict[str, list[str]]] = {}
        summary_sections: list[str] = []

        for label, entries in module_items:
            buckets: dict[str, list[str]] = {name: [] for name in _BUCKET_ORDER}
            for _file_path, class_name, stereotype in entries:
                bucket = _STEREOTYPE_TO_BUCKET.get(stereotype, "other")
                buckets[bucket].append(class_name)
            modules_out[label] = buckets

            lines = [f"{label} module:", ""]
            for bucket in _BUCKET_ORDER:
                names = buckets[bucket]
                if not names:
                    continue  # skip empty sections entirely rather than printing an empty heading
                lines.append(f"{_BUCKET_HEADING[bucket]}:")
                lines.extend(f"  {name}" for name in names)
                lines.append("")
            summary_sections.append("\n".join(lines).rstrip())

        return {"modules": modules_out, "summary": "\n\n".join(summary_sections)}

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_deps": self.class_deps,
            "class_dependents": self.class_dependents,
            "file_deps": self.file_deps,
            "classes_by_module": {
                label: [list(entry) for entry in entries] for label, entries in self.classes_by_module.items()
            },
        }

    @classmethod
    def from_dict(cls, symbol_index: SymbolIndex, data: dict[str, Any]) -> "DependencyGraph":
        graph = cls(symbol_index)
        graph.class_deps = {k: list(v) for k, v in (data.get("class_deps") or {}).items()}
        graph.class_dependents = {k: list(v) for k, v in (data.get("class_dependents") or {}).items()}
        graph.file_deps = {k: list(v) for k, v in (data.get("file_deps") or {}).items()}
        graph.classes_by_module = {
            label: [tuple(entry) for entry in entries]
            for label, entries in (data.get("classes_by_module") or {}).items()
        }
        return graph

    def save(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, symbol_index: SymbolIndex, path: str) -> Optional["DependencyGraph"]:
        in_path = Path(path)
        if not in_path.exists():
            return None
        try:
            data = json.loads(in_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return cls.from_dict(symbol_index, data)
        except Exception:  # noqa: BLE001 - malformed persisted data should degrade to None, not crash the caller
            return None
