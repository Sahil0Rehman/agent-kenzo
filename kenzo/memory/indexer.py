"""Repository indexer: walks a target repo and builds a lightweight, JSON-safe
summary of its contents (file list, sizes, language breakdown, human summary).

This index backs the "Repository Index" section of the context bundle that
`kenzo.memory.repository.RepositoryMemory` assembles for every agent node.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import pathspec

from kenzo.config.settings import DEFAULT_IGNORED_DIRS

# Directories we never want to walk into, regardless of .gitignore contents,
# unless the caller passes its own `ignored_dirs` (see IndexingConfig).
_SKIP_DIRS = set(DEFAULT_IGNORED_DIRS)

# Binary/asset extensions that add no useful text context to the index.
_BINARY_DENYLIST = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".exe",
    ".dll",
    ".so",
    ".pyc",
    ".woff",
    ".woff2",
    ".ttf",
    ".mp4",
    ".mp3",
}


def _load_gitignore_spec(repo_root: Path) -> pathspec.PathSpec | None:
    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.is_file():
        return None
    try:
        lines = gitignore_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _build_summary(top_level_dirs: list[str], file_count: int, languages: dict[str, int]) -> str:
    top_langs = sorted(languages.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_langs_str = ", ".join(f"{ext or '(no ext)'} ({count})" for ext, count in top_langs)
    dirs_str = ", ".join(top_level_dirs) if top_level_dirs else "(none)"
    return (
        f"Top-level directories: {dirs_str}. "
        f"Total files indexed: {file_count}. "
        f"Top languages by file count: {top_langs_str or '(none)'}."
    )


def index_repository(
    repo_root: str, max_files: int = 2000, ignored_dirs: set[str] | list[str] | None = None
) -> dict[str, Any]:
    """Walk `repo_root` and build a JSON-serializable summary of its contents.

    Skips well-known noise directories (see `ignored_dirs`, defaulting to
    `kenzo.config.settings.DEFAULT_IGNORED_DIRS` -- includes Maven's
    `target/`, Gradle's `.gradle/`, `.idea/`, `.vscode/`, `out/`, `coverage/`
    alongside the usual VCS/dependency/venv noise), anything matched by a
    root-level .gitignore, and a small binary-extension denylist.

    Caps the number of files actually included at `max_files` (deterministic:
    sorted paths). The result reports BOTH `indexed_files` (how many made it
    in) and `total_files` (how many actually exist before truncation) so a
    caller -- and, critically, the context bundle every agent reads -- can
    tell the difference between "this repo has 2000 files" and "this repo
    has 12000 files and we only indexed the first 2000."
    """
    root = Path(repo_root).resolve()
    spec = _load_gitignore_spec(root)
    skip_dirs = set(ignored_dirs) if ignored_dirs is not None else _SKIP_DIRS

    collected_paths: list[str] = []
    top_level_dirs: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_dir = current.relative_to(root)

        # Prune skip-dirs in place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        if rel_dir != Path("."):
            rel_dir_posix = rel_dir.as_posix()
            if spec is not None and spec.match_file(rel_dir_posix + "/"):
                dirnames[:] = []
                continue

        if current == root:
            top_level_dirs = sorted(
                d for d in dirnames if not (spec is not None and spec.match_file(d + "/"))
            )

        for filename in filenames:
            file_path = current / filename
            rel_path = file_path.relative_to(root)
            rel_posix = rel_path.as_posix()

            if spec is not None and spec.match_file(rel_posix):
                continue

            ext = file_path.suffix.lower()
            if ext in _BINARY_DENYLIST:
                continue

            collected_paths.append(rel_posix)

    collected_paths.sort()
    total_found = len(collected_paths)
    truncated = total_found > max_files
    included_paths = collected_paths[:max_files]

    files: list[dict[str, Any]] = []
    languages: Counter[str] = Counter()
    for rel_posix in included_paths:
        abs_path = root / rel_posix
        try:
            size = abs_path.stat().st_size
        except OSError:
            size = 0
        files.append({"path": rel_posix, "size": size})

        ext = Path(rel_posix).suffix.lower()
        languages[ext if ext else "(no ext)"] += 1

    languages_dict = dict(languages)
    summary = _build_summary(top_level_dirs, len(files), languages_dict)

    return {
        "repo_root": str(root),
        "file_count": len(files),  # kept for backward compatibility
        "indexed_files": len(files),
        "total_files": total_found,
        "truncated": truncated,
        "files": files,
        "languages": languages_dict,
        "summary": summary,
    }
