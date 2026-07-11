"""Tests for kenzo.memory.indexer.index_repository (exclusion rules, counts)
and kenzo.memory.repository.RepositoryMemory.ensure_kenzo_dir (idempotent
seeding of the .kenzo/ directory).
"""

from __future__ import annotations

from kenzo.memory.indexer import index_repository
from kenzo.memory.repository import RepositoryMemory


# ---------------------------------------------------------------------------
# index_repository
# ---------------------------------------------------------------------------


def _build_sample_project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()

    (root / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (root / "app.js").write_text("console.log('hi');\n", encoding="utf-8")
    (root / "notes.log").write_text("debug log contents\n", encoding="utf-8")  # gitignored
    (root / ".gitignore").write_text("*.log\n", encoding="utf-8")

    node_modules_pkg = root / "node_modules" / "some_pkg"
    node_modules_pkg.mkdir(parents=True)
    (node_modules_pkg / "index.js").write_text("module.exports = {};\n", encoding="utf-8")

    return root


def test_index_repository_excludes_node_modules_and_gitignored_files(tmp_path):
    root = _build_sample_project(tmp_path)

    result = index_repository(str(root))

    paths = {f["path"] for f in result["files"]}

    assert "main.py" in paths
    assert "app.js" in paths
    assert ".gitignore" in paths
    assert "notes.log" not in paths  # excluded via .gitignore rule
    assert not any(p.startswith("node_modules/") for p in paths)  # excluded via _SKIP_DIRS

    assert paths == {"main.py", "app.js", ".gitignore"}
    assert result["file_count"] == 3
    assert result["file_count"] == len(result["files"])
    assert result["truncated"] is False


def test_index_repository_language_breakdown(tmp_path):
    root = _build_sample_project(tmp_path)

    result = index_repository(str(root))

    assert result["languages"][".py"] == 1
    assert result["languages"][".js"] == 1
    assert result["languages"]["(no ext)"] == 1  # .gitignore itself has no suffix
    assert ".log" not in result["languages"]  # gitignored file never counted


def test_index_repository_truncates_when_over_max_files(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for i in range(5):
        (root / f"file_{i}.txt").write_text(f"content {i}\n", encoding="utf-8")

    result = index_repository(str(root), max_files=2)

    assert result["truncated"] is True
    assert result["file_count"] == 2


def test_index_repository_empty_directory(tmp_path):
    root = tmp_path / "empty_proj"
    root.mkdir()

    result = index_repository(str(root))

    assert result["file_count"] == 0
    assert result["files"] == []
    assert result["truncated"] is False


# ---------------------------------------------------------------------------
# RepositoryMemory.ensure_kenzo_dir
# ---------------------------------------------------------------------------


def test_ensure_kenzo_dir_creates_exactly_three_files(tmp_path):
    repo_root = tmp_path / "proj"
    repo_root.mkdir()
    memory = RepositoryMemory(str(repo_root))

    kenzo_dir = memory.ensure_kenzo_dir()

    assert kenzo_dir == repo_root / ".kenzo"
    created = sorted(p.name for p in kenzo_dir.iterdir())
    assert created == ["architecture.md", "memory.md", "rules.md"]


def test_ensure_kenzo_dir_does_not_overwrite_existing_files_on_second_call(tmp_path):
    repo_root = tmp_path / "proj"
    repo_root.mkdir()
    memory = RepositoryMemory(str(repo_root))

    kenzo_dir = memory.ensure_kenzo_dir()
    custom_content = "custom notes that must survive\n"
    (kenzo_dir / "memory.md").write_text(custom_content, encoding="utf-8")

    memory.ensure_kenzo_dir()  # second call

    assert (kenzo_dir / "memory.md").read_text(encoding="utf-8") == custom_content
    assert sorted(p.name for p in kenzo_dir.iterdir()) == ["architecture.md", "memory.md", "rules.md"]


def test_read_helpers_return_default_content_after_ensure(tmp_path):
    repo_root = tmp_path / "proj"
    repo_root.mkdir()
    memory = RepositoryMemory(str(repo_root))
    memory.ensure_kenzo_dir()

    assert "Repository Architecture" in memory.read_architecture()
    assert "Repository Memory" in memory.read_memory()
    assert "Repository Rules" in memory.read_rules()


def test_append_memory_adds_dated_section(tmp_path):
    repo_root = tmp_path / "proj"
    repo_root.mkdir()
    memory = RepositoryMemory(str(repo_root))

    memory.append_memory("Learned something new.", heading="Discovery")

    content = memory.read_memory()
    assert "Learned something new." in content
    assert "### Discovery" in content
