"""Tests for kenzo.memory.indexer.index_repository's Phase 2 additions:
default exclusion of target/.gradle/out/coverage, a custom ignored_dirs
override, and the indexed_files/total_files truncation reporting.
"""

from __future__ import annotations

from kenzo.memory.indexer import index_repository


def _touch(path, name: str, content: str = "content\n") -> None:
    (path / name).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Default exclusions: target/, .gradle/, out/, coverage/
# ---------------------------------------------------------------------------


def test_default_exclusions_cover_target_gradle_out_and_coverage(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()

    for noisy_dir in ("target", ".gradle", "out", "coverage"):
        nested = root / noisy_dir / "nested"
        nested.mkdir(parents=True)
        _touch(nested, "noise.txt")

    _touch(root, "real_file.py")

    result = index_repository(str(root))

    paths = {f["path"] for f in result["files"]}
    assert paths == {"real_file.py"}
    for noisy_dir in ("target", ".gradle", "out", "coverage"):
        assert not any(p.startswith(f"{noisy_dir}/") for p in paths)


# ---------------------------------------------------------------------------
# Custom ignored_dirs override
# ---------------------------------------------------------------------------


def test_custom_ignored_dirs_override_replaces_default_list(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()

    custom_dir = root / "custom_noise"
    custom_dir.mkdir()
    _touch(custom_dir, "should_be_excluded.txt")

    target_dir = root / "target"
    target_dir.mkdir()
    _touch(target_dir, "no_longer_excluded.txt")

    result = index_repository(str(root), ignored_dirs=["custom_noise"])

    paths = {f["path"] for f in result["files"]}
    assert not any(p.startswith("custom_noise/") for p in paths)
    # A custom override REPLACES the default ignore list rather than
    # merging with it, so target/ (not named here) is now included.
    assert "target/no_longer_excluded.txt" in paths


# ---------------------------------------------------------------------------
# indexed_files / total_files truncation reporting
# ---------------------------------------------------------------------------


def test_indexed_files_and_total_files_reflect_true_count_when_truncated(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for i in range(10):
        _touch(root, f"file_{i:02d}.txt")

    result = index_repository(str(root), max_files=4)

    assert result["truncated"] is True
    assert result["indexed_files"] == 4
    assert result["total_files"] == 10
    assert result["total_files"] > result["indexed_files"]
    # file_count is kept for backward compatibility and matches indexed_files.
    assert result["file_count"] == result["indexed_files"]


def test_indexed_files_and_total_files_equal_when_not_truncated(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for i in range(3):
        _touch(root, f"file_{i}.txt")

    result = index_repository(str(root), max_files=10)

    assert result["truncated"] is False
    assert result["indexed_files"] == 3
    assert result["total_files"] == 3
