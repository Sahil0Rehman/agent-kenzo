"""Tests for `kenzo.tools.cache.CachedToolRegistry`, using a REAL
`kenzo.tools.registry.build_default_registry` against a real tmp git
repository (the `repo_root` fixture from conftest.py) -- no fakes for the
tools layer itself, only real filesystem tool calls.
"""

from __future__ import annotations

from kenzo.config.settings import PermissionsConfig
from kenzo.tools.cache import CachedToolRegistry
from kenzo.tools.registry import build_default_registry


def _cached_registry(repo_root, permissions=None):
    raw = build_default_registry(str(repo_root), permissions or PermissionsConfig())
    return CachedToolRegistry(raw)


# ---------------------------------------------------------------------------
# read_file caching
# ---------------------------------------------------------------------------


def test_read_file_repeated_call_served_from_cache_even_after_file_deleted(repo_root):
    cached = _cached_registry(repo_root)

    first = cached.execute("read_file", {"path": "main.py"})
    assert first.success is True
    original_content = first.data["content"]
    assert "return 42" in original_content

    # Delete the file from disk entirely.
    (repo_root / "main.py").unlink()

    # The exact same call (same tool + same arguments) must be served from
    # cache, not re-executed against the now-missing file.
    second = cached.execute("read_file", {"path": "main.py"})
    assert second.success is True
    assert second.data["content"] == original_content
    assert cached.cache_size() == 1


def test_read_file_different_arguments_are_not_conflated(repo_root):
    cached = _cached_registry(repo_root)

    main_result = cached.execute("read_file", {"path": "main.py"})
    readme_result = cached.execute("read_file", {"path": "README.md"})

    assert main_result.data["content"] != readme_result.data["content"]
    assert cached.cache_size() == 2


# ---------------------------------------------------------------------------
# write_file/edit_file invalidation of the matching read_file cache entry
# ---------------------------------------------------------------------------


def test_edit_file_invalidates_matching_read_file_cache_entry(repo_root):
    cached = _cached_registry(repo_root)

    before = cached.execute("read_file", {"path": "main.py"})
    assert "return 42" in before.data["content"]

    edit_result = cached.execute(
        "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 99"}
    )
    assert edit_result.success is True

    after = cached.execute("read_file", {"path": "main.py"})
    assert "return 99" in after.data["content"]
    assert after.data["content"] != before.data["content"]


def test_write_file_invalidates_matching_read_file_cache_entry(repo_root):
    cached = _cached_registry(repo_root)

    before = cached.execute("read_file", {"path": "README.md"})
    assert "Sample repo" in before.data["content"]

    write_result = cached.execute("write_file", {"path": "README.md", "content": "brand new content"})
    assert write_result.success is True

    after = cached.execute("read_file", {"path": "README.md"})
    assert after.data["content"] == "brand new content"


def test_edit_file_does_not_invalidate_unrelated_path_read_cache(repo_root):
    cached = _cached_registry(repo_root)

    readme_before = cached.execute("read_file", {"path": "README.md"})

    cached.execute("edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 99"})

    # README.md was never touched -- its cached read is untouched (still the
    # SAME cached ToolResult object, proving it was never re-executed).
    readme_after = cached.execute("read_file", {"path": "README.md"})
    assert readme_after is readme_before


def test_failed_edit_does_not_invalidate_any_cache(repo_root):
    cached = _cached_registry(repo_root)

    before = cached.execute("read_file", {"path": "main.py"})

    # old_text that doesn't exist -> edit_file fails.
    failed_edit = cached.execute(
        "edit_file", {"path": "main.py", "old_text": "this text does not exist", "new_text": "x"}
    )
    assert failed_edit.success is False

    after = cached.execute("read_file", {"path": "main.py"})
    assert after is before  # untouched, still the cached object


# ---------------------------------------------------------------------------
# search_files (and friends) cache entries dropped after any successful edit
# ---------------------------------------------------------------------------


def test_search_files_cache_entry_dropped_after_any_successful_edit(repo_root):
    cached = _cached_registry(repo_root)

    # Nothing matches this unique token yet.
    before = cached.execute("search_files", {"pattern": "UNIQUETOKEN12345"})
    assert before.data["matches"] == []
    assert cached.cache_size() == 1

    # Introduce a match via an edit to a DIFFERENT file than what's searched --
    # invalidation of search_files is repo-wide, not path-scoped.
    edit_result = cached.execute(
        "write_file", {"path": "new_file.py", "content": "# UNIQUETOKEN12345 marker\n"}
    )
    assert edit_result.success is True

    after = cached.execute("search_files", {"pattern": "UNIQUETOKEN12345"})
    assert len(after.data["matches"]) == 1
    assert after.data["matches"][0]["path"] == "new_file.py"


def test_repository_tree_cache_entry_dropped_after_edit(repo_root):
    cached = _cached_registry(repo_root)

    before = cached.execute("repository_tree", {"path": "."})
    assert not any(entry["path"] == "brand_new.py" for entry in before.data["entries"])

    cached.execute("write_file", {"path": "brand_new.py", "content": "x = 1\n"})

    after = cached.execute("repository_tree", {"path": "."})
    assert any(entry["path"] == "brand_new.py" for entry in after.data["entries"])


# ---------------------------------------------------------------------------
# Tools that are neither cacheable nor mutating always pass straight through
# ---------------------------------------------------------------------------


def test_list_directory_is_never_cached(repo_root):
    cached = _cached_registry(repo_root)

    cached.execute("list_directory", {"path": "."})
    assert cached.cache_size() == 0  # list_directory is not in _CACHEABLE

    cached.execute("write_file", {"path": "another_new_file.py", "content": "y = 2\n"})
    result = cached.execute("list_directory", {"path": "."})
    assert any(entry["name"] == "another_new_file.py" for entry in result.data["entries"])


# ---------------------------------------------------------------------------
# subset(): fresh, independently-cached, correctly-scoped view
# ---------------------------------------------------------------------------


def test_subset_is_independently_cached_from_parent(repo_root):
    cached = _cached_registry(repo_root)

    parent_first = cached.execute("read_file", {"path": "main.py"})
    assert parent_first.success is True

    read_only_subset = cached.subset(["read_file"])
    assert isinstance(read_only_subset, CachedToolRegistry)
    assert read_only_subset.cache_size() == 0  # does not inherit the parent's cache

    # Delete the file so a genuinely fresh (non-cached) read would fail.
    (repo_root / "main.py").unlink()

    # The subset has never read this path before, so it executes fresh
    # against the wrapped (subset) registry and sees the real, current
    # (now-missing) file state.
    subset_result = read_only_subset.execute("read_file", {"path": "main.py"})
    assert subset_result.success is False
    assert "not found" in (subset_result.error or "")

    # Meanwhile the PARENT's cache is untouched and still serves the old,
    # pre-deletion content -- proving the two caches are fully independent.
    parent_second = cached.execute("read_file", {"path": "main.py"})
    assert parent_second.success is True
    assert parent_second.data["content"] == parent_first.data["content"]


def test_subset_is_correctly_scoped_and_fails_closed_for_excluded_tools(repo_root):
    cached = _cached_registry(repo_root)
    read_only_subset = cached.subset(["read_file", "list_directory"])

    assert set(read_only_subset.names()) == {"read_file", "list_directory"}

    result = read_only_subset.execute("write_file", {"path": "should_not_exist.py", "content": "x"})
    assert result.success is False
    assert not (repo_root / "should_not_exist.py").exists()


# ---------------------------------------------------------------------------
# cache_size() / clear_cache()
# ---------------------------------------------------------------------------


def test_cache_size_and_clear_cache(repo_root):
    cached = _cached_registry(repo_root)
    cached.execute("read_file", {"path": "main.py"})
    cached.execute("search_files", {"pattern": "def"})
    assert cached.cache_size() == 2

    cached.clear_cache()
    assert cached.cache_size() == 0

    # After clearing, a previously-cached call re-executes for real.
    (repo_root / "main.py").write_text("def main():\n    return 7\n", encoding="utf-8")
    result = cached.execute("read_file", {"path": "main.py"})
    assert "return 7" in result.data["content"]
