"""Tests for kenzo.tools.tree.repository_tree: bounded recursive ASCII tree
rendering (depth cap, entry-count cap, ignored_dirs pruning), the shared
path-traversal guard, and its ToolSpec registration.
"""

from __future__ import annotations

from kenzo.config.settings import PermissionsConfig
from kenzo.tools.tree import get_tool_specs, repository_tree


def _touch(path, name: str, content: str = "x") -> None:
    (path / name).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Basic multi-level rendering shape
# ---------------------------------------------------------------------------


def test_basic_multi_level_tree_rendering_shape(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha_dir").mkdir()
    _touch(root / "alpha_dir", "inner.txt")
    _touch(root, "beta.txt")
    _touch(root, "zeta.txt")

    result = repository_tree(".", repo_root=str(root))

    assert result.success is True
    assert result.output.splitlines()[0] == "proj/"
    assert "alpha_dir/" in result.output
    assert "inner.txt" in result.output
    assert "beta.txt" in result.output
    assert "zeta.txt" in result.output
    # Directories are listed before files at the same level.
    lines = result.output.splitlines()
    assert lines.index("├── alpha_dir/") < lines.index("├── beta.txt")
    assert lines.index("├── beta.txt") < lines.index("└── zeta.txt")

    entries_by_path = {e["path"]: e for e in result.data["entries"]}
    assert entries_by_path["alpha_dir"] == {"path": "alpha_dir", "is_dir": True, "depth": 1}
    assert entries_by_path["alpha_dir/inner.txt"] == {
        "path": "alpha_dir/inner.txt",
        "is_dir": False,
        "depth": 2,
    }
    assert entries_by_path["beta.txt"]["depth"] == 1
    assert entries_by_path["zeta.txt"]["depth"] == 1
    assert result.data["truncated"] is False


# ---------------------------------------------------------------------------
# max_depth actually excludes deeper content
# ---------------------------------------------------------------------------


def test_max_depth_excludes_entries_deeper_than_the_cap(tmp_path):
    root = tmp_path / "proj"
    current = root
    current.mkdir()
    for level in range(1, 7):  # level1 .. level6, six levels deep
        current = current / f"level{level}"
        current.mkdir()
    _touch(current, "deep_file.txt")

    result = repository_tree(".", repo_root=str(root), max_depth=4)

    assert result.success is True
    depths = {e["depth"] for e in result.data["entries"]}
    assert max(depths) == 4
    assert all(d <= 4 for d in depths)

    paths = {e["path"] for e in result.data["entries"]}
    assert "level1" in paths
    assert "level1/level2" in paths
    assert "level1/level2/level3" in paths
    assert "level1/level2/level3/level4" in paths
    # depth-5-and-beyond content must be entirely absent, not merely
    # not-descended-into.
    assert "level1/level2/level3/level4/level5" not in paths
    assert not any("level5" in p or "level6" in p or "deep_file" in p for p in paths)


# ---------------------------------------------------------------------------
# max_entries truncation
# ---------------------------------------------------------------------------


def test_max_entries_truncates_and_reports_truncated_true(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for i in range(25):
        _touch(root, f"file{i:02d}.txt")

    result = repository_tree(".", repo_root=str(root), max_entries=5)

    assert result.success is True
    assert result.data["truncated"] is True
    assert len(result.data["entries"]) == 5
    assert "(truncated at 5 entries)" in result.output


def test_max_entries_not_hit_reports_truncated_false(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for i in range(3):
        _touch(root, f"file{i}.txt")

    result = repository_tree(".", repo_root=str(root), max_entries=400)

    assert result.data["truncated"] is False
    assert len(result.data["entries"]) == 3


# ---------------------------------------------------------------------------
# ignored_dirs pruning
# ---------------------------------------------------------------------------


def test_ignored_dirs_pruned_by_default(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    node_modules_pkg = root / "node_modules" / "some_pkg"
    node_modules_pkg.mkdir(parents=True)
    _touch(node_modules_pkg, "index.js")
    target_dir = root / "target" / "classes"
    target_dir.mkdir(parents=True)
    _touch(target_dir, "Compiled.class")
    _touch(root, "real_file.py")

    result = repository_tree(".", repo_root=str(root))

    paths = {e["path"] for e in result.data["entries"]}
    assert "real_file.py" in paths
    assert not any(p.startswith("node_modules") for p in paths)
    assert not any(p.startswith("target") for p in paths)
    assert "node_modules" not in paths
    assert "target" not in paths


def test_custom_ignored_dirs_override_default(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "custom_skip").mkdir()
    _touch(root / "custom_skip", "hidden.txt")
    target_dir = root / "target"
    target_dir.mkdir()
    _touch(target_dir, "not_ignored_now.txt")

    result = repository_tree(".", repo_root=str(root), ignored_dirs={"custom_skip"})

    paths = {e["path"] for e in result.data["entries"]}
    assert not any(p.startswith("custom_skip") for p in paths)
    # Overriding ignored_dirs REPLACES the default list rather than merging
    # with it, so "target" -- not named in this custom override -- shows up.
    assert "target" in paths
    assert "target/not_ignored_now.txt" in paths


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_path_traversal_rejected(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside_secret"
    outside.mkdir()

    result = repository_tree("../outside_secret", repo_root=str(root))

    assert result.success is False
    assert "outside repo_root" in result.error


def test_directory_not_found(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()

    result = repository_tree("no_such_dir", repo_root=str(root))

    assert result.success is False
    assert "directory not found" in result.error


def test_rejects_file_path_as_starting_point(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _touch(root, "a_file.txt")

    result = repository_tree("a_file.txt", repo_root=str(root))

    assert result.success is False
    assert "not a directory" in result.error


# ---------------------------------------------------------------------------
# get_tool_specs
# ---------------------------------------------------------------------------


def test_get_tool_specs_produces_single_working_repository_tree_spec(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _touch(root, "a_file.txt")

    specs = get_tool_specs(str(root), PermissionsConfig())

    assert len(specs) == 1
    assert specs[0].name == "repository_tree"

    result = specs[0].func(path=".")
    assert result.success is True
    assert "a_file.txt" in result.output
