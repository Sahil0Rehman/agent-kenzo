"""Tests for kenzo.tools.filesystem: read/write/edit/search/list, the
path-traversal guard shared by all of them, and the write_file permission
gate.
"""

from __future__ import annotations

from kenzo.config.settings import PermissionsConfig
from kenzo.tools.filesystem import (
    edit_file,
    get_tool_specs,
    list_directory,
    read_file,
    search_files,
    write_file,
)

# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_happy_path(repo_root):
    result = read_file("README.md", repo_root=str(repo_root))

    assert result.success is True
    assert "Sample repo" in result.output
    assert result.data["content"] == result.output


def test_read_file_missing_file(repo_root):
    result = read_file("missing.txt", repo_root=str(repo_root))

    assert result.success is False
    assert "not found" in result.error


def test_read_file_rejects_directory(repo_root):
    result = read_file(".", repo_root=str(repo_root))

    assert result.success is False
    assert "directory" in result.error


def test_read_file_rejects_relative_traversal(repo_root, tmp_path):
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("outside secret", encoding="utf-8")

    result = read_file("../outside_secret.txt", repo_root=str(repo_root))

    assert result.success is False
    assert "outside repo_root" in result.error


def test_read_file_rejects_absolute_path_outside_repo(repo_root, tmp_path):
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("outside secret", encoding="utf-8")

    result = read_file(str(outside), repo_root=str(repo_root))

    assert result.success is False
    assert "outside repo_root" in result.error


def test_read_file_refuses_non_utf8_file(repo_root):
    target = repo_root / "legacy.txt"
    # 0x92 is cp1252's right single quotation mark; it's not valid UTF-8
    # (a bytes literal is used here, not str.encode, since Python's \x92 in
    # a *string* literal is the Unicode control code U+0092, not this byte).
    target.write_bytes(b"caf\x92 latin1-ish bytes")

    result = read_file("legacy.txt", repo_root=str(repo_root))

    assert result.success is False
    assert "UTF-8" in result.error


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_file_happy_path_creates_parent_dirs(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = write_file(
        "nested/dir/notes.txt", "hello world", repo_root=str(repo_root), permissions=permissions
    )

    assert result.success is True
    written = repo_root / "nested" / "dir" / "notes.txt"
    assert written.read_text(encoding="utf-8") == "hello world"
    assert result.data["bytes_written"] == len("hello world")


def test_write_file_overwrites_existing_file(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = write_file("main.py", "def main():\n    return 0\n", repo_root=str(repo_root), permissions=permissions)

    assert result.success is True
    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 0\n"


def test_write_file_blocked_when_allow_write_false(repo_root):
    permissions = PermissionsConfig(allow_write=False)

    result = write_file("blocked.txt", "nope", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "allow_write" in result.error
    assert not (repo_root / "blocked.txt").exists()


def test_write_file_rejects_directory_target(repo_root):
    permissions = PermissionsConfig(allow_write=True)
    (repo_root / "a_dir").mkdir()

    result = write_file("a_dir", "content", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "directory" in result.error


def test_write_file_rejects_path_traversal(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = write_file("../escape.txt", "pwned", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "outside repo_root" in result.error
    assert not (repo_root.parent / "escape.txt").exists()


def test_write_file_returns_diff_for_overwrite(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = write_file(
        "main.py", "def main():\n    return 0\n", repo_root=str(repo_root), permissions=permissions
    )

    assert result.success is True
    assert result.data["created"] is False
    assert "return 42" in result.data["old_content"]
    assert "return 0" in result.data["new_content"]
    assert "-    return 42" in result.data["diff"]
    assert "+    return 0" in result.data["diff"]


def test_write_file_marks_new_file_as_created_with_no_old_content(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = write_file("brand_new.txt", "hello\n", repo_root=str(repo_root), permissions=permissions)

    assert result.success is True
    assert result.data["created"] is True
    assert result.data["old_content"] is None
    assert "+hello" in result.data["diff"]


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def test_edit_file_happy_path(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = edit_file("main.py", "return 42", "return 43", repo_root=str(repo_root), permissions=permissions)

    assert result.success is True
    assert "return 43" in (repo_root / "main.py").read_text(encoding="utf-8")


def test_edit_file_old_text_not_found(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = edit_file(
        "main.py", "this text does not exist", "x", repo_root=str(repo_root), permissions=permissions
    )

    assert result.success is False
    assert "not found" in result.error


def test_edit_file_rejects_ambiguous_match(repo_root):
    permissions = PermissionsConfig(allow_write=True)
    target = repo_root / "dup.txt"
    target.write_text("foo\nfoo\n", encoding="utf-8")

    result = edit_file("dup.txt", "foo", "bar", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "found 2 times" in result.error
    assert target.read_text(encoding="utf-8") == "foo\nfoo\n"  # unchanged


def test_edit_file_missing_file(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = edit_file("missing.py", "a", "b", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "not found" in result.error


def test_edit_file_rejects_path_traversal(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = edit_file("../escape.py", "a", "b", repo_root=str(repo_root), permissions=permissions)

    assert result.success is False
    assert "outside repo_root" in result.error


def test_edit_file_blocked_when_allow_write_false(repo_root):
    permissions = PermissionsConfig(allow_write=False)

    result = edit_file(
        "main.py", "return 42", "return 999", repo_root=str(repo_root), permissions=permissions
    )

    assert result.success is False
    assert "allow_write" in result.error
    assert "return 42" in (repo_root / "main.py").read_text(encoding="utf-8")  # unchanged


def test_edit_file_returns_diff_and_old_new_content(repo_root):
    permissions = PermissionsConfig(allow_write=True)

    result = edit_file("main.py", "return 42", "return 43", repo_root=str(repo_root), permissions=permissions)

    assert result.success is True
    assert result.data["old_content"].count("return 42") == 1
    assert result.data["new_content"].count("return 43") == 1
    assert "-    return 42" in result.data["diff"]
    assert "+    return 43" in result.data["diff"]


def test_edit_file_refuses_non_utf8_file_without_corrupting_it(repo_root):
    permissions = PermissionsConfig(allow_write=True)
    target = repo_root / "legacy.py"
    # 0x92 is cp1252's right single quotation mark; it's not valid UTF-8
    # (a bytes literal is used here, not str.encode, since Python's \x92 in
    # a *string* literal is the Unicode control code U+0092, not this byte).
    original_bytes = b"# Author\x92s note: do not touch\ndef f():\n    return 1\n"
    target.write_bytes(original_bytes)

    result = edit_file(
        "legacy.py", "return 1", "return 2", repo_root=str(repo_root), permissions=permissions
    )

    assert result.success is False
    assert "UTF-8" in result.error
    assert target.read_bytes() == original_bytes  # completely unchanged, byte-for-byte


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


def test_search_files_finds_matches(repo_root):
    result = search_files(r"def main", repo_root=str(repo_root))

    assert result.success is True
    paths = {match["path"] for match in result.data["matches"]}
    assert "main.py" in paths


def test_search_files_respects_glob_filter(repo_root):
    result = search_files(r"Sample", repo_root=str(repo_root), glob="*.md")

    assert result.success is True
    paths = {match["path"] for match in result.data["matches"]}
    assert paths == {"README.md"}


def test_search_files_no_matches(repo_root):
    result = search_files(r"this_pattern_should_not_exist_anywhere", repo_root=str(repo_root))

    assert result.success is True
    assert result.data["matches"] == []
    assert result.data["truncated"] is False


def test_search_files_invalid_regex(repo_root):
    result = search_files(r"(unterminated", repo_root=str(repo_root))

    assert result.success is False
    assert "invalid regex" in result.error


def test_search_files_missing_repo_root(tmp_path):
    missing = tmp_path / "does_not_exist"
    result = search_files(r"anything", repo_root=str(missing))

    assert result.success is False
    assert "does not exist" in result.error


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


def test_list_directory_happy_path(repo_root):
    result = list_directory(".", repo_root=str(repo_root))

    assert result.success is True
    names = {entry["name"] for entry in result.data["entries"]}
    assert {"README.md", "main.py", ".gitignore"}.issubset(names)


def test_list_directory_missing_dir(repo_root):
    result = list_directory("no_such_dir", repo_root=str(repo_root))

    assert result.success is False
    assert "not found" in result.error


def test_list_directory_rejects_file_path(repo_root):
    result = list_directory("main.py", repo_root=str(repo_root))

    assert result.success is False
    assert "not a directory" in result.error


def test_list_directory_rejects_path_traversal(repo_root):
    result = list_directory("../../", repo_root=str(repo_root))

    assert result.success is False
    assert "outside repo_root" in result.error


# ---------------------------------------------------------------------------
# get_tool_specs
# ---------------------------------------------------------------------------


def test_get_tool_specs_binds_repo_root_and_permissions(repo_root):
    permissions = PermissionsConfig(allow_write=True)
    specs = get_tool_specs(str(repo_root), permissions)

    names = {spec.name for spec in specs}
    assert names == {"read_file", "write_file", "edit_file", "search_files", "list_directory"}

    write_spec = next(spec for spec in specs if spec.name == "write_file")
    result = write_spec.func(path="via_spec.txt", content="bound ok")

    assert result.success is True
    assert (repo_root / "via_spec.txt").read_text(encoding="utf-8") == "bound ok"


def test_get_tool_specs_respects_allow_write_false(repo_root):
    permissions = PermissionsConfig(allow_write=False)
    specs = get_tool_specs(str(repo_root), permissions)

    write_spec = next(spec for spec in specs if spec.name == "write_file")
    result = write_spec.func(path="via_spec.txt", content="should not land")

    assert result.success is False
    assert not (repo_root / "via_spec.txt").exists()


def test_get_tool_specs_edit_file_respects_allow_write_false(repo_root):
    permissions = PermissionsConfig(allow_write=False)
    specs = get_tool_specs(str(repo_root), permissions)

    edit_spec = next(spec for spec in specs if spec.name == "edit_file")
    result = edit_spec.func(path="main.py", old_text="return 42", new_text="return 999")

    assert result.success is False
    assert "allow_write" in result.error
    assert "return 42" in (repo_root / "main.py").read_text(encoding="utf-8")  # unchanged


def test_get_tool_specs_edit_file_works_when_allow_write_true(repo_root):
    permissions = PermissionsConfig(allow_write=True)
    specs = get_tool_specs(str(repo_root), permissions)

    edit_spec = next(spec for spec in specs if spec.name == "edit_file")
    result = edit_spec.func(path="main.py", old_text="return 42", new_text="return 999")

    assert result.success is True
    assert "return 999" in (repo_root / "main.py").read_text(encoding="utf-8")
