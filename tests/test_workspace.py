"""Tests for Phase 11 ("multi-repo workspace") and the indexing
round-robin-by-language fix.

Phase 11 lets a single Kenzo run see (read-only, `--see`) and, if explicitly
told (`--also-edit`), edit more than one repository at once. Covered here:

- `kenzo.core.workspace`: `build_workspace`/`Workspace`/`WorkspaceRepo` --
  spec parsing, alias auto-derivation/dedup, and every error path.
- `kenzo.tools.cross_repo`: the five `*_other_repo_*` tool functions and
  `get_tool_specs`, all pure filesystem operations against real tmp repos.
- `kenzo.core.checkpoint.CrossRepoCheckpointingToolRegistry`: per-alias
  lazy checkpoint creation, independence across `--also-edit` repos, and
  `subset()`.
- `kenzo.intelligence.symbol_index`'s round-robin-by-language fix (a
  separate, unrelated fix landing in the same pass) plus the raised
  `max_indexed_files` default.
- Integration: a real `AgentController`/`WorkerManager` stack (only the
  router faked, mirroring `tests/test_controller_scenarios.py`'s and
  `tests/test_workers.py`'s established scaffolding) proving the whole
  "read legacy, fix current, port to sibling" workflow really happens on
  disk, that workspace context really reaches the decision prompt, and that
  a worker really gets read-only cross-repo access and never write access.

Fixture note: every fixture file whose content is later matched against an
`old_text` in a scripted `edit_file`/`edit_other_repo_file` call is written
with `newline=""` to disable Windows CRLF translation -- `Path.write_text`
without it silently turns '\n' into '\r\n', which then never exact-matches
an `old_text` written with plain '\n' in the test source.
"""

from __future__ import annotations

import json

import pytest

from kenzo.config.settings import (
    IndexingConfig,
    KenzoConfig,
    PermissionsConfig,
    WorkersConfig,
)
from kenzo.core.checkpoint import CheckpointManager, CrossRepoCheckpointingToolRegistry
from kenzo.core.controller import AgentController, ControllerAction
from kenzo.core.workspace import Workspace, WorkspaceError, WorkspaceRepo, build_workspace
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools import cross_repo
from kenzo.tools.base import ToolRegistry
from kenzo.tools.cross_repo import (
    READ_TOOL_NAMES,
    WRITE_TOOL_NAMES,
    edit_other_repo_file,
    get_tool_specs,
    list_other_repo_directory,
    read_other_repo_file,
    search_other_repo_files,
    write_other_repo_file,
)
from kenzo.tools.registry import build_default_registry
from kenzo.workers.manager import WorkerManager
from kenzo.workers.models import WorkerTask, WorkerType
from kenzo.workers.worker_agent import tool_names_for


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding (same pattern as
# tests/test_controller_scenarios.py / tests/test_workers.py).
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeModelResponse:
    def __init__(self, content, tool_calls=None):
        self.choices = [_FakeChoice(_FakeMessage(content, tool_calls))]


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _FakeFunction(name, arguments)


def _is_decide_call(tool_choice) -> bool:
    return (
        isinstance(tool_choice, dict)
        and tool_choice.get("function", {}).get("name") == "decide_next_action"
    )


class _ControllerScriptedRouter:
    """Fake router for `AgentController`: a dedicated ordered queue for the
    forced `decide_next_action` decision calls, and a separate flat ordered
    queue for every action-execution call. See
    tests/test_controller_scenarios.py's module docstring for the full
    rationale -- copied here rather than imported to keep this file
    self-contained, matching this codebase's existing per-file convention.
    """

    def __init__(self, decisions, actions):
        self._decisions = list(decisions)
        self._actions = list(actions)
        self.decision_calls: list[dict] = []
        self.action_calls: list[dict] = []

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        snapshot = [dict(m) for m in messages]
        if _is_decide_call(tool_choice):
            self.decision_calls.append({"task_type": task_type, "messages": snapshot})
            if not self._decisions:
                raise AssertionError("ran out of scripted decisions")
            return self._decisions.pop(0)

        self.action_calls.append({"task_type": task_type, "messages": snapshot})
        if not self._actions:
            raise AssertionError("ran out of scripted action responses")
        return self._actions.pop(0)


class _WorkerScriptedRouter:
    """Ordered-queue fake router for a single-threaded worker run (see
    tests/test_workers.py's identically-named class)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        snapshot = [dict(m) for m in messages]
        self.calls.append({"task_type": task_type, "messages": snapshot})
        if not self._responses:
            raise AssertionError("ran out of scripted worker responses")
        return self._responses.pop(0)


def _decision(action: ControllerAction, rationale: str, target: str = ""):
    call = _FakeToolCall(
        f"decide-{action.value}",
        "decide_next_action",
        json.dumps({"action": action.value, "rationale": rationale, "target": target}),
    )
    return _FakeModelResponse(None, tool_calls=[call])


def _tool_call_response(call_id: str, tool_name: str, arguments: dict):
    return _FakeModelResponse(None, tool_calls=[_FakeToolCall(call_id, tool_name, json.dumps(arguments))])


def _final_text(text: str):
    return _FakeModelResponse(text)


def _write(path, content: str) -> None:
    """Write a fixture file WITHOUT newline translation (see module
    docstring) -- always use this instead of a bare `Path.write_text` for
    any file that will later be matched via an exact-match `old_text`."""
    path.write_text(content, encoding="utf-8", newline="")


# ===========================================================================
# 1. kenzo.core.workspace: build_workspace / Workspace
# ===========================================================================


def test_build_workspace_primary_only_has_no_other_repos(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()

    ws = build_workspace(str(primary))

    assert ws.others == []
    assert ws.has_any_other_repo() is False
    assert ws.has_writable_other_repo() is False
    assert ws.readable_aliases() == []
    assert ws.writable_aliases() == []


def test_see_bare_path_auto_derives_alias_and_is_read_only(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    legacy = tmp_path / "CIDSaas"
    legacy.mkdir()

    ws = build_workspace(str(primary), see=[str(legacy)])

    assert len(ws.others) == 1
    repo = ws.others[0]
    assert repo.alias == "CIDSaas"
    assert repo.writable is False
    assert repo.path == str(legacy.resolve())
    assert ws.has_any_other_repo() is True
    assert ws.has_writable_other_repo() is False


def test_also_edit_explicit_alias_is_used_and_writable(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    sibling = tmp_path / "sibling_service"
    sibling.mkdir()

    ws = build_workspace(str(primary), also_edit=[f"sib={sibling}"])

    entry = ws.find("sib")
    assert entry is not None
    assert entry.alias == "sib"
    assert entry.writable is True
    assert entry.path == str(sibling.resolve())
    assert ws.has_writable_other_repo() is True
    assert ws.writable_aliases() == ["sib"]


def test_two_see_repos_with_same_folder_name_get_deduplicated_aliases(tmp_path):
    """The exact real-world case this feature was built for: multiple
    sprint/release-branch checkouts of the same service sharing a folder
    name, e.g. Sprint53/CIDSaas/CIDSaas and june-release-2026/CIDSaas
    /CIDSaas."""
    primary = tmp_path / "primary"
    primary.mkdir()
    sprint53 = tmp_path / "Sprint53" / "CIDSaas"
    sprint53.mkdir(parents=True)
    june = tmp_path / "june-release-2026" / "CIDSaas"
    june.mkdir(parents=True)

    ws = build_workspace(str(primary), see=[str(sprint53), str(june)])

    assert ws.readable_aliases() == ["CIDSaas", "CIDSaas-2"]
    first = ws.find("CIDSaas")
    second = ws.find("CIDSaas-2")
    assert first is not None and second is not None
    assert first.path == str(sprint53.resolve())
    assert second.path == str(june.resolve())
    assert first.writable is False
    assert second.writable is False


def test_explicit_alias_collision_raises_workspace_error(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    dir1 = tmp_path / "one"
    dir1.mkdir()
    dir2 = tmp_path / "two"
    dir2.mkdir()

    with pytest.raises(WorkspaceError):
        build_workspace(str(primary), see=[f"a={dir1}", f"a={dir2}"])


def test_nonexistent_path_raises_workspace_error_with_clear_message(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    missing = tmp_path / "does_not_exist_anywhere"

    with pytest.raises(WorkspaceError, match="does not exist"):
        build_workspace(str(primary), see=[str(missing)])

    with pytest.raises(WorkspaceError, match="does not exist"):
        build_workspace(str(primary), also_edit=[str(missing)])


def test_see_path_that_is_the_primary_repo_itself_raises(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()

    with pytest.raises(WorkspaceError, match="primary repo"):
        build_workspace(str(primary), see=[str(primary)])


def test_also_edit_path_that_is_the_primary_repo_itself_raises(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()

    with pytest.raises(WorkspaceError, match="primary repo"):
        build_workspace(str(primary), also_edit=[str(primary)])


def test_workspace_find_returns_none_for_unknown_and_correct_repo_for_known(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    legacy = tmp_path / "legacy_repo"
    legacy.mkdir()

    ws = build_workspace(str(primary), see=[str(legacy)])

    assert ws.find("does-not-exist") is None
    found = ws.find("legacy_repo")
    assert isinstance(found, WorkspaceRepo)
    assert found.path == str(legacy.resolve())
    assert found.writable is False


# ===========================================================================
# 1b. Workspace.add_reference -- mid-session dynamic discovery
# ===========================================================================


def test_add_reference_registers_a_new_read_only_repo(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    discovered = tmp_path / "sprint52_cidsaas"
    discovered.mkdir()

    ws = Workspace(primary_path=str(primary.resolve()))
    assert ws.has_any_other_repo() is False

    entry = ws.add_reference(str(discovered))

    assert entry.writable is False
    assert entry.path == str(discovered.resolve())
    assert ws.has_any_other_repo() is True
    assert ws.find(entry.alias) is entry


def test_add_reference_is_idempotent_for_the_same_path(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    discovered = tmp_path / "sprint52_cidsaas"
    discovered.mkdir()

    ws = Workspace(primary_path=str(primary.resolve()))
    first = ws.add_reference(str(discovered))
    second = ws.add_reference(str(discovered))

    assert first is second
    assert len(ws.others) == 1


def test_add_reference_with_explicit_alias_uses_it(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    discovered = tmp_path / "some_folder"
    discovered.mkdir()

    ws = Workspace(primary_path=str(primary.resolve()))
    entry = ws.add_reference(str(discovered), alias="entities")

    assert entry.alias == "entities"


def test_add_reference_nonexistent_path_raises(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    ws = Workspace(primary_path=str(primary.resolve()))

    with pytest.raises(WorkspaceError):
        ws.add_reference(str(tmp_path / "does-not-exist"))


def test_add_reference_primary_path_raises(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    ws = Workspace(primary_path=str(primary.resolve()))

    with pytest.raises(WorkspaceError):
        ws.add_reference(str(primary))


def test_add_reference_never_produces_a_writable_entry_even_with_an_alias_matching_an_also_edit_repo(tmp_path):
    """add_reference is the ONLY path a model can reach on its own (no user
    consent needed); it must never be able to grant write access, no matter
    what alias is requested."""
    primary = tmp_path / "primary"
    primary.mkdir()
    discovered = tmp_path / "other"
    discovered.mkdir()

    ws = Workspace(primary_path=str(primary.resolve()))
    entry = ws.add_reference(str(discovered), alias="sibling")

    assert entry.writable is False


def test_add_reference_repo_tool_success_and_error(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    discovered = tmp_path / "discovered_repo"
    discovered.mkdir()

    ws = Workspace(primary_path=str(primary.resolve()))
    result = cross_repo.add_reference_repo(str(discovered), ws)

    assert result.success is True
    assert result.data["writable"] is False
    assert ws.find(result.data["alias"]) is not None

    bad_result = cross_repo.add_reference_repo(str(tmp_path / "nope"), ws)
    assert bad_result.success is False


# ===========================================================================
# 2. kenzo.tools.cross_repo: real tmp repos, real files, no mocking
# ===========================================================================


@pytest.fixture
def cross_repo_workspace(tmp_path):
    """primary (unused directly by cross_repo, but required by
    build_workspace) + legacy (--see, read-only) + sibling (--also-edit,
    writable), each seeded with one real file."""
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "Foo.java").write_text("primary placeholder\n", encoding="utf-8")

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write(legacy / "Foo.java", "public class Foo {\n    // reference implementation\n}\n")

    sibling = tmp_path / "sibling"
    sibling.mkdir()
    _write(sibling / "Foo.java", "public class Foo {\n    // sibling placeholder\n}\n")

    ws = build_workspace(str(primary), see=[f"legacy={legacy}"], also_edit=[f"sibling={sibling}"])
    return {"primary": primary, "legacy": legacy, "sibling": sibling, "workspace": ws}


def test_read_list_search_against_a_see_repo_succeed(cross_repo_workspace):
    ws = cross_repo_workspace["workspace"]

    read_result = read_other_repo_file("legacy", "Foo.java", ws)
    assert read_result.success is True
    assert "reference implementation" in read_result.data["content"]

    list_result = list_other_repo_directory("legacy", ".", ws)
    assert list_result.success is True
    names = [e["name"] for e in list_result.data["entries"]]
    assert "Foo.java" in names

    search_result = search_other_repo_files("legacy", "reference implementation", ws)
    assert search_result.success is True
    matches = search_result.data["matches"]
    assert len(matches) == 1
    assert matches[0]["path"] == "Foo.java"


def test_read_list_search_against_unknown_repo_alias_fails_clearly(cross_repo_workspace):
    ws = cross_repo_workspace["workspace"]

    for fn, kwargs in (
        (read_other_repo_file, {"path": "Foo.java"}),
        (list_other_repo_directory, {"path": "."}),
        (search_other_repo_files, {"pattern": "x"}),
    ):
        result = fn(repo="ghost", workspace=ws, **kwargs)
        assert result.success is False
        assert "unknown workspace repo" in result.error.lower()
        assert "legacy" in result.error
        assert "sibling" in result.error


def test_write_and_edit_against_see_repo_refused_and_disk_untouched(cross_repo_workspace):
    ws = cross_repo_workspace["workspace"]
    legacy = cross_repo_workspace["legacy"]
    permissions = PermissionsConfig(allow_write=True)

    original_content = (legacy / "Foo.java").read_text(encoding="utf-8")
    original_entries = sorted(p.name for p in legacy.iterdir())

    write_result = write_other_repo_file("legacy", "new_file.txt", "hacked content", ws, permissions)
    assert write_result.success is False
    assert "not writable" in write_result.error.lower()

    edit_result = edit_other_repo_file(
        "legacy", "Foo.java", "reference implementation", "HACKED", ws, permissions
    )
    assert edit_result.success is False
    assert "not writable" in edit_result.error.lower()

    # Re-read from disk (not the returned dicts) -- prove neither call ever
    # reached filesystem.write_file/edit_file.
    assert not (legacy / "new_file.txt").exists()
    assert sorted(p.name for p in legacy.iterdir()) == original_entries
    assert (legacy / "Foo.java").read_text(encoding="utf-8") == original_content


def test_write_and_edit_against_also_edit_repo_succeed_and_disk_changes(cross_repo_workspace):
    ws = cross_repo_workspace["workspace"]
    sibling = cross_repo_workspace["sibling"]
    permissions = PermissionsConfig(allow_write=True)

    write_result = write_other_repo_file("sibling", "new_file.txt", "genuinely written", ws, permissions)
    assert write_result.success is True
    assert (sibling / "new_file.txt").read_text(encoding="utf-8") == "genuinely written"

    edit_result = edit_other_repo_file(
        "sibling", "Foo.java", "sibling placeholder", "sibling FIXED", ws, permissions
    )
    assert edit_result.success is True
    new_content = (sibling / "Foo.java").read_text(encoding="utf-8")
    assert "sibling FIXED" in new_content
    assert "sibling placeholder" not in new_content


def test_get_tool_specs_empty_workspace_has_discovery_and_read_tools_but_no_write_tools(tmp_path):
    """An empty Workspace (zero --see/--also-edit given) still gets
    add_reference_repo + the 3 read tools registered -- this is what makes
    MID-SESSION discovery possible (the model can call add_reference_repo
    the first time a request mentions another repo's path, rather than that
    capability only existing if a repo happened to be named via a CLI flag
    before the session even started). Write tools stay absent: add_reference
    _repo can only ever add READ-ONLY entries, so an empty workspace can
    never need write_other_repo_file/edit_other_repo_file."""
    primary = tmp_path / "primary"
    primary.mkdir()
    ws = Workspace(primary_path=str(primary.resolve()))

    specs = get_tool_specs(ws, PermissionsConfig(allow_write=True))
    names = {spec.name for spec in specs}

    assert names == {"add_reference_repo", "read_other_repo_file", "list_other_repo_directory", "search_other_repo_files"}
    assert "write_other_repo_file" not in names
    assert "edit_other_repo_file" not in names


def test_get_tool_specs_see_only_workspace_has_read_tools_but_not_write_tool_names(cross_repo_workspace, tmp_path):
    primary = tmp_path / "primary2"
    primary.mkdir()
    legacy = tmp_path / "legacy2"
    legacy.mkdir()
    ws = build_workspace(str(primary), see=[f"legacy={legacy}"])

    specs = get_tool_specs(ws, PermissionsConfig(allow_write=True))
    names = {s.name for s in specs}

    assert names == set(READ_TOOL_NAMES)
    for write_name in WRITE_TOOL_NAMES:
        assert write_name not in names


def test_get_tool_specs_also_edit_workspace_has_all_five_tools(cross_repo_workspace):
    ws = cross_repo_workspace["workspace"]

    specs = get_tool_specs(ws, PermissionsConfig(allow_write=True))
    names = {s.name for s in specs}

    assert names == set(READ_TOOL_NAMES) | set(WRITE_TOOL_NAMES)


# ===========================================================================
# 3. kenzo.core.checkpoint.CrossRepoCheckpointingToolRegistry
# ===========================================================================


def _cross_repo_registry(workspace, permissions) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many(cross_repo.get_tool_specs(workspace, permissions))
    return registry


def test_edit_other_repo_file_lazily_creates_checkpoint_and_calls_before_edit(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    _write(sibling / "Foo.java", "old content\n")

    ws = build_workspace(str(primary), also_edit=[f"sibling={sibling}"])
    permissions = PermissionsConfig(allow_write=True)
    registry = _cross_repo_registry(ws, permissions)

    checkpoint = CheckpointManager(str(sibling))
    assert checkpoint.has_active_checkpoint() is False

    wrapped = CrossRepoCheckpointingToolRegistry(registry, {"sibling": checkpoint})
    result = wrapped.execute(
        "edit_other_repo_file",
        {"repo": "sibling", "path": "Foo.java", "old_text": "old content", "new_text": "new content"},
    )

    assert result.success is True
    assert checkpoint.has_active_checkpoint() is True

    latest = checkpoint.load_latest()
    assert latest is not None
    # Non-git tmp dir -> "files" backend; before_edit() must have recorded
    # the exact path it was called with, snapshotting the pre-edit content.
    assert "Foo.java" in latest.get("files", {})
    assert (sibling / "Foo.java").read_text(encoding="utf-8") == "new content\n"


def test_two_also_edit_repos_get_fully_independent_checkpoints(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    repo_a = tmp_path / "repo_a"
    repo_a.mkdir()
    _write(repo_a / "A.java", "a content\n")
    repo_b = tmp_path / "repo_b"
    repo_b.mkdir()
    _write(repo_b / "B.java", "b content\n")

    ws = build_workspace(str(primary), also_edit=[f"a={repo_a}", f"b={repo_b}"])
    permissions = PermissionsConfig(allow_write=True)
    registry = _cross_repo_registry(ws, permissions)

    checkpoint_a = CheckpointManager(str(repo_a))
    checkpoint_b = CheckpointManager(str(repo_b))
    wrapped = CrossRepoCheckpointingToolRegistry(registry, {"a": checkpoint_a, "b": checkpoint_b})

    result = wrapped.execute(
        "edit_other_repo_file",
        {"repo": "a", "path": "A.java", "old_text": "a content", "new_text": "a changed"},
    )
    assert result.success is True

    assert checkpoint_a.has_active_checkpoint() is True
    assert checkpoint_b.has_active_checkpoint() is False
    assert checkpoint_b.load_latest() is None


def test_subset_preserves_checkpoints_by_alias_and_wraps_wrapped_subset(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    _write(sibling / "Foo.java", "content\n")

    ws = build_workspace(str(primary), also_edit=[f"sibling={sibling}"])
    permissions = PermissionsConfig(allow_write=True)
    registry = _cross_repo_registry(ws, permissions)
    checkpoints_by_alias = {"sibling": CheckpointManager(str(sibling))}
    wrapped = CrossRepoCheckpointingToolRegistry(registry, checkpoints_by_alias)

    subset_registry = wrapped.subset(["edit_other_repo_file"])

    assert isinstance(subset_registry, CrossRepoCheckpointingToolRegistry)
    assert subset_registry._checkpoints_by_alias is checkpoints_by_alias
    assert subset_registry.names() == ["edit_other_repo_file"]
    assert "read_other_repo_file" not in subset_registry.names()


# ===========================================================================
# 4. kenzo.intelligence.symbol_index: round-robin-by-language fix
# ===========================================================================


def test_round_robin_gives_both_languages_a_nonzero_share_when_truncated(tmp_path):
    """Reproduce the real-world scenario: a directory that sorts
    alphabetically before the Java source dir, holding far more files, must
    not starve Java out of the index entirely once max_files forces
    truncation."""
    repo = tmp_path / "enterprise_repo"
    repo.mkdir()

    # "WebContent" (capital W) sorts before "src" (lowercase s) under plain
    # ASCII/string comparison -- the exact real repo shape that exposed the
    # bug (see SymbolIndex.build's docstring).
    web_content = repo / "WebContent"
    web_content.mkdir()
    for i in range(30):
        _write(web_content / f"file{i:02d}.js", f"console.log({i});\n")

    src = repo / "src"
    src.mkdir()
    for i in range(10):
        _write(src / f"File{i:02d}.java", f"public class File{i:02d} {{}}\n")

    index = SymbolIndex(str(repo))
    index.build(ignored_dirs=[], max_files=15)

    assert len(index.files) == 15
    java_count = sum(1 for p in index.files if p.endswith(".java"))
    js_count = sum(1 for p in index.files if p.endswith(".js"))
    assert java_count > 0, "java was starved out of the index entirely -- round-robin fix regressed"
    assert js_count > 0
    assert java_count + js_count == 15


def test_total_files_found_and_truncated_report_true_combined_total(tmp_path):
    repo = tmp_path / "enterprise_repo2"
    repo.mkdir()
    web_content = repo / "WebContent"
    web_content.mkdir()
    for i in range(30):
        _write(web_content / f"file{i:02d}.js", f"console.log({i});\n")
    src = repo / "src"
    src.mkdir()
    for i in range(10):
        _write(src / f"File{i:02d}.java", f"public class File{i:02d} {{}}\n")

    index = SymbolIndex(str(repo))
    index.build(ignored_dirs=[], max_files=15)

    assert index.total_files_found == 40
    assert len(index.files) == 15
    assert index.truncated is True


def test_repo_fitting_entirely_under_max_files_is_complete_and_not_truncated(tmp_path):
    repo = tmp_path / "small_repo"
    repo.mkdir()
    _write(repo / "A.java", "public class A {}\n")
    _write(repo / "B.java", "public class B {}\n")
    _write(repo / "c.js", "console.log('c');\n")

    index = SymbolIndex(str(repo))
    index.build(ignored_dirs=[], max_files=50)

    assert index.total_files_found == 3
    assert index.truncated is False
    assert set(index.files.keys()) == {"A.java", "B.java", "c.js"}


def test_indexing_config_max_indexed_files_default_raised_to_8000():
    assert IndexingConfig().max_indexed_files == 8000


# ===========================================================================
# 5. Integration: real AgentController/WorkerManager stack, router faked
# ===========================================================================


def test_read_legacy_fix_primary_port_to_sibling_end_to_end(tmp_path):
    """The core "read legacy, fix current, port to sibling" workflow the
    whole Phase 11 feature exists for, made real end to end: real
    AgentController, real tools (build_default_registry with a real
    workspace), real RepositoryMemory, real KenzoConfig -- only the router
    is scripted."""
    primary = tmp_path / "primary"
    primary.mkdir()
    _write(primary / "Foo.java", "public class Foo {\n    int compute() { return 41; }\n}\n")

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write(legacy / "Foo.java", "public class Foo {\n    int compute() { return 42; }\n}\n")

    sibling = tmp_path / "sibling"
    sibling.mkdir()
    _write(sibling / "Foo.java", "public class Foo {\n    int compute() { return 41; }\n}\n")

    ws = build_workspace(str(primary), see=[f"legacy={legacy}"], also_edit=[f"sibling={sibling}"])
    tools = build_default_registry(str(primary), PermissionsConfig(allow_write=True), workspace=ws)
    memory = RepositoryMemory(str(primary))
    config = KenzoConfig()

    decisions = [
        _decision(ControllerAction.READ, "Check how the legacy branch implemented compute().", "Foo.java"),
        _decision(ControllerAction.EDIT, "Apply the same fix to the primary repo.", "Foo.java"),
        _decision(ControllerAction.EDIT, "Port the same fix into the sibling repo too.", "Foo.java"),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(ControllerAction.FINISH, "Fixed in both the primary and sibling repos, verified."),
    ]

    actions = [
        # READ: consult the legacy repo's reference implementation.
        _tool_call_response("call-read-legacy", "read_other_repo_file", {"repo": "legacy", "path": "Foo.java"}),
        _final_text("Legacy Foo.compute() returns 42."),
        # EDIT #1: fix the primary repo to match.
        _tool_call_response(
            "call-edit-primary",
            "edit_file",
            {"path": "Foo.java", "old_text": "return 41;", "new_text": "return 42;"},
        ),
        _final_text("Fixed primary Foo.java to return 42."),
        # EDIT #2: port the identical fix into the sibling (--also-edit) repo.
        _tool_call_response(
            "call-edit-sibling",
            "edit_other_repo_file",
            {"repo": "sibling", "path": "Foo.java", "old_text": "return 41;", "new_text": "return 42;"},
        ),
        _final_text("Ported the same fix to the sibling repo."),
        # TEST: passes.
        _final_text(json.dumps({"passed": True, "summary": "compute() returns 42 everywhere", "command": "", "raw_output": ""})),
    ]

    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, config, workspace=ws)

    result = controller.run("Fix compute() to match the legacy branch, and port the same fix to the sibling repo")

    assert result["finished"] is True
    assert result["success"] is True

    expected_fixed = "public class Foo {\n    int compute() { return 42; }\n}\n"
    assert (primary / "Foo.java").read_text(encoding="utf-8") == expected_fixed
    assert (sibling / "Foo.java").read_text(encoding="utf-8") == expected_fixed
    # legacy (read-only) was never touched.
    assert (legacy / "Foo.java").read_text(encoding="utf-8") == "public class Foo {\n    int compute() { return 42; }\n}\n"

    # The sibling repo has its OWN, independent checkpoint -- proving
    # CrossRepoCheckpointingToolRegistry really wired a per-alias
    # CheckpointManager all the way through the real AgentController.
    sibling_checkpoint = CheckpointManager(str(sibling))
    sibling_latest = sibling_checkpoint.load_latest()
    assert sibling_latest is not None
    assert "Foo.java" in sibling_latest.get("files", {})

    # The primary repo's own checkpoint is independent of the sibling's.
    primary_checkpoint = CheckpointManager(str(primary))
    primary_latest = primary_checkpoint.load_latest()
    assert primary_latest is not None
    assert primary_latest.get("checkpoint_dir", "") != sibling_latest.get("checkpoint_dir", "")

    assert controller._also_edit_checkpoints["sibling"].has_active_checkpoint() is True


def test_workspace_context_reaches_decide_next_action_messages(tmp_path):
    """Mirrors how earlier phases proved vision_context/ide_context actually
    reach the decision prompt, not just that the render function works in
    isolation: capture the router's received messages on a real
    `_decide_next_action` call."""
    primary = tmp_path / "primary"
    primary.mkdir()
    legacy = tmp_path / "legacy_branch"
    legacy.mkdir()
    sibling = tmp_path / "sibling_service"
    sibling.mkdir()

    ws = build_workspace(str(primary), see=[f"legacy_branch={legacy}"], also_edit=[f"sibling_service={sibling}"])
    tools = build_default_registry(str(primary), PermissionsConfig(allow_write=True), workspace=ws)
    memory = RepositoryMemory(str(primary))
    config = KenzoConfig()

    router = _ControllerScriptedRouter(
        [_decision(ControllerAction.FINISH, "Nothing to do, just checking workspace context.")], []
    )
    controller = AgentController(router, tools, memory, config, workspace=ws)

    controller.run("Just checking that workspace context reaches the prompt")

    assert router.decision_calls, "expected at least one decide_next_action call"
    user_message = next(m["content"] for m in router.decision_calls[0]["messages"] if m["role"] == "user")

    assert "legacy_branch" in user_message
    assert "sibling_service" in user_message
    assert "READ-ONLY" in user_message
    assert "read+write" in user_message
    assert str(legacy.resolve()) in user_message
    assert str(sibling.resolve()) in user_message


def test_worker_gets_read_only_cross_repo_tools_never_write_tools(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write(legacy / "Foo.java", "public class Foo {\n    // reference compute() impl\n}\n")
    sibling = tmp_path / "sibling"
    sibling.mkdir()

    ws = build_workspace(str(primary), see=[f"legacy={legacy}"], also_edit=[f"sibling={sibling}"])

    # Direct assertion on tool_names_for: even though the workspace HAS an
    # --also-edit repo, a worker never gets the write tool names.
    for worker_type in (WorkerType.CODE_RESEARCH, WorkerType.BUG_ANALYSIS, WorkerType.ARCHITECTURE_CRITIC):
        names = tool_names_for(worker_type, workspace=ws)
        assert "read_other_repo_file" in names
        assert "search_other_repo_files" in names
        assert "list_other_repo_directory" in names
        assert "write_other_repo_file" not in names
        assert "edit_other_repo_file" not in names

    # Integration: a real WorkerManager.run_parallel, with workspace= set,
    # really lets the worker call read_other_repo_file/search_other_repo_files
    # against the --see repo end to end.
    tools = build_default_registry(str(primary), PermissionsConfig(allow_write=True), workspace=ws)
    findings_json = json.dumps(
        {
            "findings": "Legacy Foo.java has a reference compute() implementation.",
            "confidence": 0.9,
            "files": [],
            "recommendations": [],
        }
    )
    router = _WorkerScriptedRouter(
        [
            _tool_call_response("call-read", "read_other_repo_file", {"repo": "legacy", "path": "Foo.java"}),
            _tool_call_response("call-search", "search_other_repo_files", {"repo": "legacy", "pattern": "compute"}),
            _final_text(findings_json),
        ]
    )
    manager = WorkerManager(router, tools, WorkersConfig(), workspace=ws)
    task = WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description="Investigate the legacy Foo.java implementation")

    results = manager.run_parallel([task])

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert result.tool_calls_made == 2
    assert "Foo.java" in result.files


# ===========================================================================
# 6. Mid-session dynamic discovery end to end (a repo NEVER named via
#    --see/--also-edit, discovered purely because the user's own prompt
#    mentioned its path)
# ===========================================================================


def test_add_reference_repo_discovered_mid_task_via_real_controller(tmp_path):
    """The bare `kenzo` command (zero --see/--also-edit) still lets the
    model discover and read a repo the user's OWN request mentions inline --
    this is what makes free-form prompts like "...also check C:\\...\\Sprint52"
    work without pre-declaring anything on the command line."""
    primary = tmp_path / "primary"
    primary.mkdir()
    _write(primary / "AuthController.java", "public class AuthController {\n    // TODO\n}\n")

    discovered = tmp_path / "sprint52_cidsaas"
    discovered.mkdir()
    _write(discovered / "AuthEntity.java", "public class AuthEntity {\n    // entity fields live here\n}\n")

    # Exactly what session.py now does unconditionally, even with zero flags.
    ws = build_workspace(str(primary), see=[], also_edit=[])
    assert ws.has_any_other_repo() is False

    tools = build_default_registry(str(primary), PermissionsConfig(allow_write=True), workspace=ws)
    assert "add_reference_repo" in tools.names()

    memory = RepositoryMemory(str(primary))
    config = KenzoConfig()

    decisions = [
        _decision(ControllerAction.SEARCH, "The user mentioned a repo I don't have yet -- discover and check it.", str(discovered)),
        _decision(ControllerAction.FINISH, "Checked the entity repo as asked."),
    ]
    actions = [
        _tool_call_response("call-discover", "add_reference_repo", {"path": str(discovered)}),
        _tool_call_response("call-read", "read_other_repo_file", {"repo": discovered.name, "path": "AuthEntity.java"}),
        _final_text("Found AuthEntity.java -- entity fields live there as expected."),
    ]
    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, config, workspace=ws)

    result = controller.run(f"Fix the AuthController TODO -- also check {discovered} to see how the entity is defined, no need to edit it")

    assert result["finished"] is True
    # The live Workspace object was genuinely mutated mid-task.
    assert ws.has_any_other_repo() is True
    entry = ws.others[0]
    assert entry.path == str(discovered.resolve())
    assert entry.writable is False


def test_write_to_newly_discovered_repo_fails_closed_when_no_also_edit_was_ever_given(tmp_path):
    """A session started with ZERO --also-edit repos never has write_other
    _repo_file/edit_other_repo_file registered AT ALL (has_writable_other_repo()
    is False at registration and add_reference_repo can never change that) --
    an even stronger fail-closed outcome than a per-call refusal."""
    primary = tmp_path / "primary"
    primary.mkdir()
    discovered = tmp_path / "discovered"
    discovered.mkdir()
    _write(discovered / "f.txt", "original\n")

    ws = build_workspace(str(primary), see=[], also_edit=[])
    tools = build_default_registry(str(primary), PermissionsConfig(allow_write=True), workspace=ws)
    assert "edit_other_repo_file" not in tools.names()

    entry = ws.add_reference(str(discovered))
    result = tools.execute("edit_other_repo_file", {"repo": entry.alias, "path": "f.txt", "old_text": "original", "new_text": "hacked"})

    assert result.success is False
    assert "No tool named" in result.error
    assert (discovered / "f.txt").read_text(encoding="utf-8") == "original\n"


def test_write_to_newly_discovered_repo_refused_per_call_when_another_also_edit_repo_exists(tmp_path):
    """If the session DOES have write tools registered (because of a
    DIFFERENT --also-edit repo), a newly-discovered repo still can't be
    written to -- refused per-call this time, since the tool exists but the
    specific repo alias isn't in the writable set."""
    primary = tmp_path / "primary"
    primary.mkdir()
    other_writable = tmp_path / "other_writable"
    other_writable.mkdir()
    _write(other_writable / "x.txt", "x\n")
    discovered = tmp_path / "discovered"
    discovered.mkdir()
    _write(discovered / "f.txt", "original\n")

    ws = build_workspace(str(primary), see=[], also_edit=[f"other={other_writable}"])
    tools = build_default_registry(str(primary), PermissionsConfig(allow_write=True), workspace=ws)
    assert "edit_other_repo_file" in tools.names()

    entry = ws.add_reference(str(discovered))
    result = tools.execute("edit_other_repo_file", {"repo": entry.alias, "path": "f.txt", "old_text": "original", "new_text": "hacked"})

    assert result.success is False
    assert "not writable" in result.error
    assert "also-edit" in result.error
    assert (discovered / "f.txt").read_text(encoding="utf-8") == "original\n"
