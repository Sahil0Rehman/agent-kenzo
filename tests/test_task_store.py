"""Tests for `kenzo.memory.task_store.TaskStore`: persistence of
`AgentController`'s `TaskState` dicts to `.kenzo/tasks/<task_id>.json` inside
a target repository.

Plain filesystem tests against a tmp directory -- no git repo, no model, no
tool registry needed.
"""

from __future__ import annotations

import json

from kenzo.memory.task_store import TaskStore


# ---------------------------------------------------------------------------
# save/load round trip
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips_exact_dict(tmp_path):
    store = TaskStore(str(tmp_path))
    state = {
        "task_id": "fix-login-bug",
        "goal": "Fix the login bug",
        "iteration": 3,
        "finished": False,
        "files_modified": ["main.py"],
    }
    store.save("fix-login-bug", state)

    loaded = store.load("fix-login-bug")
    assert loaded == state


def test_save_creates_tasks_dir_and_json_file_on_disk(tmp_path):
    store = TaskStore(str(tmp_path))
    store.save("task-1", {"goal": "g"})

    tasks_dir = tmp_path / ".kenzo" / "tasks"
    assert tasks_dir.is_dir()
    assert (tasks_dir / "task-1.json").exists()

    on_disk = json.loads((tasks_dir / "task-1.json").read_text(encoding="utf-8"))
    assert on_disk == {"goal": "g"}


def test_save_overwrites_existing_task_with_same_id(tmp_path):
    store = TaskStore(str(tmp_path))
    store.save("task-1", {"goal": "g", "iteration": 0})
    store.save("task-1", {"goal": "g", "iteration": 5})

    assert store.load("task-1") == {"goal": "g", "iteration": 5}


# ---------------------------------------------------------------------------
# list_tasks summaries
# ---------------------------------------------------------------------------


def test_list_tasks_returns_summaries_for_every_saved_task(tmp_path):
    store = TaskStore(str(tmp_path))
    store.save("task-a", {"goal": "goal A", "finished": True, "iteration": 4})
    store.save("task-b", {"goal": "goal B", "finished": False, "iteration": 1})

    summaries = store.list_tasks()
    by_id = {s["task_id"]: s for s in summaries}

    assert set(by_id) == {"task-a", "task-b"}
    assert by_id["task-a"] == {"task_id": "task-a", "goal": "goal A", "finished": True, "iteration": 4}
    assert by_id["task-b"] == {"task_id": "task-b", "goal": "goal B", "finished": False, "iteration": 1}


def test_list_tasks_empty_when_no_tasks_dir_exists(tmp_path):
    store = TaskStore(str(tmp_path))
    assert store.list_tasks() == []


def test_list_tasks_skips_malformed_json_and_non_dict_payloads(tmp_path):
    store = TaskStore(str(tmp_path))
    store.save("good-task", {"goal": "g", "finished": False, "iteration": 0})

    tasks_dir = tmp_path / ".kenzo" / "tasks"
    (tasks_dir / "corrupt.json").write_text("{not valid json", encoding="utf-8")
    (tasks_dir / "not_a_dict.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    summaries = store.list_tasks()
    assert len(summaries) == 1
    assert summaries[0]["task_id"] == "good-task"


def test_list_tasks_defaults_missing_fields_to_none(tmp_path):
    store = TaskStore(str(tmp_path))
    store.save("bare-task", {})

    summaries = store.list_tasks()
    assert summaries[0] == {"task_id": "bare-task", "goal": "", "finished": None, "iteration": None}


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_removes_task_and_returns_true(tmp_path):
    store = TaskStore(str(tmp_path))
    store.save("to-delete", {"goal": "g"})

    assert store.delete("to-delete") is True
    assert store.load("to-delete") is None
    assert store.list_tasks() == []


def test_delete_nonexistent_task_returns_false(tmp_path):
    store = TaskStore(str(tmp_path))
    assert store.delete("never-existed") is False


# ---------------------------------------------------------------------------
# graceful None on missing/malformed files
# ---------------------------------------------------------------------------


def test_load_returns_none_for_missing_task(tmp_path):
    store = TaskStore(str(tmp_path))
    assert store.load("does-not-exist") is None


def test_load_returns_none_for_malformed_json(tmp_path):
    store = TaskStore(str(tmp_path))
    tasks_dir = tmp_path / ".kenzo" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "broken.json").write_text("{this is not json", encoding="utf-8")

    assert store.load("broken") is None


# ---------------------------------------------------------------------------
# task-id sanitization: path-traversal-ish ids can never escape tasks_dir
# ---------------------------------------------------------------------------


def test_task_id_sanitization_prevents_path_traversal_escape(tmp_path):
    store = TaskStore(str(tmp_path))
    store.save("../../evil", {"goal": "should stay contained"})

    tasks_dir = tmp_path / ".kenzo" / "tasks"
    json_files = list(tasks_dir.glob("*.json"))
    assert len(json_files) == 1
    assert json_files[0].parent.resolve() == tasks_dir.resolve()

    # Nothing escaped above/around tasks_dir.
    assert not (tmp_path / "evil").exists()
    assert not (tmp_path / ".kenzo" / "evil").exists()
    assert not (tmp_path.parent / "evil").exists()

    # And the same raw id used for save() correctly loads it back, since
    # sanitization is applied identically in both save() and load().
    assert store.load("../../evil") == {"goal": "should stay contained"}


def test_task_id_sanitization_replaces_path_separators_with_underscore(tmp_path):
    store = TaskStore(str(tmp_path))
    store.save("nested/path/id", {"goal": "x"})

    tasks_dir = tmp_path / ".kenzo" / "tasks"
    json_files = list(tasks_dir.glob("*.json"))
    assert len(json_files) == 1
    assert "/" not in json_files[0].name
    assert store.load("nested/path/id") == {"goal": "x"}


def test_task_id_sanitization_handles_empty_and_fully_stripped_ids(tmp_path):
    store = TaskStore(str(tmp_path))
    # An id made ENTIRELY of characters that get stripped/replaced.
    store.save("///", {"goal": "all-stripped"})
    # Should not raise, and should be retrievable via the identical raw id.
    assert store.load("///") == {"goal": "all-stripped"}


# ---------------------------------------------------------------------------
# generate_task_id
# ---------------------------------------------------------------------------


def test_generate_task_id_is_filesystem_safe_and_seed_derived(tmp_path):
    store = TaskStore(str(tmp_path))
    task_id = store.generate_task_id("Fix the login bug!!")

    assert task_id.startswith("fix-the-login-bug")
    # Filesystem-safe: no spaces, no punctuation beyond '-'/'_'.
    assert " " not in task_id
    assert "!" not in task_id


def test_generate_task_id_is_unique_for_identical_seed_text(tmp_path):
    store = TaskStore(str(tmp_path))
    first = store.generate_task_id("same request")
    second = store.generate_task_id("same request")
    assert first != second


def test_generate_task_id_falls_back_to_task_prefix_for_empty_seed(tmp_path):
    store = TaskStore(str(tmp_path))
    task_id = store.generate_task_id("   ")
    assert task_id.startswith("task-")
