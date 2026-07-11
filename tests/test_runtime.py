"""Tests for `kenzo.core.runtime.AgentRuntime`: the top-level entry point
that wires config, memory, tools, the model router, and the compiled graph
together and drives a single request end-to-end.

`KENZO_HOME` is pointed at an isolated tmp dir via the `kenzo_home` fixture
so nothing here ever touches a real `~/.kenzo/config.yaml`, and
`AgentRuntime` is always constructed with an explicit, hand-built
`KenzoConfig` (never `load_config()`).

Faking approach: rather than monkeypatching `litellm.completion` (which
would require also satisfying `ModelRouter.route()`'s provider/model/api_key
plumbing), this file monkeypatches `kenzo.models.router.ModelRouter.complete`
directly at the class level with a scripted, `TaskType`-keyed stand-in. This
is simpler and more reliable here: `AgentRuntime`/`ModelRouter` construction
still exercises the real config/provider validation code paths, but every
actual "model call" made through the compiled graph is deterministic and
network-free.

Tools and memory are always real (`build_default_registry` /
`RepositoryMemory` via `AgentRuntime`'s own construction), running against
the real tmp git repo from the `repo_root` fixture, so real edits really
land on disk and `.kenzo/memory.md` is really read/written.

Covers the runtime-level hardening fixes:
- `AgentRuntime.run()` uses `graph.stream(..., stream_mode="updates")` so a
  crash mid-pipeline still returns everything completed so far
  (`crashed=True`, `error` set, `completed_steps` accurate) instead of
  raising and discarding real work already done on disk.
- `AgentRuntime._record_completed_task()` only ever appends to
  `.kenzo/memory.md` after an approved run -- never on an unapproved or
  crashed one.
- `AgentRuntime.__init__` always calls `self.memory.ensure_kenzo_dir()`.
"""

from __future__ import annotations

import json

from kenzo.config.settings import KenzoConfig, PermissionsConfig, ProviderConfig
from kenzo.core.runtime import AgentRuntime
from kenzo.core.schemas import TaskType
from kenzo.models.router import ModelRouter


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding (see test_router.py's pattern).
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeModelResponse:
    """Minimal stand-in shaped like a real litellm ModelResponse."""

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


def _make_scripted_complete(scripts: dict):
    """Build a function suitable for `monkeypatch.setattr(ModelRouter, "complete", ...)`.

    `scripts` maps TaskType -> list of items, where each item is either a
    `_FakeModelResponse` to return or an `Exception` instance to raise. Items
    are consumed in order per task_type, independent of the other task types.
    """
    queues = {task_type: list(items) for task_type, items in scripts.items()}

    def scripted_complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        queue = queues.get(task_type)
        if not queue:
            raise AssertionError(f"scripted ModelRouter.complete ran out of responses for {task_type}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return scripted_complete


def _build_config() -> KenzoConfig:
    return KenzoConfig(
        providers={"nemotron": ProviderConfig(api_key="fake-key", default_model="fake-model")},
        permissions=PermissionsConfig(allow_write=True, allow_shell_exec=False),
    )


def _plan_response(goal="Fix simple bug"):
    return _FakeModelResponse(
        json.dumps(
            {
                "goal": goal,
                "steps": [{"id": 1, "description": "fix main.py", "rationale": "bug"}],
                "context_notes": "",
            }
        )
    )


def _research_response():
    return _FakeModelResponse(
        json.dumps(
            {
                "summary": "main.py returns the wrong value",
                "relevant_files": ["main.py"],
                "snippets": {},
                "open_questions": [],
            }
        )
    )


def _coder_edit_responses(old_text: str, new_text: str, call_id: str = "call_1"):
    edit_call = _FakeToolCall(
        call_id, "edit_file", json.dumps({"path": "main.py", "old_text": old_text, "new_text": new_text})
    )
    return [
        _FakeModelResponse(None, tool_calls=[edit_call]),
        _FakeModelResponse(f"Changed main.py: {old_text!r} -> {new_text!r}."),
    ]


def _tester_response(passed: bool, summary: str):
    return _FakeModelResponse(
        json.dumps({"passed": passed, "summary": summary, "command": "", "raw_output": ""})
    )


def _reviewer_response(approved: bool, issues=None, summary="reviewed"):
    return _FakeModelResponse(
        json.dumps({"approved": approved, "summary": summary, "issues": issues or []})
    )


# ---------------------------------------------------------------------------
# (a) Full successful "fix simple bug" run
# ---------------------------------------------------------------------------


def test_runtime_full_successful_run_applies_edit_and_writes_memory(repo_root, kenzo_home, monkeypatch):
    config = _build_config()
    scripts = {
        TaskType.PLANNING: [_plan_response()],
        TaskType.RESEARCH: [_research_response()],
        TaskType.CODING: _coder_edit_responses("return 42", "return 43"),
        TaskType.DEBUGGING: [_tester_response(True, "tests pass")],
        TaskType.REVIEW: [_reviewer_response(True)],
    }
    monkeypatch.setattr(ModelRouter, "complete", _make_scripted_complete(scripts))

    runtime = AgentRuntime(repo_path=str(repo_root), config=config)
    result = runtime.run("Fix simple bug")

    assert result["crashed"] is False
    assert result["completed_steps"] == ["planner", "researcher", "coder", "tester", "reviewer"]

    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 43\n"

    memory_text = (repo_root / ".kenzo" / "memory.md").read_text(encoding="utf-8")
    assert "Fix simple bug" in memory_text


# ---------------------------------------------------------------------------
# (b) Hard gate through the full runtime + memory NOT written when unapproved
# ---------------------------------------------------------------------------


def test_runtime_hard_gate_blocks_approval_and_skips_memory_write(repo_root, kenzo_home, monkeypatch):
    config = _build_config()
    # The hard gate overrides the reviewer's "approved: true" to False on
    # every pass (test_result.passed stays False throughout), so the retry
    # loop runs until MAX_REVIEW_RETRIES is reached -- script two full
    # coder/tester/reviewer rounds so the fake router never runs dry.
    scripts = {
        TaskType.PLANNING: [_plan_response()],
        TaskType.RESEARCH: [_research_response()],
        TaskType.CODING: _coder_edit_responses("return 42", "return 43", "call_1")
        + _coder_edit_responses("return 43", "return 44", "call_2"),
        TaskType.DEBUGGING: [
            _tester_response(False, "1 test failed"),
            _tester_response(False, "1 test failed again"),
        ],
        TaskType.REVIEW: [
            _reviewer_response(True, summary="looks fine to me"),
            _reviewer_response(True, summary="looks fine to me again"),
        ],
    }
    monkeypatch.setattr(ModelRouter, "complete", _make_scripted_complete(scripts))

    runtime = AgentRuntime(repo_path=str(repo_root), config=config)

    memory_path = repo_root / ".kenzo" / "memory.md"
    before = memory_path.read_text(encoding="utf-8")

    result = runtime.run("Fix simple bug")

    after = memory_path.read_text(encoding="utf-8")

    assert result["crashed"] is False
    assert result["review"]["approved"] is False
    assert any("overridden" in issue.lower() for issue in result["review"]["issues"])

    # The task never ended up approved, so no summary should have been
    # appended to memory.md -- it must be byte-for-byte unchanged.
    assert after == before


# ---------------------------------------------------------------------------
# (c) Crash safety: a mid-pipeline exception still returns completed work
# ---------------------------------------------------------------------------


def test_runtime_crash_preserves_completed_steps_and_disk_changes(repo_root, kenzo_home, monkeypatch):
    config = _build_config()
    scripts = {
        TaskType.PLANNING: [_plan_response()],
        TaskType.RESEARCH: [_research_response()],
        TaskType.CODING: _coder_edit_responses("return 42", "return 99"),
        TaskType.DEBUGGING: [RuntimeError("simulated transient provider failure")],
    }
    monkeypatch.setattr(ModelRouter, "complete", _make_scripted_complete(scripts))

    runtime = AgentRuntime(repo_path=str(repo_root), config=config)
    result = runtime.run("Fix simple bug")

    assert result["crashed"] is True
    assert "RuntimeError" in result["error"]
    assert "simulated transient provider failure" in result["error"]
    assert result["completed_steps"] == ["planner", "researcher", "coder"]

    # The coder's real edit record survived the crash...
    assert len(result["changes"]) == 1
    assert result["changes"][0]["file_path"] == "main.py"

    # ...and the actual file on disk was really changed.
    assert (repo_root / "main.py").read_text(encoding="utf-8") == "def main():\n    return 99\n"


# ---------------------------------------------------------------------------
# (d) AgentRuntime.__init__ ensures .kenzo/ exists before .run() is ever called
# ---------------------------------------------------------------------------


def test_runtime_init_ensures_kenzo_dir_before_run(repo_root, kenzo_home):
    config = _build_config()

    runtime = AgentRuntime(repo_path=str(repo_root), config=config)

    kenzo_dir = repo_root / ".kenzo"
    assert (kenzo_dir / "architecture.md").exists()
    assert (kenzo_dir / "memory.md").exists()
    assert (kenzo_dir / "rules.md").exists()
    assert runtime.repo_path == str(repo_root.resolve())
