"""Regression test: DELEGATE (Phase 9 parallel workers) must need strictly
FEWER outer-loop `AgentController` iterations than the old serial SEARCH-by-
SEARCH investigation style to gather equivalent information, for the class
of task the Phase 9 spec names -- "find why authentication sometimes fails"
-- spread across independent angles (controller/service code, an existing
test, and a recent config change).

This is a smaller, FASTER sibling of scripts/benchmark_workers.py -- see
that script for the full-size synthetic repo, real wall-clock timing (with
artificial per-call delays), and a printed BEFORE/AFTER comparison table.
Here we only assert the MECHANICAL claim (strictly fewer iterations, same
information gathered), so there are no `time.sleep()` calls and the repo is
tiny -- fast and fully deterministic, safe to run on every CI invocation. A
future change that accidentally breaks delegation's efficiency (e.g. workers
stop actually gathering their findings, or DELEGATE starts taking more than
one outer-loop iteration) will fail this test.

Faking approach: only the model/router layer is faked, following the exact
convention established by tests/test_controller_scenarios.py's
`_ControllerScriptedRouter` (this file duplicates the small
`_FakeMessage`/`_FakeChoice`/`_FakeModelResponse`/`_FakeToolCall` scaffolding
rather than importing it, matching how test_agents.py/test_graph.py/
test_runtime.py/test_controller_scenarios.py each keep their own copy).
Tools (`build_default_registry`) and memory (`RepositoryMemory`) are real,
running against the real tmp repo the `repo_root` fixture provides, extended
here with a few extra files representing the auth bug.
"""

from __future__ import annotations

import json
import threading

from kenzo.config.settings import KenzoConfig, PermissionsConfig
from kenzo.core.controller import AgentController, ControllerAction
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.registry import build_default_registry


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding (same pattern as
# tests/test_controller_scenarios.py's _ControllerScriptedRouter).
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


# ---------------------------------------------------------------------------
# Thread-safe scripted router (no artificial delay -- see module docstring).
#
# Thread-safety note (see kenzo.core.tool_loop.run_tool_loop /
# kenzo.core.controller.AgentController._propose_worker_tasks): a
# `decide_next_action`-forced call is identifiable via `tool_choice`;
# `_propose_worker_tasks` calls `router.complete_text(...)`, which passes NO
# `tools` kwarg at all (`tools is None` there specifically, vs a list for
# every `run_tool_loop`-driven call, worker or not). The DELEGATE scenario
# below runs 3 workers CONCURRENTLY from real OS threads -- there is no
# strict FIFO order across those 3 calls, so worker responses are served
# from a queue private to each worker (selected by inspecting message
# content for a marker embedded in that worker's own task description), and
# every mutable queue is protected by one `threading.Lock`.
# ---------------------------------------------------------------------------


class _BenchRouter:
    def __init__(self, decisions, actions, delegate_planning=None, worker_scripts=None):
        self._decisions = list(decisions)
        self._actions = list(actions)
        self._delegate_planning = list(delegate_planning or [])
        self._worker_scripts = {k: list(v) for k, v in (worker_scripts or {}).items()}
        self._lock = threading.Lock()

    def _worker_key_for(self, combined_content: str) -> str:
        for key in self._worker_scripts:
            if key in combined_content:
                return key
        raise AssertionError(
            f"could not determine which worker this call belongs to from content: {combined_content[:200]!r}"
        )

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        if _is_decide_call(tool_choice):
            with self._lock:
                if not self._decisions:
                    raise AssertionError("ran out of scripted decisions")
                return self._decisions.pop(0)

        combined = " ".join(str(m.get("content", "")) for m in messages)
        if "Investigation task:" in combined:
            key = self._worker_key_for(combined)
            with self._lock:
                queue = self._worker_scripts.get(key)
                if not queue:
                    raise AssertionError(f"ran out of scripted worker responses for {key!r}")
                return queue.pop(0)

        if tools is None:
            with self._lock:
                if not self._delegate_planning:
                    raise AssertionError("ran out of scripted delegate-planning responses")
                return self._delegate_planning.pop(0)

        with self._lock:
            if not self._actions:
                raise AssertionError("ran out of scripted action responses")
            return self._actions.pop(0)

    def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.complete(task_type, messages, **kwargs)
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# A small "authentication sometimes fails" bug, extending the shared
# repo_root fixture -- miniature sibling of scripts/benchmark_workers.py's
# full-size synthetic repo (same bug shape, far fewer files).
# ---------------------------------------------------------------------------

_ANGLE_MARKER = {"CODE": "[ANGLE:CODE]", "TEST": "[ANGLE:TEST]", "CONFIG": "[ANGLE:CONFIG]"}

_FIX_OLD_TEXT = "_REFRESH_WINDOW_SECONDS = 300"
_FIX_NEW_TEXT = "from auth.oidc_config import TOKEN_EXPIRY_SECONDS\n\n_REFRESH_WINDOW_SECONDS = TOKEN_EXPIRY_SECONDS"

_TEST_PASS_JSON = json.dumps({"passed": True, "summary": "refresh test passes", "command": "", "raw_output": ""})


def _add_auth_bug_files(repo_root) -> None:
    (repo_root / "auth").mkdir()
    (repo_root / "auth" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "auth" / "oidc_config.py").write_text(
        "# extended from 300 -> 3600 on 2026-06-01\nTOKEN_EXPIRY_SECONDS = 3600\n",
        encoding="utf-8",
    )
    (repo_root / "auth" / "session_manager.py").write_text(
        "# BUG: stale pre-2026-06-01 value, never updated to match oidc_config\n"
        "_REFRESH_WINDOW_SECONDS = 300\n\n\n"
        "class SessionManager:\n"
        "    def is_refreshable(self, session, now):\n"
        "        return (now - session['issued_at']) < _REFRESH_WINDOW_SECONDS\n",
        encoding="utf-8",
    )
    (repo_root / "auth" / "auth_controller.py").write_text(
        "from auth.session_manager import SessionManager\n\n\n"
        "class AuthController:\n"
        "    def __init__(self):\n"
        "        self.session_manager = SessionManager()\n",
        encoding="utf-8",
    )
    tests_auth = repo_root / "tests" / "auth"
    tests_auth.mkdir(parents=True)
    (repo_root / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (tests_auth / "__init__.py").write_text("", encoding="utf-8")
    (tests_auth / "test_session_refresh.py").write_text(
        "def test_refresh_succeeds_for_session_within_token_expiry():\n"
        "    pass  # asserts refresh() reports 'refreshed', not 'relogin_required'\n",
        encoding="utf-8",
    )
    # A little unrelated filler so this isn't a literal 2-file toy either.
    (repo_root / "billing.py").write_text("class InvoiceService:\n    pass\n", encoding="utf-8")
    (repo_root / "notifications.py").write_text("class EmailSender:\n    pass\n", encoding="utf-8")


def _real_controller(repo_root, router) -> AgentController:
    permissions = PermissionsConfig(allow_write=True)
    tools = build_default_registry(str(repo_root), permissions)
    memory = RepositoryMemory(str(repo_root))
    return AgentController(router, tools, memory, KenzoConfig())


def _before_scenario():
    """3 separate, sequential SEARCH/READ/ANALYZE actions -- one per angle
    -- then EDIT/TEST/FINISH."""
    decisions = [
        _decision(ControllerAction.SEARCH, "Angle 1/3: controller/service code.", "auth/"),
        _decision(ControllerAction.READ, "Angle 2/3: existing test coverage.", "tests/auth/test_session_refresh.py"),
        _decision(ControllerAction.ANALYZE, "Angle 3/3: recent config change.", "auth/oidc_config.py"),
        _decision(ControllerAction.EDIT, "Apply the fix.", "auth/session_manager.py"),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(ControllerAction.FINISH, "Fixed and verified."),
    ]
    actions = [
        # Angle 1 (SEARCH's tool subset excludes read_file -- use search_files,
        # a real call against the real repo).
        _tool_call_response("call-1a", "search_files", {"pattern": "SessionManager"}),
        _final_text("Found SessionManager wired from auth_controller.py into session_manager.py."),
        # Angle 2.
        _tool_call_response("call-2a", "read_file", {"path": "tests/auth/test_session_refresh.py"}),
        _final_text("The test expects refresh() to succeed for a session within TOKEN_EXPIRY_SECONDS."),
        # Angle 3.
        _tool_call_response("call-3a", "read_file", {"path": "auth/oidc_config.py"}),
        _final_text("TOKEN_EXPIRY_SECONDS was extended to 3600; session_manager.py wasn't updated."),
        # EDIT.
        _tool_call_response(
            "call-edit",
            "edit_file",
            {"path": "auth/session_manager.py", "old_text": _FIX_OLD_TEXT, "new_text": _FIX_NEW_TEXT},
        ),
        _final_text("Updated SessionManager to read TOKEN_EXPIRY_SECONDS from oidc_config."),
        # TEST.
        _final_text(_TEST_PASS_JSON),
    ]
    return decisions, actions


def _after_scenario():
    """1 DELEGATE action (3 concurrent workers covering the same 3 angles),
    then EDIT/TEST/FINISH identically."""
    decisions = [
        _decision(ControllerAction.DELEGATE, "3 independent angles -- investigate them at once.", "auth"),
        _decision(ControllerAction.EDIT, "Apply the fix.", "auth/session_manager.py"),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(ControllerAction.FINISH, "Fixed and verified."),
    ]
    delegate_planning = [
        _final_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "worker_type": "code_research",
                            "description": f"Investigate auth/auth_controller.py and auth/session_manager.py. {_ANGLE_MARKER['CODE']}",
                        },
                        {
                            "worker_type": "test",
                            "description": f"Read tests/auth/test_session_refresh.py and report what it asserts. {_ANGLE_MARKER['TEST']}",
                        },
                        {
                            "worker_type": "bug_analysis",
                            "description": f"Read auth/oidc_config.py for a recent config change. {_ANGLE_MARKER['CONFIG']}",
                        },
                    ]
                }
            )
        )
    ]
    worker_scripts = {
        _ANGLE_MARKER["CODE"]: [
            _tool_call_response("w-code-1", "read_file", {"path": "auth/auth_controller.py"}),
            _tool_call_response("w-code-2", "read_file", {"path": "auth/session_manager.py"}),
            _final_text(
                json.dumps(
                    {
                        "findings": "SessionManager.is_refreshable() uses a local stale constant.",
                        "confidence": 0.8,
                        "files": ["auth/auth_controller.py", "auth/session_manager.py"],
                        "recommendations": [],
                    }
                )
            ),
        ],
        _ANGLE_MARKER["TEST"]: [
            _tool_call_response("w-test-1", "read_file", {"path": "tests/auth/test_session_refresh.py"}),
            _final_text(
                json.dumps(
                    {
                        "findings": "The test expects refresh() to succeed within TOKEN_EXPIRY_SECONDS.",
                        "confidence": 0.9,
                        "files": ["tests/auth/test_session_refresh.py"],
                        "recommendations": [],
                    }
                )
            ),
        ],
        _ANGLE_MARKER["CONFIG"]: [
            _tool_call_response("w-config-1", "read_file", {"path": "auth/oidc_config.py"}),
            _final_text(
                json.dumps(
                    {
                        "findings": "TOKEN_EXPIRY_SECONDS was extended to 3600; session_manager.py wasn't updated.",
                        "confidence": 0.85,
                        "files": ["auth/oidc_config.py"],
                        "recommendations": [],
                    }
                )
            ),
        ],
    }
    actions = [
        _tool_call_response(
            "call-edit",
            "edit_file",
            {"path": "auth/session_manager.py", "old_text": _FIX_OLD_TEXT, "new_text": _FIX_NEW_TEXT},
        ),
        _final_text("Updated SessionManager to read TOKEN_EXPIRY_SECONDS from oidc_config."),
        _final_text(_TEST_PASS_JSON),
    ]
    return decisions, actions, delegate_planning, worker_scripts


def _iterations(result: dict) -> int:
    return result.get("iteration", len(result["task_state"]["decisions"]))


def test_delegate_needs_strictly_fewer_iterations_than_sequential_search(repo_root):
    """The core Phase 9 efficiency claim, made assertable: gathering the SAME
    information (same files discovered, same fix applied, same passing
    test) via DELEGATE takes strictly fewer outer-loop iterations than doing
    the equivalent investigation as 3 separate sequential actions."""
    _add_auth_bug_files(repo_root)

    before_decisions, before_actions = _before_scenario()
    before_router = _BenchRouter(before_decisions, before_actions)
    before_controller = _real_controller(repo_root, before_router)
    before_result = before_controller.run("Find why authentication sometimes fails")

    # A second, independent physical copy for the AFTER run: both scenarios
    # apply a real EDIT to auth/session_manager.py, so reusing the same
    # repo_root (already fixed by the BEFORE run above) would make the
    # AFTER run's EDIT target already-patched text and would not be a fair
    # "same ground truth" comparison. See scripts/benchmark_workers.py's
    # build_synthetic_repo docstring for the identical rationale.
    after_repo_root = repo_root.parent / "repo_after_delegate"
    after_repo_root.mkdir()
    for name in ("README.md", "main.py", ".gitignore"):
        (after_repo_root / name).write_text((repo_root / name).read_text(encoding="utf-8"), encoding="utf-8")
    _add_auth_bug_files(after_repo_root)

    after_decisions, after_actions, after_delegate_planning, after_worker_scripts = _after_scenario()
    after_router = _BenchRouter(after_decisions, after_actions, after_delegate_planning, after_worker_scripts)
    after_controller = _real_controller(after_repo_root, after_router)
    after_result = after_controller.run("Find why authentication sometimes fails")

    # Both scenarios actually finished, fixed the bug, and verified it.
    assert before_result["finished"] is True
    assert after_result["finished"] is True
    assert before_result["success"] is True
    assert after_result["success"] is True
    assert before_result["files_modified"] == ["auth/session_manager.py"]
    assert after_result["files_modified"] == ["auth/session_manager.py"]

    # Equivalent information was gathered by both -- same set of files.
    assert set(before_result["files_seen"]) == set(after_result["files_seen"])
    assert set(before_result["files_seen"]) == {
        "auth/auth_controller.py",
        "auth/session_manager.py",
        "tests/auth/test_session_refresh.py",
        "auth/oidc_config.py",
    }

    # The actual regression guard: DELEGATE must take STRICTLY FEWER
    # outer-loop iterations than the old serial-search style for the same
    # investigation. If this ever fails, either DELEGATE stopped collapsing
    # independent angles into one iteration, or the BEFORE scenario
    # (accidentally) got cheaper -- both are worth knowing about.
    before_iterations = _iterations(before_result)
    after_iterations = _iterations(after_result)
    assert after_iterations < before_iterations, (
        f"expected DELEGATE ({after_iterations} iteration(s)) to need strictly fewer outer-loop "
        f"iterations than sequential SEARCH/READ/ANALYZE ({before_iterations} iteration(s))"
    )

    # Consistency: task_state.decisions is logged once per executed
    # (non-FINISH) action, same count as `iteration`.
    assert len(before_result["task_state"]["decisions"]) == before_iterations
    assert len(after_result["task_state"]["decisions"]) == after_iterations
