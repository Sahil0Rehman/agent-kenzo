"""Formal regression tests for the Phase 7 server foundation
(`kenzo/server/{protocol,edit_approval,session,app}.py`) -- a FastAPI +
WebSocket server exposing `AgentController` to IDE clients.

This suite proves, against the REAL stack (a real `AgentRuntime`/
`AgentController`, real tools, a real tmp git repo, only the model/router
layer scripted -- exactly the `test_controller_scenarios.py` pattern), the
safety guarantees Phase 7 exists for:

- streaming events over a real WebSocket connection (task_started -> event*
  -> task_complete), including real tool calls;
- the propose-then-approve edit flow ("Do NOT directly overwrite") for
  accept, reject, AND the applied-by-client case, verified by re-reading the
  real fixture file from disk -- not just by trusting a returned dict;
- graceful mid-task cancellation;
- one-task-per-repo enforcement (`SessionBusyError`);
- clean handling of malformed/unknown client messages;
- `ide_context` actually reaching `AgentController.set_ide_context`.

Most importantly, test_edit_approval_accept_flow_real_write_confirmed_on_disk
and the cancellation test below are the ones that would have caught a
regression of the real deadlock bug found and fixed during Phase 7
development: `SessionManager` used a plain `threading.Lock()` where
`start_task()` held the lock and then called `get_runtime()`, which
acquired the SAME lock again -- self-deadlocking every single
`start_task()` call for a not-yet-running repo. The fix was switching to
`threading.RLock()` in `kenzo/server/session.py`. Every WebSocket
interaction in this file goes through `_recv`/`_drain_until`, a manual
thread-based watchdog (no `pytest-timeout` is installed in this project's
venv -- confirmed before writing this) that fails LOUDLY with a clear
`AssertionError` if the server ever hangs, rather than blocking the whole
test run forever the way that deadlock originally did.

Faking approach, matching `tests/test_controller_scenarios.py`: only
`ModelRouter.complete` is monkeypatched (class-level, restored automatically
per-test by the `monkeypatch` fixture). Tools, memory, `AgentRuntime`,
`AgentController`, `SessionManager`, and the FastAPI app/WebSocket transport
are all real. `repo_root` (from `tests/conftest.py`, unmodified) is a real
tmp git repository.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Callable, Optional

from fastapi.testclient import TestClient

from kenzo.config.settings import KenzoConfig, PermissionsConfig
from kenzo.core.controller import ControllerAction
from kenzo.models.router import ModelRouter
from kenzo.server.app import create_app
from kenzo.server.edit_approval import EditApprovalToolRegistry
from kenzo.tools.registry import build_default_registry

# ---------------------------------------------------------------------------
# WebSocket receive watchdog -- see module docstring. `pytest-timeout` is NOT
# installed in this project's venv (checked via `pip show pytest-timeout`
# before writing this), so a manual thread-based watchdog is the correct,
# available mechanism: run the (synchronous, potentially-hanging)
# `ws.receive_json()` call on a daemon thread and bound how long the TEST
# thread will wait for it, rather than bounding the call itself (which isn't
# possible to interrupt cleanly). A daemon thread that never returns (a
# genuine hang) does not block pytest from finishing and reporting the
# failure.
# ---------------------------------------------------------------------------

_DEFAULT_RECV_TIMEOUT = 25.0


def _recv(ws, timeout: float = _DEFAULT_RECV_TIMEOUT) -> dict:
    """Receive one decoded JSON message from `ws`, bounded by `timeout`
    real wall-clock seconds. Raises a clear `AssertionError` (not a bare
    hang) if the server never responds in time -- exactly the failure mode
    a regression of the `SessionManager` RLock deadlock would produce.
    """
    result: "queue.Queue[tuple[str, Any]]" = queue.Queue()

    def _target() -> None:
        try:
            result.put(("ok", ws.receive_json()))
        except Exception as exc:  # noqa: BLE001 - relay any exception to the waiting thread
            result.put(("error", exc))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise AssertionError(
            f"timed out after {timeout}s waiting for a websocket message -- the server "
            "appears to have hung (e.g. a SessionManager lock regression)."
        )
    status, payload = result.get_nowait()
    if status == "error":
        raise payload
    return payload


def _drain_until(
    ws, predicate: Callable[[dict], bool], timeout: float = _DEFAULT_RECV_TIMEOUT
) -> list[dict]:
    """Receive messages from `ws` until one satisfies `predicate`, returning
    every message collected along the way (predicate-satisfying one last).
    Bounded by `timeout` TOTAL wall-clock seconds across the whole drain,
    not per-message -- a hang partway through still fails loudly and
    promptly instead of accumulating unbounded waits.
    """
    deadline = time.monotonic() + timeout
    messages: list[dict] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"timed out after {timeout}s draining messages; collected so far: {messages}"
            )
        message = _recv(ws, timeout=remaining)
        messages.append(message)
        if predicate(message):
            return messages


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding + scripted router -- same pattern
# as tests/test_controller_scenarios.py, duplicated locally per this task's
# instructions (do not modify conftest.py; keep new fixtures/helpers local
# to this file).
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


class _ScriptedRouter:
    """Fake router: a dedicated ordered queue for `decide_next_action`
    decision calls, a separate flat ordered queue for action-execution
    calls (see test_controller_scenarios.py's module docstring for why this
    split is unambiguous). `delay`, if set, sleeps briefly on EVERY call --
    used only where a test genuinely needs a real wall-clock window to act
    in (sending a cancel/second start_task) before the scripted script would
    otherwise race ahead in a few milliseconds; always small enough that the
    whole test still completes in well under the required bound.
    """

    def __init__(self, decisions, actions, delay: float = 0.0):
        self._decisions = list(decisions)
        self._actions = list(actions)
        self.decision_calls: list[dict] = []
        self.action_calls: list[dict] = []
        self._delay = delay

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        if self._delay:
            time.sleep(self._delay)
        if _is_decide_call(tool_choice):
            self.decision_calls.append({"task_type": task_type})
            if not self._decisions:
                raise AssertionError("ran out of scripted decisions")
            return self._decisions.pop(0)

        self.action_calls.append({"task_type": task_type})
        if not self._actions:
            raise AssertionError("ran out of scripted action responses")
        return self._actions.pop(0)


def _patch_router(monkeypatch, router: _ScriptedRouter) -> None:
    def _scripted_complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        return router.complete(task_type, messages, tools=tools, tool_choice=tool_choice, **kwargs)

    monkeypatch.setattr(ModelRouter, "complete", _scripted_complete)


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


def _build_app():
    """A fresh app (fresh `SessionManager`, fresh `KenzoConfig()` --
    default `permissions.allow_write=True`) per test -- never `config=None`,
    which would fall through to `load_config()` and touch the real
    `~/.kenzo/config.yaml` on the machine running the suite.
    """
    return create_app(config=KenzoConfig())


# ---------------------------------------------------------------------------
# 1. WebSocket connection / health
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_ok():
    client = TestClient(_build_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_websocket_connects_and_disconnects_cleanly():
    client = TestClient(_build_app())
    with client.websocket_connect("/ws"):
        pass  # accepting and cleanly tearing down the connection IS the assertion


# ---------------------------------------------------------------------------
# 2. start_task -> streaming events -> task_complete (real AgentRuntime/
#    AgentController stack, scripted router, through the real WebSocket)
# ---------------------------------------------------------------------------


def test_start_task_finish_only_streams_task_started_then_task_complete(repo_root, monkeypatch):
    decisions = [_decision(ControllerAction.FINISH, "nothing to do, goal already satisfied")]
    router = _ScriptedRouter(decisions, actions=[])
    _patch_router(monkeypatch, router)

    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "do nothing"})

        started = _recv(ws)
        assert started["type"] == "task_started"
        assert started["task_id"]
        assert started["repo_path"] == str(repo_root)

        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    assert any(m["type"] == "event" for m in messages), f"expected at least one event message, got: {messages}"

    complete = messages[-1]
    assert complete["type"] == "task_complete"
    assert complete["task_id"] == started["task_id"]
    assert complete["result"]["finished"] is True


def test_start_task_with_real_tool_call_streams_tool_start_and_tool_result(repo_root, monkeypatch):
    decisions = [
        _decision(ControllerAction.READ, "read main.py first", "main.py"),
        _decision(ControllerAction.FINISH, "read it, done"),
    ]
    actions = [
        _tool_call_response("call-read", "read_file", {"path": "main.py"}),
        _final_text("main.py currently returns 42."),
    ]
    router = _ScriptedRouter(decisions, actions)
    _patch_router(monkeypatch, router)

    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "what does main.py return?"})
        ack = _recv(ws)
        assert ack["type"] == "task_started"

        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    tool_starts = [m for m in messages if m.get("type") == "event" and m.get("event_type") == "tool_start"]
    tool_results = [m for m in messages if m.get("type") == "event" and m.get("event_type") == "tool_result"]

    assert any(m["data"].get("tool") == "read_file" for m in tool_starts), f"no read_file tool_start: {messages}"
    assert any(m["data"].get("tool") == "read_file" for m in tool_results), f"no read_file tool_result: {messages}"
    assert messages[-1]["result"]["finished"] is True


# ---------------------------------------------------------------------------
# 3. Edit-approval flow: accept and reject, through the real WebSocket.
#    These are the tests that would have caught the RLock deadlock
#    regression -- see module docstring.
# ---------------------------------------------------------------------------


def test_edit_approval_accept_flow_real_write_confirmed_on_disk(repo_root, monkeypatch):
    """The single most important test in this file. Exercises the exact
    bridge (AgentSession's background thread -> asyncio.Queue via
    `loop.call_soon_threadsafe` -> WebSocket send, and SessionManager's
    RLock-protected active-session bookkeeping) that hid the RLock deadlock
    bug this server foundation was built around: if `SessionManager._lock`
    ever regresses from `threading.RLock()` to a plain `threading.Lock()`,
    `start_task()` self-deadlocks on its own call to `get_runtime()` and
    this test HANGS instead of completing -- our `_recv`/`_drain_until`
    watchdog turns that into a fast, loud failure instead of blocking the
    whole suite forever.
    """
    decisions = [
        _decision(ControllerAction.EDIT, "apply the real fix", "main.py"),
        _decision(ControllerAction.FINISH, "done, the user approved the edit"),
    ]
    actions = [
        _tool_call_response(
            "call-edit", "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 100"}
        ),
        _final_text("Changed main.py to return 100."),
    ]
    router = _ScriptedRouter(decisions, actions)
    _patch_router(monkeypatch, router)

    original_content = (repo_root / "main.py").read_text(encoding="utf-8")
    assert "return 42" in original_content

    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "fix main.py to return 100"})
        ack = _recv(ws)
        assert ack["type"] == "task_started" and ack["task_id"]

        proposal = _drain_until(ws, lambda m: m.get("type") == "edit_proposal")[-1]
        assert proposal["path"] == "main.py"
        assert proposal["action"] == "edit"
        assert "return 42" in proposal["diff"]
        assert "return 100" in proposal["diff"]
        assert "return 100" in proposal["new_content"]
        proposal_id = proposal["proposal_id"]
        assert proposal_id

        # The core safety guarantee, checked BEFORE any decision is sent:
        # a proposal must never write before it is accepted.
        assert (repo_root / "main.py").read_text(encoding="utf-8") == original_content

        ws.send_json({"type": "edit_decision", "proposal_id": proposal_id, "decision": "accept", "applied": False})

        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    assert messages[-1]["result"]["finished"] is True

    # THE real, load-bearing assertion: re-read the fixture file from disk
    # and confirm the edit was genuinely applied by Kenzo's own write path.
    final_content = (repo_root / "main.py").read_text(encoding="utf-8")
    assert final_content == "def main():\n    return 100\n"
    assert "return 42" not in final_content


def test_edit_approval_reject_flow_leaves_file_byte_for_byte_unchanged(repo_root, monkeypatch):
    """The other half of the core safety guarantee: a REJECTED proposal
    must leave the real file on disk completely untouched -- asserted
    precisely as byte-for-byte identity, not a loose "looks the same"
    check.
    """
    decisions = [
        _decision(ControllerAction.EDIT, "apply the fix", "main.py"),
        _decision(ControllerAction.FINISH, "stopping, the user rejected the edit"),
    ]
    actions = [
        _tool_call_response(
            "call-edit", "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 100"}
        ),
        _final_text("The user rejected my proposed edit; nothing further to do."),
    ]
    router = _ScriptedRouter(decisions, actions)
    _patch_router(monkeypatch, router)

    original_bytes = (repo_root / "main.py").read_bytes()

    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "fix main.py to return 100"})
        ack = _recv(ws)
        assert ack["type"] == "task_started"

        proposal = _drain_until(ws, lambda m: m.get("type") == "edit_proposal")[-1]
        proposal_id = proposal["proposal_id"]

        ws.send_json({"type": "edit_decision", "proposal_id": proposal_id, "decision": "reject", "applied": False})

        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    assert messages[-1]["result"]["finished"] is True

    # Precise, byte-for-byte proof of "Do NOT directly overwrite" on reject.
    final_bytes = (repo_root / "main.py").read_bytes()
    assert final_bytes == original_bytes

    # Second, independent signal: the tool_result event's own error text
    # proves the REJECT branch of EditApprovalToolRegistry.execute() fired.
    tool_results = [m for m in messages if m.get("type") == "event" and m.get("event_type") == "tool_result"]
    edit_results = [m for m in tool_results if m["data"].get("tool") == "edit_file"]
    assert edit_results, f"expected an edit_file tool_result event, got: {messages}"
    assert edit_results[0]["data"]["success"] is False
    assert "rejected" in (edit_results[0]["data"].get("error") or "").lower()
    assert "not modified" in (edit_results[0]["data"].get("error") or "").lower()


def test_edit_approval_applied_by_client_flow_does_not_double_write(repo_root, monkeypatch):
    """`edit_decision` with `applied=true` simulates a real IDE client that
    already applied the edit itself (e.g. VS Code's `WorkspaceEdit`) --
    Kenzo must trust that and NOT perform its own write a second time.

    Our TestClient is not a real VS Code client, so it never actually
    changes the fixture file itself -- the correct, precise assertion is
    therefore that the file stays BYTE-FOR-BYTE identical to before the
    proposal (proving Kenzo's own write path was never invoked a second
    time; if it had been, "return 42" would have become "return 100" on
    disk exactly like the accept/applied=false test above proves it does
    when the write path IS taken), plus the file's mtime is unchanged
    (nothing touched it), AND -- a second, independent, precise signal --
    the tool_result event's own `output` text is the client-trust branch's
    EXACT wording from kenzo/server/edit_approval.py
    ("... applied by the connected client."), never the real edit_file
    tool's own success wording ("applied 1 edit to ...", from
    kenzo/tools/filesystem.py) which would appear if
    `EditApprovalToolRegistry.execute()` had fallen through to
    `self._wrapped.execute(...)` instead of returning early on the
    `pending.applied_by_client` branch.
    """
    decisions = [
        _decision(ControllerAction.EDIT, "apply the fix", "main.py"),
        _decision(ControllerAction.FINISH, "done, the client already applied it"),
    ]
    actions = [
        _tool_call_response(
            "call-edit", "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 100"}
        ),
        _final_text("The client already applied this edit via WorkspaceEdit."),
    ]
    router = _ScriptedRouter(decisions, actions)
    _patch_router(monkeypatch, router)

    original_bytes = (repo_root / "main.py").read_bytes()
    original_mtime_ns = (repo_root / "main.py").stat().st_mtime_ns

    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "fix main.py to return 100"})
        ack = _recv(ws)
        assert ack["type"] == "task_started"

        proposal = _drain_until(ws, lambda m: m.get("type") == "edit_proposal")[-1]
        proposal_id = proposal["proposal_id"]

        ws.send_json({"type": "edit_decision", "proposal_id": proposal_id, "decision": "accept", "applied": True})

        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    assert messages[-1]["result"]["finished"] is True

    final_bytes = (repo_root / "main.py").read_bytes()
    assert final_bytes == original_bytes
    assert (repo_root / "main.py").stat().st_mtime_ns == original_mtime_ns

    tool_results = [m for m in messages if m.get("type") == "event" and m.get("event_type") == "tool_result"]
    edit_results = [m for m in tool_results if m["data"].get("tool") == "edit_file"]
    assert edit_results, f"expected an edit_file tool_result event, got: {messages}"
    assert edit_results[0]["data"]["success"] is True
    assert "applied by the connected client" in edit_results[0]["data"]["output"]
    # Contrast check: the real edit_file tool's own wording never appears --
    # proof the server-side write path was never invoked a second time.
    assert "applied 1 edit to" not in edit_results[0]["data"]["output"]


# ---------------------------------------------------------------------------
# 5. Cancellation
# ---------------------------------------------------------------------------


def test_cancel_stops_task_before_scripted_decisions_exhausted(repo_root, monkeypatch):
    """Cancels a task scripted to run many more iterations than it actually
    gets to -- proving cancellation genuinely cut it short, not that it
    coincidentally finished on its own. Small per-call delays give the test
    a real (but tiny -- well under a second per call) wall-clock window to
    send `cancel` before the background thread could race through several
    scripted decisions; the whole test still completes in well under the
    required bound.
    """
    decisions = [
        _decision(ControllerAction.SEARCH, f"search step {i}", f"pattern-{i}") for i in range(6)
    ] + [_decision(ControllerAction.FINISH, "would have finished naturally -- should never be reached")]
    actions = []
    for i in range(6):
        actions.append(_tool_call_response(f"call-search-{i}", "search_files", {"pattern": f"pattern-{i}"}))
        actions.append(_final_text(f"search {i} done"))

    router = _ScriptedRouter(decisions, actions, delay=0.2)
    _patch_router(monkeypatch, router)

    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "search around a lot"})
        ack = _recv(ws)
        assert ack["type"] == "task_started"

        ws.send_json({"type": "cancel", "repo_path": str(repo_root)})

        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    complete = messages[-1]
    result = complete["result"]
    assert result["finished"] is True
    # The REAL observed text from AgentController._stop/_build_stopped_summary
    # (kenzo/core/controller.py) -- not a guess.
    assert "cancelled by the user" in result["final_response"]

    # Genuinely cut short: nowhere near all 7 scripted decisions were
    # consumed (in practice 0 or 1 -- cancel is sent immediately after the
    # task_started ack, well before the 0.2s-delayed second decision call
    # could complete).
    assert len(router.decision_calls) < len(decisions)


# ---------------------------------------------------------------------------
# 6. SessionBusyError / one-task-per-repo enforcement
# ---------------------------------------------------------------------------


def test_second_start_task_for_busy_repo_gets_server_error(repo_root, monkeypatch):
    """`SessionManager.start_task` must refuse a second concurrent task for
    a repo that already has one running -- this is precisely the code path
    the RLock fix (see module docstring) made reachable at all: under the
    original plain `threading.Lock()` bug, `start_task()` never returned
    for the FIRST call, so a second call could never even be attempted.
    """
    decisions = [
        _decision(ControllerAction.SEARCH, "keep the first task alive for a moment", "x"),
        _decision(ControllerAction.FINISH, "done"),
    ]
    actions = [
        _tool_call_response("call-search", "search_files", {"pattern": "return"}),
        _final_text("found it"),
    ]
    router = _ScriptedRouter(decisions, actions, delay=0.15)
    _patch_router(monkeypatch, router)

    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "first task"})
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "second task, must be rejected"})

        # The server processes one client message at a time on a single
        # loop, so the first ack is guaranteed to arrive before the second
        # is even looked at.
        first_ack = _recv(ws)
        assert first_ack["type"] == "task_started"
        assert first_ack["task_id"]

        second_ack = _recv(ws)
        assert second_ack["type"] == "server_error"
        assert "already running" in second_ack["message"].lower()
        # Never a real second task_started/corrupted stream -- exactly one
        # task_id is ever handed out for this repo_path in this exchange.

        # Drain the first (real, still-running) task to a clean completion
        # so no background thread outlives the test.
        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    assert messages[-1]["result"]["finished"] is True


# ---------------------------------------------------------------------------
# 7. Malformed/unknown message handling
# ---------------------------------------------------------------------------


def test_unknown_message_type_gets_server_error_and_connection_survives(repo_root, monkeypatch):
    decisions = [_decision(ControllerAction.FINISH, "noop -- only used to prove the connection still works")]
    router = _ScriptedRouter(decisions, actions=[])
    _patch_router(monkeypatch, router)

    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "totally_unknown_message_type", "foo": "bar"})
        error = _recv(ws)
        assert error["type"] == "server_error"
        assert "unknown client message type" in error["message"].lower()
        assert error["task_id"] is None

        # Prove the connection is genuinely still alive -- not just assume
        # it -- by successfully starting a real task afterward and seeing
        # the normal task_started/task_complete exchange.
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "noop"})
        started = _recv(ws)
        assert started["type"] == "task_started"
        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    assert messages[-1]["result"]["finished"] is True


def test_start_task_missing_required_field_gets_server_error(repo_root, monkeypatch):
    decisions = [_decision(ControllerAction.FINISH, "noop -- only used to prove the connection still works")]
    router = _ScriptedRouter(decisions, actions=[])
    _patch_router(monkeypatch, router)

    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        # "start_task" with no "request" field -- StartTaskMessage.request
        # has no default, so this fails pydantic validation.
        ws.send_json({"type": "start_task", "repo_path": str(repo_root)})
        error = _recv(ws)
        assert error["type"] == "server_error"
        assert "invalid message" in error["message"].lower()

        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "noop"})
        started = _recv(ws)
        assert started["type"] == "task_started"
        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    assert messages[-1]["result"]["finished"] is True


# ---------------------------------------------------------------------------
# 8. ide_context
# ---------------------------------------------------------------------------


def test_ide_context_reaches_agent_controller(repo_root, monkeypatch):
    decisions = [_decision(ControllerAction.FINISH, "noop -- only used to synchronize on message ordering")]
    router = _ScriptedRouter(decisions, actions=[])
    _patch_router(monkeypatch, router)

    app = _build_app()
    client = TestClient(app)

    context_payload = {
        "current_file": "main.py",
        "selection": "def main():",
        "selection_start_line": 1,
        "selection_end_line": 1,
        "open_tabs": ["main.py", "README.md"],
        "diagnostics": [{"file": "main.py", "line": 2, "message": "unreachable code", "severity": "warning"}],
        "workspace_path": str(repo_root),
        "git_status": "clean",
    }

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ide_context", "repo_path": str(repo_root), "context": context_payload})

        # IdeContextMessage has NO ack by design (see protocol.py). Use a
        # subsequent start_task's task_started ack as the synchronization
        # point instead: the server processes one client message at a time
        # on a single loop, so receiving task_started here PROVES the
        # ide_context message was already fully processed (including
        # AgentController.set_ide_context) before this point.
        ws.send_json({"type": "start_task", "repo_path": str(repo_root), "request": "noop"})
        started = _recv(ws)
        assert started["type"] == "task_started"

        # Real, observable signal: inspect AgentController.ide_context
        # directly on the runtime SessionManager is holding for this repo --
        # proves the context actually reached
        # AgentController.set_ide_context and was not silently dropped.
        runtime = app.state.session_manager.get_runtime(str(repo_root))
        assert runtime.controller.ide_context is not None
        assert runtime.controller.ide_context["current_file"] == "main.py"
        assert runtime.controller.ide_context["selection"] == "def main():"
        assert runtime.controller.ide_context["open_tabs"] == ["main.py", "README.md"]
        assert runtime.controller.ide_context["diagnostics"][0]["message"] == "unreachable code"

        messages = _drain_until(ws, lambda m: m.get("type") == "task_complete")

    assert messages[-1]["result"]["finished"] is True


# ---------------------------------------------------------------------------
# Bonus: EditApprovalToolRegistry's approval-timeout path, isolated. NOT
# reachable through the full websocket stack above, since AgentController
# wires IDE mode up using the module's 300s DEFAULT_APPROVAL_TIMEOUT_SECONDS
# with no controller-level override (by design -- see edit_approval.py's and
# controller.py's docstrings, and this task's own instructions not to modify
# kenzo/core/controller.py to add one). The constructor DOES expose
# `approval_timeout` directly, so this constructs the registry standalone
# (real write_file/edit_file underneath, via a real ToolRegistry) and proves
# the timeout's safety behavior with a short, test-only timeout.
# ---------------------------------------------------------------------------


_APPROVAL_TIMEOUT_SECONDS = 0.5


def test_edit_approval_registry_timeout_fails_closed_without_writing(repo_root):
    tools = build_default_registry(str(repo_root), PermissionsConfig(allow_write=True))
    registry = EditApprovalToolRegistry(
        tools, str(repo_root), PermissionsConfig(allow_write=True), approval_timeout=_APPROVAL_TIMEOUT_SECONDS
    )
    received_proposals: list[dict] = []
    registry.on_proposal = received_proposals.append

    original_bytes = (repo_root / "main.py").read_bytes()

    start = time.monotonic()
    result = registry.execute(
        "edit_file", {"path": "main.py", "old_text": "return 42", "new_text": "return 100"}
    )
    elapsed = time.monotonic() - start

    assert result.success is False
    assert "timed out" in (result.error or "").lower()
    assert "not modified" in (result.error or "").lower()
    # Genuinely waited out (most of) the configured timeout rather than a
    # false-negative fast-fail -- a >=90% threshold (not the full exact
    # value) tolerates ordinary OS timer/scheduling imprecision around
    # threading.Event.wait()'s deadline without weakening the real intent
    # of this assertion.
    assert elapsed >= _APPROVAL_TIMEOUT_SECONDS * 0.9
    assert elapsed < 5.0  # ...but did not hang well past it either

    assert len(received_proposals) == 1  # the proposal WAS sent to "the client"...
    assert (repo_root / "main.py").read_bytes() == original_bytes  # ...but never resolved, so never written
