"""`AgentSession`/`SessionManager`: bridges `AgentController.run()` (fully
synchronous) into the async WebSocket world, and owns one persistent
`AgentRuntime` per repository across a `kenzo server` process's lifetime.

The bridge pattern (a background thread running the synchronous agent loop,
its `on_event` callback pushing into an `asyncio.Queue` via `loop
.call_soon_threadsafe`, cancellation via a `threading.Event` checked once per
`AgentController.run()` iteration) was verified end-to-end -- including
mid-stream cancellation -- against a real FastAPI `TestClient` WebSocket
connection before this file was written.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Optional

from kenzo.config.settings import KenzoConfig
from kenzo.core.events import AgentEvent
from kenzo.core.runtime import AgentRuntime

# Sentinel pushed onto an AgentSession's queue to signal "no more events are
# coming" -- distinct from any real AgentEvent, so the WebSocket send loop
# knows to stop awaiting the queue once it sees this.
STREAM_DONE = object()


class AgentSession:
    """One task's server-side execution state. Constructed fresh per task
    (unlike `AgentRuntime`, which `SessionManager` keeps alive across many
    tasks for the same repo)."""

    def __init__(self, runtime: AgentRuntime, loop: asyncio.AbstractEventLoop) -> None:
        self.runtime = runtime
        self.task_id: Optional[str] = None
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.queue: "asyncio.Queue[AgentEvent | object]" = asyncio.Queue()
        self.cancel_event = threading.Event()
        self.done = threading.Event()
        self._loop = loop
        self._thread: Optional[threading.Thread] = None

    def start(self, request: str, task_id: Optional[str] = None) -> None:
        # Resolve the task_id UP FRONT (same generator AgentController._load_
        # or_create_task would use internally) so it's known immediately for
        # the task_started ack and every event from the very first one --
        # without this, every event emitted before the background thread's
        # run() call returns would carry an empty task_id, since the real
        # controller.run() result (which reports it) isn't available yet.
        resolved_task_id = task_id or self.runtime.controller.task_store.generate_task_id(request)
        self.task_id = resolved_task_id

        def on_event(event: AgentEvent) -> None:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, event)

        # AgentController.on_event is a plain mutable attribute (not fixed at
        # AgentRuntime construction) -- see kenzo.core.controller's docstring
        # for why swapping it per-task is correct under this project's
        # one-task-at-a-time-per-repo model (SessionManager enforces that).
        self.runtime.controller.on_event = on_event
        self.runtime.controller.set_edit_proposal_handler(
            lambda proposal: self._loop.call_soon_threadsafe(
                self.queue.put_nowait, ("edit_proposal", proposal)
            )
        )

        def _run() -> None:
            try:
                self.result = self.runtime.controller.run(
                    request, task_id=resolved_task_id, cancel_event=self.cancel_event
                )
                self.task_id = self.result.get("task_id") or resolved_task_id
                # AgentController.run() itself does no memory bookkeeping --
                # that lives in AgentRuntime.run_autonomous(), which this
                # bridge bypasses (it calls controller.run() directly). Call
                # it explicitly so server-driven tasks get the same session/
                # architecture memory recording an in-process run_autonomous
                # caller would -- see AgentRuntime.record_task_outcome.
                self.runtime.record_task_outcome(request, self.result)
            except Exception as exc:  # noqa: BLE001 - session boundary: a crash must reach the client cleanly
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.runtime.controller.set_edit_proposal_handler(None)
                # Phase 8: vision_context is a one-time screenshot artifact,
                # not ongoing state like ide_context -- it must not leak into
                # a later, unrelated task on this same persistent
                # AgentRuntime (SessionManager reuses one per repo across
                # many tasks).
                self.runtime.controller.set_vision_context(None)
                self.done.set()
                self._loop.call_soon_threadsafe(self.queue.put_nowait, STREAM_DONE)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self.cancel_event.set()

    def resolve_edit_proposal(self, proposal_id: str, decision: str, applied: bool) -> bool:
        return self.runtime.controller.resolve_edit_proposal(proposal_id, decision, applied)

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)


class SessionManager:
    """Owns one `AgentRuntime` per repository (`ide_mode=True`) for the
    lifetime of the `kenzo server` process -- a "persistent session" per
    the Phase 7 spec, reused across reconnects/multiple tasks exactly like
    the CLI already reuses one `AgentRuntime` across multiple turns.

    Refuses to start a second concurrent task against a repo that already
    has one running, rather than silently cross-wiring their event streams
    (`AgentController.on_event`/edit-proposal handler are single mutable
    slots per controller, not per-call) -- `start_task` raises
    `SessionBusyError` in that case, which the WebSocket handler turns into
    a clear `ServerErrorEvent` rather than corrupting another task's stream.
    """

    def __init__(self, config: Optional[KenzoConfig] = None) -> None:
        self._config = config
        self._runtimes: dict[str, AgentRuntime] = {}
        self._active: dict[str, AgentSession] = {}
        # RLock, not Lock: start_task() holds this lock and calls
        # get_runtime(), which acquires the SAME lock again -- a plain,
        # non-reentrant Lock would self-deadlock the very first start_task()
        # call for a not-yet-running repo (confirmed empirically: it hung
        # every single time, indefinitely, before this fix).
        self._lock = threading.RLock()

    def get_runtime(self, repo_path: str) -> AgentRuntime:
        resolved = str(Path(repo_path).resolve())
        with self._lock:
            runtime = self._runtimes.get(resolved)
            if runtime is None:
                runtime = AgentRuntime(resolved, config=self._config, ide_mode=True)
                self._runtimes[resolved] = runtime
            return runtime

    def start_task(
        self, repo_path: str, request: str, loop: asyncio.AbstractEventLoop, task_id: Optional[str] = None
    ) -> AgentSession:
        resolved = str(Path(repo_path).resolve())
        with self._lock:
            existing = self._active.get(resolved)
            if existing is not None and not existing.done.is_set():
                raise SessionBusyError(
                    f"A task is already running for '{resolved}'. Cancel it, or wait for it to finish, first."
                )
            runtime = self.get_runtime(resolved)
            session = AgentSession(runtime, loop)
            self._active[resolved] = session
        session.start(request, task_id=task_id)
        return session

    def get_active_session(self, repo_path: str) -> Optional[AgentSession]:
        resolved = str(Path(repo_path).resolve())
        with self._lock:
            session = self._active.get(resolved)
        if session is not None and not session.done.is_set():
            return session
        return None

    def close_all(self) -> None:
        with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            runtime.close()


class SessionBusyError(RuntimeError):
    """Raised by `SessionManager.start_task` when a task is already running
    for the requested repo_path."""
