"""`WorkerManager`: owns parallel execution of `WorkerTask`s.

Architecture note on "async": `AgentController`'s entire action loop is
synchronous by design (`ModelRouter.complete` is a blocking `litellm
.completion()` call; only `kenzo.server`'s WebSocket layer is async, and even
that bridges into the sync world via `asyncio.to_thread`/a background thread
-- see `kenzo.server.session`). Introducing real `asyncio` into the
controller loop itself would be an invasive, viral rewrite for no benefit
`run_parallel` doesn't already get from real OS threads: the work being
parallelized (LLM API calls, file reads, git commands) is I/O-bound, so a
`concurrent.futures.ThreadPoolExecutor` achieves genuine wall-clock
concurrency (the actual goal of "parallel execution") without touching the
synchronous design everywhere else in this codebase. `run_parallel` is a
plain, synchronous, blocking-until-done call from the caller's point of view
-- it just does its work concurrently underneath, exactly like `AgentRuntime
.__init__` already does for `IndexingService`'s background thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Optional

from kenzo.workers.models import WorkerResult, WorkerTask, WorkerType
from kenzo.workers.worker_agent import WorkerAgent


# Deliberately plain ASCII, not emoji: this label is embedded in AgentEvent
# .message, which flows through kenzo.cli.ui.components.render_event's
# console.print() -- Rich's legacy-Windows-console text writer encodes via
# the process's ANSI codepage (cp1252 on a stock Windows terminal, not
# UTF-8), which raises UnicodeEncodeError on most emoji. Confirmed by an
# actual crash during Phase 10 development: `kenzo improve` reached
# run_parallel's "Starting N worker(s)" emission and crashed with exactly
# this error before ever reaching an edit. VS Code's OWN webview UI (an
# always-UTF-8-safe browser context) already renders delegate-stage events
# with distinct styling independently (see chatViewProvider.ts's
# `.msg.delegate` class) and never relied on these labels containing emoji.
_WORKER_LABEL: dict[WorkerType, str] = {
    WorkerType.CODE_RESEARCH: "[research]",
    WorkerType.BUG_ANALYSIS: "[bug]",
    WorkerType.TEST: "[test]",
    WorkerType.DOCUMENTATION: "[docs]",
    WorkerType.GIT: "[git]",
    WorkerType.ARCHITECTURE_CRITIC: "[architecture]",
    WorkerType.TEST_GAP: "[test-gap]",
    WorkerType.PERFORMANCE: "[perf]",
}
_WORKER_LABEL_DEFAULT = "[worker]"

EmitFn = Callable[..., None]  # matches AgentController._emit's (stage, message, event_type=..., data=...) shape


class WorkerManager:
    """Constructed once per `AgentController` (mirrors `self.checkpoint`/
    `self.context`'s lifetime), reused across every `DELEGATE` action in a
    task's lifetime."""

    def __init__(
        self,
        router: Any,
        tools: Any,
        workers_config: Any,
        mcp_client: Optional[Any] = None,
        emit: Optional[EmitFn] = None,
        workspace: Optional[Any] = None,
    ) -> None:
        self.router = router
        self.tools = tools
        self.config = workers_config
        self.mcp_client = mcp_client
        self.emit = emit
        self.workspace = workspace
        self._counter = 0

    def _next_worker_id(self) -> str:
        self._counter += 1
        return f"worker-{self._counter}"

    def _emit(self, message: str, event_type: str = "thinking", data: Optional[dict] = None) -> None:
        if self.emit is not None:
            self.emit("delegate", message, event_type=event_type, data=data or {})

    def run_parallel(self, tasks: list[WorkerTask]) -> list[WorkerResult]:
        """Run every task in `tasks` concurrently and return their results in
        the SAME order `tasks` was given (not completion order), so a caller
        can zip `tasks`/results back together if it needs to.

        Item 8 ("avoid infinite worker spawning"): `tasks` is hard-capped to
        `workers_config.max_parallel` -- any excess tasks are silently
        dropped (not queued for a later batch), since the outer
        `AgentController` iteration/tool-call/runtime budgets are the real
        backstop against a runaway loop of repeated DELEGATE calls, not this
        one call's task count. If `workers_config.enabled` is False, every
        task fails closed with a clear error rather than silently running
        anyway.
        """
        if not tasks:
            return []

        if not getattr(self.config, "enabled", True):
            return [
                WorkerResult(
                    worker_id=task.worker_id or self._next_worker_id(),
                    worker_type=task.worker_type,
                    task_description=task.description,
                    success=False,
                    error="workers are disabled (workers.enabled=false)",
                )
                for task in tasks
            ]

        max_parallel = max(1, int(getattr(self.config, "max_parallel", 5)))
        timeout_seconds = max(1, int(getattr(self.config, "timeout_seconds", 120)))
        accepted = tasks[:max_parallel]

        for task in accepted:
            if not task.worker_id:
                task.worker_id = self._next_worker_id()

        started_lines = [
            f"{_WORKER_LABEL.get(t.worker_type, _WORKER_LABEL_DEFAULT)} {t.description}" for t in accepted
        ]
        self._emit(
            f"Starting {len(accepted)} worker(s):\n" + "\n".join(started_lines),
            event_type="tool_start",
            data={"workers": [{"worker_id": t.worker_id, "worker_type": t.worker_type.value, "task": t.description} for t in accepted]},
        )

        # `wait=False` below is deliberate, not an oversight: a plain `with
        # ThreadPoolExecutor(...) as pool:` block's __exit__ calls
        # shutdown(wait=True), which BLOCKS until every submitted thread
        # finishes -- including one we already gave up waiting on via
        # future.result(timeout=...). That would silently defeat the whole
        # point of a per-worker timeout. A worker whose future timed out
        # keeps running in its thread in the background (Python's threading
        # model has no way to hard-kill a thread); harmless since workers
        # are read-only, just a wasted API call, and the thread exits
        # naturally once its own call returns.
        pool = ThreadPoolExecutor(max_workers=min(len(accepted), max_parallel))
        futures = [
            (task, pool.submit(WorkerAgent(self.router, self.tools, self.mcp_client, workspace=self.workspace).run, task))
            for task in accepted
        ]

        results: list[WorkerResult] = []
        for task, future in futures:
            try:
                result = future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                result = WorkerResult(
                    worker_id=task.worker_id,
                    worker_type=task.worker_type,
                    task_description=task.description,
                    success=False,
                    error=f"timed out after {timeout_seconds}s",
                )
            except Exception as exc:  # noqa: BLE001 - one worker's crash must not lose the others' results
                result = WorkerResult(
                    worker_id=task.worker_id,
                    worker_type=task.worker_type,
                    task_description=task.description,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._emit(
                f"{_WORKER_LABEL.get(task.worker_type, _WORKER_LABEL_DEFAULT)} {task.description} -- "
                + ("done" if result.success else f"failed ({result.error})"),
                event_type="tool_result",
                data={"worker_id": task.worker_id, "worker_type": task.worker_type.value, "success": result.success},
            )
            results.append(result)

        pool.shutdown(wait=False)
        self._emit("Completed.", event_type="tool_result", data={"count": len(results)})
        return results
