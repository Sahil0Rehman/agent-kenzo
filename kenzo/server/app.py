"""FastAPI application: one `/ws` WebSocket endpoint implementing the
protocol in `kenzo.server.protocol`. `create_app()` is the factory `kenzo
server` (see `kenzo.cli.commands.server`) and tests both use, so a test can
build an app against an isolated `SessionManager`/`KenzoConfig` without
touching global state.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from kenzo.config.settings import KenzoConfig
from kenzo.core.events import AgentEvent
from kenzo.server.protocol import (
    AgentEventMessage,
    CancelMessage,
    EditDecisionMessage,
    EditProposalEvent,
    IdeContext,
    IdeContextMessage,
    ServerErrorEvent,
    StartTaskMessage,
    TaskCompleteEvent,
    TaskStartedEvent,
    parse_client_message,
)
from kenzo.server.session import STREAM_DONE, SessionBusyError, SessionManager

logger = logging.getLogger("kenzo.server")


def create_app(config: Optional[KenzoConfig] = None) -> FastAPI:
    app = FastAPI(title="Kenzo Agent Server")
    app.state.session_manager = SessionManager(config)
    # repo_path -> most recently pushed IdeContext, consumed (and cleared) by
    # the next start_task for that repo_path that doesn't supply its own --
    # see IdeContextMessage's docstring in kenzo.server.protocol.
    app.state.pending_ide_context = {}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        session_manager: SessionManager = app.state.session_manager
        send_lock = asyncio.Lock()
        drain_tasks: dict[str, asyncio.Task] = {}

        async def send(message) -> None:
            async with send_lock:
                await websocket.send_json(message.model_dump(mode="json"))

        async def drain_session(repo_path: str, session) -> None:
            try:
                while True:
                    item = await session.queue.get()
                    if item is STREAM_DONE:
                        await send(
                            TaskCompleteEvent(
                                task_id=session.task_id or "",
                                result=session.result
                                or {"success": False, "finished": True, "final_response": session.error or ""},
                            )
                        )
                        break
                    if isinstance(item, tuple) and item[0] == "edit_proposal":
                        proposal = item[1]
                        await send(
                            EditProposalEvent(
                                task_id=session.task_id or "",
                                proposal_id=proposal["proposal_id"],
                                path=proposal["path"],
                                action=proposal["action"],
                                diff=proposal["diff"],
                                new_content=proposal["new_content"],
                            )
                        )
                        continue
                    event: AgentEvent = item
                    await send(
                        AgentEventMessage(
                            task_id=session.task_id or "",
                            event_type=event.event_type,
                            stage=event.stage,
                            message=event.message,
                            data=event.data,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - a drain-loop bug must not silently kill the connection
                logger.exception("error draining session events for %s", repo_path)
                try:
                    await send(ServerErrorEvent(task_id=session.task_id, message=f"internal error streaming events: {exc}"))
                except Exception:  # noqa: BLE001 - the websocket itself may already be gone
                    pass

        try:
            while True:
                raw = await websocket.receive_json()
                try:
                    message = parse_client_message(raw)
                except (ValueError, ValidationError) as exc:
                    await send(ServerErrorEvent(message=f"invalid message: {exc}"))
                    continue

                if isinstance(message, StartTaskMessage):
                    # `request` has no pydantic default of "" purely to support Phase
                    # 8's image-only attach flow (the request text is derived from the
                    # screenshot analysis instead) -- but a start_task with NEITHER a
                    # typed request NOR an attached image is still meaningless and must
                    # be rejected here, the same way a missing `request` field used to
                    # fail pydantic validation outright before `request` gained that
                    # default.
                    if not message.request.strip() and not message.image_data:
                        await send(
                            ServerErrorEvent(
                                message="invalid message: 'request' or 'image_data' must be provided"
                            )
                        )
                        continue

                    ide_context: Optional[IdeContext] = message.ide_context or app.state.pending_ide_context.pop(
                        message.repo_path, None
                    )
                    if ide_context is not None:
                        _stash_ide_context(session_manager, message.repo_path, ide_context)

                    request_text = message.request
                    if message.image_data:
                        await send(
                            AgentEventMessage(
                                task_id=message.task_id or "",
                                event_type="thinking",
                                stage="vision",
                                message="Analyzing attached screenshot...",
                            )
                        )
                        try:
                            image_bytes = base64.b64decode(message.image_data)
                            runtime = session_manager.get_runtime(message.repo_path)
                            analysis = await asyncio.to_thread(
                                runtime.router.analyze_image,
                                image_bytes,
                                message.request,
                                filename_hint=f"screenshot.{message.image_format or 'png'}",
                            )
                        except Exception as exc:  # noqa: BLE001 - covers UnsupportedImageError + any provider/network error; analysis failure must not crash the connection
                            await send(ServerErrorEvent(message=f"could not analyze attached image: {exc}"))
                            continue

                        runtime.controller.set_vision_context(analysis.to_context_dict())
                        request_text = analysis.derive_request(message.request)
                        location = (
                            f" ({analysis.primary_file}:{analysis.primary_line})"
                            if analysis.primary_file
                            else ""
                        )
                        await send(
                            AgentEventMessage(
                                task_id=message.task_id or "",
                                event_type="thinking",
                                stage="vision",
                                message=(
                                    f"Detected {analysis.image_type.value}"
                                    + (f": {analysis.primary_error}{location}" if analysis.primary_error else "")
                                ),
                                data=analysis.to_context_dict(),
                            )
                        )

                    loop = asyncio.get_event_loop()
                    try:
                        session = session_manager.start_task(
                            message.repo_path, request_text, loop, task_id=message.task_id
                        )
                    except SessionBusyError as exc:
                        await send(ServerErrorEvent(message=str(exc)))
                        continue
                    except Exception as exc:  # noqa: BLE001 - starting a task must never crash the connection
                        await send(ServerErrorEvent(message=f"could not start task: {exc}"))
                        continue

                    # task_id may not be resolved until the background thread
                    # actually creates/loads the TaskState -- send what we
                    # have; the client also learns the real one for certain
                    # from the first AgentEventMessage/TaskCompleteEvent.
                    await send(TaskStartedEvent(task_id=session.task_id or message.task_id or "", repo_path=message.repo_path))
                    drain_tasks[message.repo_path] = asyncio.create_task(drain_session(message.repo_path, session))

                elif isinstance(message, CancelMessage):
                    session = session_manager.get_active_session(message.repo_path)
                    if session is not None:
                        session.cancel()

                elif isinstance(message, EditDecisionMessage):
                    resolved = False
                    for session in _all_sessions_with_pending(session_manager, drain_tasks):
                        if session.resolve_edit_proposal(message.proposal_id, message.decision, message.applied):
                            resolved = True
                            break
                    if not resolved:
                        await send(
                            ServerErrorEvent(
                                message=f"no pending edit proposal '{message.proposal_id}' (already resolved or expired)"
                            )
                        )

                elif isinstance(message, IdeContextMessage):
                    _stash_ide_context(session_manager, message.repo_path, message.context)
                    app.state.pending_ide_context[message.repo_path] = message.context

        except WebSocketDisconnect:
            pass
        finally:
            for task in drain_tasks.values():
                task.cancel()

    return app


def _stash_ide_context(session_manager: SessionManager, repo_path: str, context: IdeContext) -> None:
    """Push `context` onto the repo's (persistent) `AgentController` via
    `set_ide_context` -- rendered as priority context in every subsequent
    decision (`AgentController._render_ide_context`) until replaced or
    cleared."""
    runtime = session_manager.get_runtime(repo_path)
    runtime.controller.set_ide_context(context.model_dump())


def _all_sessions_with_pending(session_manager: SessionManager, drain_tasks: dict) -> list:
    sessions = []
    for repo_path in list(drain_tasks.keys()):
        session = session_manager.get_active_session(repo_path)
        if session is not None:
            sessions.append(session)
    return sessions
