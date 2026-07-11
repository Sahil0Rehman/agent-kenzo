"""The WebSocket JSON message contract between `kenzo server` and any client
(the `vscode-extension/` package, or any other client speaking the same
protocol). This module is the SINGLE SOURCE OF TRUTH for the wire format --
`vscode-extension/src/protocol.ts` is a hand-written TypeScript mirror of the
exact same shapes and MUST be kept in sync with this file by hand (there is
no code generation step in this project).

Every message is a JSON object with a `type` field used to dispatch it.
Pydantic models here are used for SERVER-SIDE validation/serialization only
-- the wire format is plain JSON, not a Python-specific encoding.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Client -> Server
# ---------------------------------------------------------------------------


class StartTaskMessage(BaseModel):
    """Start a new task, or resume an existing one (`task_id` given, and
    not yet finished) -- see `kenzo.core.controller.AgentController
    ._load_or_create_task`, which this maps directly onto.

    `image_data`/`image_format` (Phase 8, both optional) carry an attached
    screenshot -- base64-encoded raw bytes and its format (one of
    `kenzo.vision.image_loader.SUPPORTED_FORMATS`: png/jpg/jpeg/webp). When
    present, the server analyzes it via `ModelRouter.analyze_image()` BEFORE
    starting the task, sets the result as the controller's vision context
    (priority context for every decision, see `AgentController.set_vision_
    context`), and -- if `request` is empty -- derives a reasonable default
    request from the analysis itself (`VisionAnalysis.derive_request`)
    rather than requiring the client to always type something.
    """

    type: Literal["start_task"] = "start_task"
    repo_path: str
    request: str = ""
    task_id: Optional[str] = None
    ide_context: Optional["IdeContext"] = None
    image_data: Optional[str] = None
    image_format: Optional[str] = None


class CancelMessage(BaseModel):
    """Cancel the currently-running task for `repo_path` (there can only
    ever be one running task per repo_path at a time -- see
    `kenzo.server.session.SessionManager`)."""

    type: Literal["cancel"] = "cancel"
    repo_path: str


class EditDecisionMessage(BaseModel):
    """Respond to a previously-sent `EditProposalEvent`. `applied` should be
    `true` when the client already applied the edit itself (e.g. via VS
    Code's `WorkspaceEdit`, the correct mechanism for a real IDE client) --
    Kenzo then trusts that as the actual on-disk result instead of writing
    again; `false` means "I accepted, please write it yourself" (a non-IDE
    client, or a client whose own apply step failed)."""

    type: Literal["edit_decision"] = "edit_decision"
    proposal_id: str
    decision: Literal["accept", "reject"]
    applied: bool = False


class IdeContext(BaseModel):
    """IDE state sent alongside `start_task` (and updatable independently
    via `IdeContextMessage`) -- Kenzo treats this as PRIORITY context,
    ahead of anything it would otherwise have to search for. Every field is
    optional: a non-VS-Code client, or one that simply has nothing open, can
    omit all of them."""

    current_file: Optional[str] = None
    selection: Optional[str] = None
    selection_start_line: Optional[int] = None
    selection_end_line: Optional[int] = None
    open_tabs: list[str] = Field(default_factory=list)
    diagnostics: list["DiagnosticInfo"] = Field(default_factory=list)
    workspace_path: Optional[str] = None
    git_status: Optional[str] = None


class DiagnosticInfo(BaseModel):
    """One compiler/linter diagnostic, as reported by VS Code's Problems
    tab (`vscode.languages.getDiagnostics`)."""

    file: str
    line: int
    message: str
    severity: Literal["error", "warning", "info", "hint"] = "error"


class IdeContextMessage(BaseModel):
    """Push updated IDE context independently of starting a task (e.g. the
    user switched files or the selection changed) -- stored server-side and
    merged into the NEXT `start_task` for the same `repo_path` if that one
    doesn't include its own `ide_context`."""

    type: Literal["ide_context"] = "ide_context"
    repo_path: str
    context: IdeContext


ClientMessage = StartTaskMessage | CancelMessage | EditDecisionMessage | IdeContextMessage


def parse_client_message(raw: dict[str, Any]) -> ClientMessage:
    """Dispatch a raw decoded JSON object to the right client-message model
    by its `type` field. Raises `ValueError` for an unknown/missing type or
    a payload that fails validation -- callers (the WebSocket receive loop)
    should catch this and send back a `ServerErrorEvent` rather than
    crashing the connection.
    """
    msg_type = raw.get("type")
    if msg_type == "start_task":
        return StartTaskMessage.model_validate(raw)
    if msg_type == "cancel":
        return CancelMessage.model_validate(raw)
    if msg_type == "edit_decision":
        return EditDecisionMessage.model_validate(raw)
    if msg_type == "ide_context":
        return IdeContextMessage.model_validate(raw)
    raise ValueError(f"unknown client message type: {msg_type!r}")


# ---------------------------------------------------------------------------
# Server -> Client
# ---------------------------------------------------------------------------


class TaskStartedEvent(BaseModel):
    """Acknowledges a `start_task` with the resolved `task_id` -- the client
    must remember this to cancel or reconnect-and-resume later."""

    type: Literal["task_started"] = "task_started"
    task_id: str
    repo_path: str


class AgentEventMessage(BaseModel):
    """Wire form of `kenzo.core.events.AgentEvent`, tagged with which task
    it belongs to (a client may have started tasks for more than one
    repo_path over the connection's lifetime, though only one at a time per
    repo_path -- see `SessionManager`)."""

    type: Literal["event"] = "event"
    task_id: str
    event_type: str  # kenzo.core.events.AgentEventType's value
    stage: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class EditProposalEvent(BaseModel):
    """A proposed `write_file`/`edit_file` call, awaiting an
    `EditDecisionMessage` with this `proposal_id` before Kenzo proceeds --
    see `kenzo.server.edit_approval.EditApprovalToolRegistry`. The agent's
    background thread is BLOCKED until the decision arrives or the server's
    configured approval timeout elapses."""

    type: Literal["edit_proposal"] = "edit_proposal"
    task_id: str
    proposal_id: str
    path: str
    action: Literal["create", "edit"]
    diff: str
    new_content: str


class TaskCompleteEvent(BaseModel):
    """The task finished (successfully, budget-exhausted, or cancelled) --
    `result` is `AgentController.run()`'s full result dict."""

    type: Literal["task_complete"] = "task_complete"
    task_id: str
    result: dict[str, Any]


class ServerErrorEvent(BaseModel):
    """A server-side problem unrelated to any specific task (a malformed
    client message, an unhandled exception starting a task, etc.) --
    distinct from `AgentEventMessage`'s `event_type="error"`, which is a
    normal in-task error the agent itself recovered from."""

    type: Literal["server_error"] = "server_error"
    task_id: Optional[str] = None
    message: str


ServerMessage = TaskStartedEvent | AgentEventMessage | EditProposalEvent | TaskCompleteEvent | ServerErrorEvent
