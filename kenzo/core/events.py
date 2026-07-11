"""Event contract for streaming progress out of the agent pipeline.

`AgentEvent` is the single shape every node emits through when it starts and
finishes; `EventSink` is the callback type UIs (CLI, web, etc.) implement to
receive them. Kept dependency-free of the rest of `kenzo.core` so it can be
imported from anywhere without cycles.

Phase 7 adds `event_type` -- a closed taxonomy (`AGENT_EVENT_TYPES`) letting a
UI (VS Code extension, a future web client) render each event distinctly
("Searching authentication flow..." vs. "Editing LogoutService.java...")
instead of a flat progress log. Purely additive: `event_type` defaults to
"thinking" (the old pipeline's coarse per-node start/finish events in
`kenzo.core.graph` still construct `AgentEvent(stage=..., message=...)` with
no changes needed, and every existing test/caller keeps working unmodified).
"""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, Field

# The Phase 7 event taxonomy every richer emission point (see
# kenzo.core.tool_loop.run_tool_loop's on_tool_start/on_tool_result,
# kenzo.core.controller.AgentController._emit) picks from. "thinking" is the
# generic default for narration that isn't one of the more specific kinds.
AgentEventType = Literal[
    "thinking",
    "tool_start",
    "tool_result",
    "file_changed",
    "test_running",
    "error",
    "complete",
]

AGENT_EVENT_TYPES: tuple[str, ...] = (
    "thinking",
    "tool_start",
    "tool_result",
    "file_changed",
    "test_running",
    "error",
    "complete",
)


class AgentEvent(BaseModel):
    """A single progress event emitted by a node in the agent pipeline."""

    stage: str
    message: str
    data: dict = Field(default_factory=dict)
    event_type: AgentEventType = "thinking"


EventSink = Callable[[AgentEvent], None]
