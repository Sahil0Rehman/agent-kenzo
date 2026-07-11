"""Shared data contract for the Phase 8 vision pipeline: `VisionAnalysis` is
what `kenzo.models.router.ModelRouter.analyze_image()` produces from a
screenshot, and what `kenzo.core.controller.AgentController.set_vision_
context()` consumes as priority context for the next decision.

Kept dependency-free of the rest of `kenzo` (mirrors `kenzo.core.schemas`'s
convention) so it can be imported from anywhere without cycles.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ImageType(str, Enum):
    """What kind of screenshot this is -- shapes which fields are likely
    populated and how `AgentController` should frame the resulting task."""

    ERROR_SCREEN = "ERROR_SCREEN"
    UI_SCREEN = "UI_SCREEN"
    CODE_SCREEN = "CODE_SCREEN"
    LOG_SCREEN = "LOG_SCREEN"


class VisionAnalysis(BaseModel):
    """Structured understanding extracted from one screenshot.

    `primary_error`/`primary_file`/`primary_line` are the model's own best
    single guess at "the one thing that most likely needs fixing" (e.g. the
    example from the spec: a `NullPointerException` screenshot ->
    `{error: "NullPointerException", file: "UserService.java", line: 240}`)
    -- deliberately asked of the model directly (see `_VISION_SYSTEM_PROMPT`
    in `kenzo.models.router`) rather than mechanically derived as "the first
    entry of each list," since picking the ROOT-CAUSE line out of a stack
    trace with several frames is a judgment call the model is better
    positioned to make than a naive positional pick.
    """

    image_type: ImageType = ImageType.CODE_SCREEN
    error_messages: list[str] = Field(default_factory=list)
    visible_text: str = ""
    stack_traces: list[str] = Field(default_factory=list)
    ui_elements: list[str] = Field(default_factory=list)
    file_names: list[str] = Field(default_factory=list)
    line_numbers: list[int] = Field(default_factory=list)
    possible_causes: list[str] = Field(default_factory=list)

    primary_error: Optional[str] = None
    primary_file: Optional[str] = None
    primary_line: Optional[int] = None

    raw_model_output: str = ""  # best-effort audit trail; never re-parsed

    def to_context_dict(self) -> dict:
        """Render into the same plain-dict shape `AgentController.set_vision_
        context`/`_render_vision_context` expect (mirrors `IdeContext.model_
        dump()`'s role in the Phase 7 IDE-context pattern).

        Uses `mode="json"` specifically so `image_type` comes out as its
        plain string value ("ERROR_SCREEN") rather than the raw `ImageType`
        enum member -- Python's default `str(ImageType.ERROR_SCREEN)` (and
        therefore f-string interpolation, which `_render_vision_context`
        relies on) renders "ImageType.ERROR_SCREEN", not the clean value, so
        leaving this as the enum instance would have shown that ugly form
        directly in the agent's own prompt text. Confirmed by direct check
        before fixing, not assumed.
        """
        return self.model_dump(mode="json", exclude={"raw_model_output"})

    def derive_request(self, user_note: str = "") -> str:
        """Build a reasonable default agent request string from this
        analysis, for callers (the `kenzo image` CLI command, the server's
        `attach_image` handling) that don't have -- or want to override --
        their own request text. `user_note`, if given, is prepended and
        takes priority framing."""
        if user_note.strip():
            return user_note.strip()

        if self.image_type == ImageType.ERROR_SCREEN and self.primary_error:
            location = f" in {self.primary_file}" + (f":{self.primary_line}" if self.primary_line else "") if self.primary_file else ""
            return f"Fix this error: {self.primary_error}{location}"
        if self.image_type == ImageType.UI_SCREEN:
            return "Fix the UI issue shown in this screenshot: " + (self.visible_text[:300] or "see attached screenshot")
        if self.image_type == ImageType.LOG_SCREEN and self.error_messages:
            return f"Investigate and fix: {self.error_messages[0]}"
        return "Investigate and fix the issue shown in this screenshot: " + (self.visible_text[:300] or "(no readable text extracted)")
