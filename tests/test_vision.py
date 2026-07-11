"""Formal tests for the Phase 8 "Vision Debugging Engine"
(`kenzo/vision/{models,image_loader}.py`, `ModelRouter.analyze_image`, and the
`AgentController`/`AgentRuntime` vision-context wiring it feeds into):
screenshot -> structured `VisionAnalysis` -> priority context injected into
the controller's decision loop -> a real code fix.

Faking approach, matching `tests/test_controller_scenarios.py` and
`tests/test_phase7_server.py`: only `ModelRouter.complete` is ever
monkeypatched (class-level, restored automatically per-test by the
`monkeypatch` fixture). No real API keys, no real vision model, no network
call anywhere in this file. Tools (`build_default_registry`), memory
(`RepositoryMemory`), and `AgentController` are real, running against a real
tmp git repository (the shared `repo_root` fixture from `tests/conftest.py`,
unmodified).

Sections:
1. `VisionAnalysis.to_context_dict()` -- the `mode="json"` regression test.
2. `VisionAnalysis.derive_request()` -- one test per branch.
3. `image_loader.load_image` -- format detection, fallback, and every error
   path.
4. `image_loader.to_data_url` -- base64 data URL round-trip.
5. `ModelRouter.analyze_image` -- valid JSON, graceful non-JSON degradation,
   and the "bad image never reaches `complete`" ordering guarantee.
6. `_render_vision_context` -- empty/None and populated rendering.
7. End-to-end scenario A: an ERROR_SCREEN (NullPointerException,
   UserService.java:42) drives a real controller run that finds, edits, and
   verifies a fix.
8. End-to-end scenario B: a UI_SCREEN (broken Submit button) drives a real
   controller run via search -> read -> edit -> finish.
9. `set_vision_context(None)` clears the controller's vision context.
10. `kenzo.memory.importance.is_worth_remembering("vision_fix", ...)`.
"""

from __future__ import annotations

import base64
import json

import pytest

from kenzo.config.settings import KenzoConfig, PermissionsConfig
from kenzo.core.controller import AgentController, ControllerAction, _render_vision_context
from kenzo.memory.importance import IMPORTANCE_THRESHOLD, is_worth_remembering, score_importance
from kenzo.memory.repository import RepositoryMemory
from kenzo.models.router import ModelRouter
from kenzo.tools.registry import build_default_registry
from kenzo.vision.image_loader import (
    MAX_IMAGE_BYTES,
    SUPPORTED_FORMATS,
    UnsupportedImageError,
    load_image,
    to_data_url,
)
from kenzo.vision.models import ImageType, VisionAnalysis


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding + scripted router -- same pattern
# as tests/test_controller_scenarios.py / tests/test_phase7_server.py,
# duplicated locally per those files' own convention of keeping this local
# to each test module rather than growing conftest.py.
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


class _ControllerScriptedRouter:
    """Fake router for `AgentController`: a dedicated ordered queue for the
    forced `decide_next_action` decision calls, and a separate flat ordered
    queue for every action-execution call (see
    `test_controller_scenarios.py`'s module docstring for why this split is
    unambiguous)."""

    def __init__(self, decisions, actions):
        self._decisions = list(decisions)
        self._actions = list(actions)
        self.decision_calls: list[dict] = []
        self.action_calls: list[dict] = []

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        # Snapshot (copy) messages rather than keeping the live list
        # reference -- `run_tool_loop` mutates `messages` in place across
        # turns.
        snapshot = [dict(m) for m in messages]
        if _is_decide_call(tool_choice):
            self.decision_calls.append({"task_type": task_type, "messages": snapshot})
            if not self._decisions:
                raise AssertionError("ran out of scripted decisions")
            return self._decisions.pop(0)

        self.action_calls.append({"task_type": task_type, "messages": snapshot})
        if not self._actions:
            raise AssertionError("ran out of scripted action responses")
        return self._actions.pop(0)


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


def _real_tools(repo_root, permissions=None):
    return build_default_registry(str(repo_root), permissions or PermissionsConfig(allow_write=True))


def _real_memory(repo_root):
    return RepositoryMemory(str(repo_root))


def _real_config():
    return KenzoConfig()


# ---------------------------------------------------------------------------
# Raw magic-byte fixtures for the three real image formats plus assorted
# invalid inputs -- kept minimal (magic bytes + a little filler), since
# `_sniff_format` only ever inspects a fixed-size prefix.
# ---------------------------------------------------------------------------

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body-not-a-real-image-but-thats-fine"
_JPEG_BYTES = b"\xff\xd8\xff" + b"fake-jpeg-body-not-a-real-image-but-thats-fine"
_WEBP_BYTES = b"RIFF" + b"\x24\x00\x00\x00" + b"WEBP" + b"fake-webp-body-filler"


# ===========================================================================
# 1. VisionAnalysis.to_context_dict() -- mode="json" regression test
# ===========================================================================


def test_to_context_dict_renders_image_type_as_plain_string_not_enum_repr():
    """Regression test for the real bug documented in `VisionAnalysis
    .to_context_dict`'s docstring: a plain `self.model_dump()` (no
    `mode="json"`) leaves `image_type` as the raw `ImageType` enum member,
    which `dict.get("image_type")` then hands back as `ImageType.ERROR_SCREEN`
    rather than the clean string -- this test fails if that fix is reverted.
    """
    analysis = VisionAnalysis(image_type=ImageType.ERROR_SCREEN, primary_error="NullPointerException")
    context = analysis.to_context_dict()

    assert context["image_type"] == "ERROR_SCREEN"
    assert isinstance(context["image_type"], str)
    assert context["image_type"] != str(ImageType.ERROR_SCREEN)  # "ImageType.ERROR_SCREEN" -- must NOT match
    assert "ImageType" not in context["image_type"]


def test_to_context_dict_excludes_raw_model_output():
    analysis = VisionAnalysis(
        image_type=ImageType.LOG_SCREEN,
        visible_text="some log lines",
        raw_model_output='{"image_type": "LOG_SCREEN", "visible_text": "some log lines"}',
    )
    context = analysis.to_context_dict()

    assert "raw_model_output" not in context
    assert context["visible_text"] == "some log lines"


def test_to_context_dict_all_other_image_type_values_also_render_as_plain_strings():
    for image_type in ImageType:
        context = VisionAnalysis(image_type=image_type).to_context_dict()
        assert context["image_type"] == image_type.value
        assert "ImageType" not in context["image_type"]


# ===========================================================================
# 2. VisionAnalysis.derive_request() -- one test per branch
# ===========================================================================


def test_derive_request_user_note_takes_priority_over_everything():
    analysis = VisionAnalysis(
        image_type=ImageType.ERROR_SCREEN,
        primary_error="NullPointerException",
        primary_file="UserService.java",
        primary_line=42,
    )
    result = analysis.derive_request("   Please only fix the login page, nothing else.   ")
    assert result == "Please only fix the login page, nothing else."


def test_derive_request_error_screen_with_primary_error_mentions_error_and_location():
    analysis = VisionAnalysis(
        image_type=ImageType.ERROR_SCREEN,
        primary_error="NullPointerException",
        primary_file="UserService.java",
        primary_line=42,
    )
    result = analysis.derive_request()
    assert "NullPointerException" in result
    assert "UserService.java" in result
    assert "42" in result


def test_derive_request_error_screen_without_primary_file_omits_location_but_keeps_error():
    analysis = VisionAnalysis(image_type=ImageType.ERROR_SCREEN, primary_error="KeyError: 'user_id'")
    result = analysis.derive_request()
    assert "KeyError: 'user_id'" in result


def test_derive_request_ui_screen_uses_visible_text():
    analysis = VisionAnalysis(
        image_type=ImageType.UI_SCREEN,
        visible_text="The Submit button is greyed out and unclickable on the checkout page.",
    )
    result = analysis.derive_request()
    assert "UI" in result or "ui" in result.lower()
    assert "Submit button is greyed out" in result


def test_derive_request_log_screen_with_error_messages_uses_first_message():
    analysis = VisionAnalysis(
        image_type=ImageType.LOG_SCREEN,
        error_messages=["ECONNREFUSED 127.0.0.1:5432", "retrying in 5s"],
    )
    result = analysis.derive_request()
    assert "ECONNREFUSED 127.0.0.1:5432" in result
    assert "retrying in 5s" not in result  # only the FIRST error message is used


def test_derive_request_default_generic_fallback_uses_visible_text():
    analysis = VisionAnalysis(image_type=ImageType.CODE_SCREEN, visible_text="def broken_function():\n    pass")
    result = analysis.derive_request()
    assert "def broken_function()" in result


def test_derive_request_default_fallback_with_no_visible_text_still_returns_something_sane():
    analysis = VisionAnalysis(image_type=ImageType.CODE_SCREEN)
    result = analysis.derive_request()
    assert isinstance(result, str)
    assert result.strip()


def test_derive_request_log_screen_without_error_messages_falls_through_to_default():
    analysis = VisionAnalysis(image_type=ImageType.LOG_SCREEN, visible_text="INFO: server started on :8080")
    result = analysis.derive_request()
    assert "INFO: server started on :8080" in result


# ===========================================================================
# 3. image_loader.load_image
# ===========================================================================


def test_load_image_detects_valid_png_bytes():
    loaded = load_image(_PNG_BYTES)
    assert loaded.format == "png"
    assert loaded.mime_type == "image/png"
    assert loaded.data == _PNG_BYTES
    assert loaded.source == "<bytes>"


def test_load_image_detects_valid_jpeg_bytes():
    loaded = load_image(_JPEG_BYTES)
    assert loaded.format == "jpeg"
    assert loaded.mime_type == "image/jpeg"
    assert loaded.data == _JPEG_BYTES


def test_load_image_detects_valid_webp_bytes():
    loaded = load_image(_WEBP_BYTES)
    assert loaded.format == "webp"
    assert loaded.mime_type == "image/webp"
    assert loaded.data == _WEBP_BYTES


def test_load_image_from_real_path_with_matching_extension(tmp_path):
    path = tmp_path / "screenshot.png"
    path.write_bytes(_PNG_BYTES)
    loaded = load_image(path)
    assert loaded.format == "png"
    assert loaded.source == str(path)


def test_load_image_from_path_with_str_argument(tmp_path):
    path = tmp_path / "error.jpg"
    path.write_bytes(_JPEG_BYTES)
    loaded = load_image(str(path))
    assert loaded.format == "jpeg"


def test_load_image_path_with_unrecognized_magic_bytes_falls_back_to_extension(tmp_path):
    """A supported extension (`.png`) whose content doesn't sniff to any
    known magic-byte header still loads successfully, using the extension
    as a fallback -- per `load_image`'s own documented "a sniff failure
    still falls back to a valid extension" behavior."""
    path = tmp_path / "renamed.png"
    path.write_bytes(b"this is just plain text, not really a PNG at all, but long enough")
    loaded = load_image(path)
    assert loaded.format == "png"
    assert loaded.mime_type == "image/png"


def test_load_image_bytes_with_filename_hint_falls_back_to_hint_extension_when_sniff_inconclusive():
    unrecognizable = b"plain bytes with no recognizable image header whatsoever, but nonempty"
    loaded = load_image(unrecognizable, filename_hint="my-screenshot.webp")
    assert loaded.format == "webp"
    assert loaded.mime_type == "image/webp"
    assert loaded.source == "my-screenshot.webp"


def test_load_image_magic_bytes_win_over_a_disagreeing_extension(tmp_path):
    """A file named `.jpg` but whose actual bytes are a real PNG must be
    detected as png -- magic bytes win over a mislabeled extension."""
    path = tmp_path / "mislabeled.jpg"
    path.write_bytes(_PNG_BYTES)
    loaded = load_image(path)
    assert loaded.format == "png"
    assert loaded.mime_type == "image/png"


def test_load_image_empty_bytes_raises():
    with pytest.raises(UnsupportedImageError, match="empty"):
        load_image(b"")


def test_load_image_empty_file_on_disk_raises(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(UnsupportedImageError, match="empty"):
        load_image(path)


def test_load_image_oversized_bytes_raises():
    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_IMAGE_BYTES  # over the limit
    with pytest.raises(UnsupportedImageError, match="too large"):
        load_image(oversized)


def test_load_image_unrecognized_format_raises():
    junk = b"this is definitely not any kind of image, just some plain filler text data"
    with pytest.raises(UnsupportedImageError, match="unsupported|undetectable"):
        load_image(junk)


def test_load_image_nonexistent_path_raises():
    with pytest.raises(UnsupportedImageError, match="not found"):
        load_image("C:/definitely/does/not/exist/screenshot.png")


def test_load_image_directory_path_raises(tmp_path):
    directory = tmp_path / "a_directory.png"
    directory.mkdir()
    with pytest.raises(UnsupportedImageError, match="not a file"):
        load_image(directory)


def test_supported_formats_constant_matches_documented_set():
    assert SUPPORTED_FORMATS == frozenset({"png", "jpg", "jpeg", "webp"})


# ===========================================================================
# 4. image_loader.to_data_url
# ===========================================================================


def test_to_data_url_png_round_trips_base64_and_mime():
    loaded = load_image(_PNG_BYTES)
    url = to_data_url(loaded)
    assert url.startswith("data:image/png;base64,")
    encoded = url.split(",", 1)[1]
    assert base64.b64decode(encoded) == _PNG_BYTES


def test_to_data_url_jpeg_uses_image_jpeg_mime():
    loaded = load_image(_JPEG_BYTES)
    url = to_data_url(loaded)
    assert url.startswith("data:image/jpeg;base64,")
    encoded = url.split(",", 1)[1]
    assert base64.b64decode(encoded) == _JPEG_BYTES


def test_to_data_url_webp_uses_image_webp_mime():
    loaded = load_image(_WEBP_BYTES)
    url = to_data_url(loaded)
    assert url.startswith("data:image/webp;base64,")
    encoded = url.split(",", 1)[1]
    assert base64.b64decode(encoded) == _WEBP_BYTES


# ===========================================================================
# 5. ModelRouter.analyze_image
# ===========================================================================

# The spec's own worked example, reused verbatim per this task's instructions.
_NPE_VISION_JSON = {
    "image_type": "ERROR_SCREEN",
    "error_messages": ["NullPointerException: Cannot invoke \"User.getName()\" because \"user\" is null"],
    "visible_text": "Exception in thread \"main\" java.lang.NullPointerException",
    "stack_traces": ["at com.example.UserService.getUserName(UserService.java:240)"],
    "ui_elements": [],
    "file_names": ["UserService.java"],
    "line_numbers": [240],
    "possible_causes": ["user argument was not null-checked before calling getName()"],
    "primary_error": "NullPointerException",
    "primary_file": "UserService.java",
    "primary_line": 240,
}


def test_analyze_image_parses_valid_vision_json_into_structured_analysis(monkeypatch):
    def _fake_complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        assert task_type.value == "vision" if hasattr(task_type, "value") else task_type == "vision"
        # Confirm the multimodal message shape really was built.
        user_content = messages[1]["content"]
        assert isinstance(user_content, list)
        assert user_content[1]["type"] == "image_url"
        assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        return _FakeModelResponse(json.dumps(_NPE_VISION_JSON))

    monkeypatch.setattr(ModelRouter, "complete", _fake_complete)

    router = ModelRouter(KenzoConfig())
    result = router.analyze_image(_PNG_BYTES)

    assert isinstance(result, VisionAnalysis)
    assert result.image_type == ImageType.ERROR_SCREEN
    assert result.primary_error == "NullPointerException"
    assert result.primary_file == "UserService.java"
    assert result.primary_line == 240
    assert result.file_names == ["UserService.java"]
    assert result.line_numbers == [240]
    assert "NullPointerException" in result.error_messages[0]
    assert result.raw_model_output == json.dumps(_NPE_VISION_JSON)


def test_analyze_image_degrades_gracefully_on_non_json_model_response(monkeypatch):
    prose = "I can see a code editor with a red squiggly underline but no readable error text."

    def _fake_complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        return _FakeModelResponse(prose)

    monkeypatch.setattr(ModelRouter, "complete", _fake_complete)

    router = ModelRouter(KenzoConfig())
    result = router.analyze_image(_JPEG_BYTES)  # never raises

    assert isinstance(result, VisionAnalysis)
    assert result.visible_text == prose.strip()[:2000]
    assert result.raw_model_output == prose
    # Degraded default -- no structured fields could be extracted.
    assert result.primary_error is None
    assert result.error_messages == []


def test_analyze_image_degrades_gracefully_on_json_that_fails_validation(monkeypatch):
    """Parses as JSON but doesn't validate against `VisionAnalysis` (a
    non-integer inside `line_numbers`) -- must still degrade, not raise."""
    bad_shape = json.dumps({"image_type": "ERROR_SCREEN", "line_numbers": ["not-an-int"]})

    def _fake_complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        return _FakeModelResponse(bad_shape)

    monkeypatch.setattr(ModelRouter, "complete", _fake_complete)

    router = ModelRouter(KenzoConfig())
    result = router.analyze_image(_PNG_BYTES)  # never raises

    assert isinstance(result, VisionAnalysis)
    assert result.raw_model_output == bad_shape
    assert result.visible_text == bad_shape.strip()[:2000]


def test_analyze_image_bad_image_raises_without_ever_calling_complete(monkeypatch):
    def _fail_if_called(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        raise AssertionError("ModelRouter.complete must never be called for an invalid image")

    monkeypatch.setattr(ModelRouter, "complete", _fail_if_called)

    router = ModelRouter(KenzoConfig())
    with pytest.raises(UnsupportedImageError):
        router.analyze_image(b"")  # empty bytes -- fails validation before any model call


def test_analyze_image_oversized_image_raises_without_ever_calling_complete(monkeypatch):
    def _fail_if_called(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        raise AssertionError("ModelRouter.complete must never be called for an invalid image")

    monkeypatch.setattr(ModelRouter, "complete", _fail_if_called)

    router = ModelRouter(KenzoConfig())
    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_IMAGE_BYTES
    with pytest.raises(UnsupportedImageError):
        router.analyze_image(oversized)


# ===========================================================================
# 6. _render_vision_context
# ===========================================================================


def test_render_vision_context_none_returns_empty_string():
    assert _render_vision_context(None) == ""


def test_render_vision_context_empty_dict_returns_empty_string():
    assert _render_vision_context({}) == ""


def test_render_vision_context_populated_renders_expected_labeled_content():
    analysis = VisionAnalysis(
        image_type=ImageType.ERROR_SCREEN,
        error_messages=["NullPointerException"],
        stack_traces=["at UserService.getUserName(UserService.java:42)"],
        file_names=["UserService.java"],
        line_numbers=[42],
        ui_elements=["error dialog"],
        possible_causes=["user was never null-checked"],
        visible_text="a red error banner across the top of the screen",
        primary_error="NullPointerException",
        primary_file="UserService.java",
        primary_line=42,
    )
    rendered = _render_vision_context(analysis.to_context_dict())

    assert rendered.startswith("Screenshot analysis (priority")
    assert "ERROR_SCREEN" in rendered
    assert "NullPointerException" in rendered
    assert "UserService.java" in rendered
    assert "42" in rendered
    assert "error dialog" in rendered
    assert "user was never null-checked" in rendered
    assert "a red error banner across the top of the screen" in rendered


def test_render_vision_context_minimal_analysis_only_shows_image_type_line():
    analysis = VisionAnalysis(image_type=ImageType.CODE_SCREEN)
    rendered = _render_vision_context(analysis.to_context_dict())
    assert rendered == "Screenshot analysis (priority -- the user attached this image):\n- Screenshot type: CODE_SCREEN"


# ===========================================================================
# 7. End-to-end scenario A: ERROR_SCREEN (NullPointerException,
#    UserService.java:42) -- find, edit, verify.
# ===========================================================================

_USER_SERVICE_JAVA_BUGGY = """\
public class UserService {
    public String getUserName(User user) {
        return user.getName();
    }
}
"""

_USER_SERVICE_JAVA_FIXED = """\
public class UserService {
    public String getUserName(User user) {
        if (user == null) {
            return "unknown";
        }
        return user.getName();
    }
}
"""


def test_e2e_error_screen_null_pointer_exception_finds_edits_and_verifies_fix(repo_root):
    # newline="" avoids Windows' text-mode \n -> \r\n translation on write,
    # matching what kenzo.tools.filesystem.edit_file/write_file themselves do
    # (see their own "newline="" disables platform newline translation"
    # comments) -- so `old_text`'s embedded "\n" literals below match the
    # file's real on-disk bytes exactly, cross-platform.
    (repo_root / "UserService.java").write_text(_USER_SERVICE_JAVA_BUGGY, encoding="utf-8", newline="")

    vision = VisionAnalysis(
        image_type=ImageType.ERROR_SCREEN,
        error_messages=["NullPointerException: Cannot invoke \"User.getName()\" because \"user\" is null"],
        stack_traces=["at com.example.UserService.getUserName(UserService.java:42)"],
        file_names=["UserService.java"],
        line_numbers=[42],
        possible_causes=["getUserName() never null-checks its user argument"],
        primary_error="NullPointerException",
        primary_file="UserService.java",
        primary_line=42,
        raw_model_output="(scripted test fixture, never actually sent to a model)",
    )

    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    decisions = [
        _decision(ControllerAction.READ, "The screenshot points at UserService.java -- read it first.", "UserService.java"),
        _decision(ControllerAction.EDIT, "Add a null check to prevent the NullPointerException.", "UserService.java"),
        _decision(ControllerAction.TEST, "Verify the fix resolves the crash."),
        _decision(ControllerAction.FINISH, "Fixed and verified by a passing test."),
    ]
    actions = [
        # READ
        _tool_call_response("call-read", "read_file", {"path": "UserService.java"}),
        _final_text("UserService.java calls user.getName() without checking user for null -- matches the screenshot's NullPointerException."),
        # EDIT
        _tool_call_response(
            "call-edit",
            "edit_file",
            {
                "path": "UserService.java",
                "old_text": "        return user.getName();\n",
                "new_text": '        if (user == null) {\n            return "unknown";\n        }\n        return user.getName();\n',
            },
        ),
        _final_text("Added a null check before calling user.getName()."),
        # TEST
        _final_text(json.dumps({"passed": True, "summary": "no more NullPointerException", "command": "", "raw_output": ""})),
    ]

    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    # Vision context is set BEFORE run(), exactly as `kenzo image`/kenzo.server's
    # attach_image handling would do.
    controller.set_vision_context(vision.to_context_dict())

    result = controller.run("Fix the crash shown in this screenshot")

    # 1. The vision context's error text genuinely reached the model, via a
    #    recorded decide_next_action prompt (not just that the fake router
    #    was asked for SOMETHING) -- inspect the real, captured messages.
    decision_prompts = [m["content"] for call in router.decision_calls for m in call["messages"]]
    assert any("NullPointerException" in p for p in decision_prompts)
    assert any("UserService.java" in p for p in decision_prompts)
    assert any("Screenshot analysis" in p for p in decision_prompts)

    # 2. The file was genuinely edited on disk (not just claimed).
    final_content = (repo_root / "UserService.java").read_text(encoding="utf-8")
    assert final_content == _USER_SERVICE_JAVA_FIXED
    assert "if (user == null)" in final_content

    # 3. The run reports overall success, gated by the passing test result.
    assert result["success"] is True
    assert result["finished"] is True
    assert result["test_results"][-1]["passed"] is True
    assert any(c["file_path"] == "UserService.java" for c in result["changes"])


# ===========================================================================
# 8. End-to-end scenario B: UI_SCREEN (broken Submit button) -- search, read,
#    edit, finish (no test step, per this task's own scripted sequence).
# ===========================================================================

_SUBMIT_BUTTON_JSX_BROKEN = """\
export function SubmitButton() {
  return <button className="btn-disabled">Submit</button>;
}
"""

_SUBMIT_BUTTON_JSX_FIXED = """\
export function SubmitButton() {
  return <button className="btn-primary">Submit</button>;
}
"""


def test_e2e_ui_screen_broken_submit_button_searches_reads_edits(repo_root):
    (repo_root / "SubmitButton.jsx").write_text(_SUBMIT_BUTTON_JSX_BROKEN, encoding="utf-8", newline="")

    vision = VisionAnalysis(
        image_type=ImageType.UI_SCREEN,
        ui_elements=["Submit button", "checkout form"],
        visible_text="The Submit button on the checkout page is greyed out and cannot be clicked.",
        possible_causes=["the button is stuck using its disabled/greyed-out CSS class"],
        raw_model_output="(scripted test fixture, never actually sent to a model)",
    )

    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)

    decisions = [
        _decision(ControllerAction.SEARCH, "Find the component rendering the Submit button.", "btn-disabled"),
        _decision(ControllerAction.READ, "Read the file that defines the Submit button.", "SubmitButton.jsx"),
        _decision(ControllerAction.EDIT, "Remove the disabled styling class so the button is clickable.", "SubmitButton.jsx"),
        _decision(ControllerAction.FINISH, "The button now uses the primary (enabled) style."),
    ]
    actions = [
        # SEARCH
        _tool_call_response("call-search", "search_files", {"pattern": "btn-disabled"}),
        _final_text("Found 'btn-disabled' in SubmitButton.jsx."),
        # READ
        _tool_call_response("call-read", "read_file", {"path": "SubmitButton.jsx"}),
        _final_text("SubmitButton.jsx renders <button className=\"btn-disabled\">, matching the greyed-out button in the screenshot."),
        # EDIT
        _tool_call_response(
            "call-edit",
            "edit_file",
            {"path": "SubmitButton.jsx", "old_text": "btn-disabled", "new_text": "btn-primary"},
        ),
        _final_text("Changed the button's class from btn-disabled to btn-primary."),
    ]

    router = _ControllerScriptedRouter(decisions, actions)
    controller = AgentController(router, tools, memory, _real_config())

    controller.set_vision_context(vision.to_context_dict())

    result = controller.run("Fix the UI issue shown in this screenshot")

    # 1. The vision context's descriptive text genuinely reached a decision prompt.
    decision_prompts = [m["content"] for call in router.decision_calls for m in call["messages"]]
    assert any("Submit button" in p for p in decision_prompts)
    assert any("greyed out" in p for p in decision_prompts)
    assert any("Screenshot analysis" in p for p in decision_prompts)
    assert any("UI_SCREEN" in p for p in decision_prompts)

    # 2. The file was genuinely edited on disk.
    final_content = (repo_root / "SubmitButton.jsx").read_text(encoding="utf-8")
    assert final_content == _SUBMIT_BUTTON_JSX_FIXED
    assert "btn-disabled" not in final_content

    # 3. The run finished (no TEST step in this script, so `success` is not
    #    asserted True here -- see `_build_result`: `success` is gated on a
    #    passing test result, and this scenario intentionally has none).
    assert result["finished"] is True
    assert any(c["file_path"] == "SubmitButton.jsx" for c in result["changes"])


# ===========================================================================
# 9. set_vision_context(None) clears it
# ===========================================================================


def test_set_vision_context_none_clears_it(repo_root):
    tools = _real_tools(repo_root)
    memory = _real_memory(repo_root)
    router = _ControllerScriptedRouter(decisions=[], actions=[])
    controller = AgentController(router, tools, memory, _real_config())

    assert controller.vision_context is None  # never set yet

    analysis = VisionAnalysis(image_type=ImageType.ERROR_SCREEN, primary_error="Boom")
    controller.set_vision_context(analysis.to_context_dict())
    assert controller.vision_context is not None
    assert controller.vision_context["primary_error"] == "Boom"
    assert _render_vision_context(controller.vision_context) != ""

    controller.set_vision_context(None)
    assert controller.vision_context is None
    assert _render_vision_context(controller.vision_context) == ""


# ===========================================================================
# 10. importance.py -- "vision_fix" kind
# ===========================================================================


def test_vision_fix_succeeded_scores_high_and_clears_threshold():
    content = "Fixed the NullPointerException in UserService.getUserName by null-checking the argument"
    score = score_importance("vision_fix", content, succeeded=True)
    assert score >= IMPORTANCE_THRESHOLD
    assert score == 0.85  # same shape/value as "bug_fix" succeeded=True, per the spec
    assert is_worth_remembering("vision_fix", content, succeeded=True)


def test_vision_fix_failed_scores_low_and_misses_threshold():
    content = "Attempted a fix for the NullPointerException shown in the screenshot but it did not work"
    score = score_importance("vision_fix", content, succeeded=False)
    assert score < IMPORTANCE_THRESHOLD
    assert score == 0.3  # same shape/value as "bug_fix" succeeded=False, per the spec
    assert not is_worth_remembering("vision_fix", content, succeeded=False)


def test_vision_fix_matches_bug_fix_scoring_shape_exactly():
    for succeeded in (True, False):
        content = "identical content string used for both kinds so only `kind` differs between the two scores"
        assert score_importance("vision_fix", content, succeeded=succeeded) == score_importance(
            "bug_fix", content, succeeded=succeeded
        )


def test_vision_fix_very_short_content_still_capped_low():
    score = score_importance("vision_fix", "fixed it", succeeded=True)  # under _MIN_CONTENT_LENGTH
    assert score < IMPORTANCE_THRESHOLD
