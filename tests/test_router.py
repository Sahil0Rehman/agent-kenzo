"""Tests for kenzo.models.router.ModelRouter: task-type -> provider routing,
text completion against a mocked litellm.completion, and the missing-api-key
error path. litellm is monkeypatched everywhere so no network call is ever
made.
"""

from __future__ import annotations

import litellm
import pytest

from kenzo.config.settings import KenzoConfig, PermissionsConfig, ProviderConfig, RoutingConfig
from kenzo.core.schemas import TaskType
from kenzo.models.router import ModelRouter


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


def _build_config(**provider_overrides: ProviderConfig) -> KenzoConfig:
    providers = {
        "nemotron": ProviderConfig(
            api_key="nem-key", base_url="https://nemotron.example/v1", default_model="nemotron-model"
        ),
        "gemini": ProviderConfig(api_key="gem-key", base_url=None, default_model="gemini-1.5-pro"),
        "openai_compatible": ProviderConfig(api_key="oai-key", base_url=None, default_model="gpt-4o-mini"),
    }
    providers.update(provider_overrides)
    return KenzoConfig(providers=providers, routing=RoutingConfig(), permissions=PermissionsConfig())


# ---------------------------------------------------------------------------
# route()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_type,expected_provider",
    [
        (TaskType.CODING, "nemotron"),
        (TaskType.DEBUGGING, "nemotron"),
        (TaskType.VISION, "gemini"),
        (TaskType.REVIEW, "nemotron"),
        (TaskType.PLANNING, "nemotron"),
        (TaskType.RESEARCH, "nemotron"),
        (TaskType.GENERAL, "nemotron"),
    ],
)
def test_route_selects_provider_per_default_routing(task_type, expected_provider):
    config = _build_config()
    router = ModelRouter(config)

    provider_name, provider_config = router.route(task_type)

    assert provider_name == expected_provider
    assert provider_config is config.providers[expected_provider]


def test_route_accepts_plain_string_task_type():
    config = _build_config()
    router = ModelRouter(config)

    provider_name, _ = router.route("vision")

    assert provider_name == "gemini"


def test_route_raises_runtime_error_when_api_key_missing():
    config = _build_config(nemotron=ProviderConfig(api_key="", default_model="nemotron-model"))
    router = ModelRouter(config)

    with pytest.raises(RuntimeError, match="No API key configured for provider 'nemotron'"):
        router.route(TaskType.CODING)


def test_route_error_message_mentions_config_set_hint():
    config = _build_config(gemini=ProviderConfig(api_key="", default_model="gemini-1.5-pro"))
    router = ModelRouter(config)

    with pytest.raises(RuntimeError, match=r"kenzo config set providers\.gemini\.api_key"):
        router.route(TaskType.VISION)


# ---------------------------------------------------------------------------
# complete_text() / complete()
# ---------------------------------------------------------------------------


def test_complete_text_returns_mocked_content(monkeypatch):
    config = _build_config()
    router = ModelRouter(config)
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeModelResponse("mocked reply")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    result = router.complete_text(TaskType.CODING, "system prompt", "user prompt")

    assert result == "mocked reply"
    assert captured["model"] == "openai/nemotron-model"
    assert captured["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert captured["api_key"] == "nem-key"
    assert captured["api_base"] == "https://nemotron.example/v1"


def test_complete_text_returns_empty_string_for_none_content(monkeypatch):
    config = _build_config()
    router = ModelRouter(config)

    monkeypatch.setattr(litellm, "completion", lambda **kwargs: _FakeModelResponse(None))

    result = router.complete_text(TaskType.GENERAL, "sys", "user")

    assert result == ""


def test_complete_uses_gemini_prefix_for_gemini_provider(monkeypatch):
    config = _build_config()
    router = ModelRouter(config)
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeModelResponse("ok")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    router.complete_text(TaskType.VISION, "sys", "user")

    assert captured["model"] == "gemini/gemini-1.5-pro"
    assert captured["api_key"] == "gem-key"
    assert "api_base" not in captured  # gemini provider has no base_url set


def test_complete_passes_tools_and_tool_choice_when_given(monkeypatch):
    config = _build_config()
    router = ModelRouter(config)
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeModelResponse("ok", tool_calls=None)

    monkeypatch.setattr(litellm, "completion", fake_completion)

    tools = [{"type": "function", "function": {"name": "read_file"}}]
    router.complete(TaskType.CODING, [{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto")

    assert captured["tools"] == tools
    assert captured["tool_choice"] == "auto"


def test_complete_without_tools_omits_tool_kwargs(monkeypatch):
    config = _build_config()
    router = ModelRouter(config)
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeModelResponse("ok")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    router.complete(TaskType.CODING, [{"role": "user", "content": "hi"}])

    assert "tools" not in captured
    assert "tool_choice" not in captured
