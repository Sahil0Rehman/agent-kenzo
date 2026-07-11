"""Task-based model routing on top of litellm.

`ModelRouter` is the single place that decides *which* provider/model handles
a given piece of work (coding, debugging, review, ...) and shapes the actual
`litellm.completion` call. Agents should depend on this class rather than
calling litellm directly, so routing policy stays centralized and testable.

Phase 8 adds `analyze_image()` -- routed through the SAME `TaskType.VISION` /
`RoutingConfig.vision` machinery that has existed since Phase 1 (the default
config already points `vision` at the "gemini" provider), just never had a
caller until now. Vision-capable OpenAI-compatible models work identically,
since litellm normalizes the same multimodal message shape
(`content: [{"type": "text", ...}, {"type": "image_url", ...}]`) across
providers -- confirmed empirically before writing this (both `gpt-4o-mini`
and `gemini/gemini-2.0-flash` report `litellm.supports_vision(...) is True`
and accept the identical message shape).
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any, Union

import litellm

from kenzo.config.settings import KenzoConfig, ProviderConfig
from kenzo.core.schemas import TaskType
from kenzo.core.tool_loop import extract_json
from kenzo.models.providers import build_completion_kwargs, resolve_model_string
from kenzo.vision.image_loader import load_image, to_data_url
from kenzo.vision.models import VisionAnalysis

_VISION_SYSTEM_PROMPT = """\
You are Kenzo's vision analysis component: you look at a screenshot from a \
developer's screen (an error page, a broken UI, a code editor, a terminal or \
log output) and extract every piece of structured information a coding agent \
would need to find and fix the underlying problem, without guessing beyond \
what is actually visible.

Respond with ONLY a single JSON object (no prose, no markdown fences) \
matching exactly this shape:
{
  "image_type": "ERROR_SCREEN" | "UI_SCREEN" | "CODE_SCREEN" | "LOG_SCREEN",
  "error_messages": ["<any error/exception messages visible, verbatim>"],
  "visible_text": "<all other meaningfully readable text, concatenated>",
  "stack_traces": ["<each stack trace frame or full trace block visible, verbatim>"],
  "ui_elements": ["<UI component/element names or descriptions visible, e.g. 'Submit button', 'Header nav'>"],
  "file_names": ["<every file name visible, e.g. from a stack trace or editor tab>"],
  "line_numbers": [<every line number visible, as integers>],
  "possible_causes": ["<your own best-effort hypotheses for the root cause, grounded in what's visible>"],
  "primary_error": "<the single most likely root error/exception, or null>",
  "primary_file": "<the single most likely file to look at first, or null>",
  "primary_line": <the single most likely line number to look at first, as an integer, or null>
}

Only extract what is ACTUALLY VISIBLE in the image -- do not invent file \
names, line numbers, or errors that are not shown. If a field has nothing to \
report, use an empty list/string, not a guess.\
"""

_DEFAULT_VISION_PROMPT = "Analyze this screenshot and extract the structured information described in your instructions."


def _parse_vision_analysis(content: str) -> VisionAnalysis:
    """Never raises: a response that isn't parseable JSON, or that parses
    but doesn't validate against `VisionAnalysis` (e.g. `line_numbers`
    containing a non-integer), degrades to a best-effort analysis carrying
    the raw text in `visible_text`/`raw_model_output` rather than losing the
    model's answer entirely."""
    parsed = extract_json(content)
    if parsed:
        try:
            return VisionAnalysis.model_validate({**parsed, "raw_model_output": content})
        except Exception:  # noqa: BLE001 - a malformed vision response must degrade, not crash the caller
            pass
    return VisionAnalysis(visible_text=content.strip()[:2000], raw_model_output=content)


class ModelRouter:
    def __init__(self, config: KenzoConfig) -> None:
        self.config = config

    def route(self, task_type: TaskType | str) -> tuple[str, ProviderConfig]:
        """Resolve a task type to (provider_name, provider_config).

        Raises RuntimeError if the resolved provider has no api_key set.
        """
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        provider_name = self.config.routing.for_task(task_type.value)
        provider_config = self.config.provider(provider_name)

        if not provider_config.api_key:
            raise RuntimeError(
                f"No API key configured for provider '{provider_name}' "
                f"(routed for task '{task_type.value}'). Set it with "
                f"`kenzo config set providers.{provider_name}.api_key <value>`, "
                f"or set the environment variable referenced by "
                f"providers.{provider_name}.api_key in ~/.kenzo/config.yaml "
                f"(e.g. via `${{{provider_name.upper()}_API_KEY}}`)."
            )

        return provider_name, provider_config

    def complete(
        self,
        task_type: TaskType | str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run a single (non-streaming) completion for the given task type."""
        provider_name, provider_config = self.route(task_type)
        model = resolve_model_string(provider_name, provider_config)
        completion_kwargs = build_completion_kwargs(provider_config)

        if tools is not None:
            completion_kwargs["tools"] = tools
            completion_kwargs["tool_choice"] = tool_choice

        return litellm.completion(
            model=model,
            messages=messages,
            **completion_kwargs,
            **kwargs,
        )

    def complete_text(
        self,
        task_type: TaskType | str,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        """Convenience wrapper for plain system/user prompt -> text completions."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.complete(task_type, messages, **kwargs)
        content = response.choices[0].message.content
        return content or ""

    def analyze_image(self, image: Union[str, Path, bytes], prompt: str = "", *, filename_hint: str = "") -> VisionAnalysis:
        """Analyze a screenshot (file path or raw bytes) and return a
        structured `VisionAnalysis`. Routed through `TaskType.VISION` --
        `RoutingConfig.vision` (default: "gemini") picks the provider, same
        as every other task type.

        `image` may be a path/Path (format detected from the extension) or
        raw bytes (format detected from magic bytes -- `filename_hint`, if
        given, is only a fallback for an inconclusive sniff, never trusted
        over it). Raises `kenzo.vision.image_loader.UnsupportedImageError`
        for a missing/oversized/unrecognized image -- this is a genuine input
        error worth surfacing to the caller, unlike a malformed MODEL
        response (see `_parse_vision_analysis`), which degrades gracefully
        instead of raising.
        """
        loaded = load_image(image, filename_hint=filename_hint)
        data_url = to_data_url(loaded)

        messages = [
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or _DEFAULT_VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        response = self.complete(TaskType.VISION, messages)
        content = response.choices[0].message.content or ""
        return _parse_vision_analysis(content)

    def stream(
        self,
        task_type: TaskType | str,
        messages: list[dict],
        **kwargs: Any,
    ) -> Generator[Any, None, None]:
        """Same as `complete`, but streams chunks back as they arrive."""
        provider_name, provider_config = self.route(task_type)
        model = resolve_model_string(provider_name, provider_config)
        completion_kwargs = build_completion_kwargs(provider_config)

        response: Iterable[Any] = litellm.completion(
            model=model,
            messages=messages,
            stream=True,
            **completion_kwargs,
            **kwargs,
        )
        for chunk in response:
            yield chunk
