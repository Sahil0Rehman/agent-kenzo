"""Shared agentic tool-calling loop.

Extracted out of `kenzo.agents.base.BaseAgent._tool_loop` (which now just
prepends the memory context bundle and delegates here, with byte-for-byte
identical behavior) so `kenzo.core.controller.AgentController`'s per-action
execution step can reuse the exact same tool-observation format (the
Phase 1.5 fix: models see structured success/output/data/error, not a bare
string) instead of re-implementing it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional


def run_tool_loop(
    router: Any,
    tools: Any,
    task_type: Any,
    agent_name: str,
    full_system_prompt: str,
    user_prompt: str,
    max_iterations: int = 6,
    on_tool_start: Optional[Callable[[str, dict], None]] = None,
    on_tool_result: Optional[Callable[[str, dict, Any], None]] = None,
) -> tuple[str, list[dict]]:
    """Repeatedly call the model, executing any requested tool calls and
    feeding their results back in, until it returns a final (non-tool-call)
    answer or `max_iterations` is exhausted.

    Each tool observation fed back to the model is a JSON object with
    `success`, `output` (human summary), `data` (the tool's structured
    payload), and `error` -- see the Phase 1.5 hardening note in
    `kenzo.agents.base.BaseAgent._tool_loop` for why this matters.

    `on_tool_start(tool_name, arguments)`/`on_tool_result(tool_name,
    arguments, result)` are optional Phase 7 hooks -- called immediately
    before/after `tools.execute(...)`, letting a caller (e.g.
    `AgentController`, wired to `kenzo.server`'s WebSocket event stream)
    stream `tool_start`/`tool_result`/`file_changed`/`test_running` events
    without this module needing to know anything about events itself. Both
    default to `None` (no-op) -- every existing caller (`BaseAgent
    ._tool_loop`, `AgentController._execute_action`'s pre-Phase-7 call sites)
    is unaffected.

    Returns (final_text, tool_transcript) where tool_transcript is a list of
    {"tool": str, "arguments": dict, "result": dict} records in the order
    the tools were invoked.
    """
    messages: list[dict] = [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tool_transcript: list[dict] = []

    for _ in range(max_iterations):
        response = router.complete(task_type, messages, tools=tools.to_openai_tools())
        message = response.choices[0].message

        if message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                if on_tool_start is not None:
                    on_tool_start(tool_call.function.name, arguments)

                result = tools.execute(tool_call.function.name, arguments)

                if on_tool_result is not None:
                    on_tool_result(tool_call.function.name, arguments, result)

                observation = {
                    "success": result.success,
                    "output": result.output,
                    "data": result.data,
                    "error": result.error,
                }
                content = json.dumps(observation, default=str)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": content,
                    }
                )
                tool_transcript.append(
                    {
                        "tool": tool_call.function.name,
                        "arguments": arguments,
                        "result": result.model_dump(),
                    }
                )
            continue

        return message.content or "", tool_transcript

    placeholder = (
        f"[{agent_name}] tool loop exhausted after {max_iterations} iterations "
        "without a final answer."
    )
    return placeholder, tool_transcript


def extract_json(text: str) -> dict | None:
    """Best-effort extraction of the first balanced JSON object in `text`.

    Models often wrap JSON in prose or markdown code fences. This strips a
    leading/trailing ``` (optionally ```json) fence if present, then scans
    for the first balanced `{...}` substring and attempts to parse it.
    Returns None if nothing parseable is found.
    """
    if not text:
        return None

    cleaned = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*)\n?```$", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(cleaned)):
            char = cleaned[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)

    return None
