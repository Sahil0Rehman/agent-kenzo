"""Shared base class for all Kenzo pipeline agents (Planner, Researcher, Coder,
Tester, Reviewer).

Each concrete agent subclasses `BaseAgent`, sets `name`/`task_type`, and
implements `run(state) -> dict` as the actual LangGraph node body. This module
also provides the common agentic tool-calling loop (`_tool_loop`) and a
best-effort JSON extraction helper (`_extract_json`) that every subclass
relies on to talk to the model and to parse its final structured answer.
"""

from __future__ import annotations

from typing import Any

from kenzo.core.schemas import TaskType
from kenzo.core.tool_loop import extract_json, run_tool_loop


class BaseAgent:
    """Common scaffolding shared by every Kenzo agent.

    Subclasses override `name` and `task_type` and implement `run`.
    """

    name: str = "base"
    task_type: TaskType = TaskType.GENERAL

    def __init__(self, router: Any, tools: Any, memory: Any) -> None:
        self.router = router
        self.tools = tools
        self.memory = memory

    def run(self, state: dict) -> dict:
        """Execute this agent's LangGraph node body.

        Must return a PARTIAL state update dict (only the keys this node
        changes), never the full state. Subclasses must override this.
        """
        raise NotImplementedError("Subclasses of BaseAgent must implement run()")

    def _tool_loop(
        self, system_prompt: str, user_prompt: str, max_iterations: int = 6
    ) -> tuple[str, list[dict]]:
        """Run an agentic tool-calling loop against the model router.

        Prepends the repository context bundle to the system prompt, then
        repeatedly calls the model, executing any requested tool calls and
        feeding their results back in, until the model returns a final
        (non-tool-call) answer or `max_iterations` is exhausted.

        Each tool observation fed back to the model is a JSON object with
        `success`, `output` (human summary), `data` (the tool's structured
        payload -- e.g. search_files' real matches, list_directory's real
        entries, git status's real branch/changed-files), and `error`. Tools
        like search_files/list_directory/git status put their actually
        useful information in `.data`, not `.output`; sending only `.output`
        back (as earlier versions of this loop did) left the model unable to
        see what its own tools found, so it could only use them as blind
        existence/count probes. Sending the full ToolResult lets the model
        reason from real tool output the way it's supposed to.

        Returns (final_text, tool_transcript) where tool_transcript is a list
        of {"tool": str, "arguments": dict, "result": dict} records in the
        order the tools were invoked.
        """
        context_bundle = self.memory.build_context_bundle()
        full_system_prompt = f"{context_bundle}\n\n{system_prompt}"
        return run_tool_loop(
            self.router, self.tools, self.task_type, self.name, full_system_prompt, user_prompt, max_iterations
        )

    def _coerce_bool(self, value: Any, default: bool = False) -> bool:
        """Coerce a JSON-decoded field that is supposed to be a boolean into an
        actual `bool`, tolerating models that emit it as a string (e.g.
        `"approved": "false"`) instead of a real JSON boolean literal.

        This matters because `bool("false")` is `True` in Python (any
        non-empty string is truthy) -- naively casting a model's stringified
        boolean with `bool(...)` silently inverts a "false"/"no" answer into
        an approved-looking truthy value. Every call site that reads a
        model-authored boolean (`approved`, `passed`, ...) must go through
        this instead of a bare `bool(...)` cast.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "y", "1"):
                return True
            if lowered in ("false", "no", "n", "0", ""):
                return False
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    _WORKING_CONTEXT_NOTE_MAX_CHARS = 220
    _WORKING_CONTEXT_MAX_ITEMS = 100

    def _render_working_context(self, working_context: dict) -> str:
        """Render the accumulated cross-agent WorkingContext (see
        kenzo.core.schemas.WorkingContext) into a short text block prepended
        to this agent's user prompt, so -- unlike before, when only a terse
        plan/research JSON survived between stages -- the Coder/Tester/
        Reviewer can see which files earlier agents actually opened and what
        they noticed, without needing to rediscover it from scratch.
        """
        files = working_context.get("files_discovered") or []
        notes = working_context.get("notes") or []
        if not files and not notes:
            return ""

        lines = ["Working context accumulated by earlier agents this run:"]
        if files:
            lines.append("Files already discovered: " + ", ".join(files))
        if notes:
            lines.append("Observations so far:")
            lines.extend(f"  - {note}" for note in notes)
        return "\n".join(lines)

    def _update_working_context(self, working_context: dict, tool_transcript: list[dict]) -> dict:
        """Fold this agent's tool_transcript into the shared WorkingContext.

        Deliberately stores only file paths and short (<=220 char) notes --
        never full file contents -- so this stays cheap to carry through
        every remaining stage of the run instead of re-inflating the same
        context-bloat problem full transcripts would cause.
        """
        files = list(working_context.get("files_discovered") or [])
        notes = list(working_context.get("notes") or [])
        seen = set(files)

        def add_file(path: str) -> None:
            if path and path not in seen:
                seen.add(path)
                files.append(path)

        def add_note(text: str) -> None:
            notes.append(f"[{self.name}] {text}"[: self._WORKING_CONTEXT_NOTE_MAX_CHARS])

        for entry in tool_transcript:
            tool_name = entry.get("tool")
            result = entry.get("result") or {}
            if not result.get("success"):
                continue
            data = result.get("data") or {}
            arguments = entry.get("arguments") or {}

            if tool_name in ("read_file", "write_file", "edit_file"):
                add_file(arguments.get("path") or data.get("path", ""))
            elif tool_name == "search_files":
                matches = data.get("matches") or []
                for match in matches[:5]:
                    add_file(match.get("path", ""))
                if matches:
                    add_note(f"search '{arguments.get('pattern', '')}' found {len(matches)} match(es)")
            elif tool_name == "repository_tree":
                under = arguments.get("path")
                add_note("inspected repository tree" + (f" under '{under}'" if under else ""))
            elif tool_name == "list_directory":
                entries = data.get("entries") or []
                add_note(f"listed {len(entries)} entries in '{arguments.get('path', '.')}'")
            elif tool_name == "find_symbol":
                symbol = data.get("symbol") or {}
                if symbol:
                    add_file(symbol.get("file", ""))
                    kind = symbol.get("type") or symbol.get("kind") or ""
                    add_note(f"found symbol '{symbol.get('name', '')}' ({kind}) in {symbol.get('file', '')}")
                else:
                    add_note(f"find_symbol('{arguments.get('name', '')}') found nothing")
            elif tool_name == "map_flow":
                add_note(f"mapped flow for endpoint '{arguments.get('endpoint', '')}'")

        return {
            "files_discovered": files[-self._WORKING_CONTEXT_MAX_ITEMS :],
            "notes": notes[-self._WORKING_CONTEXT_MAX_ITEMS :],
        }

    def _extract_json(self, text: str) -> dict | None:
        """Best-effort extraction of the first balanced JSON object in `text`.

        Models often wrap JSON in prose or markdown code fences. See
        `kenzo.core.tool_loop.extract_json` for the actual implementation
        (shared with `AgentController`, which needs the identical heuristic
        for parsing its own decision/action responses).
        """
        return extract_json(text)
