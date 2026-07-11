"""Tool registry: the contract between tool implementations (filesystem, git,
terminal) and the agents/model layer that calls them via LLM function-calling.

Each concrete tool module (kenzo.tools.filesystem, .git, .terminal) exposes a
`get_tool_specs(repo_root: str, permissions: PermissionsConfig) -> list[ToolSpec]`
factory. `ToolRegistry` collects those specs, exposes them to litellm in
OpenAI tool-calling format, and dispatches calls by name.
"""

from __future__ import annotations

from kenzo.core.schemas import ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool '{spec.name}' is already registered")
        self._specs[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"No tool named '{name}'. Known tools: {sorted(self._specs)}")
        return self._specs[name]

    def list_specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def names(self) -> list[str]:
        return sorted(self._specs)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """Return a new ToolRegistry containing only the named tools.

        Used to give each pipeline agent role access to only the tools its
        job requires (e.g. the Reviewer gets read-only tools and can never
        reach `write_file`/`edit_file`, even if a model hallucinates a call
        to one). Unknown names are silently ignored rather than raising, so
        a role's tool list can safely name a tool that happens not to be
        registered in a given build without crashing registry construction.

        This is a genuine access-control boundary, not just a hint to the
        model: a tool excluded from a subset is absent from both
        `to_openai_tools()` (so the model is never offered it) and
        `execute()` (so even a hallucinated call to it fails closed with a
        clear "no tool named" error instead of running).
        """
        filtered = ToolRegistry()
        for name in names:
            if name in self._specs:
                filtered.register(self._specs[name])
        return filtered

    def to_openai_tools(self) -> list[dict]:
        """Render every registered tool as an OpenAI/litellm function-calling schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters
                    or {"type": "object", "properties": {}, "required": []},
                },
            }
            for spec in self._specs.values()
        ]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        try:
            spec = self.get(name)
        except KeyError as exc:
            return ToolResult(tool_name=name, success=False, error=str(exc))
        try:
            result = spec.func(**arguments)
        except Exception as exc:  # noqa: BLE001 - tool execution boundary, must not crash the agent loop
            return ToolResult(tool_name=name, success=False, error=f"{type(exc).__name__}: {exc}")
        if isinstance(result, ToolResult):
            return result
        return ToolResult(tool_name=name, success=True, output=str(result))
