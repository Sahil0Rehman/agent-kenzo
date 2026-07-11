"""`MCPToolAdapter`: converts one discovered MCP tool
(`kenzo.mcp.connection.MCPToolInfo`) into a native Kenzo `ToolSpec`
(`kenzo.core.schemas.ToolSpec`), permission-gated at CALL time -- not at
discovery time.

Every MCP tool is registered and visible to the model regardless of its
permission level (so the agent can see what capability exists and explain to
the user why it's blocked); a WRITE/DESTRUCTIVE-level tool's wrapped `func`
checks `kenzo.mcp.permissions.is_call_allowed` on every invocation and fails
closed with a clear, actionable error if not currently allowed. This mirrors
the existing pattern native tools already use (e.g. `write_file`/
`run_command` checking `PermissionsConfig.allow_write`/`allow_shell_exec`
inside the tool function itself, rather than being conditionally registered)
-- a hallucinated or attempted call to a disallowed MCP tool fails closed
with a clear message, it never silently executes.

`MCPToolInfo.input_schema` is used directly as `ToolSpec.parameters` with no
conversion -- the MCP SDK already reports it in the same JSON-Schema shape
(`{"type": "object", "properties": {...}, "required": [...]}`) Kenzo's
`ToolSpec.parameters` expects, confirmed empirically against a real server
before writing this module.
"""

from __future__ import annotations

from typing import Any

from kenzo.core.schemas import ToolResult, ToolSpec
from kenzo.mcp.connection import MCPServerConnection, MCPToolInfo
from kenzo.mcp.permissions import ToolPermission, is_call_allowed


def build_tool_spec(
    server_name: str,
    tool_info: MCPToolInfo,
    connection: MCPServerConnection,
    permission: ToolPermission,
    allow_write: bool,
    allow_destructive: bool,
) -> ToolSpec:
    """Build one ToolSpec for `tool_info` from `server_name`. The registered
    tool name is `"{server_name}_{tool_info.name}"` (underscore-joined, per
    the Phase 6 spec's own examples -- "github_create_pr", "database_query")
    -- both to disambiguate identically-named tools across two different
    servers, and because dots aren't universally safe in every provider's
    function-calling name validation, underscores always are.
    """
    full_name = f"{server_name}_{tool_info.name}"
    description = f"[MCP:{server_name}, permission={permission.value}] {tool_info.description}".strip()

    def _call(**arguments: Any) -> ToolResult:
        if not is_call_allowed(permission, allow_write, allow_destructive):
            required_flag = "allow_write" if permission is ToolPermission.WRITE else "allow_destructive"
            return ToolResult(
                tool_name=full_name,
                success=False,
                error=(
                    f"'{full_name}' is a {permission.value.upper()}-level MCP tool and requires approval -- "
                    f"set mcp.{required_flag}: true in ~/.kenzo/config.yaml (or "
                    f"`kenzo config set mcp.{required_flag} true`) to allow it."
                ),
            )
        return connection.call_tool(tool_info.name, arguments)

    return ToolSpec(
        name=full_name,
        description=description,
        parameters=tool_info.input_schema or {"type": "object", "properties": {}, "required": []},
        func=_call,
    )
