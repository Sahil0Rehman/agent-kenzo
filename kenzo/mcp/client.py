"""`MCPClient`: the single entry point `kenzo.tools.registry
.build_default_registry` uses to connect every configured MCP server and
discover its tools as native-shaped `ToolSpec`s.

Connection happens synchronously, once, at construction-adjacent `connect_
all()` time (unlike Phase 5's `IndexingService`, which backgrounds a
genuinely slow full-repo walk) -- MCP tool AVAILABILITY needs to be known
before `AgentController`'s action-scoping logic can decide whether to offer
them for a given action (see `kenzo.core.controller`'s dynamic MCP tool-
source scoping), and connecting a handful of servers takes low single-digit
seconds per server (empirically measured), not minutes.
"""

from __future__ import annotations

from kenzo.config.settings import MCPConfig
from kenzo.core.schemas import ToolSpec
from kenzo.mcp.adapter import build_tool_spec
from kenzo.mcp.connection import MCPConnectResult, MCPServerConnection
from kenzo.mcp.permissions import ToolPermission, resolve_tool_permission


class MCPClient:
    """Owns every configured MCP server's `MCPServerConnection` for the
    lifetime of one `AgentRuntime`."""

    def __init__(self, config: MCPConfig) -> None:
        self.config = config
        self.connections: dict[str, MCPServerConnection] = {
            name: MCPServerConnection(name, server_config, config.connect_timeout_seconds, config.call_timeout_seconds)
            for name, server_config in config.servers.items()
        }
        self._connect_results: dict[str, MCPConnectResult] = {}
        # Populated by discover_tool_specs(); registered tool name ("server_
        # tool") -> permission. Cached so tool_names_by_permission (called
        # once per AgentController action-scoping decision, potentially many
        # times per task) never re-hits every MCP server's list_tools() --
        # it just filters this dict.
        self._tool_permissions: dict[str, ToolPermission] = {}

    def connect_all(self) -> dict[str, MCPConnectResult]:
        """Connect every configured server. NEVER raises: a server that's
        offline, misconfigured, or times out is recorded in the returned
        dict and simply contributes no tools -- the other servers connect
        independently and are unaffected (item 8: one dead server can't take
        the rest down, or `AgentRuntime` construction, with it).
        """
        for name, connection in self.connections.items():
            self._connect_results[name] = connection.connect()
        return dict(self._connect_results)

    def disconnect_all(self) -> None:
        for connection in self.connections.values():
            connection.disconnect()

    def discover_tool_specs(self) -> list[ToolSpec]:
        """Build ToolSpecs for every tool on every currently-connected
        server, and (re)populate the permission cache `tool_names_by_
        permission` reads. A server whose `list_tools()` call itself fails (a
        mid-session error, distinct from a connect-time failure) simply
        contributes no tools for this call -- never raises, never affects
        other servers. Safe to call more than once (e.g. after a
        reconnect); each call fully replaces the previous permission cache.
        """
        specs: list[ToolSpec] = []
        tool_permissions: dict[str, ToolPermission] = {}

        for name, connection in self.connections.items():
            if not connection.is_connected:
                continue
            tools, error = connection.list_tools()
            if error or not tools:
                continue
            server_config = self.config.servers[name]
            for tool_info in tools:
                permission = resolve_tool_permission(
                    tool_info.name, tool_info.description, server_config.tool_permission_overrides
                )
                spec = build_tool_spec(
                    name, tool_info, connection, permission, self.config.allow_write, self.config.allow_destructive
                )
                specs.append(spec)
                tool_permissions[spec.name] = permission

        self._tool_permissions = tool_permissions
        return specs

    def tool_names_by_permission(self, permission: ToolPermission) -> list[str]:
        """The registered (`"{server}_{tool}"`) names of every discovered
        tool at exactly `permission`'s level, from the cache `discover_tool_
        specs()` populates -- used by `kenzo.core.controller`'s dynamic
        tool-source scoping to decide which MCP tools an action's tool
        subset should include."""
        return [name for name, perm in self._tool_permissions.items() if perm is permission]

    @property
    def connect_results(self) -> dict[str, MCPConnectResult]:
        return dict(self._connect_results)
