"""Phase 6 MCP (Model Context Protocol) client integration: connects to
external MCP servers (stdio subprocesses or HTTP services), discovers their
tools, converts them into Kenzo's native `ToolSpec` shape, and exposes them
to `AgentController` alongside native/code-intelligence/memory tools --
permission-gated by `kenzo.mcp.permissions.ToolPermission`.

Additive to every prior phase: MCP tools are just more entries in the same
`ToolRegistry` (`kenzo.tools.base.ToolRegistry`) native tools already use --
nothing about how Kenzo executes/scopes/checkpoints tool calls changes.
"""
