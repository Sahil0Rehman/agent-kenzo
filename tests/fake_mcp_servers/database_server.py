"""Fake stdio MCP server shaped like a real database MCP integration, for
Phase 6 component/error-recovery tests. Spawned as a real subprocess by
`kenzo.mcp.connection.MCPServerConnection` -- never imported directly by the
tests that use it.

Tool-name verbs are chosen deliberately so `kenzo.mcp.permissions
.classify_tool_permission`'s heuristic classifies each one correctly without
needing a `tool_permission_overrides` entry:

- query           -> READ        ("query")
- describe_schema -> READ        ("describe")
- execute_update  -> WRITE       ("execute"/"update" -- checked before READ)
- drop_table      -> DESTRUCTIVE ("drop" -- checked before WRITE/READ)
- always_fails    -> (unclassified name, defaults to WRITE) -- deliberately
  raises server-side so tests can assert a real MCP `isError=True` result
  surfaces a real, non-empty error message through
  `MCPServerConnection.call_tool`.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp_app = FastMCP("database-fake")


@mcp_app.tool()
def query(sql: str) -> str:
    """Run a read-only SQL query and return the matching rows."""
    if sql.strip().lower().startswith("select"):
        return "id | username | email\n1 | jdoe | jdoe@example.com\n2 | asmith | asmith@example.com"
    return f"Error: query() only supports SELECT statements, got: {sql!r}"


@mcp_app.tool()
def describe_schema(table_name: str) -> str:
    """Describe the columns of a table."""
    return f"Table '{table_name}': id (int, pk), username (varchar), email (varchar), created_at (timestamp)"


@mcp_app.tool()
def execute_update(sql: str) -> str:
    """Execute an UPDATE/INSERT statement against the database."""
    return f"OK: 1 row affected by: {sql}"


@mcp_app.tool()
def drop_table(table_name: str) -> str:
    """Permanently drop a table from the database."""
    return f"OK: table '{table_name}' dropped."


@mcp_app.tool()
def always_fails() -> str:
    """A tool that always raises server-side, for error-recovery tests."""
    raise ValueError("simulated failure: always_fails() never succeeds")


if __name__ == "__main__":
    mcp_app.run(transport="stdio")
