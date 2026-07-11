"""Fake stdio MCP server with one deliberately slow tool, for the Phase 6
timeout-enforcement test (`tests/test_phase6_mcp_error_recovery.py`). Spawned
as a real subprocess by `kenzo.mcp.connection.MCPServerConnection` -- never
imported directly by the tests that use it.

`slow_tool` sleeps far longer than the short `call_timeout_seconds` those
tests configure (e.g. 1-2s), so a passing test proves
`MCPServerConnection.call_tool`'s `asyncio.wait_for`/`future.result(timeout=
...)` timeout is actually enforced -- the call returns
`ToolResult(success=False, ...)` close to the configured timeout, not after
the tool's full sleep duration.
"""

from __future__ import annotations

import time

from mcp.server.fastmcp import FastMCP

mcp_app = FastMCP("slow-fake")


@mcp_app.tool()
def slow_tool() -> str:
    """Sleep for 3 seconds, then return -- deliberately slower than the
    short call_timeout_seconds used by the tests that call this tool."""
    time.sleep(3)
    return "done after sleep"


if __name__ == "__main__":
    mcp_app.run(transport="stdio")
