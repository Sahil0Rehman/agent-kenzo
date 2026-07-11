"""`MCPServerConnection`: owns one MCP server's connection lifecycle and
bridges its fundamentally ASYNC protocol (the official `mcp` SDK's
`ClientSession`, built on `anyio`/`asyncio`) into the synchronous call shape
every other part of Kenzo already expects (`kenzo.tools.base.ToolRegistry
.execute(name, arguments) -> ToolResult`).

The bridge: a dedicated background thread runs its OWN `asyncio` event loop
for the lifetime of the connection. The real `stdio_client`/`streamablehttp
_client`/`sse_client` async context manager and `ClientSession` are entered
and kept alive INSIDE that loop (a persistent connection, not reconnected
per call -- MCP sessions are stateful). Synchronous callers submit
coroutines via `asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=
...)`, which blocks the calling thread until the async operation completes
(or times out) -- this is what lets `call_tool`/`list_tools` below present a
plain, blocking, exception-safe method to the rest of Kenzo.

This exact pattern (background event-loop thread + `run_coroutine_threadsafe`
+ `AsyncExitStack` to hold the nested `stdio_client`/`ClientSession` context
managers open across many calls) was empirically verified end-to-end before
writing this file: a real stdio subprocess server, `list_tools()`/
`call_tool()` round trips, a failed/unknown tool call (`CallToolResult
.isError=True`, no exception), a nonexistent command (`FileNotFoundError`),
and a process that exits immediately without speaking MCP (`ExceptionGroup`
from the underlying `anyio` task group) -- all confirmed to fail cleanly and
quickly rather than hanging, which is what makes the graceful-degradation
behavior below trustworthy rather than aspirational.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Optional

from kenzo.config.settings import MCPServerConfig
from kenzo.core.schemas import ToolResult


@dataclass
class MCPConnectResult:
    success: bool
    error: Optional[str] = None


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


class MCPServerConnection:
    """One configured MCP server's live connection. `connect()` NEVER
    raises -- a server that's offline, misconfigured, or not a real MCP
    endpoint degrades to `MCPConnectResult(success=False, error=...)`, so a
    caller connecting several servers can skip the failed ones and continue
    with whatever DID connect (see `kenzo.mcp.client.MCPClient.connect_all`).
    """

    def __init__(self, name: str, config: MCPServerConfig, connect_timeout: float = 15.0, call_timeout: float = 30.0) -> None:
        self.name = name
        self.config = config
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Any = None
        self._ready = threading.Event()
        self._stop_event: Optional[asyncio.Event] = None
        self._connect_error: Optional[BaseException] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> MCPConnectResult:
        """Start the background event-loop thread and block until the
        connection is ready, fails, or `connect_timeout_seconds` elapses.
        Idempotent: a second call on an already-connected instance returns
        success immediately without reconnecting.
        """
        if self._connected:
            return MCPConnectResult(success=True)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f"mcp-{self.name}")
        self._thread.start()

        if not self._ready.wait(timeout=self._connect_timeout):
            self._connected = False
            return MCPConnectResult(
                success=False,
                error=f"Timed out connecting to MCP server '{self.name}' after {self._connect_timeout}s.",
            )

        if self._connect_error is not None:
            self._connected = False
            return MCPConnectResult(success=False, error=f"{type(self._connect_error).__name__}: {self._connect_error}")

        self._connected = True
        return MCPConnectResult(success=True)

    def disconnect(self) -> None:
        if self._loop is not None and self._loop.is_running() and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._connected = False

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_hold())

    async def _connect_and_hold(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                read, write = await self._open_transport(stack)
                session = await stack.enter_async_context(_client_session(read, write))
                await asyncio.wait_for(session.initialize(), timeout=self._connect_timeout)
                self._session = session
                self._stop_event = asyncio.Event()
                self._ready.set()
                await self._stop_event.wait()
        except Exception as exc:  # noqa: BLE001 - connection boundary: every failure mode must degrade, never crash the process
            self._connect_error = exc
            self._ready.set()

    async def _open_transport(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        if self.config.command:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self.config.command,
                args=list(self.config.args),
                env=dict(self.config.env) or None,
            )
            return await stack.enter_async_context(stdio_client(params))

        if self.config.url:
            if self.config.transport == "sse":
                from mcp.client.sse import sse_client

                return await stack.enter_async_context(sse_client(self.config.url))

            from mcp.client.streamable_http import streamablehttp_client

            read, write, _get_session_id = await stack.enter_async_context(streamablehttp_client(self.config.url))
            return read, write

        raise ValueError(f"MCP server '{self.name}' has neither 'command' nor 'url' configured.")

    # ------------------------------------------------------------------
    # Operations (synchronous, exception-safe)
    # ------------------------------------------------------------------

    def list_tools(self) -> tuple[list[MCPToolInfo], Optional[str]]:
        """Returns `(tools, error)` -- exactly one of which is meaningful.
        Never raises: a timeout or protocol error on an already-connected
        session returns `([], "<error>")` rather than propagating."""
        if not self._connected or self._session is None:
            return [], f"MCP server '{self.name}' is not connected."

        try:
            future = asyncio.run_coroutine_threadsafe(self._list_tools_coro(), self._loop)
            result = future.result(timeout=self._call_timeout)
            return result, None
        except Exception as exc:  # noqa: BLE001 - call boundary: never raise into ToolRegistry.execute's caller
            return [], f"{type(exc).__name__}: {exc}"

    async def _list_tools_coro(self) -> list[MCPToolInfo]:
        result = await asyncio.wait_for(self._session.list_tools(), timeout=self._call_timeout)
        return [
            MCPToolInfo(name=t.name, description=t.description or "", input_schema=t.inputSchema or {})
            for t in result.tools
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Calls `tool_name` and converts the MCP `CallToolResult` into
        Kenzo's `ToolResult`. Never raises -- a dead connection, a timeout,
        or a server-side tool failure (`CallToolResult.isError`) all become
        `ToolResult(success=False, error=...)`, matching how every other
        tool in Kenzo already fails closed rather than propagating."""
        full_name = f"{self.name}.{tool_name}"
        if not self._connected or self._session is None:
            return ToolResult(tool_name=full_name, success=False, error=f"MCP server '{self.name}' is not connected.")

        try:
            future = asyncio.run_coroutine_threadsafe(self._call_tool_coro(tool_name, arguments), self._loop)
            call_result = future.result(timeout=self._call_timeout)
        except Exception as exc:  # noqa: BLE001 - call boundary, see docstring
            return ToolResult(tool_name=full_name, success=False, error=f"{type(exc).__name__}: {exc}")

        text = "\n".join(getattr(block, "text", "") for block in (call_result.content or []) if getattr(block, "text", None))
        if call_result.isError:
            return ToolResult(tool_name=full_name, success=False, error=text or "MCP tool call failed.")
        return ToolResult(tool_name=full_name, success=True, output=text, data={"content": text})

    async def _call_tool_coro(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return await asyncio.wait_for(self._session.call_tool(tool_name, arguments), timeout=self._call_timeout)


def _client_session(read: Any, write: Any) -> Any:
    from mcp import ClientSession

    return ClientSession(read, write)
