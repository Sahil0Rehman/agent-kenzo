"""Phase 6 "MCP Integration" error-recovery tests: every failure mode the
foundation (`kenzo.mcp.connection`/`.adapter`/`.client`) is supposed to
degrade gracefully from, exercised for real -- a genuinely offline server, a
genuinely slow tool call against a short timeout, an intentionally minimal
tool schema, a tool that raises server-side, and (the "item 8" graceful-
degradation requirement) proof that a totally-unavailable MCP layer leaves
Kenzo's native tools -- and, since the orchestrator's parallel wiring of
`AgentRuntime`/`kenzo.tools.registry.build_default_registry` finished before
this file was written (see module docstring notes below and the final
report), the FULL runtime construction path -- completely unaffected.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from kenzo.config.settings import KenzoConfig, MCPConfig, MCPServerConfig, PermissionsConfig, ProviderConfig
from kenzo.core.runtime import AgentRuntime
from kenzo.mcp.adapter import build_tool_spec
from kenzo.mcp.client import MCPClient
from kenzo.mcp.connection import MCPServerConnection, MCPToolInfo
from kenzo.mcp.permissions import ToolPermission
from kenzo.tools.registry import build_default_registry

FAKE_SERVERS_DIR = Path(__file__).parent / "fake_mcp_servers"
PY = sys.executable


def _server_config(script_name: str) -> MCPServerConfig:
    return MCPServerConfig(command=PY, args=[str(FAKE_SERVERS_DIR / script_name)])


# ---------------------------------------------------------------------------
# Server offline
# ---------------------------------------------------------------------------


def test_connect_all_with_only_dead_servers_does_not_raise_and_yields_zero_tools():
    """Direct `MCPClient`-level check of the "server offline" scenario --
    the orchestrator's `AgentRuntime`/`AgentController` wiring (see
    `test_agent_runtime_construction_survives_dead_mcp_server_and_native_
    tools_still_work` below) had already landed by the time this file was
    written, so that fuller path is also covered below; this test isolates
    the same guarantee at the `MCPClient` layer alone."""
    config = MCPConfig(
        servers={
            "dead_a": MCPServerConfig(command="nonexistent-cmd-offline-a-kenzo"),
            "dead_b": MCPServerConfig(command="nonexistent-cmd-offline-b-kenzo"),
        }
    )
    client = MCPClient(config)
    try:
        results = client.connect_all()  # must not raise
        assert set(results) == {"dead_a", "dead_b"}
        assert all(not r.success and r.error for r in results.values())

        specs = client.discover_tool_specs()  # must not raise
        assert specs == []
        assert client.tool_names_by_permission(ToolPermission.READ) == []
    finally:
        client.disconnect_all()


def test_agent_runtime_construction_survives_dead_mcp_server_and_native_tools_still_work(repo_root, kenzo_home):
    """Full `AgentRuntime` construction (the orchestrator's parallel wiring
    target) with only a dead MCP server configured: construction must not
    raise, `mcp_client.discover_tool_specs()` must be empty, and native
    tools (`read_file` et al, via `build_default_registry`) must be fully
    present and functional regardless. Never calls `.run()`/`ModelRouter
    .complete` -- pure construction-time behavior, no model calls."""
    config = KenzoConfig(
        providers={"nemotron": ProviderConfig(api_key="fake-key", default_model="fake-model")},
        permissions=PermissionsConfig(allow_write=True, allow_shell_exec=False),
        mcp=MCPConfig(servers={"dead": MCPServerConfig(command="nonexistent-cmd-runtime-kenzo")}),
    )

    runtime = AgentRuntime(repo_path=str(repo_root), config=config)
    try:
        assert runtime.mcp_client.connect_results["dead"].success is False
        assert runtime.mcp_client.discover_tool_specs() == []

        assert "read_file" in runtime.tools.names()
        result = runtime.tools.execute("read_file", {"path": "README.md"})
        assert result.success is True
        assert "Sample repo" in result.output
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------


def test_timeout_is_enforced_close_to_configured_timeout_not_tool_sleep_duration():
    """`slow_server.py`'s `slow_tool` sleeps 3s; with `call_timeout=1.5`,
    the call must return `ToolResult(success=False, ...)` in real wall-clock
    time close to 1.5s -- proving the timeout is actually enforced, not
    merely configured and ignored (which would show ~3s elapsed instead)."""
    conn = MCPServerConnection("slow", _server_config("slow_server.py"), connect_timeout=10, call_timeout=1.5)
    try:
        connect_result = conn.connect()
        assert connect_result.success, f"fake slow_server failed to connect: {connect_result.error}"

        t0 = time.perf_counter()
        result = conn.call_tool("slow_tool", {})
        elapsed = time.perf_counter() - t0

        assert result.success is False
        assert result.error
        # Empirically observed ~1.50s (matches the configured call_timeout
        # almost exactly). Bounds: comfortably above the timeout itself
        # (some scheduling overhead is expected) but clearly, unambiguously
        # short of the tool's full 3s sleep.
        assert 1.4 <= elapsed <= 2.5, f"expected ~1.5s (the configured timeout), got {elapsed}s"
    finally:
        conn.disconnect()


# ---------------------------------------------------------------------------
# Invalid/unusual schema handling
# ---------------------------------------------------------------------------


def test_build_tool_spec_falls_back_to_default_schema_for_empty_input_schema():
    """`MCPToolInfo` constructed directly (bypassing any real server) with
    an empty `input_schema` -- verifies `kenzo.mcp.adapter.build_tool_spec`'s
    documented fallback (`tool_info.input_schema or {"type": "object",
    "properties": {}, "required": []}`) actually triggers and produces a
    well-formed dict, not that some other schema-repair logic exists (it
    doesn't, and this test does not invent one)."""
    tool_info = MCPToolInfo(name="mystery_tool", description="a tool with no declared schema", input_schema={})

    # `connection` is never touched at build time (only inside `.func`,
    # which this test never calls) -- `None` is safe and avoids spawning a
    # server for a test that is purely about the schema fallback.
    spec = build_tool_spec(
        "unknownserver", tool_info, connection=None, permission=ToolPermission.READ, allow_write=True, allow_destructive=True
    )

    assert spec.name == "unknownserver_mystery_tool"
    assert spec.parameters == {"type": "object", "properties": {}, "required": []}


# ---------------------------------------------------------------------------
# Tool failure (server-side exception -> clean ToolResult, not a crash)
# ---------------------------------------------------------------------------


def test_server_side_tool_exception_surfaces_as_clean_failed_toolresult():
    """`database_server.py`'s `always_fails` tool deliberately raises
    server-side (`ValueError`). Real observed behavior: FastMCP catches it
    and returns `CallToolResult(isError=True, ...)`, which
    `MCPServerConnection.call_tool` converts into `ToolResult(success=False,
    error=<real, non-empty message>)` -- not an empty/generic error, and not
    an exception propagating out of `call_tool` itself."""
    conn = MCPServerConnection("database", _server_config("database_server.py"), connect_timeout=10, call_timeout=10)
    try:
        connect_result = conn.connect()
        assert connect_result.success, f"fake database_server failed to connect: {connect_result.error}"

        result = conn.call_tool("always_fails", {})

        assert result.success is False
        assert result.error
        assert "simulated failure" in result.error
    finally:
        conn.disconnect()


# ---------------------------------------------------------------------------
# Graceful fallback proof: native tools unaffected by a totally-unavailable
# MCP layer, through the real integration point AgentRuntime actually uses.
# ---------------------------------------------------------------------------


def test_build_default_registry_native_tools_unaffected_by_dead_mcp_client(repo_root):
    """`kenzo.tools.registry.build_default_registry` has already been
    extended (by the orchestrator's parallel wiring, complete by the time
    this test was written) to accept an optional `mcp_client`. With an
    `MCPClient` whose only configured server is dead, the resulting registry
    must contain exactly the same native tool set as one built with no
    `mcp_client` at all -- zero MCP tools merged in, and no degradation of
    the native ones."""
    permissions = PermissionsConfig(allow_write=True, allow_shell_exec=False)

    baseline_registry = build_default_registry(str(repo_root), permissions)

    dead_client = MCPClient(MCPConfig(servers={"dead": MCPServerConfig(command="nonexistent-cmd-registry-kenzo")}))
    try:
        dead_client.connect_all()
        registry_with_dead_mcp = build_default_registry(str(repo_root), permissions, mcp_client=dead_client)

        assert registry_with_dead_mcp.names() == baseline_registry.names()
        assert "read_file" in registry_with_dead_mcp.names()
        assert "write_file" in registry_with_dead_mcp.names()

        result = registry_with_dead_mcp.execute("read_file", {"path": "README.md"})
        assert result.success is True
        assert "Sample repo" in result.output
    finally:
        dead_client.disconnect_all()
