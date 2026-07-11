"""Phase 6 "MCP Integration" component tests: exercises the ALREADY-BUILT,
empirically-verified foundation (`kenzo.mcp.connection`, `.adapter`,
`.client`, `.permissions`) against real fake stdio MCP servers under
`tests/fake_mcp_servers/`. Nothing here mocks the MCP protocol -- every
"real" claim below is a real subprocess, spawned via
`kenzo.mcp.connection.MCPServerConnection`, speaking real MCP over stdio.

Fixture strategy: connecting a real MCP server costs several real wall-clock
seconds (subprocess spawn + `mcp` SDK session handshake, empirically ~3-5s
per server on this machine) -- see the module-scoped `github_connection` /
`database_connection` / `docs_connection` fixtures below. Each is connected
ONCE per test module run and reused read-only across every test that needs
it, rather than reconnecting per test, to keep the full suite's wall-clock
time reasonable. Permission-gate tests that need a specific
`(allow_write, allow_destructive)` combination call
`kenzo.mcp.adapter.build_tool_spec` directly against one of these already-
open connections (still a fully real call through the real subprocess --
only the connection's `.connect()` step is shared, never the tool call
itself). Every fixture that opens a connection disconnects it at module
teardown via `yield` + `finally`, per the "no leaked subprocesses/threads"
requirement.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from kenzo.config.settings import MCPConfig, MCPServerConfig
from kenzo.mcp.adapter import build_tool_spec
from kenzo.mcp.client import MCPClient
from kenzo.mcp.connection import MCPServerConnection
from kenzo.mcp.permissions import ToolPermission, resolve_tool_permission

FAKE_SERVERS_DIR = Path(__file__).parent / "fake_mcp_servers"
PY = sys.executable


def _server_config(script_name: str) -> MCPServerConfig:
    return MCPServerConfig(command=PY, args=[str(FAKE_SERVERS_DIR / script_name)])


def _tool_info(tools, name: str):
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"tool '{name}' not found among {[t.name for t in tools]}")


# ---------------------------------------------------------------------------
# Module-scoped real connections, reused read-only across many tests below.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def github_connection():
    conn = MCPServerConnection("github", _server_config("github_server.py"), connect_timeout=10, call_timeout=10)
    result = conn.connect()
    assert result.success, f"fake github_server failed to connect: {result.error}"
    try:
        yield conn
    finally:
        conn.disconnect()


@pytest.fixture(scope="module")
def database_connection():
    conn = MCPServerConnection("database", _server_config("database_server.py"), connect_timeout=10, call_timeout=10)
    result = conn.connect()
    assert result.success, f"fake database_server failed to connect: {result.error}"
    try:
        yield conn
    finally:
        conn.disconnect()


@pytest.fixture(scope="module")
def docs_connection():
    conn = MCPServerConnection("docs", _server_config("docs_server.py"), connect_timeout=10, call_timeout=10)
    result = conn.connect()
    assert result.success, f"fake docs_server failed to connect: {result.error}"
    try:
        yield conn
    finally:
        conn.disconnect()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def test_connect_to_real_server_succeeds(github_connection):
    assert github_connection.is_connected is True


def test_connect_to_nonexistent_command_fails_fast():
    config = MCPServerConfig(command="this-command-does-not-exist-kenzo-phase6", args=[])
    conn = MCPServerConnection("dead", config, connect_timeout=5, call_timeout=5)
    try:
        t0 = time.perf_counter()
        result = conn.connect()
        elapsed = time.perf_counter() - t0

        assert result.success is False
        assert result.error  # a real, non-empty error string
        assert conn.is_connected is False
        # Empirically ~0.03s for a FileNotFoundError -- generous bound well
        # under the 5s connect_timeout proves this is a fast failure, not a
        # disguised timeout.
        assert elapsed < 2.5, f"nonexistent-command connect took {elapsed}s -- looks like it hung, not failed fast"
    finally:
        conn.disconnect()


# ---------------------------------------------------------------------------
# Discovery: list_tools() against a real server
# ---------------------------------------------------------------------------


def test_list_tools_github_returns_expected_names_and_schema(github_connection):
    tools, error = github_connection.list_tools()
    assert error is None
    names = {t.name for t in tools}
    assert names == {"get_issue", "search_code", "list_open_issues", "create_pull_request"}

    get_issue = _tool_info(tools, "get_issue")
    assert get_issue.input_schema["type"] == "object"
    assert "issue_number" in get_issue.input_schema["properties"]
    assert get_issue.input_schema["properties"]["issue_number"]["type"] == "integer"
    assert "issue_number" in get_issue.input_schema["required"]

    create_pr = _tool_info(tools, "create_pull_request")
    assert set(create_pr.input_schema["required"]) == {"title", "body", "branch"}
    assert set(create_pr.input_schema["properties"]) == {"title", "body", "branch"}

    list_issues = _tool_info(tools, "list_open_issues")
    assert list_issues.input_schema["properties"] == {}
    assert list_issues.input_schema.get("required", []) == []


def test_list_tools_database_returns_expected_names_and_schema(database_connection):
    tools, error = database_connection.list_tools()
    assert error is None
    names = {t.name for t in tools}
    assert names == {"query", "describe_schema", "execute_update", "drop_table", "always_fails"}

    query = _tool_info(tools, "query")
    assert query.input_schema["properties"]["sql"]["type"] == "string"
    assert query.input_schema["required"] == ["sql"]

    drop_table = _tool_info(tools, "drop_table")
    assert drop_table.input_schema["properties"]["table_name"]["type"] == "string"
    assert drop_table.input_schema["required"] == ["table_name"]


# ---------------------------------------------------------------------------
# Schema conversion: MCPClient.discover_tool_specs() vs raw list_tools()
# ---------------------------------------------------------------------------


def test_discover_tool_specs_name_format_and_parameters_match_list_tools(github_connection, database_connection):
    """Builds a real `MCPClient` but rewires its (never-connected) internal
    connections to the already-open `github_connection`/`database_connection`
    fixtures above, instead of spawning two more subprocesses for
    connections that would be functionally identical to ones already open.
    `MCPClient.connections` is a plain public attribute (see
    `kenzo.mcp.client.MCPClient.__init__`), so this only reuses a real,
    already-connected session -- `discover_tool_specs()` below still issues
    real `list_tools()` calls over that real connection, nothing is mocked.
    """
    config = MCPConfig(
        servers={"github": _server_config("github_server.py"), "database": _server_config("database_server.py")},
        allow_write=True,
        allow_destructive=False,
    )
    client = MCPClient(config)
    client.connections["github"] = github_connection
    client.connections["database"] = database_connection

    specs = client.discover_tool_specs()
    specs_by_name = {s.name: s for s in specs}

    assert set(specs_by_name) == {
        "github_get_issue",
        "github_search_code",
        "github_list_open_issues",
        "github_create_pull_request",
        "database_query",
        "database_describe_schema",
        "database_execute_update",
        "database_drop_table",
        "database_always_fails",
    }

    github_tools, _ = github_connection.list_tools()
    database_tools, _ = database_connection.list_tools()
    raw_by_full_name = {f"github_{t.name}": t for t in github_tools}
    raw_by_full_name.update({f"database_{t.name}": t for t in database_tools})

    for full_name, spec in specs_by_name.items():
        raw_tool = raw_by_full_name[full_name]
        expected_params = raw_tool.input_schema or {"type": "object", "properties": {}, "required": []}
        assert spec.parameters == expected_params, f"{full_name}: ToolSpec.parameters diverged from list_tools()"


# ---------------------------------------------------------------------------
# Execution: a real fake tool through the full ToolSpec.func(**kwargs) path
# ---------------------------------------------------------------------------


def test_execute_real_tool_through_toolspec_func_matches_canned_response(github_connection):
    tools, _ = github_connection.list_tools()
    get_issue = _tool_info(tools, "get_issue")
    permission = resolve_tool_permission(get_issue.name, get_issue.description, {})
    assert permission is ToolPermission.READ

    spec = build_tool_spec("github", get_issue, github_connection, permission, allow_write=False, allow_destructive=False)
    assert spec.name == "github_get_issue"

    result = spec.func(issue_number=101)
    assert result.success is True
    assert "Logout button does nothing after OIDC redirect" in result.output


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


def test_read_tool_always_allowed_regardless_of_allow_flags(database_connection):
    tools, _ = database_connection.list_tools()
    query = _tool_info(tools, "query")
    permission = resolve_tool_permission(query.name, query.description, {})
    assert permission is ToolPermission.READ

    locked_down = build_tool_spec("database", query, database_connection, permission, allow_write=False, allow_destructive=False)
    result = locked_down.func(sql="select * from users")
    assert result.success is True
    assert "jdoe" in result.output

    wide_open = build_tool_spec("database", query, database_connection, permission, allow_write=True, allow_destructive=True)
    result2 = wide_open.func(sql="select * from users")
    assert result2.success is True


def test_write_tool_blocked_without_allow_write_and_succeeds_with_it(github_connection):
    tools, _ = github_connection.list_tools()
    create_pr = _tool_info(tools, "create_pull_request")
    permission = resolve_tool_permission(create_pr.name, create_pr.description, {})
    assert permission is ToolPermission.WRITE

    blocked_spec = build_tool_spec("github", create_pr, github_connection, permission, allow_write=False, allow_destructive=False)
    blocked_result = blocked_spec.func(title="Fix logout", body="body", branch="fix/logout")
    assert blocked_result.success is False
    assert blocked_result.error
    assert "allow_write" in blocked_result.error

    allowed_spec = build_tool_spec("github", create_pr, github_connection, permission, allow_write=True, allow_destructive=False)
    allowed_result = allowed_spec.func(title="Fix logout", body="body", branch="fix/logout")
    assert allowed_result.success is True
    assert "pull request" in allowed_result.output.lower()


def test_destructive_tool_blocked_without_allow_destructive_even_with_allow_write(database_connection):
    tools, _ = database_connection.list_tools()
    drop_table = _tool_info(tools, "drop_table")
    permission = resolve_tool_permission(drop_table.name, drop_table.description, {})
    assert permission is ToolPermission.DESTRUCTIVE

    still_blocked = build_tool_spec(
        "database", drop_table, database_connection, permission, allow_write=True, allow_destructive=False
    )
    blocked_result = still_blocked.func(table_name="users")
    assert blocked_result.success is False
    assert blocked_result.error
    assert "allow_destructive" in blocked_result.error

    allowed_spec = build_tool_spec(
        "database", drop_table, database_connection, permission, allow_write=True, allow_destructive=True
    )
    allowed_result = allowed_spec.func(table_name="users")
    assert allowed_result.success is True
    assert "dropped" in allowed_result.output.lower()


def test_tool_permission_overrides_changes_classification_and_gating(database_connection):
    """`query` heuristically classifies as READ (always allowed); an
    override to "write" for this one tool name must change both its
    resolved permission AND its actual execution-time gating."""
    tools, _ = database_connection.list_tools()
    query = _tool_info(tools, "query")

    default_permission = resolve_tool_permission(query.name, query.description, {})
    assert default_permission is ToolPermission.READ

    overridden_permission = resolve_tool_permission(query.name, query.description, {"query": "write"})
    assert overridden_permission is ToolPermission.WRITE

    spec = build_tool_spec(
        "database", query, database_connection, overridden_permission, allow_write=False, allow_destructive=False
    )
    result = spec.func(sql="select * from users")
    assert result.success is False
    assert "allow_write" in result.error

    spec_allowed = build_tool_spec(
        "database", query, database_connection, overridden_permission, allow_write=True, allow_destructive=False
    )
    result_allowed = spec_allowed.func(sql="select * from users")
    assert result_allowed.success is True


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_call_with_missing_required_argument_returns_clean_failure_not_a_crash(github_connection):
    """Real observed behavior (confirmed by hand before writing this test):
    FastMCP validates arguments server-side with pydantic and returns an
    MCP `CallToolResult(isError=True)` -- `MCPServerConnection.call_tool`
    converts that into a clean `ToolResult(success=False, error=...)`; it
    does not raise, hang, or silently succeed with wrong data."""
    result = github_connection.call_tool("get_issue", {})
    assert result.success is False
    assert result.error
    assert "issue_number" in result.error


def test_call_with_wrong_argument_type_returns_clean_failure_not_a_crash(github_connection):
    result = github_connection.call_tool("get_issue", {"issue_number": "not-a-number"})
    assert result.success is False
    assert result.error
    assert "issue_number" in result.error


def test_discover_tool_specs_returns_empty_list_when_all_servers_dead():
    config = MCPConfig(
        servers={
            "dead_one": MCPServerConfig(command="nonexistent-cmd-alpha-kenzo"),
            "dead_two": MCPServerConfig(command="nonexistent-cmd-beta-kenzo"),
        }
    )
    client = MCPClient(config)
    try:
        connect_results = client.connect_all()
        assert all(not r.success for r in connect_results.values())

        specs = client.discover_tool_specs()
        assert specs == []
    finally:
        client.disconnect_all()


def test_discover_tool_specs_falls_back_to_working_server_when_one_is_dead(database_connection):
    """Graceful degradation (item 8): one dead server contributes zero
    tools but does not prevent a working server's tools from being
    discovered in the same `discover_tool_specs()` call."""
    config = MCPConfig(
        servers={"database": _server_config("database_server.py"), "dead": MCPServerConfig(command="nonexistent-cmd-gamma-kenzo")}
    )
    client = MCPClient(config)
    try:
        dead_connect_result = client.connections["dead"].connect()
        assert dead_connect_result.success is False
        client.connections["database"] = database_connection  # reuse the already-open real connection

        specs = client.discover_tool_specs()
        names = {s.name for s in specs}
        assert names == {
            "database_query",
            "database_describe_schema",
            "database_execute_update",
            "database_drop_table",
            "database_always_fails",
        }
    finally:
        # Only tear down the "dead" connection here -- "database" is owned
        # by the module-scoped `database_connection` fixture and must stay
        # alive for any later test in this module that reuses it.
        client.connections["dead"].disconnect()
