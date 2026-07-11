"""Fake stdio MCP server shaped like a real GitHub MCP integration, for Phase
6 component/error-recovery tests. Spawned as a real subprocess by
`kenzo.mcp.connection.MCPServerConnection` -- never imported directly by the
tests that use it.

Tool-name verbs are chosen deliberately so `kenzo.mcp.permissions
.classify_tool_permission`'s heuristic classifies each one correctly without
needing a `tool_permission_overrides` entry:

- get_issue        -> READ  ("get")
- search_code      -> READ  ("search")
- list_open_issues -> READ  ("list")
- create_pull_request -> WRITE ("create")
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp_app = FastMCP("github-fake")


@mcp_app.tool()
def get_issue(issue_number: int) -> str:
    """Fetch a single GitHub issue by number."""
    return (
        f"Issue #{issue_number}: Logout button does nothing after OIDC redirect\n\n"
        "Description: After a user completes SSO login via the OIDC provider and "
        "then clicks 'Logout', the button click is registered (network tab shows "
        "no outgoing request at all) but the user session remains active and the "
        "page just sits on the dashboard. Expected: clicking Logout should call "
        "/api/auth/logout and redirect the user to the identity provider's "
        "post-logout page. Affects all users who logged in via OIDC SSO; users on "
        "local username/password auth are not affected."
    )


@mcp_app.tool()
def search_code(query: str) -> str:
    """Search the repository's code for a query string."""
    return (
        f"3 results for '{query}':\n"
        "src/main/java/com/example/auth/LogoutController.java:14: public String logout()\n"
        "src/main/java/com/example/auth/AuthService.java:22: public void terminateSession()\n"
        "src/main/resources/static/js/dashboard.js:87: function onLogoutClick()"
    )


@mcp_app.tool()
def list_open_issues() -> str:
    """List all currently open issues."""
    return (
        "#101 Logout button does nothing after OIDC redirect\n"
        "#98 Session timeout message shows wrong duration\n"
        "#87 Dashboard widget flickers on refresh"
    )


@mcp_app.tool()
def create_pull_request(title: str, body: str, branch: str) -> str:
    """Open a new pull request."""
    return f"Created pull request #245 on branch '{branch}': {title}\nhttps://github.com/example/repo/pull/245"


if __name__ == "__main__":
    mcp_app.run(transport="stdio")
