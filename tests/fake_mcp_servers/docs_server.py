"""Fake stdio MCP server shaped like a real documentation-search MCP
integration, for Phase 6 component/error-recovery tests. Spawned as a real
subprocess by `kenzo.mcp.connection.MCPServerConnection` -- never imported
directly by the tests that use it.

- search_docs -> READ ("search")
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp_app = FastMCP("docs-fake")


@mcp_app.tool()
def search_docs(query: str) -> str:
    """Search the documentation for a query string."""
    return (
        f"Top result for '{query}': 'Configuring Logout in Spring Security'\n\n"
        "To customize post-logout behavior, register a LogoutSuccessHandler bean:\n\n"
        "    @Bean\n"
        "    public LogoutSuccessHandler logoutSuccessHandler() {\n"
        "        return (request, response, authentication) -> {\n"
        "            response.sendRedirect(oidcProvider.getPostLogoutRedirectUri());\n"
        "        };\n"
        "    }\n\n"
        "Then wire it into HttpSecurity via "
        ".logout(logout -> logout.logoutSuccessHandler(logoutSuccessHandler())). "
        "A common cause of a logout button doing nothing is the frontend never "
        "issuing the POST to /logout at all -- check the button's click handler "
        "before assuming the backend handler is misconfigured."
    )


if __name__ == "__main__":
    mcp_app.run(transport="stdio")
