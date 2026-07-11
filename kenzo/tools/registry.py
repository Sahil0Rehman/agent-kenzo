"""Factory that assembles the default ToolRegistry from every tool module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from kenzo.config.settings import IndexingConfig, PermissionsConfig
from kenzo.tools import code_intelligence, cross_repo, filesystem, semantic_search, symbols, terminal, tree
from kenzo.tools import git as git_tools
from kenzo.tools.base import ToolRegistry

if TYPE_CHECKING:
    from kenzo.core.indexing_service import IndexingService
    from kenzo.core.workspace import Workspace
    from kenzo.mcp.client import MCPClient


def build_default_registry(
    repo_root: str,
    permissions: PermissionsConfig,
    indexing: Optional[IndexingConfig] = None,
    indexing_service: Optional["IndexingService"] = None,
    mcp_client: Optional["MCPClient"] = None,
    workspace: Optional["Workspace"] = None,
) -> ToolRegistry:
    """Build a ToolRegistry populated with the filesystem, git, terminal,
    repository-tree, symbol, code-intelligence, (Phase 5) semantic-search,
    (Phase 6) MCP, and (Phase 11) cross-repo workspace tools, all bound to
    `repo_root`/`permissions`/`indexing`.

    `indexing_service`/`mcp_client`/`workspace` are all optional -- only
    `AgentRuntime` (which owns each of them) passes them; when omitted,
    `semantic_search` registers no tool, no MCP tools are added, and no
    cross-repo tools are added, rather than a broken partial registration.
    `mcp_client.discover_tool_specs()` is called here, AFTER `mcp_client
    .connect_all()` is expected to have already run (see `AgentRuntime
    .__init__`) -- this function itself never connects/disconnects any MCP
    server, it only reads whatever the client has already discovered.
    """
    indexing = indexing or IndexingConfig()
    registry = ToolRegistry()
    registry.register_many(filesystem.get_tool_specs(repo_root, permissions, indexing))
    registry.register_many(git_tools.get_tool_specs(repo_root, permissions))
    registry.register_many(terminal.get_tool_specs(repo_root, permissions))
    registry.register_many(tree.get_tool_specs(repo_root, permissions, indexing))
    registry.register_many(symbols.get_tool_specs(repo_root, permissions))
    registry.register_many(code_intelligence.get_tool_specs(repo_root, permissions))
    registry.register_many(semantic_search.get_tool_specs(repo_root, permissions, indexing_service))
    if mcp_client is not None:
        registry.register_many(mcp_client.discover_tool_specs())
    if workspace is not None:
        registry.register_many(cross_repo.get_tool_specs(workspace, permissions, indexing))
    return registry
