"""MCP tool permission classification: extends Kenzo's existing safety model
(`kenzo.config.settings.PermissionsConfig`'s static booleans for native
tools) to dynamically-discovered MCP tools, whose names aren't known ahead of
time and so can't be hand-classified the way native tools are.

Three levels, in ascending order of what a human should want to gate:

- READ: side-effect-free lookups (fetch an issue, run a SELECT, search docs).
  Always allowed once a server is connected -- reading data is not something
  Kenzo's existing safety model gates for native tools either (read_file has
  no permission check).
- WRITE: mutates external state, but non-destructively/reversibly (create a
  PR, post a comment, run an UPDATE). Gated by `MCPConfig.allow_write`.
- DESTRUCTIVE: mutates external state in a way that's hard or impossible to
  reverse (drop a table, force-delete a branch, close an issue permanently).
  Gated by `MCPConfig.allow_destructive`, which defaults to `False` -- fails
  closed, exactly like `PermissionsConfig.allow_shell_exec`/`allow_git_write`
  already do for native tools.

Classification is a heuristic over the tool's name (and, as a weaker signal,
its description) -- MCP servers don't self-report a permission level, so this
is the best available signal short of an explicit per-tool override (see
`MCPServerConfig.tool_permission_overrides`, which always wins when present
and is the correct fix for any tool this heuristic gets wrong, rather than
trying to make the heuristic itself smarter for one specific tool).
"""

from __future__ import annotations

from enum import Enum


class ToolPermission(str, Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


# Order matters: DESTRUCTIVE patterns are checked first, so a name matching
# both a destructive and a write/read pattern (e.g. "delete_draft_pr" -- both
# "delete" and "pr") is classified at the MORE cautious level. Within a tier,
# longer/more specific patterns are not prioritized over shorter ones -- any
# match at the higher tier wins outright, which is the conservative (fail
# toward more caution) choice.
_DESTRUCTIVE_PATTERNS = (
    "drop",
    "truncate",
    "delete",
    "destroy",
    "purge",
    "force_push",
    "force-push",
    "wipe",
    "remove_all",
    "revoke",
    "reset_hard",
)

_WRITE_PATTERNS = (
    "create",
    "update",
    "write",
    "insert",
    "add",
    "set",
    "modify",
    "edit",
    "merge",
    "push",
    "close",
    "approve",
    "comment",
    "assign",
    "label",
    "publish",
    "deploy",
    "execute",
    "run",
    "send",
)

_READ_PATTERNS = (
    "read",
    "get",
    "list",
    "search",
    "query",
    "fetch",
    "describe",
    "inspect",
    "check",
    "show",
    "find",
    "view",
    "lookup",
)


def classify_tool_permission(tool_name: str, description: str = "") -> ToolPermission:
    """Heuristic classification of one MCP tool's permission level.

    Checks the tool name first (the stronger, more deliberate signal --
    server authors generally name tools after the verb they perform), then
    falls back to the description if the name alone doesn't match anything.
    An MCP tool matching nothing recognizable defaults to WRITE, not READ --
    an unrecognized name is more likely a mutating action a server author
    just phrased unusually (e.g. "sync_repository") than a pure lookup, and
    the safe direction to default an ambiguous tool is toward requiring
    `allow_write` rather than silently allowing it unconditionally like a
    confirmed READ tool would be.
    """
    for text in (tool_name, description):
        normalized = text.lower().replace("-", "_")
        if any(pattern in normalized for pattern in _DESTRUCTIVE_PATTERNS):
            return ToolPermission.DESTRUCTIVE
        if any(pattern in normalized for pattern in _WRITE_PATTERNS):
            return ToolPermission.WRITE
        if any(pattern in normalized for pattern in _READ_PATTERNS):
            return ToolPermission.READ

    return ToolPermission.WRITE


def resolve_tool_permission(
    tool_name: str,
    description: str,
    overrides: dict[str, str],
) -> ToolPermission:
    """`overrides` (from `MCPServerConfig.tool_permission_overrides`) always
    wins when present for this exact tool name; otherwise falls back to
    `classify_tool_permission`. An invalid override value (a typo, e.g.
    "raed") degrades to the heuristic result rather than raising -- a
    misconfigured override should never crash tool discovery.
    """
    override = overrides.get(tool_name)
    if override:
        try:
            return ToolPermission(override.lower())
        except ValueError:
            pass
    return classify_tool_permission(tool_name, description)


def is_call_allowed(permission: ToolPermission, allow_write: bool, allow_destructive: bool) -> bool:
    """The actual execution-time gate -- see module docstring for the
    rationale behind each level's default."""
    if permission is ToolPermission.READ:
        return True
    if permission is ToolPermission.WRITE:
        return allow_write
    return allow_destructive
