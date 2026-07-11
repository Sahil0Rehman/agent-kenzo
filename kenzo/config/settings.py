"""Configuration schema and load/save logic for ~/.kenzo/config.yaml.

This is the other foundational contract module (alongside
`kenzo.core.schemas`): the model router, tool permission checks, and every
CLI command all import `KenzoConfig` / `load_config` from here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from kenzo.config.defaults import DEFAULT_CONFIG_YAML
from kenzo.core.schemas import AgentBudgetConfig

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProviderConfig(BaseModel):
    """Connection details for one model provider."""

    api_key: str = ""
    base_url: Optional[str] = None
    default_model: str = ""


class RoutingConfig(BaseModel):
    """Maps a task type (see `kenzo.core.schemas.TaskType`) to a provider name."""

    coding: str = "nemotron"
    debugging: str = "nemotron"
    vision: str = "gemini"
    review: str = "nemotron"
    planning: str = "nemotron"
    research: str = "nemotron"
    general: str = "nemotron"

    def for_task(self, task_type: str) -> str:
        return getattr(self, task_type, self.general)


class PermissionsConfig(BaseModel):
    """Guardrails enforced by the tools layer, independent of model behavior."""

    allow_shell_exec: bool = False
    allow_write: bool = True
    allow_git_write: bool = False
    command_allowlist: list[str] = Field(
        default_factory=lambda: [
            "git",
            "python",
            "pytest",
            "npm",
            "pip",
            "node",
            "ls",
            "dir",
            # Java build tools, so Tester can actually validate a Java/Spring
            # repo out of the box instead of only ever having Python-flavored
            # commands available (see kenzo.core.project.ProjectAnalyzer).
            "mvn",
            "mvnw",
            "gradle",
            "gradlew",
            "java",
        ]
    )


# Directories skipped by every repo-scanning tool (search_files, repository_tree,
# index_repository, the Java analyzer). A single shared default so "ignore this
# folder" only has to be configured once; individual tools accept an override
# but fall back to this list.
DEFAULT_IGNORED_DIRS: list[str] = [
    ".git",
    ".kenzo",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    # Enterprise/Java-specific build and tooling output.
    "target",
    ".gradle",
    "out",
    "coverage",
]


class IndexingConfig(BaseModel):
    """Configuration for repository scanning (indexing, search, tree, symbols).

    `max_indexed_files` was raised from 2000 to 8000 after a real enterprise
    Java/Spring repo (8037 candidate files) showed the old default was too
    low to reasonably cover a large monorepo-style service -- combined with
    `SymbolIndex.build`'s round-robin-by-language selection (see its
    docstring), a still-truncated repo now at least gets a proportional
    slice of every language instead of one language potentially getting
    zero representation.
    """

    ignored_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_IGNORED_DIRS))
    max_indexed_files: int = 8000


class EmbeddingConfig(BaseModel):
    """Configuration for the Phase 5 semantic embedding layer
    (`kenzo.memory.vector.embeddings.EmbeddingRouter`).

    `provider` is deliberately NOT restricted to a fixed enum -- "local" is
    the one special-cased value (routes to the dependency-free, offline
    `HashingEmbeddingProvider`); anything else is treated as a litellm
    provider prefix (e.g. "openai", "gemini", "ollama") and routed through
    `LiteLLMEmbeddingProvider`, exactly like `RoutingConfig`/`ProviderConfig`
    already do for chat completions -- this is what keeps the provider
    un-hardcoded and covers both "local" (e.g. a local Ollama server) and
    "OpenAI-compatible" embeddings through the same mechanism.
    """

    provider: str = "local"
    model: str = "hashing-v1"
    api_key: str = ""
    api_base: Optional[str] = None
    dimensions: int = 256


class VectorStoreConfig(BaseModel):
    """Configuration for the Phase 5 vector store
    (`kenzo.memory.vector.store.VectorStore`).

    `url` defaults to Qdrant's embedded ":memory:" mode -- no external server
    required, verified to work fully offline -- so semantic search works out
    of the box. Point at a real Qdrant server (e.g. "http://localhost:6333")
    for a persistent, cross-session store instead.
    """

    backend: str = "qdrant"
    url: str = ":memory:"
    collection_prefix: str = "kenzo"


class MCPServerConfig(BaseModel):
    """One configured MCP (Model Context Protocol) server.

    Exactly one transport should be configured: `command` (stdio -- Kenzo
    spawns the server as a subprocess) or `url` (HTTP -- Kenzo connects to an
    already-running server). Both being set is treated as an error at
    connect time, not here (this is a plain data model with no validation
    logic beyond field types, matching every other *Config class in this
    module).
    """

    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    url: Optional[str] = None
    transport: str = "streamable_http"  # "streamable_http" | "sse" -- only relevant when `url` is set

    # Per-server permission overrides -- see kenzo.mcp.permissions. Keys are
    # exact tool names as reported by the server's own list_tools(); values
    # are "read" | "write" | "destructive". Only needed when the heuristic
    # classifier (kenzo.mcp.permissions.classify_tool_permission) gets a
    # specific tool wrong.
    tool_permission_overrides: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    """Configuration for the Phase 6 MCP client layer
    (`kenzo.mcp.client.MCPClient`).

    `allow_write`/`allow_destructive` gate MCP tool EXECUTION by permission
    level (see `kenzo.mcp.permissions.ToolPermission`) -- independent of, and
    in addition to, `PermissionsConfig.allow_write` (which only governs
    Kenzo's own native filesystem/git tools). READ-level MCP tools are always
    allowed once a server is connected; WRITE requires `allow_write=true`;
    DESTRUCTIVE requires `allow_destructive=true` -- both fail closed by
    default, exactly like every other safety gate in this codebase.
    """

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    allow_write: bool = True
    allow_destructive: bool = False
    connect_timeout_seconds: float = 15.0
    call_timeout_seconds: float = 30.0


class WorkersConfig(BaseModel):
    """Configuration for the Phase 9 parallel worker system
    (`kenzo.workers.manager.WorkerManager`).

    `max_parallel` bounds a SINGLE `DELEGATE` action's worker count (item 8's
    "avoid infinite worker spawning") -- the outer `AgentBudgetConfig`
    (`max_iterations`/`max_tool_calls`/`max_runtime_minutes`) remains the
    backstop against repeated DELEGATE calls across a task's lifetime, the
    same budget every other action already shares.
    """

    enabled: bool = True
    max_parallel: int = 5
    timeout_seconds: int = 120


class KenzoConfig(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    agent: AgentBudgetConfig = Field(default_factory=AgentBudgetConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    default_repo_path: Optional[str] = None

    def provider(self, name: str) -> ProviderConfig:
        if name not in self.providers:
            raise KeyError(
                f"No provider named '{name}' configured. "
                f"Known providers: {sorted(self.providers)}. Run `kenzo config show`."
            )
        return self.providers[name]


def _resolve_env_vars(value):
    """Recursively replace ${VAR_NAME} placeholders with the environment
    variable's value (or "" if unset) inside a parsed YAML structure.

    This operates on the *parsed* dict/list/str tree rather than the raw YAML
    text. Substituting on raw text before parsing is unsafe: an unquoted
    placeholder (e.g. ``api_key: ${GEMINI_API_KEY}``, which is exactly what
    `yaml.safe_dump` produces since `$`, `{`, `}` don't require quoting) that
    resolves to "" would leave a bare `api_key: ` line, which YAML parses as
    `None` rather than an empty string. Resolving after parsing guarantees we
    only ever substitute within an already-typed string value.
    """

    def _sub(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {key: _resolve_env_vars(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def kenzo_home() -> Path:
    override = os.environ.get("KENZO_HOME")
    return Path(override).expanduser() if override else Path.home() / ".kenzo"


def config_path() -> Path:
    return kenzo_home() / "config.yaml"


def ensure_config_exists() -> Path:
    """Create ~/.kenzo/config.yaml with defaults if it doesn't already exist."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
    return path


def load_config() -> KenzoConfig:
    """Load config from disk, creating a default file on first run.

    Environment-variable placeholders (${VAR}) in string values are resolved
    at load time so raw secrets never need to live in the YAML file itself.
    """
    path = ensure_config_exists()
    raw_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text) or {}
    data = _resolve_env_vars(data)
    return KenzoConfig.model_validate(data)


def save_config(config: KenzoConfig) -> Path:
    """Persist a config back to ~/.kenzo/config.yaml.

    Note: this writes resolved values, not ${VAR} placeholders. Prefer
    `set_value` for targeted edits (e.g. `kenzo config set`) so unrelated
    placeholders in the file aren't clobbered with resolved secrets.
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def set_value(dotted_key: str, value: str) -> KenzoConfig:
    """Set a single dotted key (e.g. 'routing.coding' or 'providers.gemini.api_key')
    directly in the on-disk YAML, preserving everything else untouched, then
    return the reloaded config.
    """
    path = ensure_config_exists()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    parts = dotted_key.split(".")
    cursor = data
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = yaml.safe_load(value) if value.lower() in ("true", "false", "null") else value

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_config()
