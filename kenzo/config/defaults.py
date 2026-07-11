"""Default values baked into a freshly generated ~/.kenzo/config.yaml."""

from __future__ import annotations

DEFAULT_CONFIG_YAML = """\
# Agent Kenzo configuration.
# Environment variables referenced as ${VAR_NAME} are resolved at load time.

providers:
  nemotron:
    api_key: "${NVIDIA_API_KEY}"
    base_url: "https://integrate.api.nvidia.com/v1"
    default_model: "openai/nvidia/llama-3.1-nemotron-70b-instruct"
  gemini:
    api_key: "${GEMINI_API_KEY}"
    base_url: null
    default_model: "gemini/gemini-1.5-pro"
  openai_compatible:
    api_key: "${OPENAI_API_KEY}"
    base_url: null
    default_model: "gpt-4o-mini"

routing:
  coding: nemotron
  debugging: nemotron
  vision: gemini
  review: nemotron
  planning: nemotron
  research: nemotron
  general: nemotron

permissions:
  allow_shell_exec: false
  allow_write: true
  allow_git_write: false
  command_allowlist:
    - git
    - python
    - pytest
    - npm
    - pip
    - node
    - ls
    - dir
    - mvn
    - mvnw
    - gradle
    - gradlew
    - java

indexing:
  ignored_dirs:
    - .git
    - .kenzo
    - __pycache__
    - node_modules
    - venv
    - .venv
    - dist
    - build
    - .mypy_cache
    - .pytest_cache
    - .idea
    - .vscode
    - target
    - .gradle
    - out
    - coverage
  max_indexed_files: 2000

agent:
  max_iterations: 50
  max_tool_calls: 100
  max_runtime_minutes: 30
  small_task_iterations: 10

# Phase 9 parallel worker system (kenzo.workers) -- the DELEGATE action
# spawns up to max_parallel temporary, read-only sub-agents at once, each
# bounded by timeout_seconds; set enabled: false to disable delegation
# entirely (the main agent then always investigates everything itself).
workers:
  enabled: true
  max_parallel: 5
  timeout_seconds: 120

# MCP (Model Context Protocol) servers -- external tools (issue trackers,
# databases, documentation search, etc.) Kenzo can call alongside its native
# tools. Empty by default: nothing connects, nothing changes, until you add
# a server here. Exactly one of `command` (stdio -- Kenzo spawns it as a
# subprocess) or `url` (HTTP -- an already-running server) per entry, e.g.:
#
# mcp:
#   servers:
#     github:
#       command: github-mcp-server
#       args: ["--stdio"]
#     database:
#       url: "http://localhost:9000"
#   allow_write: true          # gate WRITE-level MCP tools (create/update/etc.)
#   allow_destructive: false   # gate DESTRUCTIVE-level MCP tools (drop/delete/etc.)
mcp:
  servers: {}
  allow_write: true
  allow_destructive: false

default_repo_path: null
"""

DEFAULT_ARCHITECTURE_MD = """\
# Repository Architecture

_This file is maintained by Agent Kenzo. Run `kenzo index` to (re)generate the
repository index that backs this summary, and edit this file freely — Kenzo
reads it as durable context but never overwrites your edits wholesale._

## Overview

(No architecture notes yet. Run `kenzo index` inside your project.)
"""

DEFAULT_MEMORY_MD = """\
# Repository Memory

_Running log of decisions, conventions, and context Agent Kenzo has learned
about this repository. Newest entries are appended at the bottom._
"""

DEFAULT_RULES_MD = """\
# Repository Rules

_Hard constraints Agent Kenzo must follow in this repository. Edit this file
to add project-specific rules (coding style, forbidden paths, required
review steps, etc.) — every agent node reads this file as system context._

- Never force-push or rewrite shared git history.
- Never delete files outside the repository root.
- Always run the test suite before proposing a change as complete.
"""
