# Agent Kenzo

Kenzo is a local, repository-aware AI coding agent that runs entirely on
your own machine. Point it at a project and it can search your codebase,
understand how the code is structured, make real edits, run your tests,
debug from a screenshot, delegate research to parallel sub-agents, and even
propose and safely verify improvements to its own source code.

Nothing about your code leaves your machine except the actual text sent to
whichever AI model provider you configure (you bring your own API key).

## Quick Start (2 minutes)

You do **not** need to know Python, Git, or anything technical to do this.

1. **Download this project.** Click the green "Code" button on this GitHub
   page → "Download ZIP" → unzip it somewhere. (Or, if you have Git:
   `git clone <this repo's URL>`.)
2. **Open a terminal in that folder.**
   - **Windows:** open the unzipped folder in File Explorer, click the
     address bar, type `powershell`, press Enter.
   - **macOS:** open the folder in Finder, right-click → "Open Terminal at
     Folder" (or open Terminal and `cd` into the folder).
3. **Run the setup script:**
   - **Windows (PowerShell):**
     ```powershell
     .\setup.ps1
     ```
     If you see a "running scripts is disabled" error, run this once first,
     then try again:
     ```powershell
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     ```
   - **macOS / Linux:**
     ```bash
     ./setup.sh
     ```
     If you see a "permission denied" error, run this once first:
     ```bash
     chmod +x setup.sh
     ```

The script checks you have Python 3.12+, sets up an isolated environment,
installs Kenzo, and walks you through adding an API key. That's it — see
["Your first command"](#your-first-command) below.

No Python installed? Get it free from
[python.org/downloads](https://www.python.org/downloads/) — on Windows,
**make sure to check "Add python.exe to PATH"** during install.

## Your first command

Once setup finishes, go into any code project you want Kenzo to help with:

```bash
cd /path/to/your/project

# one-time: let Kenzo learn the shape of this repo
kenzo init
kenzo index

# start an interactive session
kenzo
```

Then just type what you want, e.g. `fix the bug where login fails after a
password reset`, and Kenzo will investigate, edit files, and run your tests
— showing you every step.

> If you skipped activating the environment, use the full path instead of
> `kenzo`: `..\path\to\Ken\.venv\Scripts\kenzo.exe` (Windows) or
> `../path/to/Ken/.venv/bin/kenzo` (macOS/Linux).

## What Kenzo can do

| Capability | Command | What it does |
|---|---|---|
| Autonomous coding agent | `kenzo` | Dynamically plans, searches, reads, edits, tests, and reviews — revisiting steps as needed until the goal is verified, not a fixed script. |
| Lightweight Q&A | `kenzo chat` | Repo-aware conversation that never edits files. |
| Screenshot debugging | `kenzo image screenshot.png` | Reads an error screen, broken UI, or log output from an image and drives a real fix from it. |
| Parallel investigation | *(automatic)* | For big/unfamiliar/multi-cause problems, Kenzo can delegate to several read-only sub-agents (code research, bug analysis, tests, docs, git history) running **concurrently**, then merges their findings. |
| Self-improvement | `kenzo improve` | Analyzes a codebase (including Kenzo's own) for duplicate logic, dependency cycles, oversized files, and test gaps, then safely proposes and applies one fix — protected by an automatic test-regression guard that rolls back any change that breaks anything. |
| Performance tracking | `kenzo benchmark` | Measures indexing speed and search quality, tracked over time. |
| VS Code integration | see below | Chat, live diffs, screenshot attach, and a changes/memory panel inside the editor. |
| Safety net | `kenzo rollback` | Undo the most recent set of edits Kenzo made, any time. |

Under the hood this is backed by real AST-level code intelligence (not just
text search) across Java, Python, JavaScript, and TypeScript; a semantic
(embedding-based) search index; long-term project memory that persists
across sessions; and optional MCP (Model Context Protocol) tool integration
for issue trackers, databases, and documentation search.

## Configuration

On first run Kenzo creates `~/.kenzo/config.yaml` (on Windows:
`%USERPROFILE%\.kenzo\config.yaml`) with three providers already wired up —
you just need to supply a key for at least one:

| Provider | Used for | Environment variable |
|---|---|---|
| `nemotron` (via NVIDIA) | coding/debugging/planning/research/review (default) | `NVIDIA_API_KEY` |
| `gemini` (Google) | vision (screenshot debugging) | `GEMINI_API_KEY` |
| `openai_compatible` | anything else you point it at | `OPENAI_API_KEY` |

Set a key either by exporting the environment variable before running
`kenzo`, or directly:

```bash
kenzo config set providers.nemotron.api_key YOUR_KEY_HERE
kenzo config show     # view resolved config (keys are masked)
kenzo config path      # show where the config file lives
```

Which provider handles which kind of work, tool permissions (shell
execution, file writes, git writes), and every other setting are all
editable in that same YAML file — see `kenzo/config/defaults.py` for the
full template with comments.

## VS Code Extension

The extension (`vscode-extension/`) adds a Kenzo chat panel, live diff
previews for proposed edits, a changes/memory sidebar, and a screenshot
attach button, all talking to a local `kenzo server` over a WebSocket.

```bash
cd vscode-extension
npm install
npm run compile
```

Then open the `vscode-extension` folder in VS Code and press **F5** — this
launches a second "Extension Development Host" VS Code window with Kenzo
active. (This is the standard, no-install way to run an unpublished
extension. If you'd rather have it permanently installed, package it with
`npx @vscode/vsce package` and install the resulting `.vsix` via VS Code's
Extensions view → "Install from VSIX...".)

Start the backend it talks to with:

```bash
kenzo server
```

## Safety

- **Checkpoint + rollback** — before Kenzo's first edit in any task, it
  snapshots the repository (via `git stash create` in a git repo, or a
  file-level snapshot otherwise). `kenzo rollback` restores it, any time.
- **Tool permissions** — shell execution is off by default; when enabled,
  commands are restricted to an explicit allowlist and shell
  metacharacters/chaining are rejected. File writes are sandboxed to the
  repository root (path traversal is rejected).
- **Role-scoped tools** — each step of the agent loop (searching, reading,
  editing, testing, reviewing) only has access to the tools that step
  actually needs; a search step can never write a file, for example, even
  if a model hallucinated the attempt.
- **Self-improvement is extra-guarded** — when `kenzo improve` targets
  Kenzo's own source, it additionally refuses to ever touch its own
  `.kenzo/` operational data or anything secret/credential-looking, and
  every change is verified by running the real test suite before AND after
  — any drop in passing tests triggers an automatic rollback before the
  command even returns.

## Full command reference

```bash
kenzo init                 # one-time setup for a project
kenzo index                # build/refresh the code-intelligence index
kenzo                      # interactive autonomous agent session
kenzo chat                 # repo-aware Q&A, never edits files
kenzo image <path> [note]  # debug from a screenshot
kenzo rollback             # undo the last checkpointed set of edits
kenzo server                # start the local WebSocket server (for the VS Code extension)
kenzo improve --analyze    # analyze a repo, print an improvement plan, make no edits
kenzo improve               # analyze, then safely apply and verify one improvement
kenzo improve history       # show past kenzo improve runs
kenzo benchmark              # measure indexing speed / search quality, tracked over time
kenzo benchmark history      # show past benchmark runs
kenzo config show|set|path  # view/edit configuration
```

## Architecture

```
kenzo/
  cli/            Typer commands + Rich terminal UI
  core/           the autonomous agent loop (AgentController), the older
                   fixed pipeline, task/checkpoint/context management
  agents/         the fixed-pipeline agents (planner/researcher/coder/
                   tester/reviewer) and their shared base
  models/         LiteLLM-based model router, provider resolution, vision
  vision/         screenshot loading + structured vision analysis
  workers/        parallel, read-only, isolated sub-agents + their manager
  improve/        self-analysis, regression guard, safe self-modification
  intelligence/   AST-derived symbol index, dependency graph, reference graph
  memory/         short/long-term project memory, semantic (vector) search
  mcp/            Model Context Protocol client + permission model
  tools/          filesystem, git, terminal, code-intelligence tool impls
  server/         the WebSocket server the VS Code extension talks to
  config/         ~/.kenzo/config.yaml schema, defaults, load/save

vscode-extension/  the VS Code client (TypeScript)
tests/             pytest suite (offline — every model call is mocked)
scripts/           standalone benchmark scripts
```

## Development

```bash
pip install -e ".[dev]"
pytest -q                              # Python test suite
cd vscode-extension && npm test        # TypeScript test suite
```

Both suites run fully offline: every AI model call is mocked, so no API
keys or network access are required to run the tests.

## Requirements

- Python 3.12 or newer
- Node.js (any LTS version) — only needed for the VS Code extension
- An API key for at least one supported provider to actually run tasks
  (indexing/analysis works without one)
