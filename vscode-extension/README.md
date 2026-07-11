# Kenzo for VS Code

VS Code client for `kenzo server` (Phase 7) -- a Cursor/Claude-Code-style chat and
inline-editing experience backed by the local Kenzo agent.

## Requirements

Start the Kenzo server first:

```
kenzo server --port 8756
```

The extension connects to `ws://127.0.0.1:8756/ws` by default (configurable via
`kenzo.serverHost` / `kenzo.serverPort`).

## Features

- **Kenzo Chat** (activity bar) -- chat with the agent, accept/reject proposed edits inline.
- **Kenzo Changes** -- files touched by the current/last task; click an entry for a real diff view.
- **Kenzo Memory** -- read-only summary of the last task's result.
- Commands: `Kenzo: Chat`, `Kenzo: Explain File`, `Kenzo: Fix Error`, `Kenzo: Review Selection`, `Kenzo: Show Terminal`.
- IDE context (active file, selection, open tabs, diagnostics, git status) is streamed to the
  agent automatically and on demand.
- Edits are always shown as a real diff and applied only via `vscode.WorkspaceEdit` after you
  accept them -- nothing is ever written to disk directly by the extension.

## Development

```
npm install
npm run compile
npm test
```
