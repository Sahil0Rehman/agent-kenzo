"""Phase 7 "VS Code Extension Integration": `kenzo server` runs a local
FastAPI + WebSocket server exposing `AgentRuntime` to an IDE client (the
`vscode-extension/` package, or any other WebSocket client speaking the same
protocol -- see `kenzo.server.protocol`) with streaming events, cancellation,
reconnect, and propose-then-approve file edits instead of silent overwrites.

Additive to every prior phase: this package is the ONLY thing that knows
about WebSockets/FastAPI. `AgentRuntime`/`AgentController` gained a small,
optional, backward-compatible surface (`ide_mode`, `on_event` mutability,
`cancel_event`, edit-proposal handling) that CLI mode never touches -- `kenzo`
(the interactive CLI session) and `kenzo server` are two different front
ends over the exact same agent core, not two different agents.
"""
