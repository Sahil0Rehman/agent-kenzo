"""Phase 9 "Parallel Agent Workers": temporary, isolated, read-only sub-agents
`kenzo.core.controller.AgentController` can delegate independent investigation
work to via the new `DELEGATE` action, instead of doing every search/trace/
lookup itself, serially, in its own single reasoning thread.

    Main agent: "I need to understand authentication"
        -> WorkerTask(CODE_RESEARCH, "trace the login flow")
        -> WorkerTask(TEST, "find tests covering auth")
        -> WorkerTask(GIT, "check recent auth-related commits")
    WorkerManager.run_parallel(tasks)  -- real OS-thread concurrency
        -> [WorkerResult, WorkerResult, WorkerResult]
    WorkerResultMerger.merge(results)  -- one unified investigation summary
        -> fed back into the SAME task loop as any other action's findings

Workers never edit files or run destructive MCP tools (see
`kenzo.workers.worker_agent.WORKER_TOOL_NAMES` -- a curated read-only
allowlist enforced via `ToolRegistry.subset()`, the same access-control
mechanism every other role-scoped action already relies on). They are a new
INPUT to the main agent's own judgment, not a second tier of agents that can
themselves act on the repository -- see the module docstring in
`kenzo.core.controller` for how `DELEGATE` fits into the existing action loop.
"""
