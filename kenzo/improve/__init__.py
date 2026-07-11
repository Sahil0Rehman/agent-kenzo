"""Phase 10 "Self Improvement System": Kenzo analyzing and safely modifying
its own (or, generically, any) codebase.

    kenzo improve --analyze   -> self_repo + analysis + 3 workers -> a plan, no edits
    kenzo improve             -> the above, then ONE real AgentController.run()
                                  through the SAME EDIT/TEST/REVIEW loop every
                                  other task uses, wrapped in an extra
                                  regression guard (tests before/after,
                                  automatic rollback on failure) and, when the
                                  target repo IS Kenzo's own source, an extra
                                  tool guard that refuses to touch `.kenzo/`
                                  operational data or anything secret-looking.
    kenzo improve history     -> read back `.kenzo/improvements/*.json`

Deliberately NOT a new agent loop: `SelfImprovementAgent` (self_improvement
_agent.py) is a thin orchestrator around the EXISTING `AgentRuntime`/
`AgentController`/`WorkerManager` (Phases 3/7/9) plus four new, narrowly-
scoped pieces:

- `self_repo.py` -- self-repository detection + the protected-path tool guard.
- `analysis.py` -- pure-Python structural analysis (large files, dependency
  cycles, unused modules, duplicate functions, test gaps) over the SAME
  SymbolIndex/DependencyGraph Phase 4 already builds -- feeds the three new
  worker types as pre-computed context, the same way Phase 4/5's graph/
  semantic retrieval feeds the main decision prompt.
- `regression_guard.py` -- run the repo's real test suite(s) before/after a
  change and decide whether it regressed.
- `history.py` -- `.kenzo/improvements/*.json` read/write.
"""
