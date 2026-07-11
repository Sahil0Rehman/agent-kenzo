"""Named, typed convenience wrappers around `WorkerAgent` for each of the
Phase 9 spec's five worker responsibilities. The actual execution engine is
unified in `WorkerAgent`/`WORKER_TOOL_NAMES` (one tool-loop implementation,
selected by `WorkerType`) -- these classes exist so callers (the controller,
tests, or a future CLI) can construct/run a SPECIFIC worker type by name
without threading a `WorkerType` enum value through by hand, not because
each one needs its own execution logic.
"""

from __future__ import annotations

from typing import Any, Optional

from kenzo.workers.models import WorkerTask, WorkerType
from kenzo.workers.worker_agent import WorkerAgent


class SpecializedWorker(WorkerAgent):
    """Base class for a worker fixed to one `WorkerType`. Subclasses set the
    `worker_type` class attribute; `run_task` is a one-call convenience that
    builds the `WorkerTask` and runs it, for callers that don't need to
    construct one by hand."""

    worker_type: WorkerType

    def make_task(self, description: str, context: str = "", worker_id: str = "") -> WorkerTask:
        return WorkerTask(
            worker_type=self.worker_type, description=description, context=context, worker_id=worker_id
        )

    def run_task(self, description: str, context: str = ""):
        return self.run(self.make_task(description, context))


class CodeResearchWorker(SpecializedWorker):
    """Responsibilities: explore the repo, trace flows, summarize modules."""

    worker_type = WorkerType.CODE_RESEARCH


class BugAnalysisWorker(SpecializedWorker):
    """Responsibilities: analyze stack traces, investigate causes."""

    worker_type = WorkerType.BUG_ANALYSIS


class TestWorker(SpecializedWorker):
    """Responsibilities: discover tests, suggest validation."""

    worker_type = WorkerType.TEST


class DocumentationWorker(SpecializedWorker):
    """Responsibilities: MCP docs lookup."""

    worker_type = WorkerType.DOCUMENTATION


class GitWorker(SpecializedWorker):
    """Responsibilities: analyze history."""

    worker_type = WorkerType.GIT


class ArchitectureCriticWorker(SpecializedWorker):
    """Phase 10. Responsibilities: check bad abstractions, large files,
    dependency cycles, unused modules, technical debt -- interpreting
    pre-computed structural facts (see `kenzo.improve.analysis`) rather than
    rediscovering them from scratch. Returns severity/recommendation/
    affected files via the base `WorkerResult` shape."""

    worker_type = WorkerType.ARCHITECTURE_CRITIC


class TestGapWorker(SpecializedWorker):
    """Phase 10. Responsibilities: find untested modules, weak tests,
    missing integration coverage, and suggest what to test."""

    worker_type = WorkerType.TEST_GAP


class PerformanceWorker(SpecializedWorker):
    """Phase 10. Responsibilities: analyze slow paths, large loops,
    expensive indexing, cache misses; suggest optimizations. Never executes
    anything itself (no run_command access)."""

    worker_type = WorkerType.PERFORMANCE


WORKER_CLASSES: dict[WorkerType, type[SpecializedWorker]] = {
    WorkerType.CODE_RESEARCH: CodeResearchWorker,
    WorkerType.BUG_ANALYSIS: BugAnalysisWorker,
    WorkerType.TEST: TestWorker,
    WorkerType.DOCUMENTATION: DocumentationWorker,
    WorkerType.GIT: GitWorker,
    WorkerType.ARCHITECTURE_CRITIC: ArchitectureCriticWorker,
    WorkerType.TEST_GAP: TestGapWorker,
    WorkerType.PERFORMANCE: PerformanceWorker,
}


def build_worker(worker_type: WorkerType, router: Any, tools: Any, mcp_client: Optional[Any] = None) -> SpecializedWorker:
    """Construct the right `SpecializedWorker` subclass for `worker_type`."""
    return WORKER_CLASSES[worker_type](router, tools, mcp_client=mcp_client)
