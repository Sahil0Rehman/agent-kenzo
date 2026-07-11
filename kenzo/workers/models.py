"""Shared data contracts for the Phase 9 worker system: `WorkerTask` is what
`AgentController` (or a test) hands to a worker; `WorkerResult` is what comes
back. Kept dependency-free of the rest of `kenzo`, mirroring the convention
in `kenzo.core.schemas`/`kenzo.vision.models`, so it can be imported from
anywhere without cycles.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class WorkerType(str, Enum):
    """Which specialized worker (see `kenzo.workers.specialized`) handles a
    `WorkerTask` -- selects both its system prompt and its read-only tool
    subset (see `kenzo.workers.worker_agent.WORKER_TOOL_NAMES`)."""

    CODE_RESEARCH = "code_research"
    BUG_ANALYSIS = "bug_analysis"
    TEST = "test"
    DOCUMENTATION = "documentation"
    GIT = "git"
    # Phase 10 ("Self Improvement System") additions -- same generic
    # WorkerAgent engine, just three more specialized prompts/tool subsets.
    ARCHITECTURE_CRITIC = "architecture_critic"
    TEST_GAP = "test_gap"
    PERFORMANCE = "performance"


class WorkerTask(BaseModel):
    """One independent investigation assignment for a single worker.

    `worker_id` is assigned by `WorkerManager.run_parallel` if left empty --
    callers constructing a `WorkerTask` directly (e.g. a test, or
    `kenzo.workers.specialized.SpecializedWorker.make_task`) don't need to
    invent one themselves.
    """

    worker_type: WorkerType
    description: str
    context: str = ""
    worker_id: str = ""


class WorkerResult(BaseModel):
    """What a worker reports back. Workers never edit files -- `files` is a
    list of paths the worker looked at or believes are relevant, not a
    change record (compare `kenzo.core.schemas.CodeChange`, which `EDIT`
    actions produce and this never does).
    """

    worker_id: str = ""
    worker_type: WorkerType
    task_description: str = ""
    success: bool = True
    findings: str = ""
    confidence: float = 0.0  # 0.0 (pure guess) .. 1.0 (verified by evidence)
    files: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls_made: int = 0
    # Phase 10: how serious a finding is -- populated by ArchitectureCriticWorker
    # (and usable by any worker type) as "low" | "medium" | "high" | "critical".
    # Optional/default None so every Phase 9 worker type and test is unaffected.
    severity: Optional[str] = None
