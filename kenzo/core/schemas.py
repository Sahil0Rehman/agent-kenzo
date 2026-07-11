"""Shared data contracts used across tools, agents, models, and the CLI.

Every module that crosses a package boundary (tools -> agents -> core -> cli)
speaks these types. Keep this file dependency-free of the rest of `kenzo` so
it can be imported from anywhere without cycles.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """The kind of work a model call is being routed for."""

    CODING = "coding"
    DEBUGGING = "debugging"
    VISION = "vision"
    REVIEW = "review"
    PLANNING = "planning"
    RESEARCH = "research"
    GENERAL = "general"


class ToolResult(BaseModel):
    """The outcome of executing a single tool call."""

    tool_name: str
    success: bool
    output: str = ""
    error: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A tool invocation requested by a model, in OpenAI function-call shape."""

    id: str = ""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    """Registration record for a callable tool.

    `func` is intentionally excluded from serialization (it is a live
    Python callable, not JSON-safe) but is required to execute the tool.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    func: Any = Field(exclude=True)


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class PlanStep(BaseModel):
    id: int
    description: str
    rationale: str = ""
    status: PlanStepStatus = PlanStepStatus.PENDING


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    context_notes: str = ""


class ResearchFinding(BaseModel):
    summary: str
    relevant_files: list[str] = Field(default_factory=list)
    snippets: dict[str, str] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)


class CodeChange(BaseModel):
    file_path: str
    action: Literal["create", "edit", "delete"]
    diff: str = ""
    content: Optional[str] = None
    applied: bool = False


class TestResult(BaseModel):
    passed: bool
    summary: str
    command: str = ""
    raw_output: str = ""


class ReviewVerdict(BaseModel):
    approved: bool
    summary: str
    issues: list[str] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    """Result of `kenzo.core.project.ProjectAnalyzer` -- what kind of project
    this repository actually is, so the rest of Kenzo (Tester's command
    choice, the context bundle every agent reads) can stop assuming Python.
    """

    language: str = "unknown"  # "java" | "python" | "node" | "unknown"
    framework: Optional[str] = None  # e.g. "spring-boot"
    build: Optional[str] = None  # e.g. "maven" | "gradle" | "pip" | "npm"
    test_command: Optional[str] = None  # e.g. "mvn test", "pytest", "npm test"


class WorkingContext(BaseModel):
    """Accumulates across planner -> researcher -> coder -> tester -> reviewer
    within a single run, so later agents see what earlier agents actually
    discovered (file paths touched, short observations) instead of starting
    cold every stage. Deliberately holds only short notes and file paths --
    never full file contents -- see BaseAgent._update_working_context.
    """

    files_discovered: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ControllerAction(str, Enum):
    """The vocabulary of actions `kenzo.core.controller.AgentController`
    chooses from on every iteration of its reasoning loop, replacing the
    fixed planner->researcher->coder->tester->reviewer pipeline's rigid
    ordering with a dynamically-chosen sequence (the model decides what to
    do next, informed by TaskState, not a hardcoded graph edge).
    """

    PLAN = "plan"
    SEARCH = "search"
    READ = "read"
    ANALYZE = "analyze"
    EDIT = "edit"
    TEST = "test"
    REVIEW = "review"
    DELEGATE = "delegate"
    ASK_USER = "ask_user"
    FINISH = "finish"


class ActionDecision(BaseModel):
    """One iteration's decision: what to do next, and why."""

    action: ControllerAction
    rationale: str = ""
    target: str = ""  # a file path, search pattern, endpoint, or "" if not applicable


class FailedAttempt(BaseModel):
    """A recorded failure, so the next decision is informed by what already
    didn't work rather than repeating it blindly."""

    action: str
    target: str = ""
    error: str = ""


class Hypothesis(BaseModel):
    """A working theory about the bug/task, tracked so the agent can tell
    the difference between "confirmed by evidence" and "still guessing"."""

    description: str
    status: Literal["untested", "confirmed", "rejected"] = "untested"


class TaskState(BaseModel):
    """The running state of one `AgentController` task -- deliberately NOT
    the same thing as `KenzoState` (the old pipeline's LangGraph state).
    `KenzoState` threads a fixed sequence of node outputs; `TaskState`
    answers "what am I doing, why, and what have I already tried" for a
    controller loop that can revisit any phase in any order, any number of
    times, and persists to `.kenzo/tasks/<task_id>.json` between iterations
    so a task can be inspected -- or resumed -- mid-run.
    """

    task_id: str = ""
    goal: str
    repo_path: str = ""
    current_phase: str = "plan"
    files_seen: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    failed_attempts: list[FailedAttempt] = Field(default_factory=list)
    test_results: list[dict] = Field(default_factory=list)  # history of TestResult-shaped dicts
    decisions: list[str] = Field(default_factory=list)  # one-line "[iter N] action: rationale" log
    remaining_work: list[str] = Field(default_factory=list)
    changes: list[dict] = Field(default_factory=list)  # CodeChange-shaped dicts, same shape as KenzoState's
    iteration: int = 0
    tool_call_count: int = 0
    finished: bool = False
    needs_user_input: bool = False
    final_response: Optional[str] = None


class AgentBudgetConfig(BaseModel):
    """Configurable reasoning budget for AgentController -- replaces the old
    pipeline's hardcoded `max_iterations=6` per stage with limits that scale
    to the task (a config default, further scaled per-task by
    AgentController's own heuristic -- see `_compute_iteration_budget`)."""

    max_iterations: int = 50
    max_tool_calls: int = 100
    max_runtime_minutes: int = 30
    small_task_iterations: int = 10  # used for tasks whose plan has very few steps


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


MAX_REVIEW_RETRIES = 2


class KenzoState(TypedDict, total=False):
    """The LangGraph state threaded through planner -> researcher -> coder ->
    tester -> reviewer. Every node receives the full state dict and returns a
    partial dict of the keys it updates; LangGraph merges updates in.

    Field values are plain dict/str/list/int, not pydantic model instances,
    so the state stays trivially serializable across LangGraph checkpoints.
    Nodes should validate into/out of the pydantic models above at their own
    boundaries via `.model_dump()` / `Model.model_validate(...)`.
    """

    request: str
    repo_path: str
    plan: Optional[dict]  # Plan.model_dump()
    research: Optional[dict]  # ResearchFinding.model_dump()
    changes: list[dict]  # list[CodeChange.model_dump()]
    test_result: Optional[dict]  # TestResult.model_dump()
    review: Optional[dict]  # ReviewVerdict.model_dump()
    messages: list[dict]  # transcript for UI/logging, list[ChatMessage-like dict]
    final_response: Optional[str]
    error: Optional[str]
    iteration: int
    review_retries: int
    completed_steps: list[str]  # node names that finished, in order; set by AgentRuntime.run()
    crashed: bool  # True if AgentRuntime.run() caught an unhandled node exception
    working_context: dict  # WorkingContext.model_dump(); accumulated across all 5 agents


def new_state(request: str, repo_path: str) -> KenzoState:
    """Build the initial state for a fresh run."""
    return KenzoState(
        request=request,
        repo_path=repo_path,
        plan=None,
        research=None,
        changes=[],
        test_result=None,
        review=None,
        messages=[],
        final_response=None,
        error=None,
        iteration=0,
        review_retries=0,
        completed_steps=[],
        crashed=False,
        working_context=WorkingContext().model_dump(),
    )
