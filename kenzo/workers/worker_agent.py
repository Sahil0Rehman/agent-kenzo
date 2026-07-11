"""`WorkerAgent`: runs exactly one `WorkerTask` to completion.

A worker is a small, self-contained `run_tool_loop` (the SAME sub-loop
`AgentController._execute_action` uses for a normal action -- see
`kenzo.core.tool_loop`) scoped to a curated READ-ONLY tool subset. It is
temporary (constructed fresh per task, discarded after `run()` returns),
isolated (no access to `AgentController`'s `TaskState`, `ContextManager`, or
checkpoint -- it only sees what `WorkerTask.description`/`.context` hand it),
and cannot write files or run destructive MCP tools: `WORKER_TOOL_NAMES`
never includes `write_file`/`edit_file`/`run_command`, and
`ToolRegistry.subset()` is a genuine access-control boundary (a tool excluded
from a subset is absent from both the model's offered tools AND `execute()`
-- see its docstring), the exact same mechanism `ACTION_TOOLS` already relies
on to keep e.g. `SEARCH`/`READ` from ever reaching `write_file`.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from kenzo.core.schemas import TaskType
from kenzo.core.tool_loop import extract_json, run_tool_loop
from kenzo.mcp.permissions import ToolPermission
from kenzo.workers.models import WorkerResult, WorkerTask, WorkerType

# Per-worker-type read-only native tool names. Deliberately never includes
# write_file/edit_file/run_command (or any git tool beyond the read-only
# status/diff/branch/history quartet `ControllerAction.REVIEW` already
# trusts) -- see module docstring.
WORKER_TOOL_NAMES: dict[WorkerType, list[str]] = {
    WorkerType.CODE_RESEARCH: [
        "read_file",
        "list_directory",
        "search_files",
        "repository_tree",
        "find_symbol",
        "map_flow",
        "trace_flow",
        "find_references",
        "find_callers",
        "find_usages",
        "explain_architecture",
        "semantic_search",
    ],
    WorkerType.BUG_ANALYSIS: [
        "read_file",
        "search_files",
        "find_symbol",
        "trace_flow",
        "find_references",
        "find_callers",
        "find_usages",
        "explain_architecture",
        "semantic_search",
    ],
    WorkerType.TEST: [
        "read_file",
        "list_directory",
        "search_files",
        "repository_tree",
        "find_related_tests",
    ],
    WorkerType.DOCUMENTATION: [
        # Local fallback so a DocumentationWorker isn't useless with zero MCP
        # docs servers configured -- its real job (MCP docs lookup) is added
        # dynamically in `tool_names_for`, mirroring how AgentController
        # ._action_tool_names appends MCP tools by permission level.
        "search_files",
        "read_file",
    ],
    WorkerType.GIT: ["status", "diff", "branch", "history"],
    # Phase 10: read-only architecture/test/performance analysis workers.
    # Deliberately no run_command for PERFORMANCE -- "analyzes slow paths"
    # means static analysis of pre-computed facts + code reading, not
    # actually executing anything (that would risk running arbitrary code
    # during a supposedly read-only investigation).
    WorkerType.ARCHITECTURE_CRITIC: [
        "read_file",
        "list_directory",
        "search_files",
        "repository_tree",
        "find_symbol",
        "find_references",
        "find_callers",
        "explain_architecture",
        "semantic_search",
    ],
    WorkerType.TEST_GAP: [
        "read_file",
        "list_directory",
        "search_files",
        "repository_tree",
        "find_related_tests",
        "find_symbol",
        "semantic_search",
    ],
    WorkerType.PERFORMANCE: [
        "read_file",
        "search_files",
        "find_symbol",
        "trace_flow",
        "find_callers",
        "explain_architecture",
        "semantic_search",
    ],
}

_WORKER_TASK_TYPE: dict[WorkerType, TaskType] = {
    WorkerType.CODE_RESEARCH: TaskType.RESEARCH,
    WorkerType.BUG_ANALYSIS: TaskType.DEBUGGING,
    WorkerType.TEST: TaskType.RESEARCH,
    WorkerType.DOCUMENTATION: TaskType.RESEARCH,
    WorkerType.GIT: TaskType.RESEARCH,
    WorkerType.ARCHITECTURE_CRITIC: TaskType.REVIEW,
    WorkerType.TEST_GAP: TaskType.RESEARCH,
    WorkerType.PERFORMANCE: TaskType.REVIEW,
}

_WORKER_SYSTEM_PROMPTS: dict[WorkerType, str] = {
    WorkerType.CODE_RESEARCH: (
        "You are a CodeResearchWorker: a focused, read-only investigator. "
        "Explore the repository, trace call flows/dependencies, and summarize "
        "the relevant module(s) for the task below. You cannot edit anything -- "
        "your job is understanding, not fixing."
    ),
    WorkerType.BUG_ANALYSIS: (
        "You are a BugAnalysisWorker: a focused, read-only investigator. "
        "Analyze stack traces/error messages and investigate likely root causes "
        "for the task below, using the repository's real code -- never guess "
        "a cause you have not traced through actual references. You cannot "
        "edit anything -- your job is diagnosis, not fixing."
    ),
    WorkerType.TEST: (
        "You are a TestWorker: a focused, read-only investigator. Discover "
        "which existing tests are relevant to the task below and suggest how "
        "the eventual fix should be validated. You cannot edit anything or run "
        "commands -- your job is finding and recommending, not executing."
    ),
    WorkerType.DOCUMENTATION: (
        "You are a DocumentationWorker: a focused, read-only investigator. "
        "Look up relevant documentation for the task below -- prefer any "
        "connected MCP documentation/search tools; fall back to searching "
        "this repository's own README/docs files if none are connected or "
        "relevant. You cannot edit anything."
    ),
    WorkerType.GIT: (
        "You are a GitWorker: a focused, read-only investigator. Analyze this "
        "repository's history (status/diff/branch/history tools) for the task "
        "below -- e.g. what changed recently, and whether a recent commit is "
        "plausibly related. You cannot edit anything or make any git write."
    ),
    WorkerType.ARCHITECTURE_CRITIC: (
        "You are an ArchitectureCriticWorker: a focused, read-only code-quality "
        "reviewer. You are handed PRE-COMPUTED structural facts (large files, "
        "dependency cycles, unused modules, duplicate logic -- see the task "
        "context below) derived directly from the repository's own symbol/"
        "dependency graphs, not guessed. Your job is to INTERPRET those facts: "
        "confirm which are genuine technical debt vs. acceptable, judge "
        "severity, and read the actual affected files (read_file/"
        "explain_architecture) before recommending anything -- never rubber-"
        "stamp a precomputed fact without checking the real code first. You "
        "cannot edit anything -- your job is critique, not fixing."
    ),
    WorkerType.TEST_GAP: (
        "You are a TestGapWorker: a focused, read-only test-coverage "
        "investigator. Find modules with no corresponding test file, tests "
        "that only cover a narrow happy path, and missing integration "
        "coverage for the task below -- use find_related_tests/read_file to "
        "confirm gaps against the REAL test files, not assumptions. You "
        "cannot edit anything or run tests -- your job is finding gaps and "
        "suggesting what a new test should cover, not writing or running it."
    ),
    WorkerType.PERFORMANCE: (
        "You are a PerformanceWorker: a focused, read-only performance "
        "reviewer. Analyze the task's code for likely slow paths -- nested "
        "loops over large collections, repeated I/O/network calls inside a "
        "loop, expensive re-computation that could be cached, obviously "
        "quadratic patterns -- by reading the actual code (read_file/"
        "trace_flow/find_callers), not by running anything (you have no "
        "run_command access; suggest a fix, never execute a benchmark "
        "yourself). You cannot edit anything -- your job is diagnosis, not "
        "fixing."
    ),
}

_RESULT_FORMAT_INSTRUCTIONS = """\
When you are done investigating, respond with a final answer that is ONLY a \
single JSON object (no prose, no markdown fences) matching exactly this shape:
{
  "findings": "<your summary of what you found, in plain text>",
  "confidence": <float 0.0-1.0: how confident you are in these findings>,
  "files": ["<every file path you found relevant>"],
  "recommendations": ["<concrete next step(s) for whoever picks this up>"],
  "severity": "<one of: low, medium, high, critical -- omit or null if not applicable>"
}
Only report what you actually found through your tools -- do not invent file \
paths or claims you did not verify.\
"""

_VALID_SEVERITIES = frozenset({"low", "medium", "high", "critical"})

_WORKER_MAX_ITERATIONS = 6


def tool_names_for(worker_type: WorkerType, mcp_client: Optional[Any] = None) -> list[str]:
    """Resolve the read-only tool subset for `worker_type`. `mcp_client`, if
    given, additionally offers READ-permission MCP tools -- WRITE and
    DESTRUCTIVE are NEVER offered to a worker, regardless of `mcp.
    allow_write`/`allow_destructive` (a worker is unconditionally read-only,
    unlike the main agent's EDIT action)."""
    names = list(WORKER_TOOL_NAMES.get(worker_type, []))
    if mcp_client is not None:
        names.extend(mcp_client.tool_names_by_permission(ToolPermission.READ))
    return names


def _parse_worker_result(content: str) -> WorkerResult:
    """Never raises: a response that isn't parseable JSON, or that parses but
    doesn't validate, degrades to a best-effort result carrying the raw text
    as `findings` at a low confidence, mirroring `kenzo.models.router
    ._parse_vision_analysis`'s degrade-gracefully pattern -- a malformed
    worker response must never crash the parent delegation."""
    parsed = extract_json(content)
    if parsed:
        try:
            severity = str(parsed.get("severity") or "").strip().lower() or None
            if severity not in _VALID_SEVERITIES:
                severity = None
            return WorkerResult(
                worker_type=WorkerType.CODE_RESEARCH,  # overwritten by the caller
                findings=str(parsed.get("findings", ""))[:4000],
                confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0))),
                files=[str(f) for f in (parsed.get("files") or []) if f],
                recommendations=[str(r) for r in (parsed.get("recommendations") or []) if r],
                severity=severity,
            )
        except (TypeError, ValueError):
            pass
    text = content.strip()
    return WorkerResult(
        worker_type=WorkerType.CODE_RESEARCH,  # overwritten by the caller
        findings=text[:2000] if text else "(worker returned no parseable findings)",
        confidence=0.2 if text else 0.0,
    )


class WorkerAgent:
    """Generic worker execution engine -- one instance can run any number of
    `WorkerTask`s sequentially (each `run()` call is fully self-contained,
    fresh `run_tool_loop` messages every time); `WorkerManager` constructs
    one per task and runs them concurrently across threads instead.
    """

    def __init__(
        self,
        router: Any,
        tools: Any,
        mcp_client: Optional[Any] = None,
        max_iterations: int = _WORKER_MAX_ITERATIONS,
    ) -> None:
        self.router = router
        self.tools = tools
        self.mcp_client = mcp_client
        self.max_iterations = max_iterations

    def run(self, task: WorkerTask) -> WorkerResult:
        start = time.time()
        tool_names = tool_names_for(task.worker_type, self.mcp_client)
        scoped_tools = self.tools.subset(tool_names)
        system_prompt = _WORKER_SYSTEM_PROMPTS.get(
            task.worker_type, "You are a focused, read-only investigator. You cannot edit anything."
        )
        task_type = _WORKER_TASK_TYPE.get(task.worker_type, TaskType.RESEARCH)
        user_prompt = "\n\n".join(
            part
            for part in (
                f"Investigation task: {task.description}",
                task.context,
                _RESULT_FORMAT_INSTRUCTIONS,
            )
            if part
        )

        try:
            final_text, tool_transcript = run_tool_loop(
                self.router,
                scoped_tools,
                task_type,
                f"worker:{task.worker_type.value}",
                system_prompt,
                user_prompt,
                self.max_iterations,
            )
        except Exception as exc:  # noqa: BLE001 - a worker crashing must not crash the parent delegation
            return WorkerResult(
                worker_id=task.worker_id,
                worker_type=task.worker_type,
                task_description=task.description,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.time() - start,
            )

        result = _parse_worker_result(final_text)
        result.worker_id = task.worker_id
        result.worker_type = task.worker_type
        result.task_description = task.description
        result.duration_seconds = time.time() - start
        result.tool_calls_made = len(tool_transcript)

        # Merge files the worker's OWN tools actually touched (read_file
        # paths, search hits) with whatever it self-reported in its JSON --
        # a worker that forgets to list a file in "files" but did read it
        # should still surface that file to the main agent.
        observed_files = _files_from_transcript(tool_transcript)
        result.files = list(dict.fromkeys([*result.files, *observed_files]))
        return result


def _files_from_transcript(tool_transcript: list[dict]) -> list[str]:
    files: list[str] = []
    for entry in tool_transcript:
        arguments = entry.get("arguments") or {}
        data = (entry.get("result") or {}).get("data") or {}
        for path in (arguments.get("path"), data.get("path")):
            if path and path not in files:
                files.append(path)
        for match in data.get("matches") or []:
            path = match.get("path") if isinstance(match, dict) else None
            if path and path not in files:
                files.append(path)
    return files
