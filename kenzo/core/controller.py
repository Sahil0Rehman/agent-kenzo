"""AgentController: the Phase 3 "true autonomous agent brain."

Replaces the fixed planner -> researcher -> coder -> tester -> reviewer
pipeline's rigid node ordering with a dynamic reasoning loop:

    while not complete:
        observe current state (TaskState + ContextManager)
        decide next action (PLAN/SEARCH/READ/ANALYZE/EDIT/TEST/REVIEW/
                             ASK_USER/FINISH)
        execute it (a small, ROLE-SCOPED tool-calling sub-loop)
        update context
        continue

The old pipeline (`kenzo.core.graph`, `AgentRuntime.run`) is left completely
untouched -- this is an additive, alternative execution engine, not a
replacement. `AgentRuntime.run_autonomous` is the new entry point that uses
it; `AgentRuntime.run` still drives the original fixed graph exactly as
before.

Safety guarantees carried over from the pipeline, preserved deliberately:
- Role boundaries: every action is scoped to a specific tool subset (see
  `ACTION_TOOLS`) exactly like the old `ROLE_TOOLS` -- e.g. SEARCH/READ/
  ANALYZE/REVIEW can never write a file or run a command; only EDIT can
  write, only TEST can run commands.
- Permissions: unchanged -- the underlying `ToolRegistry` still enforces
  `allow_write`/`allow_shell_exec`/`allow_git_write`/the command allowlist;
  this controller only decides WHICH scoped view of that same registry to
  use each iteration, it does not bypass any of it.
- Diffs: EDIT actions still flow through the same `write_file`/`edit_file`
  tools that compute real unified diffs; `_apply_updates` builds the same
  CodeChange-shaped dicts the old Coder built.
- Test gate: `_build_final_summary` refuses to let a FINISH decision read as
  an unqualified success if the most recently observed test result failed
  -- mirroring the old Reviewer's hard gate, adapted to a loop that doesn't
  have a single dedicated Reviewer stage anymore.

New in this phase: `CheckpointingToolRegistry` (checkpoint before the first
edit, `kenzo rollback` to undo), `CachedToolRegistry` (no repeated identical
reads within a run), `ContextManager` (bounded, relevance-retrievable
memory), and `TaskStore` (persist `TaskState` to `.kenzo/tasks/` every
iteration, enabling inspection and resume).
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from kenzo.core.checkpoint import CheckpointingToolRegistry, CheckpointManager, CrossRepoCheckpointingToolRegistry
from kenzo.core.code_search import CodeSearchEngine
from kenzo.core.context_manager import ContextManager
from kenzo.core.events import AgentEvent, EventSink
from kenzo.core.graph_context import render_graph_relevant
from kenzo.core.reindex import ReindexingToolRegistry
from kenzo.core.schemas import (
    ActionDecision,
    ControllerAction,
    FailedAttempt,
    Hypothesis,
    TaskState,
    TaskType,
)
from kenzo.core.semantic_context import render_semantic_relevant
from kenzo.core.tool_loop import extract_json, run_tool_loop
from kenzo.core.workspace import Workspace
from kenzo.mcp.permissions import ToolPermission
from kenzo.memory.task_store import TaskStore
from kenzo.server.edit_approval import EditApprovalToolRegistry
from kenzo.tools import cross_repo
from kenzo.tools.cache import CachedToolRegistry
from kenzo.workers.manager import WorkerManager
from kenzo.workers.merger import WorkerResultMerger
from kenzo.workers.models import WorkerResult, WorkerTask, WorkerType

# Per-action tool scoping -- this is the "role boundary" safety guarantee,
# reframed from fixed pipeline stages to dynamically-chosen actions. A role
# without write_file/edit_file in its subset cannot execute them even if a
# model hallucinates the call, exactly as under the old ROLE_TOOLS.
# Phase 11: cross-repo workspace tools (see kenzo.tools.cross_repo) -- only
# ever ACTUALLY registered/offered when a workspace with other repos is
# active (ToolRegistry.subset() silently drops an unregistered name, exactly
# like an MCP tool nobody connected), so listing them here unconditionally
# is safe for every single-repo caller, which is still the overwhelming
# majority of usage. Read-only tools go wherever the equivalent primary-repo
# read tool already goes; the two writable ones go ONLY on EDIT, mirroring
# write_file/edit_file exactly -- and even there, write_other_repo_file/
# edit_other_repo_file additionally refuse at the TOOL layer (not just by
# absence) against any repo not explicitly named in --also-edit.
_CROSS_REPO_READ_TOOLS = cross_repo.READ_TOOL_NAMES
_CROSS_REPO_WRITE_TOOLS = cross_repo.WRITE_TOOL_NAMES

ACTION_TOOLS: dict[ControllerAction, list[str]] = {
    ControllerAction.PLAN: [
        "read_file",
        "list_directory",
        "search_files",
        "repository_tree",
        "find_symbol",
        "explain_architecture",
        "semantic_search",
        *_CROSS_REPO_READ_TOOLS,
    ],
    ControllerAction.SEARCH: [
        "search_files",
        "find_symbol",
        "repository_tree",
        "list_directory",
        "trace_flow",
        "find_references",
        "find_callers",
        "find_usages",
        "semantic_search",
        *_CROSS_REPO_READ_TOOLS,
    ],
    ControllerAction.READ: ["read_file", "list_directory", *_CROSS_REPO_READ_TOOLS],
    ControllerAction.ANALYZE: [
        "read_file",
        "find_symbol",
        "map_flow",
        "search_files",
        "trace_flow",
        "find_references",
        "find_callers",
        "find_usages",
        "explain_architecture",
        "semantic_search",
        *_CROSS_REPO_READ_TOOLS,
    ],
    ControllerAction.EDIT: [
        "read_file",
        "list_directory",
        "search_files",
        "repository_tree",
        "write_file",
        "edit_file",
        *_CROSS_REPO_READ_TOOLS,
        *_CROSS_REPO_WRITE_TOOLS,
    ],
    ControllerAction.TEST: ["read_file", "run_command", "find_related_tests"],
    ControllerAction.REVIEW: ["read_file", "diff", "status", "history", "branch"],
    # DELEGATE calls no tools directly -- the main agent's own tool access is
    # irrelevant here; all real tool use happens INSIDE each isolated worker
    # (see kenzo.workers.worker_agent.WORKER_TOOL_NAMES), never in this loop.
    ControllerAction.DELEGATE: [],
    ControllerAction.ASK_USER: [],
    ControllerAction.FINISH: [],
}

# Phase 6 "Tool Planning Upgrade": which MCP permission levels (see
# kenzo.mcp.permissions.ToolPermission) get dynamically appended to an
# action's tool subset -- see AgentController._action_tool_names. MCP tool
# NAMES aren't known statically (they're discovered per-repo at runtime from
# whichever servers are configured), so unlike ACTION_TOOLS above this is a
# rule keyed by PERMISSION LEVEL, resolved against the live MCPClient at
# execution time, not a literal name list.
#
# Deliberately narrow: only SEARCH/ANALYZE/EDIT ever get MCP tools offered
# at all (READ/TEST/REVIEW/PLAN have well-defined, purely-local jobs already
# and gain nothing from external calls -- offering MCP tools there would
# just be unnecessary external-call surface, the opposite of what this
# upgrade is for). DESTRUCTIVE-permission tools are NEVER added to any
# action's scope here, regardless of `mcp.allow_destructive` -- that config
# flag only controls whether an ALREADY-OFFERED destructive tool's call-time
# gate (kenzo.mcp.adapter.build_tool_spec) allows it through; this is a
# second, independent layer of caution (a destructive MCP tool is
# unreachable through the controller's normal action loop at all, not just
# blocked-with-a-clear-error if reached).
_ACTION_MCP_PERMISSIONS: dict[ControllerAction, list[ToolPermission]] = {
    ControllerAction.SEARCH: [ToolPermission.READ],
    ControllerAction.ANALYZE: [ToolPermission.READ],
    ControllerAction.EDIT: [ToolPermission.READ, ToolPermission.WRITE],
}

# Which configured provider/model (TaskType -> routing -> provider) backs
# each action's execution sub-loop.
ACTION_TASK_TYPE: dict[ControllerAction, TaskType] = {
    ControllerAction.PLAN: TaskType.PLANNING,
    ControllerAction.SEARCH: TaskType.RESEARCH,
    ControllerAction.READ: TaskType.RESEARCH,
    ControllerAction.ANALYZE: TaskType.RESEARCH,
    ControllerAction.EDIT: TaskType.CODING,
    ControllerAction.TEST: TaskType.DEBUGGING,
    ControllerAction.REVIEW: TaskType.REVIEW,
    ControllerAction.DELEGATE: TaskType.PLANNING,
}


def _render_architecture_relevant(architecture_memory: Any, goal: str, top_k: int = 3) -> str:
    """Render `ArchitectureMemory.find_related(goal, top_k)` into the same
    short-fact-block style as `render_relevant`/`render_graph_relevant`/
    `render_semantic_relevant` -- long-term, cross-session project knowledge
    (past bugs/decisions/patterns) surfaced alongside this run's short-term
    and structural context. Returns "" if nothing related is on file yet.
    """
    entries = architecture_memory.find_related(goal, top_k=top_k)
    if not entries:
        return ""

    lines = []
    for entry in entries:
        files_note = f" (files: {', '.join(entry.files)})" if entry.files else ""
        detail = "; ".join(
            part
            for part in (
                f"root cause: {entry.root_cause}" if entry.root_cause else "",
                f"solution: {entry.solution}" if entry.solution else "",
                entry.notes,
            )
            if part
        )
        summary = f"'{entry.issue}'" if entry.issue else f"[{entry.kind}]"
        lines.append(f"- {summary}{(' -- ' + detail) if detail else ''}{files_note}")

    return "Related past issues (from long-term project memory):\n" + "\n".join(lines)


def _render_ide_context(context: Optional[dict[str, Any]]) -> str:
    """Render the Phase 7 IDE-context dict (see `kenzo.server.protocol
    .IdeContext`) into a short, priority-labeled block. Returns "" if no
    context is set (CLI mode, or an IDE client that hasn't pushed one yet).
    """
    if not context:
        return ""

    lines: list[str] = []
    if context.get("current_file"):
        lines.append(f"- Currently open file: {context['current_file']}")
    if context.get("selection"):
        start = context.get("selection_start_line")
        end = context.get("selection_end_line")
        location = f" (lines {start}-{end})" if start and end else ""
        lines.append(f"- User's current selection{location}:\n```\n{context['selection']}\n```")
    if context.get("open_tabs"):
        lines.append(f"- Other open tabs: {', '.join(context['open_tabs'][:15])}")
    if context.get("diagnostics"):
        diag_lines = [
            f"  - {d.get('severity', 'error')}: {d.get('file')}:{d.get('line')}: {d.get('message')}"
            for d in context["diagnostics"][:10]
        ]
        lines.append("- Current Problems/diagnostics:\n" + "\n".join(diag_lines))
    if context.get("git_status"):
        lines.append(f"- Git status: {context['git_status']}")

    if not lines:
        return ""
    return "IDE context (priority -- reflects what the user is actually looking at right now):\n" + "\n".join(lines)


def _render_vision_context(context: Optional[dict[str, Any]]) -> str:
    """Render the Phase 8 vision-analysis dict (see `kenzo.vision.models
    .VisionAnalysis.to_context_dict()`) into a short, priority-labeled block.
    Returns "" if no screenshot has been analyzed for this task."""
    if not context:
        return ""

    lines: list[str] = [f"- Screenshot type: {context.get('image_type', 'unknown')}"]
    if context.get("primary_error"):
        location = f" ({context['primary_file']}" + (f":{context['primary_line']})" if context.get("primary_line") else ")") if context.get("primary_file") else ""
        lines.append(f"- Primary error: {context['primary_error']}{location}")
    if context.get("error_messages"):
        lines.append(f"- Error message(s) visible: {'; '.join(context['error_messages'][:5])}")
    if context.get("stack_traces"):
        lines.append("- Stack trace(s) visible:\n" + "\n".join(f"  {t}" for t in context["stack_traces"][:10]))
    if context.get("file_names"):
        lines.append(f"- File name(s) visible: {', '.join(context['file_names'][:10])}")
    if context.get("line_numbers"):
        lines.append(f"- Line number(s) visible: {', '.join(str(n) for n in context['line_numbers'][:10])}")
    if context.get("ui_elements"):
        lines.append(f"- UI element(s) visible: {', '.join(context['ui_elements'][:10])}")
    if context.get("possible_causes"):
        lines.append("- Model's own hypotheses:\n" + "\n".join(f"  - {c}" for c in context["possible_causes"][:5]))
    if context.get("visible_text"):
        lines.append(f"- Other visible text: {context['visible_text'][:500]}")

    return "Screenshot analysis (priority -- the user attached this image):\n" + "\n".join(lines)


def _render_workspace_context(workspace: Optional[Workspace]) -> str:
    """Render the Phase 11 multi-repo workspace (see `kenzo.core.workspace
    .Workspace`) into a short block telling the model what other repos it
    can reach and how. Returns "" if no workspace is active (the default --
    single-repo callers are completely unaffected) or the workspace has no
    other repos configured. Unlike ide_context/vision_context, this is
    STATIC for the whole run (decided once via CLI flags, not settable
    mid-task), so it's rendered every decision, not cleared/replaced.
    """
    if workspace is None or not workspace.has_any_other_repo():
        return ""

    lines: list[str] = []
    for repo in workspace.others:
        kind = "read+write" if repo.writable else "READ-ONLY"
        lines.append(f"- '{repo.alias}' ({kind}): {repo.path}")

    guidance = (
        "Use read_other_repo_file/list_other_repo_directory/search_other_repo_files to "
        "reference these (e.g. checking how a legacy/other branch implemented something "
        "before changing the current one)."
    )
    if workspace.has_writable_other_repo():
        guidance += (
            " write_other_repo_file/edit_other_repo_file work ONLY against the repo(s) marked "
            "read+write above -- e.g. porting the same fix into another branch. Every other repo "
            "is refused at the tool level even if you try."
        )
    else:
        guidance += " None of them are editable this run -- writes there will be refused."

    return (
        "Other repos available in this workspace (in addition to the primary repo you're "
        f"working in):\n" + "\n".join(lines) + "\n" + guidance
    )


_DECIDE_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "decide_next_action",
        "description": (
            "Decide the single next action to take toward the goal, given everything "
            "known so far. You must call this exactly once per turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [a.value for a in ControllerAction],
                    "description": "The next action to take.",
                },
                "rationale": {
                    "type": "string",
                    "description": "One or two sentences: why this action, right now.",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "The specific file path, search pattern, symbol name, or endpoint "
                        "this action concerns, or an empty string if not applicable."
                    ),
                },
            },
            "required": ["action", "rationale"],
        },
    },
}

_DECISION_SYSTEM_PROMPT = """\
You are the reasoning core of Kenzo, a local autonomous coding agent. Unlike a \
rigid fixed pipeline, YOU decide what to do next at every step, informed by \
everything discovered so far. Call decide_next_action exactly once with your \
choice.

Available actions:
- plan: form or revise a plan for how to achieve the goal
- search: look for relevant files/symbols/code you don't already know about
- read: read a specific already-located file's contents
- analyze: reason about what you've read/found, tracing dependencies, call \
chains, or root causes
- edit: make a real change to a file
- test: validate the current state of the code by running its test suite
- review: critically check that a change is correct and complete before finishing
- delegate: spin up several independent, read-only worker agents to \
investigate different angles IN PARALLEL, then get their findings back as one \
merged summary (see below)
- ask_user: you are stuck and need clarification only a human can provide
- finish: the goal is achieved, or you have concluded further progress is not \
currently possible (explain which, honestly, in your rationale)

Guidance:
- Consult the task state (files seen/modified, hypotheses, failed attempts, \
test results, past decisions, remaining work) before deciding -- know what you \
are doing and why.
- Never blindly repeat an action that failed_attempts shows already failed with \
the same target. Choose a genuinely different approach instead: search with \
different terms, read a different but related file, or reconsider your \
hypothesis.
- Prefer testing after editing. Do not finish claiming success unless the most \
recent test result actually passed, or you have clearly explained in your \
rationale why you are finishing without one.
- Do not finish while remaining_work still lists items you have a realistic way \
to act on.

When to delegate (uncertainty is genuinely high and the investigation splits \
into independent angles):
- A large or unfamiliar repository where you don't yet know where things live.
- An unknown framework/stack you have not already established conventions for.
- Multiple plausible root causes that could be checked independently (e.g. the \
controller, the service it calls, AND whether a recent commit changed \
something -- three separate things to look at, not one).
Do NOT delegate a single, already-well-scoped lookup you could just search/read \
yourself -- delegation has overhead (multiple model calls) and is only worth it \
when the sub-investigations are genuinely independent and parallelizable. \
Workers are read-only: they report findings, they never edit files themselves.
"""

_ACTION_INSTRUCTIONS: dict[ControllerAction, str] = {
    ControllerAction.PLAN: (
        "You are forming or revising a plan for the current goal. Inspect the "
        "repository with your tools if you need to before committing to a plan -- "
        "never plan from assumptions alone. Respond with a final answer that is "
        "ONLY a single JSON object (no prose, no markdown fences) matching: "
        '{"goal": "<restated goal>", "steps": [{"id": 1, "description": "...", '
        '"rationale": "..."}], "context_notes": "..."}.'
    ),
    ControllerAction.SEARCH: (
        "You are looking for files/symbols/code relevant to the current goal or "
        "target that you have not already found. Use search_files/find_symbol/"
        "repository_tree/list_directory, or -- when the indexed code intelligence "
        "is available -- the more precise trace_flow/find_references/find_callers/"
        "find_usages tools, which are backed by real parsed symbols rather than "
        "text matching. If you don't know the exact symbol/file name -- e.g. the "
        "goal describes a PROBLEM rather than naming code ('users can't sign out') "
        "-- try semantic_search first: it combines symbol/graph/vector/keyword "
        "signals and can surface the right code even without an exact name match. "
        "Prefer these LOCAL tools first -- only reach for an external MCP tool "
        "(e.g. an issue tracker, database, or documentation lookup, if any are "
        "connected) when the goal genuinely references something outside this "
        "repository (e.g. 'fix issue #431', 'why is the login query failing') that "
        "local search cannot answer. When done, summarize in plain text what you "
        "found and where."
    ),
    ControllerAction.READ: (
        "You are reading a specific file to understand its current contents "
        "before acting on it. Use read_file (and list_directory if you need to "
        "locate the exact path first). Summarize the relevant parts in plain text."
    ),
    ControllerAction.ANALYZE: (
        "You are reasoning about relationships between what you've already found "
        "-- call chains, dependency injection, root cause of a bug. Prefer "
        "trace_flow/find_references/find_callers/find_usages/explain_architecture "
        "over the older find_symbol/map_flow when available -- they are backed by "
        "a real AST-derived reference and dependency graph, not regex guessing. "
        "Use read_file/search_files as needed too. If any external MCP tools are "
        "connected (e.g. a database schema/query tool) and the root cause plausibly "
        "lives outside this repository's own code, use them -- but only after local "
        "analysis has narrowed down what to check externally, not as a first resort. "
        "Summarize your conclusion in plain text, including which hypothesis it "
        "confirms or rejects."
    ),
    ControllerAction.EDIT: (
        "You are making a real change to the repository, or finalizing external work "
        "related to it (e.g. preparing a pull request once local edits and tests are "
        "done). For a code change you MUST use write_file/edit_file -- never just "
        "describe it in prose. Read the current contents first with read_file if you "
        "have not already confirmed them this run. If a previous attempt failed, fix "
        "the specific problem rather than repeating the identical call. If the goal "
        "involves an external system (e.g. 'prepare a PR for this fix') and a "
        "matching MCP tool is available, use it only after the local code change is "
        "actually made and tested -- never prepare external follow-up for a change "
        "that hasn't happened yet. Summarize what you changed and why in plain text "
        "when done."
    ),
    ControllerAction.TEST: (
        "You are validating the current state of the code. If you just edited a "
        "specific file and are unsure which tests cover it, call "
        "find_related_tests(file_path) first to target the right test(s) instead "
        "of guessing from the file name alone. Use the run_command tool to "
        "execute the project's real test suite -- the context above tells you "
        "the detected language/build system/recommended test command; use that "
        "command with an explicit, generous timeout rather than guessing. If "
        "shell execution is disabled or the command is not allowed, say so "
        "explicitly and instead do a careful static read-through via read_file. "
        'Respond with a final answer that is ONLY a single JSON object matching: '
        '{"passed": true or false, "summary": "...", "command": "...", "raw_output": "..."}.'
    ),
    ControllerAction.REVIEW: (
        "You are critically checking whether the work done so far actually "
        "satisfies the goal and is safe to consider finished. Use read_file/diff/"
        "status/history/branch to double-check rather than trusting summaries "
        "alone. Summarize your verdict in plain text, including any concrete "
        "issues that still need fixing before this can be finished."
    ),
}

_DELEGATE_PLANNING_PROMPT = """\
You are splitting an investigation into 2-5 INDEPENDENT worker tasks that can \
run in parallel, each handled by one specialized read-only worker. Available \
worker types (choose the one best suited to each sub-task):
- code_research: explore the repo, trace flows, summarize modules
- bug_analysis: analyze stack traces/errors, investigate root causes
- test: discover relevant tests, suggest how to validate a fix
- documentation: look up relevant documentation (MCP docs tools, or local \
README/docs as a fallback)
- git: analyze recent history (status/diff/branch/log) for related changes

Each task's "description" must be a SELF-CONTAINED instruction a worker with \
zero other context could act on -- not a one-word label. Tasks should be \
genuinely independent (not one depending on another's output) since they all \
run at the same time. Respond with a final answer that is ONLY a single JSON \
object (no prose, no markdown fences) matching exactly:
{"tasks": [{"worker_type": "code_research", "description": "..."}, ...]}\
"""

_PER_ACTION_SUB_ITERATIONS = 4
_REPEAT_WARNING_THRESHOLD = 3
_READ_SUMMARY_MAX_CHARS = 500


class AgentController:
    """Drives one request through the dynamic action loop for one repository."""

    def __init__(
        self,
        router: Any,
        tools: Any,
        memory: Any,
        config: Any,
        on_event: Optional[EventSink] = None,
        indexing_service: Optional[Any] = None,
        session_memory: Optional[Any] = None,
        architecture_memory: Optional[Any] = None,
        mcp_client: Optional[Any] = None,
        ide_mode: bool = False,
        workspace: Optional[Workspace] = None,
    ) -> None:
        self.router = router
        self.memory = memory
        self.config = config
        self.on_event = on_event
        self.indexing_service = indexing_service
        self.session_memory = session_memory
        self.architecture_memory = architecture_memory
        self.mcp_client = mcp_client
        # Phase 11: other repos this run can see (--see) and, if explicitly
        # granted, edit (--also-edit) -- see kenzo.core.workspace. None (the
        # default, and every pre-Phase-11 caller) means "single repo,"
        # preserving exact prior behavior.
        self.workspace = workspace

        self.checkpoint = CheckpointManager(str(memory.repo_root))
        scoped_tools: Any = CachedToolRegistry(tools)
        # Phase 4/5: if code intelligence was built for this repo (see
        # AgentRuntime.__init__), keep it incrementally fresh -- symbol index,
        # reference/dependency graphs, AND the semantic index -- as EDIT
        # actions change files. Optional and additive, so callers/tests that
        # build an AgentController without it (still exercising the exact
        # same safety guarantees below) are unaffected.
        if indexing_service is not None:
            scoped_tools = ReindexingToolRegistry(scoped_tools, indexing_service)
        self.tools = CheckpointingToolRegistry(scoped_tools, self.checkpoint)

        # Phase 11: a SEPARATE CheckpointManager per --also-edit repo (keyed
        # by workspace alias), so a bad cross-repo edit is exactly as
        # recoverable via `kenzo rollback <that repo's path>` as a bad edit
        # to the primary repo already is. Wrapped OUTSIDE the primary
        # checkpoint wrapper above -- see CrossRepoCheckpointingToolRegistry's
        # docstring for why the two never intercept the same tool call.
        self._also_edit_checkpoints: dict[str, CheckpointManager] = {}
        if workspace is not None and workspace.has_writable_other_repo():
            self._also_edit_checkpoints = {
                repo.alias: CheckpointManager(repo.path) for repo in workspace.others if repo.writable
            }
            self.tools = CrossRepoCheckpointingToolRegistry(self.tools, self._also_edit_checkpoints)

        # Phase 7: IDE mode (kenzo.server only -- CLI never sets this) wraps
        # write_file/edit_file so they propose-then-wait for approval instead
        # of writing immediately, OUTERMOST in the chain so an accepted
        # proposal's actual write still flows through checkpoint/reindex/
        # cache-invalidation exactly as before. `self._edit_approval` stays
        # `None` in CLI mode, making `set_edit_proposal_handler`/
        # `resolve_edit_proposal` safe (harmless) no-ops there.
        self._edit_approval: Optional[EditApprovalToolRegistry] = None
        if ide_mode:
            self._edit_approval = EditApprovalToolRegistry(self.tools, str(memory.repo_root), config.permissions)
            self.tools = self._edit_approval

        self.context = ContextManager()
        self.task_store = TaskStore(str(memory.repo_root))
        # Phase 7: current IDE state (open file, selection, diagnostics, ...)
        # -- see set_ide_context. None until kenzo.server pushes one; CLI
        # mode never sets this.
        self.ide_context: Optional[dict[str, Any]] = None
        # Phase 8: the most recently analyzed screenshot -- see
        # set_vision_context. None until `kenzo image`/kenzo.server's
        # attach_image handling sets one.
        self.vision_context: Optional[dict[str, Any]] = None

        # Phase 9: parallel worker system, backing the DELEGATE action (see
        # _execute_delegate). Built on `self.tools` (the FULLY wrapped chain
        # above -- checkpoint/reindex/edit-approval) so a worker's `.subset()`
        # goes through the exact same access-control mechanism SEARCH/READ/
        # ANALYZE/REVIEW already trust; irrelevant in practice since a
        # worker's tool subset never includes write_file/edit_file anyway,
        # but there is no reason to bypass it. `self._emit` is a bound
        # method reference (not yet invokable at this point in __init__
        # since Python resolves it lazily on first call, by which point
        # __init__ has long finished), so this is safe despite being defined
        # further down this class.
        workers_config = getattr(config, "workers", None)
        if workers_config is None:
            from kenzo.config.settings import WorkersConfig

            workers_config = WorkersConfig()
        self.worker_manager = WorkerManager(
            router, self.tools, workers_config, mcp_client=mcp_client, emit=self._emit, workspace=workspace
        )

    def set_ide_context(self, context: Optional[dict[str, Any]]) -> None:
        """IDE mode only: set (or clear, with `None`) the current IDE state
        pushed by the connected client (current file, selection, open tabs,
        diagnostics, workspace path, git status). Rendered as PRIORITY
        context ahead of everything else in `_decide_next_action` -- see
        `_render_ide_context`."""
        self.ide_context = context

    def set_vision_context(self, context: Optional[dict[str, Any]]) -> None:
        """Set (or clear, with `None`) the current screenshot analysis
        (`kenzo.vision.models.VisionAnalysis.to_context_dict()`) -- rendered
        as PRIORITY context, ahead of everything except IDE context, in
        `_decide_next_action`. See `_render_vision_context`."""
        self.vision_context = context

    def set_edit_proposal_handler(self, on_proposal: Optional[Any]) -> None:
        """IDE mode only: set (or clear, with `None`) the callback invoked
        with each proposed `write_file`/`edit_file` edit before it's
        applied. No-op if `ide_mode=False` (CLI). See
        `kenzo.server.edit_approval.EditApprovalToolRegistry`."""
        if self._edit_approval is not None:
            self._edit_approval.on_proposal = on_proposal

    def resolve_edit_proposal(self, proposal_id: str, decision: str, applied_by_client: bool = False) -> bool:
        """IDE mode only: resolve a pending edit proposal (called when a
        WebSocket client sends an `edit_decision` message). Returns `False`
        (harmless) if `ide_mode=False` or `proposal_id` is unknown/stale."""
        if self._edit_approval is None:
            return False
        return self._edit_approval.resolve_proposal(proposal_id, decision, applied_by_client)

    # -- public entry point ---------------------------------------------

    def run(self, request: str, task_id: Optional[str] = None, cancel_event: Optional[Any] = None) -> dict:
        """Drive `request` through the dynamic action loop and return a
        result dict (see `_build_result`). If `task_id` names an existing,
        unfinished saved task, resumes it instead of starting fresh.

        `cancel_event` (Phase 7) is an optional `threading.Event`-like object
        (anything with `.is_set()`) checked once per iteration, alongside the
        existing budget checks -- `kenzo.server` sets one when a WebSocket
        client sends a cancel message mid-task. `None` (the default, and
        every pre-Phase-7 caller) means "never cancellable," preserving
        exact prior behavior.
        """
        task_state = self._load_or_create_task(request, task_id)

        # A resumed (not fresh) task may already have an active checkpoint
        # from its own earlier iterations, which must stay active so a
        # resumed task's first NEW edit doesn't create a second checkpoint
        # partway through what is still logically one task. Only reset for
        # a genuinely fresh task -- otherwise `self.checkpoint` (reused
        # across an entire session for ContextManager continuity) would
        # keep pointing at whichever task happened to edit first, and
        # `kenzo rollback` would silently revert every later task too.
        if task_state.iteration == 0 and not task_state.files_modified:
            self.checkpoint.reset()

        budget = self.config.agent
        iteration_ceiling = budget.max_iterations
        start_time = time.time()
        last_signature: Optional[tuple] = None
        repeat_count = 0

        while not task_state.finished:
            elapsed_minutes = (time.time() - start_time) / 60.0
            if cancel_event is not None and cancel_event.is_set():
                self._stop(task_state, "cancelled by the user")
                break
            if task_state.iteration >= iteration_ceiling:
                self._stop(task_state, "reached the maximum iteration budget")
                break
            if task_state.tool_call_count >= budget.max_tool_calls:
                self._stop(task_state, "reached the maximum tool-call budget")
                break
            if elapsed_minutes >= budget.max_runtime_minutes:
                self._stop(task_state, "reached the maximum runtime budget")
                break

            decision = self._decide_next_action(task_state)
            signature = (decision.action, decision.target)
            repeat_count = repeat_count + 1 if signature == last_signature else 0
            last_signature = signature

            if repeat_count >= _REPEAT_WARNING_THRESHOLD:
                nudge = (
                    f"NOTE: '{decision.action.value}' on target '{decision.target}' has been "
                    f"attempted {repeat_count + 1} times without success -- a fundamentally "
                    "different approach is needed, not a repeat of the same call."
                )
                if nudge not in task_state.remaining_work:
                    task_state.remaining_work.insert(0, nudge)

            self._emit("controller", f"{decision.action.value}: {decision.rationale}")

            if decision.action == ControllerAction.FINISH:
                task_state.finished = True
                task_state.final_response = self._build_final_summary(task_state, decision.rationale)
                self._save(task_state)
                break

            if decision.action == ControllerAction.ASK_USER:
                task_state.finished = True
                task_state.needs_user_input = True
                task_state.final_response = f"Kenzo needs your input: {decision.rationale}"
                if decision.target and decision.target not in task_state.remaining_work:
                    task_state.remaining_work.append(decision.target)
                self._save(task_state)
                break

            final_text, tool_transcript = self._execute_action(decision, task_state)
            self._apply_updates(task_state, decision, final_text, tool_transcript)

            if decision.action == ControllerAction.PLAN:
                step_count = len(task_state.remaining_work)
                if step_count and step_count <= 2:
                    iteration_ceiling = min(iteration_ceiling, budget.small_task_iterations)

            task_state.iteration += 1
            self._save(task_state)

        result = self._build_result(task_state)
        self._emit(
            "controller",
            task_state.final_response or "Task finished.",
            event_type="complete",
            data={"success": result.get("success", False), "task_id": task_state.task_id},
        )
        return result

    # -- task lifecycle ----------------------------------------------------

    def _load_or_create_task(self, request: str, task_id: Optional[str]) -> TaskState:
        if task_id:
            existing = self.task_store.load(task_id)
            if existing is not None and not existing.get("finished"):
                return TaskState.model_validate(existing)
        resolved_id = task_id or self.task_store.generate_task_id(request)
        return TaskState(task_id=resolved_id, goal=request, repo_path=str(self.memory.repo_root))

    def _save(self, task_state: TaskState) -> None:
        self.task_store.save(task_state.task_id, task_state.model_dump())

    def _stop(self, task_state: TaskState, reason: str) -> None:
        task_state.finished = True
        task_state.final_response = self._build_stopped_summary(task_state, reason)
        self._save(task_state)

    def _emit(self, stage: str, message: str, event_type: str = "thinking", data: Optional[dict] = None) -> None:
        if self.on_event is not None:
            self.on_event(AgentEvent(stage=stage, message=message, event_type=event_type, data=data or {}))

    def _on_tool_start(self, tool_name: str, arguments: dict) -> None:
        self._emit("tool", f"Calling {tool_name}...", event_type="tool_start", data={"tool": tool_name, "arguments": arguments})
        if tool_name == "run_command":
            self._emit(
                "tool",
                f"Running: {arguments.get('command', '')}",
                event_type="test_running",
                data={"command": arguments.get("command", "")},
            )

    def _on_tool_result(self, tool_name: str, arguments: dict, result: Any) -> None:
        self._emit(
            "tool",
            f"{tool_name} {'succeeded' if result.success else 'failed'}",
            event_type="tool_result",
            data={"tool": tool_name, "success": result.success, "output": result.output[:500], "error": result.error},
        )
        if tool_name in ("write_file", "edit_file") and result.success:
            path = arguments.get("path") or (result.data or {}).get("path", "")
            self._emit(
                "tool",
                f"Edited {path}",
                event_type="file_changed",
                data={"path": path, "diff": (result.data or {}).get("diff", ""), "action": "create" if tool_name == "write_file" else "edit"},
            )

    # -- decision step -----------------------------------------------------

    def _decide_next_action(self, task_state: TaskState) -> ActionDecision:
        context_bundle = self.memory.build_context_bundle()
        relevant = self.context.render_relevant(task_state.goal, top_k=8)

        # Phase 4: graph-aware retrieval, additive to the keyword-overlap
        # `relevant` above -- surfaces structurally-connected code (call
        # chains, dependencies, related tests) that a plain keyword match
        # would miss, e.g. "Fix logout" pulling in the endpoint's controller,
        # the service/repository it calls into, and its test file. Reads
        # `self.indexing_service`'s CURRENT symbol_index/reference_graph/
        # dependency_graph at call time (never cached), so this stays correct
        # even while a background full rebuild is still in flight. Silently
        # empty if no code intelligence has been built for this repo yet.
        symbol_index = self.indexing_service.symbol_index if self.indexing_service else None
        reference_graph = self.indexing_service.reference_graph if self.indexing_service else None
        dependency_graph = self.indexing_service.dependency_graph if self.indexing_service else None
        graph_relevant = render_graph_relevant(task_state.goal, symbol_index, reference_graph, dependency_graph, top_k=8)

        # Phase 5: hybrid semantic search, additive to the two blocks above --
        # a `CodeSearchEngine` is cheap to construct (pure query-time reader,
        # no I/O in __init__) so a fresh one is built every decision, always
        # reflecting the indexing service's live state.
        semantic_engine = None
        if self.indexing_service is not None and symbol_index is not None:
            semantic_engine = CodeSearchEngine(
                symbol_index,
                reference_graph,
                dependency_graph,
                self.indexing_service.embedding_router,
                self.indexing_service.vector_store,
            )
        semantic_relevant = render_semantic_relevant(task_state.goal, semantic_engine, top_k=8)

        # Phase 5: cross-session and long-term project memory -- what has
        # this SESSION already done that might be "the same issue" (recency-
        # ordered, plain text, letting the model itself infer the reference),
        # and what has this REPOSITORY's history already taught us about
        # similar issues (keyword-ranked structured entries).
        session_relevant = self.session_memory.render_recent(top_k=3) if self.session_memory else ""
        architecture_relevant = (
            _render_architecture_relevant(self.architecture_memory, task_state.goal, top_k=3)
            if self.architecture_memory
            else ""
        )

        # Phase 7: IDE context (current file/selection/diagnostics/open tabs)
        # is PRIORITY context -- rendered first among the retrieval blocks,
        # ahead of anything Kenzo would otherwise have to go search for. See
        # kenzo.server.app._stash_ide_context for how this gets set.
        ide_context_block = _render_ide_context(self.ide_context)
        vision_context_block = _render_vision_context(self.vision_context)
        workspace_context_block = _render_workspace_context(self.workspace)

        task_summary = self._render_task_state(task_state)

        system_prompt = f"{context_bundle}\n\n{_DECISION_SYSTEM_PROMPT}"
        user_prompt = "\n\n".join(
            part
            for part in (
                ide_context_block,
                vision_context_block,
                workspace_context_block,
                task_summary,
                relevant,
                graph_relevant,
                semantic_relevant,
                session_relevant,
                architecture_relevant,
                "Decide the next action now.",
            )
            if part
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.router.complete(
                TaskType.GENERAL,
                messages,
                tools=[_DECIDE_ACTION_TOOL],
                tool_choice={"type": "function", "function": {"name": "decide_next_action"}},
            )
        except Exception as exc:  # noqa: BLE001 - a routing/provider error must not crash the loop
            self._emit("controller", f"Model/routing error: {exc}", event_type="error", data={"error": str(exc)})
            return ActionDecision(
                action=ControllerAction.FINISH,
                rationale=f"Stopping: could not get a decision from the model ({exc}).",
            )

        message = response.choices[0].message
        parsed: Optional[dict] = None
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            try:
                parsed = json.loads(tool_calls[0].function.arguments or "{}")
            except (json.JSONDecodeError, TypeError, IndexError, AttributeError):
                parsed = None
        if parsed is None:
            parsed = extract_json(getattr(message, "content", "") or "")

        if not parsed or "action" not in parsed:
            return ActionDecision(
                action=ControllerAction.SEARCH,
                rationale="Fallback: the model's decision could not be parsed; exploring further.",
            )

        try:
            action = ControllerAction(str(parsed.get("action", "")).strip().lower())
        except ValueError:
            action = ControllerAction.SEARCH

        return ActionDecision(
            action=action,
            rationale=str(parsed.get("rationale", "")),
            target=str(parsed.get("target", "") or ""),
        )

    # -- execution step ------------------------------------------------

    def _action_tool_names(self, action: ControllerAction) -> list[str]:
        """Static `ACTION_TOOLS[action]` plus, for actions where external
        tool use is plausible (see `_ACTION_MCP_PERMISSIONS`), whichever MCP
        tools are CURRENTLY discovered at an allowed permission level. MCP
        tool names aren't known until a repo's configured servers connect,
        so -- unlike native tools -- this can't be a static module-level
        list; it's resolved fresh against `self.mcp_client` every time an
        action executes, which also means a server that reconnects/changes
        its tool set mid-session is picked up on the next action without
        any extra wiring.
        """
        names = list(ACTION_TOOLS.get(action, []))
        if self.mcp_client is not None:
            for permission in _ACTION_MCP_PERMISSIONS.get(action, []):
                names.extend(self.mcp_client.tool_names_by_permission(permission))
        return names

    def _execute_action(self, decision: ActionDecision, task_state: TaskState) -> tuple[str, list[dict]]:
        if decision.action == ControllerAction.DELEGATE:
            return self._execute_delegate(decision, task_state)

        action_tool_names = self._action_tool_names(decision.action)
        scoped_tools = self.tools.subset(action_tool_names)
        task_type = ACTION_TASK_TYPE.get(decision.action, TaskType.GENERAL)
        instructions = _ACTION_INSTRUCTIONS.get(decision.action, "")

        context_bundle = self.memory.build_context_bundle()
        query = decision.target or task_state.goal
        relevant = self.context.render_relevant(query, top_k=8)
        task_summary = self._render_task_state(task_state)

        system_prompt = f"{context_bundle}\n\n{instructions}"
        user_prompt = "\n\n".join(
            part
            for part in (
                f"Current goal: {task_state.goal}",
                f"Rationale for this step: {decision.rationale}",
                f"Target: {decision.target}" if decision.target else "",
                task_summary,
                relevant,
            )
            if part
        )

        remaining_calls = max(1, self.config.agent.max_tool_calls - task_state.tool_call_count)
        max_iterations = min(_PER_ACTION_SUB_ITERATIONS, remaining_calls)

        return run_tool_loop(
            self.router,
            scoped_tools,
            task_type,
            "controller",
            system_prompt,
            user_prompt,
            max_iterations,
            on_tool_start=self._on_tool_start,
            on_tool_result=self._on_tool_result,
        )

    # -- delegation (Phase 9) -----------------------------------------------

    def _propose_worker_tasks(self, decision: ActionDecision, task_state: TaskState) -> list[WorkerTask]:
        """A single, non-tool-calling model call (same style as PLAN's own
        dedicated JSON-producing sub-call) that splits the current
        uncertainty into 2-5 independent `WorkerTask`s. Degrades to one
        CODE_RESEARCH task using the decision's own target/rationale if the
        model's response can't be parsed -- delegation must never end up
        spawning zero workers."""
        context_bundle = self.memory.build_context_bundle()
        task_summary = self._render_task_state(task_state)
        user_prompt = "\n\n".join(
            part
            for part in (
                f"Current goal: {task_state.goal}",
                f"Why delegation was chosen now: {decision.rationale}",
                f"Target/focus: {decision.target}" if decision.target else "",
                task_summary,
            )
            if part
        )
        system_prompt = f"{context_bundle}\n\n{_DELEGATE_PLANNING_PROMPT}"

        try:
            content = self.router.complete_text(TaskType.PLANNING, system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - a routing/provider error must still yield a fallback task
            self._emit("delegate", f"Could not plan worker tasks ({exc}); falling back to a single research task.", event_type="error")
            content = ""

        parsed = extract_json(content) or {}
        raw_tasks = parsed.get("tasks") or []
        tasks: list[WorkerTask] = []
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                continue
            try:
                worker_type = WorkerType(str(raw.get("worker_type", "")).strip().lower())
            except ValueError:
                continue
            description = str(raw.get("description", "")).strip()
            if not description:
                continue
            tasks.append(WorkerTask(worker_type=worker_type, description=description))

        if not tasks:
            fallback_description = decision.target or task_state.goal
            tasks = [WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description=fallback_description)]

        return tasks

    def _execute_delegate(self, decision: ActionDecision, task_state: TaskState) -> tuple[str, list[dict]]:
        """DELEGATE's execution: propose independent worker tasks, run them
        in parallel (`WorkerManager.run_parallel`), and merge their findings
        into ONE `final_text` -- returned in the exact same `(final_text,
        tool_transcript)` shape `_execute_action` returns, so `run()`'s
        caller and `_apply_updates` need no special-casing beyond
        recognizing the synthetic `"delegate_worker"` tool name below.

        Each worker's result becomes one synthetic tool_transcript entry
        (`tool="delegate_worker"`) rather than a real tool call -- this
        deliberately reuses `_apply_updates`'s EXISTING generic per-entry
        loop (files_seen merging, failed_attempts recording on a failed
        worker) instead of adding a second, parallel state-mutation path
        just for this one action.
        """
        tasks = self._propose_worker_tasks(decision, task_state)
        results = self.worker_manager.run_parallel(tasks)
        final_text = WorkerResultMerger.merge(results, task_state.goal)

        tool_transcript = [
            {
                "tool": "delegate_worker",
                "arguments": {
                    "worker_type": result.worker_type.value,
                    "target": result.task_description,
                    "task": result.task_description,
                },
                "result": {
                    "success": result.success,
                    "output": result.findings or (result.error or ""),
                    "error": result.error,
                    "data": {
                        "files": result.files,
                        "confidence": result.confidence,
                        "recommendations": result.recommendations,
                    },
                },
            }
            for result in results
        ]
        return final_text, tool_transcript

    # -- state updates -----------------------------------------------------

    def _render_task_state(self, task_state: TaskState) -> str:
        lines = [
            f"Goal: {task_state.goal}",
            f"Current phase: {task_state.current_phase}",
            f"Iteration: {task_state.iteration}",
        ]
        if task_state.files_seen:
            lines.append("Files seen: " + ", ".join(task_state.files_seen[-30:]))
        if task_state.files_modified:
            lines.append("Files modified: " + ", ".join(task_state.files_modified))
        if task_state.hypotheses:
            lines.append("Hypotheses:")
            lines.extend(f"  - [{h.status}] {h.description}" for h in task_state.hypotheses)
        if task_state.failed_attempts:
            lines.append("Failed attempts (do not blindly repeat these):")
            lines.extend(
                f"  - action={fa.action} target={fa.target!r}: {fa.error}"
                for fa in task_state.failed_attempts[-10:]
            )
        if task_state.test_results:
            last_test = task_state.test_results[-1]
            lines.append(
                f"Most recent test result: passed={last_test.get('passed')} "
                f"summary={last_test.get('summary', '')}"
            )
        if task_state.decisions:
            lines.append("Recent decisions:")
            lines.extend(f"  - {d}" for d in task_state.decisions[-10:])
        if task_state.remaining_work:
            lines.append("Remaining work:")
            lines.extend(f"  - {w}" for w in task_state.remaining_work)
        return "\n".join(lines)

    def _apply_updates(
        self,
        task_state: TaskState,
        decision: ActionDecision,
        final_text: str,
        tool_transcript: list[dict],
    ) -> None:
        task_state.tool_call_count += len(tool_transcript)
        task_state.current_phase = decision.action.value

        for entry in tool_transcript:
            tool_name = entry.get("tool")
            arguments = entry.get("arguments") or {}
            result = entry.get("result") or {}
            success = bool(result.get("success"))
            data = result.get("data") or {}

            if not success:
                target = (
                    arguments.get("path")
                    or arguments.get("pattern")
                    or arguments.get("name")
                    or arguments.get("endpoint")
                    or arguments.get("target")
                    or arguments.get("symbol")
                    or arguments.get("method")
                    or arguments.get("class_name")
                    or arguments.get("file_path")
                    or arguments.get("query")
                    or decision.target
                    or ""
                )
                task_state.failed_attempts.append(
                    FailedAttempt(action=decision.action.value, target=target, error=result.get("error") or "")
                )
                continue

            if tool_name == "read_file":
                path = arguments.get("path", "")
                content = data.get("content", "") or ""
                summary = (
                    content if len(content) <= _READ_SUMMARY_MAX_CHARS else content[:_READ_SUMMARY_MAX_CHARS] + "...(truncated)"
                )
                if path:
                    if path not in task_state.files_seen:
                        task_state.files_seen.append(path)
                    self.context.add_context("file", path, summary)

            elif tool_name in ("write_file", "edit_file"):
                path = arguments.get("path") or data.get("path", "")
                action_kind = "create" if tool_name == "write_file" else "edit"
                if path and path not in task_state.files_modified:
                    task_state.files_modified.append(path)
                task_state.changes.append(
                    {
                        "file_path": path,
                        "action": action_kind,
                        "diff": data.get("diff", ""),
                        "content": arguments.get("content") if action_kind == "create" else None,
                        "applied": True,
                    }
                )
                self.context.add_context("note", "", f"{action_kind} on {path}")

            elif tool_name == "run_command":
                task_state.test_results.append(
                    {
                        "passed": success,
                        "summary": (result.get("output") or "")[:300],
                        "command": arguments.get("command", ""),
                        "raw_output": (result.get("output") or "")[:2000],
                    }
                )

            elif tool_name == "search_files":
                matches = data.get("matches") or []
                self.context.add_context(
                    "note", "", f"search '{arguments.get('pattern', '')}' found {len(matches)} match(es)"
                )
                for match in matches[:5]:
                    path = match.get("path")
                    if path and path not in task_state.files_seen:
                        task_state.files_seen.append(path)

            elif tool_name == "find_symbol":
                symbol = data.get("symbol")
                if symbol:
                    self.context.add_context("symbol", symbol.get("name", "") or arguments.get("name", ""), symbol)
                    path = symbol.get("file")
                    if path and path not in task_state.files_seen:
                        task_state.files_seen.append(path)

            elif tool_name == "map_flow":
                chain = data.get("chain") or []
                chain_str = " -> ".join(str(hop.get("class", "")) for hop in chain)
                self.context.add_context("note", "", f"mapped flow for '{arguments.get('endpoint', '')}': {chain_str}")

            elif tool_name == "repository_tree":
                self.context.add_context("note", "", f"inspected tree under '{arguments.get('path', '.')}'")

            elif tool_name == "trace_flow":
                chain = data.get("chain") or []
                chain_str = " -> ".join(
                    f"{hop.get('class')}.{hop.get('method')}" if hop.get("class") else str(hop.get("method", ""))
                    for hop in chain
                )
                self.context.add_context("note", "", f"traced flow for '{arguments.get('target', '')}': {chain_str}")
                for hop in chain:
                    path = hop.get("file")
                    if path and path not in task_state.files_seen:
                        task_state.files_seen.append(path)

            elif tool_name in ("find_references", "find_callers"):
                edges = data.get("references") if tool_name == "find_references" else data.get("callers")
                edges = edges or []
                self.context.add_context(
                    "note", "", f"{tool_name}('{arguments.get('symbol') or arguments.get('method', '')}') "
                    f"found {len(edges)} result(s)"
                )
                for edge in edges:
                    for key in ("caller_file", "callee_file"):
                        path = edge.get(key)
                        if path and path not in task_state.files_seen:
                            task_state.files_seen.append(path)

            elif tool_name == "find_usages":
                usages = data.get("usages") or []
                self.context.add_context(
                    "note", "", f"find_usages('{arguments.get('class_name', '')}') found {len(usages)} usage(s)"
                )
                for usage in usages:
                    path = usage.get("used_in_file")
                    if path and path not in task_state.files_seen:
                        task_state.files_seen.append(path)

            elif tool_name == "explain_architecture":
                summary = str(data.get("summary") or "")
                self.context.add_context(
                    "note", "", f"explained architecture{' for ' + arguments['module'] if arguments.get('module') else ''}: "
                    f"{summary[:300]}"
                )

            elif tool_name == "find_related_tests":
                tests = data.get("tests") or []
                self.context.add_context(
                    "note", "", f"find_related_tests('{arguments.get('file_path', '')}') found {len(tests)} test file(s)"
                )
                for path in tests:
                    if path not in task_state.files_seen:
                        task_state.files_seen.append(path)

            elif tool_name == "semantic_search":
                hits = data.get("results") or []
                self.context.add_context(
                    "note", "", f"semantic_search('{arguments.get('query', '')}') found {len(hits)} result(s)"
                )
                for hit in hits[:5]:
                    path = hit.get("file")
                    if path and path not in task_state.files_seen:
                        task_state.files_seen.append(path)

            elif tool_name == "delegate_worker":
                # Phase 9: one synthetic entry per worker (see
                # _execute_delegate) -- a failed worker was already recorded
                # as a failed_attempt by the `if not success` branch above,
                # same as any other failed tool call. Deliberately does NOT
                # push worker recommendations into task_state.remaining_work
                # -- they are informational (visible to the next decision via
                # this note and via final_text/decisions below), not
                # commitments the main agent is obligated to act on; forcing
                # them into remaining_work could make a speculative or wrong
                # worker recommendation read as mandatory.
                for path in data.get("files") or []:
                    if path and path not in task_state.files_seen:
                        task_state.files_seen.append(path)
                worker_type = arguments.get("worker_type", "worker")
                self.context.add_context(
                    "note", "", f"[{worker_type}] {(result.get('output') or '')[:300]}"
                )

        log_line = f"[iter {task_state.iteration}] {decision.action.value}: {decision.rationale}"
        if final_text:
            log_line += f" -> {final_text[:200]}"
        task_state.decisions.append(log_line)

        # If a test explicitly failed with a parseable JSON verdict (per
        # _ACTION_INSTRUCTIONS' TEST format), prefer that structured
        # judgment over the raw run_command guess recorded above.
        if decision.action == ControllerAction.TEST:
            parsed_test = extract_json(final_text)
            if parsed_test and "passed" in parsed_test:
                task_state.test_results.append(
                    {
                        "passed": bool(parsed_test.get("passed")),
                        "summary": str(parsed_test.get("summary", "")),
                        "command": str(parsed_test.get("command", "")),
                        "raw_output": str(parsed_test.get("raw_output", ""))[:2000],
                    }
                )

        if decision.action == ControllerAction.PLAN:
            parsed_plan = extract_json(final_text)
            if parsed_plan and "steps" in parsed_plan:
                steps = parsed_plan.get("steps") or []
                task_state.remaining_work = [
                    str(step.get("description", "")) for step in steps if isinstance(step, dict)
                ]
                goal_restated = str(parsed_plan.get("goal", task_state.goal))
                if not any(h.description == goal_restated for h in task_state.hypotheses):
                    task_state.hypotheses.append(Hypothesis(description=goal_restated))

    # -- final summaries -----------------------------------------------

    def _build_stopped_summary(self, task_state: TaskState, reason: str) -> str:
        lines = [f"Kenzo stopped after {task_state.iteration} iteration(s): {reason}."]
        if task_state.files_modified:
            lines.append("Files modified so far: " + ", ".join(task_state.files_modified))
        if task_state.test_results:
            last_test = task_state.test_results[-1]
            lines.append(f"Most recent test result: passed={last_test.get('passed')}")
        if task_state.remaining_work:
            lines.append("Remaining work not yet addressed:")
            lines.extend(f"  - {w}" for w in task_state.remaining_work)
        return "\n".join(lines)

    def _build_final_summary(self, task_state: TaskState, rationale: str) -> str:
        last_test = task_state.test_results[-1] if task_state.test_results else None
        # Hard gate, mirroring the old pipeline's reviewer test-gate: never
        # let a FINISH read as an unqualified success if the most recent
        # test result is a known failure.
        tests_known_failing = last_test is not None and last_test.get("passed") is False

        lines = [f"Goal: {task_state.goal}"]
        if task_state.files_modified:
            lines.append("Files changed: " + ", ".join(task_state.files_modified))
        if last_test:
            status = "PASSED" if last_test.get("passed") else "FAILED"
            lines.append(f"Tests: {status} -- {last_test.get('summary', '')}")
        lines.append(f"Summary: {rationale}")

        if tests_known_failing:
            lines.insert(
                0,
                "WARNING: finishing while the most recent test result is FAILING. "
                "This task is NOT verified as working.",
            )
        return "\n".join(lines)

    def _build_result(self, task_state: TaskState) -> dict:
        last_test = task_state.test_results[-1] if task_state.test_results else None
        success = bool(last_test.get("passed")) if last_test else False
        return {
            "task_id": task_state.task_id,
            "goal": task_state.goal,
            "final_response": task_state.final_response,
            "success": success,
            "finished": task_state.finished,
            "needs_user_input": task_state.needs_user_input,
            "iteration": task_state.iteration,
            "changes": task_state.changes,
            "test_results": task_state.test_results,
            "files_modified": task_state.files_modified,
            "files_seen": task_state.files_seen,
            "decisions": task_state.decisions,
            "remaining_work": task_state.remaining_work,
            "task_state": task_state.model_dump(),
        }
