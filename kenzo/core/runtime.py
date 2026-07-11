"""Top-level runtime entry point that wires config, memory, tools, the model
router, and the compiled LangGraph pipeline together for a single repository.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from kenzo.config.settings import KenzoConfig, load_config
from kenzo.core.controller import AgentController
from kenzo.core.events import EventSink
from kenzo.core.graph import build_graph
from kenzo.core.indexing_service import IndexingService
from kenzo.core.project import ProjectAnalyzer
from kenzo.core.schemas import new_state
from kenzo.core.workspace import Workspace
from kenzo.mcp.client import MCPClient
from kenzo.memory.architecture_memory import ArchitectureMemory, MemoryEntry
from kenzo.memory.importance import is_worth_remembering
from kenzo.memory.java_analyzer import analyze_java_repository
from kenzo.memory.repository import RepositoryMemory
from kenzo.memory.session_memory import SessionEntry, SessionMemory
from kenzo.models.router import ModelRouter
from kenzo.tools.registry import build_default_registry


class AgentRuntime:
    """Owns everything needed to drive a single request through the Kenzo
    agent pipeline for one repository."""

    def __init__(
        self,
        repo_path: str,
        config: Optional[KenzoConfig] = None,
        on_event: Optional[EventSink] = None,
        ide_mode: bool = False,
        workspace: Optional[Workspace] = None,
    ) -> None:
        self.config = config or load_config()
        self.repo_path = str(Path(repo_path).resolve())
        self.memory = RepositoryMemory(self.repo_path)
        self.memory.ensure_kenzo_dir()
        self.router = ModelRouter(self.config)
        self.on_event = on_event
        # Phase 11: other repos this run can see/edit -- see
        # kenzo.core.workspace. None (the default) means single-repo,
        # preserving exact prior behavior for every pre-Phase-11 caller.
        self.workspace = workspace

        # Repository intelligence: detect what kind of project this is (so
        # Tester stops assuming Python) and, for Java, extract a lightweight
        # symbol table -- both persisted so every agent's context bundle
        # (RepositoryMemory.build_context_bundle) picks them up automatically.
        self.project_info = ProjectAnalyzer().analyze(self.repo_path)
        self.memory.save_project_info(self.project_info.model_dump())
        if self.project_info.language == "java":
            symbols = analyze_java_repository(self.repo_path, self.config.indexing.ignored_dirs)
            self.memory.save_symbols(symbols)

        # Phase 4/5: full AST-derived code intelligence (all 4 supported
        # languages) plus the semantic index (embeddings + vector store) on
        # top of it, owned by one `IndexingService` for the lifetime of this
        # runtime. `.start(background=True)` loads whatever `.kenzo/` already
        # has persisted immediately (fast) and refreshes it on a background
        # thread -- construction here never blocks on a full repo walk, no
        # matter how large the repository is. See `IndexingService`'s module
        # docstring for the thread-safety model.
        self.indexing_service = IndexingService(self.repo_path, self.memory, self.config)
        self.indexing_service.start(background=True)

        # Phase 6: connect every configured MCP server, synchronously (unlike
        # IndexingService above, MCP tool AVAILABILITY must be known before
        # AgentController can decide whether to offer them for a given
        # action -- see kenzo.core.controller's dynamic MCP tool-source
        # scoping). connect_all() never raises: an offline/misconfigured
        # server just contributes no tools, the others are unaffected, and a
        # config with zero servers configured (the default) connects
        # instantly and yields zero MCP tools -- native tool availability is
        # completely unaffected either way.
        self.mcp_client = MCPClient(self.config.mcp)
        self.mcp_client.connect_all()

        self.tools = build_default_registry(
            self.repo_path,
            self.config.permissions,
            self.config.indexing,
            self.indexing_service,
            self.mcp_client,
            workspace=self.workspace,
        )

        # Phase 5: cross-session (this process's lifetime) and long-term
        # (persisted, cross-restart) memory -- see kenzo.memory.session_memory
        # / kenzo.memory.architecture_memory. One instance per AgentRuntime,
        # same lifetime rationale as `self.controller` below.
        self.session_memory = SessionMemory(self.repo_path)
        self.architecture_memory = ArchitectureMemory(self.repo_path)

        self.graph = build_graph(self.router, self.tools, self.memory, on_event)

        # Phase 3: the dynamic Agent Controller loop, an ADDITIVE alternative
        # to the fixed pipeline above -- see `run_autonomous`. One instance
        # per AgentRuntime (not per call), so its ContextManager/checkpoint
        # state persists across multiple turns of the same interactive
        # session, the same way a human's own accumulated understanding of a
        # repo would.
        self.controller = AgentController(
            self.router,
            self.tools,
            self.memory,
            self.config,
            on_event,
            indexing_service=self.indexing_service,
            session_memory=self.session_memory,
            architecture_memory=self.architecture_memory,
            mcp_client=self.mcp_client,
            ide_mode=ide_mode,
            workspace=self.workspace,
        )

    def close(self) -> None:
        """Best-effort cleanup: disconnect every MCP server (terminates
        their subprocesses / closes their HTTP connections cleanly). Safe to
        call even if no MCP servers were configured (no-op). Not required
        for correctness -- `MCPServerConnection`'s background thread is a
        daemon thread, so the process still exits cleanly without this --
        but calling it avoids leaving spawned MCP server subprocesses
        orphaned longer than necessary during a long interactive session.
        """
        self.mcp_client.disconnect_all()

    def run(self, request: str) -> dict:
        """Drive `request` through the full planner -> researcher -> coder ->
        tester -> reviewer pipeline and return the final state dict.

        This is the ONLY entry point for driving a request through the agent
        pipeline -- there is deliberately no shortcut that lets a caller get
        a direct answer without going through the full
        planner -> researcher -> coder -> tester -> reviewer sequence.

        Failure safety: this method never lets an unhandled node exception
        (e.g. a transient LLM/API error) propagate and silently discard
        whatever the pipeline had already done. Nodes run via `graph.stream`
        instead of `graph.invoke` so state is accumulated locally as each
        node finishes; if a later node raises, the returned dict still
        reflects every completed step -- including any real files the Coder
        already wrote to disk -- with `crashed=True` and `error` set, instead
        of raising and losing that information. Callers (the CLI) must check
        `result["crashed"]` and warn the user if `result["changes"]` is
        non-empty, since those changes were NOT reverted.
        """
        initial = new_state(request, self.repo_path)
        accumulated: dict = dict(initial)
        completed_steps: list[str] = []

        try:
            for step in self.graph.stream(initial, config={"recursion_limit": 25}, stream_mode="updates"):
                for node_name, partial in step.items():
                    accumulated.update(partial)
                    completed_steps.append(node_name)
        except Exception as exc:  # noqa: BLE001 - runtime safety boundary: never hide partial disk changes
            accumulated["error"] = f"{type(exc).__name__}: {exc}"
            accumulated["crashed"] = True
            accumulated["completed_steps"] = completed_steps
            return accumulated

        accumulated["completed_steps"] = completed_steps
        accumulated["crashed"] = False

        review = accumulated.get("review") or {}
        if review.get("approved"):
            self._record_completed_task(accumulated, request)

        return accumulated

    def run_autonomous(self, request: str, task_id: Optional[str] = None) -> dict:
        """Drive `request` through the Phase 3 dynamic Agent Controller loop
        (see `kenzo.core.controller.AgentController`) instead of the fixed
        pipeline. This is what the interactive `kenzo` session uses by
        default now -- `run()` above is untouched and still fully available
        for anything that wants the original fixed pipeline's exact,
        deterministic five-stage behavior.

        Unlike `run()`, this never raises for a mid-run model/tool error --
        `AgentController.run` already returns a well-formed result dict in
        every case (including budget exhaustion) -- and it records a memory
        entry on success exactly like the pipeline's `run()` does.
        """
        result = self.controller.run(request, task_id=task_id)
        self.record_task_outcome(request, result)
        return result

    def record_task_outcome(self, request: str, result: dict) -> None:
        """Session + architecture memory bookkeeping for one completed
        `AgentController.run()` result. Extracted out of `run_autonomous` so
        `kenzo.server.session.AgentSession` -- which calls `self.controller
        .run()` directly (bridging it into the async WebSocket world, not
        going through `run_autonomous`) -- can perform the exact same
        recording. Without this, every server-driven task (the VS Code
        extension's only path) would silently skip session/architecture
        memory entirely, including Phase 8's vision-fix recording below.
        """
        succeeded = bool(result.get("success"))

        # Phase 5: session memory records EVERY completed task, success or
        # not -- its own hygiene gate (kenzo.memory.importance) decides what
        # actually survives long-term (see SessionMemory.record); this is
        # what lets a later, vaguer request in the same session ("fix the
        # same issue in admin login") retrieve what an earlier task did.
        files_discovered = list(dict.fromkeys((result.get("files_modified") or []) + (result.get("files_seen") or [])))
        self.session_memory.record(
            SessionEntry(
                request=request,
                task_id=result.get("task_id", "") or "",
                succeeded=succeeded,
                summary=(result.get("final_response") or "")[:500],
                files_discovered=files_discovered[:10],
                timestamp=datetime.now().isoformat(),
            )
        )

        if succeeded:
            self._record_completed_task(
                {
                    "plan": {"goal": result.get("goal", request)},
                    "research": {"relevant_files": result.get("files_seen", [])},
                    "changes": result.get("changes", []),
                },
                request,
                summary=result.get("final_response") or "",
            )

    def _record_completed_task(self, state: dict, request: str, summary: str = "") -> None:
        """Append a short, noise-free summary of a successfully approved run
        to `.kenzo/memory.md`, so future runs' context bundle reflects what
        was actually done -- what changed, which files mattered, and what
        the research step discovered -- rather than nothing at all.

        Also (Phase 5) considers the same completion for `ArchitectureMemory`
        -- structured, long-term, cross-restart project knowledge -- gated by
        `kenzo.memory.importance.is_worth_remembering` so routine/trivial
        completions don't bloat it the way every completed task bloating
        memory.md would be undesirable too, if that log weren't already
        bounded by `RepositoryMemory.build_context_bundle`'s own tail-only
        rendering.
        """
        plan = state.get("plan") or {}
        research = state.get("research") or {}
        changes = state.get("changes") or []

        goal = (plan.get("goal") or request or "").strip()
        changed_files = sorted({c.get("file_path", "") for c in changes if c.get("file_path")})
        relevant_files = [f for f in (research.get("relevant_files") or []) if f][:10]

        lines: list[str] = []
        if goal:
            lines.append(f"Completed: {goal}")
        if changed_files:
            lines.append("Files changed: " + ", ".join(changed_files))
        if relevant_files:
            lines.append("Related files discovered: " + ", ".join(relevant_files))
        if not lines:
            return  # nothing worth recording

        heading = f"Completed: {goal[:80]}" if goal else "Completed task"
        self.memory.append_memory("\n".join(lines), heading=heading)

        # Phase 8: a task driven by an attached screenshot (kenzo image /
        # kenzo.server's attach_image handling -- see AgentController.set_
        # vision_context) is recorded with a DEDICATED kind, "vision_fix",
        # and its own richer issue/root_cause fields sourced from the
        # analysis, rather than the generic module_note path below -- so a
        # LATER similar screenshot's derived request (which tends to reuse
        # the same wording, e.g. "white page after login") can retrieve this
        # fix via ArchitectureMemory.find_related's plain keyword overlap,
        # per the Phase 8 spec's "similar screenshot retrieves fix."
        vision_context = self.controller.vision_context
        if vision_context and is_worth_remembering("vision_fix", f"{goal} {summary}", succeeded=True):
            possible_causes = vision_context.get("possible_causes") or []
            self.architecture_memory.record(
                MemoryEntry(
                    kind="vision_fix",
                    issue=(vision_context.get("primary_error") or goal)[:200],
                    root_cause=possible_causes[0][:300] if possible_causes else "",
                    files=changed_files,
                    solution=summary[:500],
                    notes=f"Detected from a {vision_context.get('image_type', 'screenshot')} screenshot.",
                    timestamp=datetime.now().isoformat(),
                )
            )
        elif is_worth_remembering("module_note", f"{goal} {summary}", succeeded=True):
            self.architecture_memory.record(
                MemoryEntry(
                    kind="module_note",
                    issue=goal[:200],
                    files=changed_files,
                    solution=summary[:500],
                    timestamp=datetime.now().isoformat(),
                )
            )
