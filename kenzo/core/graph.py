"""LangGraph wiring: assembles the planner -> researcher -> coder -> tester ->
reviewer pipeline into a single compiled graph.

`build_graph` is the only place that knows the shape of the pipeline; the
review/retry loop (reviewer -> coder on rejection, reviewer -> END on
approval or after MAX_REVIEW_RETRIES) lives here via `_review_router`.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from kenzo.agents.coder import build_coder_node
from kenzo.agents.planner import build_planner_node
from kenzo.agents.researcher import build_researcher_node
from kenzo.agents.reviewer import build_reviewer_node
from kenzo.agents.tester import build_tester_node
from kenzo.core.events import AgentEvent, EventSink
from kenzo.core.schemas import MAX_REVIEW_RETRIES, KenzoState

# Per-role tool access. Every agent shared one unrestricted registry before
# this: any of Planner/Researcher/Tester/Reviewer could call write_file or
# edit_file directly, with zero record of it anywhere (only the Coder node
# ever converts a tool call into a CodeChange). Restricting each role's
# registry to exactly what its own system prompt says it needs closes that
# gap at the tool-dispatch layer itself -- a role without write_file/edit_file
# in its subset cannot execute them even if a model hallucinates the call.
#
# Phase 2 adds repository_tree (bounded recursive orientation, replacing
# one-level-at-a-time list_directory calls) and the Java symbol tools
# (find_symbol, map_flow) to the read-only/exploration roles. Coder also
# regains search_files/repository_tree (Phase 1.5 had removed all search
# capability from Coder, leaving it unable to recover when Research missed a
# file -- see the Phase 1.5 capability benchmark) while still never getting
# terminal or git-write access.
#
# Phase 4 adds the AST-backed code intelligence tools (trace_flow,
# find_references, find_callers, find_usages, explain_architecture) to the
# same read-only/exploration roles as find_symbol/map_flow -- they are a more
# accurate alternative to those, not a replacement, so both stay available.
# find_related_tests goes to Tester only (see kenzo.agents.tester), matching
# its actual use.
#
# Phase 5 adds semantic_search (hybrid symbol/graph/vector/keyword search --
# see kenzo.tools.semantic_search) to the same roles; it only actually
# registers when AgentRuntime was built with an IndexingService (see
# kenzo.tools.registry.build_default_registry), so listing it here is a no-op
# for any registry built without one.
_READ_ONLY_TOOLS = [
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
    "status",
    "diff",
    "branch",
    "history",
]
_CODER_TOOLS = [
    "read_file",
    "list_directory",
    "search_files",
    "repository_tree",
    "write_file",
    "edit_file",
]
_TESTER_TOOLS = ["read_file", "run_command", "find_related_tests"]

ROLE_TOOLS: dict[str, list[str]] = {
    "planner": _READ_ONLY_TOOLS,
    "researcher": _READ_ONLY_TOOLS,
    "coder": _CODER_TOOLS,
    "tester": _TESTER_TOOLS,
    # Reviewer must never modify code: read-only tools only, same as
    # planner/researcher -- it may double-check via read_file/git diff but
    # has no path to write_file/edit_file.
    "reviewer": _READ_ONLY_TOOLS,
}


def _with_events(node_name: str, node_fn: Any, on_event: EventSink | None) -> Any:
    """Wrap a node callable so it emits start/finish events around invocation.

    If `on_event` is None, the wrapped callable just calls `node_fn(state)`
    directly with no overhead.
    """
    if on_event is None:
        return node_fn

    def wrapped(state: dict) -> dict:
        on_event(AgentEvent(stage=node_name, message=f"{node_name} starting"))
        result = node_fn(state)
        on_event(AgentEvent(stage=node_name, message=f"{node_name} finished"))
        return result

    return wrapped


def _review_router(state: dict) -> str:
    """Decide whether to retry the coder or finish, based on review state."""
    review = state.get("review") or {}
    if review.get("approved") or state.get("review_retries", 0) >= MAX_REVIEW_RETRIES:
        return "done"
    return "retry"


def build_graph(router: Any, tools: Any, memory: Any, on_event: EventSink | None = None) -> Any:
    """Build and compile the Kenzo agent pipeline as a LangGraph graph."""
    graph = StateGraph(KenzoState)

    def _scoped(role: str) -> Any:
        return tools.subset(ROLE_TOOLS[role])

    graph.add_node(
        "planner", _with_events("planner", build_planner_node(router, _scoped("planner"), memory), on_event)
    )
    graph.add_node(
        "researcher",
        _with_events("researcher", build_researcher_node(router, _scoped("researcher"), memory), on_event),
    )
    graph.add_node(
        "coder", _with_events("coder", build_coder_node(router, _scoped("coder"), memory), on_event)
    )
    graph.add_node(
        "tester", _with_events("tester", build_tester_node(router, _scoped("tester"), memory), on_event)
    )
    graph.add_node(
        "reviewer",
        _with_events("reviewer", build_reviewer_node(router, _scoped("reviewer"), memory), on_event),
    )

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "coder")
    graph.add_edge("coder", "tester")
    graph.add_edge("tester", "reviewer")
    graph.add_conditional_edges("reviewer", _review_router, {"retry": "coder", "done": END})

    return graph.compile()
