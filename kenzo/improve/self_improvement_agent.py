"""`SelfImprovementAgent`: Phase 10's orchestrator.

Deliberately NOT a new agent loop -- it is a thin wrapper around the
EXISTING `AgentRuntime`/`AgentController` (Phase 3) and `WorkerManager`
(Phase 9), adding exactly four things on top:

1. Pre-computed structural analysis (`kenzo.improve.analysis`) turned into
   worker tasks for the three new Phase 10 worker types.
2. The extra self-modification tool guard when the target repo IS Kenzo's
   own source (`kenzo.improve.self_repo`).
3. A regression guard wrapped around the ONE real `AgentController.run()`
   call that actually makes the change: tests before, tests after, automatic
   `checkpoint.rollback()` if the comparison fails (`kenzo.improve
   .regression_guard`).
4. Recording the outcome to `.kenzo/improvements/` (`kenzo.improve.history`).

    Understand Kenzo architecture  -> analyze() (SymbolIndex/DependencyGraph
                                       facts + ArchitectureCritic/TestGap/
                                       Performance workers, via the SAME
                                       WorkerManager.run_parallel Phase 9 built)
    Find weakness                  -> _pick_goal() (highest-severity worker
                                       recommendation, else the largest
                                       duplicate-function finding, else a
                                       large-file finding)
    Create proposal + Modify code
    + Run tests + Review changes   -> ONE AgentController.run(goal) call --
                                       the SAME PLAN/SEARCH/READ/EDIT/TEST/
                                       REVIEW/FINISH loop every other task
                                       already uses, with its own existing
                                       checkpoint/diff safety intact
    (regression guard wraps the run() call: tests before/after, rollback if
    the after count doesn't meet the before count)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from kenzo.improve.analysis import render_issue_summary, run_full_analysis
from kenzo.improve.history import ImprovementHistory, ImprovementRecord
from kenzo.improve.regression_guard import check_for_regression, run_test_suite
from kenzo.improve.self_repo import SelfModificationGuardRegistry, is_self_repo
from kenzo.workers.merger import WorkerResultMerger
from kenzo.workers.models import WorkerTask, WorkerType

# Which fact keys feed which worker type's context -- keeps each worker
# focused on the facts its own responsibility actually covers (item 3/4/5's
# separate "Checks"/"Finds"/"Analyzes" lists), rather than every worker
# seeing the entire analysis dump.
_ARCHITECTURE_FACT_KEYS = ("large_files", "dependency_cycles", "unused_modules", "duplicate_functions")
_TEST_GAP_FACT_KEYS = ("test_gaps",)
_PERFORMANCE_FACT_KEYS = ("large_files", "dependency_cycles")


def _render_facts_for_worker(facts: dict, keys: tuple[str, ...]) -> str:
    subset = {key: facts.get(key) for key in keys if facts.get(key)}
    if not subset:
        return "Pre-computed analysis found nothing in your area -- form your own opinion by reading the code directly."
    return "Pre-computed structural facts from the repository's own symbol/dependency graphs (verify before recommending, don't just repeat them):\n" + render_issue_summary(subset)


class SelfImprovementAgent:
    def __init__(self, runtime: Any, *, force_self: Optional[bool] = None) -> None:
        self.runtime = runtime
        self.repo_root = runtime.repo_path
        self.is_self = is_self_repo(self.repo_root) if force_self is None else force_self
        if self.is_self:
            # Outermost wrap -- see SelfModificationGuardRegistry's docstring
            # for why this must see (and can refuse) a mutating call before
            # checkpointing/edit-approval ever run. AgentController._execute
            # _action reads self.tools fresh on every call, so reassigning
            # it here (after AgentController.__init__ already ran) still
            # takes effect for every EDIT action from this point on.
            self.runtime.controller.tools = SelfModificationGuardRegistry(self.runtime.controller.tools)
        self.history = ImprovementHistory(self.repo_root)

    # -- 1. self analysis (no edits) ---------------------------------------

    def analyze(self) -> dict:
        """`kenzo improve --analyze`: pure structural analysis + the three
        Phase 10 workers running in parallel, in the SAME WorkerManager
        every DELEGATE action already uses. Makes NO edits."""
        indexing = self.runtime.indexing_service
        symbol_index = indexing.symbol_index if indexing else None
        dependency_graph = indexing.dependency_graph if indexing else None
        if symbol_index is None or dependency_graph is None:
            return {
                "facts": {},
                "worker_results": [],
                "plan": "No code intelligence index is available yet for this repository -- run `kenzo index` first.",
            }

        facts = run_full_analysis(symbol_index, dependency_graph, self.repo_root)
        tasks = self._build_worker_tasks(facts)
        results = self.runtime.controller.worker_manager.run_parallel(tasks) if tasks else []
        plan = self._render_plan(facts, results)
        return {"facts": facts, "worker_results": [r.model_dump() for r in results], "plan": plan}

    def _build_worker_tasks(self, facts: dict) -> list[WorkerTask]:
        tasks = [
            WorkerTask(
                worker_type=WorkerType.ARCHITECTURE_CRITIC,
                description=(
                    "Review the repository's architecture for bad abstractions, "
                    "large files, dependency cycles, unused modules, and technical debt."
                ),
                context=_render_facts_for_worker(facts, _ARCHITECTURE_FACT_KEYS),
            ),
            WorkerTask(
                worker_type=WorkerType.TEST_GAP,
                description="Identify untested modules, weak tests, and missing integration coverage.",
                context=_render_facts_for_worker(facts, _TEST_GAP_FACT_KEYS),
            ),
            WorkerTask(
                worker_type=WorkerType.PERFORMANCE,
                description=(
                    "Analyze the repository's largest/most-depended-on modules for "
                    "likely slow paths, expensive indexing, or cache misses."
                ),
                context=_render_facts_for_worker(facts, _PERFORMANCE_FACT_KEYS),
            ),
        ]
        return tasks

    def _render_plan(self, facts: dict, results: list) -> str:
        parts = [render_issue_summary(facts)]
        if results:
            parts.append("")
            parts.append(WorkerResultMerger.merge(results))
        return "\n".join(parts)

    def _pick_goal(self, analysis: dict) -> str:
        """The single improvement to actually attempt -- priority: a
        critical/high-severity worker recommendation, else the largest
        duplicate-function finding (concrete, safe, easy to verify with
        tests), else any worker recommendation, else a large-file finding,
        else a generic fallback. Never returns an empty goal."""
        for result in analysis["worker_results"]:
            if result.get("severity") in ("critical", "high") and result.get("recommendations"):
                return f"{result['recommendations'][0]} (context: {result.get('findings', '')[:300]})"

        facts = analysis.get("facts") or {}
        if facts.get("duplicate_functions"):
            dup = facts["duplicate_functions"][0]
            locations = ", ".join(f"{loc['file']}:{loc['name']}" for loc in dup["locations"])
            return (
                f"Remove the duplicate logic found at {locations} (identical "
                f"{dup['line_count']}-line implementations) by consolidating it into "
                "a single shared implementation, keeping behavior identical, and "
                "update all call sites accordingly."
            )

        for result in analysis["worker_results"]:
            if result.get("recommendations"):
                return result["recommendations"][0]

        if facts.get("large_files"):
            large = facts["large_files"][0]
            return f"Investigate whether {large['file']} ({large['lines']} lines) can be reasonably split or simplified, without changing behavior."

        return "Investigate the codebase for one safe, well-scoped improvement and make it, verifying with tests."

    # -- 2. full improve workflow (with the regression guard) -------------

    def improve(self, request: Optional[str] = None) -> dict:
        """`kenzo improve`: analyze -> pick a weakness -> ONE real
        AgentController.run() (its own checkpoint/diff safety intact) ->
        regression guard (tests before/after) -> automatic rollback if it
        regressed -> record to `.kenzo/improvements/`.
        """
        analysis = self.analyze()
        goal = (request or "").strip() or self._pick_goal(analysis)

        before_tests = run_test_suite(self.repo_root, self.runtime.project_info, self.is_self)
        agent_result = self.runtime.controller.run(goal)
        after_tests = run_test_suite(self.repo_root, self.runtime.project_info, self.is_self)
        check = check_for_regression(before_tests, after_tests)

        rolled_back = False
        rollback_message = ""
        if check.regressed:
            rollback_result = self.runtime.controller.checkpoint.rollback()
            rolled_back = bool(rollback_result.get("success"))
            rollback_message = rollback_result.get("message", "")

        overall_success = bool(agent_result.get("success")) and not check.regressed

        checkpoint_record = self.runtime.controller.checkpoint.load_latest() or {}
        record = ImprovementRecord(
            timestamp=datetime.now().isoformat(),
            problem=goal,
            solution=agent_result.get("final_response") or "",
            files_changed=list(agent_result.get("files_modified") or []),
            tests_before={"regressed_baseline": False, "runs": [_test_run_to_dict(r) for r in before_tests]},
            tests_after={"runs": [_test_run_to_dict(r) for r in after_tests]},
            success=overall_success,
            rolled_back=rolled_back,
            rollback_reason=check.reason if check.regressed else "",
            checkpoint_ref=checkpoint_record.get("git_ref") or checkpoint_record.get("checkpoint_dir"),
            task_id=agent_result.get("task_id", ""),
        )
        history_path = self.history.record(record)

        # Reuses the SAME session/architecture memory bookkeeping every
        # in-process AgentController caller gets (see AgentRuntime
        # .record_task_outcome, and kenzo.server.session's Phase 8 fix that
        # gave the server path the same recording) -- "Uses: ... Memory"
        # per the spec, via the existing infrastructure rather than a
        # bespoke parallel memory path.
        if hasattr(self.runtime, "record_task_outcome"):
            self.runtime.record_task_outcome(goal, agent_result)

        return {
            "analysis": analysis,
            "goal": goal,
            "agent_result": agent_result,
            "regression_check": {
                "regressed": check.regressed,
                "reason": check.reason,
                "before": [_test_run_to_dict(r) for r in before_tests],
                "after": [_test_run_to_dict(r) for r in after_tests],
            },
            "rolled_back": rolled_back,
            "rollback_message": rollback_message,
            "success": overall_success,
            "history_path": str(history_path),
        }


def _test_run_to_dict(result: Any) -> dict:
    return {
        "command": result.command,
        "cwd": result.cwd,
        "ran": result.ran,
        "passed": result.passed,
        "failed": result.failed,
        "error": result.error,
    }
