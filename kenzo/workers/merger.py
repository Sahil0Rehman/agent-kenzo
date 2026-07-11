"""`WorkerResultMerger`: combines N independent `WorkerResult`s into ONE
unified investigation summary, in the same plain-text style every other
`AgentController` action's `final_text` already uses (see `_execute_action`'s
return shape) -- so `DELEGATE`'s output slots into the existing task loop
without any special rendering elsewhere.
"""

from __future__ import annotations

from kenzo.workers.models import WorkerResult

_WORKER_LABELS: dict[str, str] = {
    "code_research": "Code research",
    "bug_analysis": "Bug analysis",
    "test": "Test discovery",
    "documentation": "Documentation lookup",
    "git": "Git history",
}


class WorkerResultMerger:
    """Stateless -- `merge` is the entire contract."""

    @staticmethod
    def merge(results: list[WorkerResult], goal: str = "") -> str:
        """Render `results` into one investigation summary. Never raises and
        never returns an empty string, even for an empty `results` list, so
        callers can always use the return value directly as an action's
        `final_text`."""
        if not results:
            return "Delegated investigation returned no results."

        header = "Unified investigation summary" + (f" for: {goal}" if goal else "") + ":"
        lines = [header]

        for result in results:
            label = _WORKER_LABELS.get(result.worker_type.value, result.worker_type.value)
            if not result.success:
                lines.append(f"\n[{label}] FAILED -- {result.error or 'no error detail'}")
                continue

            lines.append(f"\n[{label}] (confidence {result.confidence:.2f})")
            if result.task_description:
                lines.append(f"  Task: {result.task_description}")
            if result.findings:
                lines.append(f"  Findings: {result.findings}")
            if result.files:
                lines.append(f"  Files: {', '.join(result.files[:15])}")
            if result.recommendations:
                lines.append("  Recommendations:")
                lines.extend(f"    - {rec}" for rec in result.recommendations[:5])

        all_files = WorkerResultMerger.collect_files(results)
        if all_files:
            lines.append(f"\nAll files discovered across workers: {', '.join(all_files)}")

        failed = [r for r in results if not r.success]
        if failed:
            lines.append(f"\n{len(failed)}/{len(results)} worker(s) failed -- see above for detail.")

        return "\n".join(lines)

    @staticmethod
    def collect_files(results: list[WorkerResult]) -> list[str]:
        """All distinct files any SUCCESSFUL worker reported, in first-seen
        order across workers."""
        files: list[str] = []
        for result in results:
            if not result.success:
                continue
            for path in result.files:
                if path not in files:
                    files.append(path)
        return files

    @staticmethod
    def collect_recommendations(results: list[WorkerResult]) -> list[str]:
        recs: list[str] = []
        for result in results:
            if not result.success:
                continue
            for rec in result.recommendations:
                if rec not in recs:
                    recs.append(rec)
        return recs
