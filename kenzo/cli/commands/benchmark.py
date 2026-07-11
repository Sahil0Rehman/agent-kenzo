"""`kenzo benchmark` -- Phase 10 item 9: measure indexing speed, search
quality, agent-loop framework overhead, and worker-loop framework overhead,
tracked over time under `.kenzo/benchmarks/`.

Honesty note (also printed by the command itself): this environment has no
real LLM API call available for a benchmark run to make (every other phase's
tests/demos in this project use a scripted/mocked `ModelRouter` for exactly
this reason). "Agent latency"/"worker speed" below therefore measure this
project's own FRAMEWORK overhead around a model call (prompt assembly,
context retrieval, thread/tool-subset setup) with the model call itself
mocked to return instantly -- NOT real end-to-end latency, which is
dominated by whichever provider/model you actually configure. Indexing speed
and search quality, by contrast, involve no model call at all and are
genuinely measured end to end.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from kenzo.cli.ui.console import console
from kenzo.config.settings import load_config
from kenzo.core.controller import AgentController
from kenzo.core.runtime import AgentRuntime
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.registry import build_default_registry
from kenzo.workers.models import WorkerTask, WorkerType

app = typer.Typer(help="Benchmark Kenzo's own indexing/search/agent-loop performance, tracked over time.")


class _InstantRouter:
    """Returns a final-text answer immediately, no tool calls -- isolates
    FRAMEWORK overhead (prompt assembly, retrieval, thread setup) from
    real LLM API latency, which this environment has no access to."""

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        class _Msg:
            content = None
            tool_calls = None

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        if isinstance(tool_choice, dict) and tool_choice.get("function", {}).get("name") == "decide_next_action":
            import json as _json

            class _Fn:
                name = "decide_next_action"
                arguments = _json.dumps({"action": "finish", "rationale": "benchmark", "target": ""})

            class _Call:
                id = "bench-call"
                function = _Fn()

            resp = _Resp()
            resp.choices[0].message.tool_calls = [_Call()]
            return resp

        resp = _Resp()
        resp.choices[0].message.content = json.dumps(
            {"findings": "benchmark", "confidence": 0.5, "files": [], "recommendations": []}
        )
        return resp

    def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
        return json.dumps({"tasks": []})


def _benchmark_indexing(repo_root: str, config: Any) -> dict:
    from kenzo.intelligence.dependency_graph import DependencyGraph
    from kenzo.intelligence.reference_graph import ReferenceGraph
    from kenzo.intelligence.symbol_index import SymbolIndex

    start = time.monotonic()
    symbol_index = SymbolIndex(repo_root)
    symbol_index.build(ignored_dirs=config.indexing.ignored_dirs, max_files=config.indexing.max_indexed_files)
    symbols_elapsed = time.monotonic() - start

    start = time.monotonic()
    reference_graph = ReferenceGraph(symbol_index)
    reference_graph.build()
    refs_elapsed = time.monotonic() - start

    start = time.monotonic()
    dependency_graph = DependencyGraph(symbol_index)
    dependency_graph.build()
    deps_elapsed = time.monotonic() - start

    file_count = len(symbol_index.files)
    total = symbols_elapsed + refs_elapsed + deps_elapsed
    return {
        "files_indexed": file_count,
        "symbol_build_seconds": round(symbols_elapsed, 4),
        "reference_graph_seconds": round(refs_elapsed, 4),
        "dependency_graph_seconds": round(deps_elapsed, 4),
        "total_seconds": round(total, 4),
        "files_per_second": round(file_count / total, 2) if total > 0 else 0.0,
    }, symbol_index


def _benchmark_search_quality(symbol_index: Any, sample_size: int = 20) -> dict:
    """Recall self-check: sample real class names the index itself just
    extracted, then confirm a plain linear find-by-name recovers each one --
    a genuine, reproducible quality signal for ANY repo (not a Kenzo-
    specific canned query list), independent of any model call."""
    all_classes = symbol_index.all_classes()
    sample = all_classes[:: max(1, len(all_classes) // sample_size)][:sample_size] if all_classes else []

    start = time.monotonic()
    hits = 0
    for _file_path, cls in sample:
        found = any(c.name == cls.name for _fp, c in symbol_index.all_classes())
        if found:
            hits += 1
    elapsed = time.monotonic() - start

    recall = round(hits / len(sample), 4) if sample else 1.0
    return {
        "sample_size": len(sample),
        "hits": hits,
        "recall": recall,
        "total_seconds": round(elapsed, 4),
        "avg_query_ms": round((elapsed / len(sample)) * 1000, 3) if sample else 0.0,
    }


def _benchmark_agent_loop(runtime: AgentRuntime, iterations: int = 5) -> dict:
    """Time `_decide_next_action`'s prompt assembly + retrieval with the
    model call itself mocked to return instantly -- see module docstring."""
    controller: AgentController = runtime.controller
    original_router = controller.router
    controller.router = _InstantRouter()
    try:
        from kenzo.core.schemas import TaskState

        durations = []
        for i in range(iterations):
            task_state = TaskState(task_id=f"bench-{i}", goal="Benchmark decision latency", repo_path=runtime.repo_path)
            start = time.monotonic()
            controller._decide_next_action(task_state)
            durations.append(time.monotonic() - start)
    finally:
        controller.router = original_router

    return {
        "iterations": iterations,
        "avg_seconds": round(sum(durations) / len(durations), 4),
        "min_seconds": round(min(durations), 4),
        "max_seconds": round(max(durations), 4),
    }


def _benchmark_worker_speed(runtime: AgentRuntime, worker_count: int = 3) -> dict:
    controller: AgentController = runtime.controller
    original_router = controller.worker_manager.router
    controller.worker_manager.router = _InstantRouter()
    try:
        tasks = [
            WorkerTask(worker_type=WorkerType.CODE_RESEARCH, description=f"benchmark task {i}")
            for i in range(worker_count)
        ]
        start = time.monotonic()
        results = controller.worker_manager.run_parallel(tasks)
        elapsed = time.monotonic() - start
    finally:
        controller.worker_manager.router = original_router

    return {
        "worker_count": worker_count,
        "total_seconds": round(elapsed, 4),
        "all_succeeded": all(r.success for r in results),
    }


def _benchmarks_dir(repo_root: str) -> Path:
    return Path(repo_root) / ".kenzo" / "benchmarks"


@app.callback(invoke_without_command=True)
def run(
    ctx: typer.Context,
    repo: str = typer.Option(".", "--repo", "-r", help="Repository to benchmark."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    console.print("[kenzo.title]Kenzo Benchmark[/kenzo.title]")
    console.print(
        "[dim]No real LLM API call is made -- 'agent latency'/'worker speed' measure this project's own "
        "framework overhead with the model call mocked to return instantly, not real provider latency.[/dim]\n"
    )

    config = load_config()
    memory = RepositoryMemory(repo)
    memory.ensure_kenzo_dir()
    tools = build_default_registry(repo, config.permissions, config.indexing)

    console.print("Indexing...")
    indexing_result, symbol_index = _benchmark_indexing(repo, config)

    console.print("Search quality...")
    search_result = _benchmark_search_quality(symbol_index)

    console.print("Agent loop + worker speed...")
    runtime = AgentRuntime(repo, config=config)
    try:
        agent_result = _benchmark_agent_loop(runtime)
        worker_result = _benchmark_worker_speed(runtime)
    finally:
        runtime.close()

    record = {
        "timestamp": datetime.now().isoformat(),
        "repo": str(Path(repo).resolve()),
        "indexing": indexing_result,
        "search_quality": search_result,
        "agent_loop": agent_result,
        "worker_speed": worker_result,
    }

    _render_results(record)

    bench_dir = _benchmarks_dir(repo)
    bench_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (bench_dir / f"{stamp}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    previous = _load_previous(bench_dir, stamp)
    if previous:
        _render_comparison(previous, record)

    console.print(f"[dim]Saved to {bench_dir / f'{stamp}.json'}[/dim]")


def _render_results(record: dict) -> None:
    table = Table(title="Benchmark Results")
    table.add_column("Metric")
    table.add_column("Value")
    idx = record["indexing"]
    table.add_row("Files indexed", str(idx["files_indexed"]))
    table.add_row("Indexing total", f"{idx['total_seconds']}s ({idx['files_per_second']} files/s)")
    sq = record["search_quality"]
    table.add_row("Search recall", f"{sq['recall'] * 100:.1f}% ({sq['hits']}/{sq['sample_size']})")
    table.add_row("Avg query time", f"{sq['avg_query_ms']}ms")
    al = record["agent_loop"]
    table.add_row("Agent decision (framework overhead)", f"{al['avg_seconds']}s avg over {al['iterations']} iterations")
    ws = record["worker_speed"]
    table.add_row("Worker parallel run (framework overhead)", f"{ws['total_seconds']}s for {ws['worker_count']} workers")
    console.print(table)


def _load_previous(bench_dir: Path, current_stamp: str) -> dict | None:
    files = sorted(f for f in bench_dir.glob("*.json") if f.stem != current_stamp)
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pct_change(before: float, after: float) -> str:
    if before == 0:
        return "n/a"
    change = ((after - before) / before) * 100
    arrow = "faster" if change < 0 else "slower"
    return f"{abs(change):.1f}% {arrow}"


def _render_comparison(previous: dict, current: dict) -> None:
    table = Table(title=f"Change since {previous.get('timestamp', 'previous run')}")
    table.add_column("Metric")
    table.add_column("Before")
    table.add_column("After")
    table.add_column("Change")

    pairs = [
        ("Indexing total (s)", previous["indexing"]["total_seconds"], current["indexing"]["total_seconds"]),
        ("Search recall", previous["search_quality"]["recall"], current["search_quality"]["recall"]),
        ("Agent decision (s)", previous["agent_loop"]["avg_seconds"], current["agent_loop"]["avg_seconds"]),
        ("Worker run (s)", previous["worker_speed"]["total_seconds"], current["worker_speed"]["total_seconds"]),
    ]
    for label, before, after in pairs:
        change = _pct_change(before, after) if "recall" not in label.lower() else f"{(after - before) * 100:+.1f}pp"
        table.add_row(label, str(before), str(after), change)
    console.print(table)


@app.command("history")
def history_cmd(repo: str = typer.Option(".", "--repo", "-r"), limit: int = typer.Option(10, "--limit")) -> None:
    """Show past `kenzo benchmark` runs."""
    bench_dir = _benchmarks_dir(repo)
    if not bench_dir.is_dir():
        console.print("[dim]No benchmark history yet -- run `kenzo benchmark` first.[/dim]")
        return
    files = sorted(bench_dir.glob("*.json"))[-limit:]
    table = Table(title=f"Benchmark History ({len(files)} shown)")
    table.add_column("Timestamp")
    table.add_column("Files")
    table.add_column("Indexing (s)")
    table.add_column("Search Recall")
    for file_path in files:
        try:
            record = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        table.add_row(
            record.get("timestamp", ""),
            str(record.get("indexing", {}).get("files_indexed", "")),
            str(record.get("indexing", {}).get("total_seconds", "")),
            f"{record.get('search_quality', {}).get('recall', 0) * 100:.1f}%",
        )
    console.print(table)
