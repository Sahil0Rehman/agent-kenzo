"""`kenzo index` -- walk a repository, build its lightweight content index,
persist it to `.kenzo/index.json`, and summarize the result.
"""

from __future__ import annotations

import typer
from rich.table import Table

from kenzo.cli.ui.console import console
from kenzo.config.settings import load_config
from kenzo.core.project import ProjectAnalyzer
from kenzo.intelligence.dependency_graph import DependencyGraph
from kenzo.intelligence.reference_graph import ReferenceGraph
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.memory.indexer import index_repository
from kenzo.memory.java_analyzer import analyze_java_repository
from kenzo.memory.repository import RepositoryMemory


def run(path: str = typer.Argument(".", help="Repository path to index")) -> None:
    """Index a repository, detect its project type, extract a lightweight
    Java symbol table (Java repositories only), and build the Phase 4 code
    intelligence artifacts (symbol index, reference graph, dependency graph
    -- Java/Python/JavaScript/TypeScript) -- persisting all of it to `.kenzo/`.
    """
    config = load_config()
    ignored_dirs = config.indexing.ignored_dirs

    result = index_repository(path, max_files=config.indexing.max_indexed_files, ignored_dirs=ignored_dirs)

    memory = RepositoryMemory(path)
    memory.save_index(result)

    project_info = ProjectAnalyzer().analyze(path)
    memory.save_project_info(project_info.model_dump())

    symbol_count = 0
    if project_info.language == "java":
        symbols = analyze_java_repository(path, ignored_dirs)
        memory.save_symbols(symbols)
        symbol_count = len(symbols.get("symbols", []))

    code_index = SymbolIndex(path)
    code_index.build(ignored_dirs=ignored_dirs, max_files=config.indexing.max_indexed_files)
    code_index.save(str(memory.code_index_path))

    reference_graph = ReferenceGraph(code_index)
    reference_graph.build()
    reference_graph.save(str(memory.references_path))

    dependency_graph = DependencyGraph(code_index)
    dependency_graph.build()
    dependency_graph.save(str(memory.dependencies_path))

    code_stats = code_index.stats()

    languages: dict[str, int] = result.get("languages", {}) or {}
    top_languages = sorted(languages.items(), key=lambda kv: kv[1], reverse=True)[:8]
    indexed_files = result.get("indexed_files", result.get("file_count", 0))
    total_files = result.get("total_files", indexed_files)

    table = Table(title="Repository Index")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Language", project_info.language)
    table.add_row("Framework", project_info.framework or "(none detected)")
    table.add_row("Build system", project_info.build or "(none detected)")
    table.add_row("Test command", project_info.test_command or "(unknown)")
    table.add_row("Indexed files", f"{indexed_files} of {total_files}")
    table.add_row(
        "Truncated",
        f"[yellow]true -- {total_files - indexed_files} file(s) not indexed[/yellow]"
        if result.get("truncated")
        else "false",
    )
    table.add_row(
        "Top languages",
        ", ".join(f"{ext or '(no ext)'} ({count})" for ext, count in top_languages) or "(none)",
    )
    table.add_row("Summary", result.get("summary", ""))
    if project_info.language == "java":
        table.add_row("Java symbols indexed", str(symbol_count))

    code_langs = ", ".join(f"{lang} ({count})" for lang, count in code_stats.get("by_language", {}).items())
    table.add_row("Code intelligence files parsed", str(code_stats.get("file_count", 0)))
    table.add_row("Classes / functions found", f"{code_stats.get('class_count', 0)} / {code_stats.get('function_count', 0)}")
    table.add_row("Code intelligence languages", code_langs or "(none)")

    console.print(table)
