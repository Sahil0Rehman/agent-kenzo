"""Standalone benchmark for Kenzo's Phase 4 "Deep Code Intelligence Engine"
(kenzo.intelligence.{symbol_index,reference_graph,dependency_graph}).

NOT a pytest test. It is slow by design (it generates synthetic repos of up
to tens of thousands of files) and its output is a human-readable report,
not a pass/fail assertion -- it lives under scripts/, has no test_*
functions, and is never collected by the test suite.

What it does
------------
1. Generates a synthetic Java "Spring Boot"-shaped repository on disk at a
   few scales: N controllers, N services, N repositories, organized into
   packageN/{controller,service,repository} directories. Each controller
   constructor-injects one service and calls one method on it; each service
   constructor-injects one repository and calls one method on it -- real
   call chains and DI relationships, not N unconnected classes, so
   ReferenceGraph/DependencyGraph have genuine cross-file work to do.
2. Actually runs SymbolIndex.build() / ReferenceGraph.build() /
   DependencyGraph.build() / SymbolIndex.reindex_file() against each
   generated repo and times each with time.perf_counter().
3. Picks its largest ("large") tier adaptively: it extrapolates from the
   two smaller tiers' *measured* growth rate to estimate how long
   ReferenceGraph.build() (the step with the most repo-wide work per call
   site -- see its module docstring) would take at a series of candidate
   scales, and picks the largest candidate expected to finish within a
   soft time budget. This is only used to CHOOSE the scale; the numbers
   printed are always real, measured wall-clock times from actually
   running the pipeline, never estimates.
4. Prints a results table and cleans up every generated repo (written
   under a Python `tempfile` temp directory, removed as each tier finishes
   and once more at exit).

Usage
-----
    C:\\Users\\sahil.r\\Ken\\.venv\\Scripts\\python.exe scripts\\benchmark_indexing.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Allow running as `python scripts/benchmark_indexing.py` from the repo root
# without requiring kenzo to already be on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kenzo.intelligence.dependency_graph import DependencyGraph  # noqa: E402
from kenzo.intelligence.reference_graph import ReferenceGraph  # noqa: E402
from kenzo.intelligence.symbol_index import SymbolIndex  # noqa: E402

# Real default from kenzo.config.settings.IndexingConfig.max_indexed_files --
# both `kenzo index` and AgentRuntime.__init__ build the real SymbolIndex
# with this cap.
DEFAULT_MAX_INDEXED_FILES = 2000

# Soft budgets purely for picking the "large" tier's scale up front (NOT for
# aborting a run in progress -- once a tier starts, it runs to completion so
# every printed number is real).
OVERALL_SOFT_BUDGET_SECONDS = 280.0
REFERENCE_GRAPH_SOFT_CAP_SECONDS = 90.0

# Candidate total-file counts for the "large" tier, largest first. The
# script walks this list and picks the first (largest) one whose
# extrapolated ReferenceGraph.build() time fits the soft cap above.
LARGE_TIER_CANDIDATES = [18000, 12000, 9000, 6000, 4500]


# ---------------------------------------------------------------------------
# Synthetic Java Spring-Boot-shaped repo generator
# ---------------------------------------------------------------------------

_CONTROLLER_TEMPLATE = """\
package {package}.controller;

import {package}.service.{service_name};
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * {name} exposes the HTTP surface for synthetic domain entity #{idx}.
 *
 * Auto-generated for kenzo's Phase 4 indexing-scale benchmark; not real
 * production code. Constructor-injects {service_name} and delegates the
 * primary GET endpoint to it, mirroring a typical Spring Boot controller.
 */
@RestController
@RequestMapping("/api/v1/entity{idx}")
public class {name} {{

    private final {service_name} {service_var};

    public {name}({service_name} {service_var}) {{
        this.{service_var} = {service_var};
    }}

    @GetMapping("/items")
    public String getItems() {{
        return {service_var}.performAction();
    }}

    @GetMapping("/status")
    public String getStatus() {{
        return "OK";
    }}

    @PostMapping("/items")
    public String createItem() {{
        return "created";
    }}

    @GetMapping("/health")
    public String health() {{
        return "healthy";
    }}
}}
"""

_SERVICE_TEMPLATE = """\
package {package}.service;

import {package}.repository.{repo_name};
import org.springframework.stereotype.Service;

/**
 * {name} contains the business logic for synthetic domain entity #{idx}.
 *
 * Auto-generated for kenzo's Phase 4 indexing-scale benchmark; not real
 * production code. Constructor-injects {repo_name} and delegates its
 * primary action to it, mirroring a typical Spring Boot service.
 */
@Service
public class {name} {{

    private final {repo_name} {repo_var};

    public {name}({repo_name} {repo_var}) {{
        this.{repo_var} = {repo_var};
    }}

    public String performAction() {{
        {repo_var}.findByCode("default");
        return "done";
    }}

    public long countActive() {{
        return {repo_var}.countActive();
    }}

    public String describe() {{
        return "service #{idx}";
    }}
}}
"""

_REPOSITORY_TEMPLATE = """\
package {package}.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * {name} provides persistence access for synthetic domain entity #{idx}.
 *
 * Auto-generated for kenzo's Phase 4 indexing-scale benchmark; not real
 * production code.
 */
@Repository
public interface {name} extends JpaRepository<Object, Long> {{

    Object findByCode(String code);

    List<Object> findAllActive();

    List<Object> findByStatus(String status);

    boolean existsByCode(String code);

    long countActive();
}}
"""


@dataclass
class GenStats:
    file_count: int
    total_loc: int
    num_packages: int
    reindex_target: str  # repo-relative path of a real, already-indexed file


def generate_spring_boot_repo(root: Path, n_features: int, features_per_package: int = 25) -> GenStats:
    """Generate `n_features` controllers + `n_features` services +
    `n_features` repositories under `root`, grouped into
    `com.example.appNN.{controller,service,repository}` packages of
    `features_per_package` features each.

    Feature `i`'s controller constructor-injects feature `i`'s service and
    calls one method on it; feature `i`'s service constructor-injects
    feature `i`'s repository and calls one method on it -- a real,
    resolvable call chain and DI relationship per feature, and a real
    cross-file import edge at every hop (for DependencyGraph's file_deps).
    """
    num_packages = max(1, (n_features + features_per_package - 1) // features_per_package)
    base_java = root / "src" / "main" / "java" / "com" / "example"

    created_dirs: set[Path] = set()

    def ensure_dir(d: Path) -> None:
        if d not in created_dirs:
            d.mkdir(parents=True, exist_ok=True)
            created_dirs.add(d)

    total_loc = 0
    reindex_target = ""

    for i in range(n_features):
        pkg_idx = i % num_packages
        package = f"com.example.app{pkg_idx}"
        suffix = f"{i:06d}"

        controller_name = f"Ctrl{suffix}"
        service_name = f"Svc{suffix}"
        repo_name = f"Repo{suffix}"
        service_var = f"svc{suffix}"
        repo_var = f"repo{suffix}"

        pkg_dir = base_java / f"app{pkg_idx}"
        controller_dir = pkg_dir / "controller"
        service_dir = pkg_dir / "service"
        repository_dir = pkg_dir / "repository"
        ensure_dir(controller_dir)
        ensure_dir(service_dir)
        ensure_dir(repository_dir)

        controller_src = _CONTROLLER_TEMPLATE.format(
            package=package, name=controller_name, idx=i,
            service_name=service_name, service_var=service_var,
        )
        service_src = _SERVICE_TEMPLATE.format(
            package=package, name=service_name, idx=i,
            repo_name=repo_name, repo_var=repo_var,
        )
        repository_src = _REPOSITORY_TEMPLATE.format(package=package, name=repo_name, idx=i)

        controller_path = controller_dir / f"{controller_name}.java"
        service_path = service_dir / f"{service_name}.java"
        repository_path = repository_dir / f"{repo_name}.java"

        controller_path.write_text(controller_src, encoding="utf-8")
        service_path.write_text(service_src, encoding="utf-8")
        repository_path.write_text(repository_src, encoding="utf-8")

        total_loc += controller_src.count("\n") + service_src.count("\n") + repository_src.count("\n")

        if i == 0:
            reindex_target = controller_path.relative_to(root).as_posix()

    return GenStats(
        file_count=n_features * 3,
        total_loc=total_loc,
        num_packages=num_packages,
        reindex_target=reindex_target,
    )


# ---------------------------------------------------------------------------
# One benchmark tier
# ---------------------------------------------------------------------------


@dataclass
class TierResult:
    label: str
    n_features: int
    file_count: int
    total_loc: int
    num_packages: int
    generate_seconds: float
    truncation_demo: "TruncationDemo | None"
    build_seconds: float
    build_max_files: int
    files_indexed: int
    total_files_found: int
    truncated: bool
    class_count: int
    function_count: int
    reference_seconds: float
    dependency_seconds: float
    reindex_seconds: float

    @property
    def files_per_second(self) -> float:
        return self.files_indexed / self.build_seconds if self.build_seconds > 0 else float("inf")


@dataclass
class TruncationDemo:
    max_files: int
    total_files_found: int
    files_indexed: int
    truncated: bool
    build_seconds: float


def run_tier(
    tmp_root: Path,
    label: str,
    n_features: int,
    features_per_package: int,
    demo_truncation: bool = False,
) -> TierResult:
    repo_dir = tmp_root / f"repo_{label}"
    print(f"\n=== Tier '{label}': generating {n_features} controllers + "
          f"{n_features} services + {n_features} repositories "
          f"({n_features * 3} files) ===", flush=True)

    t0 = time.perf_counter()
    gen_stats = generate_spring_boot_repo(repo_dir, n_features, features_per_package=features_per_package)
    generate_seconds = time.perf_counter() - t0
    print(f"  generated {gen_stats.file_count} files, ~{gen_stats.total_loc} LOC, "
          f"{gen_stats.num_packages} packages in {generate_seconds:.2f}s", flush=True)

    truncation_demo: TruncationDemo | None = None
    if demo_truncation:
        # One explicit data point showing SymbolIndex.build()'s truncation
        # behavior at the real default cap (kenzo.config.settings
        # .IndexingConfig.max_indexed_files == 2000), on a repo big enough
        # to exceed it.
        capped_index = SymbolIndex(str(repo_dir))
        t0 = time.perf_counter()
        capped_index.build(max_files=DEFAULT_MAX_INDEXED_FILES)
        capped_seconds = time.perf_counter() - t0
        capped_stats = capped_index.stats()
        truncation_demo = TruncationDemo(
            max_files=DEFAULT_MAX_INDEXED_FILES,
            total_files_found=capped_stats["total_files_found"],
            files_indexed=capped_stats["file_count"],
            truncated=capped_stats["truncated"],
            build_seconds=capped_seconds,
        )
        print(f"  [truncation demo] max_files={DEFAULT_MAX_INDEXED_FILES}: "
              f"indexed {capped_stats['file_count']}/{capped_stats['total_files_found']} "
              f"files, truncated={capped_stats['truncated']}, {capped_seconds:.2f}s", flush=True)

    # Main measurement: build with max_files high enough that nothing is
    # truncated, so this reflects the true full-repo indexing cost at this
    # scale (not a partial/truncated one).
    index = SymbolIndex(str(repo_dir))
    t0 = time.perf_counter()
    index.build(max_files=gen_stats.file_count + 1000)
    build_seconds = time.perf_counter() - t0
    stats = index.stats()
    print(f"  SymbolIndex.build(): {build_seconds:.3f}s "
          f"({stats['file_count']}/{stats['total_files_found']} files, "
          f"truncated={stats['truncated']}, "
          f"{stats['class_count']} classes, {stats['function_count']} functions)", flush=True)

    ref_graph = ReferenceGraph(index)
    t0 = time.perf_counter()
    ref_graph.build()
    reference_seconds = time.perf_counter() - t0
    n_callee_edges = sum(len(v) for v in ref_graph.callees.values())
    print(f"  ReferenceGraph.build(): {reference_seconds:.3f}s ({n_callee_edges} resolved call edges)", flush=True)

    dep_graph = DependencyGraph(index)
    t0 = time.perf_counter()
    dep_graph.build()
    dependency_seconds = time.perf_counter() - t0
    print(f"  DependencyGraph.build(): {dependency_seconds:.3f}s "
          f"({len(dep_graph.class_deps)} classes, {len(dep_graph.file_deps)} files)", flush=True)

    t0 = time.perf_counter()
    index.reindex_file(gen_stats.reindex_target)
    reindex_seconds = time.perf_counter() - t0
    print(f"  SymbolIndex.reindex_file() [1 file, incremental]: {reindex_seconds:.4f}s", flush=True)

    shutil.rmtree(repo_dir, ignore_errors=True)

    return TierResult(
        label=label,
        n_features=n_features,
        file_count=gen_stats.file_count,
        total_loc=gen_stats.total_loc,
        num_packages=gen_stats.num_packages,
        generate_seconds=generate_seconds,
        truncation_demo=truncation_demo,
        build_seconds=build_seconds,
        build_max_files=gen_stats.file_count + 1000,
        files_indexed=stats["file_count"],
        total_files_found=stats["total_files_found"],
        truncated=stats["truncated"],
        class_count=stats["class_count"],
        function_count=stats["function_count"],
        reference_seconds=reference_seconds,
        dependency_seconds=dependency_seconds,
        reindex_seconds=reindex_seconds,
    )


# ---------------------------------------------------------------------------
# Adaptive selection of the "large" tier's scale
# ---------------------------------------------------------------------------


def _growth_exponent(f1: int, t1: float, f2: int, t2: float) -> float:
    """Empirical power-law exponent k such that time ~ file_count**k,
    fit from two measured (file_count, seconds) points. Guards against
    zero/negative durations (measurement noise on a very fast small tier).
    """
    import math

    t1 = max(t1, 1e-4)
    t2 = max(t2, 1e-4)
    if f2 <= f1:
        return 1.0
    return math.log(t2 / t1) / math.log(f2 / f1)


def pick_large_tier_n_features(
    tier1: TierResult, tier2: TierResult, elapsed_so_far: float, features_per_package: int,
) -> tuple[int, str]:
    """Pick the largest candidate scale (from LARGE_TIER_CANDIDATES) whose
    ReferenceGraph.build() time -- extrapolated from the tier1/tier2
    *measured* growth rate, since that step does repo-wide work per call
    site and is the most likely to scale worse than linearly -- fits under
    REFERENCE_GRAPH_SOFT_CAP_SECONDS, and whose extrapolated total time
    still leaves the overall run under OVERALL_SOFT_BUDGET_SECONDS.

    Returns (n_features, note) where `note` explains the choice (including
    if/why it had to scale down from the largest candidate).
    """
    ref_exp = _growth_exponent(tier1.file_count, tier1.reference_seconds, tier2.file_count, tier2.reference_seconds)
    build_exp = _growth_exponent(tier1.file_count, tier1.build_seconds, tier2.file_count, tier2.build_seconds)

    notes: list[str] = [
        f"empirical growth exponents from measured tiers: "
        f"ReferenceGraph.build() ~ files^{ref_exp:.2f}, SymbolIndex.build() ~ files^{build_exp:.2f}"
    ]

    remaining_budget = max(OVERALL_SOFT_BUDGET_SECONDS - elapsed_so_far, 30.0)

    for candidate_files in LARGE_TIER_CANDIDATES:
        est_ref = tier2.reference_seconds * (candidate_files / tier2.file_count) ** ref_exp
        est_build = tier2.build_seconds * (candidate_files / tier2.file_count) ** build_exp
        est_total = est_build + est_ref + est_build * 0.5  # + dependency/reindex/generate slack
        notes.append(
            f"candidate {candidate_files} files -> est. build={est_build:.1f}s, "
            f"est. reference={est_ref:.1f}s, est. total~{est_total:.1f}s"
        )
        if est_ref <= REFERENCE_GRAPH_SOFT_CAP_SECONDS and est_total <= remaining_budget:
            n_features = candidate_files // 3
            notes.append(
                f"-> selected {candidate_files} files (n_features={n_features}) as the 'large' tier "
                f"(remaining budget ~{remaining_budget:.0f}s)"
            )
            return n_features, "\n  ".join(notes)

    # Nothing fit within budget -- fall back to the smallest candidate and
    # say so explicitly, per the requirement to note any scale-down rather
    # than silently guessing.
    smallest = LARGE_TIER_CANDIDATES[-1]
    n_features = smallest // 3
    notes.append(
        f"-> no candidate fit the soft budget "
        f"(ref cap {REFERENCE_GRAPH_SOFT_CAP_SECONDS:.0f}s / remaining {remaining_budget:.0f}s); "
        f"SCALED DOWN to the smallest candidate, {smallest} files (n_features={n_features})"
    )
    return n_features, "\n  ".join(notes)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_results_table(tiers: list[TierResult]) -> None:
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)

    header = (
        f"{'Tier':<10} {'Files':>7} {'~LOC':>9} {'Pkgs':>5} "
        f"{'Build(s)':>9} {'files/s':>8} {'RefGraph(s)':>11} {'DepGraph(s)':>11} {'Reindex(s)':>11}"
    )
    print(header)
    print("-" * len(header))
    for t in tiers:
        print(
            f"{t.label:<10} {t.file_count:>7} {t.total_loc:>9} {t.num_packages:>5} "
            f"{t.build_seconds:>9.3f} {t.files_per_second:>8.1f} "
            f"{t.reference_seconds:>11.3f} {t.dependency_seconds:>11.3f} {t.reindex_seconds:>11.4f}"
        )

    print()
    print("Incremental update cost (SymbolIndex.reindex_file(), 1 file, regardless of repo size):")
    for t in tiers:
        print(f"  {t.label:<10} ({t.file_count} files in repo): {t.reindex_seconds:.4f}s")

    for t in tiers:
        if t.truncation_demo is not None:
            d = t.truncation_demo
            print()
            print(f"Truncation behavior demo (tier '{t.label}', max_files={d.max_files} == real default "
                  f"IndexingConfig.max_indexed_files):")
            print(f"  total_files_found={d.total_files_found}, files_indexed={d.files_indexed}, "
                  f"truncated={d.truncated}, build time at this cap={d.build_seconds:.3f}s")


def main() -> None:
    overall_t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="kenzo_bench_"))
    print(f"Working directory for synthetic repos: {tmp_root}")

    tiers: list[TierResult] = []
    try:
        # Tier 1: below the real default max_indexed_files cap (2000) --
        # the realistic default-config case.
        tier1 = run_tier(tmp_root, "small", n_features=200, features_per_package=25)
        tiers.append(tier1)

        # Tier 2: meaningfully above the 2000-file default cap. Also used
        # as the truncation-behavior data point (one extra capped build at
        # max_files=2000 on this same repo, before the uncapped main run).
        tier2 = run_tier(tmp_root, "medium", n_features=1000, features_per_package=25, demo_truncation=True)
        tiers.append(tier2)

        elapsed_so_far = time.perf_counter() - overall_t0
        large_n_features, selection_note = pick_large_tier_n_features(
            tier1, tier2, elapsed_so_far, features_per_package=40
        )
        print(f"\n=== Picking 'large' tier scale ===\n  {selection_note}")

        tier3 = run_tier(tmp_root, "large", n_features=large_n_features, features_per_package=40)
        tiers.append(tier3)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print_results_table(tiers)

    total_elapsed = time.perf_counter() - overall_t0
    print(f"\nTotal benchmark wall-clock time (generation + indexing across all tiers): {total_elapsed:.1f}s")
    print(f"Synthetic repos were generated under a temp dir and deleted after each tier; "
          f"nothing left on disk at: {tmp_root}")


if __name__ == "__main__":
    main()
