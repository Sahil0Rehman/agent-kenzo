"""Phase 5 "Semantic Memory and Retrieval Engine" tests -- Part 3: structured
long-term project memory (`kenzo/memory/architecture_memory.py`), memory
hygiene scoring (`kenzo/memory/importance.py`), and cross-task session memory
(`kenzo/memory/session_memory.py`), including the required "Task 1: fix bug.
Task 2: 'fix same issue elsewhere'. Agent retrieves previous fix." scenario.
"""

from __future__ import annotations

import json

from kenzo.memory.architecture_memory import ArchitectureMemory, MemoryEntry
from kenzo.memory.importance import IMPORTANCE_THRESHOLD, is_worth_remembering, score_importance
from kenzo.memory.repository import RepositoryMemory
from kenzo.memory.session_memory import SessionEntry, SessionMemory

# ---------------------------------------------------------------------------
# 6. ArchitectureMemory
# ---------------------------------------------------------------------------


def _oidc_logout_entry(timestamp: str = "2026-01-01T00:00:00") -> MemoryEntry:
    """Mirrors the exact OIDC-logout example used in
    kenzo/core/semantic_context.py's module docstring."""
    return MemoryEntry(
        kind="bug_fix",
        issue="OIDC logout not called -- session persisted after user clicked sign out",
        root_cause="SecurityConfig never registered the OIDC logout handler with the provider",
        files=["SecurityConfig.java", "LogoutController.java"],
        solution="Registered OidcClientInitiatedLogoutSuccessHandler in SecurityConfig",
        notes="Confirmed via manual test: session cookie cleared after logout redirect",
        timestamp=timestamp,
    )


def test_architecture_memory_records_and_persists_oidc_logout_entry(tmp_path):
    memory = ArchitectureMemory(str(tmp_path))
    entry = _oidc_logout_entry()

    memory.record(entry)

    assert memory.all_entries() == [entry]
    persisted_path = RepositoryMemory(str(tmp_path)).architecture_memory_path
    assert persisted_path.exists()
    on_disk = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert on_disk[0]["issue"] == entry.issue
    assert on_disk[0]["kind"] == "bug_fix"


def test_architecture_memory_find_related_ranks_relevant_entry_above_unrelated(tmp_path):
    memory = ArchitectureMemory(str(tmp_path))
    oidc_entry = _oidc_logout_entry()
    unrelated_entry = MemoryEntry(
        kind="decision",
        issue="Chose PostgreSQL over MySQL for the reporting database",
        root_cause="",
        files=["docs/adr-003-database-choice.md"],
        solution="",
        notes="Better JSONB support for the audit trail feature",
        timestamp="2026-01-02T00:00:00",
    )

    memory.record(unrelated_entry)
    memory.record(oidc_entry)

    results = memory.find_related("logout session not clearing after OIDC sign out", top_k=5)

    assert results
    assert results[0] == oidc_entry
    assert unrelated_entry not in results


def test_architecture_memory_persistence_round_trips_across_instances(tmp_path):
    first = ArchitectureMemory(str(tmp_path))
    first.record(_oidc_logout_entry())
    first.record(
        MemoryEntry(kind="pattern", issue="Repositories always constructor-inject their DAO", timestamp="2026-01-03T00:00:00")
    )

    second = ArchitectureMemory(str(tmp_path))

    assert len(second.all_entries()) == 2
    assert second.all_entries()[0].issue == first.all_entries()[0].issue
    assert second.all_entries()[1].kind == "pattern"


def test_architecture_memory_missing_file_degrades_to_empty(tmp_path):
    memory = ArchitectureMemory(str(tmp_path))  # nothing recorded, nothing on disk
    assert memory.all_entries() == []
    assert memory.find_related("anything at all", top_k=5) == []


def test_architecture_memory_malformed_file_degrades_to_empty(tmp_path):
    repo_memory = RepositoryMemory(str(tmp_path))
    path = repo_memory.architecture_memory_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json !!", encoding="utf-8")

    memory = ArchitectureMemory(str(tmp_path))

    assert memory.all_entries() == []


def test_architecture_memory_skips_individually_malformed_entries(tmp_path):
    repo_memory = RepositoryMemory(str(tmp_path))
    path = repo_memory.architecture_memory_path
    path.parent.mkdir(parents=True, exist_ok=True)
    # A well-formed list containing one entry missing the required `kind`
    # field alongside one valid entry -- the bad one must not sink the store.
    path.write_text(
        json.dumps(
            [
                {"issue": "missing kind field entirely"},
                {"kind": "bug_fix", "issue": "a valid entry", "timestamp": "2026-01-01T00:00:00"},
            ]
        ),
        encoding="utf-8",
    )

    memory = ArchitectureMemory(str(tmp_path))

    assert len(memory.all_entries()) == 1
    assert memory.all_entries()[0].issue == "a valid entry"


def test_architecture_memory_find_related_returns_empty_for_empty_query(tmp_path):
    memory = ArchitectureMemory(str(tmp_path))
    memory.record(_oidc_logout_entry())

    assert memory.find_related("", top_k=5) == []
    assert memory.find_related("   ", top_k=5) == []


# ---------------------------------------------------------------------------
# 7. importance.py -- memory hygiene directional correctness
# ---------------------------------------------------------------------------


def test_bug_fix_succeeded_scores_high_and_clears_threshold():
    score = score_importance("bug_fix", "Fixed the null pointer exception in the login redirect handler", succeeded=True)
    assert score >= IMPORTANCE_THRESHOLD
    assert is_worth_remembering("bug_fix", "Fixed the null pointer exception in the login redirect handler", succeeded=True)


def test_bug_fix_failed_scores_low_and_misses_threshold():
    score = score_importance("bug_fix", "Attempted a fix for the null pointer exception but it did not work", succeeded=False)
    assert score < IMPORTANCE_THRESHOLD
    assert not is_worth_remembering(
        "bug_fix", "Attempted a fix for the null pointer exception but it did not work", succeeded=False
    )


def test_tool_output_scores_low_regardless_of_succeeded():
    succeeded_score = score_importance("tool_output", "ls -la output: total 48 drwxr-xr-x ...", succeeded=True)
    failed_score = score_importance("tool_output", "ls -la output: total 48 drwxr-xr-x ...", succeeded=False)
    assert succeeded_score < IMPORTANCE_THRESHOLD
    assert failed_score < IMPORTANCE_THRESHOLD
    assert failed_score <= succeeded_score


def test_decision_stays_high_even_when_succeeded_is_false():
    """Decisions/patterns represent knowledge, not an outcome -- they stay
    high-value regardless of whether the surrounding attempt succeeded."""
    score = score_importance("decision", "Chose PostgreSQL over MySQL for the reporting database", succeeded=False)
    assert score >= IMPORTANCE_THRESHOLD


def test_pattern_stays_high_even_when_succeeded_is_false():
    score = score_importance("pattern", "Repositories always constructor-inject their DAO dependency", succeeded=False)
    assert score >= IMPORTANCE_THRESHOLD


def test_very_short_content_is_capped_low_regardless_of_kind():
    score = score_importance("bug_fix", "fixed it", succeeded=True)  # under _MIN_CONTENT_LENGTH
    assert score < IMPORTANCE_THRESHOLD


def test_empty_content_scores_zero():
    assert score_importance("bug_fix", "", succeeded=True) == 0.0
    assert score_importance("bug_fix", "   ", succeeded=True) == 0.0
    assert not is_worth_remembering("decision", "", succeeded=True)


def test_unrecognized_kind_defaults_cautiously_below_threshold():
    score = score_importance("some_totally_unknown_kind", "a reasonably long piece of unclassified content here", succeeded=True)
    assert score < IMPORTANCE_THRESHOLD


def test_failed_attempt_and_temporary_kinds_score_low():
    assert score_importance("failed_attempt", "Tried running the build but it failed with a compile error", succeeded=True) < IMPORTANCE_THRESHOLD
    assert score_importance("temporary", "Just a scratch note to self, ignore this later please", succeeded=True) < IMPORTANCE_THRESHOLD


# ---------------------------------------------------------------------------
# 8. SessionMemory -- the required "same issue" retrieval scenario, hygiene
#    fallback-slot capping, and confirmation of the "task_outcome" fix.
# ---------------------------------------------------------------------------


def test_task_outcome_kind_is_now_recognized_as_high_value_when_succeeded():
    """Confirms the fix `kenzo.memory.session_memory`'s own module docstring
    flags as needed: kind="task_outcome" (what SessionMemory.record() always
    passes to is_worth_remembering) must score >= IMPORTANCE_THRESHOLD for a
    realistic successful fix summary -- this is what lets more than one
    SessionEntry accumulate permanently instead of only ever the fallback
    slot surviving."""
    content = "Fixed null pointer bug in UserLoginController's login redirect handler by adding a null check"
    score = score_importance("task_outcome", content, succeeded=True)

    assert score >= IMPORTANCE_THRESHOLD
    assert is_worth_remembering("task_outcome", content, succeeded=True)


def test_session_memory_finds_earlier_fix_for_a_vaguer_later_request(tmp_path):
    """Required scenario: Task 1 fixes a bug. Task 2 asks (vaguely) to "fix
    the same issue elsewhere". SessionMemory.find_related must surface the
    first task's entry so a later agent turn can build on it."""
    session = SessionMemory(str(tmp_path))

    fix_entry = SessionEntry(
        request="Fix null pointer bug in UserLoginController's login redirect handler",
        task_id="task-1",
        succeeded=True,
        summary=(
            "Fixed null pointer bug in UserLoginController's login redirect handler "
            "by adding a null check before redirect"
        ),
        files_discovered=["UserLoginController.java"],
        timestamp="2026-01-01T00:00:00",
    )
    unrelated_entry = SessionEntry(
        request="Update the footer CSS spacing",
        task_id="task-2",
        succeeded=True,
        summary="Adjusted padding in footer.css for the mobile breakpoint",
        files_discovered=["footer.css"],
        timestamp="2026-01-01T00:05:00",
    )

    session.record(fix_entry)
    session.record(unrelated_entry)

    results = session.find_related("fix the same issue in admin login", top_k=3)

    assert results, "expected at least one related session entry"
    assert results[0].task_id == "task-1"
    assert results[0].request == fix_entry.request
    assert all(r.task_id != "task-2" for r in results)


def test_session_memory_render_recent_surfaces_the_fix_and_its_files(tmp_path):
    session = SessionMemory(str(tmp_path))
    session.record(
        SessionEntry(
            request="Fix null pointer bug in UserLoginController's login redirect handler",
            succeeded=True,
            summary="Fixed null pointer bug in UserLoginController's login redirect handler",
            files_discovered=["UserLoginController.java"],
            timestamp="2026-01-01T00:00:00",
        )
    )

    rendered = session.render_recent(top_k=3)

    assert "UserLoginController" in rendered
    assert "Recent session activity" in rendered


def test_session_memory_fallback_slot_caps_a_run_of_low_value_entries(tmp_path):
    """Several low-value/failed entries in a row must not balloon the
    persisted file -- the fallback-slot mechanism pins them to at most one
    trailing slot."""
    session = SessionMemory(str(tmp_path))

    for i in range(5):
        session.record(
            SessionEntry(
                request=f"attempt {i}",
                succeeded=False,
                summary="",  # short/empty content -> capped low regardless of kind
                timestamp=f"2026-01-01T00:0{i}:00",
            )
        )

    assert len(session._entries) == 1

    persisted_path = RepositoryMemory(str(tmp_path)).session_memory_path
    on_disk = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert len(on_disk) == 1


def test_session_memory_real_successful_entries_persist_permanently(tmp_path):
    """A real successful entry (kind="task_outcome", succeeded=True, a
    realistic summary) DOES persist permanently now -- confirms the
    importance.py fix (task_outcome recognized as high-value-success-
    sensitive) actually changes SessionMemory's real accumulation behavior,
    not just score_importance() in isolation."""
    session = SessionMemory(str(tmp_path))

    first = SessionEntry(
        request="Fix null pointer bug in UserLoginController's login redirect handler",
        succeeded=True,
        summary="Fixed null pointer bug in UserLoginController's login redirect handler by adding a null check",
        files_discovered=["UserLoginController.java"],
        timestamp="2026-01-01T00:00:00",
    )
    second = SessionEntry(
        request="Fix a similar null pointer bug in AdminLoginController",
        succeeded=True,
        summary="Fixed a similar null pointer bug in AdminLoginController by adding the same null check pattern",
        files_discovered=["AdminLoginController.java"],
        timestamp="2026-01-01T00:10:00",
    )

    session.record(first)
    session.record(second)

    # Both survive -- NOT capped down to a single fallback slot, since both
    # are genuinely "worth remembering" per the task_outcome fix.
    assert len(session._entries) == 2

    persisted_path = RepositoryMemory(str(tmp_path)).session_memory_path
    on_disk = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert len(on_disk) == 2
    assert on_disk[0]["task_id"] == "" and on_disk[0]["request"] == first.request
    assert on_disk[1]["request"] == second.request


def test_session_memory_persistence_round_trips_across_instances(tmp_path):
    first_session = SessionMemory(str(tmp_path))
    first_session.record(
        SessionEntry(
            request="Fix null pointer bug in UserLoginController",
            succeeded=True,
            summary="Fixed null pointer bug in UserLoginController by adding a null check before redirect",
            timestamp="2026-01-01T00:00:00",
        )
    )

    second_session = SessionMemory(str(tmp_path))

    assert len(second_session._entries) == 1
    assert second_session._entries[0].request == "Fix null pointer bug in UserLoginController"


def test_session_memory_find_related_returns_empty_for_empty_query(tmp_path):
    session = SessionMemory(str(tmp_path))
    session.record(SessionEntry(request="Fix a bug", succeeded=True, summary="a fix", timestamp="2026-01-01T00:00:00"))

    assert session.find_related("", top_k=3) == []
