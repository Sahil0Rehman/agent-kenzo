"""Tests for `kenzo.core.context_manager.ContextManager`: the bounded,
retrieval-friendly memory `kenzo.core.controller.AgentController` uses to keep
a long-running loop's prompts from growing unboundedly.

No repo/model fixtures needed -- `ContextManager` is a pure in-memory data
structure with no I/O of its own (serialization is a plain dict, not a file).
"""

from __future__ import annotations

from kenzo.core.context_manager import ContextManager


# ---------------------------------------------------------------------------
# has_seen_file / re-adding updates rather than duplicates
# ---------------------------------------------------------------------------


def test_has_seen_file_false_before_any_add():
    cm = ContextManager()
    assert cm.has_seen_file("main.py") is False
    assert cm.get_file_summary("main.py") is None


def test_add_file_context_then_has_seen_file_true():
    cm = ContextManager()
    cm.add_context("file", "main.py", "defines main(), returns 42")
    assert cm.has_seen_file("main.py") is True
    assert cm.get_file_summary("main.py") == "defines main(), returns 42"


def test_re_adding_same_file_path_updates_summary_without_duplicating():
    cm = ContextManager()
    cm.add_context("file", "main.py", "original summary")
    cm.add_context("file", "main.py", "updated summary after edit")

    assert len(cm.files_seen) == 1
    assert cm.get_file_summary("main.py") == "updated summary after edit"


def test_re_adding_same_symbol_key_updates_rather_than_duplicates():
    cm = ContextManager()
    cm.add_context("symbol", "AuthService", {"methods": ["authenticate"]})
    cm.add_context("symbol", "AuthService", {"methods": ["authenticate", "terminateSession"]})

    assert len(cm.symbols_seen) == 1
    assert cm.symbols_seen["AuthService"] == {"methods": ["authenticate", "terminateSession"]}


# ---------------------------------------------------------------------------
# compress_context: only notes are ever trimmed
# ---------------------------------------------------------------------------


def test_compress_context_trims_notes_by_count_but_never_touches_files_or_symbols():
    cm = ContextManager(max_notes=3, max_total_chars=1_000_000)
    cm.add_context("file", "main.py", "a short file summary")
    cm.add_context("symbol", "AuthService", "a short symbol summary")

    for i in range(10):
        cm.add_context("note", "", f"note-{i}")

    # Notes count-capped at max_notes, keeping only the most recent ones.
    assert len(cm.notes) == 3
    assert cm.notes == ["note-7", "note-8", "note-9"]

    # files_seen/symbols_seen are durable and untouched by note pressure.
    assert cm.files_seen == {"main.py": "a short file summary"}
    assert cm.symbols_seen == {"AuthService": "a short symbol summary"}


def test_compress_context_trims_by_total_chars_oldest_first():
    # A tiny char budget forces trimming well before the note-count cap.
    cm = ContextManager(max_notes=100, max_total_chars=120)
    cm.add_context("file", "config.py", "s")

    cm.add_context("note", "", "A" * 50)
    cm.add_context("note", "", "B" * 50)
    cm.add_context("note", "", "C" * 50)

    # Oldest ("A"*50) must have been dropped first to get back under budget;
    # the most recent note must survive.
    assert "C" * 50 in cm.notes
    assert "A" * 50 not in cm.notes
    # files_seen is never touched by char-budget trimming either.
    assert cm.files_seen == {"config.py": "s"}


def test_compress_context_never_removes_files_seen_or_symbols_seen_even_far_over_budget():
    cm = ContextManager(max_notes=1, max_total_chars=10)
    for i in range(20):
        cm.add_context("file", f"file_{i}.py", "x" * 40)
        cm.add_context("symbol", f"Symbol{i}", "y" * 40)
    cm.add_context("note", "", "some note text")

    assert len(cm.files_seen) == 20
    assert len(cm.symbols_seen) == 20
    # The char budget is dwarfed by files_seen/symbols_seen alone, so notes
    # gets trimmed all the way to empty -- but that's fine: compress_context
    # only ever removes from `notes`, and files_seen/symbols_seen above are
    # still fully intact regardless.
    assert cm.notes == []


# ---------------------------------------------------------------------------
# retrieve_relevant: keyword-overlap ranking, empty-list vs. fallback behavior
# ---------------------------------------------------------------------------


def test_retrieve_relevant_ranks_multi_keyword_match_above_unrelated_note():
    cm = ContextManager()
    cm.add_context("note", "", "The login endpoint validates password and username tokens.")
    cm.add_context("note", "", "Unrelated deployment note about docker containers and disk space.")
    cm.add_context("note", "", "password token validation logic lives in AuthService.")

    results = cm.retrieve_relevant("password token validation", top_k=8)

    assert results  # non-empty
    # The note hitting all three query keywords ranks first.
    assert results[0] == "password token validation logic lives in AuthService."
    # The genuinely unrelated note must not appear at all (zero-scoring items excluded).
    assert not any("docker" in r for r in results)


def test_retrieve_relevant_breaks_ties_by_recency():
    cm = ContextManager()
    cm.add_context("note", "", "auth bug in login flow")
    cm.add_context("note", "", "auth bug in logout flow")

    results = cm.retrieve_relevant("auth bug", top_k=8)

    # Both notes score identically (2 matching tokens each); the more
    # recently added one must be ranked first.
    assert results[0] == "auth bug in logout flow"
    assert results[1] == "auth bug in login flow"


def test_retrieve_relevant_returns_genuinely_empty_list_when_nothing_matches():
    cm = ContextManager()
    cm.add_context("file", "main.py", "a summary")
    cm.add_context("symbol", "AuthService", "a symbol fact")
    cm.add_context("note", "", "some note about the login flow")

    results = cm.retrieve_relevant("xylophone jamboree zzznomatch", top_k=8)

    assert results == []


def test_retrieve_relevant_empty_query_falls_back_to_recent_notes():
    cm = ContextManager()
    cm.add_context("note", "", "note one")
    cm.add_context("note", "", "note two")
    cm.add_context("note", "", "note three")

    results = cm.retrieve_relevant("", top_k=2)

    # Falls back to the most recent 2 notes, most-recent-first.
    assert results == ["note three", "note two"]


def test_retrieve_relevant_empty_query_with_no_notes_returns_empty_list():
    cm = ContextManager()
    cm.add_context("file", "main.py", "a summary")
    assert cm.retrieve_relevant("", top_k=8) == []


# ---------------------------------------------------------------------------
# render_all / render_relevant smoke tests
# ---------------------------------------------------------------------------


def test_render_all_includes_all_three_sections():
    cm = ContextManager()
    cm.add_context("file", "main.py", "summary")
    cm.add_context("symbol", "AuthService", "fact")
    cm.add_context("note", "", "an observation")

    rendered = cm.render_all()
    assert "Files already examined" in rendered
    assert "main.py: summary" in rendered
    assert "Symbols already examined" in rendered
    assert "AuthService: fact" in rendered
    assert "Observations" in rendered
    assert "an observation" in rendered


def test_render_relevant_returns_empty_string_when_nothing_matches():
    cm = ContextManager()
    cm.add_context("note", "", "some note about the login flow")
    assert cm.render_relevant("zzznomatch", top_k=8) == ""


# ---------------------------------------------------------------------------
# to_dict / from_dict round trip
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip_preserves_state():
    cm = ContextManager(max_notes=50, max_total_chars=5000)
    cm.add_context("file", "main.py", "a summary")
    cm.add_context("symbol", "AuthService", {"methods": ["authenticate"]})
    cm.add_context("note", "", "note one")
    cm.add_context("note", "", "note two")

    data = cm.to_dict()
    restored = ContextManager.from_dict(data)

    assert isinstance(restored, ContextManager)
    assert restored.max_notes == 50
    assert restored.max_total_chars == 5000
    assert restored.files_seen == cm.files_seen
    assert restored.symbols_seen == cm.symbols_seen
    assert restored.notes == cm.notes


def test_from_dict_defaults_when_fields_missing():
    restored = ContextManager.from_dict({})
    assert restored.max_notes == 200
    assert restored.max_total_chars == 20000
    assert restored.files_seen == {}
    assert restored.symbols_seen == {}
    assert restored.notes == []


def test_restored_context_manager_retrieve_relevant_still_works_post_reload():
    cm = ContextManager()
    cm.add_context("note", "", "password token validation logic")
    cm.add_context("note", "", "unrelated docker deployment note")

    restored = ContextManager.from_dict(cm.to_dict())
    results = restored.retrieve_relevant("password token", top_k=8)

    assert results == ["password token validation logic"]
