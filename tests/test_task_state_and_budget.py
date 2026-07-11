"""Unit tests for the Phase 3 data contracts added to `kenzo.core.schemas`:
`ControllerAction`, `ActionDecision`, `FailedAttempt`, `Hypothesis`,
`TaskState`, and `AgentBudgetConfig` (plus its wiring into `KenzoConfig`).

No model/tool/repo fixtures are needed here -- these are plain pydantic
construction/default/round-trip checks.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kenzo.config.settings import KenzoConfig
from kenzo.core.schemas import (
    ActionDecision,
    AgentBudgetConfig,
    ControllerAction,
    FailedAttempt,
    Hypothesis,
    TaskState,
)


# ---------------------------------------------------------------------------
# AgentBudgetConfig defaults
# ---------------------------------------------------------------------------


def test_agent_budget_config_defaults():
    budget = AgentBudgetConfig()
    assert budget.max_iterations == 50
    assert budget.max_tool_calls == 100
    assert budget.max_runtime_minutes == 30
    assert budget.small_task_iterations == 10


def test_agent_budget_config_overrides():
    budget = AgentBudgetConfig(max_iterations=3, max_tool_calls=10, max_runtime_minutes=1, small_task_iterations=2)
    assert budget.max_iterations == 3
    assert budget.max_tool_calls == 10
    assert budget.max_runtime_minutes == 1
    assert budget.small_task_iterations == 2


def test_kenzo_config_agent_field_is_agent_budget_config_with_defaults():
    config = KenzoConfig()
    assert isinstance(config.agent, AgentBudgetConfig)
    assert config.agent.max_iterations == 50
    assert config.agent.max_tool_calls == 100
    assert config.agent.max_runtime_minutes == 30
    assert config.agent.small_task_iterations == 10


# ---------------------------------------------------------------------------
# ControllerAction
# ---------------------------------------------------------------------------


def test_controller_action_has_expected_members():
    expected = {
        "plan",
        "search",
        "read",
        "analyze",
        "edit",
        "test",
        "review",
        "delegate",
        "ask_user",
        "finish",
    }
    assert {member.value for member in ControllerAction} == expected


def test_controller_action_constructible_from_value():
    assert ControllerAction("edit") is ControllerAction.EDIT
    with pytest.raises(ValueError):
        ControllerAction("not_a_real_action")


# ---------------------------------------------------------------------------
# ActionDecision
# ---------------------------------------------------------------------------


def test_action_decision_defaults():
    decision = ActionDecision(action=ControllerAction.SEARCH)
    assert decision.rationale == ""
    assert decision.target == ""


def test_action_decision_full_construction():
    decision = ActionDecision(action=ControllerAction.EDIT, rationale="fix the bug", target="main.py")
    assert decision.action == ControllerAction.EDIT
    assert decision.rationale == "fix the bug"
    assert decision.target == "main.py"


# ---------------------------------------------------------------------------
# FailedAttempt
# ---------------------------------------------------------------------------


def test_failed_attempt_defaults():
    attempt = FailedAttempt(action="edit")
    assert attempt.target == ""
    assert attempt.error == ""


def test_failed_attempt_full_construction():
    attempt = FailedAttempt(action="edit", target="main.py", error="old_text not found")
    assert attempt.action == "edit"
    assert attempt.target == "main.py"
    assert attempt.error == "old_text not found"


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


def test_hypothesis_default_status_is_untested():
    hypothesis = Hypothesis(description="The bug is in AuthService.authenticate")
    assert hypothesis.status == "untested"


def test_hypothesis_accepts_confirmed_and_rejected():
    confirmed = Hypothesis(description="d", status="confirmed")
    rejected = Hypothesis(description="d", status="rejected")
    assert confirmed.status == "confirmed"
    assert rejected.status == "rejected"


def test_hypothesis_rejects_invalid_status():
    with pytest.raises(ValidationError):
        Hypothesis(description="d", status="maybe")


# ---------------------------------------------------------------------------
# TaskState defaults
# ---------------------------------------------------------------------------


def test_task_state_defaults_with_only_goal_supplied():
    state = TaskState(goal="Fix the login bug")
    assert state.task_id == ""
    assert state.goal == "Fix the login bug"
    assert state.repo_path == ""
    assert state.current_phase == "plan"
    assert state.files_seen == []
    assert state.files_modified == []
    assert state.hypotheses == []
    assert state.failed_attempts == []
    assert state.test_results == []
    assert state.decisions == []
    assert state.remaining_work == []
    assert state.changes == []
    assert state.iteration == 0
    assert state.tool_call_count == 0
    assert state.finished is False
    assert state.needs_user_input is False
    assert state.final_response is None


def test_task_state_requires_goal():
    with pytest.raises(ValidationError):
        TaskState()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TaskState round-tripping via model_dump / model_validate
# ---------------------------------------------------------------------------


def _build_populated_task_state() -> TaskState:
    return TaskState(
        task_id="task-abc123",
        goal="Fix the logout endpoint",
        repo_path="/tmp/repo",
        current_phase="edit",
        files_seen=["main.py", "AuthService.java"],
        files_modified=["main.py"],
        hypotheses=[
            Hypothesis(description="The bug is a stale session token", status="confirmed"),
            Hypothesis(description="It's a race condition", status="rejected"),
        ],
        failed_attempts=[
            FailedAttempt(action="edit", target="main.py", error="old_text not found"),
        ],
        test_results=[{"passed": False, "summary": "1 failed"}, {"passed": True, "summary": "all pass"}],
        decisions=["[iter 0] search: looking for the bug", "[iter 1] edit: applied the fix"],
        remaining_work=["verify the fix under load"],
        changes=[{"file_path": "main.py", "action": "edit", "diff": "...", "content": None, "applied": True}],
        iteration=2,
        tool_call_count=5,
        finished=True,
        needs_user_input=False,
        final_response="Done.",
    )


def test_task_state_model_dump_round_trips_through_model_validate():
    original = _build_populated_task_state()

    dumped = original.model_dump()
    restored = TaskState.model_validate(dumped)

    assert restored == original
    assert restored.model_dump() == dumped


def test_task_state_round_trip_preserves_nested_hypothesis_and_failed_attempt_types():
    original = _build_populated_task_state()
    dumped = original.model_dump()

    # The dumped nested lists are plain dicts, not model instances.
    assert isinstance(dumped["hypotheses"][0], dict)
    assert isinstance(dumped["failed_attempts"][0], dict)

    restored = TaskState.model_validate(dumped)

    # But after round-tripping back through model_validate, they are real
    # Hypothesis/FailedAttempt instances again, not leftover dicts.
    assert all(isinstance(h, Hypothesis) for h in restored.hypotheses)
    assert all(isinstance(fa, FailedAttempt) for fa in restored.failed_attempts)

    assert restored.hypotheses[0].description == "The bug is a stale session token"
    assert restored.hypotheses[0].status == "confirmed"
    assert restored.hypotheses[1].status == "rejected"

    assert restored.failed_attempts[0].action == "edit"
    assert restored.failed_attempts[0].target == "main.py"
    assert restored.failed_attempts[0].error == "old_text not found"


def test_task_state_round_trip_through_json_string():
    """model_dump_json / model_validate_json is the same round trip a
    real TaskStore-backed persist/resume cycle uses under the hood."""
    original = _build_populated_task_state()
    json_text = original.model_dump_json()
    restored = TaskState.model_validate_json(json_text)
    assert restored == original
