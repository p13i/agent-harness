from dataclasses import replace

from agent_harness.goals import create_goal
from agent_harness.goals import evaluate_goal
from agent_harness.goals import exhausted_budget
from agent_harness.goals import goal_consumption
from agent_harness.goals import make_evidence
from agent_harness.goals import _number
from agent_harness.goals import _predicate_satisfied
from agent_harness.goals import _timestamp
from agent_harness.goals import _token_count
from agent_harness.ids import new_uuid


def test_goal_requires_declared_predicates() -> None:
    goal = create_goal(new_uuid(), "finish")
    assert not evaluate_goal(goal, []).satisfied


def test_goal_completion_is_evidence_backed() -> None:
    goal = create_goal(
        new_uuid(),
        "ship",
        predicates=(
            {
                "type": "test",
                "subject": "unit",
                "outcome": "passed",
            },
        ),
    )
    missing = evaluate_goal(goal, [])
    assert not missing.satisfied
    evidence = make_evidence(
        goal.goal_id,
        "test",
        "unit",
        "passed",
    )
    assert evaluate_goal(goal, [evidence]).satisfied


def test_invariant_goal_never_auto_completes() -> None:
    goal = create_goal(
        new_uuid(),
        "keep service healthy",
        kind="invariant",
        predicates=({"type": "probe", "outcome": "passed"},),
    )
    evidence = make_evidence(goal.goal_id, "probe", "", "passed")
    assert not evaluate_goal(goal, [evidence]).satisfied


def test_goal_budgets_are_computed_from_durable_results() -> None:
    goal = create_goal(
        new_uuid(),
        "bounded work",
        budgets={
            "turns": 2,
            "tokens": 100,
            "dollars": 1,
            "seconds": 60,
        },
    )
    consumption = goal_consumption(
        goal,
        [
            {
                "usage": {
                    "tokenUsage": {"totalTokens": 120},
                    "total_cost_usd": 0.25,
                }
            }
        ],
        1,
        observed_at=goal.created_at,
    )

    assert consumption.tokens == 120
    assert consumption.dollars == 0.25
    assert exhausted_budget(goal, consumption) == "tokens"


def test_goal_predicates_and_numeric_helpers_reject_false_matches() -> None:
    goal = replace(
        create_goal(new_uuid(), "bounded"),
        budgets={"turns": True},
    )
    evidence = make_evidence(
        goal.goal_id,
        "test",
        "unit",
        "failed",
        {"count": 1},
    )
    assert not _predicate_satisfied(
        {
            "type": "probe",
            "subject": "unit",
            "outcome": "passed",
        },
        [evidence],
    )
    assert not _predicate_satisfied(
        {
            "type": "test",
            "subject": "integration",
            "outcome": "failed",
        },
        [evidence],
    )
    assert not _predicate_satisfied(
        {
            "type": "test",
            "subject": "unit",
            "outcome": "passed",
        },
        [evidence],
    )
    assert not _predicate_satisfied(
        {
            "type": "test",
            "subject": "unit",
            "outcome": "failed",
            "field": "count",
            "equals": 2,
        },
        [evidence],
    )
    assert _predicate_satisfied(
        {
            "type": "test",
            "subject": "unit",
            "outcome": "failed",
            "field": "count",
            "equals": 1,
        },
        [evidence],
    )
    assert exhausted_budget(
        goal,
        goal_consumption(goal, [], 10, observed_at=goal.created_at),
    ) == ""
    assert _timestamp("2026-01-01T00:00:00").tzinfo is not None
    assert _token_count(
        {
            "nested": {
                "input_tokens": 3,
                "ignored_tokens": True,
            }
        }
    ) == 3
    assert _number(True) is None
