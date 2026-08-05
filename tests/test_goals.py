import datetime
from dataclasses import replace

import pytest

from agent_harness.goals import (
    GoalConsumption,
    _number,
    _predicate_satisfied,
    _safety_count,
    _timestamp,
    _token_count,
    create_goal,
    evaluate_goal,
    evaluate_milestones,
    exhausted_budget,
    goal_consumption,
    make_evidence,
    promoted_milestones,
    promoted_predicates,
)
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


def test_milestones_require_ordered_dependency_evidence() -> None:
    session_id = new_uuid()
    goal = create_goal(
        session_id,
        "complete two dependent proof stages",
        predicates=({"type": "final", "outcome": "passed"},),
        milestones=(
            {
                "milestone_id": "build",
                "title": "Build",
                "dependencies": [],
                "predicates": [{"type": "build", "outcome": "passed"}],
            },
            {
                "milestone_id": "verify",
                "title": "Verify",
                "dependencies": ["build"],
                "predicates": [{"type": "verify", "outcome": "passed"}],
            },
        ),
        permitted_providers=("claude", "codex"),
        permitted_efforts=("low", "medium"),
        max_concurrency=2,
    )
    build = make_evidence(goal.goal_id, "build", "", "passed")
    after_build = evaluate_milestones(goal, [build])
    assert [item.status for item in after_build] == ["complete", "active"]
    verify = make_evidence(goal.goal_id, "verify", "", "passed")
    final = make_evidence(goal.goal_id, "final", "", "passed")
    assert evaluate_goal(goal, [build, verify, final]).satisfied


def test_goal_policy_and_promoted_milestones_fail_closed() -> None:
    session_id = new_uuid()
    prior = create_goal(
        session_id,
        "first",
        predicates=({"type": "first", "outcome": "passed"},),
        milestones=(
            {
                "milestone_id": "first",
                "title": "First",
                "dependencies": [],
                "predicates": [{"type": "first", "outcome": "passed"}],
            },
        ),
    )
    prior = replace(
        prior,
        milestones=evaluate_milestones(
            prior,
            [make_evidence(prior.goal_id, "first", "", "passed")],
        ),
    )
    later = create_goal(
        session_id,
        "later",
        predicates=(
            {"type": "first", "outcome": "passed"},
            {"type": "later", "outcome": "passed"},
        ),
        milestones=(
            {
                "milestone_id": "first",
                "title": "First",
                "dependencies": [],
                "predicates": [{"type": "first", "outcome": "passed"}],
            },
            {
                "milestone_id": "later",
                "title": "Later",
                "dependencies": ["first"],
                "predicates": [{"type": "later", "outcome": "passed"}],
            },
        ),
    )
    promoted = promoted_milestones(prior, later)
    assert [item.status for item in promoted] == ["complete", "blocked"]
    with pytest.raises(ValueError, match="unsupported permitted effort"):
        create_goal(session_id, "invalid", permitted_efforts=("ultra",))
    with pytest.raises(ValueError, match="between 1 and 2"):
        create_goal(session_id, "invalid", max_concurrency=3)
    with pytest.raises(ValueError, match="finite"):
        create_goal(session_id, "invalid", budgets={"seconds": float("inf")})
    with pytest.raises(ValueError, match="finite"):
        create_goal(session_id, "invalid", budgets={"seconds": float("nan")})
    with pytest.raises(ValueError, match="max_concurrency must be an integer"):
        create_goal(session_id, "invalid", max_concurrency=True)
    with pytest.raises(ValueError, match="completion policy does not match"):
        create_goal(session_id, "invalid", completion_policy="never")
    with pytest.raises(ValueError, match="unsupported goal incident policy"):
        create_goal(session_id, "invalid", incident_policy="ignore")
    with pytest.raises(ValueError, match="cannot contain whitespace"):
        create_goal(session_id, "invalid", permitted_providers=("cla ude",))


def test_promoted_milestones_reject_rewritten_or_removed_stages() -> None:
    session_id = new_uuid()
    first_stage = {
        "milestone_id": "first",
        "title": "First",
        "dependencies": [],
        "predicates": [{"type": "first", "outcome": "passed"}],
    }
    later_stage = {
        "milestone_id": "later",
        "title": "Later",
        "dependencies": ["first"],
        "predicates": [{"type": "later", "outcome": "passed"}],
    }
    prior = create_goal(session_id, "first", milestones=(first_stage,))

    rewritten = create_goal(
        session_id,
        "later",
        milestones=({**first_stage, "title": "Rewritten"}, later_stage),
    )
    with pytest.raises(ValueError, match="cannot rewrite a milestone"):
        promoted_milestones(prior, rewritten)

    replaced = create_goal(
        session_id,
        "later",
        milestones=({**later_stage, "dependencies": []},),
    )
    with pytest.raises(ValueError, match="cannot remove a milestone"):
        promoted_milestones(prior, replaced)

    unchanged = create_goal(session_id, "later", milestones=(first_stage,))
    with pytest.raises(ValueError, match="requires a later milestone"):
        promoted_milestones(prior, unchanged)


def test_milestone_definitions_fail_closed_on_malformed_input() -> None:
    session_id = new_uuid()
    valid = {
        "milestone_id": "first",
        "title": "First",
        "dependencies": [],
        "predicates": [{"type": "first", "outcome": "passed"}],
    }
    cases = (
        ("must be objects", ("not-an-object",)),
        ("identifier is invalid", ({**valid, "milestone_id": "two words"},)),
        ("identifier is duplicated", (valid, valid)),
        ("title is required", ({**valid, "title": "  "},)),
        ("dependencies must be a list", ({**valid, "dependencies": "first"},)),
        ("predicates must be a list", ({**valid, "predicates": "later"},)),
        ("reference prior milestones", ({**valid, "dependencies": ["absent"]},)),
        ("typed evidence predicates", ({**valid, "predicates": []},)),
        ("typed evidence predicates", ({**valid, "predicates": ["bare"]},)),
    )
    for message, milestones in cases:
        with pytest.raises(ValueError, match=message):
            create_goal(session_id, "invalid", milestones=milestones)


def test_boolean_budget_limits_are_never_enforced() -> None:
    goal = replace(
        create_goal(new_uuid(), "bounded work"),
        budgets={"tool_calls": True},
    )
    consumption = GoalConsumption(
        turns=0,
        tokens=0.0,
        context_tokens=0.0,
        output_tokens=0.0,
        tool_calls=9.0,
        attempts=0.0,
        child_agents=0.0,
        dollars=0.0,
        elapsed_seconds=0.0,
    )

    assert exhausted_budget(goal, consumption) == ""


def test_promoted_predicates_are_immutable_ordered_and_unique() -> None:
    first = {"type": "first", "outcome": "passed"}
    second = {"type": "second", "outcome": "passed"}
    later = {"type": "later", "outcome": "passed"}
    previous = (first, second)
    assert promoted_predicates(previous, (first, second, later)) == (
        first,
        second,
        later,
    )
    with pytest.raises(ValueError, match="later predicate"):
        promoted_predicates(previous, previous)
    with pytest.raises(ValueError, match="remove or rewrite"):
        promoted_predicates(previous, (first, later, second))
    with pytest.raises(ValueError, match="remove or rewrite"):
        promoted_predicates(previous, (second, first, later))
    with pytest.raises(ValueError, match="duplicate"):
        promoted_predicates(previous, (first, second, second))


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

    guarded = goal_consumption(
        goal,
        [],
        1,
        safety_consumptions=[
            {
                "total_tokens": 80,
                "context_tokens": 50,
                "output_tokens": 30,
                "tool_calls": 4,
                "attempts": 2,
                "child_agents": 2,
                "dollars": 0.5,
            },
            {"total_tokens": -1, "dollars": True},
        ],
        observed_at=goal.created_at,
    )
    assert guarded.tokens == 80
    assert guarded.context_tokens == 50
    assert guarded.output_tokens == 30
    assert guarded.tool_calls == 4
    assert guarded.attempts == 2
    assert guarded.child_agents == 2
    assert guarded.dollars == 0.5
    attempt_goal = replace(goal, budgets={"attempts": 2})
    assert exhausted_budget(attempt_goal, guarded) == "attempts"
    zero_discretionary = replace(
        goal,
        budgets={"tool_calls": 0, "child_agents": 0, "dollars": 0},
    )
    unused = GoalConsumption(
        turns=0,
        tokens=0,
        context_tokens=0,
        output_tokens=0,
        tool_calls=0,
        attempts=0,
        child_agents=0,
        dollars=0,
        elapsed_seconds=0,
    )
    assert exhausted_budget(zero_discretionary, unused) == ""
    assert (
        exhausted_budget(
            zero_discretionary,
            replace(unused, child_agents=1),
        )
        == "child_agents"
    )
    assert _safety_count("unknown", "tokens") == 0.0


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
    assert (
        exhausted_budget(
            goal,
            goal_consumption(goal, [], 10, observed_at=goal.created_at),
        )
        == ""
    )
    assert _timestamp("2026-01-01T00:00:00").tzinfo is not None
    assert (
        _token_count(
            {
                "nested": {
                    "input_tokens": 3,
                    "ignored_tokens": True,
                }
            }
        )
        == 3
    )
    assert _number(True) is None


def test_goal_seconds_bill_active_turn_time() -> None:
    goal = create_goal(
        new_uuid(),
        "bounded work",
        budgets={"seconds": 60},
    )
    created = _timestamp(goal.created_at)
    observed = created + datetime.timedelta(seconds=400)
    consumption = goal_consumption(
        goal,
        [],
        1,
        active_seconds=30.0,
        observed_at=observed.isoformat(),
    )

    assert consumption.elapsed_seconds == 30.0
    assert consumption.wall_seconds == 400.0
    assert exhausted_budget(goal, consumption) == ""
    saturated = goal_consumption(
        goal,
        [],
        1,
        active_seconds=60.0,
        observed_at=observed.isoformat(),
    )
    assert exhausted_budget(goal, saturated) == "seconds"
    wall_clock = goal_consumption(
        goal,
        [],
        1,
        observed_at=observed.isoformat(),
    )
    assert wall_clock.elapsed_seconds == 400.0
    assert wall_clock.wall_seconds == 400.0
