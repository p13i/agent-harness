from agent_harness.goals import create_goal
from agent_harness.goals import evaluate_goal
from agent_harness.goals import exhausted_budget
from agent_harness.goals import goal_consumption
from agent_harness.goals import make_evidence
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
