"""Goal construction and evidence-backed completion."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import Any

from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Evidence
from agent_harness.models import Goal
from agent_harness.models import GoalKind
from agent_harness.models import GoalStatus


@dataclass(frozen=True)
class GoalEvaluation:
    satisfied: bool
    matched: tuple[dict[str, Any], ...]
    missing: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GoalConsumption:
    turns: int
    tokens: float
    dollars: float
    elapsed_seconds: float


def create_goal(
    session_id: str,
    objective: str,
    *,
    kind: str = GoalKind.FINITE,
    constraints: tuple[str, ...] = (),
    predicates: tuple[dict[str, Any], ...] = (),
    budgets: dict[str, Any] | None = None,
) -> Goal:
    normalized = objective.strip()
    if not normalized:
        raise ValueError("goal objective cannot be empty")
    if kind not in {GoalKind.FINITE, GoalKind.INVARIANT}:
        raise ValueError("unsupported goal kind")
    if budgets is None:
        budgets = {}
    now = utc_now()
    return Goal(
        goal_id=new_uuid(),
        session_id=session_id,
        kind=kind,
        objective=normalized,
        status=GoalStatus.ACTIVE,
        constraints=constraints,
        predicates=predicates,
        budgets=validate_budgets(budgets),
        created_at=now,
        updated_at=now,
    )


def validate_budgets(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {"tokens", "seconds", "dollars", "turns"}
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError("unsupported goal budgets: " + names)
    result: dict[str, Any] = {}
    for name, raw in value.items():
        if isinstance(raw, bool):
            raise ValueError(name + " budget must be numeric")
        if not isinstance(raw, (int, float)):
            raise ValueError(name + " budget must be numeric")
        if raw < 0:
            raise ValueError(name + " budget cannot be negative")
        result[name] = raw
    return result


def make_evidence(
    goal_id: str,
    evidence_type: str,
    subject: str,
    outcome: str,
    value: dict[str, Any] | None = None,
) -> Evidence:
    if value is None:
        value = {}
    if not evidence_type:
        raise ValueError("evidence type cannot be empty")
    if not outcome:
        raise ValueError("evidence outcome cannot be empty")
    return Evidence(
        evidence_id=new_uuid(),
        goal_id=goal_id,
        evidence_type=evidence_type,
        subject=subject,
        outcome=outcome,
        value=value,
        created_at=utc_now(),
    )


def evaluate_goal(goal: Goal, evidence: list[Evidence]) -> GoalEvaluation:
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for predicate in goal.predicates:
        if _predicate_satisfied(predicate, evidence):
            matched.append(predicate)
        else:
            missing.append(predicate)
    satisfied = bool(goal.predicates) and not missing
    if goal.kind == GoalKind.INVARIANT:
        satisfied = False
    return GoalEvaluation(
        satisfied=satisfied,
        matched=tuple(matched),
        missing=tuple(missing),
    )


def goal_consumption(
    goal: Goal,
    command_results: list[dict[str, Any]],
    turn_count: int,
    *,
    observed_at: str | None = None,
) -> GoalConsumption:
    tokens = 0.0
    dollars = 0.0
    for result in command_results:
        usage = result.get("usage", {})
        tokens += _token_count(usage)
        dollars += _dollar_count(usage)
    now = datetime.datetime.now(datetime.UTC)
    if observed_at:
        now = _timestamp(observed_at)
    created = _timestamp(goal.created_at)
    elapsed = max(0.0, (now - created).total_seconds())
    return GoalConsumption(
        turns=turn_count,
        tokens=tokens,
        dollars=dollars,
        elapsed_seconds=elapsed,
    )


def exhausted_budget(
    goal: Goal,
    consumption: GoalConsumption,
) -> str:
    values = {
        "turns": float(consumption.turns),
        "tokens": consumption.tokens,
        "dollars": consumption.dollars,
        "seconds": consumption.elapsed_seconds,
    }
    for name in ("turns", "tokens", "dollars", "seconds"):
        limit = goal.budgets.get(name)
        if not isinstance(limit, (int, float)):
            continue
        if isinstance(limit, bool):
            continue
        if values[name] >= float(limit):
            return name
    return ""


def _predicate_satisfied(
    predicate: dict[str, Any],
    evidence: list[Evidence],
) -> bool:
    predicate_type = str(predicate.get("type", ""))
    subject = str(predicate.get("subject", ""))
    required_outcome = str(predicate.get("outcome", "passed"))
    for item in evidence:
        if item.evidence_type != predicate_type:
            continue
        if subject and item.subject != subject:
            continue
        if item.outcome != required_outcome:
            continue
        expected = predicate.get("equals")
        if expected is not None:
            field = str(predicate.get("field", "value"))
            actual = item.value.get(field)
            if actual != expected:
                continue
        return True
    return False


def _timestamp(value: str) -> datetime.datetime:
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed.astimezone(datetime.UTC)


def _token_count(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    for key in ("total_tokens", "totalTokens"):
        total = _number(value.get(key))
        if total is not None:
            return total
    total = 0.0
    for key, child in value.items():
        normalized = key.casefold().replace("-", "_")
        if normalized.endswith("_tokens") or normalized.endswith(
            "tokencount"
        ):
            number = _number(child)
            if number is not None:
                total += number
            continue
        total += _token_count(child)
    return total


def _dollar_count(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    total = 0.0
    for key, child in value.items():
        normalized = key.casefold()
        if normalized in {
            "total_cost_usd",
            "cost_usd",
            "totalcostusd",
        }:
            number = _number(child)
            if number is not None:
                total += number
            continue
        total += _dollar_count(child)
    return total


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
