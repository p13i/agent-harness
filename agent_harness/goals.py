"""Goal construction and evidence-backed completion."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any

from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import Evidence, Goal, GoalKind, GoalStatus, Milestone

FINITE_COMPLETION_POLICY = "evidence-all"
INVARIANT_COMPLETION_POLICY = "never"
INCIDENT_POLICY = "recover-then-pause"
SUPPORTED_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


@dataclass(frozen=True)
class GoalEvaluation:
    satisfied: bool
    matched: tuple[dict[str, Any], ...]
    missing: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GoalConsumption:
    turns: int
    tokens: float
    context_tokens: float
    output_tokens: float
    tool_calls: float
    attempts: float
    child_agents: float
    dollars: float
    elapsed_seconds: float


def goal_contract(goal: Goal) -> dict[str, Any]:
    return {
        "kind": goal.kind,
        "objective": goal.objective,
        "constraints": list(goal.constraints),
        "predicates": list(goal.predicates),
        "milestones": [
            {
                "milestone_id": item.milestone_id,
                "title": item.title,
                "dependencies": list(item.dependencies),
                "predicates": list(item.predicates),
                "position": item.position,
            }
            for item in goal.milestones
        ],
        "budgets": goal.budgets,
        "permitted_providers": list(goal.permitted_providers),
        "permitted_efforts": list(goal.permitted_efforts),
        "max_concurrency": goal.max_concurrency,
        "completion_policy": goal.completion_policy,
        "incident_policy": goal.incident_policy,
    }


def goal_contract_digest(goal: Goal) -> str:
    encoded = json.dumps(
        goal_contract(goal),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_goal(
    session_id: str,
    objective: str,
    *,
    kind: str = GoalKind.FINITE,
    constraints: tuple[str, ...] = (),
    predicates: tuple[dict[str, Any], ...] = (),
    milestones: tuple[dict[str, Any], ...] = (),
    budgets: dict[str, Any] | None = None,
    permitted_providers: tuple[str, ...] = (),
    permitted_efforts: tuple[str, ...] = (),
    max_concurrency: int = 1,
    completion_policy: str = "",
    incident_policy: str = INCIDENT_POLICY,
) -> Goal:
    normalized = objective.strip()
    if not normalized:
        raise ValueError("goal objective cannot be empty")
    if kind not in {GoalKind.FINITE, GoalKind.INVARIANT}:
        raise ValueError("unsupported goal kind")
    if budgets is None:
        budgets = {}
    normalized_completion_policy = completion_policy
    if not normalized_completion_policy:
        normalized_completion_policy = FINITE_COMPLETION_POLICY
        if kind == GoalKind.INVARIANT:
            normalized_completion_policy = INVARIANT_COMPLETION_POLICY
    _validate_goal_policy(
        kind,
        permitted_providers,
        permitted_efforts,
        max_concurrency,
        normalized_completion_policy,
        incident_policy,
    )
    normalized_milestones = _milestones(milestones)
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
        milestones=normalized_milestones,
        permitted_providers=_unique_values(permitted_providers),
        permitted_efforts=_unique_values(permitted_efforts),
        max_concurrency=max_concurrency,
        completion_policy=normalized_completion_policy,
        incident_policy=incident_policy,
    )


def _validate_goal_policy(
    kind: str,
    permitted_providers: tuple[str, ...],
    permitted_efforts: tuple[str, ...],
    max_concurrency: int,
    completion_policy: str,
    incident_policy: str,
) -> None:
    _unique_values(permitted_providers)
    efforts = _unique_values(permitted_efforts)
    if set(efforts) - SUPPORTED_EFFORTS:
        raise ValueError("goal contains an unsupported permitted effort")
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise ValueError("goal max_concurrency must be an integer")
    if max_concurrency < 1 or max_concurrency > 2:
        raise ValueError("goal max_concurrency must be between 1 and 2")
    expected_completion = FINITE_COMPLETION_POLICY
    if kind == GoalKind.INVARIANT:
        expected_completion = INVARIANT_COMPLETION_POLICY
    if completion_policy != expected_completion:
        raise ValueError("goal completion policy does not match its kind")
    if incident_policy != INCIDENT_POLICY:
        raise ValueError("unsupported goal incident policy")


def _unique_values(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value or any(character.isspace() for character in value):
            raise ValueError("goal policy identifiers cannot contain whitespace")
        if value not in result:
            result.append(value)
    return tuple(result)


def _milestones(values: tuple[dict[str, Any], ...]) -> tuple[Milestone, ...]:
    result: list[Milestone] = []
    known: set[str] = set()
    for position, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError("goal milestones must be objects")
        milestone_id = str(value.get("milestone_id", value.get("id", ""))).strip()
        if not milestone_id or any(character.isspace() for character in milestone_id):
            raise ValueError("milestone identifier is invalid")
        if milestone_id in known:
            raise ValueError("milestone identifier is duplicated")
        title = str(value.get("title", "")).strip()
        if not title:
            raise ValueError("milestone title is required")
        dependencies_value = value.get("dependencies", [])
        predicates_value = value.get("predicates", [])
        if not isinstance(dependencies_value, list):
            raise ValueError("milestone dependencies must be a list")
        if not isinstance(predicates_value, list):
            raise ValueError("milestone predicates must be a list")
        dependencies = tuple(str(item) for item in dependencies_value)
        if not set(dependencies).issubset(known):
            raise ValueError("milestone dependencies must reference prior milestones")
        predicates = tuple(item for item in predicates_value if isinstance(item, dict))
        if len(predicates) != len(predicates_value) or not predicates:
            raise ValueError("milestone requires typed evidence predicates")
        status = "active"
        if dependencies:
            status = "blocked"
        result.append(
            Milestone(
                milestone_id=milestone_id,
                title=title,
                status=status,
                dependencies=dependencies,
                predicates=predicates,
                position=position,
            )
        )
        known.add(milestone_id)
    return tuple(result)


def _milestone_definition(milestone: Milestone) -> dict[str, Any]:
    return {
        "milestone_id": milestone.milestone_id,
        "title": milestone.title,
        "dependencies": list(milestone.dependencies),
        "predicates": list(milestone.predicates),
        "position": milestone.position,
    }


def validate_budgets(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "tokens",
        "context_tokens",
        "output_tokens",
        "tool_calls",
        "attempts",
        "child_agents",
        "seconds",
        "dollars",
        "turns",
    }
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
        if not math.isfinite(float(raw)):
            raise ValueError(name + " budget must be finite")
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
    milestones = evaluate_milestones(goal, evidence)
    milestones_satisfied = all(
        item.status == GoalStatus.COMPLETE for item in milestones
    )
    if not milestones:
        milestones_satisfied = True
    satisfied = bool(goal.predicates) and not missing and milestones_satisfied
    if goal.completion_policy == INVARIANT_COMPLETION_POLICY:
        satisfied = False
    return GoalEvaluation(
        satisfied=satisfied,
        matched=tuple(matched),
        missing=tuple(missing),
    )


def evaluate_milestones(
    goal: Goal,
    evidence: list[Evidence],
) -> tuple[Milestone, ...]:
    completed: set[str] = set()
    result: list[Milestone] = []
    for milestone in sorted(goal.milestones, key=lambda item: item.position):
        dependencies_satisfied = set(milestone.dependencies).issubset(completed)
        status = "blocked"
        if dependencies_satisfied:
            status = "active"
            predicates_satisfied = all(
                _predicate_satisfied(predicate, evidence)
                for predicate in milestone.predicates
            )
            if predicates_satisfied:
                status = GoalStatus.COMPLETE
                completed.add(milestone.milestone_id)
        result.append(replace(milestone, status=status))
    return tuple(result)


def promoted_milestones(
    previous: Goal,
    next_goal: Goal,
) -> tuple[Milestone, ...]:
    previous_by_id = {item.milestone_id: item for item in previous.milestones}
    result: list[Milestone] = []
    for milestone in next_goal.milestones:
        prior = previous_by_id.get(milestone.milestone_id)
        if prior is None:
            result.append(milestone)
            continue
        if _milestone_definition(prior) != _milestone_definition(milestone):
            raise ValueError("goal promotion cannot rewrite a milestone")
        result.append(replace(milestone, status=prior.status))
    if not set(previous_by_id).issubset(
        {item.milestone_id for item in next_goal.milestones}
    ):
        raise ValueError("goal promotion cannot remove a milestone")
    if len(result) == len(previous.milestones):
        raise ValueError("goal promotion requires a later milestone")
    return tuple(result)


def promoted_predicates(
    previous: tuple[dict[str, Any], ...],
    later: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    if len(later) <= len(previous):
        raise ValueError("goal promotion requires a later predicate")
    if later[: len(previous)] != previous:
        raise ValueError("goal promotion cannot remove or rewrite predicates")
    digests = [
        hashlib.sha256(
            json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for item in later
    ]
    if len(digests) != len(set(digests)):
        raise ValueError("goal promotion cannot duplicate predicates")
    return later


def goal_consumption(
    goal: Goal,
    command_results: list[dict[str, Any]],
    turn_count: int,
    *,
    safety_consumptions: list[dict[str, Any]] | None = None,
    observed_at: str | None = None,
) -> GoalConsumption:
    tokens = 0.0
    context_tokens = 0.0
    output_tokens = 0.0
    tool_calls = 0.0
    attempts = 0.0
    child_agents = 0.0
    dollars = 0.0
    if safety_consumptions is None:
        for result in command_results:
            usage = result.get("usage", {})
            tokens += _token_count(usage)
            dollars += _dollar_count(usage)
    else:
        for consumption in safety_consumptions:
            tokens += _safety_count(consumption, "total_tokens")
            context_tokens += _safety_count(
                consumption,
                "context_tokens",
            )
            output_tokens += _safety_count(
                consumption,
                "output_tokens",
            )
            tool_calls += _safety_count(consumption, "tool_calls")
            attempts += _safety_count(consumption, "attempts")
            child_agents += _safety_count(
                consumption,
                "child_agents",
            )
            dollars += _safety_count(consumption, "dollars")
    now = datetime.datetime.now(datetime.UTC)
    if observed_at:
        now = _timestamp(observed_at)
    created = _timestamp(goal.created_at)
    elapsed = max(0.0, (now - created).total_seconds())
    return GoalConsumption(
        turns=turn_count,
        tokens=tokens,
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
        attempts=attempts,
        child_agents=child_agents,
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
        "context_tokens": consumption.context_tokens,
        "output_tokens": consumption.output_tokens,
        "tool_calls": consumption.tool_calls,
        "attempts": consumption.attempts,
        "child_agents": consumption.child_agents,
        "dollars": consumption.dollars,
        "seconds": consumption.elapsed_seconds,
    }
    for name in (
        "turns",
        "tokens",
        "context_tokens",
        "output_tokens",
        "attempts",
        "seconds",
    ):
        limit = goal.budgets.get(name)
        if not isinstance(limit, (int, float)):
            continue
        if isinstance(limit, bool):
            continue
        if values[name] >= float(limit):
            return name
    for name in ("tool_calls", "child_agents", "dollars"):
        limit = goal.budgets.get(name)
        if not isinstance(limit, (int, float)):
            continue
        if isinstance(limit, bool):
            continue
        if values[name] > float(limit):
            return name
    return ""


def _safety_count(value: object, name: str) -> float:
    if not isinstance(value, dict):
        return 0.0
    number = _number(value.get(name))
    if number is None:
        return 0.0
    return max(0.0, number)


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
        if normalized.endswith("_tokens") or normalized.endswith("tokencount"):
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
