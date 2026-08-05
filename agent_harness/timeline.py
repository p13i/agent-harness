"""Read-only session timeline projection for observability.

The timeline is a deterministic read projection over existing
storage rows: turns and provider attempts, routing decisions,
goal budget consumption (active versus wall seconds),
routing-unavailable deferrals, and guard incidents. It issues
only read queries and never writes to the hot path; the
underlying tables remain the source of truth.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from agent_harness.goals import goal_consumption
from agent_harness.ids import require_uuid
from agent_harness.storage import StateStore

TIMELINE_SCHEMA = "p13i/agent-harness/timeline/v1"
ROUTING_DEFERRAL_CODE = "E_PROVIDER_UNAVAILABLE"


def project_timeline(
    store: StateStore,
    session_id: str,
    *,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Project one session's durable rows into a timeline."""
    require_uuid(session_id, "session_id")
    store.get_session(session_id)
    if now is None:
        now = datetime.datetime.now(datetime.UTC)
    turns = [
        _turn(row, now) for row in store.presentation_turn_rows(session_id)
    ]
    decisions = [
        _decision(row) for row in store.routing_decisions(session_id)
    ]
    deferrals = [
        _deferral(row)
        for row in store.failed_command_results(session_id)
        if row["result"].get("code") == ROUTING_DEFERRAL_CODE
    ]
    return {
        "schema": TIMELINE_SCHEMA,
        "session_id": session_id,
        "generated_at": now.isoformat(),
        "turns": turns,
        "routing_decisions": decisions,
        "budget": _budget(store, session_id, now),
        "deferrals": deferrals,
        "guard_incidents": store.guard_incidents(session_id),
    }


def render_timeline(timeline: dict[str, Any]) -> str:
    """Render a timeline projection as a human-readable report."""
    lines = [
        "# Session timeline",
        "",
        "- Session: `" + str(timeline["session_id"]) + "`",
        "- Schema: `" + str(timeline["schema"]) + "`",
        "- Generated: " + str(timeline["generated_at"]),
        "",
        "## Turns",
    ]
    turns = timeline["turns"]
    if turns:
        for turn in turns:
            lines.append("")
            lines.append(
                "- `"
                + str(turn["turn_id"])
                + "` · "
                + _provider_label(turn)
                + " · "
                + str(turn["turn_status"])
                + " · "
                + _seconds(turn["active_seconds"])
                + " active"
            )
            detail = []
            if turn["command_id"]:
                detail.append(
                    "command `"
                    + str(turn["command_id"])
                    + "` ("
                    + str(turn["command_status"] or "unknown")
                    + ")"
                )
            if turn["attempt_status"]:
                detail.append("attempt " + str(turn["attempt_status"]))
            if turn["replay_of"]:
                detail.append("replay of `" + str(turn["replay_of"]) + "`")
            if detail:
                lines.append("  " + " · ".join(detail))
    else:
        lines.extend(["", "- none"])
    lines.extend(["", "## Routing decisions"])
    decisions = timeline["routing_decisions"]
    if decisions:
        for decision in decisions:
            lines.append("")
            binding = _percent(decision["binding_percent"])
            ceiling = _percent(decision["binding_ceiling"])
            lines.append(
                "- "
                + str(decision["provider"])
                + "/"
                + str(decision["model"])
                + " effort "
                + str(decision["effort"])
                + " · binding "
                + binding
                + " · ceiling "
                + ceiling
                + " · turn `"
                + str(decision["turn_id"])
                + "`"
            )
            if decision["reason"]:
                lines.append("  " + str(decision["reason"]))
    else:
        lines.extend(["", "- none"])
    lines.extend(["", "## Budget"])
    budget = timeline["budget"]
    if budget:
        consumption = budget["consumption"]
        lines.extend(
            [
                "",
                "- Turns: " + str(consumption["turns"]),
                "- Tokens: "
                + _number(consumption["tokens"])
                + " (context "
                + _number(consumption["context_tokens"])
                + ", output "
                + _number(consumption["output_tokens"])
                + ")",
                "- Active seconds: "
                + _seconds(consumption["active_seconds"])
                + " · wall seconds: "
                + _seconds(consumption["wall_seconds"]),
            ]
        )
        budgets = budget.get("budgets", {})
        if budgets:
            lines.append("- Budgets: `" + _canonical(budgets) + "`")
    else:
        lines.extend(["", "- no goal recorded"])
    lines.extend(["", "## Deferrals"])
    deferrals = timeline["deferrals"]
    if deferrals:
        for deferral in deferrals:
            lines.append("")
            lines.append(
                "- command `"
                + str(deferral["command_id"])
                + "` · "
                + str(deferral["code"])
                + " · "
                + str(deferral["message"])
            )
    else:
        lines.extend(["", "- none"])
    lines.extend(["", "## Guard incidents"])
    incidents = timeline["guard_incidents"]
    if incidents:
        for incident in incidents:
            lines.append("")
            lines.append(
                "- "
                + str(incident["reason"])
                + " → "
                + str(incident["action"])
                + " · command `"
                + str(incident["command_id"])
                + "` · attempt `"
                + str(incident["attempt_id"])
                + "` · "
                + str(incident["created_at"])
            )
    else:
        lines.extend(["", "- none"])
    return "\n".join(lines).rstrip() + "\n"


def _turn(row: dict[str, Any], now: datetime.datetime) -> dict[str, Any]:
    started = str(row.get("started_at", ""))
    completed = str(row.get("completed_at", ""))
    active_seconds = 0.0
    if started:
        end = now
        if completed:
            end = _timestamp(completed)
        active_seconds = max(0.0, (end - _timestamp(started)).total_seconds())
    usage: dict[str, Any] = {}
    result = row.get("command_result", {})
    if isinstance(result, dict):
        raw_usage = result.get("usage", {})
        if isinstance(raw_usage, dict):
            usage = raw_usage
    return {
        "turn_id": str(row.get("turn_id", "")),
        "attempt_id": str(row.get("attempt_id", "")),
        "provider": str(row.get("provider", "")),
        "model": str(row.get("model", "")),
        "effort": str(row.get("effort", "")),
        "turn_status": str(row.get("turn_status", "")),
        "attempt_status": str(row.get("attempt_status", "")),
        "replay_of": str(row.get("replay_of", "")),
        "started_at": started,
        "completed_at": completed,
        "active_seconds": active_seconds,
        "command_id": str(row.get("command_id", "")),
        "command_status": str(row.get("command_status", "")),
        "usage": usage,
    }


def _decision(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["payload"]
    return {
        "decision_id": row["decision_id"],
        "turn_id": row["turn_id"],
        "provider": row["provider"],
        "model": row["model"],
        "effort": row["effort"],
        "binding_percent": payload.get("binding_percent"),
        "binding_ceiling": payload.get("binding_ceiling"),
        "credits_engaged": bool(payload.get("credits_engaged", False)),
        "workload": str(payload.get("workload", "")),
        "execution_profile": str(payload.get("execution_profile", "")),
        "reason": str(payload.get("reason", "")),
        "ranked": payload.get("ranked", []),
        "rejected": payload.get("rejected", []),
        "created_at": row["created_at"],
    }


def _budget(
    store: StateStore,
    session_id: str,
    now: datetime.datetime,
) -> dict[str, Any]:
    goal = store.goal_for_session(session_id)
    if goal is None:
        return {}
    consumptions = [
        envelope["consumption"]
        for envelope in store.session_envelopes(session_id)
    ]
    consumption = goal_consumption(
        goal,
        [],
        store.countable_turn_count(session_id),
        safety_consumptions=consumptions,
        active_seconds=store.active_turn_seconds(session_id, now),
        observed_at=now.isoformat(),
    )
    return {
        "goal_id": goal.goal_id,
        "budgets": goal.budgets,
        "consumption": {
            "turns": consumption.turns,
            "tokens": consumption.tokens,
            "context_tokens": consumption.context_tokens,
            "output_tokens": consumption.output_tokens,
            "tool_calls": consumption.tool_calls,
            "attempts": consumption.attempts,
            "child_agents": consumption.child_agents,
            "dollars": consumption.dollars,
            "active_seconds": consumption.elapsed_seconds,
            "wall_seconds": consumption.wall_seconds,
        },
    }


def _deferral(row: dict[str, Any]) -> dict[str, Any]:
    result = row["result"]
    return {
        "command_id": row["command_id"],
        "command_type": row["command_type"],
        "code": str(result.get("code", "")),
        "message": str(result.get("message", "")),
        "retryable": bool(result.get("retryable", False)),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _provider_label(turn: dict[str, Any]) -> str:
    provider = str(turn["provider"])
    if not provider:
        return "harness"
    model = str(turn["model"])
    if not model:
        return provider
    return provider + "/" + model


def _timestamp(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value)


def _seconds(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "0.0s"
    return str(round(float(value), 1)) + "s"


def _percent(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return str(round(float(value), 1)) + "%"


def _number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "0"
    number = float(value)
    if number == int(number):
        return str(int(number))
    return str(round(number, 2))


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
