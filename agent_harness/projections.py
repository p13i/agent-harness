"""Inspectable, non-canonical projections of durable session state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent_harness.context import CompiledContext
from agent_harness.models import Goal
from agent_harness.models import SessionEvent


def write_session_projections(
    destination: Path,
    payload: dict[str, Any],
    context: CompiledContext,
    events: list[SessionEvent],
    goal: Goal | None,
) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    session = payload.get("session", {})
    if not isinstance(session, dict):
        raise ValueError("session export has no session")
    session_id = str(session.get("session_id", ""))
    if not session_id:
        raise ValueError("session export has no session identifier")

    export_path = destination / (session_id + ".json")
    context_path = destination / (session_id + ".run-context.gpt.json")
    transcript_path = destination / (session_id + ".transcript.jsonl")
    markdown_path = destination / (session_id + ".transcript.md")
    goal_path = destination / (session_id + ".goal.gpt.json")

    _write_json(export_path, payload)
    context_value = dict(context.projection)
    context_value["compiled_context"] = context.text
    _write_json(context_path, context_value)
    _write_text(transcript_path, _jsonl(events))
    _write_text(markdown_path, _markdown(session, events, goal))
    goal_value: dict[str, Any] = {
        "schema": "p13i/agent-harness/goal-projection/v1",
        "session_id": session_id,
        "goal": None,
    }
    if goal is not None:
        goal_value["goal"] = goal.as_dict()
    _write_json(goal_path, goal_value)
    return {
        "export": export_path,
        "run_context": context_path,
        "transcript_jsonl": transcript_path,
        "transcript_markdown": markdown_path,
        "goal": goal_path,
    }


def _jsonl(events: list[SessionEvent]) -> str:
    lines = [
        json.dumps(
            event.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        for event in events
    ]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _markdown(
    session: dict[str, Any],
    events: list[SessionEvent],
    goal: Goal | None,
) -> str:
    lines = [
        "# " + str(session.get("name", "Agent session")),
        "",
        "- Session: `" + str(session.get("session_id", "")) + "`",
        "- Lifecycle: `" + str(session.get("lifecycle", "")) + "`",
        "- Provider: `" + str(session.get("active_provider", "")) + "`",
    ]
    if goal is not None:
        lines.extend(
            [
                "",
                "## Goal",
                "",
                goal.objective,
                "",
                "Status: `" + goal.status + "`",
            ]
        )
    lines.extend(["", "## Transcript", ""])
    for event in events:
        label = event.event_type
        if event.role:
            label += " · " + event.role
        lines.extend(
            [
                "### " + str(event.sequence) + " · " + label,
                "",
                event.text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
