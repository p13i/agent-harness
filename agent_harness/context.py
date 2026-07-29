"""Deterministic provider-neutral context compilation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from agent_harness.models import Evidence
from agent_harness.models import Goal
from agent_harness.models import Session
from agent_harness.models import SessionEvent


@dataclass(frozen=True)
class CompiledContext:
    text: str
    estimated_tokens: int
    included_sequences: tuple[int, ...]
    omitted_events: int
    projection: dict[str, object]


def compile_context(
    session: Session,
    events: Iterable[SessionEvent],
    *,
    goal: Goal | None = None,
    evidence: Iterable[Evidence] = (),
    instructions: Iterable[str] = (),
    workspace_summary: str = "",
    max_input_tokens: int = 100_000,
    reserve_output_tokens: int = 8_000,
) -> CompiledContext:
    if max_input_tokens <= reserve_output_tokens:
        raise ValueError("context budget must reserve positive input capacity")
    event_list = list(events)
    evidence_list = list(evidence)
    instruction_list = [item for item in instructions if item.strip()]
    budget_chars = (max_input_tokens - reserve_output_tokens) * 4
    fixed = _fixed_sections(
        session,
        goal,
        evidence_list,
        instruction_list,
        workspace_summary,
    )
    fixed_text = "\n\n".join(fixed)
    remaining = max(0, budget_chars - len(fixed_text))
    selected: list[tuple[SessionEvent, str]] = []
    for event in reversed(event_list):
        rendered = _render_event(event)
        cost = len(rendered) + 2
        if cost > remaining:
            continue
        selected.append((event, rendered))
        remaining -= cost
    selected.reverse()
    sections = list(fixed)
    sequences: list[int] = []
    for event, rendered in selected:
        sections.append(rendered)
        sequences.append(event.sequence)
    text = "\n\n".join(sections)
    projection: dict[str, object] = {
        "schema": "p13i/agent-harness/run-context/v1",
        "session_id": session.session_id,
        "provider": session.active_provider,
        "model": session.model,
        "effort": session.effort,
        "goal_id": session.goal_id,
        "max_input_tokens": max_input_tokens,
        "reserve_output_tokens": reserve_output_tokens,
        "estimated_tokens": estimate_tokens(text),
        "included_event_sequences": sequences,
        "omitted_events": len(event_list) - len(selected),
    }
    return CompiledContext(
        text=text,
        estimated_tokens=estimate_tokens(text),
        included_sequences=tuple(sequences),
        omitted_events=len(event_list) - len(selected),
        projection=projection,
    )


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def workspace_instructions(workspace: Path) -> tuple[str, ...]:
    result: list[str] = []
    for name in ("AGENTS.md", "CLAUDE.md"):
        path = workspace / name
        if not path.is_file():
            continue
        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        if len(content) > 200_000:
            content = content[:200_000]
        result.append("# " + name + "\n\n" + content)
    return tuple(result)


def _fixed_sections(
    session: Session,
    goal: Goal | None,
    evidence: list[Evidence],
    instructions: list[str],
    workspace_summary: str,
) -> list[str]:
    sections = [
        "# Harness session",
        "Session UUID: " + session.session_id,
        "Workspace: " + session.worktree,
    ]
    if goal is not None:
        goal_payload = goal.as_dict()
        sections.append(
            "# Goal\n\n```json\n"
            + json.dumps(goal_payload, indent=2, sort_keys=True)
            + "\n```"
        )
    if instructions:
        sections.append("# Instructions\n\n" + "\n\n".join(instructions))
    if workspace_summary:
        sections.append("# Workspace checkpoint\n\n" + workspace_summary)
    if evidence:
        payload = [item.as_dict() for item in evidence]
        sections.append(
            "# Evidence\n\n```json\n"
            + json.dumps(payload, indent=2, sort_keys=True)
            + "\n```"
        )
    sections.append(
        "# Continuity contract\n\n"
        "Continue the objective from this canonical observable state. "
        "Do not claim access to hidden reasoning from a prior provider. "
        "Inspect the workspace and harness history when more detail is needed."
    )
    return sections


def _render_event(event: SessionEvent) -> str:
    label = event.event_type
    if event.role:
        label += " (" + event.role + ")"
    body = event.text
    if not body:
        body = json.dumps(event.metadata, sort_keys=True)
    return "## Event " + str(event.sequence) + ": " + label + "\n\n" + body
