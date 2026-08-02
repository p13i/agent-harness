"""Deterministic provider-neutral context compilation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agent_harness.models import Evidence, Goal, Session, SessionEvent


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
    total_event_count: int | None = None,
    inherited_context: str = "",
    inherited_context_digest: str = "",
    compacted_history: dict[str, object] | None = None,
    durable_unresolved_decisions: Iterable[dict[str, object]] = (),
) -> CompiledContext:
    if max_input_tokens <= reserve_output_tokens:
        raise ValueError("context budget must reserve positive input capacity")
    event_list = list(events)
    canonical_event_count = len(event_list)
    if total_event_count is not None:
        canonical_event_count = max(canonical_event_count, total_event_count)
    evidence_list = list(evidence)
    instruction_list = [item for item in instructions if item.strip()]
    unresolved_decisions = _unresolved_decisions(event_list)
    known_decisions = {
        (str(item.get("kind", "")), str(item.get("id", "")))
        for item in unresolved_decisions
    }
    for item in durable_unresolved_decisions:
        key = (str(item.get("kind", "")), str(item.get("id", "")))
        if key in known_decisions:
            continue
        unresolved_decisions.append(dict(item))
        known_decisions.add(key)
    unresolved_decisions.sort(
        key=lambda item: (str(item.get("kind", "")), str(item.get("id", "")))
    )
    history = compacted_history
    if history is None:
        history = {}
    budget_chars = (max_input_tokens - reserve_output_tokens) * 4
    fixed_budget_chars = int(budget_chars * 0.70)
    fixed = _fixed_sections(
        session,
        goal,
        evidence_list,
        instruction_list,
        workspace_summary,
        unresolved_decisions,
        inherited_context,
        history,
        fixed_budget_chars,
    )
    fixed_text = "\n\n".join(fixed)
    if len(fixed_text) > fixed_budget_chars:
        raise ValueError("fixed context components exceed their bounded allocation")
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
    goal_digest_value: dict[str, object] = {}
    if goal is not None:
        goal_digest_value = goal.as_dict()
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
        "omitted_events": canonical_event_count - len(selected),
        "included_components": [
            "session",
            "goal",
            "instructions-plans-skills",
            "workspace-checkpoint-diff",
            "evidence",
            "unresolved-decisions",
            "recent-events",
        ],
        "component_digests": {
            "goal": _object_digest(goal_digest_value),
            "instructions_plans_skills": _object_digest(instruction_list),
            "workspace_checkpoint_diff": _text_digest(workspace_summary),
            "evidence": _object_digest([item.as_dict() for item in evidence_list]),
            "unresolved_decisions": _object_digest(unresolved_decisions),
            "recent_events": _object_digest(sequences),
            "fork_source_context": inherited_context_digest,
            "compacted_history": _object_digest(history),
        },
    }
    return CompiledContext(
        text=text,
        estimated_tokens=estimate_tokens(text),
        included_sequences=tuple(sequences),
        omitted_events=canonical_event_count - len(selected),
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


def workspace_context_artifacts(workspace: Path) -> tuple[str, ...]:
    result = list(workspace_instructions(workspace))
    retained = sum(len(item) for item in result)
    patterns = (
        "plans/**/*.md",
        ".agents/plans/**/*.md",
        "skills/**/SKILL.md",
        ".agents/skills/**/SKILL.md",
        ".codex/skills/**/SKILL.md",
    )
    root = workspace.resolve()
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.glob(pattern))
    for path in sorted(paths):
        if retained >= 200_000:
            break
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        remaining = 200_000 - retained
        content = resolved.read_text(
            encoding="utf-8",
            errors="replace",
        )[:remaining]
        relative = resolved.relative_to(root).as_posix()
        result.append("# " + relative + "\n\n" + content)
        retained += len(content)
    return tuple(result)


def _fixed_sections(
    session: Session,
    goal: Goal | None,
    evidence: list[Evidence],
    instructions: list[str],
    workspace_summary: str,
    unresolved_decisions: list[dict[str, object]],
    inherited_context: str,
    compacted_history: dict[str, object],
    fixed_budget_chars: int,
) -> list[str]:
    session_section = (
        "# Harness session\n\n"
        + "Session UUID: "
        + session.session_id
        + "\n\nWorkspace: "
        + session.worktree
    )
    continuity_section = (
        "# Continuity contract\n\n"
        "Continue the objective from this canonical observable state. "
        "Do not claim access to hidden reasoning from a prior provider. "
        "Inspect the workspace and harness history when more detail is needed."
    )
    minimum_chars = len(session_section) + len(continuity_section) + 2
    if minimum_chars > fixed_budget_chars:
        raise ValueError("context budget cannot fit the continuity contract")
    components: list[tuple[str, str, int]] = []
    if goal is not None:
        goal_payload = goal.as_dict()
        components.append(
            (
                "Goal",
                "```json\n"
                + json.dumps(goal_payload, indent=2, sort_keys=True)
                + "\n```",
                20,
            )
        )
    if inherited_context:
        components.append(("Fork source context", inherited_context, 15))
    if compacted_history:
        components.append(
            (
                "Compacted observable history",
                "```json\n"
                + json.dumps(compacted_history, indent=2, sort_keys=True)
                + "\n```",
                15,
            )
        )
    if instructions:
        components.append(("Instructions", "\n\n".join(instructions), 20))
    if workspace_summary:
        components.append(("Workspace checkpoint", workspace_summary, 15))
    if evidence:
        payload = [item.as_dict() for item in evidence]
        components.append(
            (
                "Evidence",
                "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```",
                10,
            )
        )
    if unresolved_decisions:
        components.append(
            (
                "Unresolved decisions",
                "```json\n"
                + json.dumps(unresolved_decisions, indent=2, sort_keys=True)
                + "\n```",
                15,
            )
        )
    separators = 2 * (len(components) + 1)
    available = fixed_budget_chars - minimum_chars - separators
    if available < 0:
        raise ValueError("context budget cannot fit fixed component headers")
    total_weight = sum(item[2] for item in components)
    sections = [session_section]
    for title, content, weight in components:
        allocation = available
        if total_weight > 0:
            allocation = int(available * weight / total_weight)
        sections.append(_bounded_section(title, content, allocation))
    sections.append(continuity_section)
    return sections


def _bounded_section(title: str, content: str, allocation: int) -> str:
    header = "# " + title + "\n\n"
    if len(header) + len(content) <= allocation:
        return header + content
    digest = _text_digest(content)
    marker = (
        "\n\n[Compacted: sha256="
        + digest
        + " original_chars="
        + str(len(content))
        + "]"
    )
    retained = allocation - len(header) - len(marker)
    if retained < 0:
        minimal = "# " + title + " [sha256=" + digest + "]"
        if len(minimal) > allocation:
            raise ValueError("context budget cannot fit the " + title + " digest")
        return minimal
    return header + content[:retained] + marker


def _render_event(event: SessionEvent) -> str:
    label = event.event_type
    if event.role:
        label += " (" + event.role + ")"
    body = event.text
    if not body:
        body = json.dumps(event.metadata, sort_keys=True)
    elif event.metadata:
        body += "\n\nMetadata: " + json.dumps(event.metadata, sort_keys=True)
    return "## Event " + str(event.sequence) + ": " + label + "\n\n" + body


def _unresolved_decisions(events: list[SessionEvent]) -> list[dict[str, object]]:
    resolved: set[tuple[str, str]] = set()
    for event in events:
        if event.event_type == "approval.resolved":
            resolved.add(("approval", str(event.metadata.get("approval_id", ""))))
        if event.event_type == "reconciliation.resolved":
            resolved.add(
                (
                    "reconciliation",
                    str(event.metadata.get("reconciliation_id", "")),
                )
            )
    pending: list[dict[str, object]] = []
    for event in events:
        kind = ""
        identifier = ""
        if event.event_type == "approval.requested":
            kind = "approval"
            identifier = str(event.metadata.get("approval_id", ""))
        if event.event_type == "reconciliation.requested":
            kind = "reconciliation"
            identifier = str(event.metadata.get("reconciliation_id", ""))
        if not kind or not identifier or (kind, identifier) in resolved:
            continue
        pending.append(
            {
                "kind": kind,
                "id": identifier,
                "sequence": event.sequence,
                "metadata": event.metadata,
            }
        )
    return pending


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _object_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
