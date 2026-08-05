"""Canonical provider-neutral transcript projection and rendering.

The durable events table stays the source of truth. A transcript is
a deterministic projection of that log, with blob dereference and
per-turn provider attribution from the turns and provider_attempts
tables, so it can be rebuilt on demand at any time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from agent_harness.blobs import BlobStore
from agent_harness.ids import require_uuid
from agent_harness.models import SessionEvent
from agent_harness.storage import StateStore

TRANSCRIPT_SCHEMA = "p13i/agent-harness/transcript/v1"
DEFAULT_TOKEN_BUDGET = 8192
MAX_TOKEN_BUDGET = 1_000_000
DEFAULT_TAIL_TURNS = 4
MAX_TAIL_TURNS = 64
CHARS_PER_TOKEN = 4

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
TOOL_CALL_ROLE = "tool_call"
TOOL_RESULT_ROLE = "tool_result"
FILE_CHANGE_ROLE = "file_change"
SYSTEM_ROLE = "system"


@dataclass(frozen=True)
class TranscriptEntry:
    """One provider-neutral message in the canonical transcript."""

    seq: int
    turn_id: str
    provider: str
    role: str
    name: str
    text: str
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "turn_id": self.turn_id,
            "provider": self.provider,
            "role": self.role,
            "name": self.name,
            "text": self.text,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class Transcript:
    """Versioned, digest-bound projection of one session's events."""

    session_id: str
    goal: str
    entries: tuple[TranscriptEntry, ...]
    digest: str
    schema: str = TRANSCRIPT_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "session_id": self.session_id,
            "goal": self.goal,
            "digest": self.digest,
            "entries": [entry.as_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class RenderPolicy:
    token_budget: int = DEFAULT_TOKEN_BUDGET
    tail_turns: int = DEFAULT_TAIL_TURNS


def project_transcript(
    store: StateStore,
    session_id: str,
    *,
    blobs: BlobStore | None = None,
) -> Transcript:
    """Project the durable event log into a canonical transcript."""
    require_uuid(session_id, "session_id")
    store.get_session(session_id)
    providers: dict[str, str] = {}
    for row in store.presentation_turn_rows(session_id):
        turn_id = str(row["turn_id"])
        if turn_id and turn_id not in providers:
            providers[turn_id] = str(row["provider"])
    goal_text = ""
    goal = store.goal_for_session(session_id)
    if goal is not None:
        goal_text = goal.objective
    entries: list[TranscriptEntry] = []
    for event in store.all_events(session_id):
        if event.event_type.endswith(".delta"):
            # Streaming fragments duplicate the completed message or
            # tool event that follows them in the log.
            continue
        text = event.text
        if event.blob_digest and blobs is not None:
            text = blobs.get_text(event.blob_digest)
        entries.append(_entry(event, providers.get(event.turn_id, ""), text))
    digest = _transcript_digest(session_id, goal_text, entries)
    return Transcript(
        session_id=session_id,
        goal=goal_text,
        entries=tuple(entries),
        digest=digest,
    )


def render(
    transcript: Transcript,
    policy: RenderPolicy | None = None,
) -> str:
    """Render a transcript within a deterministic token budget.

    The head (goal block and original instructions) and the last
    ``policy.tail_turns`` turns stay verbatim. When the budget is
    exceeded the middle elides to per-entry digest lines first, then
    the oldest tail turns join the elision, and only as a last resort
    is head text truncated, always keeping its digest reachable.
    """
    if policy is None:
        policy = RenderPolicy()
    if policy.token_budget < 1:
        raise ValueError("token budget must be positive")
    if policy.tail_turns < 0:
        raise ValueError("tail turns must not be negative")
    head, middle, tail = _partition(transcript.entries, policy.tail_turns)
    candidate = _document(transcript, head, middle, tail, elide_middle=False)
    if _tokens(candidate) <= policy.token_budget:
        return candidate
    candidate = _document(transcript, head, middle, tail, elide_middle=True)
    if _tokens(candidate) <= policy.token_budget:
        return candidate
    tail_turns = policy.tail_turns
    while tail_turns > 1:
        tail_turns -= 1
        head, middle, tail = _partition(transcript.entries, tail_turns)
        candidate = _document(transcript, head, middle, tail, elide_middle=True)
        if _tokens(candidate) <= policy.token_budget:
            return candidate
    return _truncate_head(transcript, head, middle, tail, policy.token_budget)


def validate_render_policy(policy: RenderPolicy) -> RenderPolicy:
    if policy.tail_turns < 0 or policy.tail_turns > MAX_TAIL_TURNS:
        raise ValueError(
            "tail turns must be between 0 and " + str(MAX_TAIL_TURNS)
        )
    if policy.token_budget < 1 or policy.token_budget > MAX_TOKEN_BUDGET:
        raise ValueError(
            "token budget must be between 1 and " + str(MAX_TOKEN_BUDGET)
        )
    return policy


def _entry(
    event: SessionEvent,
    provider: str,
    text: str,
) -> TranscriptEntry:
    role = _role(event)
    name = ""
    if role in {TOOL_CALL_ROLE, TOOL_RESULT_ROLE}:
        name = str(event.metadata.get("name", ""))
    digest = _entry_digest(event, provider, role, name, text)
    return TranscriptEntry(
        seq=event.sequence,
        turn_id=event.turn_id,
        provider=provider,
        role=role,
        name=name,
        text=text,
        digest=digest,
    )


def _role(event: SessionEvent) -> str:
    if event.role == "user":
        return USER_ROLE
    if event.role == "assistant":
        return ASSISTANT_ROLE
    event_type = event.event_type
    if event_type.startswith("file.change."):
        return FILE_CHANGE_ROLE
    if event_type == "tool.started":
        return TOOL_CALL_ROLE
    if event_type == "tool.completed":
        return TOOL_RESULT_ROLE
    if event_type.startswith("tool.") and event_type.endswith(".started"):
        return TOOL_CALL_ROLE
    if event_type.startswith("tool.") and event_type.endswith(".completed"):
        return TOOL_RESULT_ROLE
    return SYSTEM_ROLE


def _entry_digest(
    event: SessionEvent,
    provider: str,
    role: str,
    name: str,
    text: str,
) -> str:
    payload = {
        "seq": event.sequence,
        "turn_id": event.turn_id,
        "provider": provider,
        "role": role,
        "name": name,
        "text": text,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _transcript_digest(
    session_id: str,
    goal: str,
    entries: list[TranscriptEntry],
) -> str:
    payload = {
        "schema": TRANSCRIPT_SCHEMA,
        "session_id": session_id,
        "goal": goal,
        "entries": [entry.as_dict() for entry in entries],
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _partition(
    entries: tuple[TranscriptEntry, ...],
    tail_turns: int,
) -> tuple[list[TranscriptEntry], list[TranscriptEntry], list[TranscriptEntry]]:
    head_end = 0
    for index, entry in enumerate(entries):
        if entry.role == USER_ROLE:
            head_end = index + 1
            break
    ordered_ids: list[str] = []
    for entry in entries[head_end:]:
        if entry.turn_id not in ordered_ids:
            ordered_ids.append(entry.turn_id)
    tail_ids: set[str] = set()
    if tail_turns > 0:
        tail_ids = set(ordered_ids[len(ordered_ids) - tail_turns :])
    boundary = len(entries)
    for index in range(head_end, len(entries)):
        if entries[index].turn_id in tail_ids:
            boundary = index
            break
    head = list(entries[:head_end])
    middle = list(entries[head_end:boundary])
    tail = list(entries[boundary:])
    return head, middle, tail


def _document(
    transcript: Transcript,
    head: list[TranscriptEntry],
    middle: list[TranscriptEntry],
    tail: list[TranscriptEntry],
    *,
    elide_middle: bool,
) -> str:
    lines = [
        "# Session transcript",
        "",
        "- Session: `" + transcript.session_id + "`",
        "- Schema: `" + transcript.schema + "`",
        "- Digest: `" + transcript.digest + "`",
    ]
    if transcript.goal:
        lines.extend(["", "## Goal", "", transcript.goal])
    lines.extend(["", "## Messages"])
    blocks: list[str] = []
    for entry in head:
        blocks.append(_entry_block(entry))
    if middle:
        if elide_middle:
            elided = (
                "_"
                + str(len(middle))
                + " entries elided; digests retained._"
            )
            blocks.append(elided)
            for entry in middle:
                blocks.append(_digest_line(entry))
        else:
            for entry in middle:
                blocks.append(_entry_block(entry))
    for entry in tail:
        blocks.append(_entry_block(entry))
    if blocks:
        lines.extend(["", "\n\n".join(blocks)])
    return "\n".join(lines).rstrip() + "\n"


def _entry_block(entry: TranscriptEntry) -> str:
    heading = (
        "### seq "
        + str(entry.seq)
        + " · "
        + entry.role
        + " · "
        + _provider_label(entry)
    )
    if entry.turn_id:
        heading += " · turn " + entry.turn_id
    parts = [heading]
    if entry.name:
        parts.extend(["", "`" + entry.name + "`"])
    if entry.text:
        parts.extend(["", entry.text])
    return "\n".join(parts)


def _digest_line(entry: TranscriptEntry) -> str:
    return (
        "- seq "
        + str(entry.seq)
        + " · "
        + entry.role
        + " · "
        + _provider_label(entry)
        + " · `"
        + entry.digest
        + "`"
    )


def _provider_label(entry: TranscriptEntry) -> str:
    if entry.provider:
        return entry.provider
    return "harness"


def _truncate_head(
    transcript: Transcript,
    head: list[TranscriptEntry],
    middle: list[TranscriptEntry],
    tail: list[TranscriptEntry],
    token_budget: int,
) -> str:
    working = list(head)
    while True:
        candidate = _document(
            transcript,
            working,
            middle,
            tail,
            elide_middle=True,
        )
        excess = _tokens(candidate) - token_budget
        if excess <= 0:
            return candidate
        target = -1
        longest = 0
        for index, entry in enumerate(working):
            if len(entry.text) > longest:
                longest = len(entry.text)
                target = index
        if target < 0:
            return _hard_cut(candidate, token_budget)
        entry = working[target]
        marker = "\n[truncated; digest " + entry.digest + "]"
        if len(entry.text) <= len(marker) + CHARS_PER_TOKEN:
            return _hard_cut(candidate, token_budget)
        keep = (
            len(entry.text)
            - excess * CHARS_PER_TOKEN
            - len(marker)
            - CHARS_PER_TOKEN
        )
        if keep < 0:
            keep = 0
        working[target] = replace(entry, text=entry.text[:keep] + marker)


def _hard_cut(document: str, token_budget: int) -> str:
    limit = token_budget * CHARS_PER_TOKEN
    if len(document) <= limit:
        return document
    marker = "\n[truncated to token budget]\n"
    head_keep = (limit - len(marker)) // 2
    tail_keep = limit - len(marker) - head_keep
    return document[:head_keep] + marker + document[len(document) - tail_keep :]
