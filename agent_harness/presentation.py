"""Provider-neutral read projections for human-facing agent work."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
import re
from typing import Any
from typing import Iterable

from agent_harness.blobs import BlobStore
from agent_harness.ids import require_uuid
from agent_harness.models import Checkpoint
from agent_harness.models import SessionEvent
from agent_harness.storage import StateStore


PRESENTATION_SCHEMA = "p13i/agent-harness/presentation/v1"
DEFAULT_TURN_LIMIT = 50
MAX_TURN_LIMIT = 200
DEFAULT_DIFF_LINES = 400
MAX_DIFF_LINES = 1000
MAX_ACTIVITY_TEXT = 32_768

_SAFE_METADATA_KEYS = frozenset(
    {
        "action",
        "approval_id",
        "checkpoint_id",
        "detail",
        "evidence_id",
        "is_error",
        "kind",
        "name",
        "outcome",
        "reason",
        "reconciliation_id",
        "status",
        "subject",
        "summary",
    }
)
_SENSITIVE_PATH_PARTS = (
    ".env",
    ".pem",
    ".key",
    "api-token",
    "credential",
    "id_rsa",
    "machine-keys",
    "secret",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?"
    r"(?:api[_-]?key|authorization|password|secret|token)"
    r"[\"']?)"
    r"(\s*[:=]\s*)"
    r"(?:"
    r"bearer\s+[A-Za-z0-9._~+/=-]+"
    r"|\"(?:\\.|[^\"])*\""
    r"|'(?:\\.|[^'])*'"
    r"|[^\s,;]+"
    r")"
)
_BEARER_VALUE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_SECRET_TOKEN = re.compile(
    r"\b(?:gh[opsu]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b"
)


def session_turns(
    store: StateStore,
    session_id: str,
    *,
    after_sequence: int = 0,
    limit: int = DEFAULT_TURN_LIMIT,
) -> dict[str, Any]:
    """Return bounded logical turns grouped across provider attempts."""

    require_uuid(session_id, "session_id")
    if after_sequence < 0:
        raise ValueError("after_sequence must not be negative")
    bounded_limit = max(1, min(limit, MAX_TURN_LIMIT))
    turns, events = _all_turn_summaries(store, session_id)
    selected: list[dict[str, Any]] = []
    for summary in turns:
        last_sequence = int(summary["last_sequence"])
        if last_sequence == 0:
            if after_sequence > 0:
                continue
        else:
            if last_sequence <= after_sequence:
                continue
        selected.append(summary)
    selected = selected[:bounded_limit]
    revision = store.last_sequence(session_id)
    next_after = after_sequence
    if selected:
        next_after = int(selected[-1]["last_sequence"])
    return {
        "schema": PRESENTATION_SCHEMA,
        "session_id": session_id,
        "revision": revision,
        "after_sequence": after_sequence,
        "next_after_sequence": next_after,
        "turns": selected,
        "session_activity": [
            _safe_event(event)
            for event in events
            if not event.turn_id and event.sequence > after_sequence
        ],
    }


def session_turn(
    store: StateStore,
    session_id: str,
    turn_id: str,
) -> dict[str, Any]:
    """Return one logical turn selected by any underlying attempt turn."""

    require_uuid(turn_id, "turn_id")
    require_uuid(session_id, "session_id")
    turns, unused_events = _all_turn_summaries(store, session_id)
    del unused_events
    for turn in turns:
        if turn_id == turn["turn_id"] or turn_id in turn["turn_ids"]:
            return {
                "schema": PRESENTATION_SCHEMA,
                "session_id": session_id,
                "revision": store.last_sequence(session_id),
                "turn": turn,
            }
    raise ValueError("turn does not belong to the session")


def checkpoint_diff(
    checkpoint: Checkpoint,
    blobs: BlobStore,
    *,
    start_line: int = 0,
    limit: int = DEFAULT_DIFF_LINES,
) -> dict[str, Any]:
    """Return a safe, line-paged unified diff for one checkpoint."""

    if start_line < 0:
        raise ValueError("start_line must not be negative")
    bounded_limit = max(1, min(limit, MAX_DIFF_LINES))
    content = blobs.get(checkpoint.patch_digest).decode(
        "utf-8",
        errors="replace",
    )
    sanitized, redactions, binary, files = sanitize_diff(content)
    lines = sanitized.splitlines()
    selected = lines[start_line : start_line + bounded_limit]
    next_start_line: int | None = None
    if start_line + len(selected) < len(lines):
        next_start_line = start_line + len(selected)
    return {
        "schema": PRESENTATION_SCHEMA,
        "checkpoint_id": checkpoint.checkpoint_id,
        "session_id": checkpoint.session_id,
        "language": "diff",
        "start_line": start_line,
        "next_start_line": next_start_line,
        "total_lines": len(lines),
        "truncated": next_start_line is not None,
        "redactions": redactions,
        "binary": binary,
        "changed_files": files,
        "content": "\n".join(selected),
    }


def sanitize_diff(
    content: str,
) -> tuple[str, int, bool, list[str]]:
    """Remove unsafe diff bodies and redact credential-shaped values."""

    output: list[str] = []
    files: list[str] = []
    redactions = 0
    binary = False
    omit_section = False
    binary_section = False
    for line in content.splitlines():
        if line.startswith("diff --git "):
            omit_section = _sensitive_diff_header(line)
            binary_section = False
            if omit_section:
                output.append("diff --git [sensitive path omitted]")
                output.append("[sensitive diff omitted]")
                redactions += 1
            else:
                output.append(line)
            continue
        path = ""
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:]
        if path:
            if path not in files and not _sensitive_path(path):
                files.append(path)
        if omit_section:
            continue
        if (
            line == "GIT binary patch"
            or line == "Binary files differ"
            or (
                line.startswith("Binary files ")
                and line.endswith(" differ")
            )
        ):
            binary = True
            binary_section = True
            output.append("[binary diff omitted]")
            continue
        if binary_section:
            continue
        redacted, count = _redact_line(line)
        redactions += count
        output.append(redacted)
    return "\n".join(output), redactions, binary, files


def _group_turn_rows(
    rows: Iterable[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        command_id = str(row.get("command_id", ""))
        if not command_id:
            command_id = "turn:" + str(row.get("turn_id", ""))
        grouped.setdefault(command_id, []).append(row)
    return grouped


def _all_turn_summaries(
    store: StateStore,
    session_id: str,
) -> tuple[list[dict[str, Any]], list[SessionEvent]]:
    rows = store.presentation_turn_rows(session_id)
    events = store.all_events(session_id)
    grouped = _group_turn_rows(rows)
    event_groups = _events_by_command(grouped, events)
    safety = _safety_by_command(store.session_safety(session_id))
    reconciliations = {
        item.command_id: item.as_dict()
        for item in store.all_reconciliations(session_id)
    }
    turns: list[dict[str, Any]] = []
    for command_id, group in grouped.items():
        turns.append(
            _turn_summary(
                command_id,
                group,
                event_groups.get(command_id, ()),
                safety.get(command_id, {}),
                reconciliations.get(command_id, {}),
            )
        )
    turns.sort(
        key=lambda item: (
            int(item["first_sequence"]),
            str(item["turn_id"]),
        )
    )
    return turns, events


def _events_by_command(
    grouped: OrderedDict[str, list[dict[str, Any]]],
    events: Iterable[SessionEvent],
) -> dict[str, tuple[SessionEvent, ...]]:
    commands_by_turn: dict[str, str] = {}
    for command_id, rows in grouped.items():
        for row in rows:
            commands_by_turn[str(row.get("turn_id", ""))] = command_id
    values: dict[str, list[SessionEvent]] = {}
    for event in events:
        command_id = commands_by_turn.get(event.turn_id, "")
        if not command_id:
            continue
        values.setdefault(command_id, []).append(event)
    return {
        command_id: tuple(items)
        for command_id, items in values.items()
    }


def _turn_summary(
    command_id: str,
    rows: list[dict[str, Any]],
    events: tuple[SessionEvent, ...],
    safety: dict[str, Any],
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    first = rows[0]
    sequences = [event.sequence for event in events]
    first_sequence = 0
    last_sequence = 0
    if sequences:
        first_sequence = min(sequences)
        last_sequence = max(sequences)
    turn_ids = [str(row.get("turn_id", "")) for row in rows]
    attempts = [_attempt_summary(row) for row in rows]
    checkpoint_id = ""
    evidence: list[dict[str, Any]] = []
    activity: list[dict[str, Any]] = []
    for event in events:
        safe = _safe_event(event)
        activity.append(safe)
        metadata = safe["metadata"]
        if event.event_type == "checkpoint.created":
            checkpoint_id = str(metadata.get("checkpoint_id", ""))
        if event.event_type == "goal.evidence":
            evidence.append(safe)
    status = str(first.get("command_status", ""))
    if not status:
        status = str(rows[-1].get("turn_status", ""))
    return {
        "turn_id": turn_ids[0],
        "turn_ids": turn_ids,
        "command_id": command_id,
        "turn_ref": dict(first.get("turn_ref", {})),
        "request": str(first.get("request_text", ""))[:MAX_ACTIVITY_TEXT],
        "status": status,
        "started_at": str(first.get("started_at", "")),
        "completed_at": str(rows[-1].get("completed_at", "")),
        "first_sequence": first_sequence,
        "last_sequence": last_sequence,
        "attempts": attempts,
        "result": _safe_result(first.get("command_result", {})),
        "safety": _safe_safety(safety),
        "checkpoint_id": checkpoint_id,
        "evidence": evidence,
        "reconciliation": reconciliation,
        "activity": activity,
    }


def _attempt_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": str(row.get("turn_id", "")),
        "attempt_id": str(row.get("attempt_id", "")),
        "provider": str(row.get("provider", "")),
        "model": str(row.get("model", "")),
        "effort": str(row.get("effort", "")),
        "status": str(row.get("attempt_status", "")),
        "started_at": str(row.get("started_at", "")),
        "ended_at": str(row.get("ended_at", "")),
    }


def _safe_event(event: SessionEvent) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in event.metadata.items()
        if key in _SAFE_METADATA_KEYS
        and isinstance(value, (str, int, float, bool))
    }
    return {
        "sequence": event.sequence,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "role": event.role,
        "text": event.text[:MAX_ACTIVITY_TEXT],
        "status": event.status,
        "metadata": metadata,
        "created_at": event.created_at,
    }


def _safe_result(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "checkpoint_id",
        "effort",
        "model",
        "provider",
        "status",
        "turn_id",
        "usage",
    }
    return {
        key: item
        for key, item in value.items()
        if key in allowed and isinstance(item, (str, int, float, bool, dict))
    }


def _safe_safety(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "command_id",
        "consumption",
        "guard_reason",
        "limits",
        "profile",
        "provider",
        "recovery_stage",
        "state",
    }
    return {key: value[key] for key in allowed if key in value}


def _safety_by_command(
    value: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    envelopes = value.get("envelopes", [])
    if not isinstance(envelopes, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in envelopes:
        if not isinstance(item, dict):
            continue
        command_id = str(item.get("command_id", ""))
        if command_id:
            result[command_id] = item
    return result


def _sensitive_diff_header(line: str) -> bool:
    return _sensitive_path(line)


def _sensitive_path(path: str) -> bool:
    normalized = path.casefold()
    return any(part in normalized for part in _SENSITIVE_PATH_PARTS)


def _redact_line(line: str) -> tuple[str, int]:
    redactions = 0

    def assignment(match: re.Match[str]) -> str:
        nonlocal redactions
        redactions += 1
        return match.group(1) + match.group(2) + "[REDACTED]"

    value = _SECRET_ASSIGNMENT.sub(assignment, line)

    def bearer(match: re.Match[str]) -> str:
        nonlocal redactions
        redactions += 1
        return match.group(1) + "[REDACTED]"

    value = _BEARER_VALUE.sub(bearer, value)

    def token(unused: re.Match[str]) -> str:
        nonlocal redactions
        redactions += 1
        return "[REDACTED]"

    value = _SECRET_TOKEN.sub(token, value)
    return value, redactions


@dataclass(frozen=True, slots=True)
class SessionViewCacheEntry:
    """Reader-local state retained while another session is visible."""

    session_id: str
    transcript_scroll_y: float = 0.0
    focus_id: str = "composer"
    expanded_block_ids: frozenset[str] = frozenset()
    composer: str = ""
    composer_cursor: str = "0:0"
    workspace_mode: str = "focus"
    selected_turn_id: str = ""
    detail_tab: str = "summary"
    revision: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionSwitchCoordinator:
    """Generation guard and bounded LRU for atomic session changes."""

    maximum_entries: int = 8
    generation: int = 0
    selected_session_id: str = ""
    _cache: OrderedDict[str, SessionViewCacheEntry] = field(
        default_factory=OrderedDict
    )

    def begin(self, session_id: str) -> int:
        self.generation += 1
        self.selected_session_id = session_id
        return self.generation

    def is_current(self, generation: int, session_id: str) -> bool:
        return (
            generation == self.generation
            and session_id == self.selected_session_id
        )

    def remember(self, entry: SessionViewCacheEntry) -> None:
        self._cache.pop(entry.session_id, None)
        self._cache[entry.session_id] = entry
        while len(self._cache) > self.maximum_entries:
            self._cache.popitem(last=False)

    def recall(self, session_id: str) -> SessionViewCacheEntry | None:
        entry = self._cache.pop(session_id, None)
        if entry is None:
            return None
        self._cache[session_id] = entry
        return entry
