"""Portable, deterministic records for durable chat state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agent_harness.config import HarnessPaths
from agent_harness.storage import StateStore


CHAT_RECORD_SCHEMA = "p13i/agent-harness/chat-record/v1"
CHAT_GLOBAL_SCHEMA = "p13i/agent-harness/chat-global/v1"


def materialize_session(
    paths: HarnessPaths,
    store: StateStore,
    session_id: str,
) -> dict[str, Any]:
    record = store.portable_session(session_id)
    digests = _blob_digests(record)
    missing = [
        digest
        for digest in digests
        if not _blob_path(paths.blobs, digest).is_file()
    ]
    if missing:
        raise ValueError(
            "portable record references missing blobs: "
            + ", ".join(missing)
        )
    record["blob_digests"] = digests
    session_root = paths.sessions / session_id
    session_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_json(session_root / "record.gpt.json", record)
    event_rows = _table_rows(record, "events")
    _write_text(
        session_root / "events.gpt.jsonl",
        _jsonl(event_rows),
    )
    _write_text(
        session_root / "transcript.gpt.md",
        _transcript(record, event_rows),
    )
    materialize_global(paths, store)
    return {
        "session_id": session_id,
        "last_sequence": _last_sequence(event_rows),
        "record_digest": _json_digest(record),
        "blob_count": len(digests),
    }


def materialize_all(
    paths: HarnessPaths,
    store: StateStore,
) -> list[dict[str, Any]]:
    results = []
    for session in store.list_sessions(include_archived=True):
        results.append(
            materialize_session(paths, store, session.session_id)
        )
    materialize_global(paths, store)
    return results


def materialize_global(
    paths: HarnessPaths,
    store: StateStore,
) -> Path:
    target = paths.state_dir / "global.gpt.json"
    _write_json(target, store.portable_global())
    return target


def load_portable_records(
    paths: HarnessPaths,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    if paths.sessions.is_dir():
        for record_path in sorted(
            paths.sessions.glob("*/record.gpt.json")
        ):
            record = _read_json(record_path)
            if record.get("schema") != CHAT_RECORD_SCHEMA:
                raise ValueError(
                    "unsupported portable record: " + str(record_path)
                )
            records.append(record)
    global_record = _read_json(paths.state_dir / "global.gpt.json")
    if global_record.get("schema") != CHAT_GLOBAL_SCHEMA:
        raise ValueError("unsupported portable global record")
    return (records, global_record)


def _table_rows(
    record: dict[str, Any],
    table: str,
) -> list[dict[str, Any]]:
    tables = record.get("tables", {})
    if not isinstance(tables, dict):
        raise ValueError("portable record tables are invalid")
    rows = tables.get(table, [])
    if not isinstance(rows, list):
        raise ValueError("portable record table is invalid: " + table)
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "portable record row is invalid: " + table
            )
        result.append(row)
    return result


def _blob_digests(record: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for event in _table_rows(record, "events"):
        digest = str(event.get("blob_digest", ""))
        if digest:
            found.add(_require_blob_digest(digest))
    for checkpoint in _table_rows(record, "checkpoints"):
        for field in (
            "patch_digest",
            "untracked_digest",
            "context_digest",
        ):
            digest = str(checkpoint.get(field, ""))
            if digest:
                found.add(_require_blob_digest(digest))
    return sorted(found)


def _require_blob_digest(value: str) -> str:
    if len(value) != 64:
        raise ValueError("portable blob digest must be SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            "portable blob digest must be hexadecimal"
        ) from error
    return value


def _blob_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / digest[2:]


def _jsonl(rows: list[dict[str, Any]]) -> str:
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _transcript(
    record: dict[str, Any],
    events: list[dict[str, Any]],
) -> str:
    sessions = _table_rows(record, "sessions")
    if len(sessions) != 1:
        raise ValueError("portable record must contain one session")
    session = sessions[0]
    lines = [
        "# " + str(session.get("name", "Agent session")),
        "",
        "- Session: `" + str(record.get("session_id", "")) + "`",
        "- Lifecycle: `" + str(session.get("lifecycle", "")) + "`",
        "- Provider: `" + str(session.get("active_provider", "")) + "`",
        "",
        "## Transcript",
        "",
    ]
    for event in events:
        role = str(event.get("role", ""))
        event_type = str(event.get("event_type", "event"))
        label = event_type
        if role:
            label += " · " + role
        lines.extend(
            [
                "### "
                + str(event.get("sequence", ""))
                + " · "
                + label,
                "",
                str(event.get("text", "")),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _last_sequence(events: list[dict[str, Any]]) -> int:
    if not events:
        return 0
    return max(int(event.get("sequence", 0)) for event in events)


def _json_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("portable record must be a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
