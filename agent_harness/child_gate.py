"""Durable atomic admission gate for provider child-agent launches."""

from __future__ import annotations

import datetime
import json
import sqlite3
import sys
from pathlib import Path


def admit(
    database: Path,
    command_id: str,
    limit: int,
    admission_key: str,
) -> bool:
    """Consume one child-launch permit, or reject without launching."""
    if not command_id or limit < 0 or not admission_key:
        return False
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database, isolation_level=None, timeout=5.0)
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT permit_limit, consumed FROM child_launch_gates
            WHERE command_id = ?
            """,
            (command_id,),
        ).fetchone()
        if row is None:
            connection.execute("ROLLBACK")
            return False
        permit_limit = int(row[0])
        consumed = int(row[1])
        if permit_limit != limit or consumed < 0:
            connection.execute("ROLLBACK")
            return False
        prior = connection.execute(
            """
            SELECT admitted FROM child_launch_admissions
            WHERE command_id = ? AND admission_key = ?
            """,
            (command_id, admission_key),
        ).fetchone()
        if prior is not None:
            connection.execute("COMMIT")
            return int(prior[0]) == 1
        admitted = consumed < permit_limit
        now = _utc_now()
        connection.execute(
            """
            INSERT INTO child_launch_admissions(
                command_id, admission_key, admitted, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (command_id, admission_key, int(admitted), now),
        )
        if admitted:
            cursor = connection.execute(
                """
                UPDATE child_launch_gates
                SET consumed = consumed + 1, updated_at = ?
                WHERE command_id = ? AND permit_limit = ?
                    AND consumed < permit_limit
                """,
                (now, command_id, limit),
            )
            if cursor.rowcount != 1:
                connection.execute("ROLLBACK")
                return False
        connection.execute("COMMIT")
        return admitted
    except (OSError, ValueError, sqlite3.Error):
        _rollback(connection)
        return False
    finally:
        if connection is not None:
            connection.close()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 4:
        _deny()
        return 0
    try:
        database = Path(argv[0])
        command_id = argv[1]
        limit = int(argv[2])
        provider = argv[3]
        payload = json.load(sys.stdin)
    except (OSError, ValueError, json.JSONDecodeError):
        _deny()
        return 0
    if not isinstance(payload, dict):
        _deny()
        return 0
    tool_name = str(payload.get("tool_name", ""))
    if tool_name not in {"Agent", "spawn_agent"}:
        return 0
    admission_key = _admission_key(provider, tool_name, payload)
    if admit(database, command_id, limit, admission_key):
        return 0
    _deny()
    return 0


def _deny() -> None:
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "The harness child-agent launch limit is exhausted."
            ),
        }
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


def _admission_key(
    provider: str,
    tool_name: str,
    payload: dict[str, object],
    explicit_id: str | None = None,
) -> str:
    identity = str(explicit_id or "").strip()
    if not identity:
        for name in ("tool_use_id", "call_id", "id"):
            value = payload.get(name)
            if isinstance(value, str) and value.strip():
                identity = value.strip()
                break
    if not provider or not tool_name or not identity:
        return ""
    return provider + ":" + tool_name + ":" + identity


def _rollback(connection: sqlite3.Connection | None) -> None:
    if connection is None or not connection.in_transaction:
        return
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error:
        return


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
