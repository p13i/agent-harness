"""Transactional SQLite state for the harness."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any
from typing import Iterator

from agent_harness.errors import ConflictError
from agent_harness.errors import NotFoundError
from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Checkpoint
from agent_harness.models import CommandReceipt
from agent_harness.models import CommandStatus
from agent_harness.models import Evidence
from agent_harness.models import Goal
from agent_harness.models import ProviderAttempt
from agent_harness.models import Session
from agent_harness.models import SessionEvent


SCHEMA_VERSION = 2

PORTABLE_SESSION_TABLES = (
    "sessions",
    "provider_attempts",
    "turns",
    "events",
    "commands",
    "goals",
    "milestones",
    "evidence",
    "approvals",
    "checkpoints",
    "session_safety",
    "command_envelopes",
    "guard_incidents",
    "context_deliveries",
    "routing_decisions",
    "transfers",
    "registry_entries",
    "ui_state",
)
PORTABLE_GLOBAL_TABLES = (
    "usage_samples",
    "mutation_receipts",
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    workspace TEXT NOT NULL,
    worktree TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    attention TEXT NOT NULL,
    permission_mode TEXT NOT NULL,
    active_provider TEXT NOT NULL,
    model TEXT NOT NULL,
    effort TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    owner_host TEXT NOT NULL,
    owner_epoch INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS provider_attempts (
    attempt_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    provider TEXT NOT NULL,
    native_session_id TEXT NOT NULL,
    model TEXT NOT NULL,
    effort TEXT NOT NULL,
    auth_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS provider_attempts_session
ON provider_attempts(session_id, started_at);
CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    attempt_id TEXT NOT NULL,
    status TEXT NOT NULL,
    replay_of TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    blob_digest TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(session_id, sequence)
);
CREATE TABLE IF NOT EXISTS commands (
    idempotency_key TEXT PRIMARY KEY,
    command_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    command_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS commands_dispatch
ON commands(session_id, status, created_at);
CREATE TABLE IF NOT EXISTS goals (
    goal_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    kind TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,
    constraints_json TEXT NOT NULL,
    predicates_json TEXT NOT NULL,
    budgets_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS milestones (
    milestone_id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goals(goal_id),
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    dependencies_json TEXT NOT NULL,
    predicates_json TEXT NOT NULL,
    position INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goals(goal_id),
    evidence_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    outcome TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    turn_id TEXT NOT NULL,
    provider_request_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    prompt TEXT NOT NULL,
    choices_json TEXT NOT NULL,
    status TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    sequence INTEGER NOT NULL,
    provider TEXT NOT NULL,
    native_session_id TEXT NOT NULL,
    base_commit TEXT NOT NULL,
    patch_digest TEXT NOT NULL,
    untracked_digest TEXT NOT NULL,
    context_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_samples (
    sample_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    binding_percent REAL,
    credits_engaged INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_safety (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    profile TEXT NOT NULL,
    xhigh_authorizations INTEGER NOT NULL,
    extensions_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS command_envelopes (
    command_id TEXT PRIMARY KEY REFERENCES commands(command_id),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    provider TEXT NOT NULL,
    profile TEXT NOT NULL,
    state TEXT NOT NULL,
    limits_json TEXT NOT NULL,
    consumption_json TEXT NOT NULL,
    guard_reason TEXT NOT NULL,
    recovery_stage INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS command_envelopes_active
ON command_envelopes(provider, state, profile);
CREATE TABLE IF NOT EXISTS guard_incidents (
    incident_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    command_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    action TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS context_deliveries (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    provider TEXT NOT NULL,
    context_digest TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    delivered_at TEXT NOT NULL,
    PRIMARY KEY(session_id, provider, context_digest)
);
CREATE TABLE IF NOT EXISTS process_leases (
    lease_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    profile TEXT NOT NULL,
    pid INTEGER NOT NULL,
    pid_start TEXT NOT NULL,
    state TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS process_leases_active
ON process_leases(state, expires_at);
CREATE TABLE IF NOT EXISTS mutation_receipts (
    idempotency_key TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    response_json TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routing_decisions (
    decision_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    turn_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    effort TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workers (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    pid INTEGER NOT NULL,
    incarnation TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transfers (
    transfer_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    source_host TEXT NOT NULL,
    destination_host TEXT NOT NULL,
    owner_epoch INTEGER NOT NULL,
    status TEXT NOT NULL,
    bundle_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS registry_entries (
    session_id TEXT PRIMARY KEY,
    owner_host TEXT NOT NULL,
    owner_url TEXT NOT NULL,
    owner_epoch INTEGER NOT NULL,
    lifecycle TEXT NOT NULL,
    last_sequence INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ui_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class StateStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        self.path = path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._initialize()

    def _configure(self) -> None:
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connection
            connection.executescript(SCHEMA)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT version FROM schema_meta"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_meta(version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
            elif int(row["version"]) == 1:
                connection.execute(
                    "UPDATE schema_meta SET version = ?",
                    (SCHEMA_VERSION,),
                )
            elif int(row["version"]) != SCHEMA_VERSION:
                raise RuntimeError("unsupported database schema version")
        os.chmod(self.path, 0o600)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def backup(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = sqlite3.connect(destination)
        try:
            with self._lock:
                self._connection.backup(target)
        finally:
            target.close()
        os.chmod(destination, 0o600)

    def clear_runtime_state(self) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM workers")
            connection.execute("DELETE FROM process_leases")

    def rewrite_worktree_prefix(
        self,
        source_prefix: str,
        destination_prefix: str,
    ) -> int:
        changed = 0
        boundary = source_prefix + os.sep
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT session_id, worktree FROM sessions"
            ).fetchall()
            for row in rows:
                current = str(row["worktree"])
                if not current.startswith(boundary):
                    continue
                rewritten = destination_prefix + current[len(source_prefix) :]
                connection.execute(
                    """
                    UPDATE sessions SET worktree = ?
                    WHERE session_id = ?
                    """,
                    (rewritten, row["session_id"]),
                )
                changed += 1
        return changed

    def create_session(self, session: Session) -> Session:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sessions(
                    session_id, name, workspace, worktree, lifecycle,
                    attention, permission_mode, active_provider, model,
                    effort, goal_id, owner_host, owner_epoch, created_at,
                    updated_at, archived
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.name,
                    session.workspace,
                    session.worktree,
                    session.lifecycle,
                    session.attention,
                    session.permission_mode,
                    session.active_provider,
                    session.model,
                    session.effort,
                    session.goal_id,
                    session.owner_host,
                    session.owner_epoch,
                    session.created_at,
                    session.updated_at,
                    int(session.archived),
                ),
            )
        return session

    def get_session(self, session_id: str) -> Session:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("session")
        return _session(row)

    def list_sessions(self, include_archived: bool = False) -> list[Session]:
        query = "SELECT * FROM sessions"
        parameters: tuple[Any, ...] = ()
        if not include_archived:
            query += " WHERE archived = 0"
        query += " ORDER BY updated_at DESC, session_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [_session(row) for row in rows]

    def update_session(self, session_id: str, **changes: Any) -> Session:
        allowed = {
            "name",
            "workspace",
            "worktree",
            "lifecycle",
            "attention",
            "permission_mode",
            "active_provider",
            "model",
            "effort",
            "goal_id",
            "owner_host",
            "owner_epoch",
            "archived",
        }
        unknown = set(changes) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError("unsupported session fields: " + names)
        if not changes:
            return self.get_session(session_id)
        changes["updated_at"] = utc_now()
        assignments = ", ".join(name + " = ?" for name in changes)
        values: list[Any] = []
        for value in changes.values():
            if isinstance(value, bool):
                values.append(int(value))
            else:
                values.append(value)
        values.append(session_id)
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET "
                + assignments
                + " WHERE session_id = ?",
                tuple(values),
            )
            if cursor.rowcount != 1:
                raise NotFoundError("session")
        return self.get_session(session_id)

    def append_event(
        self,
        session_id: str,
        event_type: str,
        *,
        role: str = "",
        text: str = "",
        status: str = "",
        metadata: dict[str, Any] | None = None,
        blob_digest: str = "",
        turn_id: str = "",
        event_id: str | None = None,
    ) -> SessionEvent:
        if metadata is None:
            metadata = {}
        if event_id is None:
            event_id = new_uuid()
        created_at = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS sequence
                FROM events WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            sequence = int(row["sequence"]) + 1
            connection.execute(
                """
                INSERT INTO events(
                    session_id, sequence, event_id, event_type, role,
                    text, status, metadata_json, blob_digest, turn_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    sequence,
                    event_id,
                    event_type,
                    role,
                    text,
                    status,
                    _dump(metadata),
                    blob_digest,
                    turn_id,
                    created_at,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (created_at, session_id),
            )
        return SessionEvent(
            session_id=session_id,
            sequence=sequence,
            event_id=event_id,
            event_type=event_type,
            role=role,
            text=text,
            status=status,
            metadata=metadata,
            blob_digest=blob_digest,
            turn_id=turn_id,
            created_at=created_at,
        )

    def events(
        self,
        session_id: str,
        *,
        after: int = 0,
        limit: int = 1000,
    ) -> list[SessionEvent]:
        bounded = max(1, min(limit, 5000))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM events
                WHERE session_id = ? AND sequence > ?
                ORDER BY sequence LIMIT ?
                """,
                (session_id, after, bounded),
            ).fetchall()
        return [_event(row) for row in rows]

    def all_events(self, session_id: str) -> list[SessionEvent]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM events
                WHERE session_id = ?
                ORDER BY sequence
                """,
                (session_id,),
            ).fetchall()
        return [_event(row) for row in rows]

    def last_sequence(self, session_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS sequence
                FROM events WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return int(row["sequence"])

    def enqueue_command(
        self,
        session_id: str,
        command_type: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> CommandReceipt:
        now = utc_now()
        command_id = new_uuid()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM commands WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return _command(existing)
            connection.execute(
                """
                INSERT INTO commands(
                    idempotency_key, command_id, session_id,
                    command_type, payload_json, status, result_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    command_id,
                    session_id,
                    command_type,
                    _dump(payload),
                    CommandStatus.QUEUED,
                    "{}",
                    now,
                    now,
                ),
            )
        return CommandReceipt(
            command_id=command_id,
            idempotency_key=idempotency_key,
            session_id=session_id,
            command_type=command_type,
            status=CommandStatus.QUEUED,
            result={},
            created_at=now,
            updated_at=now,
        )

    def claim_command(
        self,
        session_id: str,
        command_types: frozenset[str] = frozenset(),
    ) -> CommandReceipt | None:
        type_clause = ""
        parameters: list[Any] = [session_id, CommandStatus.QUEUED]
        if command_types:
            placeholders = ", ".join("?" for _ in command_types)
            type_clause = " AND command_type IN (" + placeholders + ")"
            parameters.extend(sorted(command_types))
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM commands
                WHERE session_id = ? AND status = ?
                """
                + type_clause
                + """
                ORDER BY created_at, command_id LIMIT 1
                """,
                tuple(parameters),
            ).fetchone()
            if row is None:
                return None
            now = utc_now()
            cursor = connection.execute(
                """
                UPDATE commands SET status = ?, updated_at = ?
                WHERE command_id = ? AND status = ?
                """,
                (
                    CommandStatus.DISPATCHING,
                    now,
                    row["command_id"],
                    CommandStatus.QUEUED,
                ),
            )
            if cursor.rowcount != 1:
                return None
            payload = dict(row)
            payload["status"] = CommandStatus.DISPATCHING
            payload["updated_at"] = now
            return _command(payload)

    def command_payload(self, command_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("command")
        return _load_object(row["payload_json"])

    def get_command(self, command_id: str) -> CommandReceipt:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("command")
        return _command(row)

    def resolve_command(
        self,
        command_id: str,
        status: str,
        result: dict[str, Any],
    ) -> CommandReceipt:
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE commands SET status = ?, result_json = ?,
                updated_at = ? WHERE command_id = ?
                """,
                (status, _dump(result), now, command_id),
            )
            if cursor.rowcount != 1:
                raise NotFoundError("command")
            row = connection.execute(
                "SELECT * FROM commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        return _command(row)

    def recover_dispatching(self, session_id: str) -> int:
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE commands SET status = ?, result_json = ?,
                updated_at = ?
                WHERE session_id = ? AND status = ?
                """,
                (
                    CommandStatus.FAILED,
                    _dump(
                        {
                            "code": "E_NEEDS_RECONCILIATION",
                            "message": (
                                "worker stopped after dispatch began; "
                                "the effect is ambiguous"
                            ),
                        }
                    ),
                    now,
                    session_id,
                    CommandStatus.DISPATCHING,
                ),
            )
        return cursor.rowcount

    def start_turn(
        self,
        session_id: str,
        attempt_id: str,
        *,
        replay_of: str = "",
    ) -> str:
        turn_id = new_uuid()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    turn_id,
                    session_id,
                    attempt_id,
                    "running",
                    replay_of,
                    utc_now(),
                    "",
                ),
            )
        return turn_id

    def finish_turn(self, turn_id: str, status: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE turns SET status = ?, completed_at = ?
                WHERE turn_id = ?
                """,
                (status, utc_now(), turn_id),
            )

    def turn_count(self, session_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["count"])

    def completed_command_results(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT result_json FROM commands
                WHERE session_id = ? AND status = ?
                ORDER BY created_at
                """,
                (session_id, CommandStatus.COMPLETE),
            ).fetchall()
        return [
            _load_object(str(row["result_json"])) for row in rows
        ]

    def create_attempt(self, attempt: ProviderAttempt) -> ProviderAttempt:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO provider_attempts VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    attempt.attempt_id,
                    attempt.session_id,
                    attempt.provider,
                    attempt.native_session_id,
                    attempt.model,
                    attempt.effort,
                    attempt.auth_mode,
                    attempt.status,
                    attempt.started_at,
                    attempt.ended_at,
                ),
            )
        return attempt

    def update_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        native_session_id: str | None = None,
    ) -> None:
        fields = ["status = ?"]
        values: list[Any] = [status]
        if native_session_id is not None:
            fields.append("native_session_id = ?")
            values.append(native_session_id)
        if status in {"complete", "failed", "interrupted", "exhausted"}:
            fields.append("ended_at = ?")
            values.append(utc_now())
        values.append(attempt_id)
        with self.transaction() as connection:
            connection.execute(
                "UPDATE provider_attempts SET "
                + ", ".join(fields)
                + " WHERE attempt_id = ?",
                tuple(values),
            )

    def attempts(self, session_id: str) -> list[ProviderAttempt]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM provider_attempts WHERE session_id = ?
                ORDER BY started_at, attempt_id
                """,
                (session_id,),
            ).fetchall()
        return [_attempt(row) for row in rows]

    def create_goal(self, goal: Goal) -> Goal:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO goals VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    goal.goal_id,
                    goal.session_id,
                    goal.kind,
                    goal.objective,
                    goal.status,
                    _dump(list(goal.constraints)),
                    _dump(list(goal.predicates)),
                    _dump(goal.budgets),
                    goal.created_at,
                    goal.updated_at,
                ),
            )
            connection.execute(
                """
                UPDATE sessions SET goal_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (goal.goal_id, goal.updated_at, goal.session_id),
            )
        return goal

    def get_goal(self, goal_id: str) -> Goal:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("goal")
        return _goal(row)

    def goal_for_session(self, session_id: str) -> Goal | None:
        session = self.get_session(session_id)
        if not session.goal_id:
            return None
        return self.get_goal(session.goal_id)

    def update_goal_status(self, goal_id: str, status: str) -> Goal:
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE goals SET status = ?, updated_at = ?
                WHERE goal_id = ?
                """,
                (status, now, goal_id),
            )
            if cursor.rowcount != 1:
                raise NotFoundError("goal")
        return self.get_goal(goal_id)

    def add_evidence(self, evidence: Evidence) -> Evidence:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    evidence.evidence_id,
                    evidence.goal_id,
                    evidence.evidence_type,
                    evidence.subject,
                    evidence.outcome,
                    _dump(evidence.value),
                    evidence.created_at,
                ),
            )
        return evidence

    def evidence(self, goal_id: str) -> list[Evidence]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM evidence WHERE goal_id = ?
                ORDER BY created_at, evidence_id
                """,
                (goal_id,),
            ).fetchall()
        return [_evidence(row) for row in rows]

    def add_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    checkpoint.checkpoint_id,
                    checkpoint.session_id,
                    checkpoint.sequence,
                    checkpoint.provider,
                    checkpoint.native_session_id,
                    checkpoint.base_commit,
                    checkpoint.patch_digest,
                    checkpoint.untracked_digest,
                    checkpoint.context_digest,
                    checkpoint.created_at,
                ),
            )
        return checkpoint

    def checkpoints(self, session_id: str) -> list[Checkpoint]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM checkpoints WHERE session_id = ?
                ORDER BY sequence, created_at
                """,
                (session_id,),
            ).fetchall()
        return [_checkpoint(row) for row in rows]

    def create_approval(
        self,
        session_id: str,
        turn_id: str,
        provider_request_id: str,
        kind: str,
        prompt: str,
        choices: list[dict[str, Any]],
    ) -> str:
        approval_id = new_uuid()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    approval_id,
                    session_id,
                    turn_id,
                    provider_request_id,
                    kind,
                    prompt,
                    _dump(choices),
                    "pending",
                    "{}",
                    utc_now(),
                    "",
                ),
            )
        return approval_id

    def pending_approvals(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM approvals
                WHERE session_id = ? AND status = 'pending'
                ORDER BY created_at
                """,
                (session_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "approval_id": str(row["approval_id"]),
                    "session_id": str(row["session_id"]),
                    "turn_id": str(row["turn_id"]),
                    "provider_request_id": str(row["provider_request_id"]),
                    "kind": str(row["kind"]),
                    "prompt": str(row["prompt"]),
                    "choices": json.loads(row["choices_json"]),
                    "status": str(row["status"]),
                    "created_at": str(row["created_at"]),
                }
            )
        return result

    def resolve_approval(
        self,
        approval_id: str,
        decision: dict[str, Any],
    ) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE approvals SET status = 'resolved',
                decision_json = ?, resolved_at = ?
                WHERE approval_id = ? AND status = 'pending'
                """,
                (_dump(decision), utc_now(), approval_id),
            )
        return cursor.rowcount == 1

    def approval_decision(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT status, decision_json FROM approvals
                WHERE approval_id = ?
                """,
                (approval_id,),
            ).fetchone()
        if row is None or row["status"] != "resolved":
            return None
        return _load_object(row["decision_json"])

    def set_session_safety(
        self,
        session_id: str,
        profile: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO session_safety(
                    session_id, profile, xhigh_authorizations,
                    extensions_json, created_at, updated_at
                ) VALUES (?, ?, 0, '{}', ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    profile = excluded.profile,
                    updated_at = excluded.updated_at
                """,
                (session_id, profile, now, now),
            )
        return self.session_safety(session_id)

    def session_safety(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM session_safety WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return {
                "session_id": session_id,
                "profile": "",
                "xhigh_authorizations": 0,
                "extensions": {},
                "created_at": "",
                "updated_at": "",
            }
        return {
            "session_id": str(row["session_id"]),
            "profile": str(row["profile"]),
            "xhigh_authorizations": int(
                row["xhigh_authorizations"]
            ),
            "extensions": _load_object(row["extensions_json"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def extend_session_safety(
        self,
        session_id: str,
        extension: dict[str, Any],
    ) -> dict[str, Any]:
        current = self.session_safety(session_id)
        if not current["profile"]:
            raise ConflictError("session execution profile is not claimed")
        extensions = dict(current["extensions"])
        extensions.update(extension)
        xhigh = int(current["xhigh_authorizations"])
        if extension.get("allow_xhigh_once") is True:
            xhigh += 1
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE session_safety
                SET xhigh_authorizations = ?, extensions_json = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    xhigh,
                    _dump(extensions),
                    utc_now(),
                    session_id,
                ),
            )
        return self.session_safety(session_id)

    def consume_xhigh_authorization(self, session_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE session_safety
                SET xhigh_authorizations = xhigh_authorizations - 1,
                    updated_at = ?
                WHERE session_id = ? AND xhigh_authorizations > 0
                """,
                (utc_now(), session_id),
            )
        return cursor.rowcount == 1

    def consume_session_extensions(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT extensions_json FROM session_safety
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                raise ConflictError(
                    "session execution profile is not claimed"
                )
            extension = _load_object(row["extensions_json"])
            connection.execute(
                """
                UPDATE session_safety
                SET extensions_json = '{}', updated_at = ?
                WHERE session_id = ?
                """,
                (utc_now(), session_id),
            )
        return extension

    def create_command_envelope(
        self,
        command_id: str,
        session_id: str,
        profile: str,
        limits: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now()
        consumption = {
            "context_tokens": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "tool_calls": 0,
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "exact_tokens": False,
        }
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO command_envelopes(
                    command_id, session_id, provider, profile, state,
                    limits_json, consumption_json, guard_reason,
                    recovery_stage, created_at, updated_at
                ) VALUES (?, ?, '', ?, 'reserved', ?, ?, '', 0, ?, ?)
                """,
                (
                    command_id,
                    session_id,
                    profile,
                    _dump(limits),
                    _dump(consumption),
                    now,
                    now,
                ),
            )
        return self.command_envelope(command_id)

    def command_envelope(self, command_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM command_envelopes WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("command envelope")
        return {
            "command_id": str(row["command_id"]),
            "session_id": str(row["session_id"]),
            "provider": str(row["provider"]),
            "profile": str(row["profile"]),
            "state": str(row["state"]),
            "limits": _load_object(row["limits_json"]),
            "consumption": _load_object(row["consumption_json"]),
            "guard_reason": str(row["guard_reason"]),
            "recovery_stage": int(row["recovery_stage"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def update_command_envelope(
        self,
        command_id: str,
        *,
        provider: str | None = None,
        state: str | None = None,
        consumption: dict[str, Any] | None = None,
        guard_reason: str | None = None,
        recovery_stage: int | None = None,
    ) -> dict[str, Any]:
        fields = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        consumption_json: str | None = None
        if consumption is not None:
            consumption_json = _dump(consumption)
        for name, value in (
            ("provider", provider),
            ("state", state),
            ("consumption_json", consumption_json),
            ("guard_reason", guard_reason),
            ("recovery_stage", recovery_stage),
        ):
            if value is None:
                continue
            fields.append(name + " = ?")
            values.append(value)
        values.append(command_id)
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE command_envelopes SET "
                + ", ".join(fields)
                + " WHERE command_id = ?",
                tuple(values),
            )
            if cursor.rowcount != 1:
                raise NotFoundError("command envelope")
        return self.command_envelope(command_id)

    def active_unattended_provider_count(self, provider: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM command_envelopes
                WHERE provider = ? AND profile = 'unattended'
                AND state IN ('reserved', 'running', 'recovering')
                """,
                (provider,),
            ).fetchone()
        return int(row["count"])

    def session_envelopes(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT command_id FROM command_envelopes
                WHERE session_id = ? ORDER BY created_at, command_id
                """,
                (session_id,),
            ).fetchall()
        return [
            self.command_envelope(str(row["command_id"])) for row in rows
        ]

    def add_guard_incident(
        self,
        session_id: str,
        command_id: str,
        attempt_id: str,
        reason: str,
        action: str,
        snapshot: dict[str, Any],
    ) -> str:
        incident_id = new_uuid()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO guard_incidents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    incident_id,
                    session_id,
                    command_id,
                    attempt_id,
                    reason,
                    action,
                    _dump(snapshot),
                    utc_now(),
                ),
            )
        return incident_id

    def guard_incidents(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM guard_incidents
                WHERE session_id = ? ORDER BY created_at, incident_id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "incident_id": str(row["incident_id"]),
                "command_id": str(row["command_id"]),
                "attempt_id": str(row["attempt_id"]),
                "reason": str(row["reason"]),
                "action": str(row["action"]),
                "snapshot": _load_object(row["snapshot_json"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def record_context_delivery(
        self,
        session_id: str,
        provider: str,
        context_digest: str,
        checkpoint_id: str,
    ) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO context_deliveries
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    provider,
                    context_digest,
                    checkpoint_id,
                    utc_now(),
                ),
            )
        return cursor.rowcount == 1

    def create_process_lease(
        self,
        session_id: str,
        provider: str,
        profile: str,
        expires_at: str,
    ) -> dict[str, Any]:
        lease_id = new_uuid()
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO process_leases
                VALUES (?, ?, ?, ?, 0, '', 'reserved', ?, ?, ?)
                """,
                (
                    lease_id,
                    session_id,
                    provider,
                    profile,
                    expires_at,
                    now,
                    now,
                ),
            )
        return self.process_lease(lease_id)

    def process_lease(self, lease_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM process_leases WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("process lease")
        return {
            "lease_id": str(row["lease_id"]),
            "session_id": str(row["session_id"]),
            "provider": str(row["provider"]),
            "profile": str(row["profile"]),
            "pid": int(row["pid"]),
            "pid_start": str(row["pid_start"]),
            "state": str(row["state"]),
            "expires_at": str(row["expires_at"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def update_process_lease(
        self,
        lease_id: str,
        *,
        pid: int | None = None,
        pid_start: str | None = None,
        state: str | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        fields = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        for name, value in (
            ("pid", pid),
            ("pid_start", pid_start),
            ("state", state),
            ("expires_at", expires_at),
        ):
            if value is None:
                continue
            fields.append(name + " = ?")
            values.append(value)
        values.append(lease_id)
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE process_leases SET "
                + ", ".join(fields)
                + " WHERE lease_id = ?",
                tuple(values),
            )
            if cursor.rowcount != 1:
                raise NotFoundError("process lease")
        return self.process_lease(lease_id)

    def active_process_leases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT lease_id FROM process_leases
                WHERE state IN ('reserved', 'active')
                ORDER BY created_at, lease_id
                """
            ).fetchall()
        return [self.process_lease(str(row["lease_id"])) for row in rows]

    def mutation_receipt(
        self,
        idempotency_key: str,
        operation: str,
        request_digest: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM mutation_receipts
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        if (
            str(row["operation"]) != operation
            or str(row["request_digest"]) != request_digest
        ):
            raise ConflictError(
                "idempotency key was already used for "
                "another mutation"
            )
        return {
            "response": _load_object(row["response_json"]),
            "status_code": int(row["status_code"]),
        }

    def record_mutation_receipt(
        self,
        idempotency_key: str,
        operation: str,
        request_digest: str,
        response: dict[str, Any],
        status_code: int,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO mutation_receipts
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    operation,
                    request_digest,
                    _dump(response),
                    status_code,
                    utc_now(),
                ),
            )
        receipt = self.mutation_receipt(
            idempotency_key,
            operation,
            request_digest,
        )
        if receipt is None:
            raise RuntimeError("mutation receipt was not recorded")
        return receipt

    def record_usage(
        self,
        provider: str,
        binding_percent: float | None,
        credits_engaged: bool,
        payload: dict[str, Any],
    ) -> str:
        sample_id = new_uuid()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO usage_samples VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sample_id,
                    provider,
                    utc_now(),
                    binding_percent,
                    int(credits_engaged),
                    _dump(payload),
                ),
            )
        return sample_id

    def latest_usage(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT current.*
                FROM usage_samples AS current
                JOIN (
                    SELECT provider, MAX(observed_at) AS observed_at
                    FROM usage_samples GROUP BY provider
                ) AS latest
                ON current.provider = latest.provider
                AND current.observed_at = latest.observed_at
                """
            ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            result[str(row["provider"])] = {
                "provider": str(row["provider"]),
                "observed_at": str(row["observed_at"]),
                "binding_percent": row["binding_percent"],
                "credits_engaged": bool(row["credits_engaged"]),
                "payload": _load_object(row["payload_json"]),
            }
        return result

    def active_provider_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT active_provider, COUNT(*) AS count
                FROM sessions
                WHERE lifecycle IN ('starting', 'running')
                GROUP BY active_provider
                """
            ).fetchall()
        return {
            str(row["active_provider"]): int(row["count"])
            for row in rows
            if row["active_provider"]
        }

    def upsert_registry_entry(
        self,
        session_id: str,
        owner_host: str,
        owner_url: str,
        owner_epoch: int,
        lifecycle: str,
        last_sequence: int,
    ) -> None:
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT owner_epoch FROM registry_entries
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if existing is not None:
                if int(existing["owner_epoch"]) > owner_epoch:
                    raise ConflictError("registry owner epoch is stale")
            connection.execute(
                """
                INSERT INTO registry_entries VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    owner_host = excluded.owner_host,
                    owner_url = excluded.owner_url,
                    owner_epoch = excluded.owner_epoch,
                    lifecycle = excluded.lifecycle,
                    last_sequence = excluded.last_sequence,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    owner_host,
                    owner_url,
                    owner_epoch,
                    lifecycle,
                    last_sequence,
                    utc_now(),
                ),
            )

    def registry_entry(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM registry_entries WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("registry entry")
        return dict(row)

    def registry_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM registry_entries
                ORDER BY updated_at DESC, session_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def record_routing(
        self,
        session_id: str,
        turn_id: str,
        provider: str,
        model: str,
        effort: str,
        payload: dict[str, Any],
    ) -> str:
        decision_id = new_uuid()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO routing_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_id,
                    session_id,
                    turn_id,
                    provider,
                    model,
                    effort,
                    _dump(payload),
                    utc_now(),
                ),
            )
        return decision_id

    def register_worker(
        self,
        session_id: str,
        pid: int,
        incarnation: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO workers VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    pid = excluded.pid,
                    incarnation = excluded.incarnation,
                    heartbeat_at = excluded.heartbeat_at
                """,
                (session_id, pid, incarnation, utc_now()),
            )

    def heartbeat_worker(self, session_id: str, incarnation: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE workers SET heartbeat_at = ?
                WHERE session_id = ? AND incarnation = ?
                """,
                (utc_now(), session_id, incarnation),
            )
        return cursor.rowcount == 1

    def remove_worker(self, session_id: str, incarnation: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                DELETE FROM workers
                WHERE session_id = ? AND incarnation = ?
                """,
                (session_id, incarnation),
            )

    def set_ui_state(self, key: str, value: dict[str, Any]) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO ui_state VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, _dump(value), utc_now()),
            )

    def get_ui_state(self, key: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT value_json FROM ui_state WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return {}
        return _load_object(row["value_json"])

    def portable_session(self, session_id: str) -> dict[str, Any]:
        self.get_session(session_id)
        with self._lock:
            goals = self._portable_rows(
                "goals",
                "session_id = ?",
                (session_id,),
            )
            goal_ids = tuple(str(item["goal_id"]) for item in goals)
            tables = {
                "sessions": self._portable_rows(
                    "sessions",
                    "session_id = ?",
                    (session_id,),
                ),
                "provider_attempts": self._portable_rows(
                    "provider_attempts",
                    "session_id = ?",
                    (session_id,),
                ),
                "turns": self._portable_rows(
                    "turns",
                    "session_id = ?",
                    (session_id,),
                ),
                "events": self._portable_rows(
                    "events",
                    "session_id = ?",
                    (session_id,),
                ),
                "commands": self._portable_rows(
                    "commands",
                    "session_id = ?",
                    (session_id,),
                ),
                "goals": goals,
                "milestones": self._portable_rows_for_values(
                    "milestones",
                    "goal_id",
                    goal_ids,
                ),
                "evidence": self._portable_rows_for_values(
                    "evidence",
                    "goal_id",
                    goal_ids,
                ),
                "approvals": self._portable_rows(
                    "approvals",
                    "session_id = ?",
                    (session_id,),
                ),
                "checkpoints": self._portable_rows(
                    "checkpoints",
                    "session_id = ?",
                    (session_id,),
                ),
                "session_safety": self._portable_rows(
                    "session_safety",
                    "session_id = ?",
                    (session_id,),
                ),
                "command_envelopes": self._portable_rows(
                    "command_envelopes",
                    "session_id = ?",
                    (session_id,),
                ),
                "guard_incidents": self._portable_rows(
                    "guard_incidents",
                    "session_id = ?",
                    (session_id,),
                ),
                "context_deliveries": self._portable_rows(
                    "context_deliveries",
                    "session_id = ?",
                    (session_id,),
                ),
                "routing_decisions": self._portable_rows(
                    "routing_decisions",
                    "session_id = ?",
                    (session_id,),
                ),
                "transfers": self._portable_rows(
                    "transfers",
                    "session_id = ?",
                    (session_id,),
                ),
                "registry_entries": self._portable_rows(
                    "registry_entries",
                    "session_id = ?",
                    (session_id,),
                ),
                "ui_state": self._portable_rows(
                    "ui_state",
                    "key = ?",
                    ("session:" + session_id,),
                ),
            }
        return {
            "schema": "p13i/agent-harness/chat-record/v1",
            "database_schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "tables": tables,
        }

    def portable_global(self) -> dict[str, Any]:
        with self._lock:
            tables = {
                "usage_samples": self._portable_rows("usage_samples"),
                "mutation_receipts": self._portable_rows(
                    "mutation_receipts"
                ),
                "ui_state": self._portable_rows(
                    "ui_state",
                    "key NOT LIKE ?",
                    ("session:%",),
                ),
            }
        return {
            "schema": "p13i/agent-harness/chat-global/v1",
            "database_schema_version": SCHEMA_VERSION,
            "tables": tables,
        }

    def import_portable(
        self,
        records: list[dict[str, Any]],
        global_record: dict[str, Any],
    ) -> None:
        if self.list_sessions(include_archived=True):
            raise ConflictError("portable import requires an empty store")
        self._apply_portable(records, global_record, merge_global=False)

    def merge_portable(
        self,
        records: list[dict[str, Any]],
        global_record: dict[str, Any],
    ) -> None:
        for record in records:
            session_id = str(record.get("session_id", ""))
            with self._lock:
                existing = self._connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            if existing is not None:
                raise ConflictError(
                    "portable session UUID already exists: " + session_id
                )
        self._apply_portable(records, global_record, merge_global=True)

    def _apply_portable(
        self,
        records: list[dict[str, Any]],
        global_record: dict[str, Any],
        *,
        merge_global: bool,
    ) -> None:
        table_rows: dict[str, list[dict[str, Any]]] = {
            table: [] for table in PORTABLE_SESSION_TABLES
        }
        for record in records:
            if record.get("schema") != (
                "p13i/agent-harness/chat-record/v1"
            ):
                raise ValueError("portable chat record schema is unsupported")
            tables = _require_object(record.get("tables"), "tables")
            for table in PORTABLE_SESSION_TABLES:
                rows = tables.get(table, [])
                if not isinstance(rows, list):
                    raise ValueError(table + " must be a list")
                for row in rows:
                    table_rows[table].append(
                        _require_object(row, table + " row")
                    )
        if global_record.get("schema") != (
            "p13i/agent-harness/chat-global/v1"
        ):
            raise ValueError("portable global schema is unsupported")
        global_tables = _require_object(
            global_record.get("tables"),
            "global tables",
        )
        global_ui = global_tables.get("ui_state", [])
        if not isinstance(global_ui, list):
            raise ValueError("global ui_state must be a list")
        validated_global_ui = [
            _require_object(row, "ui_state row")
            for row in global_ui
        ]
        if not merge_global:
            table_rows["ui_state"].extend(validated_global_ui)
        with self.transaction() as connection:
            for table in PORTABLE_SESSION_TABLES:
                self._insert_portable_rows(
                    connection,
                    table,
                    table_rows[table],
                )
            if merge_global:
                self._merge_portable_rows(
                    connection,
                    "ui_state",
                    validated_global_ui,
                )
            for table in PORTABLE_GLOBAL_TABLES:
                rows = global_tables.get(table, [])
                if not isinstance(rows, list):
                    raise ValueError(table + " must be a list")
                validated = [
                    _require_object(row, table + " row")
                    for row in rows
                ]
                if merge_global:
                    self._merge_portable_rows(
                        connection,
                        table,
                        validated,
                    )
                else:
                    self._insert_portable_rows(
                        connection,
                        table,
                        validated,
                    )

    def _portable_rows(
        self,
        table: str,
        condition: str = "",
        parameters: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        columns = [
            str(row["name"])
            for row in self._connection.execute(
                "PRAGMA table_info(" + table + ")"
            ).fetchall()
        ]
        statement = "SELECT * FROM " + table
        if condition:
            statement += " WHERE " + condition
        statement += " ORDER BY " + ", ".join(columns)
        return [
            dict(row)
            for row in self._connection.execute(
                statement,
                parameters,
            ).fetchall()
        ]

    def _portable_rows_for_values(
        self,
        table: str,
        column: str,
        values: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        if not values:
            return []
        placeholders = ", ".join("?" for unused in values)
        return self._portable_rows(
            table,
            column + " IN (" + placeholders + ")",
            values,
        )

    def _insert_portable_rows(
        self,
        connection: sqlite3.Connection,
        table: str,
        rows: list[dict[str, Any]],
    ) -> None:
        for row in rows:
            columns = tuple(row)
            placeholders = ", ".join("?" for unused in columns)
            connection.execute(
                "INSERT INTO "
                + table
                + " ("
                + ", ".join(columns)
                + ") VALUES ("
                + placeholders
                + ")",
                tuple(row[column] for column in columns),
            )

    def _merge_portable_rows(
        self,
        connection: sqlite3.Connection,
        table: str,
        rows: list[dict[str, Any]],
    ) -> None:
        primary_key = [
            str(row["name"])
            for row in sorted(
                connection.execute(
                    "PRAGMA table_info(" + table + ")"
                ).fetchall(),
                key=lambda row: int(row["pk"]),
            )
            if int(row["pk"]) > 0
        ]
        if not primary_key:
            raise RuntimeError(
                "portable merge table lacks a primary key: " + table
            )
        for row in rows:
            condition = " AND ".join(
                column + " = ?" for column in primary_key
            )
            existing = connection.execute(
                "SELECT * FROM " + table + " WHERE " + condition,
                tuple(row[column] for column in primary_key),
            ).fetchone()
            if existing is None:
                self._insert_portable_rows(connection, table, [row])
                continue
            if dict(existing) != row:
                raise ConflictError(
                    "portable global row conflicts in " + table
                )

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        goal = self.goal_for_session(session_id)
        goal_value: dict[str, Any] | None = None
        evidence_value: list[dict[str, Any]] = []
        if goal is not None:
            goal_value = goal.as_dict()
            evidence_value = [
                item.as_dict() for item in self.evidence(goal.goal_id)
            ]
        return {
            "schema": "p13i/agent-harness/session-export/v1",
            "session": session.as_dict(),
            "attempts": [
                item.as_dict() for item in self.attempts(session_id)
            ],
            "events": [
                item.as_dict() for item in self.all_events(session_id)
            ],
            "goal": goal_value,
            "evidence": evidence_value,
            "checkpoints": [
                item.as_dict() for item in self.checkpoints(session_id)
            ],
            "safety": self.session_safety(session_id),
        }

    def import_session(
        self,
        payload: dict[str, Any],
        *,
        worktree: str,
        owner_host: str,
        owner_epoch: int,
    ) -> Session:
        session_value = _require_object(payload.get("session"), "session")
        session_id = str(session_value.get("session_id", ""))
        if not session_id:
            raise ValueError("session export has no session identifier")
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT owner_epoch FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                raise ConflictError("session already exists on this host")
            now = utc_now()
            connection.execute(
                """
                INSERT INTO sessions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    session_id,
                    str(session_value.get("name", session_id)),
                    str(session_value.get("workspace", "")),
                    worktree,
                    "paused",
                    "idle",
                    str(session_value.get("permission_mode", "approval")),
                    str(session_value.get("active_provider", "")),
                    str(session_value.get("model", "")),
                    str(session_value.get("effort", "")),
                    str(session_value.get("goal_id", "")),
                    owner_host,
                    owner_epoch,
                    str(session_value.get("created_at", now)),
                    now,
                    0,
                ),
            )
            self._import_attempts(connection, payload, session_id)
            self._import_events(connection, payload, session_id)
            self._import_goal(connection, payload, session_id)
            self._import_checkpoints(connection, payload, session_id)
            self._import_safety(connection, payload, session_id)
        return self.get_session(session_id)

    def _import_attempts(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
        session_id: str,
    ) -> None:
        for value in _objects(payload.get("attempts")):
            connection.execute(
                "INSERT INTO provider_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(value.get("attempt_id", new_uuid())),
                    session_id,
                    str(value.get("provider", "")),
                    str(value.get("native_session_id", "")),
                    str(value.get("model", "")),
                    str(value.get("effort", "")),
                    str(value.get("auth_mode", "subscription")),
                    str(value.get("status", "complete")),
                    str(value.get("started_at", utc_now())),
                    str(value.get("ended_at", "")),
                ),
            )

    def _import_events(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
        session_id: str,
    ) -> None:
        for value in _objects(payload.get("events")):
            connection.execute(
                """
                INSERT INTO events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    session_id,
                    int(value.get("sequence", 0)),
                    str(value.get("event_id", new_uuid())),
                    str(value.get("event_type", "provider.event")),
                    str(value.get("role", "")),
                    str(value.get("text", "")),
                    str(value.get("status", "")),
                    _dump(_object_or_empty(value.get("metadata"))),
                    str(value.get("blob_digest", "")),
                    str(value.get("turn_id", "")),
                    str(value.get("created_at", utc_now())),
                ),
            )

    def _import_goal(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
        session_id: str,
    ) -> None:
        goal = payload.get("goal")
        if not isinstance(goal, dict):
            return
        goal_id = str(goal.get("goal_id", new_uuid()))
        connection.execute(
            "INSERT INTO goals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                goal_id,
                session_id,
                str(goal.get("kind", "finite")),
                str(goal.get("objective", "")),
                str(goal.get("status", "active")),
                _dump(goal.get("constraints", [])),
                _dump(goal.get("predicates", [])),
                _dump(_object_or_empty(goal.get("budgets"))),
                str(goal.get("created_at", utc_now())),
                str(goal.get("updated_at", utc_now())),
            ),
        )
        for value in _objects(payload.get("evidence")):
            connection.execute(
                "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(value.get("evidence_id", new_uuid())),
                    goal_id,
                    str(value.get("evidence_type", "")),
                    str(value.get("subject", "")),
                    str(value.get("outcome", "")),
                    _dump(_object_or_empty(value.get("value"))),
                    str(value.get("created_at", utc_now())),
                ),
            )

    def _import_checkpoints(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
        session_id: str,
    ) -> None:
        for value in _objects(payload.get("checkpoints")):
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(value.get("checkpoint_id", new_uuid())),
                    session_id,
                    int(value.get("sequence", 0)),
                    str(value.get("provider", "")),
                    str(value.get("native_session_id", "")),
                    str(value.get("base_commit", "")),
                    str(value.get("patch_digest", "")),
                    str(value.get("untracked_digest", "")),
                    str(value.get("context_digest", "")),
                    str(value.get("created_at", utc_now())),
                ),
            )

    def _import_safety(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
        session_id: str,
    ) -> None:
        safety = payload.get("safety")
        if not isinstance(safety, dict):
            return
        profile = str(safety.get("profile", ""))
        if not profile:
            return
        now = utc_now()
        connection.execute(
            """
            INSERT INTO session_safety VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                profile,
                int(safety.get("xhigh_authorizations", 0)),
                _dump(_object_or_empty(safety.get("extensions"))),
                str(safety.get("created_at", now)),
                str(safety.get("updated_at", now)),
            ),
        )


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(field + " must be an object")
    return value


def _object_or_empty(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _objects(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _load_object(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("stored JSON is not an object")
    return decoded


def _session(row: sqlite3.Row | dict[str, Any]) -> Session:
    return Session(
        session_id=str(row["session_id"]),
        name=str(row["name"]),
        workspace=str(row["workspace"]),
        worktree=str(row["worktree"]),
        lifecycle=str(row["lifecycle"]),
        attention=str(row["attention"]),
        permission_mode=str(row["permission_mode"]),
        active_provider=str(row["active_provider"]),
        model=str(row["model"]),
        effort=str(row["effort"]),
        goal_id=str(row["goal_id"]),
        owner_host=str(row["owner_host"]),
        owner_epoch=int(row["owner_epoch"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        archived=bool(row["archived"]),
    )


def _event(row: sqlite3.Row) -> SessionEvent:
    return SessionEvent(
        session_id=str(row["session_id"]),
        sequence=int(row["sequence"]),
        event_id=str(row["event_id"]),
        event_type=str(row["event_type"]),
        role=str(row["role"]),
        text=str(row["text"]),
        status=str(row["status"]),
        metadata=_load_object(row["metadata_json"]),
        blob_digest=str(row["blob_digest"]),
        turn_id=str(row["turn_id"]),
        created_at=str(row["created_at"]),
    )


def _command(row: sqlite3.Row | dict[str, Any]) -> CommandReceipt:
    return CommandReceipt(
        command_id=str(row["command_id"]),
        idempotency_key=str(row["idempotency_key"]),
        session_id=str(row["session_id"]),
        command_type=str(row["command_type"]),
        status=str(row["status"]),
        result=_load_object(str(row["result_json"])),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _attempt(row: sqlite3.Row) -> ProviderAttempt:
    return ProviderAttempt(
        attempt_id=str(row["attempt_id"]),
        session_id=str(row["session_id"]),
        provider=str(row["provider"]),
        native_session_id=str(row["native_session_id"]),
        model=str(row["model"]),
        effort=str(row["effort"]),
        auth_mode=str(row["auth_mode"]),
        status=str(row["status"]),
        started_at=str(row["started_at"]),
        ended_at=str(row["ended_at"]),
    )


def _goal(row: sqlite3.Row) -> Goal:
    constraints = json.loads(row["constraints_json"])
    predicates = json.loads(row["predicates_json"])
    budgets = json.loads(row["budgets_json"])
    return Goal(
        goal_id=str(row["goal_id"]),
        session_id=str(row["session_id"]),
        kind=str(row["kind"]),
        objective=str(row["objective"]),
        status=str(row["status"]),
        constraints=tuple(str(item) for item in constraints),
        predicates=tuple(dict(item) for item in predicates),
        budgets=dict(budgets),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _evidence(row: sqlite3.Row) -> Evidence:
    return Evidence(
        evidence_id=str(row["evidence_id"]),
        goal_id=str(row["goal_id"]),
        evidence_type=str(row["evidence_type"]),
        subject=str(row["subject"]),
        outcome=str(row["outcome"]),
        value=_load_object(row["value_json"]),
        created_at=str(row["created_at"]),
    )


def _checkpoint(row: sqlite3.Row) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=str(row["checkpoint_id"]),
        session_id=str(row["session_id"]),
        sequence=int(row["sequence"]),
        provider=str(row["provider"]),
        native_session_id=str(row["native_session_id"]),
        base_commit=str(row["base_commit"]),
        patch_digest=str(row["patch_digest"]),
        untracked_digest=str(row["untracked_digest"]),
        context_digest=str(row["context_digest"]),
        created_at=str(row["created_at"]),
    )
