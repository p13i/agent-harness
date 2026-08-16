"""Transactional SQLite state for the harness."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

from agent_harness.errors import (
    ConflictError,
    NotFoundError,
    WorkerOwnershipLostError,
)
from agent_harness.goals import goal_contract_digest
from agent_harness.ids import derived_uuid, new_uuid, utc_now
from agent_harness.models import (
    Attention,
    Checkpoint,
    CommandReceipt,
    CommandStatus,
    Evidence,
    Goal,
    GoalStatus,
    Lifecycle,
    Milestone,
    ProviderAttempt,
    ReconciliationDecision,
    ReconciliationRecord,
    ReconciliationStatus,
    RestartRecovery,
    Session,
    SessionEvent,
)
from agent_harness.orchestration import (
    command_envelope_digest,
    creation_digest,
    legacy_command_envelope_digest,
    normalize_command_payload,
    normalize_external_ref,
    normalize_turn_ref,
    normalized_digest,
)
from agent_harness.safety import effort_requires_xhigh_authorization
from agent_harness.workspace import workspace_matches_checkpoint_collapse
from agent_harness.workspace_state import inspect_workspace

SCHEMA_VERSION = 5
SQLITE_BEGIN_ATTEMPTS = 5
SQLITE_BEGIN_BACKOFF_SECONDS = (0.025, 0.05, 0.1, 0.2)
SQLITE_BUSY_TIMEOUT_MILLISECONDS = 1000
PROOF_SNAPSHOT_MAX_PER_SESSION = 128
PROOF_SNAPSHOT_RETENTION_HOURS = 336
TRANSITION_CONTROL_COMMANDS = frozenset(
    {"interrupt", "pause", "resume", "stop", "steer"}
)
# The refusals a control command records before it can reach the
# session. The worker returns on each of these ahead of every adapter
# call, event, checkpoint, and workspace read, so a control that carries
# one declared no stage and left no material behind it. Every other
# control failure, recognized or not, is treated as material.
INERT_CONTROL_FAILURES = frozenset({"E_CONTROL_TARGET", "E_NO_ACTIVE_TURN"})
# The whole result such a refusal records. This is an exact shape, not a
# list of forbidden claims: an unrecognized extra key could carry
# material no reader here knows how to weigh, so anything but these two
# fields is read as a control that did something.
INERT_CONTROL_RESULT_KEYS = frozenset({"code", "message"})
# The metadata keys that name the command an event is about. A
# `target_command_id` names the command an event acted upon rather than
# the command that acted, so it never attributes an effect to a control.
CONTROL_EVENT_BINDINGS = (
    "control_command_id",
    "command_id",
    "prior_command_id",
)
TERMINAL_LIFECYCLES = frozenset(
    {
        Lifecycle.STOPPED,
        Lifecycle.COMPLETED,
        Lifecycle.FAILED,
    }
)
# The commands a stopped session may still hand to a worker. A resume
# is the operator reactivation, and a stop may already be queued from a
# concurrent request that lost the race to the stop that terminalized
# the session.
STOPPED_SESSION_COMMANDS = frozenset({"resume", "stop"})
DISPATCH_TRANSITION_ANCHOR_KINDS = frozenset(
    {
        "provider-result",
        "control-command",
        "resolved-reconciliation",
        "terminal-checkpoint",
    }
)

PORTABLE_SESSION_TABLES = (
    "sessions",
    "provider_attempts",
    "turns",
    "events",
    "commands",
    "goals",
    "goal_promotions",
    "goal_contract_adoptions",
    "goal_milestones",
    "milestones",
    "evidence",
    "goal_promotion_evidence",
    "dispatch_transition_policies",
    "authorization_receipts",
    "dispatch_invalidations",
    "dispatch_transition_ledger",
    "approvals",
    "checkpoints",
    "session_safety",
    "xhigh_authorization_receipts",
    "command_envelopes",
    "child_launch_gates",
    "child_launch_admissions",
    "guard_incidents",
    "context_deliveries",
    "routing_decisions",
    "transfers",
    "registry_entries",
    "ui_state",
    "command_dispatches",
    "reconciliations",
    "session_creation_receipts",
)
PORTABLE_GLOBAL_TABLES = (
    "usage_samples",
    "mutation_receipts",
)
_GOAL_PROMOTION_COLUMNS = (
    "promotion_id",
    "session_id",
    "previous_goal_id",
    "next_goal_id",
    "stage",
    "authorization_digest",
    "request_digest",
    "idempotency_key",
    "previous_goal_digest",
    "next_goal_digest",
    "created_at",
)
_GOAL_PROMOTION_EVIDENCE_COLUMNS = (
    "promotion_id",
    "source_evidence_id",
    "copied_evidence_id",
    "value_digest",
    "created_at",
)
# The checkpoint columns that carry workspace material. A context
# digest, provider label, or native session id can change while the
# tree the turn was dispatched against stands untouched, so none of
# them is evidence that an implementation landed.
_CHECKPOINT_MATERIAL_COLUMNS = (
    "base_commit",
    "patch_digest",
    "untracked_digest",
)
_GOAL_CONTRACT_ADOPTION_COLUMNS = (
    "adoption_id",
    "session_id",
    "previous_goal_id",
    "next_goal_id",
    "external_orchestrator",
    "external_job_id",
    "authorization_digest",
    "request_digest",
    "creation_digest",
    "previous_goal_digest",
    "next_goal_digest",
    "idempotency_key",
    "created_at",
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
    archived INTEGER NOT NULL DEFAULT 0,
    external_orchestrator TEXT NOT NULL DEFAULT '',
    external_job_id TEXT NOT NULL DEFAULT '',
    creation_digest TEXT NOT NULL DEFAULT '',
    CHECK (
        (external_orchestrator = '' AND external_job_id = '')
        OR
        (external_orchestrator != '' AND external_job_id != '')
    )
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
    completed_at TEXT NOT NULL,
    turn_step_id TEXT NOT NULL DEFAULT '',
    turn_agent_role TEXT NOT NULL DEFAULT ''
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
CREATE INDEX IF NOT EXISTS events_command_instruction_v2
ON events(
    session_id,
    CASE WHEN json_valid(metadata_json)
        THEN json_extract(metadata_json, '$.command_id')
        ELSE NULL
    END,
    sequence
) WHERE event_type = 'user.message';
CREATE TABLE IF NOT EXISTS commands (
    idempotency_key TEXT PRIMARY KEY,
    command_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    command_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turn_step_id TEXT NOT NULL DEFAULT '',
    turn_agent_role TEXT NOT NULL DEFAULT ''
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
    updated_at TEXT NOT NULL,
    permitted_providers_json TEXT NOT NULL DEFAULT '[]',
    permitted_efforts_json TEXT NOT NULL DEFAULT '[]',
    max_concurrency INTEGER NOT NULL DEFAULT 1,
    completion_policy TEXT NOT NULL DEFAULT 'evidence-all',
    incident_policy TEXT NOT NULL DEFAULT 'recover-then-pause'
);
CREATE TABLE IF NOT EXISTS goal_promotions (
    promotion_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    previous_goal_id TEXT NOT NULL REFERENCES goals(goal_id),
    next_goal_id TEXT NOT NULL REFERENCES goals(goal_id),
    stage TEXT NOT NULL,
    authorization_digest TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    previous_goal_digest TEXT NOT NULL,
    next_goal_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS goal_contract_adoptions (
    adoption_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    previous_goal_id TEXT NOT NULL,
    next_goal_id TEXT NOT NULL REFERENCES goals(goal_id),
    external_orchestrator TEXT NOT NULL,
    external_job_id TEXT NOT NULL,
    authorization_digest TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    creation_digest TEXT NOT NULL,
    previous_goal_digest TEXT NOT NULL,
    next_goal_digest TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS goal_milestones (
    goal_id TEXT NOT NULL REFERENCES goals(goal_id),
    milestone_id TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    dependencies_json TEXT NOT NULL,
    predicates_json TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY(goal_id, milestone_id)
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
CREATE TABLE IF NOT EXISTS goal_promotion_evidence (
    promotion_id TEXT NOT NULL REFERENCES goal_promotions(promotion_id),
    source_evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
    copied_evidence_id TEXT NOT NULL UNIQUE REFERENCES evidence(evidence_id),
    value_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(promotion_id, source_evidence_id)
);
CREATE TABLE IF NOT EXISTS authorization_receipts (
    authorization_digest TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    operation TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    schema TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dispatch_transition_policies (
    policy_sha256 TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    goal_id TEXT NOT NULL REFERENCES goals(goal_id),
    epoch_id TEXT NOT NULL,
    schema TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, goal_id, epoch_id)
);
CREATE INDEX IF NOT EXISTS dispatch_transition_policies_session
ON dispatch_transition_policies(session_id, goal_id, epoch_id);
CREATE TABLE IF NOT EXISTS dispatch_invalidations (
    invalidation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    reason TEXT NOT NULL,
    authorization_digest TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dispatch_transition_ledger (
    invalidation_id TEXT PRIMARY KEY
        REFERENCES dispatch_invalidations(invalidation_id),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    goal_id TEXT NOT NULL REFERENCES goals(goal_id),
    epoch_id TEXT NOT NULL,
    transition_sequence INTEGER NOT NULL,
    policy_sha256 TEXT NOT NULL
        REFERENCES dispatch_transition_policies(policy_sha256),
    authorization_digest TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    prior_command_id TEXT NOT NULL,
    prior_command_type TEXT NOT NULL,
    prior_anchor_kind TEXT NOT NULL,
    prior_reconciliation_id TEXT NOT NULL,
    prior_reconciliation_resolution TEXT NOT NULL,
    prior_checkpoint_id TEXT NOT NULL,
    prior_generation_digest TEXT NOT NULL,
    prior_material_digest TEXT NOT NULL,
    next_turn_ref_json TEXT NOT NULL,
    next_command_digest TEXT NOT NULL,
    state TEXT NOT NULL,
    reserved_command_id TEXT NOT NULL,
    consumed_command_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, goal_id, epoch_id, transition_sequence)
);
CREATE INDEX IF NOT EXISTS dispatch_transition_ledger_session
ON dispatch_transition_ledger(
    session_id, goal_id, epoch_id, transition_sequence
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
CREATE TABLE IF NOT EXISTS xhigh_authorization_receipts (
    authorization_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    command_id TEXT NOT NULL UNIQUE REFERENCES commands(command_id),
    provider TEXT NOT NULL,
    effort TEXT NOT NULL,
    command_request_digest TEXT NOT NULL,
    authorization_request_digest TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    expires_at TEXT NOT NULL,
    consumed_attempt_id TEXT NOT NULL,
    consumed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS xhigh_authorizations_pending
ON xhigh_authorization_receipts(session_id, consumed_at, expires_at);
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
CREATE TABLE IF NOT EXISTS child_launch_gates (
    command_id TEXT PRIMARY KEY REFERENCES commands(command_id),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    permit_limit INTEGER NOT NULL,
    consumed INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (permit_limit >= 0),
    CHECK (consumed >= 0 AND consumed <= permit_limit)
);
CREATE TABLE IF NOT EXISTS child_launch_admissions (
    command_id TEXT NOT NULL REFERENCES child_launch_gates(command_id),
    admission_key TEXT NOT NULL,
    admitted INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(command_id, admission_key),
    CHECK (admitted IN (0, 1))
);
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
    command_id TEXT NOT NULL DEFAULT '',
    attempt_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'delivered',
    payload_digest TEXT NOT NULL DEFAULT '',
    accepted_at TEXT NOT NULL DEFAULT '',
    transport TEXT NOT NULL DEFAULT 'context-package',
    PRIMARY KEY(attempt_id)
);
CREATE TABLE IF NOT EXISTS process_leases (
    lease_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    command_id TEXT NOT NULL DEFAULT '',
    attempt_id TEXT NOT NULL DEFAULT '',
    worker_incarnation TEXT NOT NULL DEFAULT '',
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
CREATE TABLE IF NOT EXISTS session_creation_receipts (
    idempotency_key TEXT PRIMARY KEY,
    request_digest TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS command_dispatches (
    attempt_id TEXT PRIMARY KEY REFERENCES provider_attempts(attempt_id),
    command_id TEXT NOT NULL REFERENCES commands(command_id),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    turn_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL REFERENCES checkpoints(checkpoint_id),
    crossed_boundary INTEGER NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS command_dispatches_command
ON command_dispatches(command_id, crossed_boundary, state);
CREATE TABLE IF NOT EXISTS proof_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    through_sequence INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS proof_snapshots_session
ON proof_snapshots(session_id, created_at);
CREATE TABLE IF NOT EXISTS reconciliations (
    reconciliation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    command_id TEXT NOT NULL UNIQUE REFERENCES commands(command_id),
    pre_dispatch_checkpoint_id TEXT NOT NULL
        REFERENCES checkpoints(checkpoint_id),
    current_workspace_digest TEXT NOT NULL,
    current_workspace_summary TEXT NOT NULL,
    provider_attempts_json TEXT NOT NULL,
    safety_consumption_json TEXT NOT NULL,
    status TEXT NOT NULL,
    resolution TEXT NOT NULL,
    audit_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS reconciliations_pending
ON reconciliations(session_id, status, created_at);
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
        self._transaction_depth = 0
        self._savepoint_sequence = 0
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
        self._connection.execute(
            "PRAGMA busy_timeout=" + str(SQLITE_BUSY_TIMEOUT_MILLISECONDS)
        )

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connection
            connection.executescript(SCHEMA)
        with self.transaction() as connection:
            row = connection.execute("SELECT version FROM schema_meta").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_meta(version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
                self._migrate_to_v4(connection)
                self._migrate_to_v5(connection)
            else:
                version = int(row["version"])
                if version in {1, 2}:
                    self._migrate_to_v3(connection)
                    version = 3
                if version == 3:
                    self._migrate_to_v4(connection)
                    connection.execute(
                        "UPDATE schema_meta SET version = ?",
                        (4,),
                    )
                    version = 4
                if version == 4:
                    self._migrate_to_v5(connection)
                    connection.execute(
                        "UPDATE schema_meta SET version = ?",
                        (SCHEMA_VERSION,),
                    )
                    version = SCHEMA_VERSION
                if version != SCHEMA_VERSION:
                    raise RuntimeError("unsupported database schema version")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS sessions_external_ref
                ON sessions(external_orchestrator, external_job_id)
                WHERE external_orchestrator != ''
                """
            )
        os.chmod(self.path, 0o600)

    def _ensure_goal_policy_columns(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        for column, declaration in (
            ("permitted_providers_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("permitted_efforts_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("max_concurrency", "INTEGER NOT NULL DEFAULT 1"),
            (
                "completion_policy",
                "TEXT NOT NULL DEFAULT 'evidence-all'",
            ),
            (
                "incident_policy",
                "TEXT NOT NULL DEFAULT 'recover-then-pause'",
            ),
        ):
            self._add_column(connection, "goals", column, declaration)

    def _ensure_context_delivery_columns(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        for column, declaration in (
            ("command_id", "TEXT NOT NULL DEFAULT ''"),
            ("attempt_id", "TEXT NOT NULL DEFAULT ''"),
            ("state", "TEXT NOT NULL DEFAULT 'delivered'"),
            ("payload_digest", "TEXT NOT NULL DEFAULT ''"),
            ("accepted_at", "TEXT NOT NULL DEFAULT ''"),
            ("transport", "TEXT NOT NULL DEFAULT 'context-package'"),
        ):
            self._add_column(
                connection,
                "context_deliveries",
                column,
                declaration,
            )

    def _ensure_process_lease_columns(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        for column in (
            "command_id",
            "attempt_id",
            "worker_incarnation",
        ):
            self._add_column(
                connection,
                "process_leases",
                column,
                "TEXT NOT NULL DEFAULT ''",
            )

    def _migrate_to_v4(self, connection: sqlite3.Connection) -> None:
        self._ensure_goal_policy_columns(connection)
        self._ensure_context_delivery_columns(connection)
        self._ensure_process_lease_columns(connection)

    def _migrate_to_v5(self, connection: sqlite3.Connection) -> None:
        self._ensure_context_delivery_columns(connection)
        primary_key = connection.execute(
            "PRAGMA table_info(context_deliveries)"
        ).fetchall()
        primary_key_columns = [
            str(row["name"])
            for row in primary_key
            if int(row["pk"]) > 0
        ]
        if primary_key_columns != ["attempt_id"]:
            rows = connection.execute(
                "SELECT rowid, * FROM context_deliveries ORDER BY rowid"
            ).fetchall()
            attempt_counts: dict[str, int] = {}
            for row in rows:
                attempt_id = str(row["attempt_id"])
                if attempt_id:
                    attempt_counts[attempt_id] = (
                        attempt_counts.get(attempt_id, 0) + 1
                    )
            assigned_attempt_ids = {
                attempt_id
                for attempt_id, count in attempt_counts.items()
                if count == 1
            }
            connection.execute(
                "ALTER TABLE context_deliveries RENAME TO context_deliveries_v4"
            )
            connection.execute(
                """
                CREATE TABLE context_deliveries (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    provider TEXT NOT NULL,
                    context_digest TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    delivered_at TEXT NOT NULL,
                    command_id TEXT NOT NULL DEFAULT '',
                    attempt_id TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'delivered',
                    payload_digest TEXT NOT NULL DEFAULT '',
                    accepted_at TEXT NOT NULL DEFAULT '',
                    transport TEXT NOT NULL DEFAULT 'context-package',
                    PRIMARY KEY(attempt_id)
                )
                """
            )
            for row in rows:
                attempt_id = str(row["attempt_id"])
                duplicate_attempt = attempt_counts.get(attempt_id, 0) > 1
                if not attempt_id or duplicate_attempt:
                    legacy_identity = {
                        "attempt_id": attempt_id,
                        "rowid": int(row["rowid"]),
                        "session_id": str(row["session_id"]),
                        "provider": str(row["provider"]),
                        "context_digest": str(row["context_digest"]),
                    }
                    legacy_prefix = "legacy-"
                    if duplicate_attempt:
                        legacy_prefix = "legacy-duplicate-"
                    attempt_id = legacy_prefix + normalized_digest(legacy_identity)
                    collision = 0
                    while attempt_id in assigned_attempt_ids:
                        collision += 1
                        legacy_identity["collision"] = collision
                        attempt_id = legacy_prefix + normalized_digest(
                            legacy_identity
                        )
                assigned_attempt_ids.add(attempt_id)
                connection.execute(
                    """
                    INSERT INTO context_deliveries VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        str(row["session_id"]),
                        str(row["provider"]),
                        str(row["context_digest"]),
                        str(row["checkpoint_id"]),
                        str(row["delivered_at"]),
                        str(row["command_id"]),
                        attempt_id,
                        str(row["state"]),
                        str(row["payload_digest"]),
                        str(row["accepted_at"]),
                        str(row["transport"]),
                    ),
                )
            connection.execute("DROP TABLE context_deliveries_v4")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS context_deliveries_context
            ON context_deliveries(session_id, provider, context_digest)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS events_command_instruction_v2
            ON events(
                session_id,
                CASE WHEN json_valid(metadata_json)
                    THEN json_extract(metadata_json, '$.command_id')
                    ELSE NULL
                END,
                sequence
            ) WHERE event_type = 'user.message'
            """
        )
        missing = connection.execute(
            """
            SELECT d.*, a.provider
            FROM command_dispatches AS d
            JOIN provider_attempts AS a USING(attempt_id)
            JOIN commands AS m USING(command_id)
            LEFT JOIN context_deliveries AS c USING(attempt_id)
            WHERE d.crossed_boundary = 1
                AND m.status = 'dispatching'
                AND c.attempt_id IS NULL
            """
        ).fetchall()
        for row in missing:
            attempt_id = str(row["attempt_id"])
            context_digest = "legacy-ambiguous-" + hashlib.sha256(
                attempt_id.encode("utf-8")
            ).hexdigest()
            connection.execute(
                """
                INSERT INTO context_deliveries VALUES (
                    ?, ?, ?, ?, ?, ?, ?, 'legacy-ambiguous', '', '',
                    'legacy-unknown'
                )
                """,
                (
                    str(row["session_id"]),
                    str(row["provider"]),
                    context_digest,
                    str(row["checkpoint_id"]),
                    str(row["updated_at"]),
                    str(row["command_id"]),
                    attempt_id,
                ),
            )

    def _migrate_to_v3(self, connection: sqlite3.Connection) -> None:
        self._add_column(
            connection,
            "sessions",
            "external_orchestrator",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._add_column(
            connection,
            "sessions",
            "external_job_id",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._add_column(
            connection,
            "sessions",
            "creation_digest",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._add_column(
            connection,
            "turns",
            "turn_step_id",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._add_column(
            connection,
            "turns",
            "turn_agent_role",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._add_column(
            connection,
            "commands",
            "turn_step_id",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._add_column(
            connection,
            "commands",
            "turn_agent_role",
            "TEXT NOT NULL DEFAULT ''",
        )

    def _add_column(
        self,
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(" + table + ")").fetchall()
        }
        if column in columns:
            return
        connection.execute(
            "ALTER TABLE " + table + " ADD COLUMN " + column + " " + declaration
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            outermost = self._transaction_depth == 0
            savepoint = ""
            if outermost:
                self._begin_immediate()
            else:
                self._savepoint_sequence += 1
                savepoint = "nested_" + str(self._savepoint_sequence)
                self._connection.execute("SAVEPOINT " + savepoint)
            self._transaction_depth += 1
            try:
                yield self._connection
            except BaseException:
                self._transaction_depth -= 1
                if outermost:
                    self._connection.execute("ROLLBACK")
                else:
                    self._connection.execute("ROLLBACK TO " + savepoint)
                    self._connection.execute("RELEASE " + savepoint)
                raise
            else:
                self._transaction_depth -= 1
                if outermost:
                    self._connection.execute("COMMIT")
                else:
                    self._connection.execute("RELEASE " + savepoint)

    def _begin_immediate(self) -> None:
        for attempt in range(SQLITE_BEGIN_ATTEMPTS):
            try:
                self._begin_immediate_once()
                return
            except sqlite3.OperationalError as error:
                final_attempt = attempt + 1 == SQLITE_BEGIN_ATTEMPTS
                if not _sqlite_contention(error) or final_attempt:
                    raise
                time.sleep(SQLITE_BEGIN_BACKOFF_SECONDS[attempt])

    def _begin_immediate_once(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def integrity_check(self) -> str:
        with self._lock:
            row = self._connection.execute("PRAGMA quick_check").fetchone()
        if row is None:
            return ""
        return str(row[0])

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
        external_ref = normalize_external_ref(session.external_ref)
        with self.transaction() as connection:
            try:
                self._insert_session(
                    connection,
                    replace(session, external_ref=external_ref),
                    "",
                )
            except sqlite3.IntegrityError as error:
                raise ConflictError(
                    "session identifier or external reference already exists"
                ) from error
        return replace(session, external_ref=external_ref)

    def ensure_session(
        self,
        session: Session,
        creation_input: dict[str, Any],
        *,
        idempotency_key: str = "",
    ) -> tuple[Session, bool]:
        request_digest = creation_digest(creation_input)
        external_ref = normalize_external_ref(session.external_ref)
        input_ref = normalize_external_ref(creation_input.get("external_ref"))
        if external_ref != input_ref:
            raise ValueError("session and creation input external_ref must match")
        normalized = replace(
            session,
            external_ref=external_ref,
            creation_digest=request_digest,
        )
        with self.transaction() as connection:
            if idempotency_key:
                receipt = connection.execute(
                    """
                    SELECT * FROM session_creation_receipts
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if receipt is not None:
                    if str(receipt["request_digest"]) != request_digest:
                        raise ConflictError(
                            "idempotency key was already used with "
                            "different session input"
                        )
                    existing = connection.execute(
                        "SELECT * FROM sessions WHERE session_id = ?",
                        (receipt["session_id"],),
                    ).fetchone()
                    if existing is None:
                        raise RuntimeError("session creation receipt has no session")
                    return _session(existing), False
            if external_ref:
                existing = connection.execute(
                    """
                    SELECT * FROM sessions
                    WHERE external_orchestrator = ?
                    AND external_job_id = ?
                    """,
                    (
                        external_ref["orchestrator"],
                        external_ref["job_id"],
                    ),
                ).fetchone()
                if existing is not None:
                    if str(existing["creation_digest"]) != request_digest:
                        raise ConflictError(
                            "external reference already names a session "
                            "with different creation input"
                        )
                    existing_session = _session(existing)
                    if idempotency_key:
                        self._insert_creation_receipt(
                            connection,
                            idempotency_key,
                            request_digest,
                            existing_session,
                        )
                    return existing_session, False
            try:
                self._insert_session(
                    connection,
                    normalized,
                    request_digest,
                )
            except sqlite3.IntegrityError as error:
                raise ConflictError(
                    "session identifier or external reference already exists"
                ) from error
            if idempotency_key:
                self._insert_creation_receipt(
                    connection,
                    idempotency_key,
                    request_digest,
                    normalized,
                )
        return normalized, True

    def existing_ensured_session(
        self,
        creation_input: dict[str, Any],
        *,
        idempotency_key: str = "",
        external_ref: dict[str, str] | None = None,
    ) -> Session | None:
        request_digest = creation_digest(creation_input)
        normalized_ref = normalize_external_ref(external_ref)
        if external_ref is None:
            normalized_ref = normalize_external_ref(creation_input.get("external_ref"))
        by_key: Session | None = None
        by_reference: Session | None = None
        with self._lock:
            if idempotency_key:
                receipt = self._connection.execute(
                    """
                    SELECT * FROM session_creation_receipts
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if receipt is not None:
                    if str(receipt["request_digest"]) != request_digest:
                        raise ConflictError(
                            "idempotency key was already used with "
                            "different session input"
                        )
                    row = self._connection.execute(
                        "SELECT * FROM sessions WHERE session_id = ?",
                        (receipt["session_id"],),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("session creation receipt has no session")
                    by_key = _session(row)
            if normalized_ref:
                row = self._connection.execute(
                    """
                    SELECT * FROM sessions
                    WHERE external_orchestrator = ?
                    AND external_job_id = ?
                    """,
                    (
                        normalized_ref["orchestrator"],
                        normalized_ref["job_id"],
                    ),
                ).fetchone()
                if row is not None:
                    if str(row["creation_digest"]) != request_digest:
                        raise ConflictError(
                            "external reference already names a session "
                            "with different creation input"
                        )
                    by_reference = _session(row)
        if (
            by_key is not None
            and by_reference is not None
            and by_key.session_id != by_reference.session_id
        ):
            raise ConflictError(
                "idempotency key and external reference name different sessions"
            )
        if by_key is not None:
            return by_key
        return by_reference

    def _insert_session(
        self,
        connection: sqlite3.Connection,
        session: Session,
        request_digest: str,
    ) -> None:
        external_ref = normalize_external_ref(session.external_ref)
        connection.execute(
            """
                INSERT INTO sessions(
                    session_id, name, workspace, worktree, lifecycle,
                    attention, permission_mode, active_provider, model,
                    effort, goal_id, owner_host, owner_epoch, created_at,
                    updated_at, archived, external_orchestrator,
                    external_job_id, creation_digest
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
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
                external_ref.get("orchestrator", ""),
                external_ref.get("job_id", ""),
                request_digest,
            ),
        )

    def _insert_creation_receipt(
        self,
        connection: sqlite3.Connection,
        idempotency_key: str,
        request_digest: str,
        session: Session,
    ) -> None:
        connection.execute(
            """
            INSERT INTO session_creation_receipts VALUES (?, ?, ?, ?, ?)
            """,
            (
                idempotency_key,
                request_digest,
                session.session_id,
                _dump({"session": session.as_dict()}),
                utc_now(),
            ),
        )

    def create_fork(
        self,
        source_session_id: str,
        fork: Session,
        *,
        external_ref: dict[str, str] | None = None,
    ) -> Session:
        self.get_session(source_session_id)
        normalized_ref = normalize_external_ref(external_ref)
        return self.create_session(replace(fork, external_ref=normalized_ref))

    def get_session(self, session_id: str) -> Session:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("session")
        return _session(row)

    def get_session_by_external_ref(
        self,
        orchestrator: str,
        job_id: str,
    ) -> Session | None:
        external_ref = normalize_external_ref(
            {"orchestrator": orchestrator, "job_id": job_id}
        )
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM sessions
                WHERE external_orchestrator = ? AND external_job_id = ?
                """,
                (
                    external_ref["orchestrator"],
                    external_ref["job_id"],
                ),
            ).fetchone()
        if row is None:
            return None
        return _session(row)

    def find_session_by_external_ref(
        self,
        orchestrator: str,
        job_id: str,
    ) -> Session | None:
        return self.get_session_by_external_ref(orchestrator, job_id)

    def list_sessions(
        self,
        include_archived: bool = False,
        *,
        external_ref: dict[str, str] | None = None,
    ) -> list[Session]:
        query = "SELECT * FROM sessions"
        clauses: list[str] = []
        parameters: list[Any] = []
        if not include_archived:
            clauses.append("archived = 0")
        if external_ref is not None:
            normalized = normalize_external_ref(external_ref)
            if not normalized:
                raise ValueError("external_ref lookup requires identifiers")
            clauses.extend(
                [
                    "external_orchestrator = ?",
                    "external_job_id = ?",
                ]
            )
            parameters.extend([normalized["orchestrator"], normalized["job_id"]])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, session_id"
        with self._lock:
            rows = self._connection.execute(
                query,
                tuple(parameters),
            ).fetchall()
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
                "UPDATE sessions SET " + assignments + " WHERE session_id = ?",
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
        with self.transaction() as connection:
            created_at = utc_now()
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

    def context_events_for_command(
        self,
        session_id: str,
        command_id: str,
        instruction_sequence: int,
        *,
        limit: int = 5000,
    ) -> list[SessionEvent]:
        bounded = max(1, min(limit, 5000))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM events AS e
                WHERE e.session_id = ? AND (
                    e.sequence < ?
                    OR (
                        e.sequence > ?
                        AND (
                            e.turn_id IN (
                                SELECT turn_id FROM command_dispatches
                                WHERE command_id = ?
                            )
                            OR CASE
                                WHEN json_valid(e.metadata_json)
                                THEN json_extract(
                                    e.metadata_json,
                                    '$.command_id'
                                )
                                ELSE NULL
                            END = ?
                            OR CASE
                                WHEN json_valid(e.metadata_json)
                                THEN json_extract(
                                    e.metadata_json,
                                    '$.target_command_id'
                                )
                                ELSE NULL
                            END = ?
                        )
                    )
                )
                ORDER BY e.sequence DESC LIMIT ?
                """,
                (
                    session_id,
                    instruction_sequence,
                    instruction_sequence,
                    command_id,
                    command_id,
                    command_id,
                    bounded,
                ),
            ).fetchall()
        rows.reverse()
        return [_event(row) for row in rows]

    def command_instruction_sequence(
        self,
        session_id: str,
        command_id: str,
    ) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT sequence FROM events
                WHERE session_id = ? AND event_type = 'user.message'
                    AND CASE WHEN json_valid(metadata_json)
                        THEN json_extract(metadata_json, '$.command_id')
                        ELSE NULL
                    END = ?
                ORDER BY sequence LIMIT 1
                """,
                (session_id, command_id),
            ).fetchone()
            if row is not None:
                return int(row["sequence"])
        raise NotFoundError("command instruction event")

    def repair_command_instruction_event(
        self,
        session_id: str,
        command_id: str,
    ) -> int:
        with self.transaction() as connection:
            command = connection.execute(
                """
                SELECT * FROM commands
                WHERE session_id = ? AND command_id = ?
                    AND command_type = 'message'
                """,
                (session_id, command_id),
            ).fetchone()
            if command is None:
                raise NotFoundError("command instruction event")
            if str(command["status"]) not in {
                CommandStatus.QUEUED,
                CommandStatus.DISPATCHING,
            }:
                raise ConflictError(
                    "terminal message command cannot repair its instruction event: "
                    + command_id
                )
            payload = _load_object(str(command["payload_json"]))
            try:
                repaired_at = utc_now()
                return self._ensure_command_instruction_event(
                    connection,
                    session_id,
                    command_id,
                    payload,
                    created_at=repaired_at,
                    repaired=True,
                )
            except NotFoundError as error:
                raise ConflictError(
                    "message command has no recoverable instruction text: "
                    + command_id
                ) from error

    def _ensure_command_instruction_event(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        command_id: str,
        payload: dict[str, Any],
        *,
        created_at: str,
        repaired: bool,
    ) -> int:
        row = connection.execute(
            """
            SELECT sequence FROM events
            WHERE session_id = ? AND event_type = 'user.message'
                AND CASE WHEN json_valid(metadata_json)
                    THEN json_extract(metadata_json, '$.command_id')
                    ELSE NULL
                END = ?
            ORDER BY sequence LIMIT 1
            """,
            (session_id, command_id),
        ).fetchone()
        if row is not None:
            return int(row["sequence"])
        text = str(payload.get("text", "")).strip()
        if not text:
            raise NotFoundError("command instruction event")
        sequence_row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) AS sequence
            FROM events WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        sequence = int(sequence_row["sequence"]) + 1
        metadata = {
            "command_id": command_id,
            "turn_ref": normalize_turn_ref(payload.get("turn_ref")),
        }
        if repaired:
            metadata["repaired"] = True
        connection.execute(
            """
            INSERT INTO events(
                session_id, sequence, event_id, event_type, role,
                text, status, metadata_json, blob_digest, turn_id,
                created_at
            ) VALUES (?, ?, ?, 'user.message', 'user', ?, 'accepted',
                ?, '', '', ?)
            """,
            (
                session_id,
                sequence,
                new_uuid(),
                text,
                _dump(metadata),
                created_at,
            ),
        )
        connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (created_at, session_id),
        )
        return sequence

    def event_count(self, session_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["count"])

    def fork_lineage(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT metadata_json FROM events
                WHERE session_id = ? AND event_type = 'session.forked'
                ORDER BY sequence LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return {}
            metadata = _load_object(row["metadata_json"])
            checkpoint_id = str(metadata.get("source_checkpoint_id", ""))
            checkpoint = self._connection.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
        if checkpoint is None:
            raise ConflictError("fork source checkpoint is missing")
        return {
            "source_session_id": str(metadata.get("source_session_id", "")),
            "source_sequence": int(metadata.get("source_sequence", 0)),
            "source_checkpoint_id": checkpoint_id,
            "source_context_digest": str(checkpoint["context_digest"]),
        }

    def seeded_handoff(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT metadata_json FROM events
                WHERE session_id = ? AND event_type = 'session.handoff'
                ORDER BY sequence DESC
                """,
                (session_id,),
            ).fetchall()
        for row in rows:
            metadata = _load_object(row["metadata_json"])
            if str(metadata.get("origin", "")) == "fork-seed":
                return metadata
        return {}

    def context_history_summary(
        self,
        session_id: str,
        before_sequence: int,
    ) -> dict[str, Any]:
        if before_sequence <= 0:
            return {}
        with self._lock:
            aggregate = self._connection.execute(
                """
                SELECT COUNT(*) AS count, MIN(sequence) AS first_sequence,
                    MAX(sequence) AS last_sequence
                FROM events WHERE session_id = ? AND sequence <= ?
                """,
                (session_id, before_sequence),
            ).fetchone()
            type_rows = self._connection.execute(
                """
                SELECT event_type, COUNT(*) AS count
                FROM events WHERE session_id = ? AND sequence <= ?
                GROUP BY event_type ORDER BY event_type
                """,
                (session_id, before_sequence),
            ).fetchall()
            digest_rows = self._connection.execute(
                """
                SELECT sequence, event_type, role, text, status,
                    metadata_json, blob_digest, turn_id
                FROM events WHERE session_id = ? AND sequence <= ?
                ORDER BY sequence
                """,
                (session_id, before_sequence),
            ).fetchall()
            first_rows = self._connection.execute(
                """
                SELECT * FROM events
                WHERE session_id = ? AND sequence <= ?
                ORDER BY sequence LIMIT 20
                """,
                (session_id, before_sequence),
            ).fetchall()
            last_rows = self._connection.execute(
                """
                SELECT * FROM events
                WHERE session_id = ? AND sequence <= ?
                ORDER BY sequence DESC LIMIT 80
                """,
                (session_id, before_sequence),
            ).fetchall()
        if aggregate is None or int(aggregate["count"]) == 0:
            return {}
        anchor_rows: dict[int, sqlite3.Row] = {}
        for row in [*first_rows, *last_rows]:
            anchor_rows[int(row["sequence"])] = row
        anchors = []
        for sequence in sorted(anchor_rows):
            event = _event(anchor_rows[sequence])
            anchors.append(
                {
                    "sequence": event.sequence,
                    "event_type": event.event_type,
                    "role": event.role,
                    "status": event.status,
                    "text": event.text[:500],
                    "metadata": _bounded_json_value(event.metadata),
                }
            )
        history_digest = hashlib.sha256()
        for row in digest_rows:
            history_digest.update(
                _dump(
                    {
                        "sequence": int(row["sequence"]),
                        "event_type": str(row["event_type"]),
                        "role": str(row["role"]),
                        "text": str(row["text"]),
                        "status": str(row["status"]),
                        "metadata": _load_object(row["metadata_json"]),
                        "blob_digest": str(row["blob_digest"]),
                        "turn_id": str(row["turn_id"]),
                    }
                ).encode("utf-8")
            )
            history_digest.update(b"\n")
        return {
            "schema": "p13i/agent-harness/compacted-history/v1",
            "event_count": int(aggregate["count"]),
            "first_sequence": int(aggregate["first_sequence"]),
            "last_sequence": int(aggregate["last_sequence"]),
            "history_digest": history_digest.hexdigest(),
            "event_type_counts": {
                str(row["event_type"]): int(row["count"]) for row in type_rows
            },
            "anchors": anchors,
        }

    def context_unresolved_decisions(self, session_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for approval in self.pending_approvals(session_id):
            result.append(
                {
                    "kind": "approval",
                    "id": str(approval.get("approval_id", "")),
                    "status": str(approval.get("status", "")),
                    "method": str(approval.get("kind", "")),
                    "prompt": str(approval.get("prompt", "")),
                    "reason": str(approval.get("reason", "")),
                    "created_at": str(approval.get("created_at", "")),
                }
            )
        for reconciliation in self.pending_reconciliations(session_id):
            result.append(
                {
                    "kind": "reconciliation",
                    "id": reconciliation.reconciliation_id,
                    "status": reconciliation.status,
                    "command_id": reconciliation.command_id,
                    "created_at": reconciliation.created_at,
                }
            )
        result.sort(key=lambda item: (str(item["kind"]), str(item["id"])))
        return result

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

    def proof_event_count(
        self,
        session_id: str,
        through_sequence: int,
    ) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM events
                WHERE session_id = ? AND sequence <= ?
                """,
                (session_id, through_sequence),
            ).fetchone()
        return int(row["count"])

    def proof_event_rows(
        self,
        session_id: str,
        through_sequence: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("proof event limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM events
                WHERE session_id = ? AND sequence <= ?
                ORDER BY sequence LIMIT ?
                """,
                (session_id, through_sequence, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def proof_source(
        self,
        session_id: str,
        through_sequence: int | None,
        event_limit: int,
    ) -> dict[str, Any]:
        """Capture every proof input from one SQLite read snapshot."""
        if event_limit < 1:
            raise ValueError("proof event limit must be positive")
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                initial_last_sequence = self.last_sequence(session_id)
                selected_sequence = through_sequence
                if selected_sequence is None:
                    selected_sequence = initial_last_sequence
                if selected_sequence < 0:
                    raise ValueError("through_sequence must not be negative")
                if selected_sequence > initial_last_sequence:
                    raise ValueError(
                        "through_sequence exceeds current session sequence"
                    )
                event_count = self.proof_event_count(
                    session_id,
                    selected_sequence,
                )
                if event_count != selected_sequence:
                    raise ValueError("proof event sequence is not contiguous")
                if event_count > event_limit:
                    return {
                        "through_sequence": selected_sequence,
                        "event_count": event_count,
                    }
                session = self.get_session(session_id)
                portable_session = self.portable_session(
                    session_id,
                    include_events=False,
                )
                event_rows = self.proof_event_rows(
                    session_id,
                    selected_sequence,
                    event_limit,
                )
                for expected_sequence, event_row in enumerate(
                    event_rows,
                    start=1,
                ):
                    if int(event_row["sequence"]) != expected_sequence:
                        raise ValueError("proof event sequence is not contiguous")
                goal = self.goal_for_session(session_id)
                evidence: list[Evidence] = []
                if goal is not None:
                    evidence = self.evidence(goal.goal_id)
                workers = [
                    item
                    for item in self.worker_registrations()
                    if str(item.get("session_id", "")) == session_id
                ]
                return {
                    "through_sequence": selected_sequence,
                    "event_count": event_count,
                    "session": session,
                    "portable_session": portable_session,
                    "event_rows": event_rows,
                    "goal": goal,
                    "evidence": evidence,
                    "portable_global": self.portable_global(),
                    "leases": self.process_leases(session_id),
                    "workers": workers,
                    "safety": self.session_safety(session_id),
                    "envelopes": self.session_envelopes(session_id),
                    "incidents": self.guard_incidents(session_id),
                    "transition_anchor": self.dispatch_transition_anchor(session_id),
                }
            finally:
                self._connection.execute("COMMIT")

    def enqueue_command(
        self,
        session_id: str,
        command_type: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> CommandReceipt:
        receipt, unused_created = self.ensure_command(
            session_id,
            command_type,
            payload,
            idempotency_key,
        )
        del unused_created
        return receipt

    def ensure_command(
        self,
        session_id: str,
        command_type: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> tuple[CommandReceipt, bool]:
        return self._ensure_command(
            session_id,
            command_type,
            payload,
            idempotency_key,
            instruction_event=False,
        )

    def ensure_message_command(
        self,
        session_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> tuple[CommandReceipt, bool]:
        return self._ensure_command(
            session_id,
            "message",
            payload,
            idempotency_key,
            instruction_event=True,
        )

    def _ensure_command(
        self,
        session_id: str,
        command_type: str,
        payload: dict[str, Any],
        idempotency_key: str,
        *,
        instruction_event: bool,
    ) -> tuple[CommandReceipt, bool]:
        normalized_payload = normalize_command_payload(payload)
        turn_ref = normalize_turn_ref(normalized_payload.get("turn_ref"))
        with self.transaction() as connection:
            rejection = _terminal_command_rejection(
                _session_lifecycle(connection, session_id),
                command_type,
            )
            existing = connection.execute(
                "SELECT * FROM commands WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                existing_payload = normalize_command_payload(
                    _load_object(str(existing["payload_json"]))
                )
                same_request = (
                    str(existing["session_id"]) == session_id
                    and str(existing["command_type"]) == command_type
                    and _dump(existing_payload) == _dump(normalized_payload)
                )
                if not same_request:
                    raise ConflictError(
                        "idempotency key was already used with different command input"
                    )
                if not rejection:
                    existing = self._requeue_retryable_failure(
                        connection,
                        existing,
                    )
                if instruction_event and str(existing["status"]) in {
                    CommandStatus.QUEUED,
                    CommandStatus.DISPATCHING,
                }:
                    repaired_at = utc_now()
                    try:
                        self._ensure_command_instruction_event(
                            connection,
                            session_id,
                            str(existing["command_id"]),
                            existing_payload,
                            created_at=repaired_at,
                            repaired=True,
                        )
                    except NotFoundError as error:
                        raise ConflictError(
                            "message command has no recoverable instruction text: "
                            + str(existing["command_id"])
                        ) from error
                return _command(existing), False
            if rejection:
                raise ConflictError(rejection)
            now = utc_now()
            command_id = new_uuid()
            connection.execute(
                """
                INSERT INTO commands(
                    idempotency_key, command_id, session_id,
                    command_type, payload_json, status, result_json,
                    created_at, updated_at, turn_step_id,
                    turn_agent_role
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    command_id,
                    session_id,
                    command_type,
                    _dump(normalized_payload),
                    CommandStatus.QUEUED,
                    "{}",
                    now,
                    now,
                    turn_ref.get("step_id", ""),
                    turn_ref.get("agent_role", ""),
                ),
            )
            if instruction_event:
                self._ensure_command_instruction_event(
                    connection,
                    session_id,
                    command_id,
                    normalized_payload,
                    created_at=now,
                    repaired=False,
                )
            receipt = CommandReceipt(
                command_id=command_id,
                idempotency_key=idempotency_key,
                session_id=session_id,
                command_type=command_type,
                status=CommandStatus.QUEUED,
                result={},
                created_at=now,
                updated_at=now,
                turn_ref=turn_ref,
            )
        return (
            receipt,
            True,
        )

    def _requeue_retryable_failure(
        self,
        connection: sqlite3.Connection,
        existing: sqlite3.Row,
    ) -> sqlite3.Row:
        """Requeue a failed command for an identical idempotent resubmission.

        Only a failed command whose persisted result explicitly recorded
        `retryable` may run again, and only while no attempt crossed the
        provider boundary. The guarded status transition is the whole
        admission test, so two concurrent resubmissions requeue the command
        once and never fan out into duplicate work.
        """

        result = _load_object(str(existing["result_json"]))
        if result.get("retryable") is not True:
            return existing
        command_id = str(existing["command_id"])
        crossed = connection.execute(
            """
            SELECT COUNT(*) AS count FROM command_dispatches
            WHERE command_id = ? AND crossed_boundary = 1
            """,
            (command_id,),
        ).fetchone()
        crossed_count = 0
        if crossed is not None:
            crossed_count = int(crossed["count"])
        if crossed_count > 0:
            return existing
        now = utc_now()
        cursor = connection.execute(
            """
            UPDATE commands SET status = ?, result_json = '{}',
                updated_at = ? WHERE command_id = ? AND status = ?
            """,
            (
                CommandStatus.QUEUED,
                now,
                command_id,
                CommandStatus.FAILED,
            ),
        )
        if cursor.rowcount != 1:
            return existing
        connection.execute(
            """
            UPDATE command_envelopes SET state = 'reserved',
                guard_reason = '', updated_at = ?
            WHERE command_id = ?
            """,
            (now, command_id),
        )
        return connection.execute(
            "SELECT * FROM commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()

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
            if not command_types:
                reconciliation = connection.execute(
                    """
                    SELECT 1 FROM reconciliations
                    WHERE session_id = ? AND status != ?
                    LIMIT 1
                    """,
                    (session_id, ReconciliationStatus.RESOLVED),
                ).fetchone()
                if reconciliation is not None:
                    return None
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

    def requeue_command(self, command_id: str) -> None:
        """Park a claimed command back to queued for a policy deferral."""

        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE commands SET status = ?, updated_at = ?
                WHERE command_id = ? AND status = ?
                """,
                (
                    CommandStatus.QUEUED,
                    utc_now(),
                    command_id,
                    CommandStatus.DISPATCHING,
                ),
            )
            if cursor.rowcount != 1:
                raise ConflictError("command is not dispatching")

    def command_failed_before_provider_boundary(self, command_id: str) -> bool:
        with self._lock:
            command = self._connection.execute(
                "SELECT status FROM commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            crossed = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM command_dispatches
                WHERE command_id = ? AND crossed_boundary = 1
                """,
                (command_id,),
            ).fetchone()
        if command is None:
            return False
        if str(command["status"]) not in {
            CommandStatus.FAILED,
            CommandStatus.CANCELLED,
        }:
            return False
        if crossed is None:
            return True
        return int(crossed["count"]) == 0

    def has_event_type(self, session_id: str, event_type: str) -> bool:
        """Report whether a session ever recorded one kind of event.

        Answering this by reading the whole history costs the caller
        every event the session holds. The supervision tick asks it for
        each session needing input, which on this host meant reading
        83,828 events every tick to look for one type.
        """
        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1 FROM events
                WHERE session_id = ? AND event_type = ?
                LIMIT 1
                """,
                (session_id, event_type),
            ).fetchone()
        return row is not None

    def sessions_with_open_work(self) -> set[str]:
        """Sessions whose work was cut off and needs a worker again.

        Work is a turn still marked running or a command that has not
        reached a terminal status. A session outside this set has
        nothing for a worker to resume, so reviving one for it produces
        an idle process rather than progress.
        """
        with self._lock:
            turns = self._connection.execute(
                """
                SELECT DISTINCT session_id FROM turns
                WHERE status = 'running'
                """
            ).fetchall()
            commands = self._connection.execute(
                """
                SELECT DISTINCT session_id FROM commands
                WHERE status IN (?, ?, ?)
                """,
                (
                    CommandStatus.QUEUED,
                    CommandStatus.AWAITING_XHIGH_AUTHORIZATION,
                    CommandStatus.DISPATCHING,
                ),
            ).fetchall()
        result = {str(row["session_id"]) for row in turns}
        result.update(str(row["session_id"]) for row in commands)
        return result

    def active_command_summaries(self) -> list[dict[str, Any]]:
        """Enumerate prompt-free durable restart blockers."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT c.command_id, c.session_id, c.status,
                    COALESCE(e.profile, '') AS profile,
                    COALESCE(e.provider, '') AS provider,
                    COALESCE(e.state, '') AS envelope_state,
                    COALESCE(e.recovery_stage, 0) AS recovery_stage
                FROM commands AS c
                LEFT JOIN command_envelopes AS e
                    ON e.command_id = c.command_id
                WHERE c.status = ?
                ORDER BY c.created_at, c.command_id
                """,
                (CommandStatus.DISPATCHING,),
            ).fetchall()
        return [
            {
                "command_id": str(row["command_id"]),
                "session_id": str(row["session_id"]),
                "status": str(row["status"]),
                "profile": str(row["profile"]),
                "provider": str(row["provider"]),
                "envelope_state": str(row["envelope_state"]),
                "recovery_stage": int(row["recovery_stage"]),
            }
            for row in rows
        ]

    def resolve_command(
        self,
        command_id: str,
        status: str,
        result: dict[str, Any],
    ) -> CommandReceipt:
        with self.transaction() as connection:
            now = utc_now()
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

    def stop_session(
        self,
        session_id: str,
        *,
        stop_command_id: str = "",
        active_command_id: str = "",
    ) -> dict[str, Any]:
        """Stop a session and release its unaccepted provider commands.

        A stopped session keeps no worker, so any command still queued
        or claimed would hold its safety envelope in an active state
        forever and consume provider concurrency. This terminalizes
        those commands as cancelled and releases their envelopes in the
        same transaction that marks the session stopped. Cancellation
        records that the command was not accepted; checkpoints, events,
        and workspace material remain untouched even when a prior
        attempt crossed the provider boundary.

        A control that raced this transition is cancelled with every
        other unaccepted command. The worker claims controls in order,
        so nothing older than the stop is still queued here, and a
        newer control never reached an accepting session. Leaving one
        queued would strand it once this worker exits.
        """
        now = utc_now()
        released: list[dict[str, Any]] = []
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT command_id, command_type, status FROM commands
                WHERE session_id = ? AND status IN (?, ?, ?)
                ORDER BY created_at, command_id
                """,
                (
                    session_id,
                    CommandStatus.QUEUED,
                    CommandStatus.AWAITING_XHIGH_AUTHORIZATION,
                    CommandStatus.DISPATCHING,
                ),
            ).fetchall()
            sequence_row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS sequence
                FROM events WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            sequence = int(sequence_row["sequence"])
            for row in rows:
                command_id = str(row["command_id"])
                if command_id in {stop_command_id, active_command_id}:
                    continue
                prior_status = str(row["status"])
                result = {
                    "code": "E_SESSION_STOPPED",
                    "message": ("session stopped before the command was accepted"),
                    "accepted": False,
                    "prior_status": prior_status,
                }
                cursor = connection.execute(
                    """
                    UPDATE commands SET status = ?, result_json = ?,
                        updated_at = ? WHERE command_id = ? AND status = ?
                    """,
                    (
                        CommandStatus.CANCELLED,
                        _dump(result),
                        now,
                        command_id,
                        prior_status,
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                connection.execute(
                    """
                    UPDATE command_envelopes SET state = 'released',
                        guard_reason = 'session-stopped', updated_at = ?
                    WHERE command_id = ?
                    """,
                    (now, command_id),
                )
                entry = {
                    "command_id": command_id,
                    "command_type": str(row["command_type"]),
                    "prior_status": prior_status,
                    "reason": "session-stopped",
                    "accepted": False,
                }
                sequence += 1
                connection.execute(
                    """
                    INSERT INTO events(
                        session_id, sequence, event_id, event_type, role,
                        text, status, metadata_json, blob_digest, turn_id,
                        created_at
                    ) VALUES (?, ?, ?, ?, '', '', ?, ?, '', '', ?)
                    """,
                    (
                        session_id,
                        sequence,
                        new_uuid(),
                        "command.released",
                        CommandStatus.CANCELLED,
                        _dump(entry),
                        now,
                    ),
                )
                released.append(entry)
            cursor = connection.execute(
                """
                UPDATE sessions SET lifecycle = ?, attention = ?,
                    updated_at = ? WHERE session_id = ?
                """,
                (Lifecycle.STOPPED, Attention.IDLE, now, session_id),
            )
            if cursor.rowcount != 1:
                raise NotFoundError("session")
        return {
            "session": self.get_session(session_id).as_dict(),
            "released_commands": released,
        }

    def xhigh_authorization_or_park(
        self,
        command_id: str,
    ) -> dict[str, Any] | None:
        now = utc_now()
        with self.transaction() as connection:
            authorization = connection.execute(
                """
                SELECT * FROM xhigh_authorization_receipts
                WHERE command_id = ? AND consumed_at = '' AND expires_at > ?
                """,
                (command_id, now),
            ).fetchone()
            if authorization is not None:
                return dict(authorization)
            cursor = connection.execute(
                """
                UPDATE commands SET status = ?, updated_at = ?
                WHERE command_id = ? AND status = ?
                """,
                (
                    CommandStatus.AWAITING_XHIGH_AUTHORIZATION,
                    now,
                    command_id,
                    CommandStatus.DISPATCHING,
                ),
            )
            if cursor.rowcount != 1:
                raise ConflictError("xhigh command is not awaiting authorization")
        return None

    def recover_dispatching(self, session_id: str) -> int:
        recovery = self.recover_interrupted_commands(
            session_id,
            "",
            "",
        )
        return len(recovery.reconciliations)

    def start_turn(
        self,
        session_id: str,
        attempt_id: str,
        *,
        replay_of: str = "",
        turn_ref: dict[str, str] | None = None,
    ) -> str:
        normalized_ref = normalize_turn_ref(turn_ref)
        turn_id = new_uuid()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    attempt_id,
                    "running",
                    replay_of,
                    utc_now(),
                    "",
                    normalized_ref.get("step_id", ""),
                    normalized_ref.get("agent_role", ""),
                ),
            )
        return turn_id

    def turn_ref(self, turn_id: str) -> dict[str, str]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT turn_step_id, turn_agent_role FROM turns
                WHERE turn_id = ?
                """,
                (turn_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("turn")
        if not row["turn_step_id"]:
            return {}
        return {
            "step_id": str(row["turn_step_id"]),
            "agent_role": str(row["turn_agent_role"]),
        }

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

    def countable_turn_count(self, session_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM turns
                WHERE session_id = ? AND status != 'no-progress'
                """,
                (session_id,),
            ).fetchone()
        return int(row["count"])

    def active_turn_seconds(
        self,
        session_id: str,
        now: datetime.datetime,
    ) -> float:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT started_at, completed_at FROM turns
                WHERE session_id = ? AND status != 'no-progress'
                """,
                (session_id,),
            ).fetchall()
        total = 0.0
        for row in rows:
            started = _turn_timestamp(str(row["started_at"]))
            completed_text = str(row["completed_at"])
            if completed_text:
                completed = _turn_timestamp(completed_text)
            else:
                completed = now
            total += max(0.0, (completed - started).total_seconds())
        return total

    def presentation_turn_rows(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Return canonical records needed for rebuildable turn views."""

        self.get_session(session_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT
                    turns.turn_id,
                    turns.attempt_id,
                    turns.status AS turn_status,
                    turns.replay_of,
                    turns.started_at,
                    turns.completed_at,
                    turns.turn_step_id,
                    turns.turn_agent_role,
                    provider_attempts.provider,
                    provider_attempts.model,
                    provider_attempts.effort,
                    provider_attempts.status AS attempt_status,
                    provider_attempts.ended_at,
                    command_dispatches.command_id,
                    commands.status AS command_status,
                    commands.payload_json,
                    commands.result_json
                FROM turns
                LEFT JOIN provider_attempts
                    ON provider_attempts.attempt_id = turns.attempt_id
                LEFT JOIN command_dispatches
                    ON command_dispatches.turn_id = turns.turn_id
                LEFT JOIN commands
                    ON commands.command_id = command_dispatches.command_id
                WHERE turns.session_id = ?
                ORDER BY turns.started_at, turns.turn_id
                """,
                (session_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = _load_object(row["payload_json"] or "{}")
            command_result = _load_object(row["result_json"] or "{}")
            turn_ref: dict[str, str] = {}
            if row["turn_step_id"]:
                turn_ref = {
                    "step_id": str(row["turn_step_id"]),
                    "agent_role": str(row["turn_agent_role"]),
                }
            result.append(
                {
                    "turn_id": str(row["turn_id"]),
                    "attempt_id": str(row["attempt_id"]),
                    "turn_status": str(row["turn_status"]),
                    "replay_of": str(row["replay_of"]),
                    "started_at": str(row["started_at"]),
                    "completed_at": str(row["completed_at"]),
                    "turn_ref": turn_ref,
                    "provider": str(row["provider"] or ""),
                    "model": str(row["model"] or ""),
                    "effort": str(row["effort"] or ""),
                    "attempt_status": str(row["attempt_status"] or ""),
                    "ended_at": str(row["ended_at"] or ""),
                    "command_id": str(row["command_id"] or ""),
                    "command_status": str(row["command_status"] or ""),
                    "request_text": str(payload.get("text", "")),
                    "command_result": command_result,
                }
            )
        return result

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
        return [_load_object(str(row["result_json"])) for row in rows]

    def failed_command_results(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT command_id, command_type, result_json,
                    created_at, updated_at FROM commands
                WHERE session_id = ? AND status = ?
                ORDER BY created_at, command_id
                """,
                (session_id, CommandStatus.FAILED),
            ).fetchall()
        return [
            {
                "command_id": str(row["command_id"]),
                "command_type": str(row["command_type"]),
                "result": _load_object(str(row["result_json"])),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
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
        if status in {
            "cancelled",
            "complete",
            "failed",
            "interrupted",
            "exhausted",
        }:
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
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                    _dump(list(goal.permitted_providers)),
                    _dump(list(goal.permitted_efforts)),
                    goal.max_concurrency,
                    goal.completion_policy,
                    goal.incident_policy,
                ),
            )
            self._insert_milestones(connection, goal)
            connection.execute(
                """
                UPDATE sessions SET goal_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (goal.goal_id, goal.updated_at, goal.session_id),
            )
        return goal

    def _insert_milestones(
        self,
        connection: sqlite3.Connection,
        goal: Goal,
    ) -> None:
        for milestone in goal.milestones:
            connection.execute(
                "INSERT INTO goal_milestones VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    goal.goal_id,
                    milestone.milestone_id,
                    milestone.title,
                    milestone.status,
                    _dump(list(milestone.dependencies)),
                    _dump(list(milestone.predicates)),
                    milestone.position,
                ),
            )

    def _insert_authorization_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        operation: str,
        operation_id: str,
        authorization: dict[str, Any],
        created_at: str,
        authorization_digest: str = "",
    ) -> str:
        digest = authorization_digest
        if not digest:
            digest = normalized_digest(authorization)
        connection.execute(
            """
            INSERT INTO authorization_receipts VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                digest,
                session_id,
                operation,
                operation_id,
                str(authorization.get("schema", "")),
                str(authorization.get("receipt_sha256", "")),
                _dump(authorization),
                created_at,
            ),
        )
        return digest

    def _retain_dispatch_transition_policy(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        authorization: dict[str, Any],
        created_at: str,
    ) -> dict[str, Any]:
        policy = authorization.get("policy")
        if not isinstance(policy, dict) or not policy:
            raise ConflictError("dispatch transition policy is missing")
        policy_sha256 = normalized_digest(policy)
        goal_id = str(authorization.get("goal_id", ""))
        epoch_id = str(authorization.get("epoch_id", ""))
        if authorization.get("policy_sha256") != policy_sha256:
            raise ConflictError("dispatch transition policy digest changed")
        existing = connection.execute(
            """
            SELECT * FROM dispatch_transition_policies
            WHERE session_id = ? AND goal_id = ? AND epoch_id = ?
            """,
            (session_id, goal_id, epoch_id),
        ).fetchone()
        canonical_payload = _dump(policy)
        if existing is None:
            connection.execute(
                """
                INSERT INTO dispatch_transition_policies VALUES (
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    policy_sha256,
                    session_id,
                    goal_id,
                    epoch_id,
                    str(policy.get("schema", "")),
                    canonical_payload,
                    created_at,
                ),
            )
        elif (
            str(existing["policy_sha256"]) != policy_sha256
            or str(existing["payload_json"]) != canonical_payload
            or str(existing["schema"]) != str(policy.get("schema", ""))
        ):
            raise ConflictError("dispatch transition epoch policy changed")
        compact = dict(authorization)
        compact.pop("policy", None)
        compact["policy_ref"] = {
            "policy_sha256": policy_sha256,
            "session_id": session_id,
            "goal_id": goal_id,
            "epoch_id": epoch_id,
        }
        return compact

    def _insert_mutation_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        idempotency_key: str,
        operation: str,
        request_digest: str,
        response: dict[str, Any],
        status_code: int,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO mutation_receipts
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                idempotency_key,
                operation,
                request_digest,
                _dump(response),
                status_code,
                created_at,
            ),
        )

    def promote_goal(
        self,
        previous_goal: Goal,
        next_goal: Goal,
        *,
        stage: str,
        authorization_digest: str,
        authorization: dict[str, Any],
        request_digest: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM goal_promotions
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["session_id"]) != previous_goal.session_id
                    or str(existing["request_digest"]) != request_digest
                ):
                    raise ConflictError("goal promotion idempotency key was reused")
                return dict(existing)
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (previous_goal.session_id,),
            ).fetchone()
            if session is None:
                raise NotFoundError("session")
            if str(session["goal_id"]) != previous_goal.goal_id:
                raise ConflictError("goal promotion source is not current")
            if str(session["lifecycle"]) != Lifecycle.COMPLETED:
                raise ConflictError("goal promotion requires a completed session")
            current_goal = connection.execute(
                "SELECT status FROM goals WHERE goal_id = ?",
                (previous_goal.goal_id,),
            ).fetchone()
            if current_goal is None:
                raise NotFoundError("goal")
            if str(current_goal["status"]) != GoalStatus.COMPLETE:
                raise ConflictError("goal promotion requires a completed goal")
            active = connection.execute(
                """
                SELECT COUNT(*) AS count FROM commands
                WHERE session_id = ? AND status IN (
                    'queued', 'awaiting-xhigh-authorization', 'dispatching'
                )
                """,
                (previous_goal.session_id,),
            ).fetchone()
            if active is not None and int(active["count"]) > 0:
                raise ConflictError("goal promotion requires command quiescence")
            pending_approvals = connection.execute(
                """
                SELECT COUNT(*) AS count FROM approvals
                WHERE session_id = ? AND status = 'pending'
                """,
                (previous_goal.session_id,),
            ).fetchone()
            if pending_approvals is not None and int(pending_approvals["count"]) > 0:
                raise ConflictError("goal promotion has a pending approval")
            pending_reconciliations = connection.execute(
                """
                SELECT COUNT(*) AS count FROM reconciliations
                WHERE session_id = ? AND status != 'resolved'
                """,
                (previous_goal.session_id,),
            ).fetchone()
            if (
                pending_reconciliations is not None
                and int(pending_reconciliations["count"]) > 0
            ):
                raise ConflictError("goal promotion has a reconciliation barrier")
            promotion_id = new_uuid()
            created_at = utc_now()
            connection.execute(
                """
                INSERT INTO goals VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    next_goal.goal_id,
                    next_goal.session_id,
                    next_goal.kind,
                    next_goal.objective,
                    next_goal.status,
                    _dump(list(next_goal.constraints)),
                    _dump(list(next_goal.predicates)),
                    _dump(next_goal.budgets),
                    next_goal.created_at,
                    next_goal.updated_at,
                    _dump(list(next_goal.permitted_providers)),
                    _dump(list(next_goal.permitted_efforts)),
                    next_goal.max_concurrency,
                    next_goal.completion_policy,
                    next_goal.incident_policy,
                ),
            )
            recorded_authorization_digest = self._insert_authorization_receipt(
                connection,
                session_id=previous_goal.session_id,
                operation="goal-promotion",
                operation_id=promotion_id,
                authorization=authorization,
                created_at=created_at,
            )
            if recorded_authorization_digest != authorization_digest:
                raise ValueError("goal promotion authorization digest changed")
            self._insert_milestones(connection, next_goal)
            previous_digest = goal_contract_digest(previous_goal)
            next_digest = goal_contract_digest(next_goal)
            connection.execute(
                """
                INSERT INTO goal_promotions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    promotion_id,
                    previous_goal.session_id,
                    previous_goal.goal_id,
                    next_goal.goal_id,
                    stage,
                    authorization_digest,
                    request_digest,
                    idempotency_key,
                    previous_digest,
                    next_digest,
                    created_at,
                ),
            )
            prior_evidence = connection.execute(
                """
                SELECT * FROM evidence WHERE goal_id = ?
                ORDER BY created_at, evidence_id
                """,
                (previous_goal.goal_id,),
            ).fetchall()
            for row in prior_evidence:
                copied_evidence_id = new_uuid()
                value = _load_object(str(row["value_json"]))
                evidence_contract = {
                    "source_evidence_id": str(row["evidence_id"]),
                    "evidence_type": str(row["evidence_type"]),
                    "subject": str(row["subject"]),
                    "outcome": str(row["outcome"]),
                    "value": value,
                }
                connection.execute(
                    "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        copied_evidence_id,
                        next_goal.goal_id,
                        str(row["evidence_type"]),
                        str(row["subject"]),
                        str(row["outcome"]),
                        _dump(value),
                        created_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO goal_promotion_evidence VALUES (
                        ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        promotion_id,
                        str(row["evidence_id"]),
                        copied_evidence_id,
                        normalized_digest(evidence_contract),
                        created_at,
                    ),
                )
            connection.execute(
                """
                UPDATE sessions SET goal_id = ?, lifecycle = ?, attention = ?,
                    updated_at = ? WHERE session_id = ?
                """,
                (
                    next_goal.goal_id,
                    Lifecycle.STARTING,
                    Attention.IDLE,
                    created_at,
                    previous_goal.session_id,
                ),
            )
            sequence_row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS sequence
                FROM events WHERE session_id = ?
                """,
                (previous_goal.session_id,),
            ).fetchone()
            sequence = 1
            if sequence_row is not None:
                sequence = int(sequence_row["sequence"]) + 1
            connection.execute(
                """
                INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    previous_goal.session_id,
                    sequence,
                    promotion_id,
                    "goal.promoted",
                    "",
                    "",
                    "complete",
                    _dump(
                        {
                            "promotion_id": promotion_id,
                            "previous_goal_id": previous_goal.goal_id,
                            "next_goal_id": next_goal.goal_id,
                            "stage": stage,
                            "authorization_digest": authorization_digest,
                            "previous_goal_digest": previous_digest,
                            "next_goal_digest": next_digest,
                        }
                    ),
                    "",
                    "",
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM goal_promotions WHERE promotion_id = ?",
                (promotion_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("goal promotion receipt was not recorded")
            return dict(row)

    def goal_promotions(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM goal_promotions WHERE session_id = ?
                ORDER BY created_at, promotion_id
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def goal_promotion_by_key(
        self,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM goal_promotions
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def adopt_session_contract(
        self,
        session_id: str,
        next_goal: Goal,
        *,
        external_ref: dict[str, str],
        creation_input: dict[str, Any],
        authorization_digest: str,
        authorization: dict[str, Any],
        request_digest: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        request_creation_digest = creation_digest(creation_input)
        normalized_ref = normalize_external_ref(external_ref)
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM goal_contract_adoptions
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["session_id"]) != session_id
                    or str(existing["request_digest"]) != request_digest
                ):
                    raise ConflictError("contract adoption idempotency key was reused")
                return dict(existing)
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise NotFoundError("session")
            active = connection.execute(
                """
                SELECT COUNT(*) AS count FROM commands
                WHERE session_id = ? AND status IN (
                    'queued', 'awaiting-xhigh-authorization', 'dispatching'
                )
                """,
                (session_id,),
            ).fetchone()
            active_leases = connection.execute(
                """
                SELECT COUNT(*) AS count FROM process_leases
                WHERE session_id = ?
                AND state IN ('reserved', 'active', 'recovery-blocked')
                """,
                (session_id,),
            ).fetchone()
            if active is not None and int(active["count"]) > 0:
                raise ConflictError("contract adoption requires quiescence")
            if active_leases is not None and int(active_leases["count"]) > 0:
                raise ConflictError("contract adoption has an active process lease")
            if str(session["attention"]) == Attention.WORKING:
                raise ConflictError("contract adoption requires an idle session")
            conflict = connection.execute(
                """
                SELECT session_id FROM sessions
                WHERE external_orchestrator = ? AND external_job_id = ?
                """,
                (
                    normalized_ref["orchestrator"],
                    normalized_ref["job_id"],
                ),
            ).fetchone()
            if conflict is not None and str(conflict["session_id"]) != session_id:
                raise ConflictError("external reference belongs to another session")
            previous_goal_id = str(session["goal_id"])
            previous_goal: Goal | None = None
            previous_digest = ""
            if previous_goal_id:
                previous_row = connection.execute(
                    "SELECT * FROM goals WHERE goal_id = ?",
                    (previous_goal_id,),
                ).fetchone()
                if previous_row is not None:
                    milestone_rows = connection.execute(
                        """
                        SELECT * FROM goal_milestones WHERE goal_id = ?
                        ORDER BY position, milestone_id
                        """,
                        (previous_goal_id,),
                    ).fetchall()
                    previous_goal = _goal(previous_row, milestone_rows)
                    previous_digest = goal_contract_digest(previous_goal)
            adoption_id = new_uuid()
            created_at = utc_now()
            connection.execute(
                """
                INSERT INTO goals VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    next_goal.goal_id,
                    session_id,
                    next_goal.kind,
                    next_goal.objective,
                    next_goal.status,
                    _dump(list(next_goal.constraints)),
                    _dump(list(next_goal.predicates)),
                    _dump(next_goal.budgets),
                    next_goal.created_at,
                    next_goal.updated_at,
                    _dump(list(next_goal.permitted_providers)),
                    _dump(list(next_goal.permitted_efforts)),
                    next_goal.max_concurrency,
                    next_goal.completion_policy,
                    next_goal.incident_policy,
                ),
            )
            recorded_authorization_digest = self._insert_authorization_receipt(
                connection,
                session_id=session_id,
                operation="contract-adoption",
                operation_id=adoption_id,
                authorization=authorization,
                created_at=created_at,
            )
            if recorded_authorization_digest != authorization_digest:
                raise ValueError("contract adoption authorization digest changed")
            self._insert_milestones(connection, next_goal)
            if (
                previous_goal is not None
                and previous_goal.status != GoalStatus.COMPLETE
            ):
                connection.execute(
                    "UPDATE goals SET status = ?, updated_at = ? WHERE goal_id = ?",
                    (GoalStatus.CANCELLED, utc_now(), previous_goal.goal_id),
                )
            next_digest = goal_contract_digest(next_goal)
            connection.execute(
                """
                INSERT INTO goal_contract_adoptions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    adoption_id,
                    session_id,
                    previous_goal_id,
                    next_goal.goal_id,
                    normalized_ref["orchestrator"],
                    normalized_ref["job_id"],
                    authorization_digest,
                    request_digest,
                    request_creation_digest,
                    previous_digest,
                    next_digest,
                    idempotency_key,
                    created_at,
                ),
            )
            routing = creation_input.get("routing", {})
            if not isinstance(routing, dict):
                routing = {}
            connection.execute(
                """
                UPDATE sessions SET goal_id = ?, external_orchestrator = ?,
                    external_job_id = ?, lifecycle = ?, attention = ?,
                    name = ?, permission_mode = ?, model = ?, effort = ?,
                    updated_at = ? WHERE session_id = ?
                """,
                (
                    next_goal.goal_id,
                    normalized_ref["orchestrator"],
                    normalized_ref["job_id"],
                    Lifecycle.STARTING,
                    Attention.IDLE,
                    str(creation_input.get("name", session["name"])),
                    str(
                        creation_input.get(
                            "permission_mode",
                            session["permission_mode"],
                        )
                    ),
                    str(routing.get("model", session["model"])),
                    str(routing.get("effort", session["effort"])),
                    created_at,
                    session_id,
                ),
            )
            profile = str(creation_input.get("execution_profile", "unattended"))
            connection.execute(
                """
                INSERT INTO session_safety(
                    session_id, profile, xhigh_authorizations,
                    extensions_json, created_at, updated_at
                ) VALUES (?, ?, 0, '{}', ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    profile = excluded.profile,
                    extensions_json = '{}',
                    xhigh_authorizations = 0,
                    updated_at = excluded.updated_at
                """,
                (session_id, profile, created_at, created_at),
            )
            sequence_row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS sequence
                FROM events WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            sequence = 1
            if sequence_row is not None:
                sequence = int(sequence_row["sequence"]) + 1
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    sequence,
                    adoption_id,
                    "goal.contract_adopted",
                    "",
                    "",
                    "complete",
                    _dump(
                        {
                            "adoption_id": adoption_id,
                            "previous_goal_id": previous_goal_id,
                            "next_goal_id": next_goal.goal_id,
                            "previous_goal_digest": previous_digest,
                            "next_goal_digest": next_digest,
                            "authorization_digest": authorization_digest,
                        }
                    ),
                    "",
                    "",
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM goal_contract_adoptions WHERE adoption_id = ?",
                (adoption_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("contract adoption receipt was not recorded")
            return dict(row)

    def goal_contract_adoption_by_key(
        self,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM goal_contract_adoptions
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_goal(self, goal_id: str) -> Goal:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            milestone_rows = self._connection.execute(
                """
                SELECT * FROM goal_milestones WHERE goal_id = ?
                ORDER BY position, milestone_id
                """,
                (goal_id,),
            ).fetchall()
        if row is None:
            raise NotFoundError("goal")
        return _goal(row, milestone_rows)

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

    def update_milestone_statuses(
        self,
        goal_id: str,
        milestones: tuple[Milestone, ...],
    ) -> Goal:
        now = utc_now()
        with self.transaction() as connection:
            for milestone in milestones:
                cursor = connection.execute(
                    """
                    UPDATE goal_milestones SET status = ?
                    WHERE goal_id = ? AND milestone_id = ?
                    """,
                    (milestone.status, goal_id, milestone.milestone_id),
                )
                if cursor.rowcount != 1:
                    raise NotFoundError("goal milestone")
            connection.execute(
                "UPDATE goals SET updated_at = ? WHERE goal_id = ?",
                (now, goal_id),
            )
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

    def add_evidence_once(
        self,
        session_id: str,
        evidence: Evidence,
        *,
        idempotency_key: str,
        request_digest: str,
    ) -> Evidence:
        operation = "evidence-create:" + session_id
        with self.transaction() as connection:
            receipt = connection.execute(
                "SELECT * FROM mutation_receipts WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if receipt is not None:
                if (
                    str(receipt["operation"]) != operation
                    or str(receipt["request_digest"]) != request_digest
                ):
                    raise ConflictError(
                        "idempotency key was already used for another mutation"
                    )
                response = _load_object(str(receipt["response_json"]))
                value = _object_or_empty(response.get("evidence"))
                row = connection.execute(
                    "SELECT * FROM evidence WHERE evidence_id = ?",
                    (str(value.get("evidence_id", "")),),
                ).fetchone()
                if row is None:
                    raise RuntimeError("evidence receipt has no evidence row")
                return _evidence(row)
            session = connection.execute(
                "SELECT goal_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise NotFoundError("session")
            if str(session["goal_id"]) != evidence.goal_id:
                raise ConflictError("evidence goal is no longer current")
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
            sequence_row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS sequence
                FROM events WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            sequence = 1
            if sequence_row is not None:
                sequence = int(sequence_row["sequence"]) + 1
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    sequence,
                    evidence.evidence_id,
                    "goal.evidence",
                    "",
                    "",
                    "complete",
                    _dump(evidence.as_dict()),
                    "",
                    "",
                    evidence.created_at,
                ),
            )
            response = {"evidence": evidence.as_dict()}
            connection.execute(
                "INSERT INTO mutation_receipts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    idempotency_key,
                    operation,
                    request_digest,
                    _dump(response),
                    201,
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

    def repetition_generation(self, session_id: str) -> dict[str, str]:
        with self._lock:
            checkpoint = self._connection.execute(
                """
                SELECT * FROM checkpoints WHERE session_id = ?
                ORDER BY sequence DESC, created_at DESC, checkpoint_id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            invalidation = self._connection.execute(
                """
                SELECT * FROM dispatch_invalidations WHERE session_id = ?
                ORDER BY created_at DESC, invalidation_id DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        checkpoint_id = ""
        material: dict[str, Any] = {}
        if checkpoint is not None:
            checkpoint_id = str(checkpoint["checkpoint_id"])
            material = {
                "base_commit": str(checkpoint["base_commit"]),
                "patch_digest": str(checkpoint["patch_digest"]),
                "untracked_digest": str(checkpoint["untracked_digest"]),
            }
        invalidation_id = ""
        if invalidation is not None:
            invalidation_id = str(invalidation["invalidation_id"])
        return {
            "generation_digest": normalized_digest(
                {
                    "material": material,
                    "invalidation_id": invalidation_id,
                }
            ),
            "checkpoint_id": checkpoint_id,
            "invalidation_id": invalidation_id,
        }

    def dispatch_transition_anchor(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise NotFoundError("session")
            base = {
                "schema": "p13i/agent-harness/dispatch-transition-anchor/v1",
                "session_id": session_id,
                "external_ref": {
                    "orchestrator": str(session["external_orchestrator"]),
                    "job_id": str(session["external_job_id"]),
                },
                "goal_id": str(session["goal_id"]),
                "epoch_id": "",
                "eligible": False,
                "reason": "",
            }
            if str(session["attention"]) == Attention.WORKING:
                base["reason"] = "session-is-working"
                return base
            if _dispatch_transition_active_commands(self._connection, session_id) > 0:
                base["reason"] = "active-command"
                return base
            goal = self._connection.execute(
                "SELECT constraints_json FROM goals WHERE goal_id = ?",
                (str(session["goal_id"]),),
            ).fetchone()
            if goal is not None:
                constraints = json.loads(str(goal["constraints_json"]))
                for constraint in constraints:
                    prefix = "dispatch-generation-transition-epoch:"
                    if isinstance(constraint, str) and constraint.startswith(prefix):
                        base["epoch_id"] = constraint.removeprefix(prefix)
                        break
            if not base["goal_id"] or not base["epoch_id"]:
                base["reason"] = "missing-goal-epoch"
                return base
            if (
                not base["external_ref"]["orchestrator"]
                or not base["external_ref"]["job_id"]
            ):
                base["reason"] = "missing-external-reference"
                return base
            prior = _dispatch_transition_predecessor(self._connection, session_id)
            if prior is None:
                base["reason"] = "missing-prior-command"
                return base
            latest_checkpoint = self._connection.execute(
                """
                SELECT * FROM checkpoints WHERE session_id = ?
                ORDER BY sequence DESC, created_at DESC, checkpoint_id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if latest_checkpoint is None:
                base["reason"] = "missing-certified-checkpoint"
                return base
            live_material_digest, unused_summary = inspect_workspace(
                Path(str(session["worktree"]))
            )
            del unused_summary
            try:
                anchor = _dispatch_transition_anchor(
                    self._connection,
                    prior,
                    str(latest_checkpoint["checkpoint_id"]),
                    live_material_digest,
                )
            except ConflictError as error:
                base["reason"] = error.detail.message
                return base
            generation = self.repetition_generation(session_id)
            base.update(
                {
                    **anchor,
                    "eligible": True,
                    "reason": "",
                    "prior_command_id": str(prior["command_id"]),
                    "prior_command_status": str(prior["status"]),
                    "prior_checkpoint_id": str(latest_checkpoint["checkpoint_id"]),
                    "prior_material_digest": live_material_digest,
                    "prior_generation_digest": generation["generation_digest"],
                }
            )
            return base

    def dispatch_transition_policy(
        self,
        session_id: str,
        goal_id: str,
        epoch_id: str,
        policy_sha256: str,
    ) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload_json FROM dispatch_transition_policies
                WHERE session_id = ? AND goal_id = ? AND epoch_id = ?
                  AND policy_sha256 = ?
                """,
                (session_id, goal_id, epoch_id, policy_sha256),
            ).fetchone()
        if row is None:
            raise ConflictError("dispatch transition policy reference is unknown")
        policy = _load_object(str(row["payload_json"]))
        if normalized_digest(policy) != policy_sha256:
            raise ConflictError("dispatch transition retained policy is corrupt")
        return policy

    def create_dispatch_invalidation(
        self,
        session_id: str,
        *,
        reason: str,
        authorization: dict[str, Any],
        request_digest: str,
        idempotency_key: str,
        prior_command_id: str = "",
        next_turn_ref: dict[str, str] | None = None,
        authorization_digest: str = "",
    ) -> dict[str, Any]:
        if not authorization_digest:
            authorization_digest = normalized_digest(authorization)
        normalized_next_turn_ref = normalize_turn_ref(next_turn_ref)
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise NotFoundError("session")
            existing = connection.execute(
                """
                SELECT * FROM dispatch_invalidations
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["session_id"]) != session_id
                    or str(existing["request_digest"]) != request_digest
                ):
                    raise ConflictError(
                        "dispatch invalidation idempotency key was reused"
                    )
                existing_authorization = connection.execute(
                    """
                    SELECT schema, payload_json FROM authorization_receipts
                    WHERE operation = 'dispatch-invalidation'
                      AND operation_id = ?
                    """,
                    (str(existing["invalidation_id"]),),
                ).fetchone()
                transition_schema = (
                    "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
                )
                if (
                    existing_authorization is not None
                    and str(existing_authorization["schema"]) == transition_schema
                    and not _dispatch_transition_epoch_is_active(
                        connection,
                        session_id,
                        _load_object(str(existing_authorization["payload_json"])),
                    )
                ):
                    raise ConflictError("dispatch transition epoch is no longer active")
                result = dict(existing)
                result["prior_command_id"] = prior_command_id
                result["next_turn_ref"] = normalized_next_turn_ref
                if prior_command_id:
                    for name in (
                        "goal_id",
                        "prior_command_type",
                        "prior_anchor_kind",
                        "prior_reconciliation_id",
                        "prior_reconciliation_resolution",
                        "prior_checkpoint_id",
                        "prior_generation_digest",
                        "prior_material_digest",
                        "transition_sequence",
                        "epoch_id",
                        "policy_sha256",
                        "next_command_digest",
                    ):
                        result[name] = authorization.get(name)
                return result
            if str(session["attention"]) == Attention.WORKING:
                raise ConflictError("dispatch invalidation requires quiescence")
            if _dispatch_transition_active_commands(connection, session_id) > 0:
                raise ConflictError("dispatch invalidation has an active command")
            transition_schema = (
                "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
            )
            authorization_schema = str(authorization.get("schema", ""))
            if authorization_schema == transition_schema and not prior_command_id:
                raise ConflictError("dispatch transition prior command is required")
            if authorization_schema != transition_schema:
                safety = connection.execute(
                    "SELECT profile FROM session_safety WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if safety is not None and str(safety["profile"]) != "interactive":
                    raise ConflictError(
                        "managed dispatch invalidation requires a transition"
                    )
            if prior_command_id:
                if authorization_schema != transition_schema:
                    raise ConflictError("dispatch transition schema is invalid")
                if authorization.get("prior_command_id") != prior_command_id:
                    raise ConflictError("dispatch transition prior command changed")
                if normalize_turn_ref(authorization.get("next_turn_ref")) != (
                    normalized_next_turn_ref
                ):
                    raise ConflictError("dispatch transition next stage changed")
                prior = connection.execute(
                    "SELECT * FROM commands WHERE command_id = ?",
                    (prior_command_id,),
                ).fetchone()
                if prior is None or str(prior["session_id"]) != session_id:
                    raise ConflictError("dispatch transition prior command is unknown")
                latest = _dispatch_transition_predecessor(connection, session_id)
                if latest is None or str(latest["command_id"]) != prior_command_id:
                    raise ConflictError(
                        "dispatch transition prior command is not latest"
                    )
                latest_checkpoint = connection.execute(
                    """
                    SELECT * FROM checkpoints WHERE session_id = ?
                    ORDER BY sequence DESC, created_at DESC, checkpoint_id DESC
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                if latest_checkpoint is None:
                    raise ConflictError(
                        "dispatch transition has no certified checkpoint"
                    )
                checkpoint_id = str(latest_checkpoint["checkpoint_id"])
                prior_material_digest, unused_summary = inspect_workspace(
                    Path(str(session["worktree"]))
                )
                del unused_summary
                anchor = _dispatch_transition_anchor(
                    connection,
                    prior,
                    checkpoint_id,
                    prior_material_digest,
                )
                for name, expected in anchor.items():
                    if authorization.get(name) != expected:
                        raise ConflictError("dispatch transition " + name + " changed")
                if authorization.get("prior_material_digest") != (
                    prior_material_digest
                ):
                    raise ConflictError("dispatch transition material changed")
                latest_invalidation = connection.execute(
                    """
                    SELECT invalidation_id FROM dispatch_invalidations
                    WHERE session_id = ?
                    ORDER BY created_at DESC, invalidation_id DESC LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                latest_invalidation_id = ""
                if latest_invalidation is not None:
                    latest_invalidation_id = str(latest_invalidation["invalidation_id"])
                actual_generation_digest = normalized_digest(
                    {
                        "material": {
                            "base_commit": str(latest_checkpoint["base_commit"]),
                            "patch_digest": str(latest_checkpoint["patch_digest"]),
                            "untracked_digest": str(
                                latest_checkpoint["untracked_digest"]
                            ),
                        },
                        "invalidation_id": latest_invalidation_id,
                    }
                )
                if authorization.get("prior_generation_digest") != (
                    actual_generation_digest
                ):
                    raise ConflictError("dispatch transition generation changed")
                transition_sequence = authorization.get("transition_sequence")
                if isinstance(transition_sequence, bool) or not isinstance(
                    transition_sequence, int
                ):
                    raise ConflictError("dispatch transition sequence is invalid")
                epoch_id = str(authorization.get("epoch_id", ""))
                goal_id = str(session["goal_id"])
                if authorization.get("goal_id") != goal_id or not goal_id:
                    raise ConflictError("dispatch transition goal changed")
                transition_count_row = connection.execute(
                    """
                    SELECT COALESCE(MAX(transition_sequence), 0) AS count
                    FROM dispatch_transition_ledger
                    WHERE session_id = ? AND goal_id = ? AND epoch_id = ?
                    """,
                    (session_id, goal_id, epoch_id),
                ).fetchone()
                transition_count = 0
                if transition_count_row is not None:
                    transition_count = int(transition_count_row["count"])
                expected_sequence = transition_count + 1
                if transition_sequence != expected_sequence:
                    raise ConflictError("dispatch transition sequence is out of order")
                if transition_sequence > 1:
                    predecessor = connection.execute(
                        """
                        SELECT consumed_command_id
                        FROM dispatch_transition_ledger
                        WHERE session_id = ? AND goal_id = ? AND epoch_id = ?
                          AND transition_sequence = ?
                        """,
                        (
                            session_id,
                            goal_id,
                            epoch_id,
                            transition_sequence - 1,
                        ),
                    ).fetchone()
                    if predecessor is None:
                        raise ConflictError(
                            "dispatch transition predecessor is missing"
                        )
                    consumed_command_id = str(predecessor["consumed_command_id"])
                    if consumed_command_id != prior_command_id:
                        if anchor["prior_anchor_kind"] != "control-command":
                            raise ConflictError(
                                "dispatch transition predecessor was not consumed "
                                "by the prior command"
                            )
                        consumed_command = connection.execute(
                            "SELECT * FROM commands WHERE command_id = ?",
                            (consumed_command_id,),
                        ).fetchone()
                        if (
                            consumed_command is None
                            or str(consumed_command["session_id"]) != session_id
                            or str(consumed_command["command_type"]) != "message"
                            or str(consumed_command["status"])
                            not in {CommandStatus.CANCELLED, CommandStatus.FAILED}
                        ):
                            raise ConflictError(
                                "dispatch transition control lineage is missing"
                            )
                        control_result = _load_object(str(prior["result_json"]))
                        if (
                            str(control_result.get("target_command_id", ""))
                            != consumed_command_id
                            or str(control_result.get("checkpoint_id", ""))
                            != (checkpoint_id)
                            or str(
                                control_result.get(
                                    "workspace_material_digest",
                                    "",
                                )
                            )
                            != prior_material_digest
                        ):
                            raise ConflictError(
                                "dispatch transition control lineage changed"
                            )
                        dispatch_rows = connection.execute(
                            """
                            SELECT checkpoint_id, state
                            FROM command_dispatches WHERE command_id = ?
                            AND crossed_boundary = 1
                            ORDER BY created_at, attempt_id
                            """,
                            (consumed_command_id,),
                        ).fetchall()
                        if (
                            len(dispatch_rows) != 1
                            or str(dispatch_rows[0]["checkpoint_id"]) != checkpoint_id
                            or str(dispatch_rows[0]["state"])
                            not in {"interrupted", "failed", "ambiguous"}
                        ):
                            raise ConflictError(
                                "dispatch transition interrupted boundary changed"
                            )
                        interrupt_events = connection.execute(
                            """
                            SELECT metadata_json FROM events
                            WHERE session_id = ?
                            AND event_type = 'turn.interrupted'
                            ORDER BY sequence
                            """,
                            (session_id,),
                        ).fetchall()
                        expected_interrupt = {
                            "control_command_id": prior_command_id,
                            "target_command_id": consumed_command_id,
                            "checkpoint_id": checkpoint_id,
                            "workspace_material_digest": prior_material_digest,
                        }
                        matching_interrupts = [
                            item
                            for item in interrupt_events
                            if _load_object(str(item["metadata_json"]))
                            == expected_interrupt
                        ]
                        if len(matching_interrupts) != 1:
                            raise ConflictError(
                                "dispatch transition interrupt receipt changed"
                            )
                        command_lineage = connection.execute(
                            """
                            SELECT command_id, command_type, status
                            FROM commands WHERE session_id = ?
                            ORDER BY created_at, command_id
                            """,
                            (session_id,),
                        ).fetchall()
                        consumed_position = -1
                        prior_position = -1
                        for position, lineage_command in enumerate(command_lineage):
                            lineage_command_id = str(lineage_command["command_id"])
                            if lineage_command_id == consumed_command_id:
                                consumed_position = position
                            if lineage_command_id == prior_command_id:
                                prior_position = position
                        if consumed_position < 0 or prior_position <= consumed_position:
                            raise ConflictError(
                                "dispatch transition control lineage is out of order"
                            )
                        for lineage_command in command_lineage[
                            consumed_position + 1 : prior_position + 1
                        ]:
                            if str(lineage_command["command_type"]) not in (
                                TRANSITION_CONTROL_COMMANDS
                            ) or str(lineage_command["status"]) != (
                                CommandStatus.COMPLETE
                            ):
                                raise ConflictError(
                                    "dispatch transition control lineage is not exact"
                                )
                external_ref = {
                    "orchestrator": str(session["external_orchestrator"]),
                    "job_id": str(session["external_job_id"]),
                }
                if (
                    authorization.get("external_orchestrator")
                    != external_ref["orchestrator"]
                    or authorization.get("external_job_id") != external_ref["job_id"]
                ):
                    raise ConflictError("dispatch transition orchestrator changed")
                policy = authorization.get("policy")
                if not isinstance(policy, dict) or not policy:
                    raise ConflictError("dispatch transition policy is missing")
                if policy.get("schema") != (
                    "p13i/agent-harness/dispatch-generation-transition-policy/v1"
                ):
                    raise ConflictError("dispatch transition policy schema changed")
                policy_sha256 = normalized_digest(policy)
                if authorization.get("policy_sha256") != policy_sha256:
                    raise ConflictError("dispatch transition policy digest changed")
                if policy.get("session_id") != session_id:
                    raise ConflictError("dispatch transition policy session changed")
                if policy.get("external_ref") != external_ref:
                    raise ConflictError(
                        "dispatch transition policy orchestrator changed"
                    )
                if policy.get("epoch_id") != epoch_id or not epoch_id:
                    raise ConflictError("dispatch transition epoch changed")
                policy_ref = authorization.get("policy_ref")
                retained_policy = False
                if transition_sequence == 1:
                    if policy_ref is not None:
                        raise ConflictError(
                            "first dispatch transition has a policy reference"
                        )
                else:
                    expected_policy_ref = {
                        "policy_sha256": policy_sha256,
                        "session_id": session_id,
                        "goal_id": goal_id,
                        "epoch_id": epoch_id,
                    }
                    if policy_ref != expected_policy_ref:
                        raise ConflictError(
                            "dispatch transition policy reference changed"
                        )
                    retained_policy_row = connection.execute(
                        """
                        SELECT payload_json FROM dispatch_transition_policies
                        WHERE policy_sha256 = ? AND session_id = ?
                          AND goal_id = ? AND epoch_id = ?
                        """,
                        (policy_sha256, session_id, goal_id, epoch_id),
                    ).fetchone()
                    if retained_policy_row is None:
                        raise ConflictError(
                            "dispatch transition policy reference is unknown"
                        )
                    if str(retained_policy_row["payload_json"]) != _dump(policy):
                        raise ConflictError(
                            "dispatch transition retained policy changed"
                        )
                    retained_policy = True
                allowed_roles = policy.get("allowed_agent_roles")
                allowed_prefixes = policy.get("allowed_step_prefixes")
                max_transitions = policy.get("max_transitions")
                if (
                    not isinstance(allowed_roles, list)
                    or normalized_next_turn_ref.get("agent_role") not in allowed_roles
                ):
                    raise ConflictError("dispatch transition role is outside policy")
                step_id = normalized_next_turn_ref.get("step_id", "")
                if not isinstance(allowed_prefixes, list) or not any(
                    isinstance(prefix, str) and prefix and step_id.startswith(prefix)
                    for prefix in allowed_prefixes
                ):
                    raise ConflictError("dispatch transition step is outside policy")
                if (
                    isinstance(max_transitions, bool)
                    or not isinstance(max_transitions, int)
                    or max_transitions > 1_000
                    or transition_sequence > max_transitions
                ):
                    raise ConflictError("dispatch transition exceeds policy limit")
                transitions = policy.get("transitions")
                if not isinstance(transitions, list):
                    raise ConflictError("dispatch transition policy stages changed")
                if max_transitions != len(transitions):
                    raise ConflictError("dispatch transition policy limit changed")
                if transition_sequence > len(transitions):
                    raise ConflictError(
                        "dispatch transition sequence is outside policy"
                    )
                matching_transition = transitions[transition_sequence - 1]
                if not isinstance(matching_transition, dict):
                    raise ConflictError("dispatch transition policy stage changed")
                if matching_transition.get("sequence") != transition_sequence:
                    raise ConflictError("dispatch transition policy order changed")
                if not retained_policy:
                    for index, transition in enumerate(transitions, start=1):
                        if not isinstance(transition, dict):
                            raise ConflictError(
                                "dispatch transition policy stage changed"
                            )
                        if transition.get("sequence") != index:
                            raise ConflictError(
                                "dispatch transition policy order changed"
                            )
                        transition_ref = normalize_turn_ref(
                            transition.get("next_turn_ref")
                        )
                        if transition_ref["agent_role"] not in allowed_roles:
                            raise ConflictError(
                                "dispatch transition policy stage role changed"
                            )
                        transition_step = transition_ref["step_id"]
                        if not any(
                            isinstance(prefix, str)
                            and prefix
                            and transition_step.startswith(prefix)
                            for prefix in allowed_prefixes
                        ):
                            raise ConflictError(
                                "dispatch transition policy stage step changed"
                            )
                        transition_digest = str(
                            transition.get("next_command_digest", "")
                        )
                        if len(transition_digest) != 64 or any(
                            character not in "0123456789abcdef"
                            for character in transition_digest
                        ):
                            raise ConflictError(
                                "dispatch transition policy stage digest changed"
                            )
                if (
                    normalize_turn_ref(matching_transition.get("next_turn_ref"))
                    != normalized_next_turn_ref
                ):
                    raise ConflictError("dispatch transition policy stage changed")
                next_command_digest = str(authorization.get("next_command_digest", ""))
                if matching_transition.get("next_command_digest") != (
                    next_command_digest
                ):
                    raise ConflictError("dispatch transition command policy changed")
                goal = connection.execute(
                    "SELECT constraints_json FROM goals WHERE goal_id = ?",
                    (str(session["goal_id"]),),
                ).fetchone()
                constraint = (
                    "dispatch-generation-transition-policy-sha256:" + policy_sha256
                )
                epoch_constraint = "dispatch-generation-transition-epoch:" + epoch_id
                constraints = []
                if goal is not None:
                    constraints = json.loads(str(goal["constraints_json"]))
                if constraint not in constraints or epoch_constraint not in constraints:
                    raise ConflictError("dispatch transition policy is not authorized")
                exact_receipt = {
                    "session_id": session_id,
                    "external_ref": external_ref,
                    "goal_id": goal_id,
                    "prior_command_id": prior_command_id,
                    "prior_command_type": anchor["prior_command_type"],
                    "prior_anchor_kind": anchor["prior_anchor_kind"],
                    "prior_reconciliation_id": anchor["prior_reconciliation_id"],
                    "prior_reconciliation_resolution": anchor[
                        "prior_reconciliation_resolution"
                    ],
                    "prior_checkpoint_id": checkpoint_id,
                    "prior_generation_digest": actual_generation_digest,
                    "prior_material_digest": prior_material_digest,
                    "next_turn_ref": normalized_next_turn_ref,
                    "transition_sequence": transition_sequence,
                    "epoch_id": epoch_id,
                    "policy_sha256": policy_sha256,
                    "next_command_digest": next_command_digest,
                }
                if authorization.get("receipt") != exact_receipt:
                    raise ConflictError("dispatch transition source receipt changed")
                if authorization.get("receipt_sha256") != normalized_digest(
                    exact_receipt
                ):
                    raise ConflictError("dispatch transition receipt digest changed")
            created_at = utc_now()
            stored_authorization = authorization
            if authorization_schema == transition_schema:
                stored_authorization = self._retain_dispatch_transition_policy(
                    connection,
                    session_id=session_id,
                    authorization=authorization,
                    created_at=created_at,
                )
            invalidation_id = new_uuid()
            invalidation_metadata = {
                "invalidation_id": invalidation_id,
                "reason": reason,
                "authorization_digest": authorization_digest,
            }
            if prior_command_id:
                invalidation_metadata.update(
                    {
                        "prior_command_id": prior_command_id,
                        "prior_command_type": authorization.get("prior_command_type"),
                        "prior_anchor_kind": authorization.get("prior_anchor_kind"),
                        "prior_reconciliation_id": authorization.get(
                            "prior_reconciliation_id"
                        ),
                        "prior_reconciliation_resolution": authorization.get(
                            "prior_reconciliation_resolution"
                        ),
                        "goal_id": authorization.get("goal_id"),
                        "prior_checkpoint_id": authorization.get("prior_checkpoint_id"),
                        "prior_generation_digest": authorization.get(
                            "prior_generation_digest"
                        ),
                        "prior_material_digest": authorization.get(
                            "prior_material_digest"
                        ),
                        "next_turn_ref": normalized_next_turn_ref,
                        "transition_sequence": authorization.get("transition_sequence"),
                        "epoch_id": authorization.get("epoch_id"),
                        "policy_sha256": authorization.get("policy_sha256"),
                        "next_command_digest": authorization.get("next_command_digest"),
                    }
                )
            connection.execute(
                "INSERT INTO dispatch_invalidations VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    invalidation_id,
                    session_id,
                    reason,
                    authorization_digest,
                    request_digest,
                    idempotency_key,
                    created_at,
                ),
            )
            if authorization_schema == transition_schema:
                connection.execute(
                    """
                    INSERT INTO dispatch_transition_ledger(
                        invalidation_id, session_id, goal_id, epoch_id,
                        transition_sequence, policy_sha256,
                        authorization_digest, receipt_sha256,
                        request_digest, prior_command_id,
                        prior_command_type, prior_anchor_kind,
                        prior_reconciliation_id,
                        prior_reconciliation_resolution,
                        prior_checkpoint_id, prior_generation_digest,
                        prior_material_digest, next_turn_ref_json,
                        next_command_digest, state, reserved_command_id,
                        consumed_command_id, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        invalidation_id,
                        session_id,
                        str(authorization.get("goal_id", "")),
                        str(authorization.get("epoch_id", "")),
                        int(authorization.get("transition_sequence", 0)),
                        str(authorization.get("policy_sha256", "")),
                        authorization_digest,
                        str(authorization.get("receipt_sha256", "")),
                        request_digest,
                        prior_command_id,
                        str(authorization.get("prior_command_type", "")),
                        str(authorization.get("prior_anchor_kind", "")),
                        str(authorization.get("prior_reconciliation_id", "")),
                        str(
                            authorization.get(
                                "prior_reconciliation_resolution",
                                "",
                            )
                        ),
                        str(authorization.get("prior_checkpoint_id", "")),
                        str(authorization.get("prior_generation_digest", "")),
                        str(authorization.get("prior_material_digest", "")),
                        _dump(normalized_next_turn_ref),
                        str(authorization.get("next_command_digest", "")),
                        "authorized",
                        "",
                        "",
                        created_at,
                        created_at,
                    ),
                )
            self._insert_authorization_receipt(
                connection,
                session_id=session_id,
                operation="dispatch-invalidation",
                operation_id=invalidation_id,
                authorization=stored_authorization,
                created_at=created_at,
                authorization_digest=authorization_digest,
            )
            sequence_row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS sequence
                FROM events WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            sequence = 1
            if sequence_row is not None:
                sequence = int(sequence_row["sequence"]) + 1
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    sequence,
                    invalidation_id,
                    "dispatch.invalidation",
                    "",
                    "",
                    "complete",
                    _dump(invalidation_metadata),
                    "",
                    "",
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM dispatch_invalidations WHERE invalidation_id = ?",
                (invalidation_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("dispatch invalidation was not recorded")
            result = dict(row)
            result["prior_command_id"] = prior_command_id
            result["next_turn_ref"] = normalized_next_turn_ref
            if prior_command_id:
                for name in (
                    "goal_id",
                    "prior_command_type",
                    "prior_anchor_kind",
                    "prior_reconciliation_id",
                    "prior_reconciliation_resolution",
                    "prior_checkpoint_id",
                    "prior_generation_digest",
                    "prior_material_digest",
                    "transition_sequence",
                    "epoch_id",
                    "policy_sha256",
                    "next_command_digest",
                ):
                    result[name] = authorization.get(name)
            return result

    def dispatch_invalidation_replay(
        self,
        session_id: str,
        idempotency_key: str,
        request_digest: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            existing = self._connection.execute(
                """
                SELECT * FROM dispatch_invalidations
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is None:
                return None
            if (
                str(existing["session_id"]) != session_id
                or str(existing["request_digest"]) != request_digest
            ):
                raise ConflictError("dispatch invalidation idempotency key was reused")
            authorization_row = self._connection.execute(
                """
                SELECT schema, payload_json FROM authorization_receipts
                WHERE operation = 'dispatch-invalidation'
                  AND operation_id = ?
                """,
                (str(existing["invalidation_id"]),),
            ).fetchone()
            if authorization_row is None:
                raise RuntimeError(
                    "dispatch invalidation authorization receipt is missing"
                )
            authorization = _load_object(str(authorization_row["payload_json"]))
            transition_schema = (
                "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
            )
            if str(authorization_row["schema"]) == transition_schema and not (
                _dispatch_transition_epoch_is_active(
                    self._connection,
                    session_id,
                    authorization,
                )
            ):
                raise ConflictError("dispatch transition epoch is no longer active")
            result = dict(existing)
            prior_command_id = str(authorization.get("prior_command_id", ""))
            result["prior_command_id"] = prior_command_id
            result["next_turn_ref"] = normalize_turn_ref(
                authorization.get("next_turn_ref")
            )
            if prior_command_id:
                for name in (
                    "goal_id",
                    "prior_command_type",
                    "prior_anchor_kind",
                    "prior_reconciliation_id",
                    "prior_reconciliation_resolution",
                    "prior_checkpoint_id",
                    "prior_generation_digest",
                    "prior_material_digest",
                    "transition_sequence",
                    "epoch_id",
                    "policy_sha256",
                    "next_command_digest",
                ):
                    result[name] = authorization.get(name)
            return result

    def reserve_dispatch_generation_transition(
        self,
        session_id: str,
        command_id: str,
        turn_ref: dict[str, Any],
        live_material_digest: str,
    ) -> str:
        normalized_turn_ref = normalize_turn_ref(turn_ref)
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT d.invalidation_id, a.schema, a.payload_json,
                    l.state, l.reserved_command_id, l.consumed_command_id
                FROM dispatch_invalidations AS d
                JOIN authorization_receipts AS a
                  ON a.operation_id = d.invalidation_id
                 AND a.operation = 'dispatch-invalidation'
                LEFT JOIN dispatch_transition_ledger AS l
                  ON l.invalidation_id = d.invalidation_id
                WHERE d.session_id = ?
                ORDER BY d.created_at DESC, d.invalidation_id DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return ""
            transition_schema = (
                "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
            )
            if str(row["schema"]) != transition_schema:
                return ""
            invalidation_id = str(row["invalidation_id"])
            authorization = _load_object(str(row["payload_json"]))
            if not _dispatch_transition_epoch_is_active(
                connection,
                session_id,
                authorization,
            ):
                return "epoch-mismatch"
            session = connection.execute(
                "SELECT worktree FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return "material-mismatch"
            if not _dispatch_transition_material_is_current(
                connection,
                session_id,
                authorization,
                Path(str(session["worktree"])),
                live_material_digest,
            ):
                return "material-mismatch"
            material_binding = "digest"
            if authorization.get("prior_material_digest") != live_material_digest:
                material_binding = "checkpoint-collapse"
            expected_turn_ref = normalize_turn_ref(authorization.get("next_turn_ref"))
            if normalized_turn_ref != expected_turn_ref:
                return "stage-mismatch"
            latest_state = str(row["state"] or "")
            bound_command_id = str(row["reserved_command_id"] or "")
            if latest_state == "consumed":
                if str(row["consumed_command_id"] or "") == command_id:
                    return "consumed"
                return "already-consumed"
            command = connection.execute(
                "SELECT command_type, payload_json FROM commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            envelope = connection.execute(
                "SELECT profile FROM command_envelopes WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if command is None or envelope is None:
                return "command-mismatch"
            actual_command_digest = command_envelope_digest(
                str(command["command_type"]),
                _load_object(str(command["payload_json"])),
                str(envelope["profile"]),
            )
            legacy_command_digest = legacy_command_envelope_digest(
                str(command["command_type"]),
                _load_object(str(command["payload_json"])),
                str(envelope["profile"]),
            )
            if authorization.get("next_command_digest") not in {
                actual_command_digest,
                legacy_command_digest,
            }:
                return "command-mismatch"
            if latest_state == "reserved":
                if bound_command_id == command_id:
                    return "reserved"
                bound_command = connection.execute(
                    "SELECT status FROM commands WHERE command_id = ?",
                    (bound_command_id,),
                ).fetchone()
                crossed = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM command_dispatches
                    WHERE command_id = ? AND crossed_boundary = 1
                    """,
                    (bound_command_id,),
                ).fetchone()
                crossed_count = 0
                if crossed is not None:
                    crossed_count = int(crossed["count"])
                releasable = bound_command is not None and str(
                    bound_command["status"]
                ) in {CommandStatus.FAILED, CommandStatus.CANCELLED}
                if not releasable or crossed_count > 0:
                    return "already-consumed"
                self._append_dispatch_transition_event(
                    connection,
                    session_id,
                    "released",
                    {
                        "invalidation_id": invalidation_id,
                        "command_id": bound_command_id,
                        "reason": "terminal-pre-boundary-failure",
                    },
                )
                connection.execute(
                    """
                    UPDATE dispatch_transition_ledger
                    SET state = 'released', reserved_command_id = '',
                        updated_at = ?
                    WHERE invalidation_id = ?
                    """,
                    (utc_now(), invalidation_id),
                )
            self._append_dispatch_transition_event(
                connection,
                session_id,
                "reserved",
                {
                    "invalidation_id": invalidation_id,
                    "command_id": command_id,
                    "prior_command_id": str(authorization.get("prior_command_id", "")),
                    "epoch_id": str(authorization.get("epoch_id", "")),
                    "next_turn_ref": normalized_turn_ref,
                    "material_binding": material_binding,
                },
            )
            connection.execute(
                """
                UPDATE dispatch_transition_ledger
                SET state = 'reserved', reserved_command_id = ?,
                    updated_at = ?
                WHERE invalidation_id = ?
                """,
                (command_id, utc_now(), invalidation_id),
            )
            return "reserved"

    def _append_dispatch_transition_event(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        state: str,
        metadata: dict[str, Any],
    ) -> None:
        sequence_row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) AS sequence
            FROM events WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        sequence = 1
        if sequence_row is not None:
            sequence = int(sequence_row["sequence"]) + 1
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                sequence,
                new_uuid(),
                "dispatch.generation.transition." + state,
                "",
                "",
                state,
                _dump(metadata),
                "",
                "",
                utc_now(),
            ),
        )

    def _consume_reserved_dispatch_transition(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        command_id: str,
        workspace: Path,
    ) -> None:
        row = connection.execute(
            """
            SELECT d.invalidation_id, a.schema, a.payload_json,
                l.state, l.reserved_command_id, l.consumed_command_id
            FROM dispatch_invalidations AS d
            JOIN authorization_receipts AS a
              ON a.operation_id = d.invalidation_id
             AND a.operation = 'dispatch-invalidation'
            LEFT JOIN dispatch_transition_ledger AS l
              ON l.invalidation_id = d.invalidation_id
            WHERE d.session_id = ?
            ORDER BY d.created_at DESC, d.invalidation_id DESC LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if row is None or str(row["schema"]) != (
            "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
        ):
            return
        invalidation_id = str(row["invalidation_id"])
        authorization = _load_object(str(row["payload_json"]))
        if not _dispatch_transition_epoch_is_active(
            connection,
            session_id,
            authorization,
        ):
            raise ConflictError("dispatch transition epoch is stale")
        live_material_digest, unused_summary = inspect_workspace(workspace)
        del unused_summary
        if not _dispatch_transition_material_is_current(
            connection,
            session_id,
            authorization,
            workspace,
            live_material_digest,
        ):
            raise ConflictError("dispatch transition live material changed")
        material_binding = "digest"
        if authorization.get("prior_material_digest") != live_material_digest:
            material_binding = "checkpoint-collapse"
        command = connection.execute(
            "SELECT command_type, payload_json FROM commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        envelope = connection.execute(
            "SELECT profile FROM command_envelopes WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        if command is None or envelope is None:
            raise ConflictError("dispatch transition command is missing")
        actual_command_digest = command_envelope_digest(
            str(command["command_type"]),
            _load_object(str(command["payload_json"])),
            str(envelope["profile"]),
        )
        legacy_command_digest = legacy_command_envelope_digest(
            str(command["command_type"]),
            _load_object(str(command["payload_json"])),
            str(envelope["profile"]),
        )
        if authorization.get("next_command_digest") not in {
            actual_command_digest,
            legacy_command_digest,
        }:
            raise ConflictError("dispatch transition command changed")
        latest_state = str(row["state"] or "")
        bound_command_id = str(row["reserved_command_id"] or "")
        consumed_command_id = str(row["consumed_command_id"] or "")
        if latest_state == "consumed" and consumed_command_id == command_id:
            return
        if latest_state != "reserved" or bound_command_id != command_id:
            raise ConflictError("dispatch transition reservation is stale")
        self._append_dispatch_transition_event(
            connection,
            session_id,
            "consumed",
            {
                "invalidation_id": invalidation_id,
                "command_id": command_id,
                "epoch_id": str(authorization.get("epoch_id", "")),
                "material_binding": material_binding,
            },
        )
        cursor = connection.execute(
            """
            UPDATE dispatch_transition_ledger
            SET state = 'consumed', consumed_command_id = ?, updated_at = ?
            WHERE invalidation_id = ? AND state = 'reserved'
              AND reserved_command_id = ?
            """,
            (command_id, utc_now(), invalidation_id, command_id),
        )
        if cursor.rowcount != 1:
            raise ConflictError("dispatch transition consumption is stale")

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

    def checkpoint(self, checkpoint_id: str) -> Checkpoint:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM checkpoints WHERE checkpoint_id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("checkpoint")
        return _checkpoint(row)

    def record_dispatch_checkpoint(
        self,
        command_id: str,
        attempt_id: str,
        turn_id: str,
        checkpoint_id: str,
    ) -> None:
        command = self.get_command(command_id)
        with self.transaction() as connection:
            now = utc_now()
            connection.execute(
                """
                INSERT INTO command_dispatches VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    attempt_id,
                    command_id,
                    command.session_id,
                    turn_id,
                    checkpoint_id,
                    0,
                    "prepared",
                    now,
                    now,
                ),
            )

    def _mark_provider_boundary(
        self,
        connection: sqlite3.Connection,
        attempt_id: str,
        now: str,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE command_dispatches SET crossed_boundary = 1,
                state = 'dispatched', updated_at = ?
            WHERE attempt_id = ? AND crossed_boundary = 0
            """,
            (now, attempt_id),
        )
        if cursor.rowcount != 1:
            raise ConflictError("provider dispatch boundary is stale")

    def mark_provider_boundary(self, attempt_id: str) -> None:
        with self.transaction() as connection:
            self._mark_provider_boundary(connection, attempt_id, utc_now())

    def complete_dispatch(self, attempt_id: str, state: str) -> None:
        if state not in {
            "complete",
            "failed",
            "interrupted",
            "exhausted",
        }:
            raise ValueError("dispatch completion state is unsupported")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE command_dispatches SET state = ?, updated_at = ?
                WHERE attempt_id = ?
                """,
                (state, utc_now(), attempt_id),
            )
            if cursor.rowcount != 1:
                raise NotFoundError("provider dispatch")

    def complete_command_execution(
        self,
        command_id: str,
        turn_id: str,
        native_session_id: str,
        consumption: dict[str, Any],
        result: dict[str, Any],
        *,
        goal_evidence: tuple[Evidence, ...] = (),
    ) -> CommandReceipt:
        now = utc_now()
        with self.transaction() as connection:
            dispatch = connection.execute(
                """
                SELECT command_dispatches.attempt_id, commands.session_id,
                    commands.status
                FROM command_dispatches
                JOIN commands USING(command_id)
                WHERE command_dispatches.command_id = ? AND turn_id = ?
                    AND crossed_boundary = 1
                """,
                (command_id, turn_id),
            ).fetchone()
            if dispatch is None:
                raise ConflictError("completed command dispatch boundary is missing")
            checkpoint = connection.execute(
                "SELECT session_id FROM checkpoints WHERE checkpoint_id = ?",
                (str(result.get("checkpoint_id", "")),),
            ).fetchone()
            if checkpoint is None or str(checkpoint["session_id"]) != str(
                dispatch["session_id"]
            ):
                raise ConflictError("completed command checkpoint is missing")
            if str(dispatch["status"]) != CommandStatus.DISPATCHING:
                raise ConflictError("completed command is not dispatching")
            session = connection.execute(
                "SELECT goal_id FROM sessions WHERE session_id = ?",
                (str(dispatch["session_id"]),),
            ).fetchone()
            if session is None:
                raise NotFoundError("session")
            for evidence in goal_evidence:
                if evidence.goal_id != str(session["goal_id"]):
                    raise ConflictError("completed command evidence goal is not current")
            attempt_id = str(dispatch["attempt_id"])
            connection.execute(
                """
                UPDATE provider_attempts SET status = 'complete',
                    native_session_id = ?, ended_at = ?
                WHERE attempt_id = ?
                """,
                (native_session_id, now, attempt_id),
            )
            connection.execute(
                """
                UPDATE turns SET status = 'complete', completed_at = ?
                WHERE turn_id = ?
                """,
                (now, turn_id),
            )
            connection.execute(
                """
                UPDATE command_dispatches SET state = 'complete', updated_at = ?
                WHERE attempt_id = ?
                """,
                (now, attempt_id),
            )
            connection.execute(
                """
                UPDATE command_envelopes SET state = 'complete',
                    consumption_json = ?, updated_at = ?
                WHERE command_id = ?
                """,
                (_dump(consumption), now, command_id),
            )
            connection.execute(
                """
                UPDATE commands SET status = ?, result_json = ?, updated_at = ?
                WHERE command_id = ?
                """,
                (CommandStatus.COMPLETE, _dump(result), now, command_id),
            )
            sequence_row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS sequence
                FROM events WHERE session_id = ?
                """,
                (str(dispatch["session_id"]),),
            ).fetchone()
            sequence = int(sequence_row["sequence"])
            for evidence in goal_evidence:
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
                sequence += 1
                connection.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(dispatch["session_id"]),
                        sequence,
                        new_uuid(),
                        "goal.evidence",
                        "",
                        "",
                        "complete",
                        _dump(evidence.as_dict()),
                        "",
                        turn_id,
                        now,
                    ),
                )
            if goal_evidence:
                connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, str(dispatch["session_id"])),
                )
            row = connection.execute(
                "SELECT * FROM commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        return _command(row)

    def recover_interrupted_commands(
        self,
        session_id: str,
        current_workspace_digest: str,
        current_workspace_summary: str,
    ) -> RestartRecovery:
        now = utc_now()
        requeued: list[str] = []
        reconciliations: list[ReconciliationRecord] = []
        with self.transaction() as connection:
            commands = connection.execute(
                """
                SELECT * FROM commands
                WHERE session_id = ? AND status = ?
                ORDER BY created_at, command_id
                """,
                (session_id, CommandStatus.DISPATCHING),
            ).fetchall()
            for command in commands:
                dispatches = connection.execute(
                    """
                    SELECT * FROM command_dispatches
                    WHERE command_id = ? AND crossed_boundary = 1
                    ORDER BY created_at DESC, attempt_id DESC
                    """,
                    (command["command_id"],),
                ).fetchall()
                if not dispatches:
                    self._requeue_pre_boundary(
                        connection,
                        str(command["command_id"]),
                        now,
                    )
                    requeued.append(str(command["command_id"]))
                    continue
                ambiguous_dispatch = None
                for dispatch in dispatches:
                    if self._dispatch_known_undelivered(connection, dispatch):
                        continue
                    ambiguous_dispatch = dispatch
                    break
                if ambiguous_dispatch is None:
                    self._requeue_pre_boundary(
                        connection,
                        str(command["command_id"]),
                        now,
                    )
                    requeued.append(str(command["command_id"]))
                    continue
                record = self._create_reconciliation(
                    connection,
                    command,
                    ambiguous_dispatch,
                    current_workspace_digest,
                    current_workspace_summary,
                    now,
                )
                reconciliations.append(record)
        return RestartRecovery(
            requeued_command_ids=tuple(requeued),
            reconciliations=tuple(reconciliations),
        )

    def _dispatch_known_undelivered(
        self,
        connection: sqlite3.Connection,
        dispatch: sqlite3.Row,
    ) -> bool:
        delivery = connection.execute(
            """
            SELECT state, accepted_at FROM context_deliveries
            WHERE attempt_id = ?
            """,
            (str(dispatch["attempt_id"]),),
        ).fetchone()
        return self._context_delivery_known_undelivered(
            dispatch,
            delivery,
        )

    def _context_delivery_known_undelivered(
        self,
        dispatch: sqlite3.Row | None,
        delivery: sqlite3.Row | None,
    ) -> bool:
        if delivery is None:
            return False
        if str(delivery["state"]) not in {"prepared", "superseded"}:
            return False
        if str(delivery["accepted_at"]):
            return False
        if dispatch is None:
            return False
        crossed_boundary = int(dispatch["crossed_boundary"]) > 0
        if not crossed_boundary:
            return True
        return str(dispatch["state"]) in {
            "failed",
            "exhausted",
            "interrupted",
        }

    def _requeue_pre_boundary(
        self,
        connection: sqlite3.Connection,
        command_id: str,
        now: str,
    ) -> None:
        connection.execute(
            """
            UPDATE commands SET status = ?, result_json = '{}',
                updated_at = ? WHERE command_id = ?
            """,
            (CommandStatus.QUEUED, now, command_id),
        )
        connection.execute(
            """
            UPDATE provider_attempts SET status = 'interrupted',
                ended_at = ?
            WHERE status = 'running' AND session_id = (
                SELECT session_id FROM commands WHERE command_id = ?
            )
            """,
            (now, command_id),
        )
        connection.execute(
            """
            UPDATE command_dispatches SET state = 'interrupted',
                updated_at = ?
            WHERE command_id = ? AND crossed_boundary = 0
            """,
            (now, command_id),
        )
        connection.execute(
            """
            UPDATE command_envelopes SET state = 'reserved',
                updated_at = ? WHERE command_id = ?
            """,
            (now, command_id),
        )

    def _create_reconciliation(
        self,
        connection: sqlite3.Connection,
        command: sqlite3.Row,
        dispatch: sqlite3.Row,
        current_workspace_digest: str,
        current_workspace_summary: str,
        now: str,
    ) -> ReconciliationRecord:
        existing = connection.execute(
            """
            SELECT * FROM reconciliations WHERE command_id = ?
            """,
            (command["command_id"],),
        ).fetchone()
        if existing is not None:
            return _reconciliation(existing)
        attempt_rows = connection.execute(
            """
            SELECT provider_attempts.* FROM provider_attempts
            JOIN command_dispatches USING(attempt_id)
            WHERE command_dispatches.command_id = ?
            ORDER BY provider_attempts.started_at,
                provider_attempts.attempt_id
            """,
            (command["command_id"],),
        ).fetchall()
        attempts = [
            {
                "attempt_id": str(row["attempt_id"]),
                "provider": str(row["provider"]),
                "model": str(row["model"]),
                "effort": str(row["effort"]),
                "status": "ambiguous",
                "native_status": str(row["status"]),
                "started_at": str(row["started_at"]),
                "ended_at": str(row["ended_at"]),
            }
            for row in attempt_rows
        ]
        envelope = connection.execute(
            """
            SELECT consumption_json FROM command_envelopes
            WHERE command_id = ?
            """,
            (command["command_id"],),
        ).fetchone()
        consumption: dict[str, Any] = {}
        if envelope is not None:
            consumption = _load_object(envelope["consumption_json"])
        lease_rows = connection.execute(
            """
            SELECT * FROM process_leases
            WHERE session_id = ? AND command_id = ? AND attempt_id = ?
            AND state IN ('reserved', 'active', 'recovery-blocked')
            ORDER BY created_at, lease_id
            """,
            (
                str(command["session_id"]),
                str(command["command_id"]),
                str(dispatch["attempt_id"]),
            ),
        ).fetchall()
        if len(lease_rows) > 1:
            raise ConflictError("ambiguous dispatch has multiple active process leases")
        dispatch_identity = {
            "attempt_id": str(dispatch["attempt_id"]),
            "turn_id": str(dispatch["turn_id"]),
            "checkpoint_id": str(dispatch["checkpoint_id"]),
            "lease_id": "",
            "worker_incarnation": "",
            "pid": 0,
            "pid_start": "",
        }
        if lease_rows:
            lease = lease_rows[0]
            dispatch_identity.update(
                {
                    "lease_id": str(lease["lease_id"]),
                    "worker_incarnation": str(lease["worker_incarnation"]),
                    "pid": int(lease["pid"]),
                    "pid_start": str(lease["pid_start"]),
                }
            )
        reconciliation_id = new_uuid()
        connection.execute(
            """
            INSERT INTO reconciliations VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                reconciliation_id,
                command["session_id"],
                command["command_id"],
                dispatch["checkpoint_id"],
                current_workspace_digest,
                current_workspace_summary,
                _dump(attempts),
                _dump(consumption),
                ReconciliationStatus.PENDING,
                "",
                _dump({"dispatch_identity": dispatch_identity}),
                now,
                "",
            ),
        )
        connection.execute(
            """
            UPDATE commands SET status = ?, result_json = ?,
                updated_at = ? WHERE command_id = ?
            """,
            (
                CommandStatus.FAILED,
                _dump(
                    {
                        "code": "E_NEEDS_RECONCILIATION",
                        "message": (
                            "worker stopped after provider dispatch; "
                            "the effect is ambiguous"
                        ),
                        "reconciliation_id": reconciliation_id,
                    }
                ),
                now,
                command["command_id"],
            ),
        )
        connection.execute(
            """
            UPDATE provider_attempts SET status = 'ambiguous',
                ended_at = ?
                WHERE attempt_id IN (
                    SELECT attempt_id FROM command_dispatches
                    WHERE command_id = ? AND crossed_boundary = 1
                )
            """,
            (now, command["command_id"]),
        )
        connection.execute(
            """
            UPDATE command_dispatches SET state = 'ambiguous',
                updated_at = ?
            WHERE command_id = ? AND crossed_boundary = 1
            """,
            (now, command["command_id"]),
        )
        row = connection.execute(
            """
            SELECT * FROM reconciliations
            WHERE reconciliation_id = ?
            """,
            (reconciliation_id,),
        ).fetchone()
        return _reconciliation(row)

    def reconciliation(
        self,
        reconciliation_id: str,
    ) -> ReconciliationRecord:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("reconciliation")
        return _reconciliation(row)

    def set_reconciliation_command_error(
        self,
        reconciliation_id: str,
        code: str,
        message: str,
    ) -> CommandReceipt:
        with self.transaction() as connection:
            reconciliation = connection.execute(
                """
                SELECT command_id FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
            if reconciliation is None:
                raise NotFoundError("reconciliation")
            command = connection.execute(
                "SELECT * FROM commands WHERE command_id = ?",
                (str(reconciliation["command_id"]),),
            ).fetchone()
            if command is None or str(command["status"]) != CommandStatus.FAILED:
                raise ConflictError(
                    "reconciliation command is not in its failed state"
                )
            result = _load_object(str(command["result_json"]))
            if str(result.get("reconciliation_id", "")) != reconciliation_id:
                raise ConflictError("reconciliation command result changed")
            result["code"] = code
            result["message"] = message
            connection.execute(
                """
                UPDATE commands SET result_json = ?, updated_at = ?
                WHERE command_id = ?
                """,
                (
                    _dump(result),
                    utc_now(),
                    str(command["command_id"]),
                ),
            )
            updated = connection.execute(
                "SELECT * FROM commands WHERE command_id = ?",
                (str(command["command_id"]),),
            ).fetchone()
        return _command(updated)

    def record_reconciliation_discovery(
        self,
        reconciliation_id: str,
        checkpoint: Checkpoint,
        observed_workspace_digest: str,
    ) -> ReconciliationRecord:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("reconciliation")
            if str(row["session_id"]) != checkpoint.session_id:
                raise ConflictError(
                    "reconciliation discovery checkpoint belongs to another session"
                )
            if str(row["current_workspace_digest"]) != observed_workspace_digest:
                raise ConflictError(
                    "reconciliation discovery workspace digest is stale"
                )
            audit = _load_object(row["audit_json"])
            existing_checkpoint_id = str(audit.get("discovery_checkpoint_id", ""))
            if existing_checkpoint_id:
                existing = connection.execute(
                    "SELECT checkpoint_id FROM checkpoints WHERE checkpoint_id = ?",
                    (existing_checkpoint_id,),
                ).fetchone()
                if existing is None:
                    raise RuntimeError("reconciliation discovery checkpoint is missing")
                return _reconciliation(row)
            self.add_checkpoint(checkpoint)
            audit["discovery_checkpoint_id"] = checkpoint.checkpoint_id
            connection.execute(
                """
                UPDATE reconciliations SET audit_json = ?
                WHERE reconciliation_id = ?
                """,
                (_dump(audit), reconciliation_id),
            )
            updated = connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
            if updated is None:
                raise RuntimeError("reconciliation discovery was not recorded")
            return _reconciliation(updated)

    def pending_reconciliations(
        self,
        session_id: str,
    ) -> list[ReconciliationRecord]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE session_id = ? AND status != ?
                ORDER BY created_at, reconciliation_id
                """,
                (session_id, ReconciliationStatus.RESOLVED),
            ).fetchall()
        return [_reconciliation(row) for row in rows]

    def all_reconciliations(
        self,
        session_id: str,
    ) -> list[ReconciliationRecord]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE session_id = ?
                ORDER BY created_at, reconciliation_id
                """,
                (session_id,),
            ).fetchall()
        return [_reconciliation(row) for row in rows]

    def begin_reconciliation_resolution(
        self,
        reconciliation_id: str,
        decision: str,
        observed_workspace_digest: str,
    ) -> ReconciliationRecord:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("reconciliation")
            if str(row["status"]) == ReconciliationStatus.RESOLVED:
                raise ConflictError("reconciliation is already resolved")
            if str(row["current_workspace_digest"]) != observed_workspace_digest:
                raise ConflictError("observed workspace digest is stale")
            existing_decision = str(row["resolution"])
            if existing_decision and existing_decision != decision:
                raise ConflictError("reconciliation resolution is already in progress")
            if str(row["status"]) == ReconciliationStatus.PENDING:
                connection.execute(
                    """
                    UPDATE reconciliations SET status = ?,
                        resolution = ?
                    WHERE reconciliation_id = ? AND status = ?
                    """,
                    (
                        ReconciliationStatus.RESOLVING,
                        decision,
                        reconciliation_id,
                        ReconciliationStatus.PENDING,
                    ),
                )
            current = connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
        return _reconciliation(current)

    def project_resolved_reconciliation(
        self,
        reconciliation_id: str,
    ) -> CommandReceipt | None:
        """Project a reconciliation that was accepted before this ran.

        A reconciliation resolved by an earlier build left its command
        stranded on the ambiguous topology even though the turn had
        finished. Re-confirming the same resolution replays the
        projection; it declines once the command is already terminal,
        so it is safe to repeat.
        """
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("reconciliation")
            if (
                str(row["status"]) != ReconciliationStatus.RESOLVED
                or str(row["resolution"]) != ReconciliationDecision.ACCEPT_CURRENT
            ):
                return None
            audit = _load_object(str(row["audit_json"]))
            identity = _object_or_empty(audit.get("dispatch_identity"))
            attempt_id = str(identity.get("attempt_id", ""))
            turn_id = str(identity.get("turn_id", ""))
            command_id = str(row["command_id"])
            session_id = str(row["session_id"])
            # The resolution this replays certified one crossed
            # boundary. A second one means a dispatch the original
            # receipt never covered, so there is no single outcome to
            # project and this declines rather than picking a side.
            boundaries = connection.execute(
                """
                SELECT COUNT(*) AS total FROM command_dispatches
                WHERE session_id = ? AND command_id = ? AND crossed_boundary = 1
                """,
                (session_id, command_id),
            ).fetchone()
            if boundaries is None or int(boundaries["total"]) != 1:
                return None
            attempt = connection.execute(
                """
                SELECT provider_attempts.* FROM provider_attempts
                JOIN command_dispatches USING(attempt_id)
                WHERE provider_attempts.attempt_id = ?
                    AND provider_attempts.session_id = ?
                    AND command_dispatches.command_id = ?
                    AND command_dispatches.turn_id = ?
                    AND command_dispatches.crossed_boundary = 1
                """,
                (attempt_id, session_id, command_id, turn_id),
            ).fetchone()
            if attempt is None:
                return None
            if not _topology_receipt_binds(
                _object_or_empty(audit.get("topology_receipt")),
                reconciliation_id=reconciliation_id,
                session_id=session_id,
                command_id=command_id,
                attempt_id=attempt_id,
                turn_id=turn_id,
            ):
                return None
            completion = self._reconciled_completion_evidence(
                connection,
                row,
                attempt,
                turn_id,
                audit,
                ReconciliationDecision.ACCEPT_CURRENT,
            )
            if completion is None:
                return None
            now = utc_now()
            settled = self._settle_reconciled_topology(
                connection,
                command_id,
                turn_id,
                attempt,
                completion,
                now,
            )
            receipt = _object_or_empty(audit.get("topology_receipt"))
            receipt.update(
                {
                    "attempt_state": settled["settled_state"],
                    "turn_state": settled["settled_state"],
                    "dispatch_state": settled["settled_state"],
                    "envelope_state": settled["envelope_state"],
                    "guard_reason": settled["guard_reason"],
                    "prior_guard_reason": settled["prior_guard_reason"],
                    "command_status": str(CommandStatus.COMPLETE),
                    "completion_evidence": dict(completion),
                    "projected_at": now,
                }
            )
            audit["topology_receipt"] = receipt
            connection.execute(
                """
                UPDATE reconciliations SET audit_json = ?
                WHERE reconciliation_id = ?
                """,
                (_dump(audit), reconciliation_id),
            )
            # The original resolution event stays exactly as recorded.
            # This appends the later transition so the topology change
            # is observable, and so a proof taken at the new
            # through_sequence explains its own digest instead of
            # differing silently at the old one.
            self._append_projection_event(
                connection,
                session_id,
                reconciliation_id,
                command_id,
                receipt,
                now,
            )
            command = connection.execute(
                "SELECT * FROM commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        return _command(command)

    def _append_projection_event(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        reconciliation_id: str,
        command_id: str,
        receipt: dict[str, Any],
        now: str,
    ) -> None:
        event_id = derived_uuid(
            "p13i/agent-harness/reconciliation-projected/" + reconciliation_id
        )
        existing = connection.execute(
            """
            SELECT sequence FROM events
            WHERE session_id = ? AND event_id = ?
                AND event_type = 'reconciliation.projected'
            """,
            (session_id, event_id),
        ).fetchone()
        if existing is not None:
            return
        sequence_row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM events WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        sequence = 1
        if sequence_row is not None:
            sequence = int(sequence_row["sequence"]) + 1
        evidence = _object_or_empty(receipt.get("completion_evidence"))
        connection.execute(
            """
            INSERT INTO events VALUES (
                ?, ?, ?, 'reconciliation.projected', '', '',
                'complete', ?, '', '', ?
            )
            """,
            (
                session_id,
                sequence,
                event_id,
                _dump(
                    {
                        "reconciliation_id": reconciliation_id,
                        "command_id": command_id,
                        "decision": str(ReconciliationDecision.ACCEPT_CURRENT),
                        "resolution_checkpoint_id": str(
                            evidence.get("checkpoint_id", "")
                        ),
                        "resolution_workspace_digest": str(
                            evidence.get("workspace_material_digest", "")
                        ),
                        "topology_receipt": dict(receipt),
                    }
                ),
                now,
            ),
        )

    def resolve_reconciliation_record(
        self,
        reconciliation_id: str,
        decision: str,
        observed_workspace_digest: str,
        audit: dict[str, Any],
    ) -> ReconciliationRecord:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("reconciliation")
            if (
                str(row["status"]) == ReconciliationStatus.RESOLVED
                and str(row["resolution"]) == decision
                and str(row["current_workspace_digest"]) == observed_workspace_digest
            ):
                return _reconciliation(row)
            if str(row["status"]) not in {
                ReconciliationStatus.PENDING,
                ReconciliationStatus.RESOLVING,
            }:
                raise ConflictError("reconciliation was already resolved differently")
            if str(row["resolution"]) and str(row["resolution"]) != decision:
                raise ConflictError("reconciliation resolution is already in progress")
            if str(row["current_workspace_digest"]) != observed_workspace_digest:
                raise ConflictError("observed workspace digest is stale")
            now = utc_now()
            resolved_audit = self._normalize_reconciled_command_topology(
                connection,
                row,
                audit,
                now,
                decision,
            )
            connection.execute(
                """
                UPDATE reconciliations SET status = ?, resolution = ?,
                    audit_json = ?, resolved_at = ?
                WHERE reconciliation_id = ? AND status != ?
                """,
                (
                    ReconciliationStatus.RESOLVED,
                    decision,
                    _dump(resolved_audit),
                    now,
                    reconciliation_id,
                    ReconciliationStatus.RESOLVED,
                ),
            )
            resolved = connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
        return _reconciliation(resolved)

    def resolve_reconciliation_once(
        self,
        reconciliation_id: str,
        decision: str,
        observed_workspace_digest: str,
        audit: dict[str, Any],
        checkpoint: Checkpoint | None,
        *,
        idempotency_key: str,
        operation: str,
        request_digest: str,
    ) -> tuple[ReconciliationRecord, bool]:
        with self.transaction() as connection:
            receipt = connection.execute(
                """
                SELECT * FROM mutation_receipts
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if receipt is not None:
                if (
                    str(receipt["operation"]) != operation
                    or str(receipt["request_digest"]) != request_digest
                ):
                    raise ConflictError(
                        "idempotency key was already used for another mutation"
                    )
                response = _load_object(receipt["response_json"])
                value = _object_or_empty(response.get("reconciliation"))
                row = connection.execute(
                    """
                    SELECT * FROM reconciliations
                    WHERE reconciliation_id = ?
                    """,
                    (str(value.get("reconciliation_id", "")),),
                ).fetchone()
                if row is None:
                    raise RuntimeError(
                        "reconciliation receipt has no reconciliation row"
                    )
                return _reconciliation(row), False
            row = connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("reconciliation")
            if str(row["status"]) == ReconciliationStatus.RESOLVED and (
                str(row["resolution"]) != decision
                or str(row["current_workspace_digest"]) != observed_workspace_digest
            ):
                raise ConflictError("reconciliation was already resolved differently")
            if str(row["status"]) not in {
                ReconciliationStatus.PENDING,
                ReconciliationStatus.RESOLVING,
                ReconciliationStatus.RESOLVED,
            }:
                raise ConflictError("reconciliation was already resolved differently")
            if str(row["resolution"]) and str(row["resolution"]) != decision:
                raise ConflictError("reconciliation resolution is already in progress")
            if str(row["current_workspace_digest"]) != observed_workspace_digest:
                raise ConflictError("observed workspace digest is stale")
            if str(row["status"]) == ReconciliationStatus.RESOLVED:
                record = _reconciliation(row)
                self._insert_mutation_receipt(
                    connection,
                    idempotency_key=idempotency_key,
                    operation=operation,
                    request_digest=request_digest,
                    response={"reconciliation": record.as_dict()},
                    status_code=200,
                    created_at=utc_now(),
                )
                return record, True
            if checkpoint is not None:
                existing_checkpoint = connection.execute(
                    """
                    SELECT checkpoint_id FROM checkpoints
                    WHERE checkpoint_id = ?
                    """,
                    (checkpoint.checkpoint_id,),
                ).fetchone()
                if existing_checkpoint is None:
                    self.add_checkpoint(checkpoint)
            now = utc_now()
            resolved_audit = self._normalize_reconciled_command_topology(
                connection,
                row,
                audit,
                now,
                decision,
            )
            connection.execute(
                """
                UPDATE reconciliations SET status = ?, resolution = ?,
                    audit_json = ?, resolved_at = ?
                WHERE reconciliation_id = ?
                """,
                (
                    ReconciliationStatus.RESOLVED,
                    decision,
                    _dump(resolved_audit),
                    now,
                    reconciliation_id,
                ),
            )
            lifecycle = Lifecycle.RUNNING
            if decision == ReconciliationDecision.STOP:
                lifecycle = Lifecycle.STOPPED
            connection.execute(
                """
                UPDATE sessions SET lifecycle = ?, attention = ?,
                    updated_at = ? WHERE session_id = ?
                """,
                (
                    lifecycle,
                    Attention.IDLE,
                    now,
                    str(row["session_id"]),
                ),
            )
            event = connection.execute(
                """
                SELECT sequence FROM events
                WHERE session_id = ? AND event_id = ?
                    AND event_type = 'reconciliation.resolved'
                """,
                (str(row["session_id"]), reconciliation_id),
            ).fetchone()
            if event is None:
                sequence_row = connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) AS sequence
                    FROM events WHERE session_id = ?
                    """,
                    (str(row["session_id"]),),
                ).fetchone()
                sequence = 1
                if sequence_row is not None:
                    sequence = int(sequence_row["sequence"]) + 1
                connection.execute(
                    """
                    INSERT INTO events VALUES (
                        ?, ?, ?, 'reconciliation.resolved', '', '',
                        'resolved', ?, '', '', ?
                    )
                    """,
                    (
                        str(row["session_id"]),
                        sequence,
                        reconciliation_id,
                        _dump(
                            {
                                "reconciliation_id": reconciliation_id,
                                "command_id": str(row["command_id"]),
                                "decision": decision,
                                "workspace_digest": observed_workspace_digest,
                                "discovery_checkpoint_id": str(
                                    resolved_audit.get(
                                        "discovery_checkpoint_id",
                                        "",
                                    )
                                ),
                                "resolution_checkpoint_id": str(
                                    resolved_audit.get(
                                        "resolution_checkpoint_id",
                                        "",
                                    )
                                ),
                                "resolution_workspace_digest": str(
                                    resolved_audit.get(
                                        "resolution_workspace_digest",
                                        "",
                                    )
                                ),
                                "topology_receipt": _object_or_empty(
                                    resolved_audit.get("topology_receipt")
                                ),
                            }
                        ),
                        now,
                    ),
                )
            resolved = connection.execute(
                """
                SELECT * FROM reconciliations
                WHERE reconciliation_id = ?
                """,
                (reconciliation_id,),
            ).fetchone()
            if resolved is None:
                raise RuntimeError("reconciliation resolution was not recorded")
            record = _reconciliation(resolved)
            response = {"reconciliation": record.as_dict()}
            self._insert_mutation_receipt(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_digest=request_digest,
                response=response,
                status_code=200,
                created_at=now,
            )
            return record, True

    def _normalize_reconciled_command_topology(
        self,
        connection: sqlite3.Connection,
        reconciliation: sqlite3.Row,
        audit: dict[str, Any],
        now: str,
        decision: str,
    ) -> dict[str, Any]:
        command_id = str(reconciliation["command_id"])
        session_id = str(reconciliation["session_id"])
        resolved_audit = dict(audit)
        identity = _object_or_empty(resolved_audit.get("dispatch_identity"))
        attempt_id = str(identity.get("attempt_id", ""))
        dispatches = connection.execute(
            """
            SELECT * FROM command_dispatches
            WHERE session_id = ? AND command_id = ?
            AND crossed_boundary = 1 AND state = 'ambiguous'
            ORDER BY created_at, attempt_id
            """,
            (session_id, command_id),
        ).fetchall()
        if attempt_id:
            dispatches = [
                item for item in dispatches if str(item["attempt_id"]) == attempt_id
            ]
        if len(dispatches) != 1:
            raise ConflictError(
                "reconciliation does not bind one ambiguous provider dispatch"
            )
        dispatch = dispatches[0]
        attempt_id = str(dispatch["attempt_id"])
        turn_id = str(dispatch["turn_id"])
        if identity:
            if str(identity.get("turn_id", "")) != turn_id:
                raise ConflictError("reconciliation turn identity changed")
            if str(identity.get("checkpoint_id", "")) != str(dispatch["checkpoint_id"]):
                raise ConflictError("reconciliation checkpoint identity changed")

        attempt = connection.execute(
            """
            SELECT * FROM provider_attempts
            WHERE attempt_id = ? AND session_id = ?
            """,
            (attempt_id, session_id),
        ).fetchone()
        if attempt is None:
            raise ConflictError("reconciliation provider attempt is missing")
        turn = connection.execute(
            """
            SELECT * FROM turns
            WHERE turn_id = ? AND attempt_id = ? AND session_id = ?
            """,
            (turn_id, attempt_id, session_id),
        ).fetchone()
        if turn is None:
            raise ConflictError("reconciliation turn is missing")

        lease_id = str(identity.get("lease_id", ""))
        lease_rows = connection.execute(
            """
            SELECT * FROM process_leases
            WHERE session_id = ? AND command_id = ? AND attempt_id = ?
            ORDER BY created_at, lease_id
            """,
            (session_id, command_id, attempt_id),
        ).fetchall()
        if lease_id:
            lease_rows = [
                item for item in lease_rows if str(item["lease_id"]) == lease_id
            ]
            if len(lease_rows) != 1:
                raise ConflictError("reconciliation process lease identity changed")
        active_lease_rows = [
            item
            for item in lease_rows
            if str(item["state"]) in {"reserved", "active", "recovery-blocked"}
        ]
        if len(active_lease_rows) > 1:
            raise ConflictError("reconciliation has multiple active process leases")
        lease_receipts: list[dict[str, Any]] = []
        for lease in lease_rows:
            prior_state = str(lease["state"])
            if prior_state in {"reserved", "active", "recovery-blocked"}:
                connection.execute(
                    """
                    UPDATE process_leases SET state = 'released',
                        expires_at = ?, updated_at = ?
                    WHERE lease_id = ?
                    AND state IN ('reserved', 'active', 'recovery-blocked')
                    """,
                    (now, now, str(lease["lease_id"])),
                )
            lease_receipts.append(
                {
                    "lease_id": str(lease["lease_id"]),
                    "attempt_id": attempt_id,
                    "worker_incarnation": str(lease["worker_incarnation"]),
                    "pid": int(lease["pid"]),
                    "pid_start": str(lease["pid_start"]),
                    "prior_state": prior_state,
                    "state": "released",
                }
            )

        completion = self._reconciled_completion_evidence(
            connection,
            reconciliation,
            attempt,
            turn_id,
            resolved_audit,
            decision,
        )
        settled = self._settle_reconciled_topology(
            connection,
            command_id,
            turn_id,
            attempt,
            completion,
            now,
        )
        settled_state = str(settled["settled_state"])
        envelope_state = str(settled["envelope_state"])
        guard_reason = str(settled["guard_reason"])
        prior_guard_reason = str(settled["prior_guard_reason"])
        resolved_audit["dispatch_identity"] = {
            "attempt_id": attempt_id,
            "turn_id": turn_id,
            "checkpoint_id": str(dispatch["checkpoint_id"]),
            "lease_id": "",
            "worker_incarnation": "",
            "pid": 0,
            "pid_start": "",
        }
        if lease_receipts:
            first_lease = lease_receipts[0]
            resolved_audit["dispatch_identity"].update(
                {
                    "lease_id": str(first_lease["lease_id"]),
                    "worker_incarnation": str(first_lease["worker_incarnation"]),
                    "pid": int(first_lease["pid"]),
                    "pid_start": str(first_lease["pid_start"]),
                }
            )
        resolved_audit["topology_receipt"] = {
            "schema": "p13i/agent-harness/reconciliation-topology-receipt/v1",
            "reconciliation_id": str(reconciliation["reconciliation_id"]),
            "session_id": session_id,
            "command_id": command_id,
            "attempt_id": attempt_id,
            "turn_id": turn_id,
            "checkpoint_id": str(dispatch["checkpoint_id"]),
            "attempt_state": settled_state,
            "turn_state": settled_state,
            "dispatch_state": settled_state,
            "envelope_state": envelope_state,
            "guard_reason": guard_reason,
            "leases": lease_receipts,
            "recorded_at": now,
        }
        if completion is not None:
            resolved_audit["topology_receipt"].update(
                {
                    "command_status": str(CommandStatus.COMPLETE),
                    "prior_guard_reason": prior_guard_reason,
                    "completion_evidence": dict(completion),
                }
            )
        return resolved_audit

    def _settle_reconciled_topology(
        self,
        connection: sqlite3.Connection,
        command_id: str,
        turn_id: str,
        attempt: sqlite3.Row,
        completion: dict[str, Any] | None,
        now: str,
    ) -> dict[str, Any]:
        attempt_id = str(attempt["attempt_id"])
        settled_state = "ambiguous"
        if completion is not None:
            settled_state = "complete"
        connection.execute(
            """
            UPDATE provider_attempts SET status = ?,
                ended_at = CASE WHEN ended_at = '' THEN ? ELSE ended_at END
            WHERE attempt_id = ?
            """,
            (settled_state, now, attempt_id),
        )
        connection.execute(
            """
            UPDATE turns SET status = ?,
                completed_at = CASE
                    WHEN completed_at = '' THEN ? ELSE completed_at END
            WHERE turn_id = ?
            """,
            (settled_state, now, turn_id),
        )
        connection.execute(
            """
            UPDATE command_dispatches SET state = ?, updated_at = ?
            WHERE attempt_id = ?
            """,
            (settled_state, now, attempt_id),
        )
        envelope = connection.execute(
            """
            SELECT command_id, guard_reason FROM command_envelopes
            WHERE command_id = ?
            """,
            (command_id,),
        ).fetchone()
        envelope_state = ""
        guard_reason = ""
        prior_guard_reason = ""
        if envelope is not None:
            prior_guard_reason = str(envelope["guard_reason"])
            envelope_state = "paused"
            guard_reason = "ambiguous-provider-dispatch"
            if completion is not None:
                envelope_state = "complete"
                guard_reason = ""
            connection.execute(
                """
                UPDATE command_envelopes SET state = ?, guard_reason = ?,
                    updated_at = ? WHERE command_id = ?
                """,
                (envelope_state, guard_reason, now, command_id),
            )
        if completion is not None:
            connection.execute(
                """
                UPDATE commands SET status = ?, result_json = ?, updated_at = ?
                WHERE command_id = ?
                """,
                (
                    CommandStatus.COMPLETE,
                    _dump(_reconciled_command_result(attempt, completion)),
                    now,
                    command_id,
                ),
            )
        return {
            "settled_state": settled_state,
            "envelope_state": envelope_state,
            "guard_reason": guard_reason,
            "prior_guard_reason": prior_guard_reason,
        }

    def _reconciled_completion_evidence(
        self,
        connection: sqlite3.Connection,
        reconciliation: sqlite3.Row,
        attempt: sqlite3.Row,
        turn_id: str,
        resolved_audit: dict[str, Any],
        decision: str,
    ) -> dict[str, Any] | None:
        """Project a resolved dispatch onto the completion the log already proves.

        Returns ``None`` unless the accepted turn left an observable
        terminal record and moved the workspace. Every field is read
        back from the recorded event log; nothing here reconstructs an
        unobserved outcome.
        """
        if decision != ReconciliationDecision.ACCEPT_CURRENT:
            return None
        native_session_id = str(attempt["native_session_id"])
        if not native_session_id:
            return None
        checkpoint_id = str(resolved_audit.get("resolution_checkpoint_id", ""))
        workspace_digest = str(resolved_audit.get("resolution_workspace_digest", ""))
        if not checkpoint_id or not _is_digest(workspace_digest):
            return None
        session_id = str(reconciliation["session_id"])
        command_id = str(reconciliation["command_id"])
        # The checkpoint the result points at has to be a real anchor
        # in this session, not just an id the audit happens to carry.
        checkpoint = connection.execute(
            """
            SELECT base_commit, patch_digest, untracked_digest FROM checkpoints
            WHERE checkpoint_id = ? AND session_id = ?
            """,
            (checkpoint_id, session_id),
        ).fetchone()
        if checkpoint is None:
            return None
        # A provider can end its turn cleanly and say so without the
        # implementation ever landing. Terminal provider status is not
        # implementation completion, so the accepted workspace has to
        # differ in material from the tree the dispatch started against.
        pre_dispatch = connection.execute(
            """
            SELECT base_commit, patch_digest, untracked_digest FROM checkpoints
            WHERE checkpoint_id = ? AND session_id = ?
            """,
            (str(reconciliation["pre_dispatch_checkpoint_id"]), session_id),
        ).fetchone()
        if pre_dispatch is None:
            return None
        if not _checkpoint_material_differs(pre_dispatch, checkpoint):
            return None
        command = connection.execute(
            "SELECT * FROM commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        if command is None or str(command["status"]) not in {
            CommandStatus.DISPATCHING,
            CommandStatus.FAILED,
        }:
            return None
        prior_result_json = str(command["result_json"])
        prior_result = _load_object(prior_result_json)
        bound_reconciliation = str(prior_result.get("reconciliation_id", ""))
        if bound_reconciliation and bound_reconciliation != str(
            reconciliation["reconciliation_id"]
        ):
            return None
        completions = connection.execute(
            """
            SELECT sequence FROM events
            WHERE session_id = ? AND turn_id = ?
                AND event_type = 'turn.completed' AND status = 'complete'
            ORDER BY sequence
            """,
            (session_id, turn_id),
        ).fetchall()
        if len(completions) != 1:
            return None
        # A turn that also failed, was interrupted, reported a provider
        # error, or completed under a non-complete status did not end
        # the way the single completion suggests. Providers can emit a
        # complete-looking resume hint and then fail the turn, so one
        # matching completion is not on its own proof of the outcome.
        contradictions = connection.execute(
            """
            SELECT COUNT(*) AS total FROM events
            WHERE session_id = ? AND turn_id = ?
                AND (
                    event_type IN ('turn.failed', 'turn.interrupted', 'provider.error')
                    OR (event_type = 'turn.completed' AND status != 'complete')
                )
            """,
            (session_id, turn_id),
        ).fetchone()
        if contradictions is None or int(contradictions["total"]) != 0:
            return None
        messages = connection.execute(
            """
            SELECT sequence, text FROM events
            WHERE session_id = ? AND turn_id = ?
                AND event_type = 'agent.message' AND text != ''
            ORDER BY sequence
            """,
            (session_id, turn_id),
        ).fetchall()
        if not messages:
            return None
        completion_sequence = int(completions[0]["sequence"])
        final_message = messages[-1]
        message_sequence = int(final_message["sequence"])
        if message_sequence > completion_sequence:
            return None
        return {
            "reconciliation_id": str(reconciliation["reconciliation_id"]),
            "native_session_id": native_session_id,
            "turn_id": turn_id,
            "attempt_id": str(attempt["attempt_id"]),
            "final_message_sequence": message_sequence,
            "final_message_sha256": hashlib.sha256(
                str(final_message["text"]).encode("utf-8")
            ).hexdigest(),
            "turn_completed_sequence": completion_sequence,
            "checkpoint_id": checkpoint_id,
            "workspace_material_digest": workspace_digest,
            # Keep why the command was failed, not what it said: the
            # code is a closed vocabulary, the digest proves the trace
            # without republishing provider or guard prose.
            "prior_code": str(prior_result.get("code", "")),
            "prior_result_sha256": hashlib.sha256(
                prior_result_json.encode("utf-8")
            ).hexdigest(),
        }

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

    def approval(self, approval_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("approval")
        return {
            "approval_id": str(row["approval_id"]),
            "session_id": str(row["session_id"]),
            "turn_id": str(row["turn_id"]),
            "provider_request_id": str(row["provider_request_id"]),
            "kind": str(row["kind"]),
            "prompt": str(row["prompt"]),
            "choices": json.loads(row["choices_json"]),
            "status": str(row["status"]),
            "decision": _load_object(row["decision_json"]),
            "created_at": str(row["created_at"]),
            "resolved_at": str(row["resolved_at"]),
        }

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
            authorization_row = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM xhigh_authorization_receipts
                WHERE session_id = ? AND consumed_at = '' AND expires_at > ?
                """,
                (session_id, utc_now()),
            ).fetchone()
        xhigh_authorizations = 0
        if authorization_row is not None:
            xhigh_authorizations = int(authorization_row["count"])
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
            "xhigh_authorizations": xhigh_authorizations,
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
        extensions.pop("allow_xhigh_once", None)
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE session_safety SET extensions_json = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    _dump(extensions),
                    utc_now(),
                    session_id,
                ),
            )
        return self.session_safety(session_id)

    def create_xhigh_authorization(
        self,
        session_id: str,
        command_id: str,
        provider: str,
        *,
        authorization_request_digest: str,
        idempotency_key: str,
        expires_at: str,
    ) -> dict[str, Any]:
        if not provider.strip():
            raise ValueError("xhigh authorization provider is unsupported")
        if not idempotency_key:
            raise ValueError("xhigh authorization idempotency key is required")
        now = utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM xhigh_authorization_receipts
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["session_id"]) != session_id
                    or str(existing["command_id"]) != command_id
                    or str(existing["provider"]) != provider
                    or str(existing["authorization_request_digest"])
                    != authorization_request_digest
                ):
                    raise ConflictError("xhigh authorization key was reused")
                return dict(existing)
            command = connection.execute(
                "SELECT * FROM commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if command is None:
                raise NotFoundError("command")
            if str(command["session_id"]) != session_id:
                raise ConflictError("xhigh authorization command changed session")
            if str(command["status"]) not in {
                CommandStatus.QUEUED,
                CommandStatus.AWAITING_XHIGH_AUTHORIZATION,
                CommandStatus.DISPATCHING,
            }:
                raise ConflictError("xhigh authorization command is not active")
            command_payload = _load_object(str(command["payload_json"]))
            requested_effort = (
                str(command_payload.get("effort", "")).strip().casefold()
            )
            if not effort_requires_xhigh_authorization(requested_effort):
                raise ConflictError("xhigh authorization command effort changed")
            requested_provider = str(command_payload.get("provider", ""))
            if requested_provider and requested_provider != provider:
                raise ConflictError("xhigh authorization command provider changed")
            command_authorization = connection.execute(
                """
                SELECT authorization_id FROM xhigh_authorization_receipts
                WHERE command_id = ?
                """,
                (command_id,),
            ).fetchone()
            if command_authorization is not None:
                raise ConflictError("xhigh command already has an authorization")
            command_request_digest = normalized_digest(command_payload)
            authorization_id = new_uuid()
            connection.execute(
                """
                INSERT INTO xhigh_authorization_receipts VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?
                )
                """,
                (
                    authorization_id,
                    session_id,
                    command_id,
                    provider,
                    requested_effort,
                    command_request_digest,
                    authorization_request_digest,
                    idempotency_key,
                    expires_at,
                    now,
                ),
            )
            if str(command["status"]) == CommandStatus.AWAITING_XHIGH_AUTHORIZATION:
                connection.execute(
                    """
                    UPDATE commands SET status = ?, updated_at = ?
                    WHERE command_id = ? AND status = ?
                    """,
                    (
                        CommandStatus.QUEUED,
                        now,
                        command_id,
                        CommandStatus.AWAITING_XHIGH_AUTHORIZATION,
                    ),
                )
            row = connection.execute(
                """
                SELECT * FROM xhigh_authorization_receipts
                WHERE authorization_id = ?
                """,
                (authorization_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("xhigh authorization was not recorded")
        return dict(row)

    def xhigh_authorization(self, command_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM xhigh_authorization_receipts
                WHERE command_id = ? AND consumed_at = '' AND expires_at > ?
                """,
                (command_id, utc_now()),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

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
                raise ConflictError("session execution profile is not claimed")
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
            "child_agents": 0,
            "dollars": 0.0,
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "exact_tokens": False,
            "exact_dollars": False,
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
            row = connection.execute(
                "SELECT * FROM command_envelopes WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is not None:
                if str(row["session_id"]) != session_id:
                    raise ConflictError("command envelope session changed")
                if str(row["profile"]) != profile:
                    raise ConflictError("command envelope profile changed")
                if _load_object(str(row["limits_json"])) != limits:
                    raise ConflictError("command envelope limits changed")
        return self.command_envelope(command_id)

    def command_envelope(self, command_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT e.*, c.payload_json FROM command_envelopes AS e
                JOIN commands AS c ON c.command_id = e.command_id
                WHERE e.command_id = ?
                """,
                (command_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("command envelope")
        command_payload = _load_object(str(row["payload_json"]))
        requested_limits = command_payload.get("safety_limits")
        if not isinstance(requested_limits, dict):
            requested_limits = {}
        return {
            "command_id": str(row["command_id"]),
            "session_id": str(row["session_id"]),
            "provider": str(row["provider"]),
            "profile": str(row["profile"]),
            "state": str(row["state"]),
            "limits": _load_object(row["limits_json"]),
            "requested_limits": requested_limits,
            "requested_limits_digest": normalized_digest(requested_limits),
            "consumption": _load_object(row["consumption_json"]),
            "guard_reason": str(row["guard_reason"]),
            "recovery_stage": int(row["recovery_stage"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def create_child_launch_gate(
        self,
        command_id: str,
        session_id: str,
        permit_limit: int,
    ) -> dict[str, Any]:
        if permit_limit < 0:
            raise ValueError("child launch permit limit must not be negative")
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO child_launch_gates(
                    command_id, session_id, permit_limit, consumed,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 0, ?, ?)
                """,
                (command_id, session_id, permit_limit, now, now),
            )
            row = connection.execute(
                "SELECT * FROM child_launch_gates WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("child launch gate was not created")
            if str(row["session_id"]) != session_id:
                raise ConflictError("child launch gate session changed")
            if int(row["permit_limit"]) != permit_limit:
                raise ConflictError("child launch gate permit limit changed")
        return self.child_launch_gate(command_id)

    def child_launch_gate(self, command_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM child_launch_gates WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("child launch gate")
        return {
            "command_id": str(row["command_id"]),
            "session_id": str(row["session_id"]),
            "permit_limit": int(row["permit_limit"]),
            "consumed": int(row["consumed"]),
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

    def reserve_route_admission(
        self,
        command_id: str,
        provider: str,
        profile: str,
        *,
        effort: str = "",
        attempt_id: str = "",
        worker_incarnation: str,
        goal_id: str,
        max_concurrency: int,
        lease_expires_at: str,
    ) -> dict[str, Any]:
        if max_concurrency < 1:
            raise ValueError("route admission concurrency must be positive")
        now = utc_now()
        lease_id = ""
        with self.transaction() as connection:
            envelope = connection.execute(
                "SELECT * FROM command_envelopes WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if envelope is None:
                raise NotFoundError("command envelope")
            if str(envelope["profile"]) != profile:
                raise ConflictError("route admission profile changed")
            if str(envelope["state"]) not in {"reserved", "recovering"}:
                raise ConflictError("route admission envelope is not reservable")
            session = connection.execute(
                "SELECT goal_id, worktree FROM sessions WHERE session_id = ?",
                (str(envelope["session_id"]),),
            ).fetchone()
            if session is None:
                raise NotFoundError("session")
            worker = connection.execute(
                """
                SELECT incarnation FROM workers WHERE session_id = ?
                """,
                (str(envelope["session_id"]),),
            ).fetchone()
            if worker is None or str(worker["incarnation"]) != worker_incarnation:
                raise WorkerOwnershipLostError(
                    "worker incarnation lost route-admission ownership"
                )
            if goal_id and str(session["goal_id"]) != goal_id:
                raise ConflictError("route admission goal changed")
            if profile == "unattended":
                provider_row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM command_envelopes AS active_envelope
                    JOIN commands AS active_command
                        ON active_command.command_id = active_envelope.command_id
                    WHERE active_envelope.provider = ?
                    AND active_envelope.profile = 'unattended'
                    AND active_envelope.state IN (
                        'reserved', 'running', 'fault-ready', 'recovering'
                    )
                    AND active_command.status IN (?, ?, ?)
                    AND active_envelope.command_id != ?
                    """,
                    (
                        provider,
                        CommandStatus.QUEUED,
                        CommandStatus.AWAITING_XHIGH_AUTHORIZATION,
                        CommandStatus.DISPATCHING,
                        command_id,
                    ),
                ).fetchone()
                if provider_row is not None and int(provider_row["count"]) >= 1:
                    return {
                        "admitted": False,
                        "reason": "provider-concurrency",
                        "lease_id": "",
                    }
            if goal_id:
                goal_row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM command_envelopes AS active_envelope
                    JOIN sessions AS active_session
                        ON active_session.session_id = active_envelope.session_id
                    JOIN commands AS active_command
                        ON active_command.command_id = active_envelope.command_id
                    WHERE active_session.goal_id = ?
                    AND active_envelope.state IN (
                        'reserved', 'running', 'fault-ready', 'recovering'
                    )
                    AND active_envelope.provider != ''
                    AND active_command.status IN (?, ?, ?)
                    AND active_envelope.command_id != ?
                    """,
                    (
                        goal_id,
                        CommandStatus.QUEUED,
                        CommandStatus.AWAITING_XHIGH_AUTHORIZATION,
                        CommandStatus.DISPATCHING,
                        command_id,
                    ),
                ).fetchone()
                if goal_row is not None and int(goal_row["count"]) >= max_concurrency:
                    return {
                        "admitted": False,
                        "reason": "goal-concurrency",
                        "lease_id": "",
                    }
            if profile != "interactive" and effort_requires_xhigh_authorization(
                effort
            ):
                command = connection.execute(
                    "SELECT payload_json FROM commands WHERE command_id = ?",
                    (command_id,),
                ).fetchone()
                if command is None:
                    raise NotFoundError("command")
                command_payload = _load_object(str(command["payload_json"]))
                command_request_digest = normalized_digest(command_payload)
                authorization = connection.execute(
                    """
                    SELECT authorization_id
                    FROM xhigh_authorization_receipts
                    WHERE command_id = ? AND provider = ?
                    AND lower(trim(effort)) = ?
                    AND command_request_digest = ? AND consumed_at = ''
                    AND expires_at > ?
                    """,
                    (command_id, provider, effort, command_request_digest, now),
                ).fetchone()
                if authorization is None or not attempt_id:
                    return {
                        "admitted": False,
                        "reason": "xhigh-authorization",
                        "lease_id": "",
                    }
                consumed = connection.execute(
                    """
                    UPDATE xhigh_authorization_receipts
                    SET consumed_attempt_id = ?, consumed_at = ?
                    WHERE authorization_id = ? AND consumed_at = ''
                    """,
                    (
                        attempt_id,
                        now,
                        str(authorization["authorization_id"]),
                    ),
                )
                if consumed.rowcount != 1:
                    return {
                        "admitted": False,
                        "reason": "xhigh-authorization",
                        "lease_id": "",
                    }
            self._consume_reserved_dispatch_transition(
                connection,
                str(envelope["session_id"]),
                command_id,
                Path(str(session["worktree"])),
            )
            if attempt_id:
                self._mark_provider_boundary(connection, attempt_id, now)
            connection.execute(
                """
                UPDATE command_envelopes SET provider = ?, state = 'running',
                    updated_at = ? WHERE command_id = ?
                """,
                (provider, now, command_id),
            )
            if profile != "interactive":
                lease_id = new_uuid()
                connection.execute(
                    """
                    INSERT INTO process_leases(
                        lease_id, session_id, command_id, attempt_id,
                        worker_incarnation,
                        provider, profile, pid, pid_start, state,
                        expires_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, '', 'reserved', ?, ?, ?)
                    """,
                    (
                        lease_id,
                        str(envelope["session_id"]),
                        command_id,
                        attempt_id,
                        worker_incarnation,
                        provider,
                        profile,
                        lease_expires_at,
                        now,
                        now,
                    ),
                )
        return {
            "admitted": True,
            "reason": "",
            "lease_id": lease_id,
        }

    def active_unattended_provider_count(self, provider: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM command_envelopes AS envelope
                JOIN commands AS command
                    ON command.command_id = envelope.command_id
                WHERE envelope.provider = ?
                AND envelope.profile = 'unattended'
                AND envelope.state IN (
                    'reserved', 'running', 'fault-ready', 'recovering'
                )
                AND command.status IN (?, ?, ?)
                """,
                (
                    provider,
                    CommandStatus.QUEUED,
                    CommandStatus.AWAITING_XHIGH_AUTHORIZATION,
                    CommandStatus.DISPATCHING,
                ),
            ).fetchone()
        return int(row["count"])

    def active_goal_command_count(self, goal_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM command_envelopes AS envelope
                JOIN sessions AS session
                    ON session.session_id = envelope.session_id
                JOIN commands AS command
                    ON command.command_id = envelope.command_id
                WHERE session.goal_id = ?
                AND envelope.state IN (
                    'reserved', 'running', 'fault-ready', 'recovering'
                )
                AND envelope.provider != ''
                AND command.status IN (?, ?, ?)
                """,
                (
                    goal_id,
                    CommandStatus.QUEUED,
                    CommandStatus.AWAITING_XHIGH_AUTHORIZATION,
                    CommandStatus.DISPATCHING,
                ),
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
        return [self.command_envelope(str(row["command_id"])) for row in rows]

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

    def prepare_context_delivery(
        self,
        session_id: str,
        provider: str,
        context_digest: str,
        checkpoint_id: str,
        command_id: str,
        attempt_id: str,
        payload_digest: str,
        *,
        transport: str = "context-package",
    ) -> dict[str, Any]:
        if transport not in {"context-package", "native-resume"}:
            raise ValueError("context delivery transport is unsupported")
        with self.transaction() as connection:
            prior_command_rows = connection.execute(
                """
                SELECT * FROM context_deliveries
                WHERE session_id = ? AND provider = ? AND command_id = ?
                    AND attempt_id != ?
                """,
                (session_id, provider, command_id, attempt_id),
            ).fetchall()
            for prior in prior_command_rows:
                dispatch = connection.execute(
                    """
                    SELECT crossed_boundary, state FROM command_dispatches
                    WHERE attempt_id = ?
                    """,
                    (str(prior["attempt_id"]),),
                ).fetchone()
                if not self._context_delivery_known_undelivered(
                    dispatch,
                    prior,
                ):
                    raise ConflictError(
                        "prior context delivery for this command is ambiguous"
                    )
                connection.execute(
                    """
                    UPDATE context_deliveries SET state = 'superseded'
                    WHERE attempt_id = ?
                    """,
                    (str(prior["attempt_id"]),),
                )
            existing_rows = connection.execute(
                """
                SELECT * FROM context_deliveries
                WHERE session_id = ? AND provider = ? AND context_digest = ?
                ORDER BY delivered_at, attempt_id
                """,
                (session_id, provider, context_digest),
            ).fetchall()
            for existing in existing_rows:
                if str(existing["attempt_id"]) == attempt_id:
                    if str(existing["transport"]) != transport:
                        raise ConflictError("context delivery transport changed")
                    return dict(existing)
                if str(existing["state"]) == "delivered":
                    raise ConflictError(
                        "context package was already delivered without native resume"
                    )
                dispatch = connection.execute(
                    """
                    SELECT crossed_boundary, state FROM command_dispatches
                    WHERE attempt_id = ?
                    """,
                    (str(existing["attempt_id"]),),
                ).fetchone()
                if not self._context_delivery_known_undelivered(
                    dispatch,
                    existing,
                ):
                    raise ConflictError(
                        "context package delivery is ambiguous; reconcile first"
                    )
                connection.execute(
                    """
                    UPDATE context_deliveries SET state = 'superseded'
                    WHERE attempt_id = ?
                    """,
                    (str(existing["attempt_id"]),),
                )
            connection.execute(
                """
                INSERT INTO context_deliveries(
                    session_id, provider, context_digest, checkpoint_id,
                    delivered_at, command_id, attempt_id, state,
                    payload_digest, accepted_at, transport
                ) VALUES (?, ?, ?, ?, '', ?, ?, 'prepared', ?, '', ?)
                """,
                (
                    session_id,
                    provider,
                    context_digest,
                    checkpoint_id,
                    command_id,
                    attempt_id,
                    payload_digest,
                    transport,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM context_deliveries
                WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("context delivery preparation was not recorded")
            return dict(row)

    def accept_context_delivery(
        self,
        session_id: str,
        provider: str,
        context_digest: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE context_deliveries SET state = 'delivered',
                    delivered_at = ?, accepted_at = ?
                WHERE attempt_id = ? AND session_id = ? AND provider = ?
                    AND context_digest = ? AND state = 'prepared'
                """,
                (
                    now,
                    now,
                    attempt_id,
                    session_id,
                    provider,
                    context_digest,
                ),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    """
                    SELECT * FROM context_deliveries
                    WHERE attempt_id = ? AND session_id = ? AND provider = ?
                        AND context_digest = ?
                    """,
                    (attempt_id, session_id, provider, context_digest),
                ).fetchone()
                if row is None:
                    raise ConflictError("context delivery acceptance is stale")
                if str(row["state"]) != "delivered":
                    raise ConflictError("context delivery acceptance is invalid")
            row = connection.execute(
                """
                SELECT * FROM context_deliveries
                WHERE attempt_id = ? AND session_id = ? AND provider = ?
                    AND context_digest = ?
                """,
                (attempt_id, session_id, provider, context_digest),
            ).fetchone()
            if row is None:
                raise RuntimeError("context delivery acceptance was not recorded")
            return dict(row)

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
                INSERT INTO process_leases(
                    lease_id, session_id, command_id, attempt_id,
                    worker_incarnation,
                    provider, profile, pid, pid_start, state,
                    expires_at, created_at, updated_at
                ) VALUES (?, ?, '', '', '', ?, ?, 0, '', 'reserved', ?, ?, ?)
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
            "command_id": str(row["command_id"]),
            "attempt_id": str(row["attempt_id"]),
            "worker_incarnation": str(row["worker_incarnation"]),
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
                WHERE state IN ('reserved', 'active', 'recovery-blocked')
                ORDER BY created_at, lease_id
                """
            ).fetchall()
        return [self.process_lease(str(row["lease_id"])) for row in rows]

    def all_process_leases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT lease_id FROM process_leases
                ORDER BY created_at, lease_id
                """
            ).fetchall()
        return [self.process_lease(str(row["lease_id"])) for row in rows]

    def process_leases(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT lease_id FROM process_leases
                WHERE session_id = ?
                ORDER BY created_at, lease_id
                """,
                (session_id,),
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
            raise ConflictError("idempotency key was already used for another mutation")
        return {
            "response": _load_object(row["response_json"]),
            "status_code": int(row["status_code"]),
        }

    def idempotent_mutation(
        self,
        idempotency_key: str,
        operation: str,
        request_digest: str,
        mutate: Callable[[], dict[str, Any]],
        status_code: int,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM mutation_receipts
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                if (
                    str(row["operation"]) != operation
                    or str(row["request_digest"]) != request_digest
                ):
                    raise ConflictError(
                        "idempotency key was already used for another mutation"
                    )
                return {
                    "response": _load_object(row["response_json"]),
                    "status_code": int(row["status_code"]),
                }
            response = mutate()
            connection.execute(
                """
                INSERT INTO mutation_receipts
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
            return {
                "response": response,
                "status_code": status_code,
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
                "sample_id": str(row["sample_id"]),
                "provider": str(row["provider"]),
                "observed_at": str(row["observed_at"]),
                "binding_percent": row["binding_percent"],
                "credits_engaged": bool(row["credits_engaged"]),
                "payload": _load_object(row["payload_json"]),
            }
        return result

    def latest_operator_usage_attestation(
        self,
        provider: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM usage_samples
                WHERE provider = ?
                AND json_extract(payload_json, '$.payload.source')
                    = 'operator-attestation'
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                (provider,),
            ).fetchone()
        if row is None:
            return None
        return {
            "sample_id": str(row["sample_id"]),
            "provider": str(row["provider"]),
            "observed_at": str(row["observed_at"]),
            "binding_percent": row["binding_percent"],
            "credits_engaged": bool(row["credits_engaged"]),
            "payload": _load_object(row["payload_json"]),
        }

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

    def routing_decisions(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM routing_decisions
                WHERE session_id = ? ORDER BY created_at, decision_id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "decision_id": str(row["decision_id"]),
                "turn_id": str(row["turn_id"]),
                "provider": str(row["provider"]),
                "model": str(row["model"]),
                "effort": str(row["effort"]),
                "payload": _load_object(row["payload_json"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def create_proof_snapshot(
        self,
        session_id: str,
        through_sequence: int,
        payload: dict[str, Any],
        digest: str,
    ) -> dict[str, Any]:
        snapshot_id = new_uuid()
        created_at = utc_now()
        cutoff = datetime.datetime.now(datetime.UTC)
        cutoff -= datetime.timedelta(
            hours=PROOF_SNAPSHOT_RETENTION_HOURS,
        )
        with self.transaction() as connection:
            connection.execute(
                """
                DELETE FROM proof_snapshots
                WHERE session_id = ? AND created_at < ?
                """,
                (session_id, cutoff.isoformat()),
            )
            retained = connection.execute(
                """
                SELECT COUNT(*) AS count FROM proof_snapshots
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if int(retained["count"]) >= PROOF_SNAPSHOT_MAX_PER_SESSION:
                raise ConflictError("proof snapshot retention quota is full")
            connection.execute(
                "INSERT INTO proof_snapshots VALUES (?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    session_id,
                    through_sequence,
                    _dump(payload),
                    digest,
                    created_at,
                ),
            )
        return {
            "snapshot_id": snapshot_id,
            "session_id": session_id,
            "through_sequence": through_sequence,
            "payload": payload,
            "digest": digest,
            "created_at": created_at,
        }

    def proof_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM proof_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("proof snapshot")
        return {
            "snapshot_id": str(row["snapshot_id"]),
            "session_id": str(row["session_id"]),
            "through_sequence": int(row["through_sequence"]),
            "payload": _load_object(row["payload_json"]),
            "digest": str(row["digest"]),
            "created_at": str(row["created_at"]),
        }

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

    def worker_owned(self, session_id: str, incarnation: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1 FROM workers
                WHERE session_id = ? AND incarnation = ?
                """,
                (session_id, incarnation),
            ).fetchone()
        return row is not None

    def worker_registered(self, session_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM workers WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row is not None

    def remove_worker(self, session_id: str, incarnation: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                DELETE FROM workers
                WHERE session_id = ? AND incarnation = ?
                """,
                (session_id, incarnation),
            )

    def retire_worker(
        self,
        session_id: str,
        incarnation: str,
        command_types: frozenset[str],
    ) -> bool:
        """Drop this worker's registration unless queued work still needs it.

        A worker that leaves a stopped session must not strand a command
        enqueued the instant before it exits. The queue probe and the
        registration delete share one transaction, so an enqueue either
        commits first and holds the worker in its claim loop, or commits
        after the registration is gone and a fresh worker is started for
        it. Returns True when the caller may exit.
        """

        with self.transaction() as connection:
            owned = connection.execute(
                """
                SELECT 1 FROM workers
                WHERE session_id = ? AND incarnation = ?
                """,
                (session_id, incarnation),
            ).fetchone()
            if owned is None:
                return True
            if _queued_command_exists(connection, session_id, command_types):
                return False
            connection.execute(
                """
                DELETE FROM workers
                WHERE session_id = ? AND incarnation = ?
                """,
                (session_id, incarnation),
            )
        return True

    def queued_command_exists(
        self,
        session_id: str,
        command_types: frozenset[str],
    ) -> bool:
        with self._lock:
            return _queued_command_exists(
                self._connection,
                session_id,
                command_types,
            )

    def worker_registrations(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT session_id, pid, incarnation, heartbeat_at
                FROM workers ORDER BY session_id
                """
            ).fetchall()
        return [
            {
                "session_id": str(row["session_id"]),
                "pid": int(row["pid"]),
                "incarnation": str(row["incarnation"]),
                "heartbeat_at": str(row["heartbeat_at"]),
            }
            for row in rows
        ]

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

    def portable_session(
        self,
        session_id: str,
        *,
        include_events: bool = True,
    ) -> dict[str, Any]:
        self.get_session(session_id)
        with self._lock:
            goals = self._portable_rows(
                "goals",
                "session_id = ?",
                (session_id,),
            )
            goal_ids = tuple(str(item["goal_id"]) for item in goals)
            promotions = self._portable_rows(
                "goal_promotions",
                "session_id = ?",
                (session_id,),
            )
            promotion_ids = tuple(str(item["promotion_id"]) for item in promotions)
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
                "events": [],
                "commands": self._portable_rows(
                    "commands",
                    "session_id = ?",
                    (session_id,),
                ),
                "goals": goals,
                "goal_promotions": promotions,
                "goal_contract_adoptions": self._portable_rows(
                    "goal_contract_adoptions",
                    "session_id = ?",
                    (session_id,),
                ),
                "goal_milestones": self._portable_rows_for_values(
                    "goal_milestones",
                    "goal_id",
                    goal_ids,
                ),
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
                "goal_promotion_evidence": self._portable_rows_for_values(
                    "goal_promotion_evidence",
                    "promotion_id",
                    promotion_ids,
                ),
                "dispatch_transition_policies": self._portable_rows(
                    "dispatch_transition_policies",
                    "session_id = ?",
                    (session_id,),
                ),
                "authorization_receipts": self._portable_rows(
                    "authorization_receipts",
                    "session_id = ?",
                    (session_id,),
                ),
                "dispatch_invalidations": self._portable_rows(
                    "dispatch_invalidations",
                    "session_id = ?",
                    (session_id,),
                ),
                "dispatch_transition_ledger": self._portable_rows(
                    "dispatch_transition_ledger",
                    "session_id = ?",
                    (session_id,),
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
                "xhigh_authorization_receipts": self._portable_rows(
                    "xhigh_authorization_receipts",
                    "session_id = ?",
                    (session_id,),
                ),
                "command_envelopes": self._portable_rows(
                    "command_envelopes",
                    "session_id = ?",
                    (session_id,),
                ),
                "child_launch_gates": self._portable_rows(
                    "child_launch_gates",
                    "session_id = ?",
                    (session_id,),
                ),
                "child_launch_admissions": self._portable_rows_for_values(
                    "child_launch_admissions",
                    "command_id",
                    tuple(
                        str(row["command_id"])
                        for row in self._portable_rows(
                            "commands",
                            "session_id = ?",
                            (session_id,),
                        )
                    ),
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
                "command_dispatches": self._portable_rows(
                    "command_dispatches",
                    "session_id = ?",
                    (session_id,),
                ),
                "reconciliations": self._portable_rows(
                    "reconciliations",
                    "session_id = ?",
                    (session_id,),
                ),
                "session_creation_receipts": self._portable_rows(
                    "session_creation_receipts",
                    "session_id = ?",
                    (session_id,),
                ),
            }
            if include_events:
                tables["events"] = self._portable_rows(
                    "events",
                    "session_id = ?",
                    (session_id,),
                )
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
                "mutation_receipts": self._portable_rows("mutation_receipts"),
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
            if record.get("schema") != ("p13i/agent-harness/chat-record/v1"):
                raise ValueError("portable chat record schema is unsupported")
            tables = _require_object(record.get("tables"), "tables")
            for table in PORTABLE_SESSION_TABLES:
                rows = tables.get(table, [])
                if not isinstance(rows, list):
                    raise ValueError(table + " must be a list")
                for row in rows:
                    table_rows[table].append(_require_object(row, table + " row"))
        if global_record.get("schema") != ("p13i/agent-harness/chat-global/v1"):
            raise ValueError("portable global schema is unsupported")
        global_tables = _require_object(
            global_record.get("tables"),
            "global tables",
        )
        global_ui = global_tables.get("ui_state", [])
        if not isinstance(global_ui, list):
            raise ValueError("global ui_state must be a list")
        validated_global_ui = [
            _require_object(row, "ui_state row") for row in global_ui
        ]
        if not merge_global:
            table_rows["ui_state"].extend(validated_global_ui)
        table_rows["context_deliveries"] = (
            _normalize_portable_context_deliveries(
                table_rows["context_deliveries"]
            )
        )
        self._validate_portable_external_refs(table_rows["sessions"])
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
                validated = [_require_object(row, table + " row") for row in rows]
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

    def _validate_portable_external_refs(
        self,
        session_rows: list[dict[str, Any]],
    ) -> None:
        portable_refs: dict[tuple[str, str], str] = {}
        for row in session_rows:
            orchestrator = str(row.get("external_orchestrator", ""))
            job_id = str(row.get("external_job_id", ""))
            external_ref = normalize_external_ref(
                {"orchestrator": orchestrator, "job_id": job_id}
            )
            if not external_ref:
                continue
            key = (
                external_ref["orchestrator"],
                external_ref["job_id"],
            )
            session_id = str(row.get("session_id", ""))
            portable_session_id = portable_refs.get(key)
            if portable_session_id is not None and portable_session_id != session_id:
                raise ConflictError(
                    "portable external reference names two session UUIDs"
                )
            portable_refs[key] = session_id
            existing = self.get_session_by_external_ref(*key)
            if existing is not None and existing.session_id != session_id:
                raise ConflictError(
                    "portable external reference conflicts with an existing session"
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
                connection.execute("PRAGMA table_info(" + table + ")").fetchall(),
                key=lambda row: int(row["pk"]),
            )
            if int(row["pk"]) > 0
        ]
        if not primary_key:
            raise RuntimeError("portable merge table lacks a primary key: " + table)
        for row in rows:
            condition = " AND ".join(column + " = ?" for column in primary_key)
            existing = connection.execute(
                "SELECT * FROM " + table + " WHERE " + condition,
                tuple(row[column] for column in primary_key),
            ).fetchone()
            if existing is None:
                self._insert_portable_rows(connection, table, [row])
                continue
            if dict(existing) != row:
                raise ConflictError("portable global row conflicts in " + table)

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        goal = self.goal_for_session(session_id)
        portable = self.portable_session(session_id, include_events=False)
        tables = _require_object(portable.get("tables"), "portable tables")
        goal_value: dict[str, Any] | None = None
        evidence_value: list[dict[str, Any]] = []
        if goal is not None:
            goal_value = goal.as_dict()
            evidence_value = [item.as_dict() for item in self.evidence(goal.goal_id)]
        return {
            "schema": "p13i/agent-harness/session-export/v1",
            "session": session.as_dict(),
            "attempts": [item.as_dict() for item in self.attempts(session_id)],
            "events": [item.as_dict() for item in self.all_events(session_id)],
            "goal": goal_value,
            "evidence": evidence_value,
            "goals": tables["goals"],
            "goal_milestones": tables["goal_milestones"],
            "all_evidence": tables["evidence"],
            "goal_promotions": tables["goal_promotions"],
            "goal_contract_adoptions": tables["goal_contract_adoptions"],
            "goal_promotion_evidence": tables["goal_promotion_evidence"],
            "dispatch_transition_policies": tables["dispatch_transition_policies"],
            "authorization_receipts": tables["authorization_receipts"],
            "dispatch_invalidations": tables["dispatch_invalidations"],
            "dispatch_transition_ledger": tables["dispatch_transition_ledger"],
            "checkpoints": [item.as_dict() for item in self.checkpoints(session_id)],
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
        external_ref = normalize_external_ref(session_value.get("external_ref"))
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT owner_epoch FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                raise ConflictError("session already exists on this host")
            if external_ref:
                conflict = connection.execute(
                    """
                    SELECT session_id FROM sessions
                    WHERE external_orchestrator = ?
                    AND external_job_id = ?
                    """,
                    (
                        external_ref["orchestrator"],
                        external_ref["job_id"],
                    ),
                ).fetchone()
                if conflict is not None:
                    raise ConflictError(
                        "external reference already names another session"
                    )
            now = utc_now()
            connection.execute(
                """
                INSERT INTO sessions(
                    session_id, name, workspace, worktree, lifecycle,
                    attention, permission_mode, active_provider, model,
                    effort, goal_id, owner_host, owner_epoch, created_at,
                    updated_at, archived, external_orchestrator,
                    external_job_id, creation_digest
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
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
                    external_ref.get("orchestrator", ""),
                    external_ref.get("job_id", ""),
                    str(session_value.get("creation_digest", "")),
                ),
            )
            self._import_attempts(connection, payload, session_id)
            self._import_events(connection, payload, session_id)
            if isinstance(payload.get("goals"), list):
                self._import_goal_history(connection, payload, session_id)
            else:
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
            """
            INSERT INTO goals VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
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
                _dump(goal.get("permitted_providers", [])),
                _dump(goal.get("permitted_efforts", [])),
                int(goal.get("max_concurrency", 1)),
                str(goal.get("completion_policy", "evidence-all")),
                str(goal.get("incident_policy", "recover-then-pause")),
            ),
        )
        for position, value in enumerate(_objects(goal.get("milestones"))):
            connection.execute(
                "INSERT INTO goal_milestones VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    goal_id,
                    str(value.get("milestone_id", new_uuid())),
                    str(value.get("title", "")),
                    str(value.get("status", "active")),
                    _dump(value.get("dependencies", [])),
                    _dump(value.get("predicates", [])),
                    int(value.get("position", position)),
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

    def _import_goal_history(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
        session_id: str,
    ) -> None:
        goals = _objects(payload.get("goals"))
        milestones = _objects(payload.get("goal_milestones"))
        milestones_by_goal: dict[str, list[dict[str, Any]]] = {}
        for value in milestones:
            goal_id = str(value.get("goal_id", ""))
            milestones_by_goal.setdefault(goal_id, []).append(value)
        for goal in goals:
            goal_id = str(goal.get("goal_id", new_uuid()))
            connection.execute(
                """
                INSERT INTO goals VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    goal_id,
                    session_id,
                    str(goal.get("kind", "finite")),
                    str(goal.get("objective", "")),
                    str(goal.get("status", "active")),
                    str(goal.get("constraints_json", "[]")),
                    str(goal.get("predicates_json", "[]")),
                    str(goal.get("budgets_json", "{}")),
                    str(goal.get("created_at", utc_now())),
                    str(goal.get("updated_at", utc_now())),
                    str(goal.get("permitted_providers_json", "[]")),
                    str(goal.get("permitted_efforts_json", "[]")),
                    int(goal.get("max_concurrency", 1)),
                    str(goal.get("completion_policy", "evidence-all")),
                    str(goal.get("incident_policy", "recover-then-pause")),
                ),
            )
            for milestone in milestones_by_goal.get(goal_id, []):
                connection.execute(
                    "INSERT INTO goal_milestones VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        goal_id,
                        str(milestone.get("milestone_id", new_uuid())),
                        str(milestone.get("title", "")),
                        str(milestone.get("status", "active")),
                        str(milestone.get("dependencies_json", "[]")),
                        str(milestone.get("predicates_json", "[]")),
                        int(milestone.get("position", 0)),
                    ),
                )
        for value in _objects(payload.get("all_evidence")):
            connection.execute(
                "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(value.get("evidence_id", new_uuid())),
                    str(value.get("goal_id", "")),
                    str(value.get("evidence_type", "")),
                    str(value.get("subject", "")),
                    str(value.get("outcome", "")),
                    str(value.get("value_json", "{}")),
                    str(value.get("created_at", utc_now())),
                ),
            )
        for value in _objects(payload.get("goal_promotions")):
            connection.execute(
                "INSERT INTO goal_promotions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(value.get(name, "") for name in _GOAL_PROMOTION_COLUMNS),
            )
        for value in _objects(payload.get("goal_contract_adoptions")):
            connection.execute(
                """
                INSERT INTO goal_contract_adoptions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                tuple(value.get(name, "") for name in _GOAL_CONTRACT_ADOPTION_COLUMNS),
            )
        for value in _objects(payload.get("goal_promotion_evidence")):
            connection.execute(
                "INSERT INTO goal_promotion_evidence VALUES (?, ?, ?, ?, ?)",
                tuple(value.get(name, "") for name in _GOAL_PROMOTION_EVIDENCE_COLUMNS),
            )
        self._insert_portable_rows(
            connection,
            "dispatch_transition_policies",
            _objects(payload.get("dispatch_transition_policies")),
        )
        self._insert_portable_rows(
            connection,
            "authorization_receipts",
            _objects(payload.get("authorization_receipts")),
        )
        self._insert_portable_rows(
            connection,
            "dispatch_invalidations",
            _objects(payload.get("dispatch_invalidations")),
        )
        self._insert_portable_rows(
            connection,
            "dispatch_transition_ledger",
            _objects(payload.get("dispatch_transition_ledger")),
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


def _queued_command_exists(
    connection: sqlite3.Connection,
    session_id: str,
    command_types: frozenset[str],
) -> bool:
    placeholders = ", ".join("?" for _ in command_types)
    parameters: list[Any] = [session_id, CommandStatus.QUEUED]
    parameters.extend(sorted(command_types))
    row = connection.execute(
        """
        SELECT 1 FROM commands
        WHERE session_id = ? AND status = ?
        AND command_type IN ("""
        + placeholders
        + """)
        LIMIT 1
        """,
        tuple(parameters),
    ).fetchone()
    return row is not None


def _session_lifecycle(
    connection: sqlite3.Connection,
    session_id: str,
) -> str:
    row = connection.execute(
        """
        SELECT COALESCE(MAX(lifecycle), '') AS lifecycle FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return str(row["lifecycle"])


def _terminal_command_rejection(lifecycle: str, command_type: str) -> str:
    """Explain why a terminal session refuses a command, or return ''.

    A terminal session keeps no worker, so a command admitted into one
    would stay queued with nothing to claim it, holding its safety
    envelope open and blocking release quiescence forever. Admission
    therefore fails closed: only a resume, which is what returns a
    stopped session to a live lifecycle and starts the single writer
    that drains its queue, survives the guard.
    """

    if lifecycle not in TERMINAL_LIFECYCLES:
        return ""
    if lifecycle != Lifecycle.STOPPED:
        return "a " + lifecycle + " session admits no commands"
    if command_type == "resume":
        return ""
    return "a stopped session admits only a resume command"


def _sqlite_contention(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return False
    primary_code = code & 0xFF
    return primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _turn_timestamp(value: str) -> datetime.datetime:
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed.astimezone(datetime.UTC)


def _require_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(field + " must be an object")
    return value


def _normalize_portable_context_deliveries(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attempt_counts: dict[str, int] = {}
    for row in rows:
        attempt_id = _portable_attempt_id(row)
        if attempt_id:
            attempt_counts[attempt_id] = attempt_counts.get(attempt_id, 0) + 1
    assigned_attempt_ids = {
        attempt_id
        for attempt_id, count in attempt_counts.items()
        if count == 1
    }
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        attempt_id = _portable_attempt_id(normalized)
        duplicate_attempt = attempt_counts.get(attempt_id, 0) > 1
        if not attempt_id or duplicate_attempt:
            legacy_identity: dict[str, Any] = {
                "attempt_id": attempt_id,
                "row": normalized,
            }
            legacy_prefix = "legacy-"
            if duplicate_attempt:
                legacy_prefix = "legacy-duplicate-"
            attempt_id = legacy_prefix + normalized_digest(legacy_identity)
            collision = 0
            while attempt_id in assigned_attempt_ids:
                collision += 1
                legacy_identity["collision"] = collision
                attempt_id = legacy_prefix + normalized_digest(legacy_identity)
            normalized["attempt_id"] = attempt_id
        assigned_attempt_ids.add(attempt_id)
        normalized_rows.append(normalized)
    return normalized_rows


def _portable_attempt_id(row: dict[str, Any]) -> str:
    value = row.get("attempt_id", "")
    if not isinstance(value, str):
        return ""
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


def _is_digest(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _checkpoint_material_differs(
    before: sqlite3.Row,
    after: sqlite3.Row,
) -> bool:
    """Report whether two checkpoints hold different workspace material."""
    for column in _CHECKPOINT_MATERIAL_COLUMNS:
        if str(before[column]) != str(after[column]):
            return True
    return False


def _topology_receipt_binds(
    receipt: dict[str, Any],
    *,
    reconciliation_id: str,
    session_id: str,
    command_id: str,
    attempt_id: str,
    turn_id: str,
) -> bool:
    """Require the retained receipt to name this exact resolution."""
    if not receipt:
        return False
    return (
        str(receipt.get("reconciliation_id", "")) == reconciliation_id
        and str(receipt.get("session_id", "")) == session_id
        and str(receipt.get("command_id", "")) == command_id
        and str(receipt.get("attempt_id", "")) == attempt_id
        and str(receipt.get("turn_id", "")) == turn_id
    )


def _reconciled_command_result(
    attempt: sqlite3.Row,
    completion: dict[str, Any],
) -> dict[str, Any]:
    return {
        "provider": str(attempt["provider"]),
        "model": str(attempt["model"]),
        "effort": str(attempt["effort"]),
        "turn_id": str(completion["turn_id"]),
        "native_session_id": str(completion["native_session_id"]),
        "status": "complete",
        "checkpoint_id": str(completion["checkpoint_id"]),
        "workspace_material_digest": str(completion["workspace_material_digest"]),
        "final_message_sha256": str(completion["final_message_sha256"]),
        "reconciliation_id": str(completion["reconciliation_id"]),
        "reconciled_resolution": str(ReconciliationDecision.ACCEPT_CURRENT),
        "prior_code": str(completion["prior_code"]),
        "prior_result_sha256": str(completion["prior_result_sha256"]),
    }


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
        external_ref=_external_ref(row),
        creation_digest=str(row["creation_digest"]),
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
        turn_ref=_turn_ref(row),
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


def _goal(
    row: sqlite3.Row,
    milestone_rows: list[sqlite3.Row] | tuple[sqlite3.Row, ...] = (),
) -> Goal:
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
        milestones=tuple(_milestone(item) for item in milestone_rows),
        permitted_providers=tuple(
            str(item) for item in json.loads(row["permitted_providers_json"])
        ),
        permitted_efforts=tuple(
            str(item) for item in json.loads(row["permitted_efforts_json"])
        ),
        max_concurrency=int(row["max_concurrency"]),
        completion_policy=str(row["completion_policy"]),
        incident_policy=str(row["incident_policy"]),
    )


def _milestone(row: sqlite3.Row) -> Milestone:
    return Milestone(
        milestone_id=str(row["milestone_id"]),
        title=str(row["title"]),
        status=str(row["status"]),
        dependencies=tuple(str(item) for item in json.loads(row["dependencies_json"])),
        predicates=tuple(dict(item) for item in json.loads(row["predicates_json"])),
        position=int(row["position"]),
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


def _external_ref(
    row: sqlite3.Row | dict[str, Any],
) -> dict[str, str]:
    orchestrator = str(row["external_orchestrator"])
    if not orchestrator:
        return {}
    return {
        "orchestrator": orchestrator,
        "job_id": str(row["external_job_id"]),
    }


def _is_material_digest(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _dispatch_transition_terminal_checkpoint(
    connection: sqlite3.Connection,
    prior: sqlite3.Row,
    result: dict[str, Any],
    latest_checkpoint_id: str,
    live_material_digest: str,
) -> None:
    """Validate a provider-terminal guard stop that certified a checkpoint.

    An unattended guard can stop a provider after the turn already produced
    a clean, certified checkpoint. The effect is not ambiguous, so no
    reconciliation exists to anchor the next stage. This proves the soft
    success instead: the failed command must claim provider terminality, own
    the exact latest checkpoint and live material, hold exactly one failed
    post-boundary dispatch and turn, and bypass no reconciliation.
    """
    command_id = str(prior["command_id"])
    session_id = str(prior["session_id"])
    if str(result.get("checkpoint_id", "")) != latest_checkpoint_id:
        raise ConflictError("dispatch transition terminal checkpoint is not latest")
    result_material = str(result.get("workspace_material_digest", ""))
    if not _is_material_digest(result_material):
        raise ConflictError("dispatch transition terminal material is invalid")
    if result_material != live_material_digest:
        raise ConflictError("dispatch transition terminal material is not current")
    pending = connection.execute(
        "SELECT COUNT(*) AS count FROM reconciliations WHERE command_id = ?",
        (command_id,),
    ).fetchone()
    if pending is not None and int(pending["count"]) > 0:
        raise ConflictError("dispatch transition terminal result has a reconciliation")
    dispatches = connection.execute(
        """
        SELECT attempt_id, session_id, turn_id, state
        FROM command_dispatches WHERE command_id = ? AND crossed_boundary = 1
        ORDER BY created_at, attempt_id
        """,
        (command_id,),
    ).fetchall()
    if len(dispatches) != 1:
        raise ConflictError("dispatch transition terminal boundary is not exact")
    dispatch = dispatches[0]
    if str(dispatch["session_id"]) != session_id:
        raise ConflictError("dispatch transition terminal dispatch session changed")
    if str(dispatch["state"]) != "failed":
        raise ConflictError("dispatch transition terminal dispatch is not failed")
    turn_id = str(dispatch["turn_id"])
    turn = connection.execute(
        "SELECT session_id, status FROM turns WHERE turn_id = ?",
        (turn_id,),
    ).fetchone()
    if turn is None or str(turn["session_id"]) != session_id:
        raise ConflictError("dispatch transition terminal turn is unknown")
    if str(turn["status"]) != "failed":
        raise ConflictError("dispatch transition terminal turn is not failed")
    guard_events = connection.execute(
        """
        SELECT metadata_json FROM events
        WHERE session_id = ? AND event_type = 'guard.tripped' AND turn_id = ?
        ORDER BY sequence
        """,
        (session_id, turn_id),
    ).fetchall()
    matching_guards = [
        item
        for item in guard_events
        if _dispatch_transition_terminal_receipt(item) == {
            "command_id": command_id,
            "attempt_id": str(dispatch["attempt_id"]),
            "checkpoint_id": latest_checkpoint_id,
            "provider_terminal": True,
        }
    ]
    if len(matching_guards) != 1:
        raise ConflictError("dispatch transition terminal guard receipt changed")
    checkpoint_events = connection.execute(
        """
        SELECT metadata_json FROM events
        WHERE session_id = ? AND event_type = 'checkpoint.created' AND turn_id = ?
        ORDER BY sequence
        """,
        (session_id, turn_id),
    ).fetchall()
    matching_checkpoints = [
        item
        for item in checkpoint_events
        if str(_load_object(str(item["metadata_json"])).get("checkpoint_id", ""))
        == latest_checkpoint_id
    ]
    if len(matching_checkpoints) != 1:
        raise ConflictError("dispatch transition terminal checkpoint receipt changed")


def _dispatch_transition_terminal_receipt(
    row: sqlite3.Row | dict[str, Any],
) -> dict[str, Any]:
    metadata = _load_object(str(row["metadata_json"]))
    return {
        "command_id": str(metadata.get("command_id", "")),
        "attempt_id": str(metadata.get("attempt_id", "")),
        "checkpoint_id": str(metadata.get("checkpoint_id", "")),
        "provider_terminal": metadata.get("provider_terminal"),
    }


def _dispatch_transition_resolved_reconciliation(
    connection: sqlite3.Connection,
    prior: sqlite3.Row,
    latest_checkpoint_id: str,
    live_material_digest: str,
) -> sqlite3.Row:
    """Return the one safely resolved reconciliation this command owns.

    The decision must be explicit and current: exactly one
    reconciliation, an ``accept-current`` or ``restore-pre-turn``
    resolution, the latest certified checkpoint, and a protected
    resolution digest that still equals the live workspace material.
    """
    reconciliations = connection.execute(
        """
        SELECT * FROM reconciliations WHERE command_id = ?
        ORDER BY created_at, reconciliation_id
        """,
        (str(prior["command_id"]),),
    ).fetchall()
    if len(reconciliations) != 1:
        raise ConflictError("dispatch transition requires exactly one reconciliation")
    reconciliation = reconciliations[0]
    if str(reconciliation["status"]) != ReconciliationStatus.RESOLVED:
        raise ConflictError("dispatch transition reconciliation is unresolved")
    if str(reconciliation["resolution"]) not in {
        ReconciliationDecision.ACCEPT_CURRENT,
        ReconciliationDecision.RESTORE_PRE_TURN,
    }:
        raise ConflictError("dispatch transition reconciliation resolution is unsafe")
    audit = _load_object(str(reconciliation["audit_json"]))
    if str(audit.get("resolution_checkpoint_id", "")) != latest_checkpoint_id:
        raise ConflictError(
            "dispatch transition reconciliation checkpoint is not latest"
        )
    resolution_material = str(audit.get("resolution_workspace_digest", ""))
    if not _is_material_digest(resolution_material) or (
        resolution_material != live_material_digest
    ):
        raise ConflictError(
            "dispatch transition reconciliation material is not current"
        )
    return reconciliation


def _dispatch_transition_stranded_reconciliation(
    connection: sqlite3.Connection,
    prior: sqlite3.Row,
    latest_checkpoint_id: str,
    live_material_digest: str,
) -> sqlite3.Row:
    """Anchor a command an accepted reconciliation left dispatching.

    An explicit resolution can accept material no ``turn.completed``
    ever certified. Completion stays unproven, so nothing here settles
    the command, attempt, turn, or dispatch: the operator decision alone
    carries the next declared stage. The unvalidated topology must
    therefore still read exactly as the reconciliation left it, bound to
    this command and session by one crossed provider boundary, and its
    receipt must claim no completion this build never observed.
    """
    reconciliation = _dispatch_transition_resolved_reconciliation(
        connection,
        prior,
        latest_checkpoint_id,
        live_material_digest,
    )
    command_id = str(prior["command_id"])
    session_id = str(prior["session_id"])
    dispatches = connection.execute(
        """
        SELECT d.attempt_id, d.turn_id, d.checkpoint_id,
            d.state AS dispatch_state, a.status AS attempt_state,
            t.status AS turn_state
        FROM command_dispatches AS d
        JOIN provider_attempts AS a ON a.attempt_id = d.attempt_id
            AND a.session_id = d.session_id
        JOIN turns AS t ON t.turn_id = d.turn_id
            AND t.attempt_id = d.attempt_id AND t.session_id = d.session_id
        WHERE d.command_id = ? AND d.session_id = ? AND d.crossed_boundary = 1
        ORDER BY d.created_at, d.attempt_id
        """,
        (command_id, session_id),
    ).fetchall()
    if len(dispatches) != 1:
        raise ConflictError("dispatch transition accepted boundary is not exact")
    dispatch = dispatches[0]
    audit = _load_object(str(reconciliation["audit_json"]))
    identity = _object_or_empty(audit.get("dispatch_identity"))
    receipt = _object_or_empty(audit.get("topology_receipt"))
    if {
        "reconciliation_session_id": str(reconciliation["session_id"]),
        "attempt_id": str(identity.get("attempt_id", "")),
        "turn_id": str(identity.get("turn_id", "")),
        "checkpoint_id": str(identity.get("checkpoint_id", "")),
        "dispatch_state": str(dispatch["dispatch_state"]),
        "attempt_state": str(dispatch["attempt_state"]),
        "turn_state": str(dispatch["turn_state"]),
        "receipt_binds": _topology_receipt_binds(
            receipt,
            reconciliation_id=str(reconciliation["reconciliation_id"]),
            session_id=session_id,
            command_id=command_id,
            attempt_id=str(dispatch["attempt_id"]),
            turn_id=str(dispatch["turn_id"]),
        ),
        # The command is still dispatching precisely because completion
        # was never proven. A receipt that already claims one
        # contradicts that, so this refuses rather than inherit an
        # outcome nothing observed.
        "receipt_projects": bool(
            receipt.get("command_status") or receipt.get("completion_evidence")
        ),
    } != {
        "reconciliation_session_id": session_id,
        "attempt_id": str(dispatch["attempt_id"]),
        "turn_id": str(dispatch["turn_id"]),
        "checkpoint_id": str(dispatch["checkpoint_id"]),
        "dispatch_state": "ambiguous",
        "attempt_state": "ambiguous",
        "turn_state": "ambiguous",
        "receipt_binds": True,
        "receipt_projects": False,
    }:
        raise ConflictError("dispatch transition accepted topology is not bound")
    return reconciliation


def _dispatch_transition_active_commands(
    connection: sqlite3.Connection,
    session_id: str,
) -> int:
    """Count the commands that still hold live dispatch capacity.

    A dispatching command whose reconciliation was explicitly resolved
    is stranded, not running: its leases are released and its topology
    is settled ambiguous, so it does not block a transition off itself.
    The anchor revalidates that whole shape before certifying anything.
    """
    row = connection.execute(
        """
        SELECT COUNT(*) AS count FROM commands
        WHERE session_id = ? AND status IN (
            'queued', 'awaiting-xhigh-authorization', 'dispatching'
        )
        AND NOT (status = 'dispatching' AND EXISTS (
            SELECT 1 FROM reconciliations AS r
            WHERE r.command_id = commands.command_id
            AND r.session_id = commands.session_id
            AND r.status = 'resolved'
            AND r.resolution IN ('accept-current', 'restore-pre-turn')
        ))
        """,
        (session_id,),
    ).fetchone()
    count = 0
    if row is not None:
        count = int(row["count"])
    return count


def _dispatch_transition_control_is_inert(
    connection: sqlite3.Connection,
    command: sqlite3.Row,
) -> bool:
    """Report whether a control command left the stage exactly as found.

    A transition anchors on the last command that declared a stage. A
    control the worker refused before it reached the session declares
    nothing, so it must not shadow the command it followed. The refusal
    has to prove itself three ways: it recorded exactly the refusal
    result the worker writes, nothing in the store admitted it to a
    provider or reserved anything for it, and no event credits it with
    an effect. Anything short of that proof, including an unrecognized
    control failure or one unknown extra result field, stays material
    and anchors the transition itself.
    """
    if str(command["status"]) != CommandStatus.FAILED:
        return False
    if str(command["command_type"]) not in TRANSITION_CONTROL_COMMANDS:
        return False
    result = _load_object(str(command["result_json"]))
    if set(result) != INERT_CONTROL_RESULT_KEYS:
        return False
    code = result.get("code")
    if not isinstance(code, str) or code not in INERT_CONTROL_FAILURES:
        return False
    message = result.get("message")
    if not isinstance(message, str) or not message.strip():
        return False
    command_id = str(command["command_id"])
    bound = connection.execute(
        """
        SELECT (
            (SELECT COUNT(*) FROM command_dispatches WHERE command_id = ?)
            + (SELECT COUNT(*) FROM reconciliations WHERE command_id = ?)
            + (SELECT COUNT(*) FROM process_leases WHERE command_id = ?)
            + (SELECT COUNT(*) FROM guard_incidents WHERE command_id = ?)
            + (SELECT COUNT(*) FROM context_deliveries WHERE command_id = ?)
            + (SELECT COUNT(*) FROM command_envelopes WHERE command_id = ?)
            + (SELECT COUNT(*) FROM child_launch_gates WHERE command_id = ?)
            + (
                SELECT COUNT(*) FROM dispatch_transition_ledger
                WHERE prior_command_id = ? OR reserved_command_id = ?
                    OR consumed_command_id = ?
            )
            + (
                SELECT COUNT(*) FROM xhigh_authorization_receipts
                WHERE command_id = ?
            )
        ) AS count
        """,
        (command_id,) * 11,
    ).fetchone()
    if bound is None:
        return False
    if int(bound["count"]) > 0:
        return False
    attributions = " OR ".join(
        "json_extract(metadata_json, '$." + name + "') = ?"
        for name in CONTROL_EVENT_BINDINGS
    )
    attributed = connection.execute(
        """
        SELECT COUNT(*) AS count FROM events
        WHERE session_id = ? AND json_valid(metadata_json)
        AND ("""
        + attributions
        + ")",
        (str(command["session_id"]),) + (command_id,) * len(CONTROL_EVENT_BINDINGS),
    ).fetchone()
    if attributed is None:
        return False
    return int(attributed["count"]) == 0


def _dispatch_transition_predecessor(
    connection: sqlite3.Connection,
    session_id: str,
) -> sqlite3.Row | None:
    """Return the newest command a transition may anchor on.

    Commands stay in their recorded newest-first order. The walk stops
    at the first command that declared anything, so it can only ever
    step over a trailing run of refused controls, and a session holding
    nothing else anchors nothing.
    """
    commands = connection.execute(
        """
        SELECT * FROM commands WHERE session_id = ?
        ORDER BY created_at DESC, command_id DESC
        """,
        (session_id,),
    )
    for command in commands:
        if not _dispatch_transition_control_is_inert(connection, command):
            return command
    return None


def _dispatch_transition_anchor(
    connection: sqlite3.Connection,
    prior: sqlite3.Row,
    latest_checkpoint_id: str,
    live_material_digest: str,
) -> dict[str, str]:
    command_type = str(prior["command_type"])
    status = str(prior["status"])
    result = _load_object(str(prior["result_json"]))
    anchor_kind = ""
    reconciliation_id = ""
    reconciliation_resolution = ""
    if status == CommandStatus.COMPLETE and command_type == "message":
        if str(result.get("checkpoint_id", "")) != latest_checkpoint_id:
            raise ConflictError("dispatch transition provider checkpoint is not latest")
        result_material = str(result.get("workspace_material_digest", ""))
        if len(result_material) != 64 or result_material != live_material_digest:
            raise ConflictError("dispatch transition provider material is not current")
        anchor_kind = "provider-result"
    elif status == CommandStatus.COMPLETE and (
        command_type in TRANSITION_CONTROL_COMMANDS
    ):
        anchor_kind = "control-command"
    elif (
        status == CommandStatus.FAILED
        and result.get("code") == "E_SAFETY_GUARD"
        and result.get("provider_terminal") is True
    ):
        _dispatch_transition_terminal_checkpoint(
            connection,
            prior,
            result,
            latest_checkpoint_id,
            live_material_digest,
        )
        anchor_kind = "terminal-checkpoint"
    elif status == CommandStatus.FAILED and result.get("code") in {
        "E_NEEDS_RECONCILIATION",
        "E_SAFETY_GUARD",
    }:
        reconciliation = _dispatch_transition_resolved_reconciliation(
            connection,
            prior,
            latest_checkpoint_id,
            live_material_digest,
        )
        anchor_kind = "resolved-reconciliation"
        reconciliation_id = str(reconciliation["reconciliation_id"])
        reconciliation_resolution = str(reconciliation["resolution"])
    elif status == CommandStatus.DISPATCHING:
        reconciliation = _dispatch_transition_stranded_reconciliation(
            connection,
            prior,
            latest_checkpoint_id,
            live_material_digest,
        )
        anchor_kind = "resolved-reconciliation"
        reconciliation_id = str(reconciliation["reconciliation_id"])
        reconciliation_resolution = str(reconciliation["resolution"])
    else:
        raise ConflictError("dispatch transition prior command is not eligible")
    return {
        "prior_command_type": command_type,
        "prior_anchor_kind": anchor_kind,
        "prior_reconciliation_id": reconciliation_id,
        "prior_reconciliation_resolution": reconciliation_resolution,
    }


def _dispatch_transition_material_is_current(
    connection: sqlite3.Connection,
    session_id: str,
    authorization: dict[str, Any],
    workspace: Path,
    live_material_digest: str,
) -> bool:
    if authorization.get("prior_material_digest") == live_material_digest:
        return True
    checkpoint_id = str(authorization.get("prior_checkpoint_id", ""))
    checkpoint = connection.execute(
        """
        SELECT base_commit, patch_digest, untracked_digest
        FROM checkpoints
        WHERE checkpoint_id = ? AND session_id = ?
        """,
        (checkpoint_id, session_id),
    ).fetchone()
    if checkpoint is None:
        return False
    return workspace_matches_checkpoint_collapse(
        workspace,
        base_commit=str(checkpoint["base_commit"]),
        patch_digest=str(checkpoint["patch_digest"]),
        untracked_digest=str(checkpoint["untracked_digest"]),
    )


def _dispatch_transition_epoch_is_active(
    connection: sqlite3.Connection,
    session_id: str,
    authorization: dict[str, Any],
) -> bool:
    policy_sha256 = str(authorization.get("policy_sha256", ""))
    epoch_id = str(authorization.get("epoch_id", ""))
    goal_id = str(authorization.get("goal_id", ""))
    if not policy_sha256 or not epoch_id or not goal_id:
        return False
    policy = authorization.get("policy")
    if isinstance(policy, dict) and policy:
        if normalized_digest(policy) != policy_sha256:
            return False
    else:
        policy_ref = authorization.get("policy_ref")
        if policy_ref != {
            "policy_sha256": policy_sha256,
            "session_id": session_id,
            "goal_id": goal_id,
            "epoch_id": epoch_id,
        }:
            return False
        policy_row = connection.execute(
            """
            SELECT payload_json FROM dispatch_transition_policies
            WHERE policy_sha256 = ? AND session_id = ?
              AND goal_id = ? AND epoch_id = ?
            """,
            (policy_sha256, session_id, goal_id, epoch_id),
        ).fetchone()
        if policy_row is None:
            return False
        policy = _load_object(str(policy_row["payload_json"]))
        if normalized_digest(policy) != policy_sha256:
            return False
    if policy.get("session_id") != session_id:
        return False
    if policy.get("epoch_id") != epoch_id:
        return False
    goal = connection.execute(
        """
        SELECT s.goal_id, g.constraints_json FROM sessions AS s
        JOIN goals AS g ON g.goal_id = s.goal_id
        WHERE s.session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if goal is None:
        return False
    if goal_id != str(goal["goal_id"]):
        return False
    constraints = json.loads(str(goal["constraints_json"]))
    return (
        "dispatch-generation-transition-policy-sha256:" + policy_sha256 in constraints
        and "dispatch-generation-transition-epoch:" + epoch_id in constraints
    )


def _turn_ref(
    row: sqlite3.Row | dict[str, Any],
) -> dict[str, str]:
    step_id = str(row["turn_step_id"])
    if not step_id:
        return {}
    return {
        "step_id": step_id,
        "agent_role": str(row["turn_agent_role"]),
    }


def _reconciliation(row: sqlite3.Row) -> ReconciliationRecord:
    attempts = json.loads(row["provider_attempts_json"])
    if not isinstance(attempts, list):
        raise ValueError("stored provider attempts are not a list")
    return ReconciliationRecord(
        reconciliation_id=str(row["reconciliation_id"]),
        session_id=str(row["session_id"]),
        command_id=str(row["command_id"]),
        pre_dispatch_checkpoint_id=str(row["pre_dispatch_checkpoint_id"]),
        current_workspace_digest=str(row["current_workspace_digest"]),
        current_workspace_summary=str(row["current_workspace_summary"]),
        provider_attempts=tuple(dict(item) for item in attempts),
        safety_consumption=_load_object(row["safety_consumption_json"]),
        status=str(row["status"]),
        resolution=str(row["resolution"]),
        audit=_load_object(row["audit_json"]),
        created_at=str(row["created_at"]),
        resolved_at=str(row["resolved_at"]),
    )


def _bounded_json_value(
    value: object,
    *,
    depth: int = 0,
) -> object:
    if depth >= 4:
        return "[depth limit]"
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, list):
        return [_bounded_json_value(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for name in sorted(value, key=str)[:20]:
            result[str(name)] = _bounded_json_value(
                value[name],
                depth=depth + 1,
            )
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1000]
