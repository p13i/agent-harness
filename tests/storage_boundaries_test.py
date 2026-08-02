"""Deterministic coverage for durable storage boundary behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import agent_harness.storage as storage_module
from agent_harness.errors import ConflictError
from agent_harness.errors import NotFoundError
from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Checkpoint
from agent_harness.models import ProviderAttempt
from agent_harness.errors import WorkerOwnershipLostError
from agent_harness.goals import create_goal
from agent_harness.goals import make_evidence
from agent_harness.orchestration import creation_digest
from agent_harness.orchestration import normalized_digest
from agent_harness.storage import PORTABLE_GLOBAL_TABLES
from agent_harness.storage import PORTABLE_SESSION_TABLES
from agent_harness.storage import StateStore
from tests.test_support import session


def _creation(
    value: object,
    *,
    external_ref: dict[str, str] | None = None,
) -> dict[str, Any]:
    current = value
    return {
        "workspace": current.workspace,  # type: ignore[attr-defined]
        "name": current.name,  # type: ignore[attr-defined]
        "permission_mode": current.permission_mode,  # type: ignore[attr-defined]
        "execution_profile": "interactive",
        "direct": True,
        "external_ref": external_ref or {},
        "routing": {"model": "", "effort": ""},
    }


def _portable_empty() -> tuple[dict[str, Any], dict[str, Any]]:
    record = {
        "schema": "p13i/agent-harness/chat-record/v1",
        "tables": {table: [] for table in PORTABLE_SESSION_TABLES},
    }
    global_record = {
        "schema": "p13i/agent-harness/chat-global/v1",
        "tables": {
            "ui_state": [],
            **{table: [] for table in PORTABLE_GLOBAL_TABLES},
        },
    }
    return record, global_record


def test_schema_column_integrity_and_transaction_boundaries(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "unsupported.sqlite3"
    connection = sqlite3.connect(unsupported)
    connection.execute("CREATE TABLE schema_meta(version INTEGER NOT NULL)")
    connection.execute("INSERT INTO schema_meta VALUES (99)")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="schema version"):
        StateStore(unsupported)

    store = StateStore(tmp_path / "state.sqlite3")
    with store.transaction() as connection:
        store._add_column(
            connection,
            "sessions",
            "name",
            "TEXT NOT NULL DEFAULT ''",
        )
    store.close()

    class CursorProbe:
        def fetchone(self) -> None:
            return None

    class ConnectionProbe:
        def execute(self, unused: str) -> CursorProbe:
            return CursorProbe()

    probe = object.__new__(StateStore)
    probe._lock = threading.RLock()
    probe._connection = ConnectionProbe()  # type: ignore[assignment]
    assert probe.integrity_check() == ""


def test_ensured_session_conflict_and_orphan_boundaries(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    external_ref = {"orchestrator": "test", "job_id": "one"}
    created = replace(session(tmp_path), external_ref=external_ref)
    with pytest.raises(ValueError, match="external_ref"):
        store.ensure_session(
            created,
            _creation(
                created,
                external_ref={"orchestrator": "test", "job_id": "two"},
            ),
        )

    creation = _creation(created, external_ref=external_ref)
    ensured, is_new = store.ensure_session(created, creation)
    assert is_new
    replay, is_new = store.ensure_session(
        replace(created, session_id=new_uuid()),
        creation,
        idempotency_key="external-replay",
    )
    assert not is_new
    assert replay.session_id == ensured.session_id

    duplicate = replace(
        session(tmp_path),
        session_id=created.session_id,
    )
    with pytest.raises(ConflictError, match="identifier"):
        store.ensure_session(duplicate, _creation(duplicate))

    orphan_creation = _creation(session(tmp_path))
    orphan_digest = creation_digest(orphan_creation)
    store._connection.execute("PRAGMA foreign_keys = OFF")
    store._connection.execute(
        "INSERT INTO session_creation_receipts VALUES (?, ?, ?, ?, ?)",
        ("orphan", orphan_digest, new_uuid(), "{}", "now"),
    )
    store._connection.commit()
    store._connection.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(RuntimeError, match="has no session"):
        store.ensure_session(
            session(tmp_path),
            orphan_creation,
            idempotency_key="orphan",
        )
    with pytest.raises(RuntimeError, match="has no session"):
        store.existing_ensured_session(
            orphan_creation,
            idempotency_key="orphan",
        )

    keyed = session(tmp_path)
    keyed_creation = _creation(keyed)
    store.ensure_session(
        keyed,
        keyed_creation,
        idempotency_key="keyed",
    )
    referenced_ref = {"orchestrator": "test", "job_id": "referenced"}
    referenced = replace(
        session(tmp_path),
        external_ref=referenced_ref,
    )
    referenced_creation = _creation(
        referenced,
        external_ref=referenced_ref,
    )
    store.ensure_session(referenced, referenced_creation)
    target_digest = creation_digest(referenced_creation)
    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE session_creation_receipts SET request_digest = ?
            WHERE idempotency_key = ?
            """,
            (target_digest, "keyed"),
        )
    with pytest.raises(ConflictError, match="different sessions"):
        store.existing_ensured_session(
            referenced_creation,
            idempotency_key="keyed",
            external_ref=referenced_ref,
        )
    store.close()


def test_update_claim_turn_and_receipt_failure_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    with pytest.raises(NotFoundError):
        store.update_session(new_uuid(), name="missing")

    command = store.enqueue_command(
        created.session_id,
        "message",
        {
            "text": "turn",
            "turn_ref": {
                "step_id": "step",
                "agent_role": "implementer",
            },
        },
        "turn",
    )
    store._connection.execute(
        """
        CREATE TRIGGER ignore_command_claim
        BEFORE UPDATE OF status ON commands
        WHEN NEW.status = 'dispatching'
        BEGIN
            SELECT RAISE(IGNORE);
        END
        """
    )
    assert store.claim_command(created.session_id) is None
    store._connection.execute("DROP TRIGGER ignore_command_claim")

    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="codex",
        native_session_id="native",
        model="model",
        effort="low",
        auth_mode="subscription",
        status="running",
        started_at="now",
        ended_at="",
    )
    store.create_attempt(attempt)
    turn_id = store.start_turn(
        created.session_id,
        attempt.attempt_id,
        turn_ref=command.turn_ref,
    )
    rows = store.presentation_turn_rows(created.session_id)
    assert rows[0]["turn_id"] == turn_id
    assert rows[0]["turn_ref"]["step_id"] == "step"

    monkeypatch.setattr(store, "mutation_receipt", lambda *unused: None)
    with pytest.raises(RuntimeError, match="not recorded"):
        store.record_mutation_receipt(
            "receipt",
            "operation",
            "digest",
            {"ok": True},
            200,
        )
    store.close()


def test_portable_validation_import_and_merge_boundaries(
    tmp_path: Path,
) -> None:
    record, global_record = _portable_empty()
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    with pytest.raises(ConflictError, match="empty store"):
        store.import_portable([], global_record)

    existing_record = store.portable_session(created.session_id)
    with pytest.raises(ConflictError, match="already exists"):
        store.merge_portable([existing_record], global_record)
    store.close()

    invalid_values = (
        ([{**record, "schema": "invalid"}], global_record),
        ([{**record, "tables": []}], global_record),
        (
            [
                {
                    **record,
                    "tables": {
                        **record["tables"],
                        "events": {},
                    },
                }
            ],
            global_record,
        ),
        ([record], {**global_record, "schema": "invalid"}),
        ([record], {**global_record, "tables": []}),
        (
            [record],
            {
                **global_record,
                "tables": {
                    **global_record["tables"],
                    "ui_state": {},
                },
            },
        ),
        (
            [record],
            {
                **global_record,
                "tables": {
                    **global_record["tables"],
                    "usage_samples": {},
                },
            },
        ),
    )
    for index, (records, global_value) in enumerate(invalid_values):
        invalid_store = StateStore(
            tmp_path / ("invalid-" + str(index) + ".sqlite3")
        )
        with pytest.raises(ValueError):
            invalid_store.import_portable(records, global_value)
        invalid_store.close()

    merge_store = StateStore(tmp_path / "merge.sqlite3")
    with merge_store.transaction() as connection:
        connection.execute("CREATE TABLE no_key(value TEXT)")
        with pytest.raises(RuntimeError, match="primary key"):
            merge_store._merge_portable_rows(connection, "no_key", [])
        connection.execute(
            "CREATE TABLE merge_values(id TEXT PRIMARY KEY, value TEXT)"
        )
        row = {"id": "one", "value": "first"}
        merge_store._merge_portable_rows(
            connection,
            "merge_values",
            [row],
        )
        merge_store._merge_portable_rows(
            connection,
            "merge_values",
            [row],
        )
        with pytest.raises(ConflictError, match="conflicts"):
            merge_store._merge_portable_rows(
                connection,
                "merge_values",
                [{"id": "one", "value": "other"}],
            )
    merge_store.close()


def test_session_import_and_reconciliation_decode_boundaries(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    with pytest.raises(ValueError, match="identifier"):
        store.import_session(
            {"session": {}},
            worktree=str(tmp_path),
            owner_host="host",
            owner_epoch=1,
        )

    external_ref = {"orchestrator": "test", "job_id": "job"}
    existing = replace(session(tmp_path), external_ref=external_ref)
    store.create_session(existing)
    conflict = {
        "session": {
            **session(tmp_path).as_dict(),
            "external_ref": external_ref,
        }
    }
    with pytest.raises(ConflictError, match="external reference"):
        store.import_session(
            conflict,
            worktree=str(tmp_path / "conflict"),
            owner_host="host",
            owner_epoch=2,
        )

    minimal = session(tmp_path)
    imported = store.import_session(
        {"session": minimal.as_dict()},
        worktree=str(tmp_path / "minimal"),
        owner_host="host",
        owner_epoch=2,
    )
    assert imported.session_id == minimal.session_id

    empty_safety = session(tmp_path)
    store.import_session(
        {
            "session": empty_safety.as_dict(),
            "safety": {"profile": ""},
        },
        worktree=str(tmp_path / "empty-safety"),
        owner_host="host",
        owner_epoch=2,
    )

    with pytest.raises(ValueError, match="not a list"):
        storage_module._reconciliation(
            {"provider_attempts_json": "{}"}  # type: ignore[arg-type]
        )
    store.close()


def test_bounded_json_values_are_depth_width_and_length_limited() -> None:
    bounded = storage_module._bounded_json_value(
        {
            "text": "y" * 1_200,
            "items": list(range(40)),
            "nested": {"one": {"two": {"three": {"four": "deep"}}}},
            "path": Path("/tmp"),
            "flag": True,
            "absent": None,
            **{"zpad-" + str(index): index for index in range(25)},
        }
    )

    assert isinstance(bounded, dict)
    assert len(bounded) == 20
    assert len(bounded["text"]) == 1_000
    assert bounded["items"] == list(range(20))
    assert bounded["flag"] is True
    assert bounded["absent"] is None
    assert bounded["nested"]["one"]["two"]["three"] == "[depth limit]"
    assert storage_module._bounded_json_value(Path("/tmp")) == "/tmp"


def test_compacted_history_summarizes_every_prior_event(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)

    assert store.context_history_summary(created.session_id, 0) == {}
    assert store.context_history_summary(created.session_id, 5) == {}

    for sequence in range(1, 121):
        store.append_event(
            created.session_id,
            "user.message",
            role="user",
            text="turn " + str(sequence),
            status="complete",
            metadata={"sequence": sequence, "payload": "x" * 1_200},
        )

    summary = store.context_history_summary(created.session_id, 100)

    assert summary["schema"] == "p13i/agent-harness/compacted-history/v1"
    assert summary["event_count"] == 100
    assert summary["first_sequence"] == 1
    assert summary["last_sequence"] == 100
    assert len(summary["anchors"]) == 100
    assert summary["anchors"][0]["sequence"] == 1
    assert len(summary["anchors"][0]["metadata"]["payload"]) == 1_000
    assert len(summary["history_digest"]) == 64
    store.close()


def test_fork_lineage_requires_its_source_checkpoint(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)

    assert store.fork_lineage(created.session_id) == {}

    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="codex",
        native_session_id="",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context-digest",
        created_at=utc_now(),
    )
    store.add_checkpoint(checkpoint)
    store.append_event(
        created.session_id,
        "session.forked",
        status="complete",
        metadata={
            "source_session_id": "source-session",
            "source_sequence": 4,
            "source_checkpoint_id": checkpoint.checkpoint_id,
        },
    )

    lineage = store.fork_lineage(created.session_id)

    assert lineage["source_context_digest"] == "context-digest"
    assert lineage["source_sequence"] == 4

    forgotten = session(tmp_path)
    store.create_session(forgotten)
    store.append_event(
        forgotten.session_id,
        "session.forked",
        status="complete",
        metadata={"source_checkpoint_id": new_uuid()},
    )
    with pytest.raises(ConflictError, match="fork source checkpoint is missing"):
        store.fork_lineage(forgotten.session_id)
    store.close()


def test_context_unresolved_decisions_include_approvals_and_reconciliations(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    approval_id = store.create_approval(
        created.session_id,
        "",
        "provider-request",
        "tool",
        "Run the bounded command?",
        [{"id": "approve"}],
    )
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "one effect"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="codex",
        native_session_id="",
        model="",
        effort="",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    store.create_attempt(attempt)
    turn_id = store.start_turn(created.session_id, attempt.attempt_id)
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="codex",
        native_session_id="",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=utc_now(),
    )
    store.add_checkpoint(checkpoint)
    store.record_dispatch_checkpoint(
        command.command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    store.mark_provider_boundary(attempt.attempt_id)
    recovery = store.recover_interrupted_commands(
        created.session_id,
        "moved-digest",
        "summary",
    )
    assert len(recovery.reconciliations) == 1

    decisions = store.context_unresolved_decisions(created.session_id)

    assert [item["kind"] for item in decisions] == ["approval", "reconciliation"]
    assert decisions[0]["id"] == approval_id
    assert decisions[1]["id"] == recovery.reconciliations[0].reconciliation_id
    store.close()


def _attempt(store: StateStore, session_id: str) -> ProviderAttempt:
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=session_id,
        provider="codex",
        native_session_id="",
        model="",
        effort="",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    store.create_attempt(attempt)
    return attempt


def _dispatched_attempt(
    store: StateStore,
    session_id: str,
    command_id: str,
) -> ProviderAttempt:
    attempt = _attempt(store, session_id)
    turn_id = store.start_turn(session_id, attempt.attempt_id)
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=session_id,
        sequence=store.last_sequence(session_id),
        provider="codex",
        native_session_id="",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=utc_now(),
    )
    store.add_checkpoint(checkpoint)
    store.record_dispatch_checkpoint(
        command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    return attempt


def test_nested_transactions_roll_back_to_their_savepoint(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)

    with store.transaction():
        store.append_event(created.session_id, "outer.kept", status="complete")
        with pytest.raises(RuntimeError, match="inner failure"):
            with store.transaction():
                store.append_event(
                    created.session_id,
                    "inner.dropped",
                    status="complete",
                )
                raise RuntimeError("inner failure")

    kept = [item.event_type for item in store.all_events(created.session_id)]
    assert kept == ["outer.kept"]
    store.close()


def test_xhigh_authorizations_bind_one_active_command(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    other = session(tmp_path)
    store.create_session(other)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "escalate", "effort": "xhigh"},
        new_uuid(),
    )

    with pytest.raises(ValueError, match="provider is unsupported"):
        store.create_xhigh_authorization(
            created.session_id,
            command.command_id,
            "kimi",
            authorization_request_digest="a" * 64,
            idempotency_key="xhigh",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    with pytest.raises(ValueError, match="idempotency key is required"):
        store.create_xhigh_authorization(
            created.session_id,
            command.command_id,
            "codex",
            authorization_request_digest="a" * 64,
            idempotency_key="",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    with pytest.raises(NotFoundError):
        store.create_xhigh_authorization(
            created.session_id,
            new_uuid(),
            "codex",
            authorization_request_digest="a" * 64,
            idempotency_key="xhigh-missing",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    with pytest.raises(ConflictError, match="changed session"):
        store.create_xhigh_authorization(
            other.session_id,
            command.command_id,
            "codex",
            authorization_request_digest="a" * 64,
            idempotency_key="xhigh-session",
            expires_at="2099-01-01T00:00:00+00:00",
        )

    authorization = store.create_xhigh_authorization(
        created.session_id,
        command.command_id,
        "codex",
        authorization_request_digest="a" * 64,
        idempotency_key="xhigh",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    assert store.xhigh_authorization(command.command_id) == authorization
    assert (
        store.create_xhigh_authorization(
            created.session_id,
            command.command_id,
            "codex",
            authorization_request_digest="a" * 64,
            idempotency_key="xhigh",
            expires_at="2099-01-01T00:00:00+00:00",
        )
        == authorization
    )
    with pytest.raises(ConflictError, match="key was reused"):
        store.create_xhigh_authorization(
            created.session_id,
            command.command_id,
            "codex",
            authorization_request_digest="b" * 64,
            idempotency_key="xhigh",
            expires_at="2099-01-01T00:00:00+00:00",
        )

    lowered = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "ordinary", "effort": "high"},
        new_uuid(),
    )
    with pytest.raises(ConflictError, match="command effort changed"):
        store.create_xhigh_authorization(
            created.session_id,
            lowered.command_id,
            "codex",
            authorization_request_digest="c" * 64,
            idempotency_key="xhigh-effort",
            expires_at="2099-01-01T00:00:00+00:00",
        )

    finished = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "finished", "effort": "xhigh"},
        new_uuid(),
    )
    store.resolve_command(finished.command_id, "complete", {})
    with pytest.raises(ConflictError, match="command is not active"):
        store.create_xhigh_authorization(
            created.session_id,
            finished.command_id,
            "codex",
            authorization_request_digest="d" * 64,
            idempotency_key="xhigh-finished",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    store.close()


def test_route_admission_binds_envelope_worker_and_goal(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "route"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None

    def admit(**overrides: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "command_id": command.command_id,
            "provider": "codex",
            "profile": "unattended",
            "effort": "high",
            "worker_incarnation": "worker-1",
            "goal_id": "",
            "max_concurrency": 1,
            "lease_expires_at": "2099-01-01T00:00:00+00:00",
        }
        values.update(overrides)
        command_id = values.pop("command_id")
        provider = values.pop("provider")
        profile = values.pop("profile")
        return store.reserve_route_admission(
            command_id,
            provider,
            profile,
            **values,
        )

    with pytest.raises(ValueError, match="concurrency must be positive"):
        admit(max_concurrency=0)
    with pytest.raises(NotFoundError):
        admit()

    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        {"max_attempts": 1},
    )
    with pytest.raises(ConflictError, match="profile changed"):
        admit(profile="interactive")
    with pytest.raises(WorkerOwnershipLostError):
        admit()

    store.register_worker(created.session_id, 123, "worker-1")
    with pytest.raises(ConflictError, match="goal changed"):
        admit(goal_id=new_uuid())

    stale = _attempt(store, created.session_id)
    with pytest.raises(ConflictError, match="dispatch boundary is stale"):
        admit(attempt_id=stale.attempt_id)

    attempt = _dispatched_attempt(store, created.session_id, command.command_id)
    admission = admit(attempt_id=attempt.attempt_id)
    assert admission["admitted"] is True

    store.update_command_envelope(command.command_id, state="complete")
    with pytest.raises(ConflictError, match="envelope is not reservable"):
        admit()
    store.close()


def test_route_admission_requires_a_bound_xhigh_authorization(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    payload = {"text": "escalate", "effort": "xhigh"}
    command = store.enqueue_command(
        created.session_id,
        "message",
        payload,
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        {"max_attempts": 1},
    )
    store.register_worker(created.session_id, 123, "worker-1")
    attempt = _dispatched_attempt(store, created.session_id, command.command_id)

    def admit(attempt_id: str) -> dict[str, Any]:
        return store.reserve_route_admission(
            command.command_id,
            "codex",
            "unattended",
            effort="xhigh",
            attempt_id=attempt_id,
            worker_incarnation="worker-1",
            goal_id="",
            max_concurrency=1,
            lease_expires_at="2099-01-01T00:00:00+00:00",
        )

    assert admit(attempt.attempt_id)["reason"] == "xhigh-authorization"

    store.create_xhigh_authorization(
        created.session_id,
        command.command_id,
        "codex",
        authorization_request_digest=normalized_digest(payload),
        idempotency_key="xhigh",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    assert admit("")["reason"] == "xhigh-authorization"
    assert admit(attempt.attempt_id)["admitted"] is True
    store.close()


def test_context_delivery_preparation_and_acceptance_fail_closed(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "deliver"},
        new_uuid(),
    )
    first = _attempt(store, created.session_id)
    second = _attempt(store, created.session_id)

    prepared = store.prepare_context_delivery(
        created.session_id,
        "codex",
        "digest-1",
        "checkpoint-1",
        command.command_id,
        first.attempt_id,
        "payload-1",
    )
    assert prepared["state"] == "prepared"
    assert (
        store.prepare_context_delivery(
            created.session_id,
            "codex",
            "digest-1",
            "checkpoint-1",
            command.command_id,
            first.attempt_id,
            "payload-1",
        )
        == prepared
    )

    with pytest.raises(ConflictError, match="acceptance is stale"):
        store.accept_context_delivery(
            created.session_id,
            "codex",
            "digest-1",
            second.attempt_id,
        )

    accepted = store.accept_context_delivery(
        created.session_id,
        "codex",
        "digest-1",
        first.attempt_id,
    )
    assert accepted["state"] == "delivered"
    assert (
        store.accept_context_delivery(
            created.session_id,
            "codex",
            "digest-1",
            first.attempt_id,
        )["state"]
        == "delivered"
    )
    with pytest.raises(ConflictError, match="already delivered without native"):
        store.prepare_context_delivery(
            created.session_id,
            "codex",
            "digest-1",
            "checkpoint-1",
            new_uuid(),
            second.attempt_id,
            "payload-1",
        )
    store.close()


def test_context_delivery_rejects_ambiguous_prior_dispatches(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "deliver"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    first = _attempt(store, created.session_id)
    second = _dispatched_attempt(store, created.session_id, command.command_id)
    turn_id = store.start_turn(created.session_id, first.attempt_id)
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="codex",
        native_session_id="",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=utc_now(),
    )
    store.add_checkpoint(checkpoint)
    store.record_dispatch_checkpoint(
        command.command_id,
        first.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    store.prepare_context_delivery(
        created.session_id,
        "codex",
        "digest-1",
        checkpoint.checkpoint_id,
        command.command_id,
        first.attempt_id,
        "payload-1",
    )

    store.prepare_context_delivery(
        created.session_id,
        "codex",
        "digest-2",
        checkpoint.checkpoint_id,
        command.command_id,
        second.attempt_id,
        "payload-2",
    )

    third = _attempt(store, created.session_id)
    store.mark_provider_boundary(second.attempt_id)
    with pytest.raises(ConflictError, match="prior context delivery"):
        store.prepare_context_delivery(
            created.session_id,
            "codex",
            "digest-3",
            checkpoint.checkpoint_id,
            command.command_id,
            third.attempt_id,
            "payload-3",
        )
    with pytest.raises(ConflictError, match="delivery is ambiguous"):
        store.prepare_context_delivery(
            created.session_id,
            "codex",
            "digest-2",
            checkpoint.checkpoint_id,
            new_uuid(),
            third.attempt_id,
            "payload-3",
        )
    store.close()


def _forget_row(store: StateStore, statement: str, value: str) -> None:
    store._connection.execute("PRAGMA foreign_keys=OFF")
    try:
        with store.transaction() as connection:
            connection.execute(statement, (value,))
    finally:
        store._connection.execute("PRAGMA foreign_keys=ON")


def test_context_delivery_retargets_a_prepared_package(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    first = _attempt(store, created.session_id)
    second = _attempt(store, created.session_id)

    store.prepare_context_delivery(
        created.session_id,
        "codex",
        "digest-1",
        "checkpoint-1",
        new_uuid(),
        first.attempt_id,
        "payload-1",
    )
    retargeted = store.prepare_context_delivery(
        created.session_id,
        "codex",
        "digest-1",
        "checkpoint-2",
        new_uuid(),
        second.attempt_id,
        "payload-2",
    )

    assert retargeted["attempt_id"] == second.attempt_id
    assert retargeted["checkpoint_id"] == "checkpoint-2"
    assert retargeted["state"] == "prepared"
    store.close()


def test_route_admission_requires_its_session_and_command_rows(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    orphan = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "orphan"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    store.create_command_envelope(
        orphan.command_id,
        created.session_id,
        "unattended",
        {"max_attempts": 1},
    )
    # A durable session row can be lost to operator surgery while its
    # envelope survives, and admission has to fail closed on that state.
    _forget_row(store, "DELETE FROM sessions WHERE session_id = ?", created.session_id)

    with pytest.raises(NotFoundError):
        store.reserve_route_admission(
            orphan.command_id,
            "codex",
            "unattended",
            worker_incarnation="worker-1",
            goal_id="",
            max_concurrency=1,
            lease_expires_at="2099-01-01T00:00:00+00:00",
        )

    restored = session(tmp_path)
    store.create_session(restored)
    escalated = store.enqueue_command(
        restored.session_id,
        "message",
        {"text": "escalate", "effort": "xhigh"},
        new_uuid(),
    )
    assert store.claim_command(restored.session_id) is not None
    store.create_command_envelope(
        escalated.command_id,
        restored.session_id,
        "unattended",
        {"max_attempts": 1},
    )
    store.register_worker(restored.session_id, 123, "worker-1")
    _forget_row(
        store,
        "DELETE FROM commands WHERE command_id = ?",
        escalated.command_id,
    )
    with pytest.raises(NotFoundError):
        store.reserve_route_admission(
            escalated.command_id,
            "codex",
            "unattended",
            effort="xhigh",
            attempt_id=new_uuid(),
            worker_incarnation="worker-1",
            goal_id="",
            max_concurrency=1,
            lease_expires_at="2099-01-01T00:00:00+00:00",
        )
    store.close()


def test_durable_lookups_fail_closed_on_absent_rows(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)

    with pytest.raises(NotFoundError):
        store.dispatch_transition_anchor("absent-session")

    goal = create_goal(
        created.session_id,
        "Prove the bounded stage.",
        milestones=(
            {
                "milestone_id": "one",
                "title": "One",
                "dependencies": [],
                "predicates": [{"type": "report", "outcome": "passed"}],
            },
        ),
    )
    store.create_goal(goal)
    with pytest.raises(NotFoundError):
        store.update_milestone_statuses(
            goal.goal_id,
            (replace(goal.milestones[0], milestone_id="absent"),),
        )

    other = session(tmp_path)
    store.create_session(other)
    with pytest.raises(ConflictError, match="evidence goal is no longer current"):
        store.add_evidence_once(
            other.session_id,
            make_evidence(goal.goal_id, "report", "one", "passed"),
            idempotency_key="evidence-1",
            request_digest="a" * 64,
        )
    with pytest.raises(NotFoundError):
        store.add_evidence_once(
            "absent-session",
            make_evidence(goal.goal_id, "report", "one", "passed"),
            idempotency_key="evidence-2",
            request_digest="b" * 64,
        )

    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "escalate", "effort": "xhigh"},
        new_uuid(),
    )
    with pytest.raises(ConflictError, match="not awaiting authorization"):
        store.xhigh_authorization_or_park(command.command_id)
    store.close()


def test_retained_transition_policies_reject_corrupt_material(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    goal = store.create_goal(
        create_goal(created.session_id, "Advance one exact stage.")
    )
    policy = {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": created.session_id,
        "epoch_id": "epoch-1",
    }
    policy_sha256 = normalized_digest(policy)
    with store.transaction() as connection:
        retained = store._retain_dispatch_transition_policy(
            connection,
            session_id=created.session_id,
            authorization={
                "policy": policy,
                "policy_sha256": policy_sha256,
                "goal_id": goal.goal_id,
                "epoch_id": "epoch-1",
            },
            created_at=utc_now(),
        )
    assert retained["policy_ref"]["policy_sha256"] == policy_sha256
    assert (
        store.dispatch_transition_policy(
            created.session_id,
            goal.goal_id,
            "epoch-1",
            policy_sha256,
        )
        == policy
    )

    with store.transaction() as connection:
        with pytest.raises(ConflictError, match="policy is missing"):
            store._retain_dispatch_transition_policy(
                connection,
                session_id=created.session_id,
                authorization={"policy_sha256": policy_sha256},
                created_at=utc_now(),
            )
        with pytest.raises(ConflictError, match="policy digest changed"):
            store._retain_dispatch_transition_policy(
                connection,
                session_id=created.session_id,
                authorization={"policy": policy, "policy_sha256": "a" * 64},
                created_at=utc_now(),
            )
        with pytest.raises(ConflictError, match="epoch policy changed"):
            store._retain_dispatch_transition_policy(
                connection,
                session_id=created.session_id,
                authorization={
                    "policy": {**policy, "extra": 1},
                    "policy_sha256": normalized_digest({**policy, "extra": 1}),
                    "goal_id": goal.goal_id,
                    "epoch_id": "epoch-1",
                },
                created_at=utc_now(),
            )

    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE dispatch_transition_policies SET payload_json = ?
            WHERE policy_sha256 = ?
            """,
            ('{"schema": "corrupt"}', policy_sha256),
        )
    with pytest.raises(ConflictError, match="retained policy is corrupt"):
        store.dispatch_transition_policy(
            created.session_id,
            goal.goal_id,
            "epoch-1",
            policy_sha256,
        )
    store.close()


def _ambiguous(store: StateStore, created: Any) -> tuple[Any, Any]:
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "one effect"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    attempt = _dispatched_attempt(store, created.session_id, command.command_id)
    store.mark_provider_boundary(attempt.attempt_id)
    recovery = store.recover_interrupted_commands(
        created.session_id,
        "moved-digest",
        "summary",
    )
    assert len(recovery.reconciliations) == 1
    return command, recovery.reconciliations[0]


def _new_checkpoint(store: StateStore, session_id: str) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=session_id,
        sequence=store.last_sequence(session_id),
        provider="codex",
        native_session_id="",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=utc_now(),
    )


def test_reconciliation_discovery_checkpoints_are_bound_and_idempotent(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    other = session(tmp_path)
    store.create_session(other)
    unused_command, record = _ambiguous(store, created)
    del unused_command
    checkpoint = _new_checkpoint(store, created.session_id)

    with pytest.raises(NotFoundError):
        store.record_reconciliation_discovery(
            new_uuid(),
            checkpoint,
            record.current_workspace_digest,
        )
    with pytest.raises(ConflictError, match="belongs to another session"):
        store.record_reconciliation_discovery(
            record.reconciliation_id,
            replace(checkpoint, session_id=other.session_id),
            record.current_workspace_digest,
        )
    with pytest.raises(ConflictError, match="workspace digest is stale"):
        store.record_reconciliation_discovery(
            record.reconciliation_id,
            checkpoint,
            "other-digest",
        )

    discovered = store.record_reconciliation_discovery(
        record.reconciliation_id,
        checkpoint,
        record.current_workspace_digest,
    )
    assert discovered.audit["discovery_checkpoint_id"] == checkpoint.checkpoint_id

    repeated = store.record_reconciliation_discovery(
        record.reconciliation_id,
        _new_checkpoint(store, created.session_id),
        record.current_workspace_digest,
    )
    assert repeated.audit["discovery_checkpoint_id"] == checkpoint.checkpoint_id

    _forget_row(
        store,
        "DELETE FROM checkpoints WHERE checkpoint_id = ?",
        checkpoint.checkpoint_id,
    )
    with pytest.raises(RuntimeError, match="discovery checkpoint is missing"):
        store.record_reconciliation_discovery(
            record.reconciliation_id,
            _new_checkpoint(store, created.session_id),
            record.current_workspace_digest,
        )
    store.close()


def test_reconciliation_resolution_receipts_are_exactly_once(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    unused_command, record = _ambiguous(store, created)
    del unused_command

    def resolve(
        decision: str,
        digest: str,
        key: str,
        *,
        request_digest: str = "a" * 64,
    ) -> tuple[Any, bool]:
        return store.resolve_reconciliation_once(
            record.reconciliation_id,
            decision,
            digest,
            dict(record.audit),
            None,
            idempotency_key=key,
            operation="reconciliation-resolve:" + record.reconciliation_id,
            request_digest=request_digest,
        )

    with pytest.raises(NotFoundError):
        store.resolve_reconciliation_once(
            new_uuid(),
            "stop",
            record.current_workspace_digest,
            {},
            None,
            idempotency_key="absent",
            operation="reconciliation-resolve:absent",
            request_digest="a" * 64,
        )
    with pytest.raises(ConflictError, match="observed workspace digest is stale"):
        resolve("stop", "other-digest", "stale-digest")

    resolved, created_first = resolve(
        "stop",
        record.current_workspace_digest,
        "resolve-1",
    )
    assert created_first is True
    assert resolved.status == "resolved"
    assert store.get_session(created.session_id).lifecycle == "stopped"

    repeated, created_again = resolve(
        "stop",
        record.current_workspace_digest,
        "resolve-1",
    )
    assert created_again is False
    assert repeated.reconciliation_id == resolved.reconciliation_id

    with pytest.raises(ConflictError, match="already used for another mutation"):
        resolve(
            "stop",
            record.current_workspace_digest,
            "resolve-1",
            request_digest="b" * 64,
        )
    with pytest.raises(ConflictError, match="already resolved differently"):
        resolve("accept-current", record.current_workspace_digest, "resolve-2")
    store.close()


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "--import-mode=importlib",
                *sys.argv[1:],
            ]
        )
    )
