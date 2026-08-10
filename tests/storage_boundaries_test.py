"""Deterministic coverage for durable storage boundary behavior."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sqlite3
import subprocess
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
from agent_harness.models import CommandStatus
from agent_harness.models import ProviderAttempt
from agent_harness.errors import WorkerOwnershipLostError
from agent_harness.goals import create_goal
from agent_harness.goals import make_evidence
from agent_harness.orchestration import creation_digest
from agent_harness.orchestration import command_envelope_digest
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


def test_portable_import_migrates_legacy_context_delivery_identity(
    tmp_path: Path,
) -> None:
    source = StateStore(tmp_path / "source.sqlite3")
    created = session(tmp_path)
    source.create_session(created)
    record = source.portable_session(created.session_id)
    source.close()
    _, global_record = _portable_empty()

    missing_attempt = {
        "session_id": created.session_id,
        "provider": "codex",
        "context_digest": "legacy-missing-attempt",
        "checkpoint_id": "",
        "delivered_at": "2026-08-02T00:00:00+00:00",
    }
    colliding_attempt_id = "legacy-" + normalized_digest(
        {
            "attempt_id": "",
            "row": missing_attempt,
        }
    )
    source_rows = [
        missing_attempt,
        copy.deepcopy(missing_attempt),
        {
            **missing_attempt,
            "context_digest": "legacy-null-attempt",
            "attempt_id": None,
        },
        {
            **missing_attempt,
            "provider": "claude",
            "context_digest": "legacy-duplicate-one",
            "attempt_id": "duplicate-attempt",
        },
        {
            **missing_attempt,
            "context_digest": "legacy-duplicate-two",
            "attempt_id": "duplicate-attempt",
        },
        {
            **missing_attempt,
            "context_digest": "preserved-explicit-attempt",
            "attempt_id": colliding_attempt_id,
        },
    ]
    record["tables"]["context_deliveries"] = source_rows
    original_record = copy.deepcopy(record)

    first = StateStore(tmp_path / "first.sqlite3")
    first.import_portable([record], global_record)
    first_record = first.portable_session(created.session_id)
    first.close()

    reordered_record = copy.deepcopy(record)
    reordered_record["tables"]["context_deliveries"].reverse()
    second = StateStore(tmp_path / "second.sqlite3")
    second.import_portable([reordered_record], global_record)
    second_record = second.portable_session(created.session_id)
    second.close()

    assert record == original_record
    assert reordered_record["tables"]["context_deliveries"] == list(
        reversed(source_rows)
    )
    assert first_record == second_record
    rows = first_record["tables"]["context_deliveries"]
    assert len(rows) == 6
    by_context = {str(row["context_digest"]): row for row in rows}
    missing_rows = [
        row
        for row in rows
        if row["context_digest"] == "legacy-missing-attempt"
    ]
    assert len(missing_rows) == 2
    missing_ids = {str(row["attempt_id"]) for row in missing_rows}
    assert len(missing_ids) == 2
    assert all(value.startswith("legacy-") for value in missing_ids)
    assert colliding_attempt_id not in missing_ids
    null_id = str(by_context["legacy-null-attempt"]["attempt_id"])
    assert null_id.startswith("legacy-")
    assert not null_id.startswith("legacy-duplicate-")
    first_duplicate = str(by_context["legacy-duplicate-one"]["attempt_id"])
    second_duplicate = str(by_context["legacy-duplicate-two"]["attempt_id"])
    assert first_duplicate.startswith("legacy-duplicate-")
    assert second_duplicate.startswith("legacy-duplicate-")
    assert first_duplicate != second_duplicate
    assert (
        by_context["preserved-explicit-attempt"]["attempt_id"]
        == colliding_attempt_id
    )
    preserved = missing_rows[0]
    assert preserved["session_id"] == created.session_id
    assert preserved["provider"] == "codex"
    assert preserved["checkpoint_id"] == ""
    assert preserved["delivered_at"] == "2026-08-02T00:00:00+00:00"
    assert preserved["command_id"] == ""
    assert preserved["state"] == "delivered"
    assert preserved["payload_digest"] == ""
    assert preserved["accepted_at"] == ""
    assert preserved["transport"] == "context-package"
    assert by_context["legacy-duplicate-one"]["provider"] == "claude"


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


def test_dispatch_timestamp_is_sampled_under_the_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "order this dispatch"},
        new_uuid(),
    )
    attempt = _attempt(store, created.session_id)
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
    original_now = storage_module.utc_now
    lock_observations: list[bool] = []

    def observe_lock() -> str:
        lock_observations.append(store._lock._is_owned())
        return original_now()

    monkeypatch.setattr(storage_module, "utc_now", observe_lock)
    store.record_dispatch_checkpoint(
        command.command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )

    assert lock_observations == [True]
    store.close()


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
            "",
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
    assert prepared["transport"] == "context-package"
    with pytest.raises(ValueError, match="transport is unsupported"):
        store.prepare_context_delivery(
            created.session_id,
            "codex",
            "digest-invalid",
            "checkpoint-1",
            command.command_id,
            first.attempt_id,
            "payload-1",
            transport="unsupported",
        )
    with pytest.raises(ConflictError, match="transport changed"):
        store.prepare_context_delivery(
            created.session_id,
            "codex",
            "digest-1",
            "checkpoint-1",
            command.command_id,
            first.attempt_id,
            "payload-1",
            transport="native-resume",
        )
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
    with pytest.raises(ConflictError, match="acceptance is stale"):
        store.accept_context_delivery(
            created.session_id,
            "claude",
            "digest-1",
            first.attempt_id,
        )
    with pytest.raises(ConflictError, match="acceptance is stale"):
        store.accept_context_delivery(
            created.session_id,
            "codex",
            "wrong-digest",
            first.attempt_id,
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
    store.complete_dispatch(second.attempt_id, "failed")
    with store.transaction() as connection:
        connection.execute(
            "UPDATE context_deliveries SET state = 'prepared', accepted_at = ? "
            "WHERE attempt_id = ?",
            (utc_now(), second.attempt_id),
        )
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
    with store.transaction() as connection:
        connection.execute(
            "UPDATE context_deliveries SET state = 'legacy-ambiguous', "
            "accepted_at = '' WHERE attempt_id = ?",
            (second.attempt_id,),
        )
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
    store.close()


def _forget_row(store: StateStore, statement: str, value: str) -> None:
    store._connection.execute("PRAGMA foreign_keys=OFF")
    try:
        with store.transaction() as connection:
            connection.execute(statement, (value,))
    finally:
        store._connection.execute("PRAGMA foreign_keys=ON")


def test_context_delivery_without_dispatch_cannot_be_retargeted(
    tmp_path: Path,
) -> None:
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
    with pytest.raises(ConflictError, match="delivery is ambiguous"):
        store.prepare_context_delivery(
            created.session_id,
            "codex",
            "digest-1",
            "checkpoint-2",
            new_uuid(),
            second.attempt_id,
            "payload-2",
        )

    deliveries = store.portable_session(created.session_id)["tables"][
        "context_deliveries"
    ]
    assert [item["state"] for item in deliveries] == ["prepared"]
    store.close()


def test_context_delivery_retargets_a_proven_preboundary_failure(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "prove non-delivery"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    first = _dispatched_attempt(
        store,
        created.session_id,
        command.command_id,
    )
    store.prepare_context_delivery(
        created.session_id,
        "codex",
        "digest-1",
        "checkpoint-1",
        command.command_id,
        first.attempt_id,
        "payload-1",
    )
    store.complete_dispatch(first.attempt_id, "failed")
    second = _attempt(store, created.session_id)

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
    deliveries = store.portable_session(created.session_id)["tables"][
        "context_deliveries"
    ]
    assert [item["state"] for item in deliveries] == [
        "superseded",
        "prepared",
    ]
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


def test_proof_command_and_transition_helper_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StateStore(tmp_path / "proof.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    store.append_event(created.session_id, "one", status="complete")
    monkeypatch.setattr(
        store,
        "proof_event_rows",
        lambda *unused: [{"sequence": 2}],
    )
    with pytest.raises(ValueError, match="sequence is not contiguous"):
        store.proof_source(created.session_id, 1, 10)

    class Cursor:
        def __init__(self, value: object) -> None:
            self.value = value

        def fetchone(self) -> object:
            return self.value

    class MissingAggregateConnection:
        def execute(
            self,
            statement: str,
            unused_parameters: tuple[object, ...],
        ) -> Cursor:
            if "SELECT status FROM commands" in statement:
                return Cursor({"status": CommandStatus.FAILED})
            return Cursor(None)

    probe = object.__new__(StateStore)
    probe._lock = threading.RLock()
    probe._connection = MissingAggregateConnection()  # type: ignore[assignment]
    assert probe.command_failed_before_provider_boundary("command") is True

    complete = {
        "command_type": "message",
        "status": CommandStatus.COMPLETE,
        "result_json": storage_module._dump(
            {
                "checkpoint_id": "old",
                "workspace_material_digest": "a" * 64,
            }
        ),
        "command_id": "command",
    }
    with pytest.raises(ConflictError, match="checkpoint is not latest"):
        storage_module._dispatch_transition_anchor(
            store._connection,
            complete,  # type: ignore[arg-type]
            "latest",
            "a" * 64,
        )
    complete["result_json"] = storage_module._dump(
        {
            "checkpoint_id": "latest",
            "workspace_material_digest": "short",
        }
    )
    with pytest.raises(ConflictError, match="material is not current"):
        storage_module._dispatch_transition_anchor(
            store._connection,
            complete,  # type: ignore[arg-type]
            "latest",
            "a" * 64,
        )

    failed = {
        "command_type": "message",
        "status": CommandStatus.FAILED,
        "result_json": storage_module._dump({"code": "E_SAFETY_GUARD"}),
        "command_id": "missing-reconciliation",
    }
    with pytest.raises(ConflictError, match="exactly one reconciliation"):
        storage_module._dispatch_transition_anchor(
            store._connection,
            failed,  # type: ignore[arg-type]
            "latest",
            "a" * 64,
        )

    class ReconciliationCursor:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[dict[str, Any]]:
            return self.rows

    class ReconciliationConnection:
        def __init__(self, row: dict[str, Any]) -> None:
            self.row = row

        def execute(
            self,
            unused_statement: str,
            unused_parameters: tuple[object, ...],
        ) -> ReconciliationCursor:
            return ReconciliationCursor([self.row])

    reconciliation = {
        "status": "resolved",
        "resolution": "stop",
        "audit_json": storage_module._dump(
            {
                "resolution_checkpoint_id": "latest",
                "resolution_workspace_digest": "a" * 64,
            }
        ),
        "reconciliation_id": "reconciliation",
    }
    with pytest.raises(ConflictError, match="resolution is unsafe"):
        storage_module._dispatch_transition_anchor(
            ReconciliationConnection(reconciliation),  # type: ignore[arg-type]
            failed,  # type: ignore[arg-type]
            "latest",
            "a" * 64,
        )
    reconciliation["resolution"] = "accept-current"
    reconciliation["audit_json"] = storage_module._dump(
        {
            "resolution_checkpoint_id": "old",
            "resolution_workspace_digest": "a" * 64,
        }
    )
    with pytest.raises(ConflictError, match="checkpoint is not latest"):
        storage_module._dispatch_transition_anchor(
            ReconciliationConnection(reconciliation),  # type: ignore[arg-type]
            failed,  # type: ignore[arg-type]
            "latest",
            "a" * 64,
        )
    reconciliation["audit_json"] = storage_module._dump(
        {
            "resolution_checkpoint_id": "latest",
            "resolution_workspace_digest": "a" * 64,
        }
    )
    anchor = storage_module._dispatch_transition_anchor(
        ReconciliationConnection(reconciliation),  # type: ignore[arg-type]
        failed,  # type: ignore[arg-type]
        "latest",
        "a" * 64,
    )
    assert anchor["prior_anchor_kind"] == "resolved-reconciliation"
    assert anchor["prior_reconciliation_id"] == "reconciliation"
    store.close()


def test_transition_epoch_validation_rejects_each_durable_drift(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "epoch.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    goal = store.create_goal(create_goal(created.session_id, "Bind an epoch."))
    policy = {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": created.session_id,
        "epoch_id": "epoch-one",
    }
    policy_digest = normalized_digest(policy)
    connection = store._connection

    assert not storage_module._dispatch_transition_epoch_is_active(
        connection,
        created.session_id,
        {},
    )
    assert not storage_module._dispatch_transition_epoch_is_active(
        connection,
        created.session_id,
        {
            "policy_sha256": "a" * 64,
            "epoch_id": "epoch-one",
            "goal_id": goal.goal_id,
            "policy": policy,
        },
    )
    referenced = {
        "policy_sha256": policy_digest,
        "epoch_id": "epoch-one",
        "goal_id": goal.goal_id,
        "policy_ref": {},
    }
    assert not storage_module._dispatch_transition_epoch_is_active(
        connection,
        created.session_id,
        referenced,
    )
    referenced["policy_ref"] = {
        "policy_sha256": policy_digest,
        "session_id": created.session_id,
        "goal_id": goal.goal_id,
        "epoch_id": "epoch-one",
    }
    assert not storage_module._dispatch_transition_epoch_is_active(
        connection,
        created.session_id,
        referenced,
    )
    with store.transaction() as current:
        current.execute(
            "INSERT INTO dispatch_transition_policies VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                policy_digest,
                created.session_id,
                goal.goal_id,
                "epoch-one",
                str(policy["schema"]),
                "{}",
                utc_now(),
            ),
        )
    assert not storage_module._dispatch_transition_epoch_is_active(
        connection,
        created.session_id,
        referenced,
    )
    for changed_policy in (
        {**policy, "session_id": "other"},
        {**policy, "epoch_id": "other"},
    ):
        assert not storage_module._dispatch_transition_epoch_is_active(
            connection,
            created.session_id,
            {
                "policy_sha256": normalized_digest(changed_policy),
                "epoch_id": str(changed_policy["epoch_id"]),
                "goal_id": goal.goal_id,
                "policy": changed_policy,
            },
        )
    store.close()

    absent = StateStore(tmp_path / "absent-epoch.sqlite3")
    assert not storage_module._dispatch_transition_epoch_is_active(
        absent._connection,
        created.session_id,
        {
            "policy_sha256": policy_digest,
            "epoch_id": "epoch-one",
            "goal_id": goal.goal_id,
            "policy": policy,
        },
    )
    absent.close()


def test_transition_epoch_rejects_a_policy_with_a_changed_epoch(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "changed-epoch.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    goal = store.create_goal(create_goal(created.session_id, "Bind an epoch."))
    policy = {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": created.session_id,
        "epoch_id": "other-epoch",
    }

    assert not storage_module._dispatch_transition_epoch_is_active(
        store._connection,
        created.session_id,
        {
            "policy_sha256": normalized_digest(policy),
            "epoch_id": "epoch-one",
            "goal_id": goal.goal_id,
            "policy": policy,
        },
    )
    store.close()


def _promotion_store(
    root: Path,
    name: str,
    *,
    complete_session: bool = True,
    complete_goal: bool = True,
) -> tuple[StateStore, Any, Any, Any]:
    store = StateStore(root / (name + ".sqlite3"))
    created = session(root / name)
    store.create_session(created)
    previous = store.create_goal(create_goal(created.session_id, "First goal."))
    if complete_goal:
        store.update_goal_status(previous.goal_id, "complete")
        previous = store.get_goal(previous.goal_id)
    if complete_session:
        store.update_session(created.session_id, lifecycle="completed")
    next_goal = create_goal(created.session_id, "Next goal.")
    return store, created, previous, next_goal


def _promote(
    store: StateStore,
    previous: Any,
    next_goal: Any,
    key: str,
    request_digest: str = "request",
) -> dict[str, Any]:
    authorization: dict[str, Any] = {}
    return store.promote_goal(
        previous,
        next_goal,
        stage="next",
        authorization_digest=normalized_digest(authorization),
        authorization=authorization,
        request_digest=request_digest,
        idempotency_key=key,
    )


def test_goal_promotion_rejects_every_durable_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, created, previous, next_goal = _promotion_store(tmp_path, "replay")
    accepted = _promote(store, previous, next_goal, "promotion")
    assert _promote(store, previous, next_goal, "promotion") == accepted
    with pytest.raises(ConflictError, match="key was reused"):
        _promote(store, previous, next_goal, "promotion", "changed")
    assert store.goal_promotions(created.session_id) == [accepted]
    store.close()

    missing_store = StateStore(tmp_path / "missing-promotion.sqlite3")
    missing_previous = create_goal(new_uuid(), "Missing.")
    with pytest.raises(NotFoundError, match="session"):
        _promote(
            missing_store,
            missing_previous,
            create_goal(missing_previous.session_id, "Next."),
            "missing",
        )
    missing_store.close()

    store, unused_created, previous, next_goal = _promotion_store(
        tmp_path,
        "stale-source",
    )
    with pytest.raises(ConflictError, match="source is not current"):
        _promote(store, replace(previous, goal_id=new_uuid()), next_goal, "stale")
    store.close()

    store, unused_created, previous, next_goal = _promotion_store(
        tmp_path,
        "active-session",
        complete_session=False,
    )
    with pytest.raises(ConflictError, match="completed session"):
        _promote(store, previous, next_goal, "active-session")
    store.close()

    store, created, previous, next_goal = _promotion_store(tmp_path, "missing-goal")
    _forget_row(store, "DELETE FROM goals WHERE goal_id = ?", previous.goal_id)
    with pytest.raises(NotFoundError, match="goal"):
        _promote(store, previous, next_goal, "missing-goal")
    store.close()

    store, unused_created, previous, next_goal = _promotion_store(
        tmp_path,
        "active-goal",
        complete_goal=False,
    )
    with pytest.raises(ConflictError, match="completed goal"):
        _promote(store, previous, next_goal, "active-goal")
    store.close()

    store, created, previous, next_goal = _promotion_store(tmp_path, "command")
    store.enqueue_command(created.session_id, "message", {"text": "active"}, "active")
    with pytest.raises(ConflictError, match="command quiescence"):
        _promote(store, previous, next_goal, "command")
    store.close()

    store, created, previous, next_goal = _promotion_store(tmp_path, "approval")
    store.create_approval(created.session_id, "turn", "request", "tool", "Allow?", [])
    with pytest.raises(ConflictError, match="pending approval"):
        _promote(store, previous, next_goal, "approval")
    store.close()

    store, created, previous, next_goal = _promotion_store(
        tmp_path,
        "reconciliation",
    )
    unused_command, unused_record = _ambiguous(store, created)
    del unused_command, unused_record
    store.update_session(created.session_id, lifecycle="completed")
    with pytest.raises(ConflictError, match="reconciliation barrier"):
        _promote(store, previous, next_goal, "reconciliation")
    store.close()

    store, unused_created, previous, next_goal = _promotion_store(
        tmp_path,
        "authorization",
    )
    monkeypatch.setattr(store, "_insert_authorization_receipt", lambda *args, **kwargs: "changed")
    with pytest.raises(ValueError, match="authorization digest changed"):
        _promote(store, previous, next_goal, "authorization")
    monkeypatch.undo()
    store.close()

    store, unused_created, previous, next_goal = _promotion_store(
        tmp_path,
        "missing-receipt",
    )
    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER ignore_goal_promotion
            BEFORE INSERT ON goal_promotions
            BEGIN SELECT RAISE(IGNORE); END
            """
        )
    with pytest.raises(RuntimeError, match="receipt was not recorded"):
        _promote(store, previous, next_goal, "missing-receipt")
    store.close()


def _adoption_store(
    root: Path,
    name: str,
) -> tuple[StateStore, Any, Any, dict[str, str], dict[str, Any]]:
    store = StateStore(root / (name + ".sqlite3"))
    created = session(root / name)
    store.create_session(created)
    next_goal = create_goal(created.session_id, "Adopted goal.")
    external_ref = {"orchestrator": "machines", "job_id": name}
    creation_input = {
        "workspace": created.workspace,
        "name": "adopted",
        "permission_mode": created.permission_mode,
        "execution_profile": "unattended",
        "external_ref": external_ref,
        "routing": {},
    }
    return store, created, next_goal, external_ref, creation_input


def _adopt(
    store: StateStore,
    created: Any,
    next_goal: Any,
    external_ref: dict[str, str],
    creation_input: dict[str, Any],
    key: str,
    request_digest: str = "request",
) -> dict[str, Any]:
    authorization: dict[str, Any] = {}
    return store.adopt_session_contract(
        created.session_id,
        next_goal,
        external_ref=external_ref,
        creation_input=creation_input,
        authorization_digest=normalized_digest(authorization),
        authorization=authorization,
        request_digest=request_digest,
        idempotency_key=key,
    )


def test_contract_adoption_rejects_every_durable_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, created, next_goal, external_ref, creation = _adoption_store(
        tmp_path,
        "adoption-replay",
    )
    accepted = _adopt(store, created, next_goal, external_ref, creation, "adoption")
    assert _adopt(store, created, next_goal, external_ref, creation, "adoption") == accepted
    with pytest.raises(ConflictError, match="key was reused"):
        _adopt(
            store,
            created,
            next_goal,
            external_ref,
            creation,
            "adoption",
            "changed",
        )
    store.close()

    missing = StateStore(tmp_path / "missing-adoption.sqlite3")
    missing_session = session(tmp_path / "missing-adoption")
    with pytest.raises(NotFoundError, match="session"):
        _adopt(
            missing,
            missing_session,
            create_goal(missing_session.session_id, "Missing."),
            {"orchestrator": "machines", "job_id": "missing"},
            {"routing": {}},
            "missing",
        )
    missing.close()

    store, created, next_goal, external_ref, creation = _adoption_store(
        tmp_path,
        "adoption-command",
    )
    store.enqueue_command(created.session_id, "message", {"text": "active"}, "active")
    with pytest.raises(ConflictError, match="requires quiescence"):
        _adopt(store, created, next_goal, external_ref, creation, "command")
    store.close()

    store, created, next_goal, external_ref, creation = _adoption_store(
        tmp_path,
        "adoption-lease",
    )
    store.create_process_lease(
        created.session_id,
        "codex",
        "unattended",
        "2099-01-01T00:00:00+00:00",
    )
    with pytest.raises(ConflictError, match="active process lease"):
        _adopt(store, created, next_goal, external_ref, creation, "lease")
    store.close()

    store, created, next_goal, external_ref, creation = _adoption_store(
        tmp_path,
        "adoption-working",
    )
    store.update_session(created.session_id, attention="working")
    with pytest.raises(ConflictError, match="idle session"):
        _adopt(store, created, next_goal, external_ref, creation, "working")
    store.close()

    store, created, next_goal, external_ref, creation = _adoption_store(
        tmp_path,
        "adoption-ref",
    )
    other = replace(session(tmp_path / "adoption-ref-other"), external_ref=external_ref)
    store.create_session(other)
    with pytest.raises(ConflictError, match="belongs to another session"):
        _adopt(store, created, next_goal, external_ref, creation, "ref")
    store.close()

    store, created, next_goal, external_ref, creation = _adoption_store(
        tmp_path,
        "adoption-authorization",
    )
    monkeypatch.setattr(store, "_insert_authorization_receipt", lambda *args, **kwargs: "changed")
    with pytest.raises(ValueError, match="authorization digest changed"):
        _adopt(store, created, next_goal, external_ref, creation, "authorization")
    monkeypatch.undo()
    store.close()

    store, created, next_goal, external_ref, creation = _adoption_store(
        tmp_path,
        "adoption-routing",
    )

    class ChangingCreation(dict[str, Any]):
        def __init__(self, values: dict[str, Any]) -> None:
            super().__init__(values)
            self.routing_reads = 0

        def get(self, key: str, default: object = None) -> object:
            if key == "routing":
                self.routing_reads += 1
                if self.routing_reads > 1:
                    return ["malformed"]
            return super().get(key, default)

    changing_creation = ChangingCreation(creation)
    routed = _adopt(
        store,
        created,
        next_goal,
        external_ref,
        changing_creation,
        "routing",
    )
    assert routed["next_goal_id"] == next_goal.goal_id
    store.close()

    store, created, next_goal, external_ref, creation = _adoption_store(
        tmp_path,
        "adoption-missing-receipt",
    )
    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER ignore_contract_adoption
            BEFORE INSERT ON goal_contract_adoptions
            BEGIN SELECT RAISE(IGNORE); END
            """
        )
    with pytest.raises(RuntimeError, match="receipt was not recorded"):
        _adopt(store, created, next_goal, external_ref, creation, "missing-receipt")
    store.close()


def test_post_insert_guards_and_xhigh_consumption_fail_closed(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "post-insert.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    xhigh = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "maximum", "provider": "codex", "effort": "xhigh"},
        "xhigh",
    )
    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER ignore_xhigh_authorization
            BEFORE INSERT ON xhigh_authorization_receipts
            BEGIN SELECT RAISE(IGNORE); END
            """
        )
    with pytest.raises(RuntimeError, match="authorization was not recorded"):
        store.create_xhigh_authorization(
            created.session_id,
            xhigh.command_id,
            "codex",
            authorization_request_digest="a" * 64,
            idempotency_key="ignored-xhigh",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    with store.transaction() as connection:
        connection.execute("DROP TRIGGER ignore_xhigh_authorization")

    gated = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "gate"},
        "gate",
    )
    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER ignore_child_gate
            BEFORE INSERT ON child_launch_gates
            BEGIN SELECT RAISE(IGNORE); END
            """
        )
    with pytest.raises(RuntimeError, match="gate was not created"):
        store.create_child_launch_gate(gated.command_id, created.session_id, 1)
    store.close()

    route = StateStore(tmp_path / "xhigh-route.sqlite3")
    routed_session = session(tmp_path / "xhigh-route")
    route.create_session(routed_session)
    command = route.enqueue_command(
        routed_session.session_id,
        "message",
        {"text": "route", "provider": "codex", "effort": "xhigh"},
        "route",
    )
    route.create_command_envelope(
        command.command_id,
        routed_session.session_id,
        "unattended",
        {"max_attempts": 1},
    )
    route.register_worker(routed_session.session_id, 123, "worker")
    route.create_xhigh_authorization(
        routed_session.session_id,
        command.command_id,
        "codex",
        authorization_request_digest="b" * 64,
        idempotency_key="route-authorization",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    with route.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER ignore_xhigh_consumption
            BEFORE UPDATE OF consumed_at ON xhigh_authorization_receipts
            BEGIN SELECT RAISE(IGNORE); END
            """
        )
    denied = route.reserve_route_admission(
        command.command_id,
        "codex",
        "unattended",
        effort="xhigh",
        attempt_id=new_uuid(),
        worker_incarnation="worker",
        goal_id="",
        max_concurrency=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    assert denied["reason"] == "xhigh-authorization"
    route.close()


def _ambiguous_store(
    root: Path,
    name: str,
) -> tuple[StateStore, Any, Any, Any]:
    store = StateStore(root / (name + ".sqlite3"))
    created = session(root / name)
    store.create_session(created)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "ambiguous"},
        name,
    )
    assert store.claim_command(created.session_id) is not None
    attempt = _dispatched_attempt(store, created.session_id, command.command_id)
    store.mark_provider_boundary(attempt.attempt_id)
    recovery = store.recover_interrupted_commands(
        created.session_id,
        "workspace-digest",
        "summary",
    )
    assert len(recovery.reconciliations) == 1
    return store, created, command, recovery.reconciliations[0]


def _resolve_ambiguous(
    store: StateStore,
    record: Any,
    key: str,
    *,
    audit: dict[str, Any] | None = None,
    decision: str = "stop",
) -> tuple[Any, bool]:
    selected_audit = dict(record.audit)
    if audit is not None:
        selected_audit = audit
    return store.resolve_reconciliation_once(
        record.reconciliation_id,
        decision,
        record.current_workspace_digest,
        selected_audit,
        None,
        idempotency_key=key,
        operation="resolve:" + record.reconciliation_id,
        request_digest="request",
    )


def _insert_bound_lease(
    store: StateStore,
    created: Any,
    command: Any,
    attempt_id: str,
    name: str,
) -> None:
    now = utc_now()
    with store.transaction() as connection:
        connection.execute(
            """
            INSERT INTO process_leases VALUES (
                ?, ?, ?, ?, 'worker', 'codex', 'unattended',
                123, 'start', 'active', ?, ?, ?
            )
            """,
            (
                name,
                created.session_id,
                command.command_id,
                attempt_id,
                "2099-01-01T00:00:00+00:00",
                now,
                now,
            ),
        )


def test_reconciliation_write_and_topology_guards_fail_closed(
    tmp_path: Path,
) -> None:
    multiple = StateStore(tmp_path / "multiple-leases.sqlite3")
    created = session(tmp_path / "multiple-leases")
    multiple.create_session(created)
    command = multiple.enqueue_command(
        created.session_id,
        "message",
        {"text": "ambiguous"},
        "multiple-leases",
    )
    assert multiple.claim_command(created.session_id) is not None
    attempt = _dispatched_attempt(multiple, created.session_id, command.command_id)
    multiple.mark_provider_boundary(attempt.attempt_id)
    _insert_bound_lease(multiple, created, command, attempt.attempt_id, "lease-one")
    _insert_bound_lease(multiple, created, command, attempt.attempt_id, "lease-two")
    with pytest.raises(ConflictError, match="multiple active process leases"):
        multiple.recover_interrupted_commands(
            created.session_id,
            "workspace-digest",
            "summary",
        )
    multiple.close()

    store, unused_created, unused_command, record = _ambiguous_store(
        tmp_path,
        "discovery-write",
    )
    checkpoint = _new_checkpoint(store, record.session_id)
    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER erase_reconciliation_discovery
            AFTER UPDATE OF audit_json ON reconciliations
            BEGIN DELETE FROM reconciliations; END
            """
        )
    with pytest.raises(RuntimeError, match="discovery was not recorded"):
        store.record_reconciliation_discovery(
            record.reconciliation_id,
            checkpoint,
            record.current_workspace_digest,
        )
    store.close()

    store, unused_created, unused_command, record = _ambiguous_store(
        tmp_path,
        "orphan-receipt",
    )
    store.record_mutation_receipt(
        "orphan-receipt",
        "resolve:" + record.reconciliation_id,
        "request",
        {"reconciliation": {"reconciliation_id": new_uuid()}},
        200,
    )
    with pytest.raises(RuntimeError, match="has no reconciliation row"):
        _resolve_ambiguous(store, record, "orphan-receipt")
    store.close()

    store, unused_created, unused_command, record = _ambiguous_store(
        tmp_path,
        "corrupt-status",
    )
    with store.transaction() as connection:
        connection.execute(
            "UPDATE reconciliations SET status = 'corrupt' WHERE reconciliation_id = ?",
            (record.reconciliation_id,),
        )
    with pytest.raises(ConflictError, match="already resolved differently"):
        _resolve_ambiguous(store, record, "corrupt-status")
    store.close()

    store, unused_created, unused_command, record = _ambiguous_store(
        tmp_path,
        "changed-resolution",
    )
    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE reconciliations SET status = 'resolving',
                resolution = 'accept-current' WHERE reconciliation_id = ?
            """,
            (record.reconciliation_id,),
        )
    with pytest.raises(ConflictError, match="already in progress"):
        _resolve_ambiguous(store, record, "changed-resolution")
    store.close()

    store, unused_created, unused_command, record = _ambiguous_store(
        tmp_path,
        "erased-resolution",
    )
    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER erase_reconciliation_resolution
            AFTER UPDATE OF status ON reconciliations
            WHEN NEW.status = 'resolved'
            BEGIN DELETE FROM reconciliations; END
            """
        )
    with pytest.raises(RuntimeError, match="resolution was not recorded"):
        _resolve_ambiguous(store, record, "erased-resolution")
    store.close()

    store, unused_created, unused_command, record = _ambiguous_store(
        tmp_path,
        "missing-dispatch",
    )
    identity = dict(record.audit["dispatch_identity"])
    _forget_row(
        store,
        "DELETE FROM command_dispatches WHERE attempt_id = ?",
        str(identity["attempt_id"]),
    )
    with pytest.raises(ConflictError, match="one ambiguous provider dispatch"):
        _resolve_ambiguous(store, record, "missing-dispatch")
    store.close()

    for name, field, value, message in (
        ("turn-identity", "turn_id", new_uuid(), "turn identity changed"),
        (
            "checkpoint-identity",
            "checkpoint_id",
            new_uuid(),
            "checkpoint identity changed",
        ),
    ):
        store, unused_created, unused_command, record = _ambiguous_store(
            tmp_path,
            name,
        )
        audit = copy.deepcopy(record.audit)
        audit["dispatch_identity"][field] = value
        with pytest.raises(ConflictError, match=message):
            _resolve_ambiguous(store, record, name, audit=audit)
        store.close()

    store, unused_created, unused_command, record = _ambiguous_store(
        tmp_path,
        "missing-attempt",
    )
    identity = dict(record.audit["dispatch_identity"])
    _forget_row(
        store,
        "DELETE FROM provider_attempts WHERE attempt_id = ?",
        str(identity["attempt_id"]),
    )
    with pytest.raises(ConflictError, match="provider attempt is missing"):
        _resolve_ambiguous(store, record, "missing-attempt")
    store.close()

    store, unused_created, unused_command, record = _ambiguous_store(
        tmp_path,
        "missing-turn",
    )
    identity = dict(record.audit["dispatch_identity"])
    _forget_row(
        store,
        "DELETE FROM turns WHERE turn_id = ?",
        str(identity["turn_id"]),
    )
    with pytest.raises(ConflictError, match="turn is missing"):
        _resolve_ambiguous(store, record, "missing-turn")
    store.close()

    store, unused_created, unused_command, record = _ambiguous_store(
        tmp_path,
        "lease-identity",
    )
    audit = copy.deepcopy(record.audit)
    audit["dispatch_identity"]["lease_id"] = new_uuid()
    with pytest.raises(ConflictError, match="lease identity changed"):
        _resolve_ambiguous(store, record, "lease-identity", audit=audit)
    store.close()

    store, created, command, record = _ambiguous_store(
        tmp_path,
        "resolution-leases",
    )
    attempt_id = str(record.audit["dispatch_identity"]["attempt_id"])
    _insert_bound_lease(store, created, command, attempt_id, "resolution-lease-one")
    _insert_bound_lease(store, created, command, attempt_id, "resolution-lease-two")
    with pytest.raises(ConflictError, match="multiple active process leases"):
        _resolve_ambiguous(store, record, "resolution-leases")
    store.close()


def _transition_store(
    root: Path,
    name: str,
    *,
    profile: str = "interactive",
) -> tuple[
    StateStore,
    Any,
    dict[str, Any],
    tuple[dict[str, str], dict[str, str]],
    tuple[dict[str, Any], dict[str, Any]],
]:
    workspace = root / name
    workspace.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(workspace), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "--allow-empty", "-qm", "initial"],
        check=True,
    )
    created = replace(
        session(workspace),
        external_ref={"orchestrator": "machines", "job_id": name},
    )
    store = StateStore(root / (name + ".sqlite3"))
    store.create_session(created)
    store.set_session_safety(created.session_id, profile)
    refs = (
        {"step_id": "stage-one", "agent_role": "builder"},
        {"step_id": "stage-two", "agent_role": "builder"},
    )
    payloads = (
        {"text": "stage one", "turn_ref": refs[0]},
        {"text": "stage two", "turn_ref": refs[1]},
    )
    digests = (
        command_envelope_digest("message", payloads[0], profile),
        command_envelope_digest("message", payloads[1], profile),
    )
    policy = {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": created.session_id,
        "external_ref": created.external_ref,
        "epoch_id": "epoch-one",
        "allowed_agent_roles": ["builder"],
        "allowed_step_prefixes": ["stage-"],
        "max_transitions": 2,
        "transitions": [
            {
                "sequence": 1,
                "next_turn_ref": refs[0],
                "next_command_digest": digests[0],
            },
            {
                "sequence": 2,
                "next_turn_ref": refs[1],
                "next_command_digest": digests[1],
            },
        ],
    }
    policy_digest = normalized_digest(policy)
    store.create_goal(
        create_goal(
            created.session_id,
            "Advance two exact stages.",
            constraints=(
                "dispatch-generation-transition-policy-sha256:" + policy_digest,
                "dispatch-generation-transition-epoch:epoch-one",
            ),
        )
    )
    prior = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "prior"},
        name + "-prior",
    )
    material_digest, unused_summary = storage_module.inspect_workspace(workspace)
    del unused_summary
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="codex",
        native_session_id="native",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=utc_now(),
    )
    store.add_checkpoint(checkpoint)
    store.resolve_command(
        prior.command_id,
        CommandStatus.COMPLETE,
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "workspace_material_digest": material_digest,
        },
    )
    return store, created, policy, refs, payloads


def _transition_authorization(
    store: StateStore,
    created: Any,
    policy: dict[str, Any],
    next_ref: dict[str, str],
    next_payload: dict[str, Any],
    profile: str,
    sequence: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor = store.dispatch_transition_anchor(created.session_id)
    assert anchor["eligible"] is True
    goal = store.goal_for_session(created.session_id)
    assert goal is not None
    next_digest = command_envelope_digest("message", next_payload, profile)
    policy_digest = normalized_digest(policy)
    receipt = {
        "session_id": created.session_id,
        "external_ref": created.external_ref,
        "goal_id": goal.goal_id,
        "prior_command_id": anchor["prior_command_id"],
        "prior_command_type": anchor["prior_command_type"],
        "prior_anchor_kind": anchor["prior_anchor_kind"],
        "prior_reconciliation_id": anchor["prior_reconciliation_id"],
        "prior_reconciliation_resolution": anchor[
            "prior_reconciliation_resolution"
        ],
        "prior_checkpoint_id": anchor["prior_checkpoint_id"],
        "prior_generation_digest": anchor["prior_generation_digest"],
        "prior_material_digest": anchor["prior_material_digest"],
        "next_turn_ref": next_ref,
        "transition_sequence": sequence,
        "epoch_id": "epoch-one",
        "policy_sha256": policy_digest,
        "next_command_digest": next_digest,
    }
    authorization = {
        "schema": (
            "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
        ),
        **receipt,
        "reason": "advance",
        "external_orchestrator": created.external_ref["orchestrator"],
        "external_job_id": created.external_ref["job_id"],
        "receipt": receipt,
        "receipt_sha256": normalized_digest(receipt),
        "policy": policy,
    }
    if sequence > 1:
        authorization["policy_ref"] = {
            "policy_sha256": policy_digest,
            "session_id": created.session_id,
            "goal_id": goal.goal_id,
            "epoch_id": "epoch-one",
        }
    return authorization, anchor


def _create_transition(
    store: StateStore,
    created: Any,
    authorization: dict[str, Any],
    anchor: dict[str, Any],
    next_ref: dict[str, str],
    key: str,
) -> dict[str, Any]:
    return store.create_dispatch_invalidation(
        created.session_id,
        reason="advance",
        authorization=authorization,
        request_digest=key + "-request",
        idempotency_key=key,
        prior_command_id=str(anchor["prior_command_id"]),
        next_turn_ref=next_ref,
        authorization_digest=normalized_digest(authorization),
    )


def _prepare_consumed_first_transition(
    root: Path,
    name: str,
) -> tuple[
    StateStore,
    Any,
    dict[str, Any],
    tuple[dict[str, str], dict[str, str]],
    tuple[dict[str, Any], dict[str, Any]],
    Any,
    ProviderAttempt,
]:
    store, created, policy, refs, payloads = _transition_store(root, name)
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[0],
        payloads[0],
        "interactive",
        1,
    )
    _create_transition(store, created, authorization, anchor, refs[0], name + "-first")
    command = store.enqueue_command(
        created.session_id,
        "message",
        payloads[0],
        name + "-stage-one",
    )
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "interactive",
        {"max_attempts": 1},
    )
    attempt = _dispatched_attempt(store, created.session_id, command.command_id)
    store.register_worker(created.session_id, 123, "worker")
    reserved = store.reserve_dispatch_generation_transition(
        created.session_id,
        command.command_id,
        refs[0],
        str(anchor["prior_material_digest"]),
    )
    assert reserved == "reserved"
    admission = store.reserve_route_admission(
        command.command_id,
        "codex",
        "interactive",
        effort="low",
        attempt_id=attempt.attempt_id,
        worker_incarnation="worker",
        goal_id=str(store.get_session(created.session_id).goal_id),
        max_concurrency=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    assert admission["admitted"] is True
    store.complete_dispatch(attempt.attempt_id, "complete")
    checkpoint = store.checkpoints(created.session_id)[-1]
    material_digest, unused_summary = storage_module.inspect_workspace(
        Path(created.worktree)
    )
    del unused_summary
    store.resolve_command(
        command.command_id,
        CommandStatus.COMPLETE,
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "workspace_material_digest": material_digest,
        },
    )
    return store, created, policy, refs, payloads, command, attempt


def test_dispatch_transition_validation_and_write_guards(
    tmp_path: Path,
) -> None:
    store, created, policy, refs, payloads = _transition_store(
        tmp_path,
        "schema",
    )
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[0],
        payloads[0],
        "interactive",
        1,
    )
    authorization["schema"] = "operator"
    with pytest.raises(ConflictError, match="schema is invalid"):
        _create_transition(store, created, authorization, anchor, refs[0], "schema")
    store.close()

    store, created, policy, refs, payloads = _transition_store(
        tmp_path,
        "checkpoint",
    )
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[0],
        payloads[0],
        "interactive",
        1,
    )
    _forget_row(
        store,
        "DELETE FROM checkpoints WHERE checkpoint_id = ?",
        str(anchor["prior_checkpoint_id"]),
    )
    with pytest.raises(ConflictError, match="no certified checkpoint"):
        _create_transition(
            store,
            created,
            authorization,
            anchor,
            refs[0],
            "checkpoint",
        )
    store.close()

    for index, message in enumerate(
        (
            "policy stage changed",
            "policy order changed",
            "policy stage role changed",
            "policy stage step changed",
            "policy stage digest changed",
        )
    ):
        name = "policy-stage-" + str(index)
        store, created, policy, refs, payloads = _transition_store(tmp_path, name)
        changed_policy = copy.deepcopy(policy)
        if index == 0:
            changed_policy["transitions"][1] = "bare"
        if index == 1:
            changed_policy["transitions"][1]["sequence"] = 3
        if index == 2:
            changed_policy["transitions"][1]["next_turn_ref"]["agent_role"] = (
                "reviewer"
            )
        if index == 3:
            changed_policy["transitions"][1]["next_turn_ref"]["step_id"] = "other"
        if index == 4:
            changed_policy["transitions"][1]["next_command_digest"] = "bad"
        authorization, anchor = _transition_authorization(
            store,
            created,
            changed_policy,
            refs[0],
            payloads[0],
            "interactive",
            1,
        )
        with pytest.raises(ConflictError, match=message):
            _create_transition(
                store,
                created,
                authorization,
                anchor,
                refs[0],
                name,
            )
        store.close()

    (
        store,
        created,
        policy,
        refs,
        payloads,
        unused_command,
        unused_attempt,
    ) = _prepare_consumed_first_transition(tmp_path, "shrinking-policy")
    del unused_command, unused_attempt

    class ShrinkingTransitions(list[Any]):
        def __init__(self, values: list[Any]) -> None:
            super().__init__(values)
            self.length_reads = 0

        def __len__(self) -> int:
            self.length_reads += 1
            if self.length_reads == 1:
                return 2
            return 1

    shrinking_policy = copy.deepcopy(policy)
    shrinking_policy["transitions"] = ShrinkingTransitions(
        list(shrinking_policy["transitions"])
    )
    authorization, anchor = _transition_authorization(
        store,
        created,
        shrinking_policy,
        refs[1],
        payloads[1],
        "interactive",
        2,
    )
    with pytest.raises(ConflictError, match="sequence is outside policy"):
        _create_transition(
            store,
            created,
            authorization,
            anchor,
            refs[1],
            "shrinking-policy",
        )
    store.close()

    store, created, unused_policy, unused_refs, unused_payloads = _transition_store(
        tmp_path,
        "ignored-invalidation",
    )
    operator = {
        "schema": "p13i/agent-harness/dispatch-invalidation-authorization/v1",
        "session_id": created.session_id,
    }
    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER ignore_dispatch_invalidation
            BEFORE INSERT ON dispatch_invalidations
            BEGIN SELECT RAISE(IGNORE); END
            """
        )
    with pytest.raises(RuntimeError, match="invalidation was not recorded"):
        store.create_dispatch_invalidation(
            created.session_id,
            reason="operator",
            authorization=operator,
            request_digest="ignored-request",
            idempotency_key="ignored-invalidation",
        )
    store.close()

    store, created, unused_policy, unused_refs, unused_payloads = _transition_store(
        tmp_path,
        "missing-authorization",
    )
    receipt = store.create_dispatch_invalidation(
        created.session_id,
        reason="operator",
        authorization=operator,
        request_digest="operator-request",
        idempotency_key="operator",
    )
    _forget_row(
        store,
        "DELETE FROM authorization_receipts WHERE operation_id = ?",
        str(receipt["invalidation_id"]),
    )
    with pytest.raises(RuntimeError, match="authorization receipt is missing"):
        store.dispatch_invalidation_replay(
            created.session_id,
            "operator",
            "operator-request",
        )
    store.close()


def test_dispatch_transition_reservation_and_consumption_guards(
    tmp_path: Path,
) -> None:
    store, created, policy, refs, payloads = _transition_store(
        tmp_path,
        "reservation",
    )
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[0],
        payloads[0],
        "interactive",
        1,
    )
    _create_transition(store, created, authorization, anchor, refs[0], "reservation")
    assert store.reserve_dispatch_generation_transition(
        created.session_id,
        new_uuid(),
        refs[0],
        str(anchor["prior_material_digest"]),
    ) == "command-mismatch"
    command = store.enqueue_command(
        created.session_id,
        "message",
        payloads[0],
        "reserved-command",
    )
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "interactive",
        {"max_attempts": 1},
    )
    assert store.reserve_dispatch_generation_transition(
        created.session_id,
        command.command_id,
        refs[0],
        str(anchor["prior_material_digest"]),
    ) == "reserved"
    assert store.reserve_dispatch_generation_transition(
        created.session_id,
        command.command_id,
        refs[0],
        str(anchor["prior_material_digest"]),
    ) == "reserved"
    other = store.enqueue_command(
        created.session_id,
        "message",
        payloads[0],
        "other-command",
    )
    store.create_command_envelope(
        other.command_id,
        created.session_id,
        "interactive",
        {"max_attempts": 1},
    )
    assert store.reserve_dispatch_generation_transition(
        created.session_id,
        other.command_id,
        refs[0],
        str(anchor["prior_material_digest"]),
    ) == "already-consumed"
    with store.transaction() as connection:
        store._consume_reserved_dispatch_transition(
            connection,
            created.session_id,
            command.command_id,
            Path(created.worktree),
        )
    assert store.reserve_dispatch_generation_transition(
        created.session_id,
        command.command_id,
        refs[0],
        str(anchor["prior_material_digest"]),
    ) == "consumed"
    with store.transaction() as connection:
        store._consume_reserved_dispatch_transition(
            connection,
            created.session_id,
            command.command_id,
            Path(created.worktree),
        )
    store.close()

    store, created, policy, refs, payloads = _transition_store(
        tmp_path,
        "stage-mismatch",
    )
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[0],
        payloads[0],
        "interactive",
        1,
    )
    _create_transition(store, created, authorization, anchor, refs[0], "stage-mismatch")
    assert store.reserve_dispatch_generation_transition(
        created.session_id,
        new_uuid(),
        refs[1],
        str(anchor["prior_material_digest"]),
    ) == "stage-mismatch"
    store.close()

    store, created, policy, refs, payloads = _transition_store(
        tmp_path,
        "epoch-mismatch",
    )
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[0],
        payloads[0],
        "interactive",
        1,
    )
    _create_transition(store, created, authorization, anchor, refs[0], "epoch-mismatch")
    goal = store.goal_for_session(created.session_id)
    assert goal is not None
    with store.transaction() as connection:
        connection.execute(
            "UPDATE goals SET constraints_json = '[]' WHERE goal_id = ?",
            (goal.goal_id,),
        )
    assert store.reserve_dispatch_generation_transition(
        created.session_id,
        new_uuid(),
        refs[0],
        str(anchor["prior_material_digest"]),
    ) == "epoch-mismatch"
    with pytest.raises(ConflictError, match="epoch is no longer active"):
        _create_transition(
            store,
            created,
            authorization,
            anchor,
            refs[0],
            "epoch-mismatch",
        )
    with store.transaction() as connection:
        with pytest.raises(ConflictError, match="epoch is stale"):
            store._consume_reserved_dispatch_transition(
                connection,
                created.session_id,
                new_uuid(),
                Path(created.worktree),
            )
    store.close()


def test_dispatch_transition_consumption_rejects_command_and_reservation_drift(
    tmp_path: Path,
) -> None:
    for name, expected in (
        ("missing-command", "command is missing"),
        ("changed-command", "command changed"),
        ("stale-reservation", "reservation is stale"),
    ):
        store, created, policy, refs, payloads = _transition_store(tmp_path, name)
        authorization, anchor = _transition_authorization(
            store,
            created,
            policy,
            refs[0],
            payloads[0],
            "interactive",
            1,
        )
        _create_transition(store, created, authorization, anchor, refs[0], name)
        command_id = new_uuid()
        if name != "missing-command":
            changed_payload = {"text": "changed"}
            if name == "stale-reservation":
                changed_payload = payloads[0]
            command = store.enqueue_command(
                created.session_id,
                "message",
                changed_payload,
                name + "-command",
            )
            command_id = command.command_id
            store.create_command_envelope(
                command.command_id,
                created.session_id,
                "interactive",
                {"max_attempts": 1},
            )
        with store.transaction() as connection:
            with pytest.raises(ConflictError, match=expected):
                store._consume_reserved_dispatch_transition(
                    connection,
                    created.session_id,
                    command_id,
                    Path(created.worktree),
                )
        store.close()

    store, created, policy, refs, payloads = _transition_store(
        tmp_path,
        "stale-consumption",
    )
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[0],
        payloads[0],
        "interactive",
        1,
    )
    _create_transition(store, created, authorization, anchor, refs[0], "stale-consumption")
    command = store.enqueue_command(
        created.session_id,
        "message",
        payloads[0],
        "stale-consumption-command",
    )
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "interactive",
        {"max_attempts": 1},
    )
    assert store.reserve_dispatch_generation_transition(
        created.session_id,
        command.command_id,
        refs[0],
        str(anchor["prior_material_digest"]),
    ) == "reserved"
    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER ignore_transition_consumption
            BEFORE UPDATE OF state ON dispatch_transition_ledger
            WHEN NEW.state = 'consumed'
            BEGIN SELECT RAISE(IGNORE); END
            """
        )
    with store.transaction() as connection:
        with pytest.raises(ConflictError, match="consumption is stale"):
            store._consume_reserved_dispatch_transition(
                connection,
                created.session_id,
                command.command_id,
                Path(created.worktree),
            )
    store.close()


def test_followup_transition_rejects_policy_reference_drift(
    tmp_path: Path,
) -> None:
    store, created, policy, refs, payloads, unused_command, unused_attempt = (
        _prepare_consumed_first_transition(tmp_path, "policy-reference")
    )
    del unused_command, unused_attempt
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[1],
        payloads[1],
        "interactive",
        2,
    )
    authorization["policy_ref"] = {}
    with pytest.raises(ConflictError, match="policy reference changed"):
        _create_transition(
            store,
            created,
            authorization,
            anchor,
            refs[1],
            "policy-reference",
        )
    store.close()

    store, created, policy, refs, payloads, unused_command, unused_attempt = (
        _prepare_consumed_first_transition(tmp_path, "unknown-policy")
    )
    del unused_command, unused_attempt
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[1],
        payloads[1],
        "interactive",
        2,
    )
    durable_connection = store._connection

    class MissingPolicyConnection:
        def execute(
            self,
            statement: str,
            parameters: tuple[object, ...] = (),
        ) -> object:
            if "SELECT payload_json FROM dispatch_transition_policies" in statement:
                return SimpleNamespace(fetchone=lambda: None)
            return durable_connection.execute(statement, parameters)

        def __getattr__(self, name: str) -> object:
            return getattr(durable_connection, name)

    store._connection = MissingPolicyConnection()  # type: ignore[assignment]
    try:
        with pytest.raises(ConflictError, match="reference is unknown"):
            _create_transition(
                store,
                created,
                authorization,
                anchor,
                refs[1],
                "unknown-policy",
            )
    finally:
        store._connection = durable_connection
    store.close()

    store, created, policy, refs, payloads, unused_command, unused_attempt = (
        _prepare_consumed_first_transition(tmp_path, "changed-policy")
    )
    del unused_command, unused_attempt
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[1],
        payloads[1],
        "interactive",
        2,
    )
    with store.transaction() as connection:
        connection.execute(
            "UPDATE dispatch_transition_policies SET payload_json = '{}'",
        )
    with pytest.raises(ConflictError, match="retained policy changed"):
        _create_transition(
            store,
            created,
            authorization,
            anchor,
            refs[1],
            "changed-policy",
        )
    store.close()


def _control_transition_store(
    root: Path,
    name: str,
    *,
    consumed_terminal: bool = True,
    control_result_exact: bool = True,
    dispatch_state: str = "interrupted",
    interrupt_event: bool = True,
    extra_message: bool = False,
) -> tuple[
    StateStore,
    Any,
    dict[str, Any],
    tuple[dict[str, str], dict[str, str]],
    tuple[dict[str, Any], dict[str, Any]],
    Any,
    Any,
    dict[str, Any],
    dict[str, Any],
]:
    store, created, policy, refs, payloads, consumed, attempt = (
        _prepare_consumed_first_transition(root, name)
    )
    checkpoint = store.checkpoints(created.session_id)[-1]
    material_digest, unused_summary = storage_module.inspect_workspace(
        Path(created.worktree)
    )
    del unused_summary
    if consumed_terminal:
        store.resolve_command(
            consumed.command_id,
            CommandStatus.FAILED,
            {"code": "E_INTERRUPTED"},
        )
    with store.transaction() as connection:
        connection.execute(
            "UPDATE command_dispatches SET state = ? WHERE attempt_id = ?",
            (dispatch_state, attempt.attempt_id),
        )
    if extra_message:
        extra = store.enqueue_command(
            created.session_id,
            "message",
            {"text": "unexpected lineage"},
            name + "-extra",
        )
        store.resolve_command(extra.command_id, CommandStatus.COMPLETE, {})
    control = store.enqueue_command(
        created.session_id,
        "interrupt",
        {},
        name + "-control",
    )
    control_result = {
        "target_command_id": consumed.command_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "workspace_material_digest": material_digest,
    }
    if not control_result_exact:
        control_result["checkpoint_id"] = new_uuid()
    store.resolve_command(control.command_id, CommandStatus.COMPLETE, control_result)
    if interrupt_event:
        store.append_event(
            created.session_id,
            "turn.interrupted",
            status="complete",
            metadata={
                "control_command_id": control.command_id,
                "target_command_id": consumed.command_id,
                "checkpoint_id": checkpoint.checkpoint_id,
                "workspace_material_digest": material_digest,
            },
        )
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[1],
        payloads[1],
        "interactive",
        2,
    )
    return (
        store,
        created,
        policy,
        refs,
        payloads,
        consumed,
        control,
        authorization,
        anchor,
    )


def test_followup_transition_rejects_changed_provider_and_control_lineage(
    tmp_path: Path,
) -> None:
    store, created, policy, refs, payloads, consumed, unused_attempt = (
        _prepare_consumed_first_transition(tmp_path, "provider-lineage")
    )
    del unused_attempt
    later = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "later provider result"},
        "provider-lineage-later",
    )
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="codex",
        native_session_id="native-later",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context-later",
        created_at=utc_now(),
    )
    store.add_checkpoint(checkpoint)
    material_digest, unused_summary = storage_module.inspect_workspace(
        Path(created.worktree)
    )
    del unused_summary
    store.resolve_command(
        later.command_id,
        CommandStatus.COMPLETE,
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "workspace_material_digest": material_digest,
        },
    )
    authorization, anchor = _transition_authorization(
        store,
        created,
        policy,
        refs[1],
        payloads[1],
        "interactive",
        2,
    )
    assert anchor["prior_command_id"] == later.command_id
    assert anchor["prior_command_id"] != consumed.command_id
    with pytest.raises(ConflictError, match="predecessor was not consumed"):
        _create_transition(
            store,
            created,
            authorization,
            anchor,
            refs[1],
            "provider-lineage",
        )
    store.close()

    cases = (
        (
            "missing-control-lineage",
            {"consumed_terminal": False},
            "control lineage is missing",
        ),
        (
            "changed-control-result",
            {"control_result_exact": False},
            "control lineage changed",
        ),
        (
            "changed-boundary",
            {"dispatch_state": "complete"},
            "interrupted boundary changed",
        ),
        (
            "missing-interrupt",
            {"interrupt_event": False},
            "interrupt receipt changed",
        ),
        (
            "nonexact-control-lineage",
            {"extra_message": True},
            "control lineage is not exact",
        ),
    )
    for name, values, message in cases:
        (
            store,
            created,
            unused_policy,
            refs,
            unused_payloads,
            unused_consumed,
            unused_control,
            authorization,
            anchor,
        ) = _control_transition_store(tmp_path, name, **values)
        del unused_policy, unused_payloads, unused_consumed, unused_control
        with pytest.raises(ConflictError, match=message):
            _create_transition(
                store,
                created,
                authorization,
                anchor,
                refs[1],
                name,
            )
        store.close()


def test_followup_transition_rejects_control_lineage_missing_from_enumeration(
    tmp_path: Path,
) -> None:
    (
        store,
        created,
        unused_policy,
        refs,
        unused_payloads,
        unused_consumed,
        control,
        authorization,
        anchor,
    ) = _control_transition_store(tmp_path, "lineage-enumeration")
    del unused_policy, unused_payloads, unused_consumed
    durable_connection = store._connection
    control_row = durable_connection.execute(
        "SELECT command_id, command_type, status FROM commands WHERE command_id = ?",
        (control.command_id,),
    ).fetchone()
    assert control_row is not None

    class MissingLineageConnection:
        def execute(
            self,
            statement: str,
            parameters: tuple[object, ...] = (),
        ) -> object:
            if "SELECT command_id, command_type, status" in statement:
                return SimpleNamespace(fetchall=lambda: [control_row])
            return durable_connection.execute(statement, parameters)

        def __getattr__(self, name: str) -> object:
            return getattr(durable_connection, name)

    store._connection = MissingLineageConnection()  # type: ignore[assignment]
    try:
        with pytest.raises(ConflictError, match="lineage is out of order"):
            _create_transition(
                store,
                created,
                authorization,
                anchor,
                refs[1],
                "lineage-enumeration",
            )
    finally:
        store._connection = durable_connection
    store.close()


def _failed_message(
    store: StateStore,
    session_id: str,
    payload: dict[str, Any],
    idempotency_key: str,
    result: dict[str, Any],
):
    receipt, was_created = store.ensure_message_command(
        session_id,
        payload,
        idempotency_key,
    )
    assert was_created
    store.create_command_envelope(
        receipt.command_id,
        session_id,
        "unattended",
        {"max_attempts": 3},
    )
    assert store.claim_command(session_id) is not None
    store.update_command_envelope(
        receipt.command_id,
        state="paused",
        guard_reason="provider-capacity",
    )
    store.resolve_command(
        receipt.command_id,
        CommandStatus.FAILED,
        result,
    )
    return receipt


def _instruction_events(store: StateStore, session_id: str) -> list[Any]:
    return [
        event
        for event in store.events(session_id, limit=5000)
        if event.event_type == "user.message"
    ]


def test_retryable_failed_command_requeues_on_identical_resubmission(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    payload = {"text": "Review the cs-builder change."}
    first = _failed_message(
        store,
        created.session_id,
        payload,
        "builder-resubmission",
        {
            "code": "E_PROVIDER_UNAVAILABLE",
            "message": "grok is unavailable",
            "retryable": True,
        },
    )

    requeued, created_again = store.ensure_message_command(
        created.session_id,
        payload,
        "builder-resubmission",
    )

    assert not created_again
    assert requeued.command_id == first.command_id
    assert requeued.status == CommandStatus.QUEUED
    assert requeued.result == {}
    envelope = store.command_envelope(first.command_id)
    assert envelope["state"] == "reserved"
    assert envelope["guard_reason"] == ""
    instructions = _instruction_events(store, created.session_id)
    assert [event.metadata["command_id"] for event in instructions] == [
        first.command_id
    ]
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    assert claimed.command_id == first.command_id
    assert store.claim_command(created.session_id) is None
    store.close()


def test_terminal_commands_stay_terminal_on_identical_resubmission(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    defect_payload = {"text": "Run the defective instruction."}
    defect = _failed_message(
        store,
        created.session_id,
        defect_payload,
        "non-retryable-failure",
        {
            "code": "E_DEFECT",
            "message": "a genuine non-retryable defect",
            "retryable": False,
        },
    )
    complete_payload = {"text": "Run the accepted instruction."}
    complete, was_created = store.ensure_message_command(
        created.session_id,
        complete_payload,
        "complete-command",
    )
    assert was_created
    store.resolve_command(
        complete.command_id,
        CommandStatus.COMPLETE,
        {"status": "complete", "retryable": True},
    )

    replayed_defect, defect_created = store.ensure_message_command(
        created.session_id,
        defect_payload,
        "non-retryable-failure",
    )
    replayed_complete, complete_created = store.ensure_message_command(
        created.session_id,
        complete_payload,
        "complete-command",
    )

    assert not defect_created
    assert replayed_defect.command_id == defect.command_id
    assert replayed_defect.status == CommandStatus.FAILED
    assert replayed_defect.result["code"] == "E_DEFECT"
    assert not complete_created
    assert replayed_complete.command_id == complete.command_id
    assert replayed_complete.status == CommandStatus.COMPLETE
    assert replayed_complete.result["status"] == "complete"
    assert store.command_envelope(defect.command_id)["state"] == "paused"
    assert store.claim_command(created.session_id) is None
    store.close()


def test_retryable_failure_past_the_provider_boundary_stays_terminal(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    payload = {"text": "Dispatch across the provider boundary."}
    first = _failed_message(
        store,
        created.session_id,
        payload,
        "crossed-boundary",
        {
            "code": "E_PROVIDER_UNAVAILABLE",
            "message": "grok is unavailable",
            "retryable": True,
        },
    )
    attempt = _dispatched_attempt(store, created.session_id, first.command_id)
    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE command_dispatches SET crossed_boundary = 1
            WHERE attempt_id = ?
            """,
            (attempt.attempt_id,),
        )

    replayed, created_again = store.ensure_message_command(
        created.session_id,
        payload,
        "crossed-boundary",
    )

    assert not created_again
    assert replayed.command_id == first.command_id
    assert replayed.status == CommandStatus.FAILED
    assert replayed.result["code"] == "E_PROVIDER_UNAVAILABLE"
    assert store.claim_command(created.session_id) is None
    store.close()


def test_concurrent_resubmissions_requeue_a_failed_command_once(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    payload = {"text": "Retry exactly once."}
    first = _failed_message(
        store,
        created.session_id,
        payload,
        "concurrent-resubmission",
        {
            "code": "E_PROVIDER_UNAVAILABLE",
            "message": "grok is unavailable",
            "retryable": True,
        },
    )
    barrier = threading.Barrier(2)
    receipts: list[Any] = []

    def resubmit() -> None:
        barrier.wait(timeout=5)
        receipts.append(
            store.ensure_message_command(
                created.session_id,
                payload,
                "concurrent-resubmission",
            )
        )

    threads = [threading.Thread(target=resubmit) for unused in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert len(receipts) == 2
    for receipt, created_again in receipts:
        assert not created_again
        assert receipt.command_id == first.command_id
        assert receipt.status == CommandStatus.QUEUED
    assert len(_instruction_events(store, created.session_id)) == 1
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    assert claimed.command_id == first.command_id
    assert store.claim_command(created.session_id) is None
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
