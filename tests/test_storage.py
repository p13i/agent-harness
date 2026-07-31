from dataclasses import replace
import copy
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest

import agent_harness.migration as migration_module
import agent_harness.records as records_module
import agent_harness.storage as storage_module
import agent_harness.sync as sync_module
from agent_harness.config import paths
from agent_harness.config import prepare_paths
from agent_harness.errors import ConflictError
from agent_harness.errors import NotFoundError
from agent_harness.goals import create_goal
from agent_harness.goals import make_evidence
from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Checkpoint
from agent_harness.models import CommandStatus
from agent_harness.models import ProviderAttempt
from agent_harness.orchestration import creation_digest
from agent_harness.orchestration import normalize_creation_input
from agent_harness.orchestration import normalize_external_ref
from agent_harness.orchestration import normalize_turn_ref
from agent_harness.migration import migrate_state
from agent_harness.records import load_portable_records
from agent_harness.storage import StateStore
from agent_harness.sync import publish_all
from agent_harness.sync import publish_session
from agent_harness.sync import read_sync_status
from agent_harness.sync import sync_repository
from agent_harness.workspace import create_worktree
from test_support import session


def test_store_round_trip_and_idempotent_command(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    event = store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="continue",
    )
    assert event.sequence == 1
    first = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "continue"},
        "same-key",
    )
    second, was_created = store.ensure_command(
        created.session_id,
        "message",
        {"text": "continue"},
        "same-key",
    )
    assert first == second
    assert not was_created
    with pytest.raises(ConflictError):
        store.enqueue_command(
            created.session_id,
            "message",
            {"text": "different"},
            "same-key",
        )
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    assert claimed.status == CommandStatus.DISPATCHING
    assert store.recover_dispatching(created.session_id) == 0
    recovered = store.get_command(first.command_id)
    assert recovered.status == CommandStatus.QUEUED
    store.close()


def test_store_missing_state_paths_fail_closed(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        assert store.session_safety("missing") == {
            "session_id": "missing",
            "profile": "",
            "xhigh_authorizations": 0,
            "extensions": {},
            "created_at": "",
            "updated_at": "",
        }
        with pytest.raises(ConflictError, match="not claimed"):
            store.extend_session_safety("missing", {"value": 1})
        with pytest.raises(ConflictError, match="not claimed"):
            store.consume_session_extensions("missing")
        with pytest.raises(NotFoundError):
            store.command_envelope("missing")
        with pytest.raises(NotFoundError):
            store.update_command_envelope(
                "missing",
                state="running",
            )
        with pytest.raises(NotFoundError):
            store.checkpoint("missing")
        with pytest.raises(NotFoundError):
            store.complete_dispatch("missing", "complete")

        assert storage_module._object_or_empty("invalid") == {}
        assert storage_module._objects("invalid") == []
        with pytest.raises(ValueError, match="object"):
            storage_module._require_object("invalid", "value")
        with pytest.raises(ValueError, match="stored JSON"):
            storage_module._load_object("[]")
    finally:
        store.close()


def test_schema_v3_migrates_external_and_turn_columns(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_meta(version INTEGER NOT NULL);
        INSERT INTO schema_meta VALUES (2);
        CREATE TABLE sessions (
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
        CREATE TABLE turns (
            turn_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            status TEXT NOT NULL,
            replay_of TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL
        );
        CREATE TABLE commands (
            idempotency_key TEXT PRIMARY KEY,
            command_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            command_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    connection.close()

    store = StateStore(database)

    assert store._connection.execute(
        "SELECT version FROM schema_meta"
    ).fetchone()["version"] == 3
    session_columns = {
        row["name"]
        for row in store._connection.execute(
            "PRAGMA table_info(sessions)"
        )
    }
    command_columns = {
        row["name"]
        for row in store._connection.execute(
            "PRAGMA table_info(commands)"
        )
    }
    assert {
        "external_orchestrator",
        "external_job_id",
        "creation_digest",
    } <= session_columns
    assert {"turn_step_id", "turn_agent_role"} <= command_columns
    store.close()


def test_external_reference_creation_lookup_fork_and_conflicts(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    external_ref = {
        "orchestrator": "p13i/machines",
        "job_id": "job-42",
    }
    created = replace(session(tmp_path), external_ref=external_ref)
    creation_input = {
        "name": created.name,
        "workspace": created.workspace,
        "permission_mode": created.permission_mode,
        "execution_profile": "unattended",
        "external_ref": external_ref,
    }

    first, was_created = store.ensure_session(
        created,
        creation_input,
        idempotency_key="create-job-42",
    )
    replay, replay_created = store.ensure_session(
        replace(created, session_id=new_uuid()),
        dict(reversed(list(creation_input.items()))),
        idempotency_key="create-job-42",
    )
    by_reference = store.get_session_by_external_ref(
        "p13i/machines",
        "job-42",
    )

    assert was_created
    assert not replay_created
    assert replay.session_id == first.session_id
    assert store.existing_ensured_session(
        creation_input,
        idempotency_key="create-job-42",
    ) == first
    assert store.existing_ensured_session(
        creation_input,
        external_ref=external_ref,
    ) == first
    assert store.existing_ensured_session(
        {"name": "new", "workspace": str(tmp_path)}
    ) is None
    assert by_reference == first
    assert store.find_session_by_external_ref(
        "p13i/machines",
        "job-42",
    ) == first
    assert store.list_sessions(external_ref=external_ref) == [first]
    with pytest.raises(ValueError):
        store.list_sessions(external_ref={})
    with pytest.raises(ConflictError):
        store.ensure_session(
            replace(created, session_id=new_uuid()),
            {**creation_input, "name": "different"},
            idempotency_key="create-job-42",
        )
    with pytest.raises(ConflictError):
        store.existing_ensured_session(
            {**creation_input, "name": "different"},
            idempotency_key="create-job-42",
        )
    with pytest.raises(ConflictError):
        store.ensure_session(
            replace(created, session_id=new_uuid()),
            {**creation_input, "name": "different"},
        )

    fork = store.create_fork(
        first.session_id,
        replace(
            session(tmp_path),
            session_id=new_uuid(),
            external_ref=external_ref,
        ),
    )
    assert fork.external_ref == {}
    with pytest.raises(ConflictError):
        store.create_fork(
            first.session_id,
            replace(session(tmp_path), session_id=new_uuid()),
            external_ref=external_ref,
        )
    with pytest.raises(NotFoundError):
        store.create_fork(
            "missing",
            replace(session(tmp_path), session_id=new_uuid()),
        )
    assert store.get_session_by_external_ref("other", "job") is None
    store.close()


def test_orchestration_normalization_and_turn_reference(
    tmp_path: Path,
) -> None:
    assert normalize_external_ref(None) == {}
    assert normalize_external_ref({}) == {}
    assert normalize_turn_ref(None) == {}
    assert normalize_turn_ref({}) == {}
    with pytest.raises(ValueError):
        normalize_external_ref("not-an-object")
    with pytest.raises(ValueError):
        normalize_external_ref({"orchestrator": "only"})
    with pytest.raises(ValueError):
        normalize_external_ref(
            {"orchestrator": "has space", "job_id": "job"}
        )
    with pytest.raises(ValueError):
        normalize_external_ref(
            {"orchestrator": "x" * 129, "job_id": "job"}
        )
    with pytest.raises(ValueError):
        normalize_turn_ref({"step_id": "only"})
    with pytest.raises(ValueError):
        normalize_turn_ref("not-an-object")
    with pytest.raises(ValueError):
        normalize_turn_ref({"step_id": 1, "agent_role": "reviewer"})
    assert normalize_turn_ref(
        {"step_id": "step", "agent_role": "x" * 128}
    )["agent_role"] == "x" * 128
    with pytest.raises(ValueError):
        normalize_turn_ref(
            {"step_id": "step", "agent_role": "x" * 129}
        )
    with pytest.raises(ValueError):
        normalize_creation_input({"routing": "automatic"})
    with pytest.raises(ValueError):
        normalize_creation_input({"goal": "finish"})
    with pytest.raises(ValueError):
        creation_digest({"goal": {"unsupported": object()}})
    with pytest.raises(ValueError):
        normalize_external_ref(
            {"orchestrator": "control\0", "job_id": "job"}
        )
    normalized = normalize_creation_input(
        {
            "routing": {"providers": ["codex", "claude"]},
            "goal": {"predicates": ({"type": "test"},)},
        }
    )
    assert normalized["routing"]["providers"] == ["codex", "claude"]
    assert normalized["goal"]["predicates"] == [{"type": "test"}]

    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    turn_ref = {"step_id": "step-1", "agent_role": "implementer"}
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "continue", "turn_ref": turn_ref},
        "managed-turn",
    )
    assert command.turn_ref == turn_ref
    assert store.claim_command(created.session_id) is not None
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="codex",
        native_session_id="",
        model="account-default",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    store.create_attempt(attempt)
    turn_id = store.start_turn(
        created.session_id,
        attempt.attempt_id,
        turn_ref=turn_ref,
    )
    assert store.turn_ref(turn_id) == turn_ref
    empty_turn = store.start_turn(
        created.session_id,
        attempt.attempt_id,
    )
    assert store.turn_ref(empty_turn) == {}
    with pytest.raises(NotFoundError):
        store.turn_ref("missing")
    store.close()


def test_store_backup_is_queryable(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    backup = tmp_path / "backup.sqlite3"
    store.backup(backup)
    copied = StateStore(backup)
    assert copied.get_session(created.session_id).name == "test"
    copied.close()
    store.close()


def test_safety_envelope_guard_and_lease_are_durable(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    safety = store.set_session_safety(created.session_id, "unattended")
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "continue"},
        "safety-key",
    )
    envelope = store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        {"max_seconds": 900},
    )

    assert safety["profile"] == "unattended"
    assert envelope["state"] == "reserved"
    updated = store.update_command_envelope(
        command.command_id,
        provider="claude",
        state="running",
        consumption={"total_tokens": 100},
    )
    assert updated["provider"] == "claude"
    assert store.active_unattended_provider_count("claude") == 1
    incident_id = store.add_guard_incident(
        created.session_id,
        command.command_id,
        "attempt-1",
        "repeated-tool",
        "downgrade",
        {"consumption": {"tool_calls": 3}},
    )
    assert store.guard_incidents(created.session_id)[0][
        "incident_id"
    ] == incident_id

    lease = store.create_process_lease(
        created.session_id,
        "claude",
        "unattended",
        "2099-01-01T00:00:00+00:00",
    )
    attached = store.update_process_lease(
        lease["lease_id"],
        pid=123,
        pid_start="456",
        state="active",
    )
    assert attached["pid"] == 123
    assert store.active_process_leases()[0]["lease_id"] == lease["lease_id"]
    assert (
        store.mutation_receipt(
            "new-key",
            "lease-create",
            "request-digest",
        )
        is None
    )
    receipt = store.record_mutation_receipt(
        "new-key",
        "lease-create",
        "request-digest",
        {"lease": lease},
        201,
    )
    assert receipt["response"]["lease"]["lease_id"] == lease["lease_id"]
    replay = store.mutation_receipt(
        "new-key",
        "lease-create",
        "request-digest",
    )
    assert replay == receipt
    store.close()


def test_dispatch_recovery_requeues_before_boundary_and_barriers_after(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    safe = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "safe retry"},
        "safe",
    )
    assert store.claim_command(created.session_id) is not None
    store.recover_interrupted_commands(
        created.session_id,
        "digest-before",
        "summary-before",
    )
    assert store.get_command(safe.command_id).status == CommandStatus.QUEUED

    claimed_safe = store.claim_command(created.session_id)
    assert claimed_safe is not None
    store.resolve_command(
        safe.command_id,
        CommandStatus.COMPLETE,
        {},
    )
    ambiguous = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "ambiguous"},
        "ambiguous",
    )
    assert store.claim_command(created.session_id) is not None
    now = utc_now()
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="claude",
        native_session_id="native",
        model="opus",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=now,
        ended_at="",
    )
    store.create_attempt(attempt)
    turn_id = store.start_turn(created.session_id, attempt.attempt_id)
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="claude",
        native_session_id="native",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=now,
    )
    store.add_checkpoint(checkpoint)
    store.record_dispatch_checkpoint(
        ambiguous.command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    with pytest.raises(ConflictError):
        store.mark_provider_boundary("missing")
    store.mark_provider_boundary(attempt.attempt_id)
    with pytest.raises(ConflictError):
        store.mark_provider_boundary(attempt.attempt_id)
    with pytest.raises(ValueError):
        store.complete_dispatch(attempt.attempt_id, "unknown")

    recovery = store.recover_interrupted_commands(
        created.session_id,
        "digest-current",
        "summary-current",
    )

    assert recovery.requeued_command_ids == ()
    assert len(recovery.reconciliations) == 1
    record = recovery.reconciliations[0]
    assert record.command_id == ambiguous.command_id
    assert record.pre_dispatch_checkpoint_id == checkpoint.checkpoint_id
    assert record.provider_attempts[0]["provider"] == "claude"
    assert record.safety_consumption == {}
    failed = store.get_command(ambiguous.command_id)
    assert failed.status == CommandStatus.FAILED
    assert failed.result["code"] == "E_NEEDS_RECONCILIATION"
    assert store.pending_reconciliations(created.session_id) == [record]
    assert store.reconciliation(record.reconciliation_id) == record

    queued = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "must wait"},
        "barrier",
    )
    assert store.claim_command(created.session_id) is None
    control = store.enqueue_command(
        created.session_id,
        "stop",
        {},
        "control",
    )
    assert store.claim_command(
        created.session_id,
        frozenset({"stop"}),
    ).command_id == control.command_id
    assert store.get_command(queued.command_id).status == CommandStatus.QUEUED
    with pytest.raises(ConflictError):
        store.resolve_reconciliation_record(
            record.reconciliation_id,
            "stop",
            "stale",
            {},
        )
    resolving = store.begin_reconciliation_resolution(
        record.reconciliation_id,
        "stop",
        "digest-current",
    )
    assert resolving.status == "resolving"
    assert store.begin_reconciliation_resolution(
        record.reconciliation_id,
        "stop",
        "digest-current",
    ) == resolving
    with pytest.raises(ConflictError):
        store.begin_reconciliation_resolution(
            record.reconciliation_id,
            "accept-current",
            "digest-current",
        )
    resolved = store.resolve_reconciliation_record(
        record.reconciliation_id,
        "stop",
        "digest-current",
        {"actor": "test"},
    )
    assert resolved.status == "resolved"
    assert store.resolve_reconciliation_record(
        record.reconciliation_id,
        "stop",
        "digest-current",
        {"actor": "ignored-replay"},
    ) == resolved
    with pytest.raises(ConflictError):
        store.resolve_reconciliation_record(
            record.reconciliation_id,
            "accept-current",
            "digest-current",
            {},
        )
    with pytest.raises(ConflictError):
        store.begin_reconciliation_resolution(
            record.reconciliation_id,
            "stop",
            "digest-current",
        )
    with pytest.raises(NotFoundError):
        store.reconciliation("missing")
    with pytest.raises(NotFoundError):
        store.resolve_reconciliation_record(
            "missing",
            "stop",
            "digest-current",
            {},
        )
    store.close()


def test_session_export_import_preserves_resumable_state(
    tmp_path: Path,
) -> None:
    source = StateStore(tmp_path / "source.sqlite3")
    created = session(tmp_path)
    source.create_session(created)
    now = utc_now()
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="codex",
        native_session_id="native-session",
        model="account-default",
        effort="high",
        auth_mode="subscription",
        status="complete",
        started_at=now,
        ended_at=now,
    )
    source.create_attempt(attempt)
    source.append_event(
        created.session_id,
        "agent.message",
        role="assistant",
        text="durable response",
        status="complete",
    )
    goal = create_goal(
        created.session_id,
        "preserve the session",
        predicates=({"type": "test", "outcome": "passed"},),
    )
    source.create_goal(goal)
    source.add_evidence(
        make_evidence(goal.goal_id, "test", "unit", "passed")
    )
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=1,
        provider="codex",
        native_session_id="native-session",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=now,
    )
    source.add_checkpoint(checkpoint)
    source.set_session_safety(created.session_id, "unattended")
    source.extend_session_safety(
        created.session_id,
        {
            "max_seconds": 120,
            "allow_xhigh_once": True,
        },
    )

    payload = source.export_session(created.session_id)
    destination = StateStore(tmp_path / "destination.sqlite3")
    imported = destination.import_session(
        payload,
        worktree=str(tmp_path / "restored"),
        owner_host="restored-host",
        owner_epoch=2,
    )

    assert imported.session_id == created.session_id
    assert imported.lifecycle == "paused"
    assert imported.owner_host == "restored-host"
    assert destination.attempts(created.session_id) == [attempt]
    assert destination.all_events(created.session_id)[0].text == (
        "durable response"
    )
    assert destination.goal_for_session(created.session_id) is not None
    assert destination.evidence(goal.goal_id)[0].subject == "unit"
    assert destination.checkpoints(created.session_id) == [checkpoint]
    safety = destination.session_safety(created.session_id)
    assert safety["profile"] == "unattended"
    assert safety["xhigh_authorizations"] == 1
    assert safety["extensions"]["max_seconds"] == 120
    with pytest.raises(ConflictError):
        destination.import_session(
            payload,
            worktree=str(tmp_path / "duplicate"),
            owner_host="restored-host",
            owner_epoch=3,
        )
    source.close()
    destination.close()


def test_registry_worker_ui_and_failure_paths(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)

    assert store.get_ui_state("theme") == {}
    store.set_ui_state("theme", {"mode": "dark"})
    assert store.get_ui_state("theme") == {"mode": "dark"}
    store.upsert_registry_entry(
        created.session_id,
        "host-a",
        "https://host-a.example",
        2,
        "running",
        4,
    )
    assert store.registry_entry(created.session_id)["owner_epoch"] == 2
    assert len(store.registry_entries()) == 1
    with pytest.raises(ConflictError):
        store.upsert_registry_entry(
            created.session_id,
            "host-b",
            "https://host-b.example",
            1,
            "paused",
            5,
        )
    store.record_routing(
        created.session_id,
        "turn",
        "codex",
        "account-default",
        "high",
        {"reason": "headroom"},
    )
    store.register_worker(created.session_id, 123, "incarnation")
    assert store.integrity_check() == "ok"
    registrations = store.worker_registrations()
    assert len(registrations) == 1
    assert registrations[0]["session_id"] == created.session_id
    assert registrations[0]["pid"] == 123
    assert registrations[0]["incarnation"] == "incarnation"
    assert registrations[0]["heartbeat_at"]
    assert store.heartbeat_worker(created.session_id, "incarnation")
    assert not store.heartbeat_worker(created.session_id, "other")
    store.remove_worker(created.session_id, "incarnation")
    assert store.worker_registrations() == []
    approval_id = store.create_approval(
        created.session_id,
        "",
        "reconciliation-1",
        "reconciliation.restore",
        "Restore?",
        [{"id": "approve", "label": "Restore"}],
    )
    assert store.approval(approval_id)["status"] == "pending"
    assert store.resolve_approval(
        approval_id,
        {"decision": "approve"},
    )
    assert store.approval(approval_id)["decision"] == {
        "decision": "approve"
    }
    with pytest.raises(NotFoundError):
        store.approval("missing")

    with pytest.raises(NotFoundError):
        store.get_session("missing")
    with pytest.raises(ValueError):
        store.update_session(created.session_id, unsupported=True)
    assert store.update_session(created.session_id) == created
    archived = store.update_session(created.session_id, archived=True)
    assert archived.archived
    assert store.list_sessions() == []
    assert store.list_sessions(include_archived=True) == [archived]
    with pytest.raises(NotFoundError):
        store.command_payload("missing")
    with pytest.raises(NotFoundError):
        store.get_command("missing")
    with pytest.raises(NotFoundError):
        store.resolve_command("missing", CommandStatus.FAILED, {})
    with pytest.raises(NotFoundError):
        store.get_goal("missing")
    with pytest.raises(NotFoundError):
        store.update_goal_status("missing", "complete")
    with pytest.raises(NotFoundError):
        store.process_lease("missing")
    with pytest.raises(NotFoundError):
        store.update_process_lease("missing", state="expired")
    with pytest.raises(NotFoundError):
        store.registry_entry("missing")

    with pytest.raises(RuntimeError):
        with store.transaction():
            raise RuntimeError("force rollback")
    store.close()


def test_portable_records_round_trip_complete_state(
    tmp_path: Path,
) -> None:
    harness_paths = paths(tmp_path / "chats")
    prepare_paths(harness_paths)
    source = StateStore(harness_paths.database)
    created = session(tmp_path)
    source.create_session(created)
    source.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="preserve this message",
    )
    source.set_ui_state(
        "session:" + created.session_id,
        {"theme": "system"},
    )
    source.set_ui_state("workspace", {"sidebar_width": "40"})

    publish = publish_all(harness_paths, source)

    assert publish["state"] == "not-configured"
    records, global_record = load_portable_records(harness_paths)
    restored = StateStore(tmp_path / "restored.sqlite3")
    restored.import_portable(records, global_record)

    assert restored.portable_session(created.session_id) == (
        source.portable_session(created.session_id)
    )
    assert restored.portable_global() == source.portable_global()
    restored.close()
    source.close()


def test_portable_external_reference_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    external_ref = {
        "orchestrator": "p13i/autoplan",
        "job_id": "plan-7",
    }
    source = StateStore(tmp_path / "source.sqlite3")
    created = replace(session(tmp_path), external_ref=external_ref)
    source.ensure_session(
        created,
        {
            "name": created.name,
            "workspace": created.workspace,
            "external_ref": external_ref,
        },
    )
    record = source.portable_session(created.session_id)
    global_record = source.portable_global()

    destination = StateStore(tmp_path / "destination.sqlite3")
    existing = replace(
        session(tmp_path),
        session_id=new_uuid(),
        external_ref=external_ref,
    )
    destination.create_session(existing)
    with pytest.raises(ConflictError):
        destination.merge_portable([record], global_record)
    destination.close()

    duplicate = copy.deepcopy(record)
    duplicate_id = new_uuid()
    duplicate["session_id"] = duplicate_id
    duplicate["tables"]["sessions"][0]["session_id"] = duplicate_id
    empty = StateStore(tmp_path / "empty.sqlite3")
    with pytest.raises(ConflictError):
        empty.import_portable([record, duplicate], global_record)
    empty.close()
    source.close()


def test_portable_records_reject_missing_blobs_and_bad_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    source = StateStore(harness_paths.database)
    created = session(tmp_path)
    source.create_session(created)
    missing = "0" * 64
    source.add_checkpoint(
        Checkpoint(
            checkpoint_id=new_uuid(),
            session_id=created.session_id,
            sequence=0,
            provider="codex",
            native_session_id="native",
            base_commit="base",
            patch_digest=missing,
            untracked_digest=missing,
            context_digest=missing,
            created_at=utc_now(),
        )
    )

    pending = publish_session(
        harness_paths,
        source,
        created.session_id,
    )

    assert pending["state"] == "pending"
    assert pending["detail"] == "record-materialization"
    source.close()

    clean_paths = paths(tmp_path / "clean")
    prepare_paths(clean_paths)
    clean = StateStore(clean_paths.database)
    clean_session = session(tmp_path)
    clean.create_session(clean_session)
    clean.record_context_delivery(
        clean_session.session_id,
        "codex",
        "1" * 64,
        "",
    )
    published = publish_session(
        clean_paths,
        clean,
        clean_session.session_id,
    )
    assert published["state"] == "not-configured"

    def fail_sync(unused) -> dict[str, object]:
        del unused
        raise RuntimeError("Git failed")

    monkeypatch.setattr(
        sync_module,
        "sync_repository",
        fail_sync,
    )
    pending_sync = publish_session(
        clean_paths,
        clean,
        clean_session.session_id,
    )
    assert pending_sync["detail"] == "repository-synchronization"
    record_path = (
        clean_paths.sessions
        / clean_session.session_id
        / "record.gpt.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["schema"] = "unsupported"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported portable"):
        load_portable_records(clean_paths)
    record_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        records_module._read_json(record_path)
    clean.close()


def test_sync_status_and_unreachable_remote_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "chats")
    prepare_paths(harness_paths)
    harness_paths.sync_status.write_text("{", encoding="utf-8")
    assert read_sync_status(harness_paths)["state"] == "invalid"
    harness_paths.sync_status.write_text("[]", encoding="utf-8")
    assert read_sync_status(harness_paths)["state"] == "invalid"
    with pytest.raises(ValueError, match="positive"):
        sync_repository(harness_paths, attempts=0)

    _workspace_repository(harness_paths.state_dir, create=False)
    store = StateStore(harness_paths.database)
    created = session(tmp_path)
    store.create_session(created)
    records_module.materialize_all(harness_paths, store)

    result = sync_repository(harness_paths, attempts=1)

    assert result["state"] == "pending"
    assert result["detail"] == "git-fetch"

    def materialization_fails(*unused_args, **unused_kwargs):
        del unused_args
        del unused_kwargs
        raise ValueError("invalid portable record")

    monkeypatch.setattr(
        sync_module,
        "materialize_all",
        materialization_fails,
    )
    result = sync_module.publish_all(harness_paths, store)
    assert result["pending"] is True
    assert result["detail"] == "record-materialization"

    def materialization_succeeds(*unused_args, **unused_kwargs):
        del unused_args
        del unused_kwargs
        return []

    def synchronization_fails(*unused_args, **unused_kwargs):
        del unused_args
        del unused_kwargs
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(
        sync_module,
        "materialize_all",
        materialization_succeeds,
    )
    monkeypatch.setattr(
        sync_module,
        "sync_repository",
        synchronization_fails,
    )
    result = sync_module.publish_all(harness_paths, store)
    assert result["pending"] is True
    assert result["detail"] == "repository-synchronization"
    store.close()

    def time_out(*unused_args, **unused_kwargs):
        del unused_args
        del unused_kwargs
        raise subprocess.TimeoutExpired(["git"], 1)

    monkeypatch.setattr(sync_module.subprocess, "run", time_out)
    with pytest.raises(RuntimeError, match="timed out"):
        sync_module._git(tmp_path, "status")


def test_runtime_state_is_cleared_and_worktrees_are_rewritten(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    original = str(tmp_path / "old" / created.session_id)
    created = replace(created, worktree=original)
    store.create_session(created)
    store.register_worker(created.session_id, 123, "worker")
    store.create_process_lease(
        created.session_id,
        "codex",
        "interactive",
        "2099-01-01T00:00:00+00:00",
    )

    changed = store.rewrite_worktree_prefix(
        str(tmp_path / "old"),
        str(tmp_path / "new"),
    )
    store.clear_runtime_state()

    assert changed == 1
    assert store.get_session(created.session_id).worktree == str(
        tmp_path / "new" / created.session_id
    )
    assert not store.heartbeat_worker(created.session_id, "worker")
    assert store.active_process_leases() == []
    store.close()


def test_git_sync_commits_and_pushes_portable_records(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote.git"
    chat_root = tmp_path / "chats"
    _chat_repository(chat_root, remote)
    harness_paths = paths(chat_root)
    prepare_paths(harness_paths)
    store = StateStore(harness_paths.database)
    created = session(tmp_path)
    store.create_session(created)
    store.append_event(
        created.session_id,
        "agent.message",
        role="assistant",
        text="durable",
    )

    result = publish_all(harness_paths, store)

    assert result["state"] == "synced"
    assert read_sync_status(harness_paths)["pending"] is False
    assert (
        chat_root
        / "sessions"
        / created.session_id
        / "record.gpt.json"
    ).is_file()
    remote_record = _git(
        remote,
        "show",
        "main:sessions/"
        + created.session_id
        + "/transcript.gpt.md",
    )
    assert "durable" in remote_record
    store.close()


def test_legacy_migration_preserves_resume_state_and_git_worktree(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _workspace_repository(workspace)
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    source_store = StateStore(source_root / "state.sqlite3")
    created = session(workspace)
    worktree = create_worktree(
        workspace,
        source_root / "worktrees",
        created.session_id,
    )
    created = replace(created, worktree=str(worktree))
    source_store.create_session(created)
    source_store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="resume after migration",
    )
    source_store.close()

    remote = tmp_path / "remote.git"
    destination_root = tmp_path / "chats"
    _chat_repository(destination_root, remote)
    destination_paths = paths(destination_root)
    prepare_paths(destination_paths)
    existing_store = StateStore(destination_paths.database)
    existing = session(tmp_path / "existing")
    existing_store.create_session(existing)
    existing_store.append_event(
        existing.session_id,
        "agent.message",
        role="assistant",
        text="preserve destination state",
    )
    existing = existing_store.get_session(existing.session_id)
    assert publish_all(destination_paths, existing_store)["state"] == (
        "synced"
    )
    existing_store.close()

    result = migrate_state(
        source_root,
        destination_root,
        trash_source=False,
    )

    assert result["sessions"] == 1
    assert result["events"] == 1
    assert result["worktrees"] == 1
    destination_store = StateStore(destination_paths.database)
    migrated = destination_store.get_session(created.session_id)
    assert migrated.session_id == created.session_id
    assert migrated.worktree == str(
        destination_paths.worktrees / created.session_id
    )
    assert destination_store.all_events(created.session_id)[0].text == (
        "resume after migration"
    )
    assert destination_store.get_session(existing.session_id) == existing
    assert destination_store.all_events(existing.session_id)[0].text == (
        "preserve destination state"
    )
    assert len(destination_store.list_sessions()) == 2
    assert not worktree.exists()
    assert Path(migrated.worktree).is_dir()
    assert _git(Path(migrated.worktree), "status", "--porcelain=v1") == ""
    assert read_sync_status(destination_paths)["state"] == "synced"
    destination_store.close()


def test_failed_migration_restores_worktree_and_remains_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _workspace_repository(workspace)
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    source_store = StateStore(source_root / "state.sqlite3")
    created = session(workspace)
    worktree = create_worktree(
        workspace,
        source_root / "worktrees",
        created.session_id,
    )
    created = replace(created, worktree=str(worktree))
    source_store.create_session(created)
    source_store.close()
    remote = tmp_path / "remote.git"
    destination_root = tmp_path / "chats"
    _chat_repository(destination_root, remote)
    destination_paths = paths(destination_root)
    prepare_paths(destination_paths)
    destination_store = StateStore(destination_paths.database)
    existing = session(tmp_path / "existing")
    destination_store.create_session(existing)
    destination_store.append_event(
        existing.session_id,
        "agent.message",
        role="assistant",
        text="keep on rollback",
    )
    assert publish_all(destination_paths, destination_store)["state"] == (
        "synced"
    )
    destination_store.close()
    monkeypatch.setattr(
        migration_module,
        "sync_repository",
        lambda unused: {"state": "pending"},
    )

    with pytest.raises(RuntimeError, match="synchronize"):
        migrate_state(
            source_root,
            destination_root,
            trash_source=False,
        )

    assert worktree.is_dir()
    assert _git(worktree, "status", "--porcelain=v1") == ""
    restored_destination = StateStore(destination_paths.database)
    assert len(restored_destination.list_sessions()) == 1
    assert restored_destination.get_session(existing.session_id).session_id == (
        existing.session_id
    )
    assert restored_destination.all_events(existing.session_id)[0].text == (
        "keep on rollback"
    )
    restored_destination.close()
    retry_store = StateStore(source_root / "state.sqlite3")
    assert retry_store.get_session(created.session_id).worktree == str(
        worktree
    )
    retry_store.close()


def test_migration_validation_rejects_unsafe_roots_and_low_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(ValueError, match="database"):
        migration_module._validate_roots(
            migration_module.legacy_paths(missing),
            paths(destination),
        )

    source = tmp_path / "source"
    source.mkdir()
    source_store = StateStore(source / "state.sqlite3")
    source_store.close()
    with pytest.raises(ValueError, match="Git repository"):
        migration_module._validate_roots(
            migration_module.legacy_paths(source),
            paths(destination),
        )

    _workspace_repository(source, create=False)
    with pytest.raises(ValueError, match="must differ"):
        migration_module._validate_roots(
            migration_module.legacy_paths(source),
            paths(source),
        )

    monkeypatch.setattr(
        migration_module.shutil,
        "disk_usage",
        lambda unused: type("Usage", (), {"free": 0})(),
    )
    with pytest.raises(RuntimeError, match="disk space"):
        migration_module._require_headroom(source, destination)


def _chat_repository(
    root: Path,
    remote: Path,
    *,
    create: bool = True,
) -> None:
    _run(["git", "init", "--bare", str(remote)])
    if create:
        root.mkdir()
    _run(["git", "-C", str(root), "init", "-b", "main"])
    _configure_identity(root)
    (root / ".gitignore").write_text(".runtime/\n", encoding="utf-8")
    _run(["git", "-C", str(root), "add", ".gitignore"])
    _run(["git", "-C", str(root), "commit", "-m", "Initialize chats"])
    _run(["git", "-C", str(root), "remote", "add", "origin", str(remote)])
    _run(["git", "-C", str(root), "push", "-u", "origin", "main"])


def _workspace_repository(
    root: Path,
    *,
    create: bool = True,
) -> None:
    if create:
        root.mkdir()
    _run(["git", "-C", str(root), "init", "-b", "main"])
    _configure_identity(root)
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _run(["git", "-C", str(root), "add", "tracked.txt"])
    _run(["git", "-C", str(root), "commit", "-m", "Initialize workspace"])


def _configure_identity(root: Path) -> None:
    _run(["git", "-C", str(root), "config", "user.name", "Test User"])
    _run(
        [
            "git",
            "-C",
            str(root),
            "config",
            "user.email",
            "test@example.com",
        ]
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )
