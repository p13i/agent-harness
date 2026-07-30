from pathlib import Path

import pytest

from agent_harness.errors import ConflictError
from agent_harness.errors import NotFoundError
from agent_harness.goals import create_goal
from agent_harness.goals import make_evidence
from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Checkpoint
from agent_harness.models import CommandStatus
from agent_harness.models import ProviderAttempt
from agent_harness.storage import StateStore
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
    second = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "different"},
        "same-key",
    )
    assert first.command_id == second.command_id
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    assert claimed.status == CommandStatus.DISPATCHING
    assert store.recover_dispatching(created.session_id) == 1
    recovered = store.get_command(first.command_id)
    assert recovered.status == CommandStatus.FAILED
    assert recovered.result["code"] == "E_NEEDS_RECONCILIATION"
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
    assert store.heartbeat_worker(created.session_id, "incarnation")
    assert not store.heartbeat_worker(created.session_id, "other")
    store.remove_worker(created.session_id, "incarnation")

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
