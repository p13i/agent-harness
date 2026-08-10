"""Behavioral boundaries for durable storage and application service state."""

from __future__ import annotations

import asyncio
import copy
import subprocess
from dataclasses import replace
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import agent_harness.service as service_module
import agent_harness.storage as storage_module
from agent_harness.config import paths
from agent_harness.config import prepare_paths
from agent_harness.errors import ConflictError
from agent_harness.errors import NotFoundError
from agent_harness.errors import SafetyGuardError
from agent_harness.goals import create_goal
from agent_harness.goals import make_evidence
from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Checkpoint
from agent_harness.models import CommandStatus
from agent_harness.orchestration import normalized_digest
from agent_harness.service import HarnessService
from agent_harness.storage import StateStore
from tests.test_support import session

SUPPORTED_PROVIDERS = frozenset({"claude", "codex", "kimi", "grok"})


class WorkerProbe:
    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.forced: list[bool] = []

    def ensure(self, session_id: str, *, force: bool = False) -> None:
        self.sessions.append(session_id)
        self.forced.append(force)


class GoalStoreProbe:
    def __init__(self, goal: object | None) -> None:
        self.goal = goal

    def goal_for_session(self, unused_session_id: str) -> object | None:
        return self.goal


def _service(tmp_path: Path) -> HarnessService:
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    return HarnessService(
        harness_paths,
        worker_manager=WorkerProbe(),  # type: ignore[arg-type]
    )


def _owned_session(tmp_path: Path):
    return replace(
        session(tmp_path),
        external_ref={
            "orchestrator": "p13i/machines/agent-harness-proof",
            "job_id": "boundary-job",
        },
    )


def _proof_probe_payload(value: object, key: str) -> tuple[dict[str, Any], str]:
    authorization = {
        "schema": "p13i/machines/provider-fault-authorization/v1",
        "external_ref": value.external_ref,  # type: ignore[attr-defined]
        "idempotency_key": key,
        "stage": "after-lease-before-acceptance",
        "provider": "codex",
        "agent_role": "proof-fault-probe",
    }
    digest = normalized_digest(authorization)
    return (
        {
            "text": "exercise an authorized fault barrier",
            "turn_ref": {
                "step_id": "fault-step",
                "agent_role": "proof-fault-probe",
            },
            "required_capabilities": ["proof-fault-barrier"],
            "proof_fault_probe": {
                "stage": "after-lease-before-acceptance",
                "provider": "codex",
                "authorization_digest": digest,
                "authorization": authorization,
            },
        },
        digest,
    )


def _service_probe_payload(value: object, key: str) -> tuple[dict[str, Any], str]:
    authorization = {
        "schema": "p13i/machines/service-fault-authorization/v1",
        "external_ref": value.external_ref,  # type: ignore[attr-defined]
        "idempotency_key": key,
        "stage": "after-acceptance-before-terminal",
        "provider": "claude",
        "agent_role": "proof-service-fault-probe",
    }
    digest = normalized_digest(authorization)
    return (
        {
            "text": "exercise an authorized service barrier",
            "provider": "claude",
            "turn_ref": {
                "step_id": "service-fault-step",
                "agent_role": "proof-service-fault-probe",
            },
            "required_capabilities": ["proof-service-fault-barrier"],
            "proof_service_fault_probe": {
                "stage": "after-acceptance-before-terminal",
                "provider": "claude",
                "authorization_digest": digest,
                "authorization": authorization,
            },
        },
        digest,
    )


def test_nested_rollback_is_isolated_and_durable(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    store = StateStore(database)
    created = session(tmp_path)
    store.create_session(created)

    with store.transaction() as connection:
        connection.execute(
            "UPDATE sessions SET name = 'outer-durable' WHERE session_id = ?",
            (created.session_id,),
        )
        with pytest.raises(RuntimeError, match="nested failure"):
            with store.transaction() as nested:
                nested.execute(
                    "UPDATE sessions SET name = 'transient' WHERE session_id = ?",
                    (created.session_id,),
                )
                raise RuntimeError("nested failure")
        row = connection.execute(
            "SELECT name FROM sessions WHERE session_id = ?",
            (created.session_id,),
        ).fetchone()
        assert row is not None
        assert row["name"] == "outer-durable"
    store.close()

    recovered = StateStore(database)
    assert recovered.get_session(created.session_id).name == "outer-durable"
    recovered.close()


def test_history_compaction_bounds_data_and_reports_open_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    for index in range(25):
        store.append_event(
            created.session_id,
            "event." + str(index % 2),
            role="assistant",
            text=("x" * 700) + str(index),
            status="complete",
            metadata={
                "deep": {"a": {"b": {"c": {"d": "hidden"}}}},
                "items": list(range(30)),
            },
        )

    summary = store.context_history_summary(created.session_id, 25)
    assert summary["event_count"] == 25
    assert summary["first_sequence"] == 1
    assert summary["last_sequence"] == 25
    assert len(summary["history_digest"]) == 64
    assert summary["event_type_counts"] == {"event.0": 13, "event.1": 12}
    assert len(summary["anchors"]) == 25
    first = summary["anchors"][0]
    assert len(first["text"]) == 500
    assert first["metadata"]["deep"]["a"]["b"]["c"] == "[depth limit]"
    assert len(first["metadata"]["items"]) == 20
    assert store.context_history_summary(created.session_id, 0) == {}

    reconciliation = SimpleNamespace(
        reconciliation_id="reconciliation-one",
        status="pending",
        command_id="command-one",
        created_at="later",
    )
    monkeypatch.setattr(
        store,
        "pending_approvals",
        lambda unused: [
            {
                "approval_id": "approval-one",
                "status": "pending",
                "kind": "tool",
                "prompt": "safe prompt",
                "reason": "approval needed",
                "created_at": "earlier",
            }
        ],
    )
    monkeypatch.setattr(
        store,
        "pending_reconciliations",
        lambda unused: [reconciliation],
    )
    decisions = store.context_unresolved_decisions(created.session_id)
    assert [item["kind"] for item in decisions] == ["approval", "reconciliation"]
    assert decisions[0]["prompt"] == "safe prompt"
    assert decisions[1]["command_id"] == "command-one"

    bounded = storage_module._bounded_json_value
    assert bounded("z" * 1_200) == "z" * 1_000
    assert bounded(list(range(30))) == list(range(20))
    assert bounded(None) is None
    assert bounded(3.5) == 3.5
    assert bounded(Path("p")) == "p"
    store.close()


def test_fork_lineage_rejects_a_missing_source_checkpoint(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    source = session(tmp_path)
    store.create_session(created)
    store.create_session(source)
    missing_checkpoint = new_uuid()
    store.append_event(
        created.session_id,
        "session.forked",
        metadata={
            "source_session_id": source.session_id,
            "source_sequence": 8,
            "source_checkpoint_id": missing_checkpoint,
        },
    )

    with pytest.raises(ConflictError, match="source checkpoint is missing"):
        store.fork_lineage(created.session_id)
    store.add_checkpoint(
        Checkpoint(
            checkpoint_id=missing_checkpoint,
            session_id=source.session_id,
            sequence=8,
            provider="codex",
            native_session_id="native",
            base_commit="base",
            patch_digest="patch",
            untracked_digest="untracked",
            context_digest="source-context",
            created_at=utc_now(),
        )
    )
    assert store.fork_lineage(created.session_id) == {
        "source_session_id": source.session_id,
        "source_sequence": 8,
        "source_checkpoint_id": missing_checkpoint,
        "source_context_digest": "source-context",
    }
    store.close()


def test_context_delivery_replay_is_durable_and_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = StateStore(database)
    created = session(tmp_path)
    store.create_session(created)
    prepared = store.prepare_context_delivery(
        created.session_id,
        "codex",
        "context-a",
        "checkpoint-a",
        "command-a",
        "attempt-a",
        "payload-a",
    )
    replay = store.prepare_context_delivery(
        created.session_id,
        "codex",
        "context-a",
        "checkpoint-a",
        "command-a",
        "attempt-a",
        "payload-a",
    )
    assert replay == prepared
    store.close()

    recovered = StateStore(database)
    with pytest.raises(ConflictError, match="acceptance is stale"):
        recovered.accept_context_delivery(
            created.session_id,
            "codex",
            "context-a",
            "attempt-stale",
        )
    with recovered.transaction() as connection:
        connection.execute(
            "UPDATE context_deliveries SET state = 'corrupt'",
        )
    with pytest.raises(ConflictError, match="acceptance is invalid"):
        recovered.accept_context_delivery(
            created.session_id,
            "codex",
            "context-a",
            "attempt-a",
        )
    with recovered.transaction() as connection:
        connection.execute(
            "UPDATE context_deliveries SET state = 'prepared'",
        )
    accepted = recovered.accept_context_delivery(
        created.session_id,
        "codex",
        "context-a",
        "attempt-a",
    )
    assert accepted["state"] == "delivered"
    assert recovered.accept_context_delivery(
        created.session_id,
        "codex",
        "context-a",
        "attempt-a",
    ) == accepted
    with pytest.raises(ConflictError, match="already delivered"):
        recovered.prepare_context_delivery(
            created.session_id,
            "codex",
            "context-a",
            "checkpoint-b",
            "command-b",
            "attempt-b",
            "payload-b",
        )
    recovered.close()


def test_context_delivery_missing_writes_roll_back_atomically(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER erase_prepared_context
            AFTER INSERT ON context_deliveries
            BEGIN
                DELETE FROM context_deliveries;
            END
            """
        )
    with pytest.raises(RuntimeError, match="preparation was not recorded"):
        store.prepare_context_delivery(
            created.session_id,
            "codex",
            "context-a",
            "checkpoint-a",
            "command-a",
            "attempt-a",
            "payload-a",
        )
    with store.transaction() as connection:
        connection.execute("DROP TRIGGER erase_prepared_context")
    prepared = store.prepare_context_delivery(
        created.session_id,
        "codex",
        "context-a",
        "checkpoint-a",
        "command-a",
        "attempt-a",
        "payload-a",
    )
    assert prepared["state"] == "prepared"

    with store.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER erase_accepted_context
            AFTER UPDATE OF state ON context_deliveries
            WHEN NEW.state = 'delivered'
            BEGIN
                DELETE FROM context_deliveries;
            END
            """
        )
    with pytest.raises(RuntimeError, match="acceptance was not recorded"):
        store.accept_context_delivery(
            created.session_id,
            "codex",
            "context-a",
            "attempt-a",
        )
    with store.transaction() as connection:
        connection.execute("DROP TRIGGER erase_accepted_context")
    assert store.prepare_context_delivery(
        created.session_id,
        "codex",
        "context-a",
        "checkpoint-a",
        "command-a",
        "attempt-a",
        "payload-a",
    )["state"] == "prepared"
    store.close()


def test_xhigh_authorization_rejects_stale_identity_and_inactive_commands(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    first = session(tmp_path)
    second = session(tmp_path)
    store.create_session(first)
    store.create_session(second)

    with pytest.raises(ValueError, match="provider is unsupported"):
        store.create_xhigh_authorization(
            first.session_id,
            new_uuid(),
            "",
            authorization_request_digest="a" * 64,
            idempotency_key="provider-key",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    with pytest.raises(ValueError, match="idempotency key is required"):
        store.create_xhigh_authorization(
            first.session_id,
            new_uuid(),
            "codex",
            authorization_request_digest="a" * 64,
            idempotency_key="",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    with pytest.raises(NotFoundError, match="command"):
        store.create_xhigh_authorization(
            first.session_id,
            new_uuid(),
            "codex",
            authorization_request_digest="a" * 64,
            idempotency_key="missing-command",
            expires_at="2099-01-01T00:00:00+00:00",
        )

    active = store.enqueue_command(
        first.session_id,
        "message",
        {"text": "exact", "provider": "codex", "effort": "xhigh"},
        "active-xhigh",
    )
    with pytest.raises(ConflictError, match="changed session"):
        store.create_xhigh_authorization(
            second.session_id,
            active.command_id,
            "codex",
            authorization_request_digest="b" * 64,
            idempotency_key="wrong-session",
            expires_at="2099-01-01T00:00:00+00:00",
        )

    inactive = store.enqueue_command(
        first.session_id,
        "message",
        {"text": "done", "provider": "codex", "effort": "xhigh"},
        "inactive-xhigh",
    )
    store.resolve_command(inactive.command_id, CommandStatus.COMPLETE, {})
    with pytest.raises(ConflictError, match="is not active"):
        store.create_xhigh_authorization(
            first.session_id,
            inactive.command_id,
            "codex",
            authorization_request_digest="c" * 64,
            idempotency_key="inactive-command",
            expires_at="2099-01-01T00:00:00+00:00",
        )

    wrong_effort = store.enqueue_command(
        first.session_id,
        "message",
        {"text": "ordinary", "provider": "codex", "effort": "high"},
        "ordinary-command",
    )
    with pytest.raises(ConflictError, match="effort changed"):
        store.create_xhigh_authorization(
            first.session_id,
            wrong_effort.command_id,
            "codex",
            authorization_request_digest="d" * 64,
            idempotency_key="wrong-effort",
            expires_at="2099-01-01T00:00:00+00:00",
        )

    authorization = store.create_xhigh_authorization(
        first.session_id,
        active.command_id,
        "codex",
        authorization_request_digest="e" * 64,
        idempotency_key="durable-authorization",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    assert store.xhigh_authorization(active.command_id) == authorization
    with pytest.raises(ConflictError, match="key was reused"):
        store.create_xhigh_authorization(
            first.session_id,
            active.command_id,
            "codex",
            authorization_request_digest="f" * 64,
            idempotency_key="durable-authorization",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    store.close()


def test_route_admission_rejects_stale_ownership_and_boundaries(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)

    def command_with_envelope(key: str, profile: str = "unattended"):
        command = store.enqueue_command(
            created.session_id,
            "message",
            {"text": key, "effort": "high"},
            key,
        )
        store.create_command_envelope(
            command.command_id,
            created.session_id,
            profile,
            {"max_attempts": 1},
        )
        return command

    arguments = {
        "effort": "high",
        "attempt_id": "",
        "worker_incarnation": "worker-one",
        "goal_id": "",
        "max_concurrency": 1,
        "lease_expires_at": "2099-01-01T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="concurrency must be positive"):
        store.reserve_route_admission(
            new_uuid(),
            "codex",
            "unattended",
            **{**arguments, "max_concurrency": 0},
        )
    with pytest.raises(NotFoundError, match="command envelope"):
        store.reserve_route_admission(
            new_uuid(),
            "codex",
            "unattended",
            **arguments,
        )

    wrong_profile = command_with_envelope("wrong-profile")
    with pytest.raises(ConflictError, match="profile changed"):
        store.reserve_route_admission(
            wrong_profile.command_id,
            "codex",
            "interactive",
            **arguments,
        )
    terminal = command_with_envelope("terminal-envelope")
    store.update_command_envelope(terminal.command_id, state="complete")
    with pytest.raises(ConflictError, match="not reservable"):
        store.reserve_route_admission(
            terminal.command_id,
            "codex",
            "unattended",
            **arguments,
        )
    missing_worker = command_with_envelope("missing-worker")
    with pytest.raises(RuntimeError, match="lost route-admission ownership"):
        store.reserve_route_admission(
            missing_worker.command_id,
            "codex",
            "unattended",
            **arguments,
        )

    store.register_worker(created.session_id, 123, "worker-one")
    changed_goal = command_with_envelope("changed-goal")
    with pytest.raises(ConflictError, match="goal changed"):
        store.reserve_route_admission(
            changed_goal.command_id,
            "codex",
            "unattended",
            **{**arguments, "goal_id": new_uuid()},
        )
    xhigh = command_with_envelope("missing-xhigh")
    denied = store.reserve_route_admission(
        xhigh.command_id,
        "codex",
        "unattended",
        **{**arguments, "effort": "xhigh"},
    )
    assert denied == {
        "admitted": False,
        "reason": "xhigh-authorization",
        "lease_id": "",
    }
    stale_boundary = command_with_envelope("stale-boundary", "interactive")
    with pytest.raises(ConflictError, match="dispatch boundary is stale"):
        store.reserve_route_admission(
            stale_boundary.command_id,
            "codex",
            "interactive",
            **{**arguments, "attempt_id": new_uuid()},
        )
    assert store.command_envelope(stale_boundary.command_id)["state"] == "reserved"
    store.close()


def test_milestone_and_evidence_failures_roll_back_and_replay_once(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    goal = create_goal(
        created.session_id,
        "retain atomic goal state",
        milestones=(
            {
                "milestone_id": new_uuid(),
                "title": "first",
                "predicates": [{"type": "test", "outcome": "passed"}],
            },
        ),
    )
    store.create_goal(goal)
    changed = replace(goal.milestones[0], status="complete")
    missing = replace(changed, milestone_id=new_uuid())
    with pytest.raises(NotFoundError, match="goal milestone"):
        store.update_milestone_statuses(goal.goal_id, (changed, missing))
    assert store.get_goal(goal.goal_id).milestones[0].status == "active"

    evidence = make_evidence(goal.goal_id, "test", "boundary", "passed")
    with pytest.raises(NotFoundError, match="session"):
        store.add_evidence_once(
            new_uuid(),
            evidence,
            idempotency_key="missing-session",
            request_digest="missing-digest",
        )
    wrong_goal = replace(evidence, evidence_id=new_uuid(), goal_id=new_uuid())
    with pytest.raises(ConflictError, match="no longer current"):
        store.add_evidence_once(
            created.session_id,
            wrong_goal,
            idempotency_key="wrong-goal",
            request_digest="wrong-digest",
        )

    operation = "evidence-create:" + created.session_id
    store.record_mutation_receipt(
        "orphan-evidence",
        operation,
        "orphan-digest",
        {"evidence": {"evidence_id": new_uuid()}},
        201,
    )
    with pytest.raises(RuntimeError, match="has no evidence row"):
        store.add_evidence_once(
            created.session_id,
            evidence,
            idempotency_key="orphan-evidence",
            request_digest="orphan-digest",
        )
    with pytest.raises(ConflictError, match="another mutation"):
        store.mutation_receipt(
            "orphan-evidence",
            "different-operation",
            "different-digest",
        )

    barrier = threading.Barrier(3)
    results: list[object] = []
    failures: list[BaseException] = []

    def add_once() -> None:
        barrier.wait()
        try:
            result = store.add_evidence_once(
                created.session_id,
                evidence,
                idempotency_key="concurrent-evidence",
                request_digest="concurrent-digest",
            )
            results.append(result)
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=add_once) for unused in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
    assert not failures
    assert len(results) == 2
    assert results[0] == results[1] == evidence
    assert store.evidence(goal.goal_id) == [evidence]
    store.close()


def test_legacy_session_import_retains_goal_milestones_and_evidence(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    session_id = new_uuid()
    goal_id = new_uuid()
    milestone_id = new_uuid()
    evidence_id = new_uuid()
    imported = store.import_session(
        {
            "session": {
                "session_id": session_id,
                "name": "legacy",
                "workspace": str(tmp_path),
                "goal_id": goal_id,
            },
            "goal": {
                "goal_id": goal_id,
                "objective": "retain legacy goal",
                "constraints": ["durable"],
                "predicates": [{"type": "test", "outcome": "passed"}],
                "milestones": [
                    {
                        "milestone_id": milestone_id,
                        "title": "imported milestone",
                        "dependencies": [],
                        "predicates": [],
                        "position": 0,
                    }
                ],
            },
            "evidence": [
                {
                    "evidence_id": evidence_id,
                    "evidence_type": "test",
                    "subject": "legacy",
                    "outcome": "passed",
                    "value": {"durable": True},
                }
            ],
        },
        worktree=str(tmp_path / "restored"),
        owner_host="restored-host",
        owner_epoch=2,
    )
    assert imported.goal_id == goal_id
    imported_goal = store.get_goal(goal_id)
    assert imported_goal.objective == "retain legacy goal"
    assert imported_goal.milestones[0].milestone_id == milestone_id
    imported_evidence = store.evidence(goal_id)
    assert imported_evidence[0].evidence_id == evidence_id
    assert imported_evidence[0].value == {"durable": True}
    store.close()


def test_retained_dispatch_policy_detects_durable_corruption(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    goal = create_goal(created.session_id, "retain an exact transition policy")
    store.create_goal(goal)
    policy_digest = "a" * 64
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO dispatch_transition_policies VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                policy_digest,
                created.session_id,
                goal.goal_id,
                "epoch-one",
                "p13i/agent-harness/dispatch-generation-transition-policy/v1",
                "{}",
                utc_now(),
            ),
        )
    with pytest.raises(ConflictError, match="retained policy is corrupt"):
        store.dispatch_transition_policy(
            created.session_id,
            goal.goal_id,
            "epoch-one",
            policy_digest,
        )
    with pytest.raises(NotFoundError, match="session"):
        store.dispatch_transition_anchor(new_uuid())
    store.close()


def test_creation_intent_replay_cleanup_and_malformed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    creation = {
        "workspace": str(tmp_path),
        "name": "durable",
        "permission_mode": "approval",
        "execution_profile": "unattended",
        "direct": False,
        "external_ref": {},
        "routing": {"model": "", "effort": ""},
    }
    preferred = new_uuid()
    reserved, intent_path = service_module._reserve_creation_intent(
        harness_paths,
        "creation-key",
        creation,
        preferred,
    )
    assert reserved == preferred
    assert intent_path is not None
    replay, replay_path = service_module._reserve_creation_intent(
        harness_paths,
        "creation-key",
        creation,
        new_uuid(),
    )
    assert replay == preferred
    assert replay_path == intent_path

    changed = dict(creation)
    changed["name"] = "changed"
    with pytest.raises(ConflictError, match="key was reused"):
        service_module._reserve_creation_intent(
            harness_paths,
            "creation-key",
            changed,
            new_uuid(),
        )
    with pytest.raises(ConflictError, match="key was reused"):
        service_module._complete_creation_intent(
            harness_paths,
            "creation-key",
            changed,
            preferred,
        )
    with pytest.raises(ConflictError, match="identity changed"):
        service_module._complete_creation_intent(
            harness_paths,
            "creation-key",
            creation,
            new_uuid(),
        )
    service_module._complete_creation_intent(
        harness_paths,
        "creation-key",
        creation,
        preferred,
    )
    assert not intent_path.exists()

    malformed = harness_paths.runtime / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    with pytest.raises(ConflictError, match="unreadable"):
        service_module._read_creation_intent(malformed)
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ConflictError, match="invalid"):
        service_module._read_creation_intent(malformed)
    malformed.write_text("{}", encoding="utf-8")
    with pytest.raises(ConflictError, match="incompatible"):
        service_module._read_creation_intent(malformed)
    link = harness_paths.runtime / "intent-link.json"
    link.symlink_to(malformed)
    with pytest.raises(ValueError, match="invalid"):
        service_module._read_creation_intent(link)

    original_sync = service_module._sync_directory

    def fail_sync(unused_path: Path) -> None:
        raise OSError("directory sync failed")

    monkeypatch.setattr(service_module, "_sync_directory", fail_sync)
    failed_path = service_module._creation_intent_path(harness_paths, "sync-failure")
    with pytest.raises(OSError, match="directory sync failed"):
        service_module._reserve_creation_intent(
            harness_paths,
            "sync-failure",
            creation,
            new_uuid(),
        )
    assert not failed_path.exists()
    monkeypatch.setattr(service_module, "_sync_directory", original_sync)
    service_module._complete_creation_intent(
        harness_paths,
        "",
        creation,
        preferred,
    )


def test_session_creation_recovers_a_durable_intent_after_losing_a_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(
        ["git", "-C", str(workspace), "init", "-q"],
        check=True,
    )
    removed: list[tuple[Path, Path, str]] = []
    attempts = 0

    monkeypatch.setattr(
        service.store,
        "existing_ensured_session",
        lambda *unused, **unused_values: None,
    )
    monkeypatch.setattr(
        service_module,
        "create_worktree",
        lambda unused_workspace, unused_root, session_id, **unused_values: (
            tmp_path / ("worktree-" + session_id)
        ),
    )

    def remove(workspace_path: Path, worktree: Path, session_id: str) -> None:
        removed.append((workspace_path, worktree, session_id))

    monkeypatch.setattr(service_module, "remove_worktree", remove)

    def ensure(candidate: object, *unused: object, **unused_values: object):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("database unavailable")
        return candidate, False

    monkeypatch.setattr(service.store, "ensure_session", ensure)
    payload = {"workspace": str(workspace), "name": "race loser"}
    with pytest.raises(RuntimeError, match="database unavailable"):
        service.create_session(payload, idempotency_key="race-key")
    intent_path = service_module._creation_intent_path(service.paths, "race-key")
    assert intent_path.is_file()
    reserved = service_module._read_creation_intent(intent_path)["session_id"]

    recovered = service.create_session(payload, idempotency_key="race-key")
    assert recovered.session_id == reserved
    assert attempts == 2
    assert len(removed) == 2
    assert removed[0][2] == removed[1][2] == reserved
    assert not intent_path.exists()
    service.close()


def test_service_rejects_malformed_session_and_message_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    malformed_fields = (
        ("milestones", {}, "milestones must be an array"),
        ("permitted_providers", {}, "permitted_providers must be an array"),
        ("permitted_efforts", {}, "permitted_efforts must be an array"),
    )
    for name, value, message in malformed_fields:
        with pytest.raises(ValueError, match=message):
            service.create_session(
                {"workspace": str(workspace), name: value},
            )

    owned = _owned_session(tmp_path)
    service.store.create_session(owned)
    monkeypatch.setattr(
        service_module,
        "_validate_proof_fault_probe",
        lambda *unused: None,
    )
    monkeypatch.setattr(
        service_module,
        "_validate_service_fault_probe",
        lambda *unused: None,
    )
    with pytest.raises(ValueError, match="only one proof fault probe"):
        service.submit_message(
            owned.session_id,
            {
                "proof_fault_probe": {},
                "proof_service_fault_probe": {},
            },
            "both-probes",
        )

    monkeypatch.setattr(
        service_module,
        "require_state_headroom",
        lambda unused_path, unused_provider: 10 * 1024**3,
    )
    with pytest.raises(ValueError, match="unsupported per-turn permission mode"):
        service.submit_message(
            owned.session_id,
            {"text": "invalid permission", "permission_mode": "invalid"},
            "invalid-permission",
        )
    command_id = new_uuid()
    with pytest.raises(ValueError, match="provider is unsupported"):
        service.extend_budget(
            owned.session_id,
            {
                "reason": "bounded authorization",
                "allow_xhigh_once": True,
                "command_id": command_id,
                "provider": "other",
            },
            idempotency_key="xhigh-key",
        )
    with pytest.raises(ValueError, match="idempotency key is required"):
        service.extend_budget(
            owned.session_id,
            {
                "reason": "bounded authorization",
                "allow_xhigh_once": True,
                "command_id": command_id,
                "provider": "codex",
            },
        )
    service.close()


def test_service_capacity_lease_and_route_preview_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    created = session(tmp_path)
    service.store.create_session(created)
    monkeypatch.setattr(
        service.store,
        "latest_usage",
        lambda: {
            "codex": {
                "observed_at": utc_now(),
                "binding_percent": -1,
                "credits_engaged": False,
            }
        },
    )
    with pytest.raises(SafetyGuardError, match="execution safety guard stopped"):
        service._require_process_lease_capacity("codex", "unattended")

    lease = service.store.create_process_lease(
        created.session_id,
        "codex",
        "unattended",
        "2099-01-01T00:00:00+00:00",
    )
    service.update_process_lease(
        str(lease["lease_id"]),
        {"action": "attach", "pid": 101, "pid_start": "first"},
    )
    with pytest.raises(ConflictError, match="process identity is immutable"):
        service.update_process_lease(
            str(lease["lease_id"]),
            {"action": "attach", "pid": 202, "pid_start": "second"},
        )

    service.store.set_session_safety(created.session_id, "unattended")
    goal = create_goal(
        created.session_id,
        "bound route selection",
        permitted_providers=("codex",),
        permitted_efforts=("high",),
    )
    service.store.create_goal(goal)
    with pytest.raises(ValueError, match="metered budget must be positive"):
        asyncio.run(
            service.preview_route(
                created.session_id,
                {"effort": "high", "metered_budget": 0},
            )
        )
    service.close()


def test_proof_fault_probe_validation_fails_closed(tmp_path: Path) -> None:
    owned = _owned_session(tmp_path)
    key = "proof-probe-key"
    valid, digest = _proof_probe_payload(owned, key)
    goal = create_goal(
        owned.session_id,
        "authorize one proof fault",
        constraints=("proof-fault-authorization-sha256:" + digest,),
    )
    service_module._validate_proof_fault_probe(
        owned, goal, valid, key, SUPPORTED_PROVIDERS
    )

    cases: list[tuple[object, object | None, dict[str, Any], str, str]] = []
    bad = copy.deepcopy(valid)
    bad["proof_fault_probe"] = "invalid"
    cases.append((owned, goal, bad, key, "must be an object"))
    cases.append((owned, goal, valid, "", "requires an idempotency key"))
    bad = copy.deepcopy(valid)
    bad["provider"] = "codex"
    cases.append((owned, goal, bad, key, "requires automatic routing"))
    cases.append((session(tmp_path), goal, valid, key, "requires p13i/machines"))
    bad = copy.deepcopy(valid)
    bad["turn_ref"] = []
    cases.append((owned, goal, bad, key, "managed turn reference"))
    bad = copy.deepcopy(valid)
    bad["turn_ref"]["agent_role"] = "other"
    cases.append((owned, goal, bad, key, "role is unauthorized"))
    bad = copy.deepcopy(valid)
    bad["required_capabilities"] = "proof-fault-barrier"
    cases.append((owned, goal, bad, key, "requires capabilities"))
    bad = copy.deepcopy(valid)
    bad["required_capabilities"] = []
    cases.append((owned, goal, bad, key, "requires proof-fault-barrier"))
    bad = copy.deepcopy(valid)
    bad["proof_fault_probe"]["stage"] = "other"
    cases.append((owned, goal, bad, key, "stage is unsupported"))
    bad = copy.deepcopy(valid)
    bad["proof_fault_probe"]["provider"] = "other"
    cases.append((owned, goal, bad, key, "provider is unsupported"))
    bad = copy.deepcopy(valid)
    bad["proof_fault_probe"]["authorization_digest"] = "short"
    cases.append((owned, goal, bad, key, "digest is invalid"))
    bad = copy.deepcopy(valid)
    bad["proof_fault_probe"]["authorization_digest"] = "z" * 64
    cases.append((owned, goal, bad, key, "digest is invalid"))
    bad = copy.deepcopy(valid)
    bad["proof_fault_probe"]["authorization"] = []
    cases.append((owned, goal, bad, key, "authorization is required"))
    bad = copy.deepcopy(valid)
    bad["proof_fault_probe"]["authorization"]["provider"] = "claude"
    cases.append((owned, goal, bad, key, "authorization does not match"))
    bad = copy.deepcopy(valid)
    bad["proof_fault_probe"]["authorization_digest"] = "0" * 64
    cases.append((owned, goal, bad, key, "digest does not match"))
    cases.append((owned, None, valid, key, "is not precommitted"))

    for value, current_goal, payload, current_key, message in cases:
        with pytest.raises(ValueError, match=message):
            service_module._validate_proof_fault_probe(
                value,  # type: ignore[arg-type]
                current_goal,  # type: ignore[arg-type]
                payload,
                current_key,
                SUPPORTED_PROVIDERS,
            )


def test_service_fault_probe_validation_fails_closed(tmp_path: Path) -> None:
    owned = _owned_session(tmp_path)
    key = "service-probe-key"
    valid, digest = _service_probe_payload(owned, key)
    goal = create_goal(
        owned.session_id,
        "authorize one service fault",
        constraints=("proof-service-fault-authorization-sha256:" + digest,),
    )
    service_module._validate_service_fault_probe(
        owned, goal, valid, key, SUPPORTED_PROVIDERS
    )

    cases: list[tuple[object, object | None, dict[str, Any], str, str]] = []
    bad = copy.deepcopy(valid)
    bad["proof_service_fault_probe"] = "invalid"
    cases.append((owned, goal, bad, key, "must be an object"))
    cases.append((owned, goal, valid, "", "requires an idempotency key"))
    cases.append((session(tmp_path), goal, valid, key, "requires p13i/machines"))
    bad = copy.deepcopy(valid)
    bad["turn_ref"] = []
    cases.append((owned, goal, bad, key, "managed turn reference"))
    bad = copy.deepcopy(valid)
    bad["turn_ref"]["agent_role"] = "other"
    cases.append((owned, goal, bad, key, "role is unauthorized"))
    bad = copy.deepcopy(valid)
    bad["required_capabilities"] = "proof-service-fault-barrier"
    cases.append((owned, goal, bad, key, "requires capabilities"))
    bad = copy.deepcopy(valid)
    bad["required_capabilities"] = []
    cases.append((owned, goal, bad, key, "requires proof-service-fault-barrier"))
    bad = copy.deepcopy(valid)
    bad["proof_service_fault_probe"]["stage"] = "other"
    cases.append((owned, goal, bad, key, "stage is unsupported"))
    bad = copy.deepcopy(valid)
    bad["proof_service_fault_probe"]["provider"] = "other"
    cases.append((owned, goal, bad, key, "provider is unsupported"))
    bad = copy.deepcopy(valid)
    bad["provider"] = "codex"
    cases.append((owned, goal, bad, key, "provider does not match"))
    bad = copy.deepcopy(valid)
    bad["proof_service_fault_probe"]["authorization_digest"] = "short"
    cases.append((owned, goal, bad, key, "digest is invalid"))
    bad = copy.deepcopy(valid)
    bad["proof_service_fault_probe"]["authorization_digest"] = "z" * 64
    cases.append((owned, goal, bad, key, "digest is invalid"))
    bad = copy.deepcopy(valid)
    bad["proof_service_fault_probe"]["authorization"] = []
    cases.append((owned, goal, bad, key, "authorization is required"))
    bad = copy.deepcopy(valid)
    bad["proof_service_fault_probe"]["authorization"]["provider"] = "codex"
    cases.append((owned, goal, bad, key, "authorization does not match"))
    bad = copy.deepcopy(valid)
    bad["proof_service_fault_probe"]["authorization_digest"] = "0" * 64
    cases.append((owned, goal, bad, key, "digest does not match"))
    cases.append((owned, None, valid, key, "is not precommitted"))

    for value, current_goal, payload, current_key, message in cases:
        with pytest.raises(ValueError, match=message):
            service_module._validate_service_fault_probe(
                value,  # type: ignore[arg-type]
                current_goal,  # type: ignore[arg-type]
                payload,
                current_key,
                SUPPORTED_PROVIDERS,
            )


def test_typed_authorization_helpers_reject_mutated_receipts() -> None:
    receipt = {"grant": "one bounded transition"}
    receipt_digest = normalized_digest(receipt)
    promotion = {
        "schema": "p13i/agent-harness/goal-promotion-authorization/v1",
        "session_id": "session-one",
        "from_goal_id": "goal-one",
        "stage": "stage-two",
        "receipt_sha256": receipt_digest,
        "receipt": receipt,
        "budget_increases": {"turns": 1.0},
        "next_goal_contract_digest": "contract-digest",
    }
    promotion_arguments = {
        "session_id": "session-one",
        "previous_goal_id": "goal-one",
        "stage": "stage-two",
        "budget_increases": {"turns": 1.0},
        "next_goal_contract_digest": "contract-digest",
    }
    service_module._require_promotion_authorization(
        promotion,
        **promotion_arguments,
    )
    promotion_cases = (
        ("schema", "other", "schema is unsupported"),
        ("session_id", "other", "session does not match"),
        ("from_goal_id", "other", "source does not match"),
        ("stage", "other", "stage does not match"),
        ("receipt_sha256", "bad", "receipt is invalid"),
        ("budget_increases", {}, "budget does not match"),
        ("next_goal_contract_digest", "other", "contract does not match"),
    )
    for name, value, message in promotion_cases:
        changed = copy.deepcopy(promotion)
        changed[name] = value
        with pytest.raises(ValueError, match=message):
            service_module._require_promotion_authorization(
                changed,
                **promotion_arguments,
            )

    adoption = {
        "schema": "p13i/agent-harness/session-contract-adoption-authorization/v1",
        "session_id": "session-one",
        "external_ref": {"orchestrator": "test", "job_id": "one"},
        "previous_goal_digest": "previous",
        "goal_envelope_digest": "next",
        "receipt_sha256": receipt_digest,
        "receipt": receipt,
    }
    adoption_arguments = {
        "session_id": "session-one",
        "external_ref": {"orchestrator": "test", "job_id": "one"},
        "previous_goal_digest": "previous",
        "goal_envelope_digest": "next",
    }
    service_module._require_contract_adoption_authorization(
        adoption,
        **adoption_arguments,
    )
    adoption_cases = (
        ("schema", "other", "schema is unsupported"),
        ("session_id", "other", "session does not match"),
        ("external_ref", {}, "external reference does not match"),
        ("previous_goal_digest", "other", "previous goal does not match"),
        ("goal_envelope_digest", "other", "goal envelope does not match"),
        ("receipt_sha256", "bad", "receipt is invalid"),
    )
    for name, value, message in adoption_cases:
        changed = copy.deepcopy(adoption)
        changed[name] = value
        with pytest.raises(ValueError, match=message):
            service_module._require_contract_adoption_authorization(
                changed,
                **adoption_arguments,
            )

    with pytest.raises(ValueError, match="retain its source receipt"):
        service_module._require_retained_authorization_receipt(
            {"receipt": {}},
            receipt_digest,
        )
    with pytest.raises(ValueError, match="digest does not match"):
        service_module._require_retained_authorization_receipt(
            {"receipt": receipt},
            "0" * 64,
        )


def test_machine_goal_envelope_rejects_unbounded_or_untyped_fields() -> None:
    budgets = {
        "seconds": 1,
        "turns": 1,
        "tokens": 1,
        "tool_calls": 1,
        "output_tokens": 1,
        "context_tokens": 1,
        "dollars": 1,
        "attempts": 1,
        "child_agents": 1,
    }
    payload = {
        "goal_kind": "finite",
        "predicates": [{"type": "test"}],
        "constraints": ["bounded"],
        "milestones": [{"title": "one"}],
        "budgets": budgets,
        "permitted_providers": ["codex"],
        "permitted_efforts": ["high"],
        "max_concurrency": 1,
        "completion_policy": "evidence-all",
        "incident_policy": "recover-then-pause",
    }

    def validate(
        current_payload: dict[str, Any],
        *,
        objective: str = "bounded objective",
        constraints: list[Any] | None = None,
        predicates: list[Any] | None = None,
        milestones: list[Any] | None = None,
        permitted_providers: list[Any] | None = None,
        permitted_efforts: list[Any] | None = None,
        current_budgets: dict[str, Any] | None = None,
    ) -> None:
        if constraints is None:
            constraints = ["bounded"]
        if predicates is None:
            predicates = [{"type": "test"}]
        if milestones is None:
            milestones = [{"title": "one"}]
        if permitted_providers is None:
            permitted_providers = ["codex"]
        if permitted_efforts is None:
            permitted_efforts = ["high"]
        if current_budgets is None:
            current_budgets = budgets
        service_module._require_machines_goal_envelope(
            current_payload,
            objective=objective,
            constraints=constraints,
            predicates=predicates,
            milestones=milestones,
            permitted_providers=permitted_providers,
            permitted_efforts=permitted_efforts,
            budgets=current_budgets,
            supported_providers=frozenset({"claude", "codex", "kimi", "grok"}),
        )

    validate(payload)
    with pytest.raises(ValueError, match="complete typed goal envelope"):
        validate({}, objective="")
    with pytest.raises(ValueError, match="nonempty constraints"):
        validate(payload, constraints=[""])
    with pytest.raises(ValueError, match="evidence predicates"):
        validate(payload, predicates=[])
    with pytest.raises(ValueError, match="at least one goal milestone"):
        validate(payload, milestones=[])
    with pytest.raises(ValueError, match="typed goal milestones"):
        validate(payload, milestones=[{}])
    with pytest.raises(ValueError, match="explicit routing bounds"):
        validate(payload, permitted_providers=[])
    incomplete_budgets = dict(budgets)
    incomplete_budgets.pop("seconds")
    with pytest.raises(ValueError, match="all bounded budget dimensions"):
        validate(payload, current_budgets=incomplete_budgets)


def test_dispatch_transition_policy_rejects_malformed_stages(tmp_path: Path) -> None:
    created = _owned_session(tmp_path)
    next_ref = {"step_id": "step-one", "agent_role": "implementer"}
    command_digest = "a" * 64
    policy = {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": created.session_id,
        "external_ref": created.external_ref,
        "epoch_id": "epoch-one",
        "allowed_agent_roles": ["implementer"],
        "allowed_step_prefixes": ["step-"],
        "max_transitions": 1,
        "transitions": [
            {
                "sequence": 1,
                "next_turn_ref": next_ref,
                "next_command_digest": command_digest,
            }
        ],
    }
    policy_digest = normalized_digest(policy)
    goal = SimpleNamespace(
        constraints=(
            "dispatch-generation-transition-policy-sha256:" + policy_digest,
            "dispatch-generation-transition-epoch:epoch-one",
        )
    )
    store = GoalStoreProbe(goal)

    def validate(
        current_policy: dict[str, Any],
        *,
        current_ref: dict[str, str] | None = None,
        sequence: int = 1,
        current_digest: str = command_digest,
        retained: bool = False,
        current_store: object = store,
        digest: str = policy_digest,
    ) -> None:
        if current_ref is None:
            current_ref = next_ref
        service_module._require_dispatch_transition_policy(
            current_store,  # type: ignore[arg-type]
            session_id=created.session_id,
            session=created,
            policy=current_policy,
            policy_sha256=digest,
            next_turn_ref=current_ref,
            transition_sequence=sequence,
            next_command_digest=current_digest,
            retained_policy=retained,
        )

    validate(policy)

    cases = (
        ("schema", "other", "schema is unsupported"),
        ("session_id", "other", "session does not match"),
        ("external_ref", {}, "orchestrator does not match"),
        ("epoch_id", "", "epoch is invalid"),
        ("allowed_agent_roles", [], "roles are invalid"),
        ("allowed_step_prefixes", [], "step prefixes are invalid"),
        ("max_transitions", True, "limit is invalid"),
        ("transitions", [], "stages are invalid"),
    )
    for name, value, message in cases:
        changed = copy.deepcopy(policy)
        changed[name] = value
        with pytest.raises(ValueError, match=message):
            validate(changed)

    with pytest.raises(ValueError, match="role is outside policy"):
        validate(policy, current_ref={"step_id": "step-one", "agent_role": "reviewer"})
    with pytest.raises(ValueError, match="step is outside policy"):
        validate(
            policy,
            current_ref={"step_id": "other", "agent_role": "implementer"},
        )
    with pytest.raises(ValueError, match="exceeds policy limit"):
        validate(policy, sequence=2)

    changed = copy.deepcopy(policy)
    changed["max_transitions"] = 2
    with pytest.raises(ValueError, match="limit does not match stages"):
        validate(changed)
    changed = copy.deepcopy(policy)
    changed["transitions"][0] = []
    with pytest.raises(ValueError, match="stage is invalid"):
        validate(changed)
    changed = copy.deepcopy(policy)
    changed["transitions"][0]["sequence"] = 2
    with pytest.raises(ValueError, match="stages are not ordered"):
        validate(changed)

    two_stage = copy.deepcopy(policy)
    two_stage["max_transitions"] = 2
    two_stage["transitions"].append(
        {
            "sequence": 2,
            "next_turn_ref": {
                "step_id": "step-two",
                "agent_role": "implementer",
            },
            "next_command_digest": "b" * 64,
        }
    )
    invalid_second = copy.deepcopy(two_stage)
    invalid_second["transitions"][1] = []
    with pytest.raises(ValueError, match="stage is invalid"):
        validate(invalid_second, digest=normalized_digest(invalid_second))
    unordered = copy.deepcopy(two_stage)
    unordered["transitions"][1]["sequence"] = 3
    with pytest.raises(ValueError, match="stages are not ordered"):
        validate(unordered, digest=normalized_digest(unordered))
    wrong_role = copy.deepcopy(two_stage)
    wrong_role["transitions"][1]["next_turn_ref"]["agent_role"] = "reviewer"
    with pytest.raises(ValueError, match="stage role is invalid"):
        validate(wrong_role, digest=normalized_digest(wrong_role))
    wrong_step = copy.deepcopy(two_stage)
    wrong_step["transitions"][1]["next_turn_ref"]["step_id"] = "other"
    with pytest.raises(ValueError, match="stage step is invalid"):
        validate(wrong_step, digest=normalized_digest(wrong_step))
    wrong_digest = copy.deepcopy(two_stage)
    wrong_digest["transitions"][1]["next_command_digest"] = "bad"
    with pytest.raises(ValueError, match="stage digest is invalid"):
        validate(wrong_digest, digest=normalized_digest(wrong_digest))
    with pytest.raises(ValueError, match="stage is outside policy"):
        validate(
            policy,
            current_ref={"step_id": "step-two", "agent_role": "implementer"},
        )
    with pytest.raises(ValueError, match="command digest is invalid"):
        validate(policy, current_digest="b" * 64)
    with pytest.raises(ValueError, match="requires an active goal"):
        validate(policy, current_store=GoalStoreProbe(None))
    unauthorized = GoalStoreProbe(SimpleNamespace(constraints=()))
    with pytest.raises(ValueError, match="not goal-authorized"):
        validate(policy, current_store=unauthorized)
    missing_epoch = GoalStoreProbe(
        SimpleNamespace(
            constraints=(
                "dispatch-generation-transition-policy-sha256:" + policy_digest,
            )
        )
    )
    with pytest.raises(ValueError, match="epoch is not goal-authorized"):
        validate(policy, current_store=missing_epoch)


def test_dispatch_transition_validation_rejects_a_changed_receipt_shape(
    tmp_path: Path,
) -> None:
    created = _owned_session(tmp_path)
    goal_id = new_uuid()
    created = replace(created, goal_id=goal_id)
    next_ref = {"step_id": "step-one", "agent_role": "implementer"}
    prior_command_id = new_uuid()
    checkpoint_id = new_uuid()
    generation_digest = "a" * 64
    material_digest = "b" * 64
    command_digest = "c" * 64
    policy = {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": created.session_id,
        "external_ref": created.external_ref,
        "epoch_id": "epoch-one",
        "allowed_agent_roles": ["implementer"],
        "allowed_step_prefixes": ["step-"],
        "max_transitions": 1,
        "transitions": [
            {
                "sequence": 1,
                "next_turn_ref": next_ref,
                "next_command_digest": command_digest,
            }
        ],
    }
    policy_digest = normalized_digest(policy)
    receipt = {
        "source": "retained",
    }

    class ChangingAuthorization(dict[str, Any]):
        def __init__(self, values: dict[str, Any]) -> None:
            super().__init__(values)
            self.receipt_reads = 0

        def get(self, key: str, default: object = None) -> object:
            if key == "receipt":
                self.receipt_reads += 1
                if self.receipt_reads > 1:
                    return "changed"
            return super().get(key, default)

    authorization = ChangingAuthorization(
        {
            "schema": (
                "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
            ),
            "session_id": created.session_id,
            "reason": "advance",
            "receipt": receipt,
            "receipt_sha256": normalized_digest(receipt),
            "prior_command_id": prior_command_id,
            "next_turn_ref": next_ref,
            "goal_id": goal_id,
            "epoch_id": "epoch-one",
            "policy": policy,
            "policy_sha256": policy_digest,
            "transition_sequence": 1,
            "prior_command_type": "message",
            "prior_anchor_kind": "provider-result",
            "prior_reconciliation_id": "",
            "prior_reconciliation_resolution": "",
            "prior_checkpoint_id": checkpoint_id,
            "prior_generation_digest": generation_digest,
            "prior_material_digest": material_digest,
            "next_command_digest": command_digest,
            "external_orchestrator": created.external_ref["orchestrator"],
            "external_job_id": created.external_ref["job_id"],
        }
    )
    anchor = {
        "eligible": True,
        "prior_command_id": prior_command_id,
        "prior_command_type": "message",
        "prior_anchor_kind": "provider-result",
        "prior_reconciliation_id": "",
        "prior_reconciliation_resolution": "",
        "goal_id": goal_id,
        "epoch_id": "epoch-one",
        "prior_checkpoint_id": checkpoint_id,
        "prior_generation_digest": generation_digest,
        "prior_material_digest": material_digest,
    }
    store = SimpleNamespace(
        session_safety=lambda unused: {"profile": "unattended"},
        dispatch_invalidation_replay=lambda *unused: None,
        get_session=lambda unused: created,
        dispatch_transition_anchor=lambda unused: anchor,
        goal_for_session=lambda unused: SimpleNamespace(
            constraints=(
                "dispatch-generation-transition-policy-sha256:" + policy_digest,
                "dispatch-generation-transition-epoch:epoch-one",
            )
        ),
    )
    service = object.__new__(HarnessService)
    service.store = store  # type: ignore[assignment]
    payload = {
        "reason": "advance",
        "authorization": authorization,
        "prior_command_id": prior_command_id,
        "prior_command_type": "message",
        "prior_anchor_kind": "provider-result",
        "prior_reconciliation_id": "",
        "prior_reconciliation_resolution": "",
        "prior_checkpoint_id": checkpoint_id,
        "prior_generation_digest": generation_digest,
        "prior_material_digest": material_digest,
        "next_turn_ref": next_ref,
        "transition_sequence": 1,
        "next_command_digest": command_digest,
    }

    with pytest.raises(ValueError, match="receipt is invalid"):
        service.invalidate_dispatch_generation(
            created.session_id,
            payload,
            idempotency_key="changed-receipt",
        )


def test_dispatch_transition_policy_rejects_a_shrinking_stage_collection(
    tmp_path: Path,
) -> None:
    created = _owned_session(tmp_path)
    next_ref = {"step_id": "step-two", "agent_role": "implementer"}
    command_digest = "b" * 64

    class ShrinkingStages(list[dict[str, Any]]):
        def __init__(self, values: list[dict[str, Any]]) -> None:
            super().__init__(values)
            self.length_reads = 0

        def __len__(self) -> int:
            self.length_reads += 1
            if self.length_reads <= 2:
                return 2
            return 1

    stages = ShrinkingStages(
        [
            {
                "sequence": 1,
                "next_turn_ref": {
                    "step_id": "step-one",
                    "agent_role": "implementer",
                },
                "next_command_digest": "a" * 64,
            },
            {
                "sequence": 2,
                "next_turn_ref": next_ref,
                "next_command_digest": command_digest,
            },
        ]
    )
    policy = {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": created.session_id,
        "external_ref": created.external_ref,
        "epoch_id": "epoch-one",
        "allowed_agent_roles": ["implementer"],
        "allowed_step_prefixes": ["step-"],
        "max_transitions": 2,
        "transitions": stages,
    }

    with pytest.raises(ValueError, match="sequence is outside policy"):
        service_module._require_dispatch_transition_policy(
            GoalStoreProbe(SimpleNamespace(constraints=())),  # type: ignore[arg-type]
            session_id=created.session_id,
            session=created,
            policy=policy,
            policy_sha256="d" * 64,
            next_turn_ref=next_ref,
            transition_sequence=2,
            next_command_digest=command_digest,
            retained_policy=True,
        )


def test_worker_status_and_supervision_failure_are_defensive(tmp_path: Path) -> None:
    manager = service_module.WorkerManager(paths(tmp_path / "state"))
    status = manager.status()
    status["status"] = "mutated"
    assert manager.status()["status"] == "initializing"

    service = object.__new__(HarnessService)
    service._worker_supervision = {
        "status": "ok",
        "expected_sessions": "malformed",
    }
    service.record_worker_supervision_failure(RuntimeError("x" * 700))
    assert service._worker_supervision["expected_sessions"] == []
    failure = service._worker_supervision["unrecovered"][0]
    assert failure["reason"] == "worker-supervision-failed"
    assert len(failure["detail"]) == 500
    with pytest.raises(ValueError, match="must be finite"):
        service_module._optional_number(float("inf"))


def test_submit_message_requeues_a_retryable_failed_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    owned = session(tmp_path)
    service.store.create_session(owned)
    monkeypatch.setattr(
        service_module,
        "require_state_headroom",
        lambda unused_path, unused_provider: 10 * 1024**3,
    )
    payload = {"text": "Review the cs-builder change.", "provider": "grok"}
    first = service.submit_message(
        owned.session_id,
        payload,
        "cs-builder-review",
    )
    service.store.resolve_command(
        str(first["command_id"]),
        CommandStatus.FAILED,
        {
            "code": "E_PROVIDER_UNAVAILABLE",
            "message": "grok is unavailable",
            "retryable": True,
        },
    )

    replayed = service.submit_message(
        owned.session_id,
        payload,
        "cs-builder-review",
    )

    assert first["status"] == CommandStatus.QUEUED
    assert replayed["command_id"] == first["command_id"]
    assert replayed["status"] == CommandStatus.QUEUED
    assert replayed["result"] == {}
    instructions = [
        event
        for event in service.store.events(owned.session_id, limit=5000)
        if event.event_type == "user.message"
    ]
    assert len(instructions) == 1
    assert service.workers.sessions == [  # type: ignore[attr-defined]
        owned.session_id,
        owned.session_id,
    ]
    service.close()


def test_service_commands_fail_closed_on_terminal_sessions(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    stopped = session(tmp_path)
    service.store.create_session(stopped)
    service.store.stop_session(stopped.session_id)
    probe = service.workers

    for command_type, payload in (
        ("interrupt", {}),
        ("pause", {}),
        ("stop", {}),
        ("steer", {"text": "keep going"}),
    ):
        with pytest.raises(ConflictError, match="admits only a resume command"):
            service.command(
                stopped.session_id,
                command_type,
                payload,
                "stopped-" + command_type,
            )
    assert probe.sessions == []  # type: ignore[attr-defined]
    assert service.store.claim_command(stopped.session_id) is None
    assert not service.store.queued_command_exists(
        stopped.session_id,
        storage_module.STOPPED_SESSION_COMMANDS,
    )

    receipt = service.command(
        stopped.session_id,
        "resume",
        {},
        "stopped-resume",
    )
    assert receipt["status"] == CommandStatus.QUEUED
    replayed = service.command(
        stopped.session_id,
        "resume",
        {},
        "stopped-resume",
    )
    assert replayed["command_id"] == receipt["command_id"]
    assert probe.sessions == [  # type: ignore[attr-defined]
        stopped.session_id,
        stopped.session_id,
    ]
    assert probe.forced == [True, True]  # type: ignore[attr-defined]

    assert service._recoverable_worker_sessions() == [stopped.session_id]
    claimed = service.store.claim_command(
        stopped.session_id,
        storage_module.STOPPED_SESSION_COMMANDS,
    )
    assert claimed is not None
    assert claimed.command_id == receipt["command_id"]
    assert service._recoverable_worker_sessions() == []

    for lifecycle in ("completed", "failed"):
        service.store.update_session(stopped.session_id, lifecycle=lifecycle)
        with pytest.raises(ConflictError, match="admits no commands"):
            service.command(
                stopped.session_id,
                "resume",
                {},
                lifecycle + "-resume",
            )
        assert service._recoverable_worker_sessions() == []
    service.close()


def test_service_forces_a_worker_past_a_retiring_stopped_session_worker(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    stopped = session(tmp_path)
    service.store.create_session(stopped)
    service.store.stop_session(stopped.session_id)
    probe = service.workers

    service.command(stopped.session_id, "resume", {}, "resume-while-retiring")
    assert probe.forced == [True]  # type: ignore[attr-defined]

    service.store.register_worker(stopped.session_id, 4321, "incarnation-one")
    service.command(stopped.session_id, "resume", {}, "resume-with-a-worker")
    assert probe.forced == [True, False]  # type: ignore[attr-defined]

    running = session(tmp_path)
    service.store.create_session(running)
    service.command(running.session_id, "pause", {}, "pause-a-running-session")
    assert probe.forced == [True, False, False]  # type: ignore[attr-defined]
    service.close()


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
