"""Focused regressions for adversarial durable-boundary findings."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agent_harness.api as api_module
import agent_harness.proof as proof_module
import agent_harness.worker as worker_module
from agent_harness.blobs import BlobStore
from agent_harness.config import paths
from agent_harness.errors import (
    ConflictError,
    HarnessError,
    NotFoundError,
    SafetyGuardError,
)
from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import Checkpoint, CommandStatus, ProviderAttempt
from agent_harness.process_control import ProcessGroupIdentity
from agent_harness.providers import claude, codex, kimi
from agent_harness.providers.base import ChildLaunchGate, ProviderEvent
from agent_harness.safety import (
    SafetyConsumption,
    TurnGuard,
    effective_effort,
    effort_requires_xhigh_authorization,
    limits_for,
    lower_effort,
)
from agent_harness.service import HarnessService
from agent_harness.storage import StateStore
from agent_harness.worker import SessionWorker
from tests.test_support import session


def _store(tmp_path: Path) -> tuple[StateStore, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(workspace)
    store.create_session(created)
    return store, created


def _dispatch(
    store: StateStore,
    created: Any,
    command_id: str,
) -> tuple[ProviderAttempt, str, Checkpoint]:
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="codex",
        native_session_id="",
        model="default",
        effort="high",
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
        command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    return attempt, turn_id, checkpoint


def test_terminal_dispatch_is_reconciled_and_success_finalizes_atomically(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    limits = limits_for("unattended", "implementation")
    crashed = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "perform one effect"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    store.create_command_envelope(
        crashed.command_id,
        created.session_id,
        "unattended",
        limits.as_dict(),
    )
    attempt, unused_turn_id, unused_checkpoint = _dispatch(
        store,
        created,
        crashed.command_id,
    )
    del unused_turn_id, unused_checkpoint
    store.mark_provider_boundary(attempt.attempt_id)
    store.update_attempt(attempt.attempt_id, status="complete")
    store.complete_dispatch(attempt.attempt_id, "complete")

    recovery = store.recover_interrupted_commands(
        created.session_id,
        "observed-material",
        "observed summary",
    )
    assert recovery.requeued_command_ids == ()
    assert len(recovery.reconciliations) == 1
    assert store.get_command(crashed.command_id).result["code"] == (
        "E_NEEDS_RECONCILIATION"
    )
    store.close()

    completed_store, completed_session = _store(tmp_path / "completed")
    completed = completed_store.enqueue_command(
        completed_session.session_id,
        "message",
        {"text": "complete one effect"},
        new_uuid(),
    )
    assert completed_store.claim_command(completed_session.session_id) is not None
    completed_store.create_command_envelope(
        completed.command_id,
        completed_session.session_id,
        "unattended",
        limits.as_dict(),
    )
    completed_attempt, turn_id, checkpoint = _dispatch(
        completed_store,
        completed_session,
        completed.command_id,
    )
    completed_store.mark_provider_boundary(completed_attempt.attempt_id)
    result = {
        "turn_id": turn_id,
        "native_session_id": "native-one",
        "checkpoint_id": checkpoint.checkpoint_id,
        "status": "complete",
    }
    with pytest.raises(ConflictError, match="dispatch boundary is missing"):
        completed_store.complete_command_execution(
            completed.command_id,
            new_uuid(),
            "native-one",
            SafetyConsumption().as_dict(),
            result,
        )
    with pytest.raises(ConflictError, match="checkpoint is missing"):
        completed_store.complete_command_execution(
            completed.command_id,
            turn_id,
            "native-one",
            SafetyConsumption().as_dict(),
            {**result, "checkpoint_id": new_uuid()},
        )
    receipt = completed_store.complete_command_execution(
        completed.command_id,
        turn_id,
        "native-one",
        SafetyConsumption(attempts=1).as_dict(),
        result,
    )
    assert receipt.status == "complete"
    assert completed_store.command_envelope(completed.command_id)["state"] == "complete"
    assert completed_store.attempts(completed_session.session_id)[-1].status == (
        "complete"
    )
    with completed_store.transaction() as connection:
        topology = connection.execute(
            """
            SELECT turns.status AS turn_status, command_dispatches.state
            FROM turns JOIN command_dispatches USING(turn_id)
            WHERE turns.turn_id = ?
            """,
            (turn_id,),
        ).fetchone()
    assert dict(topology) == {"turn_status": "complete", "state": "complete"}
    with pytest.raises(ConflictError, match="not dispatching"):
        completed_store.complete_command_execution(
            completed.command_id,
            turn_id,
            "native-one",
            SafetyConsumption(attempts=1).as_dict(),
            result,
        )
    completed_store.close()


def test_adapters_publish_the_canonical_process_group_identity() -> None:
    identity = ProcessGroupIdentity(pid=4242, pgid=4242, pid_start="ps-start")

    codex_adapter = codex.CodexAdapter()
    codex_adapter._active_server = SimpleNamespace(
        process=SimpleNamespace(pid=4242, returncode=None),
        _process_group=identity,
    )
    assert codex_adapter.process_identity() == (4242, "ps-start")
    codex_adapter._active_server._process_group = ProcessGroupIdentity(
        pid=4343,
        pgid=4343,
        pid_start="other-start",
    )
    assert codex_adapter.process_identity() == (0, "")

    claude_adapter = claude.ClaudeAdapter()
    claude_adapter._transport = SimpleNamespace(
        _process=SimpleNamespace(pid=4242, returncode=None),
        _process_group=identity,
    )
    assert claude_adapter.process_identity() == (4242, "ps-start")
    claude_adapter._transport._process_group = None
    assert claude_adapter.process_identity() == (0, "")


def test_default_service_registry_excludes_kimi_until_it_is_safety_mapped(
    tmp_path: Path,
) -> None:
    service = HarnessService(paths(tmp_path / "state"))
    try:
        assert set(service.adapters) == {"claude", "codex"}
        assert not kimi.KimiAdapter().status().ready
    finally:
        service.close()


def test_max_effort_uses_exact_single_use_xhigh_authorization(
    tmp_path: Path,
) -> None:
    limits = limits_for("unattended", "implementation")
    assert effort_requires_xhigh_authorization("xhigh")
    assert effort_requires_xhigh_authorization("max")
    assert not effort_requires_xhigh_authorization("high")
    assert not effort_requires_xhigh_authorization("unsupported")
    assert lower_effort("max") == "xhigh"
    with pytest.raises(ValueError, match="explicit unattended authorization"):
        effective_effort("max", limits, xhigh_authorized=False)
    assert effective_effort("max", limits, xhigh_authorized=True) == "max"

    store, created = _store(tmp_path)
    payload = {
        "text": "maximum effort",
        "provider": "claude",
        "effort": " MAX ",
    }
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
        {**limits.as_dict(), "max_attempts": 1},
    )
    store.register_worker(created.session_id, 123, "worker-one")
    authorization = store.create_xhigh_authorization(
        created.session_id,
        command.command_id,
        "claude",
        authorization_request_digest="a" * 64,
        idempotency_key=new_uuid(),
        expires_at="2099-01-01T00:00:00+00:00",
    )
    assert authorization["effort"] == "max"
    attempt, unused_turn_id, unused_checkpoint = _dispatch(
        store,
        created,
        command.command_id,
    )
    del unused_turn_id, unused_checkpoint
    admission = store.reserve_route_admission(
        command.command_id,
        "claude",
        "unattended",
        effort="max",
        attempt_id=attempt.attempt_id,
        worker_incarnation="worker-one",
        goal_id="",
        max_concurrency=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    assert admission["admitted"] is True
    assert store.xhigh_authorization(command.command_id) is None
    store.close()


class _FingerprintStore:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def reserve_dispatch_generation_transition(self, *unused: object) -> str:
        del unused
        return ""

    def repetition_generation(self, unused_session_id: str) -> dict[str, str]:
        del unused_session_id
        return {"checkpoint_id": "", "invalidation_id": ""}

    def all_events(self, unused_session_id: str) -> list[Any]:
        del unused_session_id
        return list(self.events)

    def command_failed_before_provider_boundary(self, unused_command_id: str) -> bool:
        del unused_command_id
        return False

    def append_event(
        self,
        unused_session_id: str,
        event_type: str,
        *,
        metadata: dict[str, Any],
        **unused: object,
    ) -> None:
        del unused_session_id, unused
        self.events.append(SimpleNamespace(event_type=event_type, metadata=metadata))


def test_distinct_managed_steps_may_reuse_every_governing_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FingerprintStore()
    worker = SessionWorker(
        store,  # type: ignore[arg-type]
        BlobStore(tmp_path / "blobs"),
        None,  # type: ignore[arg-type]
        {},
        "session-one",
    )
    monkeypatch.setattr(
        worker_module,
        "inspect_workspace",
        lambda unused_workspace: ("material", "summary"),
    )
    first_ref = {"step_id": "one", "agent_role": "builder"}
    second_ref = {"step_id": "two", "agent_role": "builder"}
    worker._guard_repeated_dispatch(
        "command-one",
        "same instruction",
        "same context",
        tmp_path,
        first_ref,
        "unattended",
    )
    worker._guard_repeated_dispatch(
        "command-changed",
        "changed instruction",
        "changed context",
        tmp_path,
        first_ref,
        "unattended",
    )
    worker._guard_repeated_dispatch(
        "command-two",
        "same instruction",
        "same context",
        tmp_path,
        second_ref,
        "unattended",
    )
    with pytest.raises(SafetyGuardError, match="repeated-instruction"):
        worker._guard_repeated_dispatch(
            "command-three",
            "same instruction",
            "same context",
            tmp_path,
            first_ref,
            "unattended",
        )
    worker._guard_repeated_dispatch(
        "command-with-empty-components",
        "new instruction",
        "new context",
        tmp_path,
        {},
        "unattended",
    )


@pytest.mark.asyncio
async def test_kimi_accepts_neutral_hooks_and_uses_process_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = kimi.KimiAdapter()
    observed: list[tuple[int, str]] = []
    launch_options: dict[str, Any] = {}
    termination_requests: list[int] = []

    class EmptyStdout:
        def __aiter__(self):
            async def lines():
                if False:
                    yield b""

            return lines()

    class EmptyStderr:
        async def read(self) -> bytes:
            return b""

    class Process:
        pid = 42
        returncode: int | None = None
        stdout = EmptyStdout()
        stderr = EmptyStderr()

        async def wait(self) -> int:
            observed.append(adapter.process_identity())
            await adapter.interrupt()
            self.returncode = 0
            observed.append(adapter.process_identity())
            return 0

        def terminate(self) -> None:
            termination_requests.append(self.pid)
            self.returncode = -15

    async def create_process(*unused: object, **options: Any) -> Process:
        del unused
        launch_options.update(options)
        return Process()

    monkeypatch.setattr(kimi.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(
        kimi,
        "process_group_identity",
        lambda unused_pid: ProcessGroupIdentity(42, 42, "canonical-start"),
    )
    interruptions: list[tuple[int, str]] = []

    async def terminate_process(
        unused_process: object,
        identity: ProcessGroupIdentity,
    ) -> None:
        del unused_process
        interruptions.append((identity.pid, identity.pid_start))

    monkeypatch.setattr(kimi, "terminate_process_group", terminate_process)
    gate_calls: list[str] = []

    async def gate() -> None:
        gate_calls.append("opened")

    async def event_handler(unused_event: ProviderEvent) -> None:
        del unused_event

    async def approval_handler(
        unused_method: str,
        unused_request: dict[str, Any],
    ) -> dict[str, Any]:
        del unused_method, unused_request
        return {}

    result = await adapter.run_turn(
        workspace=tmp_path,
        prompt="one prompt",
        native_session_id="",
        permission_mode="full",
        model="kimi-code/k3",
        effort="high",
        event_handler=event_handler,
        approval_handler=approval_handler,
        pre_prompt_gate=gate,
    )
    assert result.status == "complete"
    assert gate_calls == ["opened"]
    assert launch_options["start_new_session"] is True
    assert observed == [(42, "canonical-start"), (0, "")]
    assert interruptions == [(42, "canonical-start")]
    assert adapter.process_identity() == (0, "")
    await adapter.interrupt()

    adapter._active_process = SimpleNamespace(pid=7, returncode=None)
    adapter._process_group = ProcessGroupIdentity(42, 42, "canonical-start")
    assert adapter.process_identity() == (0, "")
    adapter._active_process = None
    adapter._process_group = None

    with pytest.raises(RuntimeError, match="permission mode"):
        await adapter.run_turn(
            workspace=tmp_path,
            prompt="one prompt",
            native_session_id="",
            permission_mode="read-only",
            model="kimi-code/k3",
            effort="high",
            event_handler=event_handler,
            approval_handler=approval_handler,
        )
    with pytest.raises(RuntimeError, match="child-agent limit"):
        await adapter.run_turn(
            workspace=tmp_path,
            prompt="one prompt",
            native_session_id="",
            permission_mode="full",
            model="kimi-code/k3",
            effort="high",
            event_handler=event_handler,
            approval_handler=approval_handler,
            child_launch_gate=ChildLaunchGate(
                tmp_path / "state.sqlite3",
                "command",
                0,
            ),
        )

    def unavailable_identity(unused_pid: int) -> ProcessGroupIdentity:
        del unused_pid
        raise RuntimeError("identity unavailable")

    monkeypatch.setattr(kimi, "process_group_identity", unavailable_identity)
    with pytest.raises(RuntimeError, match="identity unavailable"):
        await adapter.run_turn(
            workspace=tmp_path,
            prompt="one prompt",
            native_session_id="",
            permission_mode="full",
            model="kimi-code/k3",
            effort="high",
            event_handler=event_handler,
            approval_handler=approval_handler,
        )
    assert termination_requests == [42]


@pytest.mark.asyncio
async def test_retry_hydrates_persisted_consumption_and_keeps_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, created = _store(tmp_path)
    store.set_session_safety(created.session_id, "unattended")
    store.extend_session_safety(
        created.session_id,
        {"reason": "retained", "additional_seconds": 60},
    )
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "retry safely", "effort": "high"},
        new_uuid(),
    )
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    limits = limits_for("unattended", "implementation")
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        limits.as_dict(),
    )
    persisted = SafetyConsumption(
        context_tokens=17,
        total_tokens=17,
        attempts=1,
        elapsed_seconds=9.0,
    )
    store.update_command_envelope(
        command.command_id,
        consumption=persisted.as_dict(),
    )
    with pytest.raises(ConflictError, match="limits changed"):
        store.create_command_envelope(
            command.command_id,
            created.session_id,
            "unattended",
            {**limits.as_dict(), "max_attempts": 1},
        )
    with pytest.raises(ConflictError, match="profile changed"):
        store.create_command_envelope(
            command.command_id,
            created.session_id,
            "interactive",
            limits.as_dict(),
        )
    other = session(tmp_path / "other-workspace")
    Path(other.worktree).mkdir()
    store.create_session(other)
    with pytest.raises(ConflictError, match="session changed"):
        store.create_command_envelope(
            command.command_id,
            other.session_id,
            "unattended",
            limits.as_dict(),
        )
    store.set_session_safety(created.session_id, "interactive")

    worker = SessionWorker(
        store,
        BlobStore(tmp_path / "blobs"),
        None,  # type: ignore[arg-type]
        {},
        created.session_id,
    )
    captured: list[SafetyConsumption] = []

    async def stop_after_hydration(
        unused_command_id: str,
        unused_payload: dict[str, Any],
        unused_text: str,
        guard: TurnGuard,
    ) -> dict[str, Any]:
        del unused_command_id, unused_payload, unused_text
        captured.append(guard.consumption)
        raise SafetyGuardError("persisted-stop", "automatic routing", recoverable=False)

    monkeypatch.setattr(worker, "_execute_with_failover", stop_after_hydration)
    monkeypatch.setattr(worker_module, "require_state_headroom", lambda *unused: 1)
    await worker._execute_message(claimed)
    assert captured[0].attempts == 1
    assert captured[0].context_tokens == 17
    assert captured[0].elapsed_seconds >= 9.0
    assert store.command_envelope(command.command_id)["profile"] == "unattended"
    assert store.session_safety(created.session_id)["extensions"] == {
        "reason": "retained",
        "additional_seconds": 60,
    }
    store.close()


def test_cross_provider_context_excludes_the_current_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, created = _store(tmp_path)
    current_text = "Perform the current action exactly once."
    store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="Earlier instruction.",
        status="accepted",
        metadata={"command_id": "earlier"},
    )
    command_id = new_uuid()
    store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text=current_text,
        status="accepted",
        metadata={"command_id": command_id},
    )
    store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="Future queued instruction.",
        status="accepted",
        metadata={"command_id": "future"},
    )
    assert len(store.context_events(created.session_id)) == 3
    with pytest.raises(NotFoundError, match="instruction event"):
        store.command_instruction_sequence(created.session_id, "missing")
    worker = SessionWorker(
        store,
        BlobStore(tmp_path / "blobs"),
        None,  # type: ignore[arg-type]
        {},
        created.session_id,
    )
    monkeypatch.setattr(worker_module, "workspace_summary", lambda unused: "")
    context = worker._compile_context(
        created,
        limits_for("interactive", "implementation"),
        command_id,
    )
    prompt = context.text + "\n\n# Next instruction\n\n" + current_text
    assert "Earlier instruction." in prompt
    assert "Future queued instruction." not in prompt
    assert prompt.count(current_text) == 1
    store.close()


def test_known_undelivered_terminal_dispatch_requeues_after_restart(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    limits = limits_for("unattended", "implementation")
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "retry only when not delivered"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        limits.as_dict(),
    )
    attempt, unused_turn_id, checkpoint = _dispatch(
        store,
        created,
        command.command_id,
    )
    del unused_turn_id
    store.prepare_context_delivery(
        created.session_id,
        "claude",
        "context-digest",
        checkpoint.checkpoint_id,
        command.command_id,
        attempt.attempt_id,
        "payload-digest",
    )
    store.mark_provider_boundary(attempt.attempt_id)
    store.update_attempt(attempt.attempt_id, status="failed")
    store.complete_dispatch(attempt.attempt_id, "failed")

    recovery = store.recover_interrupted_commands(
        created.session_id,
        "unchanged-material",
        "unchanged summary",
    )

    assert recovery.requeued_command_ids == (command.command_id,)
    assert recovery.reconciliations == ()
    assert store.get_command(command.command_id).status == CommandStatus.QUEUED
    next_attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="claude",
        native_session_id="",
        model="",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    store.create_attempt(next_attempt)
    prepared = store.prepare_context_delivery(
        created.session_id,
        "claude",
        "next-context-digest",
        checkpoint.checkpoint_id,
        command.command_id,
        next_attempt.attempt_id,
        "next-payload-digest",
    )
    assert prepared["attempt_id"] == next_attempt.attempt_id
    store.close()


def test_delivery_evidence_distinguishes_retry_from_reconciliation(
    tmp_path: Path,
) -> None:
    limits = limits_for("unattended", "implementation")
    no_delivery, no_delivery_session = _store(tmp_path / "no-delivery")
    command = no_delivery.enqueue_command(
        no_delivery_session.session_id,
        "message",
        {"text": "fail before delivery"},
        new_uuid(),
    )
    assert no_delivery.claim_command(no_delivery_session.session_id) is not None
    no_delivery.create_command_envelope(
        command.command_id,
        no_delivery_session.session_id,
        "unattended",
        limits.as_dict(),
    )
    attempt, unused_turn_id, unused_checkpoint = _dispatch(
        no_delivery,
        no_delivery_session,
        command.command_id,
    )
    del unused_turn_id, unused_checkpoint
    no_delivery.mark_provider_boundary(attempt.attempt_id)
    no_delivery.update_attempt(attempt.attempt_id, status="failed")
    no_delivery.complete_dispatch(attempt.attempt_id, "failed")
    retry = no_delivery.recover_interrupted_commands(
        no_delivery_session.session_id,
        "material",
        "summary",
    )
    assert retry.requeued_command_ids == (command.command_id,)
    no_delivery.close()

    delivered, delivered_session = _store(tmp_path / "delivered")
    delivered_command = delivered.enqueue_command(
        delivered_session.session_id,
        "message",
        {"text": "fail after delivery"},
        new_uuid(),
    )
    assert delivered.claim_command(delivered_session.session_id) is not None
    delivered.create_command_envelope(
        delivered_command.command_id,
        delivered_session.session_id,
        "unattended",
        limits.as_dict(),
    )
    delivered_attempt, unused_turn_id, checkpoint = _dispatch(
        delivered,
        delivered_session,
        delivered_command.command_id,
    )
    del unused_turn_id
    delivered.prepare_context_delivery(
        delivered_session.session_id,
        "codex",
        "accepted-context",
        checkpoint.checkpoint_id,
        delivered_command.command_id,
        delivered_attempt.attempt_id,
        "accepted-payload",
    )
    delivered.accept_context_delivery(
        delivered_session.session_id,
        "codex",
        "accepted-context",
        delivered_attempt.attempt_id,
    )
    delivered.mark_provider_boundary(delivered_attempt.attempt_id)
    delivered.update_attempt(delivered_attempt.attempt_id, status="failed")
    delivered.complete_dispatch(delivered_attempt.attempt_id, "failed")
    reconciliation = delivered.recover_interrupted_commands(
        delivered_session.session_id,
        "material",
        "summary",
    )
    assert reconciliation.requeued_command_ids == ()
    assert len(reconciliation.reconciliations) == 1
    delivered.close()


def test_persisted_proof_payload_must_match_its_retained_digest(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    payload = {"events": [], "session": {"session_id": created.session_id}}
    retained = store.create_proof_snapshot(
        created.session_id,
        0,
        payload,
        proof_module._digest(payload),
    )
    with store.transaction() as connection:
        connection.execute(
            "UPDATE proof_snapshots SET payload_json = ? WHERE snapshot_id = ?",
            (
                json.dumps(
                    {"events": [], "session": {"session_id": "tampered"}},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                retained["snapshot_id"],
            ),
        )
    with pytest.raises(HarnessError) as raised:
        proof_module.proof_snapshot(
            store,
            created.session_id,
            snapshot_id=str(retained["snapshot_id"]),
        )
    assert raised.value.detail.code == "E_PROOF_INTEGRITY"
    assert raised.value.detail.status == 500
    with pytest.raises(HarnessError, match="integrity check"):
        proof_module._proof_page(
            {"payload": {"events": []}, "through_sequence": 0},
            after_sequence=0,
            event_limit=1,
        )
    store.close()


@pytest.mark.asyncio
async def test_health_remains_live_while_ready_is_contractually_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervision = {"status": "failed", "unrecovered": [{"type": "RuntimeError"}]}
    service = SimpleNamespace(
        paths=SimpleNamespace(state_dir=tmp_path),
        worker_supervision=lambda: supervision,
        quiescence=lambda: {"status": "blocked"},
    )
    monkeypatch.setattr(api_module, "_service", lambda unused_request: service)
    monkeypatch.setattr(api_module, "read_sync_status", lambda unused_paths: {})
    response = await api_module._health(SimpleNamespace())
    body = json.loads(response.text)
    assert response.status == 200
    assert body["status"] == "ok"
    assert body["worker_supervision"] == supervision

    contract = (
        Path(__file__).resolve().parents[1] / "contracts" / "openapi.gpt.yaml"
    ).read_text(encoding="utf-8")
    health_contract = contract.split("  /healthz:", 1)[1].split("  /readyz:", 1)[0]
    ready_contract = contract.split("  /readyz:", 1)[1].split(
        "  /v1/capabilities:",
        1,
    )[0]
    assert '"503"' not in health_contract
    assert '"503"' in ready_contract


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
