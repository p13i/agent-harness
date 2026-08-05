"""Focused regressions for adversarial durable-boundary findings."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agent_harness.api as api_module
import agent_harness.proof as proof_module
import agent_harness.storage as storage_module
import agent_harness.worker as worker_module
from agent_harness.blobs import BlobStore
from agent_harness.config import paths
from agent_harness.errors import (
    ConflictError,
    HarnessError,
    NotFoundError,
    SafetyGuardError,
)
from agent_harness.goals import make_evidence
from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import Checkpoint, CommandStatus, ProviderAttempt
from agent_harness.orchestration import (
    command_envelope_digest,
    legacy_command_envelope_digest,
)
from agent_harness.process_control import ProcessGroupIdentity
from agent_harness.providers import claude, codex, kimi
from agent_harness.providers.base import ChildLaunchGate, ProviderEvent
from agent_harness.safety import (
    SafetyConsumption,
    TurnGuard,
    effective_effort,
    effort_requires_xhigh_authorization,
    limits_for,
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
    with pytest.raises(ConflictError, match="evidence goal is not current"):
        completed_store.complete_command_execution(
            completed.command_id,
            turn_id,
            "native-one",
            SafetyConsumption().as_dict(),
            result,
            goal_evidence=(
                make_evidence(
                    new_uuid(),
                    "command",
                    "make test",
                    "failed",
                ),
            ),
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


def test_atomic_completion_fails_closed_when_session_row_is_missing(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "complete one effect"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        limits_for("unattended", "implementation").as_dict(),
    )
    attempt, turn_id, checkpoint = _dispatch(store, created, command.command_id)
    store.mark_provider_boundary(attempt.attempt_id)
    corruptor = sqlite3.connect(store.path, isolation_level=None)
    corruptor.execute("PRAGMA foreign_keys=OFF")
    corruptor.execute(
        "DELETE FROM sessions WHERE session_id = ?",
        (created.session_id,),
    )

    with pytest.raises(NotFoundError, match="session"):
        store.complete_command_execution(
            command.command_id,
            turn_id,
            "native-one",
            SafetyConsumption().as_dict(),
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "status": "complete",
            },
        )

    corruptor.close()
    store.close()


def test_sqlite_begin_retries_only_bounded_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = StateStore(database)
    holder = sqlite3.connect(database, isolation_level=None)
    store._connection.execute("PRAGMA busy_timeout=1")
    holder.execute("BEGIN IMMEDIATE")
    delays: list[float] = []

    def release_holder(delay: float) -> None:
        delays.append(delay)
        holder.execute("COMMIT")

    with monkeypatch.context() as context:
        context.setattr(storage_module.time, "sleep", release_holder)
        with store.transaction() as connection:
            connection.execute("SELECT 1")
    assert delays == [storage_module.SQLITE_BEGIN_BACKOFF_SECONDS[0]]

    holder.execute("BEGIN IMMEDIATE")
    delays.clear()
    with monkeypatch.context() as context:
        context.setattr(
            storage_module.time,
            "sleep",
            lambda delay: delays.append(delay),
        )
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with store.transaction():
                pass
    assert delays == list(storage_module.SQLITE_BEGIN_BACKOFF_SECONDS)
    holder.execute("ROLLBACK")

    calls = 0

    def arbitrary_failure() -> None:
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as context:
        context.setattr(store, "_begin_immediate_once", arbitrary_failure)
        with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
            with store.transaction():
                pass
    assert calls == 1
    assert not storage_module._sqlite_contention(
        sqlite3.OperationalError("database is locked")
    )
    holder.close()
    store.close()


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


def test_default_service_registry_admits_kimi_with_config_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kimi.shutil, "which", lambda _name: "/usr/bin/npx")
    service = HarnessService(paths(tmp_path / "state"))
    try:
        assert set(service.adapters) == {"claude", "codex", "kimi"}
        assert service.adapters["kimi"].status().ready
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
                2,
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
        unused_evaluator: Any = None,
    ) -> dict[str, Any]:
        del unused_command_id, unused_payload, unused_text, unused_evaluator
        captured.append(guard.consumption)
        raise SafetyGuardError("persisted-stop", "automatic routing")

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
        "agent.message",
        role="assistant",
        text="The first attempt changed one bounded file.",
        status="complete",
        metadata={"command_id": command_id},
    )
    store.append_event(
        created.session_id,
        "user.steer",
        role="user",
        text="Preserve that file during failover.",
        status="accepted",
        metadata={"target_command_id": command_id},
    )
    store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="Future queued instruction.",
        status="accepted",
        metadata={"command_id": "future"},
    )
    assert len(store.events(created.session_id, limit=5000)) == 5
    instruction_sequence = store.command_instruction_sequence(
        created.session_id,
        command_id,
    )
    assert [
        event.text
        for event in store.events(created.session_id, limit=5000)
        if event.sequence < instruction_sequence + 1
    ] == ["Earlier instruction.", current_text]
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
    assert "The first attempt changed one bounded file." in prompt
    assert "Preserve that file during failover." in prompt
    assert "Future queued instruction." not in prompt
    assert prompt.count(current_text) == 1
    store.close()


@pytest.mark.parametrize("terminal_state", ["failed", "exhausted", "interrupted"])
def test_known_undelivered_terminal_dispatch_requeues_after_restart(
    tmp_path: Path,
    terminal_state: str,
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
    store.update_attempt(attempt.attempt_id, status=terminal_state)
    store.complete_dispatch(attempt.attempt_id, terminal_state)

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
    assert retry.requeued_command_ids == ()
    assert len(retry.reconciliations) == 1
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


def test_recovery_retains_every_superseded_undelivered_dispatch(
    tmp_path: Path,
) -> None:
    limits = limits_for("unattended", "implementation")
    store, created = _store(tmp_path)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "do not repeat an ambiguous earlier attempt"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        limits.as_dict(),
    )
    first, unused_turn_id, first_checkpoint = _dispatch(
        store,
        created,
        command.command_id,
    )
    del unused_turn_id
    store.prepare_context_delivery(
        created.session_id,
        "codex",
        "first-context",
        first_checkpoint.checkpoint_id,
        command.command_id,
        first.attempt_id,
        "first-payload",
    )
    store.mark_provider_boundary(first.attempt_id)
    store.update_attempt(first.attempt_id, status="failed")
    store.complete_dispatch(first.attempt_id, "failed")

    second, unused_turn_id, checkpoint = _dispatch(
        store,
        created,
        command.command_id,
    )
    del unused_turn_id
    store.prepare_context_delivery(
        created.session_id,
        "codex",
        "second-context",
        checkpoint.checkpoint_id,
        command.command_id,
        second.attempt_id,
        "second-payload",
    )
    store.mark_provider_boundary(second.attempt_id)
    store.update_attempt(second.attempt_id, status="failed")
    store.complete_dispatch(second.attempt_id, "failed")

    recovery = store.recover_interrupted_commands(
        created.session_id,
        "material",
        "summary",
    )
    assert recovery.requeued_command_ids == (command.command_id,)
    assert recovery.reconciliations == ()
    deliveries = store.portable_session(created.session_id)["tables"][
        "context_deliveries"
    ]
    assert [item["attempt_id"] for item in deliveries] == [
        first.attempt_id,
        second.attempt_id,
    ]
    assert [item["state"] for item in deliveries] == [
        "superseded",
        "prepared",
    ]
    store.close()


def test_schema_v5_backfills_legacy_missing_delivery_as_ambiguous(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    limits = limits_for("unattended", "implementation")
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "retain ambiguous legacy dispatch"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        limits.as_dict(),
    )
    attempt, unused_turn_id, unused_checkpoint = _dispatch(
        store,
        created,
        command.command_id,
    )
    del unused_turn_id, unused_checkpoint
    store.mark_provider_boundary(attempt.attempt_id)
    store.update_attempt(attempt.attempt_id, status="failed")
    store.complete_dispatch(attempt.attempt_id, "failed")

    historical = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "Do not backfill a completed historical dispatch."},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    historical_attempt, historical_turn, unused_checkpoint = _dispatch(
        store,
        created,
        historical.command_id,
    )
    del unused_checkpoint
    store.mark_provider_boundary(historical_attempt.attempt_id)
    with store.transaction() as connection:
        connection.execute(
            "UPDATE provider_attempts SET status = 'complete' "
            "WHERE attempt_id = ?",
            (historical_attempt.attempt_id,),
        )
        connection.execute(
            "UPDATE turns SET status = 'complete' WHERE turn_id = ?",
            (historical_turn,),
        )
        connection.execute(
            "UPDATE command_dispatches SET state = 'complete' "
            "WHERE attempt_id = ?",
            (historical_attempt.attempt_id,),
        )
        connection.execute(
            "UPDATE commands SET status = 'complete' WHERE command_id = ?",
            (historical.command_id,),
        )
    with store.transaction() as connection:
        connection.execute(
            "DELETE FROM context_deliveries WHERE attempt_id = ?",
            (attempt.attempt_id,),
        )
        connection.execute("UPDATE schema_meta SET version = 4")
    database = store.path
    store.close()

    recovered = StateStore(database)
    deliveries = recovered.portable_session(created.session_id)["tables"][
        "context_deliveries"
    ]
    assert len(deliveries) == 1
    delivery = deliveries[0]
    assert delivery["attempt_id"] == attempt.attempt_id
    assert delivery["attempt_id"] != historical_attempt.attempt_id
    assert delivery["state"] == "legacy-ambiguous"
    assert delivery["transport"] == "legacy-unknown"
    restart = recovered.recover_interrupted_commands(
        created.session_id,
        "material",
        "summary",
    )
    assert restart.requeued_command_ids == ()
    assert len(restart.reconciliations) == 1
    reconciliation = restart.reconciliations[0]
    updated = recovered.set_reconciliation_command_error(
        reconciliation.reconciliation_id,
        "E_SAFETY_GUARD",
        "retained safety error",
    )
    assert updated.result == {
        "code": "E_SAFETY_GUARD",
        "message": "retained safety error",
        "reconciliation_id": reconciliation.reconciliation_id,
    }
    with pytest.raises(NotFoundError, match="reconciliation"):
        recovered.set_reconciliation_command_error(
            new_uuid(),
            "E_SAFETY_GUARD",
            "missing",
        )
    with recovered.transaction() as connection:
        connection.execute(
            "UPDATE commands SET status = 'queued' WHERE command_id = ?",
            (command.command_id,),
        )
    with pytest.raises(ConflictError, match="failed state"):
        recovered.set_reconciliation_command_error(
            reconciliation.reconciliation_id,
            "E_SAFETY_GUARD",
            "wrong state",
        )
    with recovered.transaction() as connection:
        connection.execute(
            "UPDATE commands SET status = 'failed', result_json = '{}' "
            "WHERE command_id = ?",
            (command.command_id,),
        )
    with pytest.raises(ConflictError, match="result changed"):
        recovered.set_reconciliation_command_error(
            reconciliation.reconciliation_id,
            "E_SAFETY_GUARD",
            "wrong result",
        )
    recovered.close()


def test_recovery_orders_equal_timestamp_dispatches_by_attempt_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = limits_for("unattended", "implementation")
    store, created = _store(tmp_path)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "reconcile the newest deterministic attempt"},
        new_uuid(),
    )
    assert store.claim_command(created.session_id) is not None
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        limits.as_dict(),
    )
    generated = iter(
        [
            "attempt-a",
            "checkpoint-a",
            "attempt-z",
            "checkpoint-z",
        ]
    )
    monkeypatch.setattr(
        "tests.test_adversarial_regressions.new_uuid",
        lambda: next(generated),
    )
    attempts = []
    context_digests = []
    providers = []
    for suffix, provider in (("a", "codex"), ("z", "claude")):
        attempt, unused_turn_id, checkpoint = _dispatch(
            store,
            created,
            command.command_id,
        )
        del unused_turn_id
        context_digest = "context-" + suffix
        store.prepare_context_delivery(
            created.session_id,
            provider,
            context_digest,
            checkpoint.checkpoint_id,
            command.command_id,
            attempt.attempt_id,
            "payload-" + suffix,
        )
        context_digests.append(context_digest)
        attempts.append(attempt)
        providers.append(provider)
    for attempt, context_digest, provider in zip(
        attempts,
        context_digests,
        providers,
        strict=True,
    ):
        store.accept_context_delivery(
            created.session_id,
            provider,
            context_digest,
            attempt.attempt_id,
        )
        store.mark_provider_boundary(attempt.attempt_id)
        store.update_attempt(attempt.attempt_id, status="failed")
        store.complete_dispatch(attempt.attempt_id, "failed")
    with store.transaction() as connection:
        connection.execute(
            "UPDATE command_dispatches SET created_at = ? WHERE command_id = ?",
            ("2026-08-02T00:00:00+00:00", command.command_id),
        )

    recovery = store.recover_interrupted_commands(
        created.session_id,
        "material",
        "summary",
    )

    assert len(recovery.reconciliations) == 1
    assert recovery.reconciliations[0].audit["dispatch_identity"]["attempt_id"] == (
        "attempt-z"
    )
    assert [attempt.attempt_id for attempt in attempts] == [
        "attempt-a",
        "attempt-z",
    ]
    store.close()


def test_command_context_filters_future_instructions_before_limiting(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    current = store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="Current instruction.",
        status="accepted",
        metadata={"command_id": "current"},
    )
    store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="Future instruction one.",
        status="accepted",
        metadata={"command_id": "future-one"},
    )
    store.append_event(
        created.session_id,
        "agent.message",
        role="assistant",
        text="Retain this attempt output.",
        status="complete",
        metadata={"command_id": "current"},
    )
    store.append_event(
        created.session_id,
        "agent.message",
        role="assistant",
        text="Exclude another command output.",
        status="complete",
        metadata={"command_id": "future-one"},
    )
    store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="Future instruction two.",
        status="accepted",
        metadata={"command_id": "future-two"},
    )

    events = store.context_events_for_command(
        created.session_id,
        "current",
        current.sequence,
        limit=1,
    )

    assert [event.text for event in events] == ["Retain this attempt output."]
    store.close()


def test_message_command_creation_and_repair_are_idempotent(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    payload = {"text": "Record one durable instruction."}
    first, first_created = store.ensure_message_command(
        created.session_id,
        payload,
        "atomic-message",
    )
    second, second_created = store.ensure_message_command(
        created.session_id,
        payload,
        "atomic-message",
    )

    assert first_created
    assert not second_created
    assert first.command_id == second.command_id
    assert [
        event.text
        for event in store.events(created.session_id, limit=5000)
        if event.event_type == "user.message"
    ] == ["Record one durable instruction."]
    prior_event = store.append_event(
        created.session_id,
        "checkpoint.created",
        status="complete",
    )
    with store.transaction() as connection:
        connection.execute(
            "DELETE FROM events WHERE session_id = ? AND event_type = 'user.message'",
            (created.session_id,),
        )
    repaired, repaired_created = store.ensure_message_command(
        created.session_id,
        payload,
        "atomic-message",
    )
    assert repaired.command_id == first.command_id
    assert not repaired_created
    repaired_sequence = store.command_instruction_sequence(
        created.session_id,
        first.command_id,
    )
    assert repaired_sequence > prior_event.sequence
    repaired_event = store.events(
        created.session_id,
        after=repaired_sequence - 1,
        limit=1,
    )[0]
    assert repaired_event.created_at >= prior_event.created_at
    assert store.get_session(created.session_id).updated_at >= repaired_event.created_at
    store.close()


def test_command_instruction_lookup_uses_its_partial_index(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    command, was_created = store.ensure_message_command(
        created.session_id,
        {"text": "Use the command instruction index."},
        "indexed-message",
    )

    assert was_created
    plan = store._connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT sequence FROM events
        WHERE session_id = ? AND event_type = 'user.message'
            AND CASE WHEN json_valid(metadata_json)
                THEN json_extract(metadata_json, '$.command_id')
                ELSE NULL
            END = ?
        ORDER BY sequence LIMIT 1
        """,
        (created.session_id, command.command_id),
    ).fetchall()
    assert any(
        "events_command_instruction_v2" in str(row["detail"])
        for row in plan
    )
    store.close()


def test_message_timestamp_is_captured_after_transaction_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, created = _store(tmp_path)
    entered = threading.Event()
    clock_called = threading.Event()
    result: list[Any] = []

    def clock() -> str:
        clock_called.set()
        return "2026-08-02T00:00:03+00:00"

    def create_message() -> None:
        entered.set()
        result.append(
            store.ensure_message_command(
                created.session_id,
                {"text": "Retain monotonic timestamps."},
                "transaction-timestamp",
            )
        )

    monkeypatch.setattr(storage_module, "utc_now", clock)
    thread = threading.Thread(target=create_message)
    with store.transaction() as connection:
        connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            ("2026-08-02T00:00:02+00:00", created.session_id),
        )
        thread.start()
        assert entered.wait(timeout=1)
        assert not clock_called.wait(timeout=0.1)
    thread.join(timeout=1)

    assert not thread.is_alive()
    receipt, was_created = result[0]
    assert was_created
    assert receipt.created_at == "2026-08-02T00:00:03+00:00"
    assert store.get_session(created.session_id).updated_at == (
        "2026-08-02T00:00:03+00:00"
    )
    store.close()


def test_event_timestamp_is_captured_after_transaction_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, created = _store(tmp_path)
    entered = threading.Event()
    clock_called = threading.Event()
    result: list[Any] = []

    def clock() -> str:
        clock_called.set()
        return "2026-08-02T00:00:03+00:00"

    def append() -> None:
        entered.set()
        result.append(
            store.append_event(
                created.session_id,
                "transaction.admitted",
                status="complete",
            )
        )

    monkeypatch.setattr(storage_module, "utc_now", clock)
    thread = threading.Thread(target=append)
    with store.transaction() as connection:
        connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            ("2026-08-02T00:00:02+00:00", created.session_id),
        )
        thread.start()
        assert entered.wait(timeout=1)
        assert not clock_called.wait(timeout=0.1)
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert result[0].created_at == "2026-08-02T00:00:03+00:00"
    assert store.get_session(created.session_id).updated_at == (
        "2026-08-02T00:00:03+00:00"
    )
    store.close()


def test_terminal_message_replay_does_not_resurrect_missing_instruction(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    payload = {"text": "Run this terminal instruction once."}
    command, was_created = store.ensure_message_command(
        created.session_id,
        payload,
        "terminal-message",
    )
    assert was_created
    store.resolve_command(
        command.command_id,
        CommandStatus.COMPLETE,
        {"status": "complete"},
    )
    with store.transaction() as connection:
        connection.execute(
            "DELETE FROM events WHERE session_id = ? AND event_type = 'user.message'",
            (created.session_id,),
        )

    replayed, created_again = store.ensure_message_command(
        created.session_id,
        payload,
        "terminal-message",
    )

    assert replayed.command_id == command.command_id
    assert not created_again
    assert store.events(created.session_id, limit=5000) == []
    with pytest.raises(NotFoundError, match="instruction event"):
        store.command_instruction_sequence(created.session_id, command.command_id)
    with pytest.raises(ConflictError, match="terminal message command"):
        store.repair_command_instruction_event(
            created.session_id,
            command.command_id,
        )
    store.close()


def test_missing_instruction_event_is_repaired_from_the_command(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    idempotency_key = new_uuid()
    payload = {
        "text": "Recover the durable instruction.",
        "turn_ref": {"step_id": "recover", "agent_role": "builder"},
    }
    command = store.enqueue_command(
        created.session_id,
        "message",
        payload,
        idempotency_key,
    )

    with pytest.raises(NotFoundError, match="instruction event"):
        store.command_instruction_sequence(created.session_id, command.command_id)
    with pytest.raises(NotFoundError, match="instruction event"):
        store.repair_command_instruction_event(
            created.session_id,
            new_uuid(),
        )
    retried, created_again = store.ensure_message_command(
        created.session_id,
        payload,
        idempotency_key,
    )
    sequence = store.command_instruction_sequence(created.session_id, command.command_id)

    assert retried.command_id == command.command_id
    assert not created_again
    assert sequence == 1
    event = store.events(created.session_id, limit=5000)[0]
    assert event.text == "Recover the durable instruction."
    assert event.metadata == {
        "command_id": command.command_id,
        "repaired": True,
        "turn_ref": {"step_id": "recover", "agent_role": "builder"},
    }
    empty = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "   "},
        "empty-message",
    )
    with pytest.raises(NotFoundError, match="instruction event"):
        store.command_instruction_sequence(created.session_id, empty.command_id)
    with pytest.raises(ConflictError, match="no recoverable instruction text"):
        store.repair_command_instruction_event(
            created.session_id,
            empty.command_id,
        )
    with pytest.raises(ConflictError, match="no recoverable instruction text"):
        store.ensure_message_command(
            created.session_id,
            {"text": "   "},
            "empty-message",
        )
    store.close()


def test_legacy_effort_spelling_remains_idempotent_after_normalization(
    tmp_path: Path,
) -> None:
    store, created = _store(tmp_path)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "Retain one request.", "effort": "high"},
        "legacy-effort",
    )
    with store.transaction() as connection:
        connection.execute(
            "UPDATE commands SET payload_json = ? WHERE command_id = ?",
            (
                json.dumps(
                    {"text": "Retain one request.", "effort": " HIGH "},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                command.command_id,
            ),
        )

    retried, created_again = store.ensure_command(
        created.session_id,
        "message",
        {"text": "Retain one request.", "effort": "high"},
        "legacy-effort",
    )

    assert retried.command_id == command.command_id
    assert not created_again
    default_effort = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "Use the provider default."},
        "default-effort",
    )
    blank_effort, blank_created = store.ensure_command(
        created.session_id,
        "message",
        {"text": "Use the provider default.", "effort": "  "},
        "default-effort",
    )
    assert blank_effort.command_id == default_effort.command_id
    assert not blank_created
    legacy_payload = {"text": "Retain one request.", "effort": " HIGH "}
    assert command_envelope_digest(
        "message",
        legacy_payload,
        "unattended",
    ) != legacy_command_envelope_digest(
        "message",
        legacy_payload,
        "unattended",
    )

    typed = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "Keep JSON types exact.", "_effort_pinned": True},
        "typed-idempotency",
    )
    assert typed.command_id
    with pytest.raises(ConflictError, match="different command input"):
        store.ensure_command(
            created.session_id,
            "message",
            {"text": "Keep JSON types exact.", "_effort_pinned": 1},
            "typed-idempotency",
        )
    store.close()


def test_delivery_v2_preserves_the_v1_idempotency_binding() -> None:
    dispatch = {
        "attempt_id": "attempt-one",
        "turn_id": "turn-one",
    }
    base = {
        "session_id": "session-one",
        "provider": "codex",
        "context_digest": "context-one",
        "checkpoint_id": "checkpoint-one",
        "command_id": "command-one",
        "attempt_id": "attempt-one",
        "payload_digest": "payload-one",
        "state": "delivered",
        "accepted_at": "2026-08-02T00:00:00+00:00",
        "delivered_at": "2026-08-02T00:00:00+00:00",
    }
    context_package = proof_module._proof_context_deliveries(
        [{**base, "transport": "context-package"}],
        [dispatch],
    )[0]
    native_resume = proof_module._proof_context_deliveries(
        [{**base, "transport": "native-resume"}],
        [dispatch],
    )[0]

    assert context_package["delivery_version"] == 2
    assert context_package["idempotency_digest"] == native_resume[
        "idempotency_digest"
    ]
    assert context_package["context_delivery_digest"] != native_resume[
        "context_delivery_digest"
    ]


def test_command_proof_marks_the_versioned_envelope_shape() -> None:
    command = proof_module._proof_command(
        {
            "command_id": "command-one",
            "session_id": "session-one",
            "command_type": "message",
            "payload_json": '{"text":"one"}',
            "result_json": "{}",
        },
        {},
        {"command-one": "unattended"},
    )

    assert command["command_envelope_version"] == 2


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
