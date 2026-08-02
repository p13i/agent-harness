"""Deterministic coverage for orchestration boundary behavior."""

from __future__ import annotations

import asyncio
import copy
import io
import json
import signal
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness import api as api_module
from agent_harness import client as client_module
from agent_harness import process_control as process_control_module
from agent_harness import service as service_module
from agent_harness import worker as worker_module
from agent_harness import workspace as workspace_module
from agent_harness.client import wait_command
from agent_harness.config import paths, prepare_paths
from agent_harness.errors import (
    ConflictError,
    HarnessError,
    ProviderUnavailableError,
    ReconciliationRequiredError,
    SafetyGuardError,
    WorkerOwnershipLostError,
)
from agent_harness.goals import create_goal
from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import (
    Attention,
    Checkpoint,
    CommandStatus,
    Goal,
    Lifecycle,
    PermissionMode,
    ProviderAttempt,
    RoutingDecision,
    Session,
)
from agent_harness.orchestration import normalized_digest
from agent_harness.scheduler import Scheduler
from agent_harness.safety import TurnGuard, limits_for
from agent_harness.service import HarnessService, WorkerManager, _export_digests
from agent_harness.storage import StateStore
from agent_harness.transfer import (
    MachineKeys,
    load_machine_keys,
    open_transfer,
    seal_transfer,
)
from agent_harness.providers.base import (
    ProviderAdapter,
    ProviderEvent,
    ProviderModel,
    ProviderResult,
    ProviderStatus,
)
from agent_harness.worker import SessionWorker

SUPPORTED_PROVIDERS = frozenset({"claude", "codex", "kimi"})


class WorkerProbe:
    def __init__(self) -> None:
        self.sessions: list[str] = []

    def ensure(self, session_id: str) -> None:
        self.sessions.append(session_id)


@pytest.fixture(autouse=True)
def stable_state_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_module,
        "require_state_headroom",
        lambda unused_path, unused_provider: 10 * 1024**3,
    )


def _service(tmp_path: Path) -> HarnessService:
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    return HarnessService(
        harness_paths,
        worker_manager=WorkerProbe(),  # type: ignore[arg-type]
    )


def _create_direct(
    service: HarnessService,
    tmp_path: Path,
    **values: object,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").is_dir():
        subprocess.run(
            ["git", "-C", str(workspace), "init", "-q"],
            check=True,
        )
    payload: dict[str, object] = {
        "workspace": str(workspace),
        "direct": True,
        "execution_profile": "interactive",
    }
    payload.update(values)
    return service.create_session(payload)


def test_client_wait_command_polls_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def request(
            self,
            unused_method: str,
            unused_path: str,
        ) -> dict[str, object]:
            del unused_method, unused_path
            self.calls += 1
            if self.calls == 1:
                return {"command": {"status": "running"}}
            return {"command": {"status": "complete"}}

    sleeps: list[float] = []

    async def no_sleep(value: float) -> None:
        sleeps.append(value)

    monkeypatch.setattr(client_module.asyncio, "sleep", no_sleep)
    client = Client()
    result = asyncio.run(
        wait_command(
            client,  # type: ignore[arg-type]
            "command",
            timeout=10,
        )
    )

    assert result["status"] == "complete"
    assert sleeps == [0.2]


def test_scheduler_refresh_status_runs_both_sources(
    tmp_path: Path,
) -> None:
    scheduler = object.__new__(Scheduler)
    calls: list[str] = []

    async def refresh_usage() -> dict[str, object]:
        calls.append("usage")
        return {}

    async def models(
        workspace: Path,
        *,
        refresh: bool,
    ) -> dict[str, object]:
        assert workspace == tmp_path
        assert refresh
        calls.append("models")
        return {}

    scheduler.refresh_usage = refresh_usage  # type: ignore[method-assign]
    scheduler.models = models  # type: ignore[method-assign]
    asyncio.run(scheduler._refresh_status(tmp_path))

    assert sorted(calls) == ["models", "usage"]


def test_scheduler_never_admits_restored_usage_before_fresh_probe(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "scheduler-state.sqlite3")
    store.record_usage(
        "codex",
        10.0,
        False,
        {"payload": {"source": "old-process"}, "error": ""},
    )
    scheduler = Scheduler(store, {})
    refreshed: list[bool] = []

    async def refresh() -> dict[str, object]:
        refreshed.append(True)
        scheduler._usage_cache = {}
        scheduler._usage_at = asyncio.get_running_loop().time()
        return {}

    scheduler.refresh_usage = refresh  # type: ignore[method-assign]

    assert scheduler._usage_cache
    assert scheduler._usage_at is None
    assert asyncio.run(scheduler.usage()) == {}
    assert refreshed == [True]
    store.close()


def test_worker_manager_starts_reuses_and_stops_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    processes: list[SimpleNamespace] = []

    class Process:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.signals: list[int] = []

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, value: int) -> None:
            self.signals.append(value)

    def popen(
        command: list[str],
        **values: object,
    ) -> Process:
        assert command[-2:] == ["worker", "session"]
        assert values["start_new_session"] is True
        process = Process()
        processes.append(process)  # type: ignore[arg-type]
        return process

    monkeypatch.setattr(
        service_module,
        "launcher_command",
        lambda: ["agent-harness"],
    )
    monkeypatch.setattr(service_module.subprocess, "Popen", popen)
    manager = WorkerManager(harness_paths)

    manager.ensure("session")
    manager.ensure("session")
    assert len(processes) == 1
    manager.stop_all()
    assert processes[0].signals == [signal.SIGTERM]

    processes[0].returncode = 1
    manager.ensure("session")
    assert len(processes) == 2


def test_worker_supervision_bounds_spawn_failures_and_recovers_after_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    clock = {"value": 0.0}
    attempts: list[int] = []

    class Process:
        def poll(self) -> None:
            return None

        def send_signal(self, unused_value: int) -> None:
            del unused_value

    def popen(*unused_args: object, **unused_values: object) -> Process:
        del unused_args
        del unused_values
        attempts.append(1)
        if len(attempts) <= 3:
            raise OSError("bounded spawn failure")
        return Process()

    monkeypatch.setattr(
        service_module,
        "launcher_command",
        lambda: ["agent-harness"],
    )
    monkeypatch.setattr(service_module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        service_module.time,
        "monotonic",
        lambda: clock["value"],
    )
    manager = WorkerManager(harness_paths)

    for offset in range(4):
        clock["value"] = float(offset)
        report = manager.supervise(["session"])

    assert len(attempts) == 3
    assert report["status"] == "failed"
    assert report["unrecovered"][0]["reason"] == "worker-restart-limit"

    clock["value"] = 64.0
    recovered = manager.supervise(["session"])
    assert len(attempts) == 4
    assert recovered["status"] == "ok"
    assert recovered["running_sessions"] == ["session"]


def test_recovery_blocked_lease_suppresses_supervision_until_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    session = _create_direct(service, tmp_path)
    service.store.update_session(
        session.session_id,
        lifecycle=Lifecycle.PAUSED,
        attention=Attention.NEEDS_RECONCILIATION,
    )
    lease = service.store.create_process_lease(
        session.session_id,
        "codex",
        "unattended",
        "2099-01-01T00:00:00+00:00",
    )
    service.store.update_process_lease(
        str(lease["lease_id"]),
        pid=123,
        pid_start="recorded-start",
        state="recovery-blocked",
    )
    service.store.append_event(
        session.session_id,
        "lease.recovery.blocked",
        status="needs-reconciliation",
        metadata={"lease_id": str(lease["lease_id"])},
    )

    starts: list[list[str]] = []
    clock = {"value": 0.0}

    class Process:
        def __init__(self, returncode: int | None) -> None:
            self.returncode = returncode

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, unused_value: int) -> None:
            del unused_value

    def popen(command: list[str], **unused_values: object) -> Process:
        del unused_values
        starts.append(command)
        return Process(None)

    monkeypatch.setattr(
        service_module,
        "launcher_command",
        lambda: ["agent-harness"],
    )
    monkeypatch.setattr(service_module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        service_module.time,
        "monotonic",
        lambda: clock["value"],
    )
    manager = WorkerManager(service.paths)
    manager._processes[session.session_id] = Process(1)  # type: ignore[assignment]
    service.workers = manager
    try:
        for observed_at in (0.0, 70.0, 140.0):
            clock["value"] = observed_at
            report = service.supervise_workers()
            assert report["status"] == "failed"
            assert report["expected_sessions"] == []
            assert report["unrecovered"] == [
                {
                    "session_id": session.session_id,
                    "lease_id": str(lease["lease_id"]),
                    "command_id": "",
                    "attempt_id": "",
                    "provider": "codex",
                    "reason": "process-lease-recovery-blocked",
                }
            ]
        assert starts == []
        blocked_events = [
            event
            for event in service.store.all_events(session.session_id)
            if event.event_type == "lease.recovery.blocked"
        ]
        assert len(blocked_events) == 1
        with pytest.raises(ConflictError):
            service.update_process_lease(
                str(lease["lease_id"]),
                {
                    "action": "attach",
                    "pid": 456,
                    "pid_start": "new-start",
                },
            )

        released = service.update_process_lease(
            str(lease["lease_id"]),
            {"action": "release"},
        )
        assert released["state"] == "released"
        with pytest.raises(ConflictError, match="terminal lease"):
            service.update_process_lease(
                str(lease["lease_id"]),
                {
                    "action": "attach",
                    "pid": 456,
                    "pid_start": "new-start",
                },
            )
        with pytest.raises(ConflictError, match="heartbeated"):
            service.update_process_lease(
                str(lease["lease_id"]),
                {"action": "heartbeat"},
            )
        clock["value"] = 210.0
        recovered = service.supervise_workers()
        assert recovered["status"] == "ok"
        assert recovered["restarted_sessions"] == [session.session_id]
        assert len(starts) == 1
        clock["value"] = 280.0
        stable = service.supervise_workers()
        assert stable["status"] == "ok"
        assert stable["restarted_sessions"] == []
        assert len(starts) == 1
        events = service.store.all_events(session.session_id)
        assert (
            len(
                [
                    event
                    for event in events
                    if event.event_type == "lease.recovery.unblocked"
                ]
            )
            == 1
        )
        assert (
            len([event for event in events if event.event_type == "worker.supervised"])
            == 1
        )
    finally:
        service.close()


def test_service_creation_configuration_and_archive_boundaries(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    try:
        invalid_payloads = (
            {},
            {
                "workspace": str(tmp_path),
                "permission_mode": "invalid",
            },
            {
                "workspace": str(tmp_path),
                "predicates": "invalid",
            },
            {
                "workspace": str(tmp_path),
                "constraints": "invalid",
            },
            {
                "workspace": str(tmp_path),
                "budgets": [],
            },
        )
        for payload in invalid_payloads:
            with pytest.raises(ValueError):
                service.create_session(payload)

        external_ref = {
            "orchestrator": "test",
            "job_id": "job-1",
        }
        payload = {
            "workspace": str(tmp_path / "workspace"),
            "direct": True,
            "execution_profile": "interactive",
            "external_ref": external_ref,
        }
        workspace = Path(str(payload["workspace"]))
        workspace.mkdir(parents=True)
        subprocess.run(
            ["git", "-C", str(workspace), "init", "-q"],
            check=True,
        )
        first = service.create_session(
            payload,
            idempotency_key="creation-key",
        )
        second = service.create_session(
            payload,
            idempotency_key="creation-key",
        )
        assert first.session_id == second.session_id

        with pytest.raises(ValueError, match="message text"):
            service.submit_message(
                first.session_id,
                {"text": ""},
                "message-key",
            )
        with pytest.raises(ValueError, match="session name"):
            service.configure_session(first.session_id, {"name": ""})
        with pytest.raises(ValueError, match="permission"):
            service.configure_session(
                first.session_id,
                {"permission_mode": "invalid"},
            )
        with pytest.raises(ValueError, match="no supported"):
            service.configure_session(first.session_id, {})

        configured = service.configure_session(
            first.session_id,
            {"execution_profile": "interactive"},
        )
        assert configured.session_id == first.session_id
        with pytest.raises(ConflictError, match="immutable"):
            service.configure_session(
                first.session_id,
                {"execution_profile": "unattended"},
            )
        assert service.store.session_safety(first.session_id)["profile"] == (
            "interactive"
        )
        assert (
            service.set_session_archived(
                first.session_id,
                False,
            ).session_id
            == configured.session_id
        )
        archived = service.set_session_archived(first.session_id, True)
        assert archived.archived
        assert not service.set_session_archived(
            first.session_id,
            False,
        ).archived
    finally:
        service.close()


def test_service_miscellaneous_validation_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    try:
        session = _create_direct(service, tmp_path)
        other = _create_direct(
            service,
            tmp_path / "other",
        )
        checkpoint_id = new_uuid()
        monkeypatch.setattr(
            service.store,
            "checkpoint",
            lambda unused: SimpleNamespace(session_id=other.session_id),
        )
        with pytest.raises(ValueError, match="does not belong"):
            service.checkpoint_diff(session.session_id, checkpoint_id)

        assert not service.resolve_approval(
            session.session_id,
            new_uuid(),
            {"decision": "deny"},
        )["resolved"]
        with pytest.raises(ValueError, match="no goal"):
            service.add_evidence(session.session_id, {})

        goal_session = _create_direct(
            service,
            tmp_path / "goal",
            goal="Verify boundaries",
        )
        evidence = service.add_evidence(
            goal_session.session_id,
            {
                "type": "test",
                "subject": "core",
                "outcome": "passed",
                "value": [],
            },
        )
        assert evidence["outcome"] == "passed"

        service.store.update_session(
            session.session_id,
            attention=Attention.WORKING,
        )
        with pytest.raises(ValueError, match="active turn"):
            service.create_transfer(session.session_id, {})
        service.store.update_session(
            session.session_id,
            attention=Attention.IDLE,
        )
        with pytest.raises(ValueError, match="destination host"):
            service.create_transfer(session.session_id, {})
        with pytest.raises(ValueError, match="base64"):
            service.import_transfer({"envelope": "!", "source_signing_public": ""})
        with pytest.raises(ValueError, match="owner epoch"):
            service.finalize_transfer(
                session.session_id,
                "host",
                session.owner_epoch,
            )
    finally:
        service.close()


def test_machines_fault_probe_requires_exact_goal_bound_authorization(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    try:
        session = _create_direct(service, tmp_path)
        external_ref = {
            "orchestrator": "p13i/machines/agent-harness-proof-builder",
            "job_id": "fault-authorization-job",
        }
        with service.store.transaction() as connection:
            connection.execute(
                """
                UPDATE sessions SET external_orchestrator = ?,
                    external_job_id = ? WHERE session_id = ?
                """,
                (
                    external_ref["orchestrator"],
                    external_ref["job_id"],
                    session.session_id,
                ),
            )
        idempotency_key = "fault-authorized-message"
        authorization = {
            "schema": "p13i/machines/provider-fault-authorization/v1",
            "external_ref": external_ref,
            "idempotency_key": idempotency_key,
            "stage": "after-lease-before-acceptance",
            "provider": "codex",
            "agent_role": "proof-fault-probe",
        }
        authorization_digest = normalized_digest(authorization)
        service_idempotency_key = "service-fault-authorized-message"
        service_authorization = {
            "schema": "p13i/machines/service-fault-authorization/v1",
            "external_ref": external_ref,
            "idempotency_key": service_idempotency_key,
            "stage": "after-acceptance-before-terminal",
            "provider": "claude",
            "agent_role": "proof-service-fault-probe",
        }
        service_authorization_digest = normalized_digest(service_authorization)
        service.store.create_goal(
            create_goal(
                session.session_id,
                "Prove one bounded provider fault.",
                constraints=(
                    "proof-fault-authorization-sha256:" + authorization_digest,
                    "proof-service-fault-authorization-sha256:"
                    + service_authorization_digest,
                ),
            )
        )
        payload = {
            "text": "Run the precommitted provider fault probe.",
            "required_capabilities": ["proof-fault-barrier"],
            "turn_ref": {
                "step_id": "provider-fault",
                "agent_role": "proof-fault-probe",
            },
            "proof_fault_probe": {
                "stage": "after-lease-before-acceptance",
                "provider": "codex",
                "authorization": authorization,
                "authorization_digest": authorization_digest,
            },
        }
        receipt = service.submit_message(
            session.session_id,
            payload,
            idempotency_key,
        )
        assert receipt["status"] == "queued"

        service_receipt = service.submit_message(
            session.session_id,
            {
                "text": "Run the precommitted service fault probe.",
                "required_capabilities": ["proof-service-fault-barrier"],
                "turn_ref": {
                    "step_id": "service-fault",
                    "agent_role": "proof-service-fault-probe",
                },
                "proof_service_fault_probe": {
                    "stage": "after-acceptance-before-terminal",
                    "provider": "claude",
                    "authorization": service_authorization,
                    "authorization_digest": service_authorization_digest,
                },
            },
            service_idempotency_key,
        )
        assert service_receipt["status"] == "queued"

        forged = dict(payload)
        forged_probe = dict(payload["proof_fault_probe"])
        forged_probe["authorization_digest"] = "f" * 64
        forged["proof_fault_probe"] = forged_probe
        with pytest.raises(ValueError, match="digest does not match"):
            service.submit_message(
                session.session_id,
                forged,
                idempotency_key,
            )
    finally:
        service.close()


def test_service_route_wait_and_export_digest_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    try:
        session = _create_direct(service, tmp_path)
        decision = RoutingDecision(
            provider="codex",
            model="default",
            effort="low",
            reason="test",
            ranked=(),
            rejected=(),
        )

        async def choose(*unused: object, **values: object) -> RoutingDecision:
            del unused
            assert values["enforce_concurrency"] is True
            return decision

        monkeypatch.setattr(service.scheduler, "choose", choose)
        safety = service.store.session_safety(session.session_id)
        safety["profile"] = ""
        monkeypatch.setattr(
            service.store,
            "session_safety",
            lambda unused: safety,
        )
        route = asyncio.run(
            service.preview_route(
                session.session_id,
                {
                    "required_capabilities": ["tools"],
                    "metered_budget": True,
                },
            )
        )
        assert route["provider"] == "codex"

        receipts = [
            SimpleNamespace(
                status=CommandStatus.DISPATCHING,
                as_dict=lambda: {"status": "dispatching"},
            ),
            SimpleNamespace(
                status=CommandStatus.COMPLETE,
                as_dict=lambda: {"status": "complete"},
            ),
        ]
        monkeypatch.setattr(
            service.store,
            "get_command",
            lambda unused: receipts.pop(0),
        )

        async def no_sleep(unused: float) -> None:
            return

        monkeypatch.setattr(service_module.asyncio, "sleep", no_sleep)
        assert (
            asyncio.run(service.wait_for_command("command", timeout=10))["status"]
            == "complete"
        )
    finally:
        service.close()

    assert _export_digests(
        {
            "events": [
                "invalid",
                {"blob_digest": ""},
                {"blob_digest": "event"},
            ],
            "checkpoints": [
                "invalid",
                {
                    "patch_digest": "patch",
                    "untracked_digest": "",
                    "context_digest": "context",
                },
            ],
        }
    ) == {"event", "patch", "context"}
    assert _export_digests({"events": {}, "checkpoints": {}}) == set()


def test_transfer_and_workspace_reject_malformed_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_keys = tmp_path / "keys.json"
    invalid_keys.write_text("{", encoding="utf-8")
    with pytest.raises(HarnessError, match="cannot be read"):
        load_machine_keys(invalid_keys)

    source = MachineKeys.generate()
    destination = MachineKeys.generate()
    envelope = seal_transfer(
        ["invalid"],  # type: ignore[arg-type]
        destination_encryption_public=(destination.public_bundle()["encryption"]),
        source_signing_private=source.signing_private,
    )
    with pytest.raises(HarnessError, match="not an object"):
        open_transfer(
            envelope,
            destination_encryption_private=destination.encryption_private,
            source_signing_public=source.public_bundle()["signing"],
        )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    (workspace / "directory").mkdir()
    (workspace / "link").symlink_to(outside)
    inside = workspace / "inside"
    inside.write_text("inside", encoding="utf-8")
    (workspace / "inside-link").symlink_to("inside")

    def listed(*unused: object, **unused_values: object):
        del unused, unused_values
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=b"inside\0inside-link\0",
            stderr=b"",
        )

    monkeypatch.setattr(workspace_module.subprocess, "run", listed)
    archive = workspace_module._untracked_archive(workspace)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as value:
        members = {member.name: member for member in value.getmembers()}
        assert set(members) == {"inside", "inside-link"}
        assert members["inside"].isreg()
        assert members["inside-link"].issym()
        assert members["inside-link"].linkname == "inside"

    def unsafe_listed(*unused: object, **unused_values: object):
        del unused, unused_values
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=b"../outside\0",
            stderr=b"",
        )

    monkeypatch.setattr(
        workspace_module.subprocess,
        "run",
        unsafe_listed,
    )
    with pytest.raises(HarnessError, match="escapes"):
        workspace_module._untracked_archive(workspace)

    def outside_link_listed(*unused: object, **unused_values: object):
        del unused, unused_values
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=b"link\0",
            stderr=b"",
        )

    monkeypatch.setattr(
        workspace_module.subprocess,
        "run",
        outside_link_listed,
    )
    with pytest.raises(HarnessError, match="link escapes"):
        workspace_module._untracked_archive(workspace)

    def directory_listed(*unused: object, **unused_values: object):
        del unused, unused_values
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=b"directory\0",
            stderr=b"",
        )

    monkeypatch.setattr(
        workspace_module.subprocess,
        "run",
        directory_listed,
    )
    with pytest.raises(HarnessError, match="unsupported"):
        workspace_module._untracked_archive(workspace)

    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    (outside_directory / "file").write_text("outside", encoding="utf-8")
    (workspace / "directory-link").symlink_to(outside_directory)

    def nested_outside_listed(*unused: object, **unused_values: object):
        del unused, unused_values
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=b"directory-link/file\0",
            stderr=b"",
        )

    monkeypatch.setattr(
        workspace_module.subprocess,
        "run",
        nested_outside_listed,
    )
    with pytest.raises(HarnessError, match="path escapes"):
        workspace_module._untracked_archive(workspace)

    link_archive = io.BytesIO()
    with tarfile.open(fileobj=link_archive, mode="w:gz") as value:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/outside"
        value.addfile(member)
    with pytest.raises(HarnessError, match="link escapes"):
        workspace_module._extract_untracked(
            workspace,
            link_archive.getvalue(),
        )

    hardlink_archive = io.BytesIO()
    with tarfile.open(fileobj=hardlink_archive, mode="w:gz") as value:
        member = tarfile.TarInfo("hardlink")
        member.type = tarfile.LNKTYPE
        member.linkname = "inside"
        value.addfile(member)
    with pytest.raises(HarnessError, match="unsupported object"):
        workspace_module._extract_untracked(
            workspace,
            hardlink_archive.getvalue(),
        )

    escape_archive = io.BytesIO()
    with tarfile.open(fileobj=escape_archive, mode="w:gz") as value:
        content = b"escape"
        member = tarfile.TarInfo("../escape")
        member.size = len(content)
        value.addfile(member, io.BytesIO(content))
    with pytest.raises(HarnessError, match="escapes"):
        workspace_module._extract_untracked(
            workspace,
            escape_archive.getvalue(),
        )


def test_checkpoint_archive_rejects_collisions_and_malformed_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "escape-parent").symlink_to(outside)
    (workspace / "existing").write_text("existing", encoding="utf-8")

    assert workspace_module._safe_archive_path("") is None
    assert workspace_module._safe_archive_path("/absolute") is None
    assert workspace_module._safe_archive_path("a/../b") is None
    assert workspace_module._safe_archive_path("safe/path") is not None
    assert workspace_module._safe_link_target("link", "") is None
    assert workspace_module._safe_link_target("link", "/absolute") is None
    assert workspace_module._safe_link_target("link", "../escape") is None
    assert workspace_module._safe_link_target("nested/link", "..") is None
    safe_target = workspace_module._safe_link_target(
        "nested/link",
        "../target",
    )
    assert safe_target is not None
    assert safe_target.as_posix() == "target"

    duplicate = io.BytesIO()
    with tarfile.open(fileobj=duplicate, mode="w:gz") as archive:
        archive.addfile(tarfile.TarInfo("duplicate"), io.BytesIO())
        archive.addfile(tarfile.TarInfo("duplicate"), io.BytesIO())
    with pytest.raises(HarnessError, match="duplicate"):
        workspace_module._extract_untracked(workspace, duplicate.getvalue())

    topology = io.BytesIO()
    with tarfile.open(fileobj=topology, mode="w:gz") as archive:
        archive.addfile(tarfile.TarInfo("parent"), io.BytesIO())
        archive.addfile(tarfile.TarInfo("parent/child"), io.BytesIO())
    with pytest.raises(HarnessError, match="topology"):
        workspace_module._extract_untracked(workspace, topology.getvalue())

    parent_escape = io.BytesIO()
    with tarfile.open(fileobj=parent_escape, mode="w:gz") as archive:
        archive.addfile(
            tarfile.TarInfo("escape-parent/file"),
            io.BytesIO(),
        )
    with pytest.raises(HarnessError, match="escapes"):
        workspace_module._extract_untracked(
            workspace,
            parent_escape.getvalue(),
        )

    collision = io.BytesIO()
    with tarfile.open(fileobj=collision, mode="w:gz") as archive:
        archive.addfile(tarfile.TarInfo("existing"), io.BytesIO())
    with pytest.raises(HarnessError, match="collides"):
        workspace_module._extract_untracked(workspace, collision.getvalue())

    (workspace / "existing-link").write_text("collision", encoding="utf-8")
    symlink_collision = io.BytesIO()
    with tarfile.open(fileobj=symlink_collision, mode="w:gz") as archive:
        member = tarfile.TarInfo("existing-link")
        member.type = tarfile.SYMTYPE
        member.linkname = "existing"
        archive.addfile(member)
    with pytest.raises(HarnessError, match="collides"):
        workspace_module._extract_untracked(
            workspace,
            symlink_collision.getvalue(),
        )

    missing_member = tarfile.TarInfo("missing-payload")

    class MissingPayloadArchive:
        def __enter__(self) -> MissingPayloadArchive:
            return self

        def __exit__(self, *unused: object) -> None:
            del unused

        def getmembers(self) -> list[tarfile.TarInfo]:
            return [missing_member]

        def extractfile(self, unused: tarfile.TarInfo) -> None:
            del unused
            return None

    with monkeypatch.context() as context:
        context.setattr(
            workspace_module.tarfile,
            "open",
            lambda **unused: MissingPayloadArchive(),
        )
        with pytest.raises(HarnessError, match="payload is missing"):
            workspace_module._extract_untracked(workspace, b"archive")


@pytest.mark.asyncio
async def test_process_group_control_is_identity_bound_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        process_control_module.process_group_identity(0)
    monkeypatch.setattr(process_control_module.os, "getpgid", lambda pid: pid + 1)
    with pytest.raises(RuntimeError, match="leader"):
        process_control_module.process_group_identity(41)
    monkeypatch.setattr(process_control_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        process_control_module,
        "_process_start",
        lambda unused_pid: "started",
    )
    identity = process_control_module.process_group_identity(41)
    assert identity == process_control_module.ProcessGroupIdentity(
        pid=41,
        pgid=41,
        pid_start="started",
    )

    class Process:
        def __init__(self, pid: int, returncode: int | None = None) -> None:
            self.pid = pid
            self.returncode = returncode
            self.waited = 0

        async def wait(self) -> int:
            self.waited += 1
            return 0

    process = Process(41)
    with pytest.raises(RuntimeError, match="identity changed"):
        await process_control_module.terminate_process_group(
            Process(42),  # type: ignore[arg-type]
            identity,
        )
    monkeypatch.setattr(
        process_control_module,
        "process_group_identity",
        lambda unused_pid: process_control_module.ProcessGroupIdentity(
            41,
            41,
            "changed",
        ),
    )
    with pytest.raises(RuntimeError, match="start identity"):
        await process_control_module.terminate_process_group(
            process,  # type: ignore[arg-type]
            identity,
        )
    monkeypatch.setattr(
        process_control_module,
        "process_group_identity",
        lambda unused_pid: identity,
    )
    signals: list[signal.Signals] = []
    bounded_waits: list[float] = []

    def record_signal(
        unused_pgid: int,
        selected_signal: signal.Signals,
    ) -> None:
        del unused_pgid
        signals.append(selected_signal)

    async def record_wait(unused_process: object, timeout: float) -> None:
        del unused_process
        bounded_waits.append(timeout)

    monkeypatch.setattr(process_control_module, "_signal_group", record_signal)
    monkeypatch.setattr(process_control_module, "_bounded_wait", record_wait)

    outcomes = iter((True,))

    async def group_exit(
        unused_pgid: int,
        unused_timeout: float,
    ) -> bool:
        del unused_pgid
        del unused_timeout
        return next(outcomes)

    monkeypatch.setattr(process_control_module, "_wait_group_exit", group_exit)
    await process_control_module.terminate_process_group(
        process,  # type: ignore[arg-type]
        identity,
        grace_timeout=1.0,
        kill_timeout=3.0,
    )
    assert signals == []
    assert bounded_waits == [3.0]

    signals.clear()
    bounded_waits.clear()
    outcomes = iter((True,))
    await process_control_module.terminate_process_group(
        process,  # type: ignore[arg-type]
        identity,
        terminate_timeout=1.0,
        kill_timeout=4.0,
    )
    assert signals == [signal.SIGTERM]
    assert bounded_waits == [4.0]

    signals.clear()
    bounded_waits.clear()
    outcomes = iter((False, True))
    await process_control_module.terminate_process_group(
        process,  # type: ignore[arg-type]
        identity,
    )
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert bounded_waits == [2.0]

    outcomes = iter((False, False))
    with pytest.raises(RuntimeError, match="did not exit"):
        await process_control_module.terminate_process_group(
            process,  # type: ignore[arg-type]
            identity,
        )


@pytest.mark.asyncio
async def test_recorded_process_control_and_wait_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert await process_control_module.terminate_recorded_process_group(0, "") == (
        "unattached"
    )

    def missing_identity(unused_pid: int) -> object:
        del unused_pid
        raise ProcessLookupError

    monkeypatch.setattr(
        process_control_module,
        "process_group_identity",
        missing_identity,
    )
    assert await process_control_module.terminate_recorded_process_group(
        41,
        "started",
    ) == ("already-exited")

    monkeypatch.setattr(
        process_control_module,
        "_group_exists",
        lambda unused_pgid: True,
    )
    assert await process_control_module.terminate_recorded_process_group(
        41,
        "started",
    ) == ("identity-invalid")

    monkeypatch.setattr(
        process_control_module,
        "_group_exists",
        lambda unused_pgid: False,
    )

    def invalid_identity(unused_pid: int) -> object:
        del unused_pid
        raise OSError

    monkeypatch.setattr(
        process_control_module,
        "process_group_identity",
        invalid_identity,
    )
    assert await process_control_module.terminate_recorded_process_group(
        41,
        "started",
    ) == ("identity-invalid")

    identity = process_control_module.ProcessGroupIdentity(41, 41, "new-start")
    monkeypatch.setattr(
        process_control_module,
        "process_group_identity",
        lambda unused_pid: identity,
    )
    assert await process_control_module.terminate_recorded_process_group(
        41,
        "old-start",
    ) == ("identity-changed")

    signals: list[signal.Signals] = []

    def record_signal(
        unused_pgid: int,
        selected_signal: signal.Signals,
    ) -> None:
        del unused_pgid
        signals.append(selected_signal)

    monkeypatch.setattr(process_control_module, "_signal_group", record_signal)
    outcomes = iter((True,))

    async def group_exit(
        unused_pgid: int,
        unused_timeout: float,
    ) -> bool:
        del unused_pgid
        del unused_timeout
        return next(outcomes)

    monkeypatch.setattr(process_control_module, "_wait_group_exit", group_exit)
    assert await process_control_module.terminate_recorded_process_group(
        41,
        "new-start",
    ) == ("terminated")
    assert signals == [signal.SIGTERM]

    signals.clear()
    outcomes = iter((False, True))
    assert await process_control_module.terminate_recorded_process_group(
        41,
        "new-start",
    ) == ("killed")
    assert signals == [signal.SIGTERM, signal.SIGKILL]

    outcomes = iter((False, False))
    with pytest.raises(RuntimeError, match="after SIGKILL"):
        await process_control_module.terminate_recorded_process_group(
            41,
            "new-start",
        )

    class WaitProcess:
        def __init__(self, returncode: int | None) -> None:
            self.returncode = returncode
            self.waited = 0

        async def wait(self) -> int:
            self.waited += 1
            return 0

    completed = WaitProcess(0)
    await process_control_module._bounded_wait(
        completed,  # type: ignore[arg-type]
        1.0,
    )
    assert completed.waited == 0
    active = WaitProcess(None)
    await process_control_module._bounded_wait(
        active,  # type: ignore[arg-type]
        1.0,
    )
    assert active.waited == 1

    async def timeout_wait(awaitable: object, timeout: float) -> object:
        del timeout
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise TimeoutError

    monkeypatch.setattr(process_control_module.asyncio, "wait_for", timeout_wait)
    with pytest.raises(RuntimeError, match="kill deadline"):
        await process_control_module._bounded_wait(
            WaitProcess(None),  # type: ignore[arg-type]
            1.0,
        )


@pytest.mark.asyncio
async def test_process_group_poll_signal_and_start_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_group_exists = process_control_module._group_exists
    existence = iter((True, False))
    monkeypatch.setattr(
        process_control_module,
        "_group_exists",
        lambda unused_pgid: next(existence),
    )

    async def no_sleep(unused_delay: float) -> None:
        del unused_delay

    monkeypatch.setattr(process_control_module.asyncio, "sleep", no_sleep)
    assert await process_control_module._wait_group_exit(41, 1.0) is True
    monkeypatch.setattr(
        process_control_module,
        "_group_exists",
        lambda unused_pgid: True,
    )
    assert await process_control_module._wait_group_exit(41, 0.0) is False
    monkeypatch.setattr(
        process_control_module,
        "_group_exists",
        lambda unused_pgid: False,
    )
    assert await process_control_module._wait_group_exit(41, 0.0) is True

    monkeypatch.setattr(
        process_control_module,
        "_group_exists",
        original_group_exists,
    )
    monkeypatch.setattr(
        process_control_module.os,
        "killpg",
        lambda unused_pgid, unused_signal: None,
    )
    assert process_control_module._group_exists(41) is True
    process_control_module._signal_group(41, signal.SIGTERM)

    def missing_group(unused_pgid: int, unused_signal: int) -> None:
        del unused_pgid
        del unused_signal
        raise ProcessLookupError

    monkeypatch.setattr(process_control_module.os, "killpg", missing_group)
    assert process_control_module._group_exists(41) is False
    process_control_module._signal_group(41, signal.SIGTERM)

    def inaccessible_group(unused_pgid: int, unused_signal: int) -> None:
        del unused_pgid
        del unused_signal
        raise PermissionError

    monkeypatch.setattr(process_control_module.os, "killpg", inaccessible_group)
    assert process_control_module._group_exists(41) is True

    with monkeypatch.context() as context:
        context.setattr(
            process_control_module.Path,
            "read_text",
            lambda *unused, **unused_values: " ".join(
                str(index) for index in range(22)
            ),
        )
        assert process_control_module._process_start(41) == "21"

    def unreadable(*unused: object, **unused_values: object) -> str:
        del unused
        del unused_values
        raise OSError

    with monkeypatch.context() as context:
        context.setattr(process_control_module.Path, "read_text", unreadable)
        context.setattr(
            process_control_module.subprocess,
            "run",
            lambda *unused, **unused_values: SimpleNamespace(
                returncode=0,
                stdout="ps-start",
            ),
        )
        assert process_control_module._process_start(41) == "ps-start"
        context.setattr(
            process_control_module.subprocess,
            "run",
            lambda *unused, **unused_values: SimpleNamespace(
                returncode=1,
                stdout="",
            ),
        )
        assert process_control_module._process_start(41) == "41"


class RequestProbe(dict[str, object]):
    def __init__(
        self,
        *,
        match_info: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.match_info = match_info or {}
        self.query = query or {}
        self.headers = headers or {}


def test_api_internal_error_and_direct_handler_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = api_module.create_app(SimpleNamespace(), "token")  # type: ignore[arg-type]
    error_middleware = app.middlewares[1]
    request = RequestProbe()
    request["correlation_id"] = "correlation"

    async def fail(unused_request: object) -> object:
        del unused_request
        raise RuntimeError("private detail")

    response = asyncio.run(error_middleware(request, fail))
    assert response.status == 500
    assert "private detail" not in response.text

    async def body(unused_request: object) -> dict[str, object]:
        del unused_request
        return {"value": "payload"}

    class SchedulerProbe:
        async def status(self, workspace: Path) -> dict[str, str]:
            return {"workspace": str(workspace)}

    class ServiceProbe:
        def __init__(self) -> None:
            self.scheduler = SchedulerProbe()
            self.store = SimpleNamespace(
                get_session=lambda unused: SimpleNamespace(worktree=str(tmp_path))
            )

        async def preview_route(
            self,
            session_id: str,
            payload: dict[str, object],
        ) -> dict[str, object]:
            return {"session_id": session_id, "payload": payload}

        def import_transfer(
            self,
            payload: dict[str, object],
        ) -> dict[str, object]:
            return {"payload": payload}

    service = ServiceProbe()
    monkeypatch.setattr(api_module, "_body", body)
    monkeypatch.setattr(api_module, "_service", lambda unused: service)

    route_request = RequestProbe(match_info={"session_id": "session"})
    route_response = asyncio.run(api_module._route_preview(route_request))
    assert json.loads(route_response.text)["route"]["session_id"] == "session"

    provider_request = RequestProbe(query={"workspace": str(tmp_path)})
    provider_response = asyncio.run(api_module._providers(provider_request))
    assert json.loads(provider_response.text)["providers"]["workspace"] == str(tmp_path)

    import_request = RequestProbe()
    import_response = asyncio.run(api_module._import_transfer(import_request))
    assert json.loads(import_response.text)["transfer"]["payload"] == {
        "value": "payload"
    }

    terminal_calls: list[Path] = []

    async def terminal(
        unused_request: object,
        worktree: Path,
    ) -> str:
        del unused_request
        terminal_calls.append(worktree)
        return "terminal"

    monkeypatch.setattr(api_module, "terminal_socket", terminal)
    terminal_request = RequestProbe(match_info={"session_id": "session"})
    assert asyncio.run(api_module._terminal(terminal_request)) == "terminal"
    assert terminal_calls == [tmp_path]


def test_api_daemon_signal_stream_and_cleanup_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoopProbe:
        def __init__(self) -> None:
            self.calls: list[signal.Signals] = []

        def add_signal_handler(
            self,
            signal_value: signal.Signals,
            unused_callback: object,
        ) -> None:
            del unused_callback
            self.calls.append(signal_value)
            raise NotImplementedError

    loop = LoopProbe()
    api_module._register_stop_signals(
        loop,  # type: ignore[arg-type]
        asyncio.Event(),
    )
    assert loop.calls == [signal.SIGINT, signal.SIGTERM]

    class ResponseProbe:
        def __init__(self, **unused: object) -> None:
            del unused

        async def prepare(self, unused_request: object) -> None:
            del unused_request

        async def write(self, unused_content: bytes) -> None:
            del unused_content

    stream_service = SimpleNamespace(
        store=SimpleNamespace(events=lambda *unused, **values: [])
    )
    monkeypatch.setattr(
        api_module,
        "_service",
        lambda unused: stream_service,
    )
    monkeypatch.setattr(api_module.web, "StreamResponse", ResponseProbe)
    monkeypatch.setattr(api_module, "STREAM_HEARTBEAT_LIMIT", 0)
    stream_request = RequestProbe(
        match_info={"session_id": "session"},
        query={},
    )
    assert asyncio.run(api_module._stream(stream_request)).__class__ is ResponseProbe

    with tempfile.TemporaryDirectory(
        prefix="harness-",
        dir="/tmp",
    ) as state_dir:
        harness_paths = paths(Path(state_dir))
        prepare_paths(harness_paths)
        with pytest.raises(ValueError, match="TCP port"):
            asyncio.run(
                api_module.run_daemon(
                    harness_paths,
                    tcp_host="127.0.0.1",
                    tcp_port=0,
                )
            )
        assert not harness_paths.socket.exists()


def test_service_worker_recovery_budget_lease_and_ui_boundaries(
    tmp_path: Path,
) -> None:
    harness_paths = paths(tmp_path / "default-state")
    prepare_paths(harness_paths)
    default_service = HarnessService(harness_paths)
    try:
        assert isinstance(default_service.workers, WorkerManager)
    finally:
        default_service.close()

    service = _service(tmp_path / "service")
    try:
        session = _create_direct(service, tmp_path / "service")
        service.store.update_session(
            session.session_id,
            lifecycle=Lifecycle.STOPPED,
        )
        workers = service.workers
        assert isinstance(workers, WorkerProbe)
        workers.sessions.clear()
        service.recover_workers()
        assert workers.sessions == []

        invalid_extensions = (
            {},
            {"reason": "test", "additional_seconds": True},
            {"reason": "test", "additional_seconds": 0},
            {"reason": "test", "additional_seconds": 3601},
            {"reason": "test", "additional_tokens": 300001},
            {"reason": "test", "allow_xhigh_once": "yes"},
            {"reason": "test"},
        )
        for payload in invalid_extensions:
            with pytest.raises(ValueError):
                service.extend_budget(session.session_id, payload)
        xhigh_command = service.store.enqueue_command(
            session.session_id,
            "message",
            {
                "text": "one bounded maximum-effort attempt",
                "provider": "codex",
                "effort": "xhigh",
            },
            "xhigh-command",
        )
        extended = service.extend_budget(
            session.session_id,
            {
                "reason": "test",
                "additional_tokens": 1,
                "allow_xhigh_once": True,
                "command_id": xhigh_command.command_id,
                "provider": "codex",
            },
            idempotency_key="xhigh-authorization",
        )
        assert extended["xhigh_authorizations"] == 1

        with pytest.raises(ValueError, match="provider"):
            service.create_process_lease({"provider": "other"})
        with pytest.raises(ValueError, match="interactive"):
            service.create_process_lease(
                {
                    "provider": "codex",
                    "execution_profile": "interactive",
                }
            )
        with pytest.raises(ValueError, match="action"):
            service.update_process_lease("lease", {"action": "other"})
        for pid in (True, 0):
            with pytest.raises(ValueError, match="pid"):
                service.update_process_lease(
                    "lease",
                    {"action": "attach", "pid": pid},
                )
        with pytest.raises(ValueError, match="pid_start"):
            service.update_process_lease(
                "lease",
                {"action": "attach", "pid": 1},
            )

        invalid_ui = (
            {"unknown": "value"},
            {"theme": 1},
            {"theme": "x" * 129},
        )
        for payload in invalid_ui:
            with pytest.raises(ValueError):
                service.set_ui_state(session.session_id, payload)
    finally:
        service.close()

    capacity_service = object.__new__(HarnessService)
    capacity_service.store = SimpleNamespace(
        latest_usage=lambda: {
            "codex": {
                "observed_at": service_module.datetime.datetime.now().isoformat(),
                "binding_percent": 0,
                "credits_engaged": False,
            }
        }
    )
    capacity_service._require_process_lease_capacity("codex", "unattended")

    negative_service = object.__new__(HarnessService)
    negative_service.store = SimpleNamespace(
        latest_usage=lambda: {
            "codex": {
                "observed_at": service_module.datetime.datetime.now().isoformat(),
                "binding_percent": -1,
                "credits_engaged": False,
            }
        }
    )
    with pytest.raises(SafetyGuardError, match="binding usage is invalid"):
        negative_service._require_process_lease_capacity("codex", "unattended")


def test_service_turn_and_budget_extension_boundaries(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        session = _create_direct(service, tmp_path)
        with pytest.raises(ValueError, match="per-turn permission mode"):
            service.submit_message(
                session.session_id,
                {"text": "Run one turn.", "permission_mode": "root"},
                idempotency_key="unsupported-mode",
            )
        with pytest.raises(ValueError, match="only one proof fault probe"):
            service.submit_message(
                session.session_id,
                {
                    "text": "Run one turn.",
                    "proof_fault_probe": None,
                    "proof_service_fault_probe": None,
                },
                idempotency_key="both-probes",
            )
        command = service.store.enqueue_command(
            session.session_id,
            "message",
            {"text": "Run one turn.", "effort": "xhigh"},
            "xhigh-turn",
        )
        with pytest.raises(ValueError, match="authorization provider is unsupported"):
            service.extend_budget(
                session.session_id,
                {
                    "reason": "one authorized turn",
                    "allow_xhigh_once": True,
                    "command_id": command.command_id,
                    "provider": "other",
                },
                idempotency_key="xhigh-provider",
            )
        with pytest.raises(ValueError, match="idempotency key is required"):
            service.extend_budget(
                session.session_id,
                {
                    "reason": "one authorized turn",
                    "allow_xhigh_once": True,
                    "command_id": command.command_id,
                    "provider": "codex",
                },
            )
    finally:
        service.close()


def test_process_lease_attachment_identity_is_immutable(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        session = _create_direct(service, tmp_path)
        lease = service.store.create_process_lease(
            session.session_id,
            "codex",
            "unattended",
            service_module._lease_expiry(),
        )
        attached = service.update_process_lease(
            str(lease["lease_id"]),
            {"action": "attach", "pid": 4242, "pid_start": "start-1"},
        )
        assert attached["state"] == "active"

        service.update_process_lease(
            str(lease["lease_id"]),
            {"action": "attach", "pid": 4242, "pid_start": "start-1"},
        )
        with pytest.raises(ConflictError, match="identity is immutable"):
            service.update_process_lease(
                str(lease["lease_id"]),
                {"action": "attach", "pid": 4343, "pid_start": "start-1"},
            )
    finally:
        service.close()


def test_session_creation_rolls_back_its_worktree_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    workspace = tmp_path / "workspace"
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
    (workspace / "file.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(workspace), "add", "file.txt"], check=True)
    subprocess.run(["git", "-C", str(workspace), "commit", "-qm", "initial"], check=True)
    payload = {
        "workspace": str(workspace),
        "direct": False,
        "execution_profile": "interactive",
    }
    try:
        original_ensure = service.store.ensure_session

        def failing_ensure(*arguments: object, **options: object):
            raise RuntimeError("durable session write failed")

        monkeypatch.setattr(service.store, "ensure_session", failing_ensure)
        with pytest.raises(RuntimeError, match="durable session write failed"):
            service.create_session(dict(payload))
        monkeypatch.setattr(service.store, "ensure_session", original_ensure)
        assert not list(service.paths.worktrees.glob("*"))

        created = service.create_session(dict(payload), idempotency_key="repeat")
        assert len(list(service.paths.worktrees.glob("*"))) == 1

        # A concurrent creator can win the race between the pre-check and
        # the durable insert, so the loser has to drop its own worktree.
        monkeypatch.setattr(
            service.store,
            "existing_ensured_session",
            lambda *arguments, **options: None,
        )
        repeated = service.create_session(dict(payload), idempotency_key="repeat")

        assert repeated.session_id == created.session_id
        assert len(list(service.paths.worktrees.glob("*"))) == 1
    finally:
        service.close()


def test_route_preview_respects_goal_bounds_and_metered_budgets(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    try:
        session = _create_direct(service, tmp_path)
        service.store.create_goal(
            create_goal(
                session.session_id,
                "Route inside the goal contract.",
                permitted_providers=("codex",),
                permitted_efforts=("low",),
            )
        )

        async def scenario() -> None:
            with pytest.raises(ValueError, match="metered budget must be positive"):
                await service.preview_route(
                    session.session_id,
                    {"metered_budget": 0},
                )

        asyncio.run(scenario())
    finally:
        service.close()


def test_service_reconciliation_approval_boundaries(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    try:
        session = _create_direct(
            service,
            tmp_path,
            permission_mode=PermissionMode.APPROVAL,
        )
        assert not service._reconciliation_restore_approved(
            session.session_id,
            "reconciliation",
            "stop",
            "",
        )
        service.store.update_session(
            session.session_id,
            permission_mode=PermissionMode.FULL,
        )
        assert not service._reconciliation_restore_approved(
            session.session_id,
            "reconciliation",
            "restore-pre-turn",
            "",
        )
        service.store.update_session(
            session.session_id,
            permission_mode=PermissionMode.APPROVAL,
        )

        approval_id = service.store.create_approval(
            session.session_id,
            "",
            "other",
            "reconciliation.restore",
            "approve",
            [],
        )
        with pytest.raises(ConflictError, match="authorize"):
            service._reconciliation_restore_approved(
                session.session_id,
                "reconciliation",
                "restore-pre-turn",
                approval_id,
            )

        pending_id = service.store.create_approval(
            session.session_id,
            "",
            "reconciliation",
            "reconciliation.restore",
            "approve",
            [],
        )
        assert (
            service._pending_reconciliation_approval(
                session.session_id,
                "reconciliation",
            )
            == pending_id
        )
        with pytest.raises(HarnessError, match="pending"):
            service._reconciliation_restore_approved(
                session.session_id,
                "reconciliation",
                "restore-pre-turn",
                pending_id,
            )
        service.store.resolve_approval(
            pending_id,
            {"decision": "decline"},
        )
        with pytest.raises(HarnessError, match="not approved"):
            service._reconciliation_restore_approved(
                session.session_id,
                "reconciliation",
                "restore-pre-turn",
                pending_id,
            )
        approved_id = service.store.create_approval(
            session.session_id,
            "",
            "reconciliation",
            "reconciliation.restore",
            "approve",
            [],
        )
        service.store.resolve_approval(
            approved_id,
            {"decision": "approve"},
        )
        assert service._reconciliation_restore_approved(
            session.session_id,
            "reconciliation",
            "restore-pre-turn",
            approved_id,
        )

        invalid_store = service.store
        original_approval = invalid_store.approval
        invalid_store.approval = lambda unused: {  # type: ignore[method-assign]
            "session_id": session.session_id,
            "provider_request_id": "reconciliation",
            "kind": "reconciliation.restore",
            "status": "resolved",
            "decision": "invalid",
        }
        with pytest.raises(ConflictError, match="decision"):
            service._reconciliation_restore_approved(
                session.session_id,
                "reconciliation",
                "restore-pre-turn",
                pending_id,
            )
        invalid_store.approval = original_approval  # type: ignore[method-assign]

        reconciliation_id = new_uuid()
        with pytest.raises(ValueError, match="digest"):
            asyncio.run(
                service._resolve_reconciliation(
                    reconciliation_id,
                    {"decision": "stop"},
                )
            )
        with pytest.raises(ValueError, match="audit"):
            asyncio.run(
                service._resolve_reconciliation(
                    reconciliation_id,
                    {
                        "decision": "stop",
                        "observed_workspace_digest": "digest",
                        "audit": [],
                    },
                )
            )
    finally:
        service.close()


def test_service_checkpoint_fork_transfer_and_timeout_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Session(
        session_id=new_uuid(),
        name="source",
        workspace=str(tmp_path),
        worktree=str(tmp_path),
        lifecycle=Lifecycle.RUNNING,
        attention=Attention.IDLE,
        permission_mode=PermissionMode.APPROVAL,
        active_provider="codex",
        model="",
        effort="",
        goal_id="",
        owner_host="host",
        owner_epoch=1,
        created_at="now",
        updated_at="now",
    )
    forked = Session(
        **{
            **source.as_dict(),
            "session_id": new_uuid(),
            "name": "source fork",
        }
    )
    created_payloads: list[dict[str, Any]] = []

    def get_session(session_id: str) -> Session:
        if session_id == forked.session_id:
            return forked
        return source

    store = SimpleNamespace(
        get_session=get_session,
        goal_for_session=lambda unused: None,
        session_safety=lambda unused: {"profile": ""},
        checkpoints=lambda unused: [],
        append_event=lambda *unused, **values: None,
        last_sequence=lambda unused: 1,
    )
    service = object.__new__(HarnessService)
    service.store = store
    service.blobs = SimpleNamespace()
    service.checkpoint = lambda unused: {"checkpoint_id": "checkpoint"}  # type: ignore[method-assign]

    def create(payload: dict[str, Any]) -> Session:
        created_payloads.append(payload)
        return forked

    service.create_session = create  # type: ignore[method-assign]
    result = service.fork_session(
        source.session_id,
        {"external_ref": {"orchestrator": "test", "job_id": "job"}},
    )
    assert result == forked
    assert created_payloads[0]["name"] == "source fork"
    assert created_payloads[0]["execution_profile"] == "unattended"
    assert created_payloads[0]["external_ref"]["job_id"] == "job"

    timeout_service = object.__new__(HarnessService)
    timeout_service.store = SimpleNamespace(
        get_command=lambda unused: SimpleNamespace(
            status=CommandStatus.DISPATCHING,
            as_dict=lambda: {"status": "dispatching"},
        )
    )
    assert (
        asyncio.run(timeout_service.wait_for_command("command", timeout=-1))["status"]
        == "dispatching"
    )
    assert service_module._message_session_name("x" * 80).endswith("…")


def test_service_import_transfer_validation_and_success_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = new_uuid()
    imported = Session(
        session_id=session_id,
        name="imported",
        workspace=str(tmp_path),
        worktree=str(tmp_path / "worktree"),
        lifecycle=Lifecycle.PAUSED,
        attention=Attention.IDLE,
        permission_mode=PermissionMode.APPROVAL,
        active_provider="",
        model="",
        effort="",
        goal_id="",
        owner_host="host",
        owner_epoch=2,
        created_at="now",
        updated_at="now",
    )

    class BlobProbe:
        def __init__(self) -> None:
            self.digest = "digest"

        def put(self, unused_content: bytes) -> str:
            return self.digest

    class StoreProbe:
        def __init__(self) -> None:
            self.safety: list[str] = []
            self.events: list[str] = []

        def import_session(self, *unused: object, **values: object) -> Session:
            del unused, values
            return imported

        def set_session_safety(self, unused: str, profile: str) -> None:
            self.safety.append(profile)

        def checkpoints(self, unused: str) -> list[str]:
            return ["checkpoint"]

        def append_event(
            self,
            unused: str,
            event_type: str,
            **values: object,
        ) -> None:
            del values
            self.events.append(event_type)

    service = object.__new__(HarnessService)
    service.paths = SimpleNamespace(worktrees=tmp_path / "worktrees")
    service.machine_keys = SimpleNamespace(encryption_private="private")
    service.blobs = BlobProbe()
    service.store = StoreProbe()
    monkeypatch.setattr(service_module, "host_id", lambda: "destination")
    monkeypatch.setattr(
        service_module,
        "create_worktree",
        lambda *unused, **values: tmp_path / "worktree",
    )
    restored: list[object] = []
    monkeypatch.setattr(
        service_module,
        "restore_checkpoint",
        lambda *values: restored.append(values),
    )

    base = {
        "schema": "p13i/agent-harness/session-transfer/v1",
        "destination_host": "destination",
        "source_host": "source",
        "owner_epoch": 2,
        "export": {
            "session": {
                "session_id": session_id,
                "workspace": str(tmp_path),
            },
            "safety": [],
        },
        "blobs": {},
    }
    invalid_values = (
        {**base, "schema": "invalid"},
        {**base, "destination_host": "other"},
        {**base, "export": []},
        {**base, "blobs": []},
        {**base, "blobs": {1: "value"}},
        {**base, "blobs": {"digest": "!"}},
        {**base, "export": {}},
    )
    for opened in invalid_values:
        monkeypatch.setattr(
            service_module,
            "open_transfer",
            lambda *unused, value=opened, **values: value,
        )
        with pytest.raises(ValueError):
            service.import_transfer(
                {
                    "envelope": "eA==",
                    "source_signing_public": "public",
                }
            )

    service.blobs.digest = "other"
    mismatch = {**base, "blobs": {"digest": "eA=="}}
    monkeypatch.setattr(
        service_module,
        "open_transfer",
        lambda *unused, **values: mismatch,
    )
    with pytest.raises(ValueError, match="digest"):
        service.import_transfer({"envelope": "eA==", "source_signing_public": "public"})

    service.blobs.digest = "digest"
    success = {**base, "blobs": {"digest": "eA=="}}
    monkeypatch.setattr(
        service_module,
        "open_transfer",
        lambda *unused, **values: success,
    )
    result = service.import_transfer(
        {"envelope": "eA==", "source_signing_public": "public"}
    )
    assert result["session"]["session_id"] == session_id
    assert service.store.safety == ["unattended"]
    assert restored


def test_service_creation_checkpoint_and_transfer_success_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    try:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        subprocess.run(
            ["git", "-C", str(workspace), "init", "-q"],
            check=True,
        )
        monkeypatch.setattr(
            service.store,
            "existing_ensured_session",
            lambda *unused, **values: None,
        )

        def existing(
            candidate: Session,
            *unused: object,
            **values: object,
        ) -> tuple[Session, bool]:
            del unused, values
            return candidate, False

        monkeypatch.setattr(service.store, "ensure_session", existing)
        replay = service.create_session(
            {
                "workspace": str(workspace),
                "direct": True,
                "execution_profile": "interactive",
            }
        )
        assert replay.workspace == str(workspace)

        monkeypatch.undo()
        session = _create_direct(service, tmp_path / "checkpoint")
        service.store.update_session(
            session.session_id,
            active_provider="codex",
        )
        now = utc_now()
        codex_attempt = ProviderAttempt(
            attempt_id=new_uuid(),
            session_id=session.session_id,
            provider="codex",
            native_session_id="native",
            model="model",
            effort="low",
            auth_mode="subscription",
            status="complete",
            started_at=now,
            ended_at=now,
        )
        other_attempt = ProviderAttempt(
            **{
                **codex_attempt.as_dict(),
                "attempt_id": new_uuid(),
                "provider": "claude",
                "started_at": now + "z",
            }
        )
        service.store.create_attempt(codex_attempt)
        service.store.create_attempt(other_attempt)
        checkpoint = Checkpoint(
            checkpoint_id=new_uuid(),
            session_id=session.session_id,
            sequence=0,
            provider="codex",
            native_session_id="native",
            base_commit="",
            patch_digest="",
            untracked_digest="",
            context_digest="",
            created_at=now,
        )
        monkeypatch.setattr(
            service_module,
            "compile_context",
            lambda *unused, **values: SimpleNamespace(text="context"),
        )
        monkeypatch.setattr(
            service_module,
            "workspace_instructions",
            lambda unused: (),
        )
        monkeypatch.setattr(
            service_module,
            "workspace_summary",
            lambda unused: "summary",
        )
        monkeypatch.setattr(
            service_module,
            "checkpoint_workspace",
            lambda *unused, **values: checkpoint,
        )
        assert service.checkpoint(session.session_id)["native_session_id"] == ("native")

        monkeypatch.setattr(
            service.store,
            "export_session",
            lambda unused: {
                "events": [{"blob_digest": "digest"}],
                "checkpoints": [],
            },
        )
        monkeypatch.setattr(
            service.blobs,
            "get",
            lambda unused: b"content",
        )
        monkeypatch.setattr(
            service_module,
            "seal_transfer",
            lambda *unused, **values: b"sealed",
        )
        transfer = service.create_transfer(
            session.session_id,
            {
                "destination_host": "other",
                "destination_encryption_public": "public",
            },
        )
        assert transfer["session_id"] == session.session_id
    finally:
        service.close()


def _machines_session(
    tmp_path: Path,
    *,
    orchestrator: str = "p13i/machines/agent-harness-proof",
) -> Session:
    return Session(
        session_id=new_uuid(),
        name="machines",
        workspace=str(tmp_path),
        worktree=str(tmp_path),
        lifecycle=Lifecycle.RUNNING,
        attention=Attention.IDLE,
        permission_mode=PermissionMode.APPROVAL,
        active_provider="",
        model="",
        effort="",
        goal_id="",
        owner_host="host",
        owner_epoch=1,
        created_at="now",
        updated_at="now",
        external_ref={"orchestrator": orchestrator, "job_id": "job-1"},
    )


def _fault_probe_request(
    session: Session,
    idempotency_key: str,
    *,
    schema: str,
    agent_role: str,
    stage: str,
    capability: str,
    probe_field: str,
) -> tuple[dict[str, Any], str]:
    authorization = {
        "schema": schema,
        "external_ref": session.external_ref,
        "idempotency_key": idempotency_key,
        "stage": stage,
        "provider": "claude",
        "agent_role": agent_role,
    }
    digest = normalized_digest(authorization)
    payload = {
        "turn_ref": {"step_id": "probe", "agent_role": agent_role},
        "required_capabilities": [capability],
        probe_field: {
            "stage": stage,
            "provider": "claude",
            "authorization_digest": digest,
            "authorization": authorization,
        },
    }
    return payload, digest


def _probe_goal(session: Session, constraint: str) -> Goal:
    return create_goal(
        session.session_id,
        "Prove the bounded fault probe.",
        constraints=(constraint,),
    )


def test_proof_fault_probe_requests_fail_closed_on_every_binding(
    tmp_path: Path,
) -> None:
    session = _machines_session(tmp_path)
    key = "fault-probe-key"
    payload, digest = _fault_probe_request(
        session,
        key,
        schema="p13i/machines/provider-fault-authorization/v1",
        agent_role="proof-fault-probe",
        stage="after-lease-before-acceptance",
        capability="proof-fault-barrier",
        probe_field="proof_fault_probe",
    )
    goal = _probe_goal(session, "proof-fault-authorization-sha256:" + digest)

    service_module._validate_proof_fault_probe(
        session, goal, payload, key, SUPPORTED_PROVIDERS
    )
    service_module._validate_proof_fault_probe(
        session, goal, {}, key, SUPPORTED_PROVIDERS
    )

    def rejected(message: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        request = copy.deepcopy(payload)
        mutate(request)
        with pytest.raises(ValueError, match=message):
            service_module._validate_proof_fault_probe(
                session,
                goal,
                request,
                key,
                SUPPORTED_PROVIDERS,
            )

    rejected(
        "must be an object",
        lambda item: item.__setitem__("proof_fault_probe", "probe"),
    )
    rejected(
        "requires automatic routing",
        lambda item: item.__setitem__("provider", "claude"),
    )
    rejected(
        "requires a managed turn reference",
        lambda item: item.__setitem__("turn_ref", "step"),
    )
    rejected(
        "role is unauthorized",
        lambda item: item["turn_ref"].__setitem__("agent_role", "implementer"),
    )
    rejected(
        "requires capabilities",
        lambda item: item.__setitem__("required_capabilities", "barrier"),
    )
    rejected(
        "requires proof-fault-barrier",
        lambda item: item.__setitem__("required_capabilities", []),
    )
    rejected(
        "stage is unsupported",
        lambda item: item["proof_fault_probe"].__setitem__("stage", "before"),
    )
    rejected(
        "provider is unsupported",
        lambda item: item["proof_fault_probe"].__setitem__("provider", "other"),
    )
    rejected(
        "authorization digest is invalid",
        lambda item: item["proof_fault_probe"].__setitem__(
            "authorization_digest",
            "short",
        ),
    )
    rejected(
        "authorization digest is invalid",
        lambda item: item["proof_fault_probe"].__setitem__(
            "authorization_digest",
            "z" * 64,
        ),
    )
    rejected(
        "authorization is required",
        lambda item: item["proof_fault_probe"].__setitem__("authorization", []),
    )
    rejected(
        "authorization does not match",
        lambda item: item["proof_fault_probe"]["authorization"].__setitem__(
            "provider",
            "codex",
        ),
    )

    with pytest.raises(ValueError, match="requires an idempotency key"):
        service_module._validate_proof_fault_probe(
            session, goal, payload, "", SUPPORTED_PROVIDERS
        )
    foreign = _machines_session(tmp_path, orchestrator="operator")
    with pytest.raises(ValueError, match="p13i/machines ownership"):
        service_module._validate_proof_fault_probe(
            foreign, goal, payload, key, SUPPORTED_PROVIDERS
        )
    with pytest.raises(ValueError, match="is not precommitted"):
        service_module._validate_proof_fault_probe(
            session, None, payload, key, SUPPORTED_PROVIDERS
        )


def test_service_fault_probe_requests_fail_closed_on_every_binding(
    tmp_path: Path,
) -> None:
    session = _machines_session(tmp_path)
    key = "service-fault-probe-key"
    payload, digest = _fault_probe_request(
        session,
        key,
        schema="p13i/machines/service-fault-authorization/v1",
        agent_role="proof-service-fault-probe",
        stage="after-acceptance-before-terminal",
        capability="proof-service-fault-barrier",
        probe_field="proof_service_fault_probe",
    )
    goal = _probe_goal(
        session,
        "proof-service-fault-authorization-sha256:" + digest,
    )

    service_module._validate_service_fault_probe(
        session, goal, payload, key, SUPPORTED_PROVIDERS
    )
    service_module._validate_service_fault_probe(
        session, goal, {}, key, SUPPORTED_PROVIDERS
    )

    def rejected(message: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        request = copy.deepcopy(payload)
        mutate(request)
        with pytest.raises(ValueError, match=message):
            service_module._validate_service_fault_probe(
                session,
                goal,
                request,
                key,
                SUPPORTED_PROVIDERS,
            )

    rejected(
        "must be an object",
        lambda item: item.__setitem__("proof_service_fault_probe", "probe"),
    )
    rejected(
        "requires a managed turn reference",
        lambda item: item.__setitem__("turn_ref", "step"),
    )
    rejected(
        "role is unauthorized",
        lambda item: item["turn_ref"].__setitem__("agent_role", "implementer"),
    )
    rejected(
        "requires capabilities",
        lambda item: item.__setitem__("required_capabilities", "barrier"),
    )
    rejected(
        "requires proof-service-fault-barrier",
        lambda item: item.__setitem__("required_capabilities", []),
    )
    rejected(
        "stage is unsupported",
        lambda item: item["proof_service_fault_probe"].__setitem__(
            "stage",
            "before",
        ),
    )
    rejected(
        "provider is unsupported",
        lambda item: item["proof_service_fault_probe"].__setitem__(
            "provider",
            "other",
        ),
    )
    rejected(
        "provider does not match",
        lambda item: item.__setitem__("provider", "codex"),
    )
    rejected(
        "authorization digest is invalid",
        lambda item: item["proof_service_fault_probe"].__setitem__(
            "authorization_digest",
            "short",
        ),
    )
    rejected(
        "authorization digest is invalid",
        lambda item: item["proof_service_fault_probe"].__setitem__(
            "authorization_digest",
            "z" * 64,
        ),
    )
    rejected(
        "authorization is required",
        lambda item: item["proof_service_fault_probe"].__setitem__(
            "authorization",
            [],
        ),
    )
    rejected(
        "authorization does not match",
        lambda item: item["proof_service_fault_probe"][
            "authorization"
        ].__setitem__("provider", "codex"),
    )
    rejected(
        "authorization digest does not match",
        lambda item: item["proof_service_fault_probe"].__setitem__(
            "authorization_digest",
            "a" * 64,
        ),
    )

    with pytest.raises(ValueError, match="requires an idempotency key"):
        service_module._validate_service_fault_probe(
            session, goal, payload, "", SUPPORTED_PROVIDERS
        )
    foreign = _machines_session(tmp_path, orchestrator="operator")
    with pytest.raises(ValueError, match="p13i/machines ownership"):
        service_module._validate_service_fault_probe(
            foreign,
            goal,
            payload,
            key,
            SUPPORTED_PROVIDERS,
        )
    with pytest.raises(ValueError, match="is not precommitted"):
        service_module._validate_service_fault_probe(
            session, None, payload, key, SUPPORTED_PROVIDERS
        )


def test_machines_goal_envelope_requires_every_typed_dimension() -> None:
    complete = {
        "goal_kind": "finite",
        "predicates": [{"type": "proof"}],
        "constraints": ["bounded"],
        "milestones": [{"milestone_id": "one"}],
        "budgets": {},
        "permitted_providers": ["codex"],
        "permitted_efforts": ["low"],
        "max_concurrency": 1,
        "completion_policy": "evidence-all",
        "incident_policy": "recover-then-pause",
    }
    budgets = {
        "seconds": 1,
        "turns": 1,
        "tokens": 1,
        "context_tokens": 1,
        "output_tokens": 1,
        "tool_calls": 1,
        "dollars": 0,
        "attempts": 1,
        "child_agents": 0,
    }

    def envelope(**overrides: Any) -> None:
        values: dict[str, Any] = {
            "objective": "Prove the stage.",
            "constraints": complete["constraints"],
            "predicates": complete["predicates"],
            "milestones": complete["milestones"],
            "permitted_providers": complete["permitted_providers"],
            "permitted_efforts": complete["permitted_efforts"],
            "budgets": budgets,
            "supported_providers": frozenset({"claude", "codex", "kimi"}),
        }
        values.update(overrides)
        service_module._require_machines_goal_envelope(complete, **values)

    envelope()
    with pytest.raises(ValueError, match="complete typed goal envelope"):
        envelope(objective="")
    with pytest.raises(ValueError, match="typed evidence predicates"):
        envelope(predicates=["bare"])
    with pytest.raises(ValueError, match="at least one goal milestone"):
        envelope(milestones=[])
    with pytest.raises(ValueError, match="typed goal milestones"):
        envelope(milestones=["bare"])
    with pytest.raises(ValueError, match="explicit routing bounds"):
        envelope(permitted_efforts=[])
    with pytest.raises(ValueError, match="bounded budget dimensions"):
        envelope(budgets={"seconds": 1})


def test_authorization_receipts_must_be_retained_and_exact() -> None:
    receipt = {"scope": "test"}
    digest = normalized_digest(receipt)

    service_module._require_retained_authorization_receipt(
        {"receipt": receipt},
        digest,
    )
    with pytest.raises(ValueError, match="must retain its source receipt"):
        service_module._require_retained_authorization_receipt({}, digest)
    with pytest.raises(ValueError, match="receipt digest does not match"):
        service_module._require_retained_authorization_receipt(
            {"receipt": receipt},
            "a" * 64,
        )


def test_goal_promotion_authorization_binds_every_field() -> None:
    session_id = new_uuid()
    previous_goal_id = new_uuid()
    receipt = {"scope": "promotion"}
    authorization = {
        "schema": "p13i/agent-harness/goal-promotion-authorization/v1",
        "session_id": session_id,
        "from_goal_id": previous_goal_id,
        "stage": "tier-24h",
        "receipt": receipt,
        "receipt_sha256": normalized_digest(receipt),
        "budget_increases": {"seconds": 60.0},
        "next_goal_contract_digest": "b" * 64,
    }

    def require(**overrides: Any) -> None:
        values: dict[str, Any] = {
            "session_id": session_id,
            "previous_goal_id": previous_goal_id,
            "stage": "tier-24h",
            "budget_increases": {"seconds": 60.0},
            "next_goal_contract_digest": "b" * 64,
        }
        values.update(overrides)
        service_module._require_promotion_authorization(
            dict(authorization),
            **values,
        )

    require()
    for message, name, value in (
        ("authorization schema is unsupported", "schema", "other"),
        ("authorization session does not match", "session_id", new_uuid()),
        ("authorization source does not match", "from_goal_id", new_uuid()),
        ("authorization stage does not match", "stage", "tier-48h"),
        ("authorization receipt is invalid", "receipt_sha256", "short"),
    ):
        altered = dict(authorization)
        altered[name] = value
        with pytest.raises(ValueError, match=message):
            service_module._require_promotion_authorization(
                altered,
                session_id=session_id,
                previous_goal_id=previous_goal_id,
                stage="tier-24h",
                budget_increases={"seconds": 60.0},
                next_goal_contract_digest="b" * 64,
            )
    with pytest.raises(ValueError, match="budget does not match"):
        require(budget_increases={"seconds": 120.0})
    with pytest.raises(ValueError, match="contract does not match"):
        require(next_goal_contract_digest="c" * 64)


def test_contract_adoption_authorization_binds_every_field() -> None:
    session_id = new_uuid()
    external_ref = {"orchestrator": "p13i/machines/proof", "job_id": "job-1"}
    receipt = {"scope": "adoption"}
    authorization = {
        "schema": ("p13i/agent-harness/session-contract-adoption-authorization/v1"),
        "session_id": session_id,
        "external_ref": external_ref,
        "previous_goal_digest": "d" * 64,
        "goal_envelope_digest": "e" * 64,
        "receipt": receipt,
        "receipt_sha256": normalized_digest(receipt),
    }

    service_module._require_contract_adoption_authorization(
        dict(authorization),
        session_id=session_id,
        external_ref=external_ref,
        previous_goal_digest="d" * 64,
        goal_envelope_digest="e" * 64,
    )
    for message, name, value in (
        ("authorization schema is unsupported", "schema", "other"),
        ("authorization session does not match", "session_id", new_uuid()),
        ("external reference does not match", "external_ref", {}),
        ("previous goal does not match", "previous_goal_digest", "f" * 64),
        ("goal envelope does not match", "goal_envelope_digest", "0" * 64),
        ("authorization receipt is invalid", "receipt_sha256", "short"),
    ):
        altered = dict(authorization)
        altered[name] = value
        with pytest.raises(ValueError, match=message):
            service_module._require_contract_adoption_authorization(
                altered,
                session_id=session_id,
                external_ref=external_ref,
                previous_goal_digest="d" * 64,
                goal_envelope_digest="e" * 64,
            )


def _transition_policy(
    session: Session,
    next_turn_ref: dict[str, str],
    next_command_digest: str,
) -> dict[str, Any]:
    return {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": session.session_id,
        "external_ref": session.external_ref,
        "epoch_id": "epoch-1",
        "allowed_agent_roles": ["verifier"],
        "allowed_step_prefixes": ["verify"],
        "max_transitions": 1,
        "transitions": [
            {
                "sequence": 1,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": next_command_digest,
            }
        ],
    }


def test_dispatch_transition_policies_fail_closed_outside_their_contract(
    tmp_path: Path,
) -> None:
    session = _machines_session(tmp_path)
    next_turn_ref = {"step_id": "verify", "agent_role": "verifier"}
    next_command_digest = "a" * 64
    policy = _transition_policy(session, next_turn_ref, next_command_digest)
    policy_sha256 = normalized_digest(policy)
    goal = create_goal(
        session.session_id,
        "Advance one exact stage.",
        constraints=(
            "dispatch-generation-transition-policy-sha256:" + policy_sha256,
            "dispatch-generation-transition-epoch:epoch-1",
        ),
    )
    store = SimpleNamespace(goal_for_session=lambda unused: goal)

    def require(
        current: dict[str, Any],
        *,
        sequence: int = 1,
        retained: bool = False,
        current_store: object = None,
    ) -> None:
        service_module._require_dispatch_transition_policy(
            current_store or store,
            session_id=session.session_id,
            session=session,
            policy=current,
            policy_sha256=policy_sha256,
            next_turn_ref=next_turn_ref,
            transition_sequence=sequence,
            next_command_digest=next_command_digest,
            retained_policy=retained,
        )

    require(policy)
    require(policy, retained=True)

    def rejected(
        message: str,
        mutate: Callable[[dict[str, Any]], None],
        *,
        sequence: int = 1,
    ) -> None:
        current = copy.deepcopy(policy)
        mutate(current)
        with pytest.raises(ValueError, match=message):
            require(current, sequence=sequence)

    rejected("policy schema is unsupported", lambda item: item.update(schema="x"))
    rejected(
        "policy session does not match",
        lambda item: item.update(session_id=new_uuid()),
    )
    rejected(
        "policy orchestrator does not match",
        lambda item: item.update(external_ref={}),
    )
    rejected("policy epoch is invalid", lambda item: item.update(epoch_id="a b"))
    rejected(
        "policy roles are invalid",
        lambda item: item.update(allowed_agent_roles=[]),
    )
    rejected(
        "policy step prefixes are invalid",
        lambda item: item.update(allowed_step_prefixes=[]),
    )
    rejected(
        "policy limit is invalid",
        lambda item: item.update(max_transitions=True),
    )
    rejected(
        "role is outside policy",
        lambda item: item.update(allowed_agent_roles=["publisher"]),
    )
    rejected(
        "step is outside policy",
        lambda item: item.update(allowed_step_prefixes=["publish"]),
    )
    rejected("exceeds policy limit", lambda item: item, sequence=2)
    rejected(
        "policy stages are invalid",
        lambda item: item.update(transitions=[]),
    )
    rejected(
        "policy limit does not match stages",
        lambda item: item.update(max_transitions=2),
    )
    rejected(
        "policy stage is invalid",
        lambda item: item["transitions"].__setitem__(0, "bare"),
    )
    rejected(
        "policy stages are not ordered",
        lambda item: item["transitions"][0].__setitem__("sequence", 2),
    )
    rejected(
        "policy stage role is invalid",
        lambda item: item["transitions"][0]["next_turn_ref"].__setitem__(
            "agent_role",
            "publisher",
        ),
    )
    rejected(
        "policy stage step is invalid",
        lambda item: item["transitions"][0]["next_turn_ref"].__setitem__(
            "step_id",
            "publish",
        ),
    )
    rejected(
        "policy command digest is invalid",
        lambda item: item["transitions"][0].__setitem__(
            "next_command_digest",
            "b" * 64,
        ),
    )

    two_stage = copy.deepcopy(policy)
    two_stage["max_transitions"] = 2
    two_stage["transitions"].append(
        {
            "sequence": 2,
            "next_turn_ref": next_turn_ref,
            "next_command_digest": next_command_digest,
        }
    )
    two_stage_sha256 = normalized_digest(two_stage)
    two_stage_goal = create_goal(
        session.session_id,
        "Advance two exact stages.",
        constraints=(
            "dispatch-generation-transition-policy-sha256:" + two_stage_sha256,
            "dispatch-generation-transition-epoch:epoch-1",
        ),
    )
    two_stage_store = SimpleNamespace(goal_for_session=lambda unused: two_stage_goal)

    def two_stage_rejected(
        message: str,
        mutate: Callable[[dict[str, Any]], None],
    ) -> None:
        current = copy.deepcopy(two_stage)
        mutate(current)
        with pytest.raises(ValueError, match=message):
            service_module._require_dispatch_transition_policy(
                two_stage_store,
                session_id=session.session_id,
                session=session,
                policy=current,
                policy_sha256=two_stage_sha256,
                next_turn_ref=next_turn_ref,
                transition_sequence=1,
                next_command_digest=next_command_digest,
                retained_policy=False,
            )

    two_stage_rejected(
        "policy stage is invalid",
        lambda item: item["transitions"].__setitem__(1, "bare"),
    )
    two_stage_rejected(
        "policy stages are not ordered",
        lambda item: item["transitions"][1].__setitem__("sequence", 3),
    )

    stage_mismatch = copy.deepcopy(policy)
    stage_mismatch["allowed_agent_roles"] = ["verifier", "publisher"]
    stage_mismatch["allowed_step_prefixes"] = ["verify", "publish"]
    stage_mismatch["transitions"][0]["next_turn_ref"] = {
        "step_id": "publish",
        "agent_role": "publisher",
    }
    with pytest.raises(ValueError, match="stage is outside policy"):
        require(stage_mismatch, retained=True)

    with pytest.raises(ValueError, match="requires an active goal"):
        require(policy, current_store=SimpleNamespace(goal_for_session=lambda _: None))
    unauthorized = create_goal(session.session_id, "Advance without authority.")
    with pytest.raises(ValueError, match="policy is not goal-authorized"):
        require(
            policy,
            current_store=SimpleNamespace(goal_for_session=lambda _: unauthorized),
        )
    epochless = create_goal(
        session.session_id,
        "Advance without an epoch.",
        constraints=(
            "dispatch-generation-transition-policy-sha256:" + policy_sha256,
        ),
    )
    with pytest.raises(ValueError, match="epoch is not goal-authorized"):
        require(
            policy,
            current_store=SimpleNamespace(goal_for_session=lambda _: epochless),
        )


def test_session_creation_intents_fail_closed_on_unusable_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    creation_input = {"workspace": str(tmp_path), "direct": True}
    session_id = new_uuid()

    reserved, intent_path = service_module._reserve_creation_intent(
        harness_paths,
        "creation-key",
        creation_input,
        session_id,
    )
    assert reserved == session_id
    assert intent_path is not None

    repeated, repeated_path = service_module._reserve_creation_intent(
        harness_paths,
        "creation-key",
        creation_input,
        new_uuid(),
    )
    assert repeated == session_id
    assert repeated_path == intent_path

    with pytest.raises(ConflictError, match="idempotency key was reused"):
        service_module._reserve_creation_intent(
            harness_paths,
            "creation-key",
            {"workspace": str(tmp_path), "direct": False},
            session_id,
        )
    with pytest.raises(ConflictError, match="idempotency key was reused"):
        service_module._complete_creation_intent(
            harness_paths,
            "creation-key",
            {"workspace": str(tmp_path), "direct": False},
            session_id,
        )
    with pytest.raises(ConflictError, match="intent identity changed"):
        service_module._complete_creation_intent(
            harness_paths,
            "creation-key",
            creation_input,
            new_uuid(),
        )

    service_module._complete_creation_intent(
        harness_paths,
        "",
        creation_input,
        session_id,
    )
    service_module._complete_creation_intent(
        harness_paths,
        "creation-key",
        creation_input,
        session_id,
    )
    assert not intent_path.exists()

    def failing_fsync(descriptor: int) -> None:
        del descriptor
        raise OSError("fsync failed")

    monkeypatch.setattr(service_module.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        service_module._reserve_creation_intent(
            harness_paths,
            "unstable-key",
            creation_input,
            new_uuid(),
        )
    monkeypatch.undo()
    assert not service_module._creation_intent_path(
        harness_paths,
        "unstable-key",
    ).exists()


def test_creation_intent_documents_are_validated_before_use(
    tmp_path: Path,
) -> None:
    linked = tmp_path / "linked.json"
    linked.symlink_to(tmp_path / "absent.json")
    with pytest.raises(ValueError, match="intent is invalid"):
        service_module._read_creation_intent(linked)

    oversized = tmp_path / "oversized.json"
    oversized.write_text("{}" + " " * (64 * 1024), encoding="utf-8")
    with pytest.raises(ValueError, match="intent is invalid"):
        service_module._read_creation_intent(oversized)

    with pytest.raises(ConflictError, match="intent is unreadable"):
        service_module._read_creation_intent(tmp_path / "absent.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConflictError, match="intent is unreadable"):
        service_module._read_creation_intent(malformed)

    listed = tmp_path / "listed.json"
    listed.write_text("[]", encoding="utf-8")
    with pytest.raises(ConflictError, match="intent is invalid"):
        service_module._read_creation_intent(listed)

    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text('{"schema": "other"}', encoding="utf-8")
    with pytest.raises(ConflictError, match="intent is incompatible"):
        service_module._read_creation_intent(incompatible)


def test_worker_supervision_status_survives_a_corrupt_expectation(
    tmp_path: Path,
) -> None:
    manager = WorkerManager(paths(tmp_path / "state"))

    assert manager.status()["status"] == "initializing"

    service = HarnessService(
        paths(tmp_path / "service-state"),
        worker_manager=WorkerProbe(),
    )
    try:
        service._worker_supervision = {"expected_sessions": "session-1"}
        service.record_worker_supervision_failure(RuntimeError("supervision"))
        assert service._worker_supervision["expected_sessions"] == []
        assert service._worker_supervision["status"] == "failed"
    finally:
        service.close()


def test_optional_numbers_reject_nonfinite_values() -> None:
    assert service_module._optional_number(True) is None
    assert service_module._optional_number("2") is None
    assert service_module._optional_number(2) == 2.0
    with pytest.raises(ValueError, match="numeric value must be finite"):
        service_module._optional_number(float("inf"))


def test_workspace_link_normalization_ignores_current_directory_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundaryPath:
        def __init__(self, *parts: str) -> None:
            self.parts = parts

        @property
        def parent(self) -> BoundaryPath:
            return self

        def is_absolute(self) -> bool:
            return False

        def __truediv__(self, unused: object) -> BoundaryPath:
            return BoundaryPath("directory", ".", "target")

    monkeypatch.setattr(workspace_module, "PurePosixPath", BoundaryPath)
    resolved = workspace_module._safe_link_target("member", "target")

    assert resolved is not None
    assert resolved.parts == ("directory", "target")


class BoundaryAdapter(ProviderAdapter):
    provider_id = "codex"

    def __init__(
        self,
        behavior: str = "complete",
        identities: list[tuple[int, str]] | None = None,
    ) -> None:
        self.behavior = behavior
        self.identities = list(identities or [(0, "")])
        self.interruptions = 0

    async def run_turn(self, **values: Any) -> ProviderResult:
        pre_prompt_gate = values.get("pre_prompt_gate")
        if pre_prompt_gate is not None:
            await pre_prompt_gate()
        if self.behavior == "safety":
            raise SafetyGuardError("stagnation", "codex", recoverable=True)
        if self.behavior == "reconciliation":
            raise ReconciliationRequiredError("reconciliation")
        event_handler = values["event_handler"]
        await event_handler(
            ProviderEvent(
                "provider.prompt.accepted",
                status="accepted",
                native_session_id="native",
            )
        )
        return ProviderResult(
            provider="codex",
            native_session_id="native",
            native_turn_id="turn",
            status="complete",
            usage={"total_tokens": 1},
            ambiguous_mutation=self.behavior == "ambiguous",
        )

    async def models(self, workspace: Path) -> tuple[ProviderModel, ...]:
        del workspace
        return ()

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider="codex",
            ready=True,
            detail="boundary adapter",
            capabilities=frozenset(),
        )

    async def interrupt(self) -> None:
        self.interruptions += 1

    def process_identity(self) -> tuple[int, str]:
        if len(self.identities) > 1:
            return self.identities.pop(0)
        return self.identities[0]


class BoundaryScheduler:
    async def choose(self, *unused: object, **values: object) -> RoutingDecision:
        return RoutingDecision(
            provider="codex",
            model="boundary-model",
            effort=str(values.get("effort", "low")) or "low",
            reason="boundary",
            ranked=(),
            rejected=(),
            execution_profile=str(values.get("execution_profile", "")),
            workload=str(values.get("workload", "")),
        )


def _worker_attempt(
    root: Path,
    adapter: BoundaryAdapter,
    *,
    profile: str = "interactive",
) -> tuple[HarnessService, SessionWorker, Any, TurnGuard]:
    service = HarnessService(
        paths(root / "state"),
        worker_manager=WorkerProbe(),
    )
    created = _create_direct(service, root)
    workspace = Path(created.worktree)
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
    service.store.set_session_safety(created.session_id, profile)
    command = service.store.enqueue_command(
        created.session_id,
        "message",
        {"text": "boundary"},
        new_uuid(),
    )
    assert service.store.claim_command(created.session_id) is not None
    limits = limits_for(profile, "implementation")
    service.store.create_command_envelope(
        command.command_id,
        created.session_id,
        profile,
        limits.as_dict(),
    )
    service.store.create_child_launch_gate(
        command.command_id,
        created.session_id,
        limits.max_child_agents,
    )
    worker = SessionWorker(
        service.store,
        service.blobs,
        BoundaryScheduler(),  # type: ignore[arg-type]
        {"codex": adapter},
        created.session_id,
    )
    service.store.register_worker(created.session_id, 123, worker.incarnation)
    worker._compile_context = (
        lambda unused_session, unused_limits, unused_command_id: SimpleNamespace(
            text="canonical context",
            estimated_tokens=10,
        )
    )
    worker._guard_repeated_dispatch = lambda *unused, **values: None
    return service, worker, command, TurnGuard(limits)


def test_worker_rejects_a_mismatched_xhigh_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HarnessService(
        paths(tmp_path / "xhigh-state"),
        worker_manager=WorkerProbe(),
    )
    created = _create_direct(service, tmp_path / "xhigh")
    service.store.set_session_safety(created.session_id, "unattended")
    command = service.store.enqueue_command(
        created.session_id,
        "message",
        {
            "text": "maximum",
            "provider": "codex",
            "effort": "xhigh",
        },
        "mismatched-xhigh",
    )
    claimed = service.store.claim_command(created.session_id)
    assert claimed is not None
    worker = SessionWorker(
        service.store,
        service.blobs,
        BoundaryScheduler(),  # type: ignore[arg-type]
        {},
        created.session_id,
    )
    monkeypatch.setattr(
        service.store,
        "xhigh_authorization_or_park",
        lambda unused: {"provider": "claude"},
    )

    asyncio.run(worker._execute_message(claimed))

    failed = service.store.get_command(command.command_id)
    assert failed.status == "failed"
    assert failed.result["code"] == "E_SAFETY_EFFORT"
    service.close()


def test_worker_context_repetition_goal_and_reconciliation_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = object.__new__(SessionWorker)
    worker.session_id = "session"

    class TransitionStore:
        def __init__(self, result: str) -> None:
            self.result = result

        def reserve_dispatch_generation_transition(self, *unused: object) -> str:
            return self.result

    monkeypatch.setattr(
        worker_module,
        "inspect_workspace",
        lambda unused: ("material", "summary"),
    )
    for transition, reason in (
        ("stage-mismatch", "dispatch-transition-stage-mismatch"),
        ("epoch-mismatch", "dispatch-transition-epoch-mismatch"),
    ):
        worker.store = TransitionStore(transition)  # type: ignore[assignment]
        with pytest.raises(SafetyGuardError) as raised:
            worker._guard_repeated_dispatch(
                "command",
                "instruction",
                "context",
                tmp_path,
                {},
                "unattended",
            )
        assert raised.value.reason == reason

    appended: list[tuple[str, dict[str, Any]]] = []
    pair = worker_module._text_digest("pair")
    fingerprint_store = SimpleNamespace(
        all_events=lambda unused: [
            SimpleNamespace(
                event_type="tool.result.fingerprint",
                metadata={"generation_digest": "current", "pair_digest": "other"},
            ),
            SimpleNamespace(
                event_type="tool.result.fingerprint",
                metadata={"generation_digest": "old", "pair_digest": pair},
            ),
        ],
        append_event=lambda session_id, event_type, **values: appended.append(
            (event_type, values["metadata"])
        ),
    )
    worker.store = fingerprint_store  # type: ignore[assignment]
    worker._dispatch_generation = lambda: {
        "generation_digest": "current",
        "invalidation_id": "",
    }
    worker._record_tool_result_fingerprint("command", "turn", "pair")
    assert appended[0][0] == "tool.result.fingerprint"

    captured: dict[str, Any] = {}
    context_store = SimpleNamespace(
        goal_for_session=lambda unused: None,
        fork_lineage=lambda unused: {"source_context_digest": "inherited"},
        context_events=lambda *unused, **values: [],
        context_history_summary=lambda *unused: {},
        event_count=lambda unused: 0,
        context_unresolved_decisions=lambda unused: [],
    )
    worker.store = context_store  # type: ignore[assignment]
    worker.blobs = SimpleNamespace(get_text=lambda digest: "text:" + digest)
    monkeypatch.setattr(worker_module, "workspace_summary", lambda unused: "summary")
    monkeypatch.setattr(
        worker_module,
        "workspace_context_artifacts",
        lambda unused: [],
    )

    def capture_context(*unused: object, **values: Any) -> SimpleNamespace:
        captured.update(values)
        return SimpleNamespace(text="compiled")

    monkeypatch.setattr(worker_module, "compile_context", capture_context)
    created = Session(
        session_id=new_uuid(),
        name="boundary",
        workspace=str(tmp_path),
        worktree=str(tmp_path),
        permission_mode="approval",
        lifecycle="running",
        attention="idle",
        active_provider="",
        model="",
        effort="",
        goal_id="",
        external_ref={},
        owner_host="host",
        owner_epoch=1,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    worker._compile_context(
        created,
        limits_for("interactive", "implementation"),
        "command",
    )
    assert captured["inherited_context_digest"] == "inherited"
    assert captured["inherited_context"] == "text:inherited"

    updated_goal = SimpleNamespace(goal_id="goal", milestones=("new",))
    goal_store = SimpleNamespace(
        goal_for_session=lambda unused: SimpleNamespace(
            goal_id="goal",
            milestones=("old",),
        ),
        evidence=lambda unused: [],
        update_milestone_statuses=lambda unused_goal, milestones: updated_goal,
    )
    worker.store = goal_store  # type: ignore[assignment]
    worker._exhausted_budget = lambda: ""
    monkeypatch.setattr(worker_module, "evaluate_milestones", lambda *unused: ("new",))
    monkeypatch.setattr(
        worker_module,
        "evaluate_goal",
        lambda *unused: SimpleNamespace(satisfied=False),
    )
    asyncio.run(worker._evaluate_goal())

    class EmptyManager:
        async def recover_after_restart(self, unused: str) -> SimpleNamespace:
            return SimpleNamespace(reconciliations=())

    monkeypatch.setattr(
        worker_module,
        "ReconciliationManager",
        lambda unused_store, unused_blobs: EmptyManager(),
    )
    worker.store = SimpleNamespace()  # type: ignore[assignment]
    worker.blobs = SimpleNamespace()
    with pytest.raises(RuntimeError, match="did not create reconciliation"):
        asyncio.run(
            worker._pause_for_ambiguous_dispatch(
                "command",
                "turn",
                "codex",
                "reason",
            )
        )


def test_worker_fault_barrier_timing_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sleep = asyncio.sleep

    async def yield_once(unused: float) -> None:
        await original_sleep(0)

    with monkeypatch.context() as current:
        current.setattr(worker_module.asyncio, "sleep", yield_once)
        service, worker, command, guard = _worker_attempt(
            tmp_path / "service-fault",
            BoundaryAdapter(),
        )
        with pytest.raises(SafetyGuardError, match="proof-service-fault-timeout"):
            asyncio.run(
                worker._execute_attempt(
                    command.command_id,
                    {
                        "proof_service_fault_probe": {
                            "provider": "codex",
                            "stage": "after-acceptance-before-terminal",
                        }
                    },
                    "boundary",
                    frozenset(),
                    guard,
                    0,
                    enforce_concurrency=True,
                )
            )
        assert any(
            event.event_type == "proof.service-fault.timeout"
            for event in service.store.all_events(worker.session_id)
        )
        service.close()

    class TimedLoop:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def time(self) -> float:
            if len(self.values) > 1:
                return self.values.pop(0)
            return self.values[0]

    with monkeypatch.context() as current:
        current.setattr(worker_module.asyncio, "sleep", yield_once)
        current.setattr(
            worker_module.asyncio,
            "get_running_loop",
            lambda: TimedLoop([0.0, 1.0, 31.0]),
        )
        service, worker, command, guard = _worker_attempt(
            tmp_path / "readiness-timeout",
            BoundaryAdapter(),
        )
        with pytest.raises(SafetyGuardError, match="proof-fault-readiness-timeout"):
            asyncio.run(
                worker._execute_attempt(
                    command.command_id,
                    {
                        "proof_fault_probe": {
                            "provider": "codex",
                            "stage": "after-lease-before-acceptance",
                        }
                    },
                    "boundary",
                    frozenset(),
                    guard,
                    0,
                    enforce_concurrency=True,
                )
            )
        service.close()

    with monkeypatch.context() as current:
        current.setattr(worker_module.asyncio, "sleep", yield_once)
        current.setattr(
            worker_module.asyncio,
            "get_running_loop",
            lambda: TimedLoop([0.0, 1.0]),
        )
        adapter = BoundaryAdapter(identities=[(10, "start"), (11, "changed")])
        service, worker, command, guard = _worker_attempt(
            tmp_path / "identity-changed",
            adapter,
            profile="unattended",
        )
        with pytest.raises(SafetyGuardError, match="process-identity-changed"):
            asyncio.run(
                worker._execute_attempt(
                    command.command_id,
                    {
                        "proof_fault_probe": {
                            "provider": "codex",
                            "stage": "after-lease-before-acceptance",
                        }
                    },
                    "boundary",
                    frozenset(),
                    guard,
                    0,
                    enforce_concurrency=True,
                )
            )
        service.close()

    with monkeypatch.context() as current:
        current.setattr(worker_module.asyncio, "sleep", yield_once)
        current.setattr(
            worker_module.asyncio,
            "get_running_loop",
            lambda: TimedLoop([0.0, 1.0, 31.0]),
        )
        adapter = BoundaryAdapter(identities=[(10, "start")])
        service, worker, command, guard = _worker_attempt(
            tmp_path / "termination-timeout",
            adapter,
            profile="unattended",
        )
        with pytest.raises(SafetyGuardError, match="termination-timeout"):
            asyncio.run(
                worker._execute_attempt(
                    command.command_id,
                    {
                        "proof_fault_probe": {
                            "provider": "codex",
                            "stage": "after-lease-before-acceptance",
                        }
                    },
                    "boundary",
                    frozenset(),
                    guard,
                    0,
                    enforce_concurrency=True,
                )
            )
        assert any(
            event.event_type == "proof.fault.timeout"
            for event in service.store.all_events(worker.session_id)
        )
        service.close()


def test_worker_attempt_admission_ownership_and_result_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, worker, command, guard = _worker_attempt(
        tmp_path / "denied",
        BoundaryAdapter(),
    )
    monkeypatch.setattr(
        service.store,
        "reserve_route_admission",
        lambda *unused, **values: {
            "admitted": False,
            "reason": "capacity",
            "lease_id": "",
        },
    )
    with pytest.raises(ProviderUnavailableError, match="admission denied"):
        asyncio.run(
            worker._execute_attempt(
                command.command_id,
                {},
                "boundary",
                frozenset(),
                guard,
                0,
                enforce_concurrency=True,
            )
        )
    monkeypatch.undo()
    service.close()

    service, worker, command, guard = _worker_attempt(
        tmp_path / "lease-release",
        BoundaryAdapter(),
        profile="unattended",
    )

    class FailingGate:
        def __init__(self, **unused: object) -> None:
            raise RuntimeError("gate construction failed")

    monkeypatch.setattr(worker_module, "ChildLaunchGate", FailingGate)
    with pytest.raises(RuntimeError, match="gate construction failed"):
        asyncio.run(
            worker._execute_attempt(
                command.command_id,
                {},
                "boundary",
                frozenset(),
                guard,
                0,
                enforce_concurrency=True,
            )
        )
    assert service.store.process_leases(worker.session_id)[0]["state"] == "released"
    monkeypatch.undo()
    service.close()

    original_sleep = asyncio.sleep

    async def yield_once(unused: float) -> None:
        await original_sleep(0)

    service, worker, command, guard = _worker_attempt(
        tmp_path / "ownership",
        BoundaryAdapter(),
    )
    ownership = iter((True, False))
    monkeypatch.setattr(
        service.store,
        "heartbeat_worker",
        lambda *unused: next(ownership, False),
    )
    monkeypatch.setattr(worker_module.asyncio, "sleep", yield_once)
    with pytest.raises(WorkerOwnershipLostError, match="result ownership"):
        asyncio.run(
            worker._execute_attempt(
                command.command_id,
                {},
                "boundary",
                frozenset(),
                guard,
                0,
                enforce_concurrency=True,
            )
        )
    monkeypatch.undo()
    service.close()

    for stage, action in ((0, "downgrade"), (1, "failover")):
        service, worker, command, guard = _worker_attempt(
            tmp_path / ("safety-" + str(stage)),
            BoundaryAdapter("safety"),
        )
        with pytest.raises(SafetyGuardError, match="stagnation"):
            asyncio.run(
                worker._execute_attempt(
                    command.command_id,
                    {},
                    "boundary",
                    frozenset(),
                    guard,
                    stage,
                    enforce_concurrency=True,
                )
            )
        assert service.store.guard_incidents(worker.session_id)[0]["action"] == action
        service.close()

    service, worker, command, guard = _worker_attempt(
        tmp_path / "reconciliation-error",
        BoundaryAdapter("reconciliation"),
    )
    with pytest.raises(ReconciliationRequiredError):
        asyncio.run(
            worker._execute_attempt(
                command.command_id,
                {},
                "boundary",
                frozenset(),
                guard,
                0,
                enforce_concurrency=True,
            )
        )
    service.close()

    service, worker, command, guard = _worker_attempt(
        tmp_path / "ambiguous-result",
        BoundaryAdapter("ambiguous"),
    )

    async def pause(*unused: object) -> str:
        return "reconciliation"

    monkeypatch.setattr(worker, "_pause_for_ambiguous_dispatch", pause)
    with pytest.raises(ReconciliationRequiredError):
        asyncio.run(
            worker._execute_attempt(
                command.command_id,
                {},
                "boundary",
                frozenset(),
                guard,
                0,
                enforce_concurrency=True,
            )
        )
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
