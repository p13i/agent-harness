import asyncio
from pathlib import Path
import subprocess
from types import SimpleNamespace

from aiohttp.test_utils import TestClient
from aiohttp.test_utils import TestServer
import pytest

from agent_harness import api as api_module
import agent_harness.safety as safety_module
from agent_harness.api import create_app
from agent_harness.config import CONTROL_BUILD_ID
from agent_harness.config import CONTROL_PROTOCOL_VERSION
from agent_harness.config import paths
from agent_harness.errors import SafetyGuardError
from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import ProviderAttempt
from agent_harness.reconciliation import inspect_workspace
from agent_harness.service import HarnessService
from agent_harness.workspace import checkpoint_workspace


@pytest.fixture(autouse=True)
def stable_state_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        safety_module.shutil,
        "disk_usage",
        lambda unused: SimpleNamespace(free=10 * 1024**3),
    )


class Workers:
    def __init__(self) -> None:
        self.started: list[str] = []

    def ensure(self, session_id: str) -> None:
        self.started.append(session_id)

    def stop_all(self) -> None:
        return


@pytest.mark.asyncio
async def test_api_inspects_and_resolves_reconciliation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    repository(workspace)
    workers = Workers()
    service = HarnessService(
        paths(tmp_path / "state"),
        worker_manager=workers,
    )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    try:
        created = await client.post(
            "/v1/sessions",
            headers=headers,
            json={
                "workspace": str(workspace),
                "permission_mode": "approval",
            },
        )
        session_id = (await created.json())["session"]["session_id"]
        submitted = await client.post(
            "/v1/sessions/" + session_id + "/messages",
            headers={
                **headers,
                "Idempotency-Key": "ambiguous-message",
            },
            json={"text": "Make one bounded change."},
        )
        command_id = (await submitted.json())["command"]["command_id"]
        command = service.store.claim_command(session_id)
        assert command is not None
        assert command.command_id == command_id
        attempt = ProviderAttempt(
            attempt_id=new_uuid(),
            session_id=session_id,
            provider="codex",
            native_session_id="",
            model="default",
            effort="high",
            auth_mode="subscription",
            status="running",
            started_at=utc_now(),
            ended_at="",
        )
        service.store.create_attempt(attempt)
        turn_id = service.store.start_turn(
            session_id,
            attempt.attempt_id,
        )
        session = service.store.get_session(session_id)
        checkpoint = checkpoint_workspace(
            session,
            service.blobs,
            sequence=service.store.last_sequence(session_id),
            provider="codex",
            native_session_id="",
            context_text="",
        )
        service.store.add_checkpoint(checkpoint)
        service.store.record_dispatch_checkpoint(
            command_id,
            attempt.attempt_id,
            turn_id,
            checkpoint.checkpoint_id,
        )
        service.store.append_event(
            session_id,
            "checkpoint.created",
            status="complete",
            metadata={"checkpoint_id": checkpoint.checkpoint_id},
            turn_id=turn_id,
        )
        turn_page = await client.get(
            "/v1/sessions/"
            + session_id
            + "/turns?after_sequence=0&limit=10",
            headers=headers,
        )
        assert turn_page.status == 200
        turn_page_value = await turn_page.json()
        assert turn_page_value["turns"][0]["turn_id"] == turn_id
        assert turn_page_value["turns"][0]["checkpoint_id"] == (
            checkpoint.checkpoint_id
        )
        assert "payload_json" not in str(turn_page_value)
        turn_detail = await client.get(
            "/v1/sessions/" + session_id + "/turns/" + turn_id,
            headers=headers,
        )
        assert (await turn_detail.json())["turn"]["turn_ids"] == [
            turn_id
        ]
        diff = await client.get(
            "/v1/sessions/"
            + session_id
            + "/checkpoints/"
            + checkpoint.checkpoint_id
            + "/diff?start_line=0&limit=10",
            headers=headers,
        )
        assert diff.status == 200
        assert (await diff.json())["diff"]["checkpoint_id"] == (
            checkpoint.checkpoint_id
        )
        unauthorized_turns = await client.get(
            "/v1/sessions/" + session_id + "/turns"
        )
        assert unauthorized_turns.status == 401
        invalid_turn_page = await client.get(
            "/v1/sessions/"
            + session_id
            + "/turns?after_sequence=-1",
            headers=headers,
        )
        assert invalid_turn_page.status == 400
        invalid_diff_page = await client.get(
            "/v1/sessions/"
            + session_id
            + "/checkpoints/"
            + checkpoint.checkpoint_id
            + "/diff?start_line=-1",
            headers=headers,
        )
        assert invalid_diff_page.status == 400
        service.store.mark_provider_boundary(attempt.attempt_id)
        digest, summary = inspect_workspace(Path(session.worktree))
        recovery = service.store.recover_interrupted_commands(
            session_id,
            digest,
            summary,
        )
        record = recovery.reconciliations[0]

        listed = await client.get(
            "/v1/sessions/" + session_id + "/reconciliations",
            headers=headers,
        )
        assert (await listed.json())["reconciliations"][0][
            "reconciliation_id"
        ] == record.reconciliation_id
        inspected = await client.get(
            "/v1/reconciliations/" + record.reconciliation_id,
            headers=headers,
        )
        assert (await inspected.json())["reconciliation"][
            "current_workspace_digest"
        ] == digest
        resolution_headers = {
            **headers,
            "Idempotency-Key": "restore-ambiguous-turn",
        }
        approval_required = await client.post(
            "/v1/reconciliations/"
            + record.reconciliation_id
            + "/resolution",
            headers=resolution_headers,
            json={
                "decision": "restore-pre-turn",
                "observed_workspace_digest": digest,
                "audit": {"actor": "integration-test"},
            },
        )
        assert approval_required.status == 409
        assert (await approval_required.json())["error"]["code"] == (
            "E_APPROVAL_REQUIRED"
        )
        approvals = service.store.pending_approvals(session_id)
        assert len(approvals) == 1
        approval_id = approvals[0]["approval_id"]
        approved = await client.post(
            "/v1/sessions/"
            + session_id
            + "/approvals/"
            + approval_id,
            headers={
                **headers,
                "Idempotency-Key": "approve-workspace-restore",
            },
            json={"decision": "approve"},
        )
        assert (await approved.json())["resolved"]
        resolved = await client.post(
            "/v1/reconciliations/"
            + record.reconciliation_id
            + "/resolution",
            headers=resolution_headers,
            json={
                "decision": "restore-pre-turn",
                "observed_workspace_digest": digest,
                "approval_id": approval_id,
                "audit": {"actor": "integration-test"},
            },
        )
        assert resolved.status == 200
        resolved_value = (await resolved.json())["reconciliation"]
        assert resolved_value["resolution"] == "restore-pre-turn"
        repeated = await client.post(
            "/v1/reconciliations/"
            + record.reconciliation_id
            + "/resolution",
            headers=resolution_headers,
            json={
                "decision": "restore-pre-turn",
                "observed_workspace_digest": digest,
                "approval_id": approval_id,
                "audit": {"actor": "integration-test"},
            },
        )
        assert (await repeated.json())["reconciliation"] == (
            resolved_value
        )
    finally:
        await client.close()
        service.close()


def repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
    )
    (path / "file.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "file.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "initial"],
        check=True,
    )


@pytest.mark.asyncio
async def test_api_creates_session_and_accepts_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "repo"
    repository(workspace)
    workers = Workers()
    service = HarnessService(
        paths(tmp_path / "state"),
        worker_manager=workers,
    )
    app = create_app(service, "test-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    try:
        unauthorized = await client.get("/v1/sessions")
        assert unauthorized.status == 401
        unauthorized_body = await unauthorized.json()
        assert unauthorized.headers["X-Correlation-ID"]
        assert (
            unauthorized_body["error"]["correlation_id"]
            == unauthorized.headers["X-Correlation-ID"]
        )
        created = await client.post(
            "/v1/sessions",
            headers=headers,
            json={
                "workspace": str(workspace),
                "goal": "Maintain bounded operation.",
                "goal_kind": "invariant",
            },
        )
        assert created.status == 201
        session = (await created.json())["session"]
        session_id = session["session_id"]
        health = await client.get("/healthz", headers=headers)
        health_value = await health.json()
        assert health_value["status"] == "ok"
        assert health_value["control_build_id"] == CONTROL_BUILD_ID
        assert health_value["control_protocol_version"] == (
            CONTROL_PROTOCOL_VERSION
        )
        assert health_value["state_root"] == str(tmp_path / "state")
        capabilities = await client.get(
            "/v1/capabilities",
            headers=headers,
        )
        capabilities_value = await capabilities.json()
        assert capabilities_value["api_version"] == "1.4.0"
        assert capabilities_value["control_protocol_version"] == (
            CONTROL_PROTOCOL_VERSION
        )
        assert capabilities_value["paths"]["socket"].endswith(
            "/.runtime/control.sock"
        )
        assert capabilities_value["paths"]["token_path"].endswith(
            "/.runtime/secrets/api-token"
        )
        assert "reconciliation" in capabilities_value["features"]
        sync_status = await client.get("/v1/sync", headers=headers)
        assert (await sync_status.json())["sync"]["state"] == "unknown"
        synchronized = await client.post(
            "/v1/sync",
            headers=headers,
            json={},
        )
        assert (await synchronized.json())["sync"]["state"] == (
            "not-configured"
        )
        ready = await client.get("/readyz", headers=headers)
        assert (await ready.json())["status"] == "ready"
        listed = await client.get("/v1/sessions", headers=headers)
        assert (await listed.json())["sessions"][0]["session_id"] == (
            session_id
        )
        archived = await client.get(
            "/v1/sessions?archived=1",
            headers=headers,
        )
        assert len((await archived.json())["sessions"]) == 1
        archive_response = await client.post(
            "/v1/sessions/" + session_id + "/archive",
            headers={
                **headers,
                "Idempotency-Key": "integration-archive",
            },
            json={},
        )
        assert (await archive_response.json())["session"]["archived"]
        repeated_archive = await client.post(
            "/v1/sessions/" + session_id + "/archive",
            headers={
                **headers,
                "Idempotency-Key": "integration-archive",
            },
            json={},
        )
        assert (await repeated_archive.json())["session"]["archived"]
        active_after_archive = await client.get(
            "/v1/sessions",
            headers=headers,
        )
        assert (await active_after_archive.json())["sessions"] == []
        unarchive_response = await client.post(
            "/v1/sessions/" + session_id + "/unarchive",
            headers={
                **headers,
                "Idempotency-Key": "integration-unarchive",
            },
            json={},
        )
        assert not (await unarchive_response.json())["session"]["archived"]
        detail = await client.get(
            "/v1/sessions/" + session_id,
            headers=headers,
        )
        assert (await detail.json())["safety"]["session"]["profile"] == (
            "unattended"
        )
        usage = await client.get(
            "/v1/sessions/" + session_id + "/usage",
            headers=headers,
        )
        assert (await usage.json())["safety"]["envelopes"] == []
        goal = await client.get(
            "/v1/sessions/" + session_id + "/goal",
            headers=headers,
        )
        assert (await goal.json())["goal"]["kind"] == "invariant"
        evidence = await client.post(
            "/v1/sessions/" + session_id + "/evidence",
            headers=headers,
            json={
                "type": "command",
                "subject": "make test",
                "outcome": "passed",
                "value": {"exit_code": 0},
            },
        )
        assert evidence.status == 201
        approvals = await client.get(
            "/v1/sessions/" + session_id + "/approvals",
            headers=headers,
        )
        assert (await approvals.json())["approvals"] == []
        extension = await client.post(
            "/v1/sessions/" + session_id + "/budget-extensions",
            headers={
                **headers,
                "Idempotency-Key": "integration-extension",
            },
            json={
                "reason": "finish one bounded validation",
                "additional_seconds": 60,
                "allow_xhigh_once": True,
            },
        )
        assert extension.status == 201
        extension_value = (await extension.json())["safety"]
        assert extension_value["xhigh_authorizations"] == 1
        repeated_extension = await client.post(
            "/v1/sessions/" + session_id + "/budget-extensions",
            headers={
                **headers,
                "Idempotency-Key": "integration-extension",
            },
            json={
                "reason": "finish one bounded validation",
                "additional_seconds": 60,
                "allow_xhigh_once": True,
            },
        )
        repeated_value = (await repeated_extension.json())["safety"]
        assert repeated_value["xhigh_authorizations"] == 1
        conflicting_extension = await client.post(
            "/v1/sessions/" + session_id + "/budget-extensions",
            headers={
                **headers,
                "Idempotency-Key": "integration-extension",
            },
            json={
                "reason": "different request",
                "additional_seconds": 120,
            },
        )
        assert conflicting_extension.status == 409

        refused_lease = await client.post(
            "/v1/leases",
            headers={
                **headers,
                "Idempotency-Key": "integration-lease-refused",
            },
            json={
                "provider": "claude",
                "execution_profile": "unattended",
            },
        )
        assert refused_lease.status == 429
        assert (await refused_lease.json())["error"]["code"] == (
            "E_SAFETY_GUARD"
        )
        samples = [
            {
                "observed_at": "invalid",
                "binding_percent": 25,
                "credits_engaged": False,
            },
            {
                "observed_at": "2020-01-01T00:00:00+00:00",
                "binding_percent": 25,
                "credits_engaged": False,
            },
            {
                "observed_at": "2099-01-01T00:00:00+00:00",
                "binding_percent": None,
                "credits_engaged": False,
            },
            {
                "observed_at": "2099-01-01T00:00:00+00:00",
                "binding_percent": 25,
                "credits_engaged": True,
            },
            {
                "observed_at": "2099-01-01T00:00:00+00:00",
                "binding_percent": 70,
                "credits_engaged": False,
            },
        ]
        latest_usage = service.store.latest_usage
        for sample in samples:
            monkeypatch.setattr(
                service.store,
                "latest_usage",
                lambda sample=sample: {"codex": sample},
            )
            with pytest.raises(SafetyGuardError):
                service._require_process_lease_capacity(
                    "codex",
                    "unattended",
                )
        monkeypatch.setattr(
            service.store,
            "latest_usage",
            latest_usage,
        )
        service.store.record_usage(
            "codex",
            25.0,
            False,
            {"payload": {}, "error": ""},
        )
        created_lease = await client.post(
            "/v1/leases",
            headers={
                **headers,
                "Idempotency-Key": "integration-lease",
            },
            json={
                "provider": "codex",
                "session_id": session_id,
                "execution_profile": "unattended",
            },
        )
        assert created_lease.status == 201
        lease = (await created_lease.json())["lease"]
        lease_id = lease["lease_id"]
        repeated_lease = await client.post(
            "/v1/leases",
            headers={
                **headers,
                "Idempotency-Key": "integration-lease",
            },
            json={
                "provider": "codex",
                "session_id": session_id,
                "execution_profile": "unattended",
            },
        )
        assert (await repeated_lease.json())["lease"]["lease_id"] == lease_id
        attached = await client.patch(
            "/v1/leases/" + lease_id,
            headers={
                **headers,
                "Idempotency-Key": "integration-lease-attach",
            },
            json={
                "action": "attach",
                "pid": 1234,
                "pid_start": "5678",
            },
        )
        assert (await attached.json())["lease"]["state"] == "active"
        leases = await client.get("/v1/leases", headers=headers)
        assert (await leases.json())["leases"][0]["pid"] == 1234
        released = await client.patch(
            "/v1/leases/" + lease_id,
            headers={
                **headers,
                "Idempotency-Key": "integration-lease-release",
            },
            json={"action": "release"},
        )
        assert (await released.json())["lease"]["state"] == "released"
        missing_key = await client.post(
            "/v1/sessions/" + session_id + "/messages",
            headers=headers,
            json={"text": "not accepted"},
        )
        assert missing_key.status == 400
        assert (await missing_key.json())["error"]["code"] == "E_INPUT"
        with monkeypatch.context() as low_disk:
            low_disk.setattr(
                safety_module.shutil,
                "disk_usage",
                lambda unused: SimpleNamespace(free=0),
            )
            refused_message = await client.post(
                "/v1/sessions/" + session_id + "/messages",
                headers={
                    **headers,
                    "Idempotency-Key": "integration-low-disk",
                },
                json={
                    "text": "must not be queued",
                    "provider": "codex",
                },
            )
            assert refused_message.status == 429
            refused_value = await refused_message.json()
            assert refused_value["error"]["code"] == "E_SAFETY_GUARD"
            assert "state-volume-headroom" in (
                refused_value["error"]["message"]
            )
        accepted = await client.post(
            "/v1/sessions/" + session_id + "/messages",
            headers={
                **headers,
                "Idempotency-Key": "integration-message",
            },
            json={"text": "continue"},
        )
        assert accepted.status == 202
        events = await client.get(
            "/v1/sessions/" + session_id + "/events",
            headers=headers,
        )
        values = (await events.json())["events"]
        assert values[-1]["event_type"] == "user.message"
        assert workers.started == [session_id]
        command_value = (await accepted.json())["command"]
        command_status = await client.get(
            "/v1/commands/" + command_value["command_id"],
            headers=headers,
        )
        assert (await command_status.json())["command"]["status"] == (
            "queued"
        )
        paused = await client.post(
            "/v1/sessions/" + session_id + "/commands/pause",
            headers={
                **headers,
                "Idempotency-Key": "integration-pause",
            },
            json={},
        )
        assert paused.status == 202
        service.store.update_session(
            session_id,
            lifecycle="paused",
        )
        workers.started.clear()
        service.recover_workers()
        assert workers.started == [session_id]

        configured = await client.patch(
            "/v1/sessions/" + session_id,
            headers=headers,
            json={
                "name": "Configured session",
                "permission_mode": "read-only",
            },
        )
        assert configured.status == 200
        configured_session = (await configured.json())["session"]
        assert configured_session["name"] == "Configured session"
        assert configured_session["permission_mode"] == "read-only"

        saved_ui = await client.put(
            "/v1/sessions/" + session_id + "/ui-state",
            headers=headers,
            json={
                "composer": "unfinished message",
                "composer_cursor": "9",
                "events": "off",
                "expanded_blocks": "[\"tool-1\"]",
                "inspector_tab": "storage",
                "provider": "codex",
                "request_id": "request-1",
                "session_filter": "focused",
                "sidebar_width": "48",
            },
        )
        assert saved_ui.status == 200
        restored_ui = await client.get(
            "/v1/sessions/" + session_id + "/ui-state",
            headers=headers,
        )
        assert (await restored_ui.json())["ui_state"] == {
            "composer": "unfinished message",
            "composer_cursor": "9",
            "events": "off",
            "expanded_blocks": "[\"tool-1\"]",
            "inspector_tab": "storage",
            "provider": "codex",
            "request_id": "request-1",
            "session_filter": "focused",
            "sidebar_width": "48",
        }

        checkpoint = await client.post(
            "/v1/sessions/" + session_id + "/checkpoints",
            headers=headers,
            json={},
        )
        assert checkpoint.status == 201
        checkpoint_value = (await checkpoint.json())["checkpoint"]
        assert checkpoint_value["session_id"] == session_id

        exported = await client.post(
            "/v1/sessions/" + session_id + "/export",
            headers=headers,
            json={},
        )
        assert exported.status == 200
        export_path = Path((await exported.json())["path"])
        assert export_path.is_file()
        assert export_path.with_name(
            session_id + ".run-context.gpt.json"
        ).is_file()
        assert export_path.with_name(
            session_id + ".transcript.jsonl"
        ).is_file()
        assert export_path.with_name(
            session_id + ".transcript.md"
        ).is_file()

        forked = await client.post(
            "/v1/sessions/" + session_id + "/fork",
            headers=headers,
            json={"name": "Forked session"},
        )
        assert forked.status == 201
        forked_session = (await forked.json())["session"]
        assert forked_session["session_id"] != session_id
        assert forked_session["name"] == "Forked session"
        registry = await client.get("/v1/registry", headers=headers)
        assert isinstance((await registry.json())["entries"], list)
        public_keys = await client.get(
            "/v1/fleet/keys",
            headers=headers,
        )
        key_values = (await public_keys.json())["keys"]
        transfer = await client.post(
            "/v1/sessions/" + forked_session["session_id"] + "/transfers",
            headers=headers,
            json={
                "destination_host": "destination-host",
                "destination_encryption_public": key_values["encryption"],
            },
        )
        assert transfer.status == 201
        transfer_value = (await transfer.json())["transfer"]
        finalized = await client.post(
            "/v1/sessions/"
            + forked_session["session_id"]
            + "/transfers/finalize",
            headers=headers,
            json={
                "destination_host": "destination-host",
                "owner_epoch": transfer_value["owner_epoch"],
            },
        )
        assert finalized.status == 200
    finally:
        await client.close()
        service.close()


@pytest.mark.asyncio
async def test_new_chat_is_named_from_prompt_and_inherits_workspace_ui(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    repository(workspace)
    workers = Workers()
    service = HarnessService(
        paths(tmp_path / "state"),
        worker_manager=workers,
    )
    app = create_app(service, "test-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    try:
        first_response = await client.post(
            "/v1/sessions",
            headers=headers,
            json={"workspace": str(workspace)},
        )
        first = (await first_response.json())["session"]
        await client.put(
            "/v1/sessions/" + first["session_id"] + "/ui-state",
            headers=headers,
            json={
                "events": "off",
                "session_filter": "focused",
                "sidebar_width": "52",
                "theme": "system",
            },
        )

        second_response = await client.post(
            "/v1/sessions",
            headers=headers,
            json={"workspace": str(workspace)},
        )
        second = (await second_response.json())["session"]
        no_goal = await client.get(
            "/v1/sessions/" + second["session_id"] + "/goal",
            headers=headers,
        )
        assert (await no_goal.json()) == {
            "goal": None,
            "evidence": [],
        }
        inherited = await client.get(
            "/v1/sessions/" + second["session_id"] + "/ui-state",
            headers=headers,
        )
        assert (await inherited.json())["ui_state"] == {
            "events": "off",
            "session_filter": "focused",
            "sidebar_width": "52",
            "theme": "system",
        }

        message = await client.post(
            "/v1/sessions/" + second["session_id"] + "/messages",
            headers={
                **headers,
                "Idempotency-Key": "name-session",
            },
            json={
                "text": (
                    "Investigate the provider routing regression "
                    "and repair its tests."
                )
            },
        )
        assert message.status == 202
        detail = await client.get(
            "/v1/sessions/" + second["session_id"],
            headers=headers,
        )
        assert (await detail.json())["session"]["name"] == (
            "Investigate the provider routing regression and repair "
            "its tests."
        )

        legacy = service.create_session(
            {"workspace": str(workspace)}
        )
        service.store.append_event(
            legacy.session_id,
            "user.message",
            role="user",
            text="Resume the durable migration.",
            status="accepted",
        )
        service.recover_workers()
        assert service.store.get_session(legacy.session_id).name == (
            "Resume the durable migration."
        )
    finally:
        await client.close()
        service.close()


@pytest.mark.asyncio
async def test_api_socket_pid_and_validation_helpers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_path = Path("/tmp") / ("agent-harness-" + new_uuid())
    await api_module._prepare_socket(socket_path)
    assert socket_path.parent.is_dir()

    socket_path.write_text("stale", encoding="utf-8")
    await api_module._prepare_socket(socket_path)
    assert not socket_path.exists()

    server = await api_module.asyncio.start_unix_server(
        lambda reader, writer: writer.close(),
        path=socket_path,
    )
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await api_module._prepare_socket(socket_path)
    finally:
        server.close()
        await server.wait_closed()
        socket_path.unlink(missing_ok=True)

    pid_path = tmp_path / "daemon.pid"
    api_module._remove_daemon_pid(pid_path)
    pid_path.write_text("invalid\n", encoding="utf-8")
    api_module._remove_daemon_pid(pid_path)
    assert pid_path.exists()
    pid_path.write_text("999999\n", encoding="utf-8")
    api_module._remove_daemon_pid(pid_path)
    assert pid_path.exists()
    monkeypatch.setattr(api_module.os, "getpid", lambda: 42)
    api_module._write_daemon_pid(pid_path)
    assert pid_path.stat().st_mode & 0o077 == 0
    api_module._remove_daemon_pid(pid_path)
    assert not pid_path.exists()

    assert api_module._integer("7") == 7
    with pytest.raises(ValueError, match="integer"):
        api_module._integer("invalid")
    with pytest.raises(ValueError, match="explicit"):
        api_module._validate_tcp_host("0.0.0.0")

    def unresolved(
        unused_host: str,
        unused_port: object,
    ) -> None:
        del unused_host
        del unused_port
        raise api_module.socket.gaierror()

    monkeypatch.setattr(
        api_module.socket,
        "getaddrinfo",
        unresolved,
    )
    with pytest.raises(ValueError, match="resolved"):
        api_module._validate_tcp_host("unresolvable.invalid")


@pytest.mark.asyncio
async def test_daemon_runtime_starts_sites_recovers_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "state")
    calls: list[str] = []

    class FakeWorkers:
        def stop_all(self) -> None:
            calls.append("workers-stopped")

    class FakeService:
        def __init__(self) -> None:
            self.paths = harness_paths
            self.store = object()
            self.workers = FakeWorkers()

        def recover_workers(self) -> None:
            calls.append("workers-recovered")

        def close(self) -> None:
            calls.append("service-closed")

    service = FakeService()

    class Runner:
        def __init__(self, app: object, access_log: object) -> None:
            del app
            del access_log

        async def setup(self) -> None:
            calls.append("runner-setup")

        async def cleanup(self) -> None:
            calls.append("runner-cleanup")

    class Site:
        def __init__(self, *unused_args: object) -> None:
            del unused_args

        async def start(self) -> None:
            calls.append("site-started")

    class Stopped:
        async def wait(self) -> None:
            calls.append("waited")

        def set(self) -> None:
            return

    class Loop:
        def add_signal_handler(
            self,
            signal_value: object,
            callback: object,
        ) -> None:
            del callback
            calls.append("signal-" + str(signal_value))

    async def sync_loop(unused_service: object) -> None:
        del unused_service

    monkeypatch.setattr(
        api_module,
        "HarnessService",
        lambda unused_paths: service,
    )
    monkeypatch.setattr(api_module, "api_token", lambda unused: "token")
    monkeypatch.setattr(
        api_module,
        "create_app",
        lambda unused_service, unused_token: object(),
    )
    monkeypatch.setattr(api_module.web, "AppRunner", Runner)
    monkeypatch.setattr(api_module.web, "UnixSite", Site)
    monkeypatch.setattr(api_module.web, "TCPSite", Site)
    monkeypatch.setattr(
        api_module,
        "_prepare_socket",
        lambda unused: api_module.asyncio.sleep(0),
    )
    monkeypatch.setattr(
        api_module,
        "_write_daemon_pid",
        lambda unused: calls.append("pid-written"),
    )
    monkeypatch.setattr(
        api_module,
        "_remove_daemon_pid",
        lambda unused: calls.append("pid-removed"),
    )
    monkeypatch.setattr(
        api_module.os,
        "chmod",
        lambda unused_path, unused_mode: calls.append(
            "socket-chmod"
        ),
    )
    monkeypatch.setattr(api_module, "_sync_loop", sync_loop)
    monkeypatch.setattr(api_module.asyncio, "Event", Stopped)
    monkeypatch.setattr(
        api_module.asyncio,
        "get_running_loop",
        lambda: Loop(),
    )
    monkeypatch.setattr(
        api_module,
        "publish_all",
        lambda *unused: calls.append("published"),
    )

    await api_module.run_daemon(
        harness_paths,
        tcp_host="127.0.0.1",
        tcp_port=8765,
    )

    assert calls.count("site-started") == 2
    assert "socket-chmod" in calls
    assert "workers-recovered" in calls
    assert "workers-stopped" in calls
    assert "published" in calls
    assert calls[-2:] == ["service-closed", "pid-removed"]


@pytest.mark.asyncio
async def test_api_stream_emits_existing_events_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[bytes] = []
    event = SimpleNamespace(
        sequence=7,
        event_type="agent.message",
        as_dict=lambda: {
            "sequence": 7,
            "event_type": "agent.message",
            "text": "complete",
        },
    )

    class Store:
        def events(
            self,
            session_id: str,
            *,
            after: int,
            limit: int,
        ) -> list[object]:
            assert session_id == "session-1"
            assert after == 6
            assert limit == 500
            return [event]

    class Response:
        async def prepare(self, request: object) -> None:
            del request

        async def write(self, content: bytes) -> None:
            writes.append(content)

    class Request:
        match_info = {"session_id": "session-1"}
        headers = {}
        query = {"after": "6"}

        def get(self, name: str, default: object) -> object:
            del name
            return default

    async def stop_after_heartbeat(unused: float) -> None:
        del unused
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        api_module,
        "_service",
        lambda unused: SimpleNamespace(store=Store()),
    )
    monkeypatch.setattr(
        api_module.web,
        "StreamResponse",
        lambda **unused: Response(),
    )
    monkeypatch.setattr(api_module.asyncio, "sleep", stop_after_heartbeat)

    with pytest.raises(asyncio.CancelledError):
        await api_module._stream(Request())  # type: ignore[arg-type]
    assert b"id: 7" in writes[0]
    assert b"agent.message" in writes[0]
    assert writes[1] == b": heartbeat\n\n"


@pytest.mark.asyncio
async def test_api_request_helpers_reject_invalid_shapes_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Request:
        def __init__(
            self,
            *,
            can_read_body: bool,
            value: object,
        ) -> None:
            self.can_read_body = can_read_body
            self.value = value
            self.app = {"service": "invalid"}
            self.headers = {"Idempotency-Key": "x" * 129}
            self.values = {"correlation_id": 7}

        async def json(self) -> object:
            return self.value

        def get(self, name: str, default: object) -> object:
            return self.values.get(name, default)

    empty = Request(can_read_body=False, value=None)
    assert await api_module._body(empty) == {}  # type: ignore[arg-type]
    invalid = Request(can_read_body=True, value=[])
    with pytest.raises(ValueError, match="JSON object"):
        await api_module._body(invalid)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="unavailable"):
        api_module._service(invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="too long"):
        api_module._idempotency_key(invalid)  # type: ignore[arg-type]
    assert api_module._correlation_id(invalid) == ""  # type: ignore[arg-type]

    async def stop_sync(unused: float) -> None:
        del unused
        raise asyncio.CancelledError()

    monkeypatch.setattr(api_module, "publish_all", lambda *unused: None)
    monkeypatch.setattr(api_module.asyncio, "sleep", stop_sync)

    with pytest.raises(asyncio.CancelledError):
        await api_module._sync_loop(
            SimpleNamespace(paths=object(), store=object())
        )


@pytest.mark.asyncio
async def test_external_session_and_managed_turn_requests_are_retry_safe(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "state"),
        worker_manager=Workers(),
    )
    app = create_app(service, "test-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    create_payload = {
        "workspace": str(workspace),
        "name": "External job",
        "external_ref": {
            "orchestrator": "p13i/machines",
            "job_id": "job-42",
        },
    }
    try:
        created = await client.post(
            "/v1/sessions",
            headers={
                **headers,
                "Idempotency-Key": "create-job-42",
            },
            json=create_payload,
        )
        assert created.status == 201
        session = (await created.json())["session"]
        repeated = await client.post(
            "/v1/sessions",
            headers={
                **headers,
                "Idempotency-Key": "create-job-42",
            },
            json=create_payload,
        )
        assert (await repeated.json())["session"]["session_id"] == (
            session["session_id"]
        )
        by_reference = await client.post(
            "/v1/sessions",
            headers={
                **headers,
                "Idempotency-Key": "create-job-42-alternate",
            },
            json=create_payload,
        )
        assert (await by_reference.json())["session"]["session_id"] == (
            session["session_id"]
        )
        conflicting_key = await client.post(
            "/v1/sessions",
            headers={
                **headers,
                "Idempotency-Key": "create-job-42",
            },
            json={
                **create_payload,
                "name": "Different name",
            },
        )
        assert conflicting_key.status == 409
        conflicting_reference = await client.post(
            "/v1/sessions",
            headers={
                **headers,
                "Idempotency-Key": "create-job-42-conflict",
            },
            json={
                **create_payload,
                "name": "Different name",
            },
        )
        assert conflicting_reference.status == 409
        lookup = await client.get(
            "/v1/sessions"
            "?archived=1"
            "&external_orchestrator=p13i%2Fmachines"
            "&external_job_id=job-42",
            headers=headers,
        )
        values = (await lookup.json())["sessions"]
        assert [item["session_id"] for item in values] == [
            session["session_id"]
        ]
        missing_lookup_field = await client.get(
            "/v1/sessions?external_job_id=job-42",
            headers=headers,
        )
        assert missing_lookup_field.status == 400

        turn_payload = {
            "text": "Implement the bounded step.",
            "turn_ref": {
                "step_id": "step-7",
                "agent_role": "implementer",
            },
        }
        submitted = await client.post(
            "/v1/sessions/" + session["session_id"] + "/messages",
            headers={
                **headers,
                "Idempotency-Key": "turn-step-7",
            },
            json=turn_payload,
        )
        command = (await submitted.json())["command"]
        assert command["turn_ref"] == turn_payload["turn_ref"]
        repeated_turn = await client.post(
            "/v1/sessions/" + session["session_id"] + "/messages",
            headers={
                **headers,
                "Idempotency-Key": "turn-step-7",
            },
            json=turn_payload,
        )
        assert (await repeated_turn.json())["command"]["command_id"] == (
            command["command_id"]
        )
        conflicting_turn = await client.post(
            "/v1/sessions/" + session["session_id"] + "/messages",
            headers={
                **headers,
                "Idempotency-Key": "turn-step-7",
            },
            json={
                **turn_payload,
                "text": "A different request.",
            },
        )
        assert conflicting_turn.status == 409
        user_events = [
            event
            for event in service.store.events(session["session_id"])
            if event.event_type == "user.message"
        ]
        assert len(user_events) == 1

        forked = await client.post(
            "/v1/sessions/" + session["session_id"] + "/fork",
            headers={
                **headers,
                "Idempotency-Key": "fork-job-42",
            },
            json={"name": "Independent fork"},
        )
        forked_session = (await forked.json())["session"]
        assert forked_session["external_ref"] == {}
    finally:
        await client.close()
        service.close()


@pytest.mark.asyncio
async def test_reconciliation_resolutions_are_serialized() -> None:
    class ResolutionProbe(HarnessService):
        def __init__(self) -> None:
            self._reconciliation_resolution_lock = asyncio.Lock()
            self.active_resolutions = 0
            self.maximum_active_resolutions = 0

        async def _resolve_reconciliation(
            self,
            reconciliation_id: str,
            payload: dict[str, object],
        ) -> dict[str, object]:
            del payload
            self.active_resolutions += 1
            self.maximum_active_resolutions = max(
                self.maximum_active_resolutions,
                self.active_resolutions,
            )
            await asyncio.sleep(0)
            self.active_resolutions -= 1
            return {"reconciliation_id": reconciliation_id}

    service = ResolutionProbe()
    results = await asyncio.gather(
        service.resolve_reconciliation(new_uuid(), {}),
        service.resolve_reconciliation(new_uuid(), {}),
    )

    assert len(results) == 2
    assert service.maximum_active_resolutions == 1
