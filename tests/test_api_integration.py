from pathlib import Path
import subprocess

from aiohttp.test_utils import TestClient
from aiohttp.test_utils import TestServer
import pytest

from agent_harness.api import create_app
from agent_harness.config import CONTROL_BUILD_ID
from agent_harness.config import CONTROL_PROTOCOL_VERSION
from agent_harness.config import paths
from agent_harness.errors import SafetyGuardError
from agent_harness.service import HarnessService


class Workers:
    def __init__(self) -> None:
        self.started: list[str] = []

    def ensure(self, session_id: str) -> None:
        self.started.append(session_id)

    def stop_all(self) -> None:
        return


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
                "provider": "codex",
            },
        )
        assert saved_ui.status == 200
        restored_ui = await client.get(
            "/v1/sessions/" + session_id + "/ui-state",
            headers=headers,
        )
        assert (await restored_ui.json())["ui_state"] == {
            "composer": "unfinished message",
            "provider": "codex",
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
