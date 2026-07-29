from pathlib import Path
import subprocess

from aiohttp.test_utils import TestClient
from aiohttp.test_utils import TestServer
import pytest

from agent_harness.api import create_app
from agent_harness.config import paths
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
            json={"workspace": str(workspace)},
        )
        assert created.status == 201
        session = (await created.json())["session"]
        session_id = session["session_id"]
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
    finally:
        await client.close()
        service.close()
