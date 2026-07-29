import asyncio
from pathlib import Path

import pytest

from agent_harness.config import paths
from agent_harness.sdk import AgentHarnessClient
from agent_harness.sdk import _command_view
from agent_harness.sdk import _event_data
from agent_harness.sdk import _object_tuple


def test_typed_sdk_projection_helpers() -> None:
    command = _command_view(
        {
            "command_id": "command-1",
            "status": "queued",
        }
    )

    assert command.command_id == "command-1"
    assert command.status == "queued"
    assert _object_tuple([{"value": 1}, "ignored"]) == ({"value": 1},)
    assert _event_data(['{"sequence":42}']) == {"sequence": 42}


def test_typed_sdk_builds_message_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = AgentHarnessClient(paths(tmp_path / "state"))
    captured: dict[str, object] = {}

    async def request(
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ):
        captured.update(
            {
                "method": method,
                "path": path,
                "payload": payload,
                "idempotency_key": idempotency_key,
            }
        )
        return {
            "command": {
                "command_id": "command-1",
                "status": "queued",
            }
        }

    monkeypatch.setattr(client.raw, "request", request)

    command = asyncio.run(
        client.send_message(
            "session-1",
            "continue",
            provider="codex",
            effort="xhigh",
            idempotency_key="request-1",
        )
    )

    assert command.command_id == "command-1"
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/sessions/session-1/messages"
    assert captured["payload"] == {
        "text": "continue",
        "provider": "codex",
        "model": "",
        "effort": "xhigh",
        "workload": "implementation",
    }
    assert captured["idempotency_key"] == "request-1"
