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
    assert _object_tuple(None) == ()
    with pytest.raises(ValueError, match="object"):
        _event_data(["[]"])


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


def test_typed_sdk_builds_safety_and_lease_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = AgentHarnessClient(paths(tmp_path / "state"))
    requests: list[tuple[str, str, object]] = []

    async def request(
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ):
        del idempotency_key
        requests.append((method, path, payload))
        if method == "GET":
            return {"leases": [{"lease_id": "lease-1"}]}
        return {"lease": {"lease_id": "lease-1"}}

    monkeypatch.setattr(client.raw, "request", request)

    created = asyncio.run(
        client.create_process_lease(
            "codex",
            session_id="session-1",
        )
    )
    attached = asyncio.run(
        client.update_process_lease(
            "lease-1",
            action="attach",
            pid=123,
            pid_start="456",
        )
    )
    leases = asyncio.run(client.process_leases())

    assert created["lease_id"] == "lease-1"
    assert attached["lease_id"] == "lease-1"
    assert leases == ({"lease_id": "lease-1"},)
    assert requests == [
        (
            "POST",
            "/v1/leases",
            {
                "provider": "codex",
                "session_id": "session-1",
                "execution_profile": "unattended",
            },
        ),
        (
            "PATCH",
            "/v1/leases/lease-1",
            {
                "action": "attach",
                "pid": 123,
                "pid_start": "456",
            },
        ),
        ("GET", "/v1/leases", None),
    ]


def test_typed_sdk_covers_the_complete_control_plane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = AgentHarnessClient(paths(tmp_path / "state"))
    requests: list[tuple[str, str, object, str]] = []

    async def request(
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ):
        requests.append((method, path, payload, idempotency_key))
        if path == "/v1/sessions" and method == "GET":
            return {"sessions": [{"session_id": "session-1"}]}
        if path == "/v1/sessions?archived=1":
            return {"sessions": [{"session_id": "session-2"}]}
        if path == "/v1/sessions" and method == "POST":
            return {"session": {"session_id": "session-1"}}
        if path.endswith("/ui-state"):
            return {"ui_state": {"composer": "draft"}}
        if "/events?" in path:
            return {"events": [{"sequence": 1}]}
        if path.endswith("/approvals"):
            return {"approvals": [{"approval_id": "approval-1"}]}
        if "/approvals/" in path:
            return {"resolved": True}
        if path.endswith("/goal"):
            return {"goal": {"status": "active"}}
        if path.endswith("/evidence"):
            return {"evidence": {"evidence_id": "evidence-1"}}
        if path.endswith("/checkpoints"):
            return {"checkpoint": {"checkpoint_id": "checkpoint-1"}}
        if path.endswith("/fork"):
            return {"session": {"session_id": "session-2"}}
        if path.endswith("/route"):
            return {
                "route": {
                    "provider": "codex",
                    "model": "default",
                    "effort": "high",
                    "reason": "headroom",
                }
            }
        if path.startswith("/v1/providers"):
            return {"providers": {"codex": {"ready": True}}}
        if path.endswith("/usage"):
            return {"safety": {"session": {"profile": "unattended"}}}
        if path.endswith("/budget-extensions"):
            return {"safety": {"xhigh_authorizations": 1}}
        if path.endswith("/export"):
            return {"path": str(tmp_path / "export.json")}
        if path.startswith("/v1/commands/") or "/commands/" in path:
            return {
                "command": {
                    "command_id": "command-1",
                    "status": "queued",
                }
            }
        if path.endswith("/messages"):
            return {
                "command": {
                    "command_id": "command-1",
                    "status": "queued",
                }
            }
        if path.startswith("/v1/sessions/") and method == "PATCH":
            return {"session": {"name": "configured"}}
        if path.startswith("/v1/sessions/") and method == "GET":
            return {
                "session": {"session_id": "session-1"},
                "goal": {"status": "active"},
                "approvals": [],
                "safety": {"session": {"profile": "unattended"}},
                "last_sequence": 3,
            }
        return {}

    monkeypatch.setattr(client.raw, "request", request)

    async def scenario() -> None:
        assert len(await client.list_sessions()) == 1
        assert len(
            await client.list_sessions(include_archived=True)
        ) == 1
        created = await client.create_session(
            tmp_path,
            name="test",
            goal="finish",
            constraints=("bounded",),
            predicates=({"type": "command"},),
        )
        assert created["session_id"] == "session-1"
        detail = await client.session("session-1")
        assert detail.goal == {"status": "active"}
        assert detail.last_sequence == 3
        configured = await client.configure_session(
            "session-1",
            name="configured",
            permission_mode="read-only",
            execution_profile="live-smoke",
        )
        assert configured["name"] == "configured"
        assert await client.ui_state("session-1") == {
            "composer": "draft"
        }
        assert await client.update_ui_state(
            "session-1", {"composer": "draft"}
        ) == {"composer": "draft"}
        assert (
            await client.events("session-1", after=1, limit=2)
        ).events[0]["sequence"] == 1
        assert (
            await client.send_message("session-1", "continue")
        ).command_id == "command-1"
        assert (
            await client.command("session-1", "pause")
        ).command_id == "command-1"
        assert (
            await client.command_status("command-1")
        ).status == "queued"
        assert len(await client.approvals("session-1")) == 1
        assert await client.resolve_approval(
            "session-1", "approval-1", "approve"
        )
        assert (await client.goal("session-1"))["goal"]["status"] == (
            "active"
        )
        evidence = await client.add_evidence(
            "session-1",
            evidence_type="command",
            subject="make test",
            outcome="passed",
        )
        assert evidence["evidence_id"] == "evidence-1"
        assert (
            await client.checkpoint("session-1")
        )["checkpoint_id"] == "checkpoint-1"
        assert (
            await client.fork("session-1", name="fork")
        )["session_id"] == "session-2"
        route = await client.preview_route(
            "session-1",
            provider="codex",
            model="default",
            effort="high",
            required_capabilities=("tools",),
            metered_budget=1.0,
        )
        assert route.provider == "codex"
        assert (
            await client.providers(tmp_path)
        )["providers"]["codex"]["ready"]
        assert (
            await client.usage("session-1")
        )["session"]["profile"] == "unattended"
        assert (
            await client.extend_budget(
                "session-1",
                reason="bounded",
                additional_seconds=60,
                additional_tokens=1_000,
                allow_xhigh_once=True,
            )
        )["xhigh_authorizations"] == 1
        assert await client.export("session-1") == (
            tmp_path / "export.json"
        )

    asyncio.run(scenario())

    assert any("?archived=1" in item[1] for item in requests)
    assert any(
        item[1].endswith("/budget-extensions")
        and bool(item[3])
        for item in requests
    )
