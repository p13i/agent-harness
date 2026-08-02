import asyncio
from pathlib import Path

import pytest

from agent_harness import sdk as sdk_module
from agent_harness.config import paths
from agent_harness.sdk import (
    AgentHarnessClient,
    _command_view,
    _event_data,
    _object,
    _object_tuple,
)


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
    assert _object("ignored") == {}
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
            permission_mode="read-only",
            safety_limits={"max_attempts": 2, "max_seconds": 300},
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
        "permission_mode": "read-only",
        "safety_limits": {"max_attempts": 2, "max_seconds": 300},
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
        if path == "/v1/capabilities":
            return {"api_version": "1.11.0"}
        if path == "/v1/sessions" and method == "GET":
            return {"sessions": [{"session_id": "session-1"}]}
        if path == "/v1/sessions?archived=1":
            return {"sessions": [{"session_id": "session-2"}]}
        if "external_orchestrator=" in path:
            return {"sessions": [{"session_id": "external-session"}]}
        if path == "/v1/sessions" and method == "POST":
            return {"session": {"session_id": "session-1"}}
        if path.endswith("/archive"):
            return {"session": {"archived": True}}
        if path.endswith("/unarchive"):
            return {"session": {"archived": False}}
        if path.endswith("/ui-state"):
            return {"ui_state": {"composer": "draft"}}
        if "/events?" in path:
            return {"events": [{"sequence": 1}]}
        if "/proof?" in path:
            return {"proof": {"schema": "p13i/agent-harness/proof-snapshot/v1"}}
        if "/checkpoints/" in path and "/diff?" in path:
            return {
                "diff": {
                    "checkpoint_id": "checkpoint-1",
                    "content": "@@ -1 +1 @@",
                }
            }
        if "/turns/" in path:
            return {"turn": {"turn_id": "turn-1"}}
        if "/turns?" in path:
            return {"turns": [{"turn_id": "turn-1"}]}
        if path.endswith("/approvals"):
            return {"approvals": [{"approval_id": "approval-1"}]}
        if "/approvals/" in path:
            return {"resolved": True}
        if path.endswith("/goal"):
            return {"goal": {"status": "active"}}
        if path.endswith("/goal/promotions"):
            return {"promotion": {"promotion_id": "promotion-1"}}
        if path.endswith("/contract-adoptions"):
            return {"adoption": {"adoption_id": "adoption-1"}}
        if path.endswith("/dispatch-invalidations"):
            return {"invalidation": {"invalidation_id": "invalidation-1"}}
        if path.endswith("/dispatch-transition-anchor"):
            return {
                "transition_anchor": {
                    "eligible": True,
                    "prior_anchor_kind": "resolved-reconciliation",
                }
            }
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
        if path.endswith("/reconciliations"):
            return {"reconciliations": [{"reconciliation_id": "reconciliation-1"}]}
        if path.startswith("/v1/reconciliations/"):
            return {
                "reconciliation": {
                    "reconciliation_id": "reconciliation-1",
                    "resolution": "accept-current",
                }
            }
        if path == "/v1/sync":
            return {
                "state_root": str(tmp_path / "chats"),
                "sync": {"state": "synced"},
            }
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
        assert (await client.capabilities())["api_version"] == "1.11.0"
        assert len(await client.list_sessions()) == 1
        assert len(await client.list_sessions(include_archived=True)) == 1
        external = await client.session_by_external_ref(
            "p13i/machines",
            "job-1",
        )
        assert external is not None
        assert external["session_id"] == "external-session"
        created = await client.create_session(
            tmp_path,
            name="test",
            goal="finish",
            constraints=("bounded",),
            predicates=({"type": "command"},),
        )
        assert created["session_id"] == "session-1"
        ensured = await client.ensure_session(
            tmp_path,
            orchestrator="p13i/machines",
            job_id="job-1",
            idempotency_key="ensure-1",
        )
        assert ensured["session_id"] == "session-1"
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
        assert (
            await client.set_archived(
                "session-1",
                True,
                idempotency_key="archive-1",
            )
        )["archived"]
        assert not (
            await client.set_archived(
                "session-1",
                False,
            )
        )["archived"]
        assert await client.ui_state("session-1") == {"composer": "draft"}
        assert await client.update_ui_state("session-1", {"composer": "draft"}) == {
            "composer": "draft"
        }
        assert (await client.events("session-1", after=1, limit=2)).events[0][
            "sequence"
        ] == 1
        assert (
            await client.proof(
                "session-1",
                after_sequence=2,
                event_limit=3,
                through_sequence=7,
                snapshot_id="snapshot-1",
            )
        )["schema"] == "p13i/agent-harness/proof-snapshot/v1"
        assert (
            await client.turns(
                "session-1",
                after_sequence=2,
                limit=3,
            )
        )["turns"][0]["turn_id"] == "turn-1"
        assert (await client.turn("session-1", "turn-1"))["turn"]["turn_id"] == "turn-1"
        assert (
            await client.checkpoint_diff(
                "session-1",
                "checkpoint-1",
                start_line=4,
                limit=12,
            )
        )["checkpoint_id"] == "checkpoint-1"
        assert (
            await client.send_message("session-1", "continue")
        ).command_id == "command-1"
        assert (
            await client.submit_managed_turn(
                "session-1",
                "managed",
                step_id="step-1",
                agent_role="implementer",
                idempotency_key="turn-1",
            )
        ).command_id == "command-1"
        assert (await client.command("session-1", "pause")).command_id == "command-1"
        assert (await client.command_status("command-1")).status == "queued"
        assert len(await client.approvals("session-1")) == 1
        assert await client.resolve_approval("session-1", "approval-1", "approve")
        assert (await client.goal("session-1"))["goal"]["status"] == ("active")
        promotion = await client.promote_goal(
            "session-1",
            from_goal_id="goal-1",
            stage="tier-24h",
            objective="Prove the 24-hour tier.",
            constraints=["bounded"],
            predicates=[{"type": "report", "subject": "tier-24h"}],
            milestones=[
                {
                    "milestone_id": "tier-24h",
                    "title": "Prove tier 24h",
                    "dependencies": [],
                    "predicates": [{"type": "report", "subject": "tier-24h"}],
                }
            ],
            budgets={"seconds": 86_400},
            authorization={
                "schema": ("p13i/agent-harness/goal-promotion-authorization/v1"),
                "receipt": {"scope": "test"},
                "receipt_sha256": "a" * 64,
                "next_goal_contract_digest": "b" * 64,
            },
            idempotency_key="promote-24h",
        )
        assert promotion["promotion"]["promotion_id"] == "promotion-1"
        unconstrained = await client.promote_goal(
            "session-1",
            from_goal_id="goal-1",
            stage="tier-48h",
            objective="Prove the 48-hour tier.",
            predicates=[{"type": "report", "subject": "tier-48h"}],
            milestones=[
                {
                    "milestone_id": "tier-48h",
                    "title": "Prove tier 48h",
                    "dependencies": [],
                    "predicates": [{"type": "report", "subject": "tier-48h"}],
                }
            ],
            budgets={"seconds": 172_800},
            authorization={
                "schema": ("p13i/agent-harness/goal-promotion-authorization/v1"),
                "receipt": {"scope": "test"},
                "receipt_sha256": "a" * 64,
                "next_goal_contract_digest": "c" * 64,
            },
            idempotency_key="promote-48h",
        )
        assert unconstrained["promotion"]["promotion_id"] == "promotion-1"
        adoption = await client.adopt_session_contract(
            "session-1",
            tmp_path,
            name="Managed session",
            goal="Keep the service healthy.",
            goal_kind="invariant",
            constraints=["Pause after an incident."],
            predicates=[{"type": "probe", "outcome": "passed"}],
            milestones=[
                {
                    "milestone_id": "installed",
                    "title": "Install service",
                    "dependencies": [],
                    "predicates": [{"type": "probe", "outcome": "passed"}],
                }
            ],
            budgets={"seconds": 86_400},
            permitted_providers=["claude", "codex"],
            permitted_efforts=["low", "medium"],
            max_concurrency=1,
            completion_policy="never",
            incident_policy="recover-then-pause",
            permission_mode="full",
            direct=True,
            execution_profile="unattended",
            external_ref={
                "orchestrator": "p13i/machines/cs-sre",
                "job_id": "cs-sre",
            },
            authorization={"schema": "typed"},
            idempotency_key="adopt-cs-sre",
        )
        assert adoption["adoption"]["adoption_id"] == "adoption-1"
        invalidation = await client.invalidate_dispatch_generation(
            "session-1",
            reason="Material state was independently reviewed.",
            authorization={"schema": "typed"},
            idempotency_key="invalidate-reviewed-state",
            prior_command_id="command-1",
            prior_command_type="message",
            prior_anchor_kind="resolved-reconciliation",
            prior_reconciliation_id="reconciliation-1",
            prior_reconciliation_resolution="accept-current",
            prior_checkpoint_id="checkpoint-1",
            prior_generation_digest="a" * 64,
            prior_material_digest="c" * 64,
            next_turn_ref={"step_id": "verify", "agent_role": "verifier"},
            transition_sequence=1,
            next_command_digest="b" * 64,
        )
        assert invalidation["invalidation_id"] == "invalidation-1"
        transition_anchor = await client.dispatch_transition_anchor("session-1")
        assert transition_anchor["eligible"] is True
        assert transition_anchor["prior_anchor_kind"] == ("resolved-reconciliation")
        evidence = await client.add_evidence(
            "session-1",
            evidence_type="command",
            subject="make test",
            outcome="passed",
        )
        assert evidence["evidence_id"] == "evidence-1"
        assert (await client.checkpoint("session-1"))["checkpoint_id"] == "checkpoint-1"
        assert (
            await client.fork(
                "session-1",
                name="fork",
                external_ref={
                    "orchestrator": "test",
                    "job_id": "fork-job",
                },
            )
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
        assert (await client.providers(tmp_path))["providers"]["codex"]["ready"]
        assert (await client.usage("session-1"))["session"]["profile"] == "unattended"
        assert len(await client.reconciliations("session-1")) == 1
        assert (await client.reconciliation("reconciliation-1"))[
            "reconciliation_id"
        ] == "reconciliation-1"
        assert (
            await client.resolve_reconciliation(
                "reconciliation-1",
                decision="accept-current",
                observed_workspace_digest="digest-1",
                idempotency_key="reconcile-1",
                approval_id="approval-1",
                audit={"orchestrator": "test"},
            )
        )["resolution"] == "accept-current"
        assert (
            await client.extend_budget(
                "session-1",
                reason="bounded",
                additional_seconds=60,
                additional_tokens=1_000,
                allow_xhigh_once=True,
                command_id="00000000-0000-4000-8000-000000000001",
                provider="codex",
            )
        )["xhigh_authorizations"] == 1
        assert (await client.sync_status())["sync"]["state"] == "synced"
        assert (await client.sync())["sync"]["state"] == "synced"
        assert await client.export("session-1") == (tmp_path / "export.json")

    asyncio.run(scenario())

    assert any("?archived=1" in item[1] for item in requests)
    assert any(
        item[1].endswith("/messages")
        and isinstance(item[2], dict)
        and item[2].get("turn_ref")
        == {
            "step_id": "step-1",
            "agent_role": "implementer",
        }
        for item in requests
    )
    assert any(
        item[1].endswith("/budget-extensions") and bool(item[3]) for item in requests
    )
    assert any(
        item[1] == "/v1/sessions/session-1/turns?after_sequence=2&limit=3"
        for item in requests
    )
    assert any(
        item[1]
        == (
            "/v1/sessions/session-1/proof?after_sequence=2"
            "&event_limit=3&through_sequence=7&snapshot_id=snapshot-1"
        )
        for item in requests
    )
    assert any(
        item[1]
        == (
            "/v1/sessions/session-1/checkpoints/checkpoint-1/diff?start_line=4&limit=12"
        )
        for item in requests
    )
    assert any(item[1].endswith("/unarchive") and bool(item[3]) for item in requests)
    assert any(
        item[1].startswith("/v1/reconciliations/")
        and isinstance(item[2], dict)
        and item[2].get("approval_id") == "approval-1"
        for item in requests
    )
    transition_requests = [
        item for item in requests if item[1].endswith("/dispatch-invalidations")
    ]
    assert len(transition_requests) == 1
    transition_payload = transition_requests[0][2]
    assert isinstance(transition_payload, dict)
    assert transition_payload["prior_command_type"] == "message"
    assert transition_payload["prior_anchor_kind"] == "resolved-reconciliation"
    assert transition_payload["prior_reconciliation_id"] == "reconciliation-1"
    assert transition_payload["prior_material_digest"] == "c" * 64


def test_managed_sdk_requires_keys_and_waits_boundedly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = AgentHarnessClient(paths(tmp_path / "state"))
    statuses = ["queued", "complete"]

    async def request(
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ):
        del method
        del path
        del payload
        del idempotency_key
        return {
            "command": {
                "command_id": "command-1",
                "status": statuses.pop(0),
            }
        }

    monkeypatch.setattr(client.raw, "request", request)

    async def scenario() -> None:
        with pytest.raises(ValueError, match="idempotency"):
            await client.ensure_session(
                tmp_path,
                orchestrator="test",
                job_id="job",
                idempotency_key="",
            )
        with pytest.raises(ValueError, match="idempotency"):
            await client.submit_managed_turn(
                "session-1",
                "continue",
                step_id="step-1",
                agent_role="implementer",
                idempotency_key="",
            )
        with pytest.raises(ValueError, match="idempotency"):
            await client.resolve_reconciliation(
                "reconciliation-1",
                decision="stop",
                observed_workspace_digest="digest-1",
                idempotency_key="",
            )
        with pytest.raises(ValueError, match="both required"):
            await client.list_sessions(
                external_orchestrator="test",
            )
        command = await client.wait_command(
            "command-1",
            timeout=1,
            poll_interval=0.001,
        )
        assert command.status == "complete"
        with pytest.raises(ValueError, match="timeout"):
            await client.wait_command("command-1", timeout=0)
        with pytest.raises(ValueError, match="poll interval"):
            await client.wait_command(
                "command-1",
                timeout=1,
                poll_interval=0,
            )
        statuses.append("queued")
        command = await client.wait_command(
            "command-1",
            timeout=0.000000001,
            poll_interval=0.001,
        )
        assert command.status == "queued"

    asyncio.run(scenario())


def test_sdk_external_lookup_handles_absence_and_conflicting_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = AgentHarnessClient(paths(tmp_path / "state"))
    results: list[dict[str, object]] = [
        {"sessions": []},
        {
            "sessions": [
                {"session_id": "one"},
                {"session_id": "two"},
            ]
        },
    ]

    async def request(
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ):
        del method
        del path
        del payload
        del idempotency_key
        return results.pop(0)

    monkeypatch.setattr(client.raw, "request", request)

    async def scenario() -> None:
        assert await client.session_by_external_ref("test", "missing") is None
        with pytest.raises(RuntimeError, match="not unique"):
            await client.session_by_external_ref("test", "duplicate")

    asyncio.run(scenario())


def test_sdk_stream_events_resumes_and_decodes_sse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = AgentHarnessClient(paths(tmp_path / "state"))
    captured: dict[str, object] = {}

    class Content:
        def __aiter__(self):
            self.values = iter(
                (
                    b": keepalive\n",
                    b'data: {"sequence": 7}\n',
                    b"\n",
                )
            )
            return self

        async def __anext__(self):
            try:
                return next(self.values)
            except StopIteration as error:
                raise StopAsyncIteration from error

    class Response:
        content = Content()

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exception_type,
            exception,
            traceback,
        ) -> None:
            del exception_type
            del exception
            del traceback

        def raise_for_status(self) -> None:
            captured["raised"] = True

    class Session:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exception_type,
            exception,
            traceback,
        ) -> None:
            del exception_type
            del exception
            del traceback

        def get(self, path: str) -> Response:
            captured["path"] = path
            return Response()

    monkeypatch.setattr(
        sdk_module,
        "UnixConnector",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(sdk_module, "ClientSession", Session)

    async def scenario() -> None:
        events = [
            event
            async for event in client.stream_events(
                "session-1",
                after=6,
            )
        ]
        assert events == [{"sequence": 7}]

    asyncio.run(scenario())
    assert captured["headers"] == {
        "Authorization": "Bearer " + client.raw.token,
        "Last-Event-ID": "6",
    }
    assert captured["path"] == "/v1/sessions/session-1/stream"
    assert captured["raised"]
