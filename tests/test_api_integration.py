import asyncio
import copy
import hashlib
import json
import subprocess
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

import agent_harness.proof as proof_module
import agent_harness.safety as safety_module
import agent_harness.storage as storage_module
from agent_harness import api as api_module
from agent_harness import service as service_module
from agent_harness.api import create_app
from agent_harness.config import CONTROL_BUILD_ID, CONTROL_PROTOCOL_VERSION, paths
from agent_harness.errors import ConflictError, NotFoundError, SafetyGuardError
from agent_harness.goals import create_goal, goal_contract_digest, make_evidence
from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import ProviderAttempt, Session
from agent_harness.orchestration import command_envelope_digest, normalized_digest
from agent_harness.reconciliation import inspect_workspace
from agent_harness.service import HarnessService
from agent_harness.storage import StateStore
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
async def test_api_records_and_replays_operator_usage_attestation(
    tmp_path: Path,
) -> None:
    service = HarnessService(
        paths(tmp_path / "state"),
        worker_manager=Workers(),
    )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {
        "Authorization": "Bearer test-token",
        "Idempotency-Key": "claude-usage-outage",
    }
    payload = {
        "binding_percent": 44.0,
        "credits_engaged": False,
        "valid_seconds": 3600,
        "evidence_sha256": "c" * 64,
    }
    try:
        created = await client.post(
            "/v1/providers/claude/usage-attestations",
            headers=headers,
            json=payload,
        )
        assert created.status == 201
        created_value = await created.json()
        replay = await client.post(
            "/v1/providers/claude/usage-attestations",
            headers=headers,
            json=payload,
        )
        assert replay.status == 201
        assert await replay.json() == created_value
        providers = await service.scheduler.status(tmp_path)
        assert providers["claude"]["usage"]["admissible"] is True
        assert (
            providers["claude"]["usage"]["payload"]["source"]
            == "operator-attestation"
        )
        rejected = await client.post(
            "/v1/providers/unknown/usage-attestations",
            headers={
                "Authorization": "Bearer test-token",
                "Idempotency-Key": "invalid-usage-attestation",
            },
            json=payload,
        )
        assert rejected.status == 400
    finally:
        refresh = service.scheduler._status_refresh
        if refresh is not None:
            refresh.cancel()
            await asyncio.gather(refresh, return_exceptions=True)
        await client.close()
        service.close()


@pytest.mark.asyncio
async def test_api_projects_canonical_transcript(
    tmp_path: Path,
) -> None:
    service = HarnessService(
        paths(tmp_path / "state"),
        worker_manager=Workers(),
    )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    now = utc_now()
    created = Session(
        session_id=new_uuid(),
        name="transcript",
        workspace=str(tmp_path),
        worktree=str(tmp_path),
        lifecycle="running",
        attention="idle",
        permission_mode="approval",
        active_provider="",
        model="",
        effort="",
        goal_id="",
        owner_host="test-host",
        owner_epoch=1,
        created_at=now,
        updated_at=now,
    )
    service.store.create_session(created)
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="claude",
        native_session_id="",
        model="account-default",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    service.store.create_attempt(attempt)
    turn_id = service.store.start_turn(created.session_id, attempt.attempt_id)
    service.store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="fix the flaky test",
        turn_id=turn_id,
    )
    service.store.append_event(
        created.session_id,
        "agent.message",
        role="assistant",
        text="on it",
        turn_id=turn_id,
    )
    service.store.finish_turn(turn_id, "complete")
    try:
        response = await client.get(
            "/v1/sessions/" + created.session_id + "/transcript",
            headers=headers,
        )
        assert response.status == 200
        body = await response.json()
        transcript = body["transcript"]
        assert transcript["schema"] == ("p13i/agent-harness/transcript/v1")
        assert [entry["role"] for entry in transcript["entries"]] == [
            "user",
            "assistant",
        ]
        assert transcript["entries"][0]["provider"] == "claude"
        assert len(transcript["digest"]) == 64
        assert body["rendered"].startswith("# Session transcript")

        tailored = await client.get(
            "/v1/sessions/"
            + created.session_id
            + "/transcript?tail=0&token_budget=64",
            headers=headers,
        )
        assert tailored.status == 200

        invalid = await client.get(
            "/v1/sessions/" + created.session_id + "/transcript?tail=-1",
            headers=headers,
        )
        assert invalid.status == 400

        missing = await client.get(
            "/v1/sessions/" + new_uuid() + "/transcript",
            headers=headers,
        )
        assert missing.status == 404
    finally:
        refresh = service.scheduler._status_refresh
        if refresh is not None:
            refresh.cancel()
            await asyncio.gather(refresh, return_exceptions=True)
        await client.close()
        service.close()


def machines_session_payload(
    workspace: Path,
    external_ref: dict[str, str],
    **overrides: object,
) -> dict[str, object]:
    job_id = external_ref["job_id"]
    payload: dict[str, object] = {
        "workspace": str(workspace),
        "direct": True,
        "execution_profile": "unattended",
        "external_ref": external_ref,
        "goal": "Complete the bounded Machines stage.",
        "goal_kind": "finite",
        "constraints": ["Preserve the Machines execution receipt."],
        "predicates": [
            {
                "type": "machines-proof",
                "subject": job_id,
                "outcome": "passed",
            }
        ],
        "milestones": [
            {
                "milestone_id": job_id + "-complete",
                "title": "Complete " + job_id,
                "dependencies": [],
                "predicates": [
                    {
                        "type": "machines-proof",
                        "subject": job_id,
                        "outcome": "passed",
                    }
                ],
            }
        ],
        "budgets": {
            "seconds": 300,
            "turns": 2,
            "tokens": 20_000,
            "context_tokens": 16_000,
            "output_tokens": 4_000,
            "tool_calls": 10,
            "attempts": 2,
            "child_agents": 1,
            "dollars": 0,
        },
        "permitted_providers": ["claude", "codex"],
        "permitted_efforts": ["low", "medium"],
        "max_concurrency": 1,
        "completion_policy": "evidence-all",
        "incident_policy": "recover-then-pause",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_api_inspects_and_resolves_reconciliation(
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
                "external_ref": {
                    "orchestrator": "machines",
                    "job_id": "proof-job",
                },
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
            native_session_id="codex-native-session",
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
            "/v1/sessions/" + session_id + "/turns?after_sequence=0&limit=10",
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
        assert (await turn_detail.json())["turn"]["turn_ids"] == [turn_id]
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
        unauthorized_turns = await client.get("/v1/sessions/" + session_id + "/turns")
        assert unauthorized_turns.status == 401
        invalid_turn_page = await client.get(
            "/v1/sessions/" + session_id + "/turns?after_sequence=-1",
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
        service.store.record_usage(
            "codex",
            25.0,
            False,
            {"source": "integration-test"},
        )
        usage = service.store.latest_usage()["codex"]
        routing_payload = {
            "provider": "codex",
            "model": "default",
            "effort": "high",
            "usage_sample_id": usage["sample_id"],
            "usage_observed_at": usage["observed_at"],
            "binding_percent": 25.0,
            "credits_engaged": False,
            "required_capabilities": ["resume", "tools"],
            "metered_budget": None,
            "binding_ceiling": 85.0,
            "execution_profile": "unattended",
            "workload": "implementation",
        }
        decision_id = service.store.record_routing(
            session_id,
            turn_id,
            "codex",
            "default",
            "high",
            routing_payload,
        )
        service.store.record_usage(
            "codex",
            45.0,
            False,
            {
                "payload": {"window": "later"},
                "error": "later probe failed",
            },
        )
        latest_usage = service.store.latest_usage()["codex"]
        historical_observed_at = "2020-01-01T00:00:00+00:00"
        historical_admitted_at = "2020-01-01T00:00:15+00:00"
        historical_route_at = "2020-01-01T00:00:30+00:00"
        routing_payload["usage_observed_at"] = historical_observed_at
        with service.store.transaction() as connection:
            connection.execute(
                "UPDATE usage_samples SET observed_at = ? WHERE sample_id = ?",
                (historical_observed_at, usage["sample_id"]),
            )
            connection.execute(
                "UPDATE provider_attempts SET started_at = ? WHERE attempt_id = ?",
                (historical_admitted_at, attempt.attempt_id),
            )
            connection.execute(
                """
                UPDATE routing_decisions SET created_at = ?, payload_json = ?
                WHERE decision_id = ?
                """,
                (
                    historical_route_at,
                    json.dumps(
                        routing_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    decision_id,
                ),
            )
        secret_subject = str(workspace) + "/token=credential-value"
        goal = create_goal(
            session_id,
            "Complete the bounded test.",
            constraints=("Never disclose credential-value",),
            predicates=(
                {
                    "type": "test",
                    "subject": secret_subject,
                    "outcome": "passed",
                    "field": "exit_code",
                    "equals": 0,
                },
            ),
            milestones=(
                {
                    "milestone_id": "artifact-sealed",
                    "title": "Seal artifact",
                    "dependencies": [],
                    "predicates": [
                        {
                            "type": "artifact",
                            "subject": "report",
                            "outcome": "passed",
                            "field": "digest",
                            "equals": "sha256:report",
                        }
                    ],
                },
            ),
            budgets={"turns": 2},
        )
        service.store.create_goal(goal)
        service.store.add_evidence(
            make_evidence(
                goal.goal_id,
                "test",
                secret_subject,
                "passed",
                {"exit_code": 0, "secret": "credential-value"},
            )
        )
        service.store.add_evidence(
            make_evidence(
                goal.goal_id,
                "artifact",
                "report",
                "passed",
                {"digest": "sha256:report"},
            )
        )
        service.store.append_event(
            session_id,
            "budget.extended",
            status="complete",
            metadata={"reason": "credential-value in " + str(workspace)},
        )
        for event_type, metadata in (
            (
                "agent.child.started",
                {
                    "id": "child-1",
                    "receiverThreadIds": ["thread-2"],
                    "usage": {
                        "input_tokens": 2,
                        "secret": "credential-value",
                    },
                    "prompt": "credential-value",
                },
            ),
            (
                "agent.child.completed",
                {
                    "id": "child-1",
                    "usage": {"total_tokens": 4},
                    "result": "credential-value",
                },
            ),
            (
                "agent.child.started",
                {"child_id": "child-2"},
            ),
            (
                "agent.child.completed",
                {
                    "child_id": "child-2",
                    "usage": {"total_tokens": 5},
                },
            ),
            (
                "agent.child.started",
                {"child_id": "child-3"},
            ),
            (
                "agent.child.failed",
                {
                    "child_id": "child-3",
                    "usage": {"total_tokens": 6},
                },
            ),
            (
                "agent.child.started",
                {"child_id": "child-4"},
            ),
            (
                "agent.child.cancelled",
                {"child_id": "child-4"},
            ),
        ):
            service.store.append_event(
                session_id,
                event_type,
                status="complete",
                metadata=metadata,
                text="credential-value",
                turn_id=turn_id,
            )
        service.store.prepare_context_delivery(
            session_id,
            "codex",
            checkpoint.context_digest,
            checkpoint.checkpoint_id,
            command_id,
            attempt.attempt_id,
            checkpoint.context_digest,
        )
        service.store.accept_context_delivery(
            session_id,
            "codex",
            checkpoint.context_digest,
            attempt.attempt_id,
        )
        lease = service.store.create_process_lease(
            session_id,
            "codex",
            "unattended",
            "2099-01-01T00:00:00+00:00",
        )
        service.store.update_process_lease(
            lease["lease_id"],
            pid=123,
            pid_start="456",
            state="active",
        )

        proof_response = await client.get(
            "/v1/sessions/" + session_id + "/proof",
            headers=headers,
        )
        assert proof_response.status == 200
        proof = (await proof_response.json())["proof"]
        assert proof["schema"] == "p13i/agent-harness/proof-snapshot/v1"
        assert proof["complete"] is True
        assert proof["session"]["external_ref"]["job_id"] == "proof-job"
        assert proof["commands"][0]["command_id"] == command_id
        assert proof["commands"][0]["session_id"] == session_id
        assert proof["commands"][0]["turn_id"] == turn_id
        assert len(proof["commands"][0]["idempotency_key_digest"]) == 64
        assert proof["attempts"][0]["native_session_id"] == ("codex-native-session")
        assert proof["dispatches"][0]["crossed_boundary"] == 1
        assert proof["routing"][0]["command_id"] == command_id
        assert proof["routing"][0]["attempt_id"] == attempt.attempt_id
        assert proof["routing"][0]["usage_sample_id"] == usage["sample_id"]
        assert proof["routing"][0]["binding_percent"] == 25.0
        assert proof["routing"][0]["credits_engaged"] is False
        assert proof["routing"][0]["usage_sample_bound"] is True
        assert proof["routing"][0]["route_recorded_at"] == historical_route_at
        assert proof["routing"][0]["attempt_admitted_at"] == (historical_admitted_at)
        assert proof["routing"][0]["usage_age_seconds_at_route"] == 30.0
        assert proof["routing"][0]["usage_age_seconds_at_attempt_admission"] == 15.0
        assert proof["routing"][0]["fresh_at_route"] is True
        assert proof["routing"][0]["fresh_at_attempt_admission"] is True
        assert proof["routing"][0]["admissible_at_route"] is True
        assert proof["routing"][0]["required_capabilities"] == [
            "resume",
            "tools",
        ]
        assert proof["goal"]["predicates"][0]["satisfied"] is True
        assert proof["goal"]["predicates"][0]["evidence_ids"] == [
            proof["goal"]["evidence"][0]["evidence_id"]
        ]
        assert proof["goal"]["evidence"][0]["fields"]["exit_code"] == {
            "type": "integer",
            "value": 0,
        }
        artifact_evidence = next(
            item for item in proof["goal"]["evidence"] if item["type"] == "artifact"
        )
        assert artifact_evidence["fields"]["digest"] == {
            "type": "string",
            "digest": hashlib.sha256(b"sha256:report").hexdigest(),
        }
        assert proof["goal"]["milestones"][0]["predicates"][0]["satisfied"] is True
        budget_event = next(
            item for item in proof["events"] if item["event_type"] == "budget.extended"
        )
        assert len(budget_event["metadata"]["reason_digest"]) == 64
        usage_by_id = {item["sample_id"]: item for item in proof["usage"]}
        assert usage_by_id[usage["sample_id"]]["credits_engaged"] is False
        assert usage_by_id[usage["sample_id"]]["fresh_at_capture"] is False
        assert usage_by_id[usage["sample_id"]]["latest"] is False
        assert usage_by_id[usage["sample_id"]]["error_present"] is False
        assert usage_by_id[latest_usage["sample_id"]]["latest"] is True
        assert usage_by_id[latest_usage["sample_id"]]["error_present"] is True
        assert (
            usage_by_id[latest_usage["sample_id"]]["admissible_at_90_percent"] is False
        )
        assert proof["leases"][0]["pid_start"] == "456"
        assert proof["context_deliveries"][0]["context_digest"] == (
            checkpoint.context_digest
        )
        assert proof["context_deliveries"][0]["command_id"] == command_id
        assert proof["context_deliveries"][0]["turn_id"] == turn_id
        assert proof["context_deliveries"][0]["attempt_id"] == (attempt.attempt_id)
        assert proof["context_deliveries"][0]["transport"] == "context-package"
        assert len(proof["context_deliveries"][0]["idempotency_digest"]) == 64
        assert len(proof["context_deliveries"][0]["context_delivery_digest"]) == 64
        children = {item["child_id"]: item for item in proof["children"]}
        assert len(children) == 4
        assert [item["status"] for item in proof["children"]].count("completed") == 2
        assert children["child-1"]["native_thread_ids"] == ["thread-2"]
        assert children["child-1"]["provider"] == "codex"
        assert children["child-1"]["parent_command_id"] == command_id
        assert children["child-1"]["parent_turn_id"] == turn_id
        assert children["child-1"]["parent_attempt_id"] == attempt.attempt_id
        assert children["child-1"]["usage"] == {"total_tokens": 4}
        assert children["child-3"]["status"] == "failed"
        assert children["child-4"]["status"] == "cancelled"
        assert proof["reconciliations"][0]["reconciliation_id"] == (
            record.reconciliation_id
        )
        assert len(proof["event_range"]["digest"]) == 64
        canonical_events = json.dumps(
            proof["events"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert (
            hashlib.sha256(canonical_events).hexdigest()
            == proof["event_range"]["snapshot_digest"]
        )
        retained_payload = copy.deepcopy(proof)
        retained_payload.pop("snapshot_id")
        retained_payload.pop("snapshot_digest")
        retained_payload.pop("event_range")
        canonical_snapshot = json.dumps(
            retained_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert (
            hashlib.sha256(canonical_snapshot).hexdigest() == proof["snapshot_digest"]
        )
        tampered_payload = copy.deepcopy(retained_payload)
        tampered_payload["session"]["lifecycle"] = "tampered"
        tampered_snapshot = json.dumps(
            tampered_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert hashlib.sha256(tampered_snapshot).hexdigest() != proof["snapshot_digest"]
        serialized_proof = str(proof)
        assert "Make one bounded change." not in serialized_proof
        assert str(workspace) not in serialized_proof
        assert "payload_json" not in serialized_proof
        assert "credential-value" not in serialized_proof
        assert secret_subject not in serialized_proof

        retained_response = await client.get(
            "/v1/sessions/"
            + session_id
            + "/proof?snapshot_id="
            + proof["snapshot_id"]
            + "&through_sequence="
            + str(proof["event_range"]["through_sequence"]),
            headers=headers,
        )
        retained = (await retained_response.json())["proof"]
        assert retained["children"] == proof["children"]
        assert retained["snapshot_digest"] == proof["snapshot_digest"]

        proof_page = await client.get(
            "/v1/sessions/" + session_id + "/proof?after_sequence=0&event_limit=1",
            headers=headers,
        )
        paged = (await proof_page.json())["proof"]
        assert paged["complete"] is False
        assert paged["event_range"]["complete"] is False
        assert paged["truncated"] == ["events"]
        assert paged["state_stable"] is True
        assert (
            paged["event_range"]["through_sequence"]
            == paged["event_range"]["last_sequence"]
        )

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
        substituted_discovery = await client.post(
            "/v1/reconciliations/" + record.reconciliation_id + "/resolution",
            headers=resolution_headers,
            json={
                "decision": "restore-pre-turn",
                "observed_workspace_digest": digest,
                "audit": {
                    "actor": "integration-test",
                    "discovery_checkpoint_id": checkpoint.checkpoint_id,
                },
            },
        )
        assert substituted_discovery.status == 400
        assert (
            "system-owned fields"
            in (await substituted_discovery.json())["error"]["message"]
        )
        protected_record = service.store.reconciliation(record.reconciliation_id)
        assert "discovery_checkpoint_id" not in protected_record.audit
        approval_required = await client.post(
            "/v1/reconciliations/" + record.reconciliation_id + "/resolution",
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
            "/v1/sessions/" + session_id + "/approvals/" + approval_id,
            headers={
                **headers,
                "Idempotency-Key": "approve-workspace-restore",
            },
            json={"decision": "approve"},
        )
        assert (await approved.json())["resolved"]
        insert_receipt = service.store._insert_mutation_receipt

        def lose_resolution_receipt(
            *unused: object,
            **values: object,
        ) -> None:
            del unused, values
            raise RuntimeError("simulated response loss before receipt")

        monkeypatch.setattr(
            service.store,
            "_insert_mutation_receipt",
            lose_resolution_receipt,
        )
        lost_resolution = await client.post(
            "/v1/reconciliations/" + record.reconciliation_id + "/resolution",
            headers=resolution_headers,
            json={
                "decision": "restore-pre-turn",
                "observed_workspace_digest": digest,
                "approval_id": approval_id,
                "audit": {"actor": "integration-test"},
            },
        )
        assert lost_resolution.status == 500
        unresolved = service.store.reconciliation(record.reconciliation_id)
        assert unresolved.status == "resolving"
        assert "resolution_checkpoint_id" not in unresolved.audit
        assert not [
            event
            for event in service.store.all_events(session_id)
            if event.event_type == "reconciliation.resolved"
        ]
        monkeypatch.setattr(
            service.store,
            "_insert_mutation_receipt",
            insert_receipt,
        )
        ensure_worker = workers.ensure

        def lose_worker_start(unused_session_id: str) -> None:
            del unused_session_id
            raise RuntimeError("simulated crash after resolution commit")

        monkeypatch.setattr(workers, "ensure", lose_worker_start)
        committed_without_worker = await client.post(
            "/v1/reconciliations/" + record.reconciliation_id + "/resolution",
            headers=resolution_headers,
            json={
                "decision": "restore-pre-turn",
                "observed_workspace_digest": digest,
                "approval_id": approval_id,
                "audit": {"actor": "integration-test"},
            },
        )
        assert committed_without_worker.status == 500
        committed = service.store.reconciliation(record.reconciliation_id)
        assert committed.status == "resolved"
        assert (
            len(
                [
                    event
                    for event in service.store.all_events(session_id)
                    if event.event_type == "reconciliation.resolved"
                ]
            )
            == 1
        )
        monkeypatch.setattr(workers, "ensure", ensure_worker)
        replayed_resolution = await client.post(
            "/v1/reconciliations/" + record.reconciliation_id + "/resolution",
            headers=resolution_headers,
            json={
                "decision": "restore-pre-turn",
                "observed_workspace_digest": digest,
                "approval_id": approval_id,
                "audit": {"actor": "integration-test"},
            },
        )
        assert replayed_resolution.status == 200
        resolved_value = (await replayed_resolution.json())["reconciliation"]
        assert resolved_value["resolution"] == "restore-pre-turn"
        resolution_workspace_digest = inspect_workspace(
            Path(service.store.get_session(session_id).worktree)
        )[0]
        assert resolved_value["audit"]["resolution_workspace_digest"] == (
            resolution_workspace_digest
        )
        assert workers.started[-1] == session_id

        async def repeat_resolution() -> object:
            return await client.post(
                "/v1/reconciliations/" + record.reconciliation_id + "/resolution",
                headers=resolution_headers,
                json={
                    "decision": "restore-pre-turn",
                    "observed_workspace_digest": digest,
                    "approval_id": approval_id,
                    "audit": {"actor": "integration-test"},
                },
            )

        repeated_responses = await asyncio.gather(
            repeat_resolution(),
            repeat_resolution(),
        )
        repeated_values = [
            (await response.json())["reconciliation"] for response in repeated_responses
        ]
        assert repeated_values == [resolved_value, resolved_value]
        new_key_replay = await client.post(
            "/v1/reconciliations/" + record.reconciliation_id + "/resolution",
            headers={
                **headers,
                "Idempotency-Key": "restore-ambiguous-turn-new-receipt",
            },
            json={
                "decision": "restore-pre-turn",
                "observed_workspace_digest": digest,
                "approval_id": approval_id,
                "audit": {"actor": "ignored-new-receipt-replay"},
            },
        )
        assert new_key_replay.status == 200
        assert (await new_key_replay.json())["reconciliation"] == resolved_value
        resolution_events = [
            event
            for event in service.store.all_events(session_id)
            if event.event_type == "reconciliation.resolved"
        ]
        assert len(resolution_events) == 1
        assert (
            resolution_events[0].metadata["topology_receipt"]
            == (resolved_value["audit"]["topology_receipt"])
        )
        assert resolution_events[0].metadata["resolution_workspace_digest"] == (
            resolution_workspace_digest
        )
        resolution_checkpoints = [
            item
            for item in service.store.checkpoints(session_id)
            if item.checkpoint_id == resolved_value["audit"]["resolution_checkpoint_id"]
        ]
        assert len(resolution_checkpoints) == 1
    finally:
        await client.close()
        service.close()


def _single_transition_policy(
    session_id: str,
    external_ref: dict[str, str],
    epoch_id: str,
    next_turn_ref: dict[str, str],
    next_command_digest: str,
) -> dict[str, object]:
    return {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": session_id,
        "external_ref": external_ref,
        "epoch_id": epoch_id,
        "allowed_agent_roles": [next_turn_ref["agent_role"]],
        "allowed_step_prefixes": [next_turn_ref["step_id"]],
        "max_transitions": 1,
        "transitions": [
            {
                "sequence": 1,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": next_command_digest,
            }
        ],
    }


def _managed_transition_payload(
    service: HarnessService,
    session_id: str,
    policy: dict[str, object],
    next_turn_ref: dict[str, str],
    next_command_digest: str,
) -> dict[str, object]:
    anchor = service.store.dispatch_transition_anchor(session_id)
    assert anchor["eligible"] is True
    return _managed_transition_payload_from_anchor(
        service,
        session_id,
        policy,
        next_turn_ref,
        next_command_digest,
        anchor,
    )


def _managed_transition_payload_from_anchor(
    service: HarnessService,
    session_id: str,
    policy: dict[str, object],
    next_turn_ref: dict[str, str],
    next_command_digest: str,
    anchor: dict[str, object],
    transition_sequence: int = 1,
) -> dict[str, object]:
    session = service.store.get_session(session_id)
    goal = service.store.goal_for_session(session_id)
    assert goal is not None
    reason = "Advance one exact managed transition anchor."
    policy_sha256 = normalized_digest(policy)
    receipt = {
        "session_id": session_id,
        "external_ref": session.external_ref,
        "goal_id": goal.goal_id,
        "prior_command_id": anchor["prior_command_id"],
        "prior_command_type": anchor["prior_command_type"],
        "prior_anchor_kind": anchor["prior_anchor_kind"],
        "prior_reconciliation_id": anchor["prior_reconciliation_id"],
        "prior_reconciliation_resolution": anchor["prior_reconciliation_resolution"],
        "prior_checkpoint_id": anchor["prior_checkpoint_id"],
        "prior_generation_digest": anchor["prior_generation_digest"],
        "prior_material_digest": anchor["prior_material_digest"],
        "next_turn_ref": next_turn_ref,
        "transition_sequence": transition_sequence,
        "epoch_id": str(policy["epoch_id"]),
        "policy_sha256": policy_sha256,
        "next_command_digest": next_command_digest,
    }
    authorization = {
        "schema": (
            "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
        ),
        **receipt,
        "reason": reason,
        "external_orchestrator": session.external_ref["orchestrator"],
        "external_job_id": session.external_ref["job_id"],
        "receipt": receipt,
        "receipt_sha256": normalized_digest(receipt),
    }
    if transition_sequence == 1:
        authorization["policy"] = policy
    else:
        authorization["policy_ref"] = {
            "policy_sha256": policy_sha256,
            "session_id": session_id,
            "goal_id": goal.goal_id,
            "epoch_id": str(policy["epoch_id"]),
        }
    authorization.pop("external_ref")
    return {
        "reason": reason,
        "prior_command_id": anchor["prior_command_id"],
        "prior_command_type": anchor["prior_command_type"],
        "prior_anchor_kind": anchor["prior_anchor_kind"],
        "prior_reconciliation_id": anchor["prior_reconciliation_id"],
        "prior_reconciliation_resolution": anchor["prior_reconciliation_resolution"],
        "prior_checkpoint_id": anchor["prior_checkpoint_id"],
        "prior_generation_digest": anchor["prior_generation_digest"],
        "prior_material_digest": anchor["prior_material_digest"],
        "next_turn_ref": next_turn_ref,
        "transition_sequence": transition_sequence,
        "next_command_digest": next_command_digest,
        "authorization": authorization,
    }


def _transition_source_receipt(binding: dict[str, object]) -> dict[str, object]:
    return {
        "session_id": binding["session_id"],
        "external_ref": {
            "orchestrator": binding["external_orchestrator"],
            "job_id": binding["external_job_id"],
        },
        "goal_id": binding["goal_id"],
        "prior_command_id": binding["prior_command_id"],
        "prior_command_type": binding["prior_command_type"],
        "prior_anchor_kind": binding["prior_anchor_kind"],
        "prior_reconciliation_id": binding["prior_reconciliation_id"],
        "prior_reconciliation_resolution": binding["prior_reconciliation_resolution"],
        "prior_checkpoint_id": binding["prior_checkpoint_id"],
        "prior_generation_digest": binding["prior_generation_digest"],
        "prior_material_digest": binding["prior_material_digest"],
        "next_turn_ref": binding["next_turn_ref"],
        "transition_sequence": binding["transition_sequence"],
        "epoch_id": binding["epoch_id"],
        "policy_sha256": binding["policy_sha256"],
        "next_command_digest": binding["next_command_digest"],
    }


def _rewrite_transition_binding(
    payload: dict[str, object],
    name: str,
    value: object,
) -> None:
    payload[name] = value
    authorization = payload["authorization"]
    assert isinstance(authorization, dict)
    authorization[name] = value
    receipt = authorization["receipt"]
    assert isinstance(receipt, dict)
    receipt[name] = value
    authorization["receipt_sha256"] = normalized_digest(receipt)


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


def test_session_creation_recovers_crash_after_worktree_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "repo"
    repository(workspace)
    harness_paths = paths(tmp_path / "state")
    first_service = HarnessService(
        harness_paths,
        worker_manager=Workers(),
    )
    original_create_worktree = service_module.create_worktree
    crashed_worktree: Path | None = None

    class SimulatedProcessLoss(BaseException):
        pass

    def crash_after_worktree(*args: object, **kwargs: object) -> Path:
        nonlocal crashed_worktree
        crashed_worktree = original_create_worktree(*args, **kwargs)  # type: ignore[arg-type]
        raise SimulatedProcessLoss

    monkeypatch.setattr(
        service_module,
        "create_worktree",
        crash_after_worktree,
    )
    request = {"workspace": str(workspace)}
    with pytest.raises(SimulatedProcessLoss):
        first_service.create_session(
            request,
            idempotency_key="crash-retry-key",
        )
    assert crashed_worktree is not None
    assert crashed_worktree.is_dir()
    assert first_service.store.list_sessions() == []
    first_service.close()

    monkeypatch.setattr(
        service_module,
        "create_worktree",
        original_create_worktree,
    )
    recovered_service = HarnessService(
        harness_paths,
        worker_manager=Workers(),
    )
    try:
        recovered = recovered_service.create_session(
            request,
            idempotency_key="crash-retry-key",
        )
        replay = recovered_service.create_session(
            request,
            idempotency_key="crash-retry-key",
        )
        assert replay.session_id == recovered.session_id
        assert Path(recovered.worktree) == crashed_worktree
        assert len(recovered_service.store.list_sessions()) == 1
        worktrees = [
            path for path in harness_paths.worktrees.iterdir() if path.is_dir()
        ]
        assert worktrees == [crashed_worktree]
        references = subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "for-each-ref",
                "--format=%(refname)",
                "refs/agent-harness/",
            ],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.splitlines()
        assert references == [
            "refs/agent-harness/" + recovered.session_id,
        ]
        assert not list((harness_paths.runtime / "creation-intents").glob("*.json"))
    finally:
        recovered_service.close()


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
        assert health_value["runtime_build_id"] == ""
        assert health_value["control_protocol_version"] == (CONTROL_PROTOCOL_VERSION)
        assert health_value["state_root"] == str(tmp_path / "state")
        assert health_value["quiescence"] == {
            "restart_safe": True,
            "active_commands": 0,
            "active_command_details": [],
            "active_unattended_commands": [],
            "active_proofs": 0,
            "active_proof_sessions": [],
        }
        capabilities = await client.get(
            "/v1/capabilities",
            headers=headers,
        )
        capabilities_value = await capabilities.json()
        assert capabilities_value["api_version"] == "1.11.0"
        assert capabilities_value["control_protocol_version"] == (
            CONTROL_PROTOCOL_VERSION
        )
        assert capabilities_value["paths"]["socket"].endswith("/.runtime/control.sock")
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
        assert (await synchronized.json())["sync"]["state"] == ("not-configured")
        ready = await client.get("/readyz", headers=headers)
        assert (await ready.json())["status"] == "ready"
        service.record_worker_supervision_failure(
            RuntimeError("transient supervision failure")
        )
        failed_health = await client.get("/healthz", headers=headers)
        assert failed_health.status == 200
        assert (await failed_health.json())["status"] == "ok"
        not_ready = await client.get("/readyz", headers=headers)
        assert not_ready.status == 503
        assert (await not_ready.json())["status"] == "not-ready"
        recovered_supervision = service.supervise_workers()
        assert recovered_supervision["status"] == "ok"
        recovered_health = await client.get("/healthz", headers=headers)
        assert recovered_health.status == 200
        listed = await client.get("/v1/sessions", headers=headers)
        assert (await listed.json())["sessions"][0]["session_id"] == (session_id)
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
        assert (await detail.json())["safety"]["session"]["profile"] == ("unattended")
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
            },
        )
        assert extension.status == 201
        extension_value = (await extension.json())["safety"]
        assert extension_value["xhigh_authorizations"] == 0
        repeated_extension = await client.post(
            "/v1/sessions/" + session_id + "/budget-extensions",
            headers={
                **headers,
                "Idempotency-Key": "integration-extension",
            },
            json={
                "reason": "finish one bounded validation",
                "additional_seconds": 60,
            },
        )
        repeated_value = (await repeated_extension.json())["safety"]
        assert repeated_value["xhigh_authorizations"] == 0
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
        assert (await refused_lease.json())["error"]["code"] == ("E_SAFETY_GUARD")
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
                "binding_percent": 90,
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
            89.0,
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
            assert "state-volume-headroom" in (refused_value["error"]["message"])
        worker_starts_before_message = list(workers.started)
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
        assert workers.started == worker_starts_before_message + [session_id]
        command_value = (await accepted.json())["command"]
        command_status = await client.get(
            "/v1/commands/" + command_value["command_id"],
            headers=headers,
        )
        assert (await command_status.json())["command"]["status"] == ("queued")
        paused = await client.post(
            "/v1/sessions/" + session_id + "/commands/pause",
            headers={
                **headers,
                "Idempotency-Key": "integration-pause",
            },
            json={},
        )
        assert paused.status == 202
        target_command_id = new_uuid()
        targeted_interrupt = await client.post(
            "/v1/sessions/" + session_id + "/commands/interrupt",
            headers={
                **headers,
                "Idempotency-Key": "integration-targeted-interrupt",
            },
            json={"target_command_id": target_command_id},
        )
        assert targeted_interrupt.status == 202
        targeted_command = (await targeted_interrupt.json())["command"]
        assert service.store.command_payload(targeted_command["command_id"]) == {
            "target_command_id": target_command_id
        }
        steer = await client.post(
            "/v1/sessions/" + session_id + "/commands/steer",
            headers={
                **headers,
                "Idempotency-Key": "integration-steer",
            },
            json={"text": "Continue from the exact checkpoint."},
        )
        assert steer.status == 202
        worker_starts_before_invalid_controls = list(workers.started)
        invalid_controls = [
            ("unsupported", {}),
            ("pause", {"unexpected": True}),
            ("steer", {}),
            ("steer", {"text": 1}),
            ("steer", {"text": " "}),
            ("steer", {"text": "x" * 65_537}),
            ("interrupt", {"unexpected": True}),
            ("interrupt", {"target_command_id": 1}),
            ("interrupt", {"target_command_id": "not-a-uuid"}),
        ]
        for index, (command_type, command_payload) in enumerate(invalid_controls):
            invalid_control = await client.post(
                "/v1/sessions/" + session_id + "/commands/" + command_type,
                headers={
                    **headers,
                    "Idempotency-Key": "invalid-control-" + str(index),
                },
                json=command_payload,
            )
            assert invalid_control.status == 400
        assert workers.started == worker_starts_before_invalid_controls
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
                "expanded_blocks": '["tool-1"]',
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
            "expanded_blocks": '["tool-1"]',
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
        assert export_path.with_name(session_id + ".run-context.gpt.json").is_file()
        assert export_path.with_name(session_id + ".transcript.jsonl").is_file()
        assert export_path.with_name(session_id + ".transcript.md").is_file()

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
            "/v1/sessions/" + forked_session["session_id"] + "/transfers/finalize",
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


def test_service_quiescence_tracks_active_commands_and_proofs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "quiescence-repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "quiescence-state"),
        worker_manager=Workers(),
    )
    created = service.create_session({"workspace": str(workspace), "direct": True})
    assert service.quiescence()["restart_safe"] is True
    receipt = service.store.enqueue_command(
        created.session_id,
        "message",
        {"text": "active"},
        "quiescence-command",
    )
    claimed = service.store.claim_command(created.session_id)
    assert claimed is not None
    service.store.create_command_envelope(
        receipt.command_id,
        created.session_id,
        "unattended",
        {"max_seconds": 900},
    )
    active = service.quiescence()
    assert active["restart_safe"] is False
    assert active["active_commands"] == 1
    assert active["active_command_details"][0]["command_id"] == (receipt.command_id)
    assert active["active_unattended_commands"] == (active["active_command_details"])
    service.store.resolve_command(
        receipt.command_id,
        "complete",
        {"status": "complete"},
    )

    proofs_started = threading.Barrier(3)
    release = threading.Event()

    def blocked_proof(*unused: object, **unused_values: object) -> dict[str, bool]:
        del unused, unused_values
        proofs_started.wait(timeout=2)
        assert release.wait(timeout=2)
        return {"complete": True}

    monkeypatch.setattr(service_module, "proof_snapshot", blocked_proof)
    results: list[dict[str, bool]] = []

    def capture() -> None:
        results.append(service.proof(created.session_id))

    threads = [threading.Thread(target=capture) for unused in range(2)]
    for thread in threads:
        thread.start()
    proofs_started.wait(timeout=2)
    proving = service.quiescence()
    assert proving["restart_safe"] is False
    assert proving["active_proofs"] == 2
    assert proving["active_proof_sessions"] == [created.session_id]
    release.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert results == [{"complete": True}, {"complete": True}]
    assert service.quiescence()["restart_safe"] is True
    service.close()


@pytest.mark.asyncio
async def test_api_formal_goal_completes_from_bound_evidence_without_provider(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "formal-goal-repo"
    repository(workspace)
    workers = Workers()
    service = HarnessService(
        paths(tmp_path / "formal-goal-state"),
        worker_manager=workers,
    )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    try:
        created_response = await client.post(
            "/v1/sessions",
            headers=headers,
            json={
                "workspace": str(workspace),
                "direct": True,
                "goal": "Prove the managed run.",
                "goal_kind": "finite",
                "constraints": ["Use retained evidence."],
                "predicates": [
                    {
                        "type": "report",
                        "subject": "tier-8h",
                        "outcome": "passed",
                        "field": "verdict",
                        "equals": "passed",
                    }
                ],
                "budgets": {"turns": 2, "attempts": 1},
            },
        )
        assert created_response.status == 201
        created = (await created_response.json())["session"]
        session_id = created["session_id"]
        goal_response = await client.get(
            "/v1/sessions/" + session_id + "/goal",
            headers=headers,
        )
        goal = (await goal_response.json())["goal"]
        assert goal["constraints"] == ["Use retained evidence."]
        assert goal["predicates"][0]["subject"] == "tier-8h"
        assert goal["budgets"] == {"turns": 2, "attempts": 1}
        assert goal["status"] == "active"

        evidence_response = await client.post(
            "/v1/sessions/" + session_id + "/evidence",
            headers=headers,
            json={
                "type": "report",
                "subject": "tier-8h",
                "outcome": "passed",
                "value": {"verdict": "passed"},
            },
        )
        assert evidence_response.status == 201
        completed_goal = await client.get(
            "/v1/sessions/" + session_id + "/goal",
            headers=headers,
        )
        assert (await completed_goal.json())["goal"]["status"] == "complete"
        session_response = await client.get(
            "/v1/sessions/" + session_id,
            headers=headers,
        )
        session_value = (await session_response.json())["session"]
        assert session_value["lifecycle"] == "completed"
        assert session_value["attention"] == "ready"
        proof_response = await client.get(
            "/v1/sessions/" + session_id + "/proof",
            headers=headers,
        )
        proof = (await proof_response.json())["proof"]
        assert proof["goal"]["status"] == "complete"
        assert proof["goal"]["predicates"][0]["satisfied"] is True
        assert not workers.started

        promoted_predicates = [
            goal["predicates"][0],
            {
                "type": "report",
                "subject": "tier-24h",
                "outcome": "passed",
            },
        ]
        promoted_milestones = [
            {
                "milestone_id": "tier-24h",
                "title": "Prove the 24-hour tier",
                "dependencies": [],
                "predicates": [promoted_predicates[1]],
            }
        ]
        promoted_budgets = {
            "turns": 3,
            "attempts": 2,
            "seconds": 86_400,
        }
        promoted_contract = create_goal(
            session_id,
            "Prove the promoted managed run.",
            constraints=(
                "Use retained evidence.",
                "Preserve the native resume chain.",
            ),
            predicates=tuple(promoted_predicates),
            milestones=tuple(promoted_milestones),
            budgets=promoted_budgets,
        )
        authorization_receipt = {
            "actor": "test-operator",
            "scope": "tier-24h",
        }
        authorization_receipt_digest = hashlib.sha256(
            json.dumps(
                authorization_receipt,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        promotion_payload = {
            "from_goal_id": goal["goal_id"],
            "stage": "tier-24h",
            "objective": "Prove the promoted managed run.",
            "goal_kind": "finite",
            "constraints": [
                "Use retained evidence.",
                "Preserve the native resume chain.",
            ],
            "predicates": promoted_predicates,
            "milestones": promoted_milestones,
            "budgets": promoted_budgets,
            "authorization": {
                "schema": ("p13i/agent-harness/goal-promotion-authorization/v1"),
                "session_id": session_id,
                "from_goal_id": goal["goal_id"],
                "stage": "tier-24h",
                "budget_increases": {
                    "turns": 1.0,
                    "attempts": 1.0,
                    "seconds": 86_400.0,
                },
                "next_goal_contract_digest": goal_contract_digest(promoted_contract),
                "receipt": authorization_receipt,
                "receipt_sha256": authorization_receipt_digest,
            },
        }
        promotion_headers = {
            **headers,
            "Idempotency-Key": "promote-tier-24h",
        }
        promotion_response = await client.post(
            "/v1/sessions/" + session_id + "/goal/promotions",
            headers=promotion_headers,
            json=promotion_payload,
        )
        assert promotion_response.status == 201, await promotion_response.text()
        promoted = await promotion_response.json()
        promotion = promoted["promotion"]
        next_goal = promoted["goal"]
        assert promotion["previous_goal_id"] == goal["goal_id"]
        assert promotion["next_goal_id"] == next_goal["goal_id"]
        assert promoted["session"]["session_id"] == session_id
        assert promoted["session"]["lifecycle"] == "starting"
        assert next_goal["status"] == "active"
        assert next_goal["created_at"] == goal["created_at"]
        assert workers.started == [session_id]

        retried_response = await client.post(
            "/v1/sessions/" + session_id + "/goal/promotions",
            headers=promotion_headers,
            json=promotion_payload,
        )
        assert retried_response.status == 201
        assert (await retried_response.json())["promotion"] == promotion
        assert workers.started == [session_id]

        conflicting_payload = copy.deepcopy(promotion_payload)
        conflicting_payload["stage"] = "tier-72h"
        conflict = await client.post(
            "/v1/sessions/" + session_id + "/goal/promotions",
            headers=promotion_headers,
            json=conflicting_payload,
        )
        assert conflict.status == 409

        promoted_proof_response = await client.get(
            "/v1/sessions/" + session_id + "/proof",
            headers=headers,
        )
        promoted_proof = (await promoted_proof_response.json())["proof"]
        assert promoted_proof["goal"]["goal_id"] == next_goal["goal_id"]
        assert promoted_proof["goal"]["status"] == "active"
        history = {item["goal_id"]: item for item in promoted_proof["goal_history"]}
        assert history[goal["goal_id"]]["status"] == "complete"
        assert history[next_goal["goal_id"]]["status"] == "active"
        proof_promotion = promoted_proof["goal_promotions"][0]
        assert proof_promotion["promotion_id"] == promotion["promotion_id"]
        assert (
            proof_promotion["previous_goal_digest"]
            == history[goal["goal_id"]]["contract_digest"]
        )
        assert (
            proof_promotion["next_goal_digest"]
            == history[next_goal["goal_id"]]["contract_digest"]
        )
        lineage = promoted_proof["goal_promotion_evidence"][0]
        assert lineage["digest_valid"] is True
        assert lineage["copied_matches_source"] is True
        assert "value" not in lineage["source"]
        assert "value" not in lineage["copied"]
        authorization_proof = promoted_proof["authorization_receipts"][0]
        assert authorization_proof["authorization_digest_valid"] is True
        assert authorization_proof["receipt_digest_valid"] is True
        assert "authorization" not in authorization_proof
        assert (
            authorization_proof["binding"]["next_goal_contract_digest"]
            == (promotion["next_goal_digest"])
        )
        exported = service.store.export_session(session_id)
        restored = StateStore(tmp_path / "restored.sqlite3")
        restored.import_session(
            exported,
            worktree=str(workspace),
            owner_host="restored-host",
            owner_epoch=2,
        )
        restored_goal = restored.goal_for_session(session_id)
        assert restored_goal is not None
        assert goal_contract_digest(restored_goal) == promotion["next_goal_digest"]
        restored_tables = restored.portable_session(session_id)["tables"]
        assert len(restored_tables["goal_promotions"]) == 1
        assert len(restored_tables["goal_promotion_evidence"]) == 1
        assert len(restored_tables["authorization_receipts"]) == 1
        restored.close()
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
                    "Investigate the provider routing regression and repair its tests."
                )
            },
        )
        assert message.status == 202
        detail = await client.get(
            "/v1/sessions/" + second["session_id"],
            headers=headers,
        )
        assert (await detail.json())["session"]["name"] == (
            "Investigate the provider routing regression and repair its tests."
        )

        legacy = service.create_session({"workspace": str(workspace)})
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
async def test_proof_pagination_pins_concurrent_event_boundary(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "state"),
        worker_manager=Workers(),
    )
    session = service.create_session({"workspace": str(workspace)})
    for index in range(1_002):
        service.store.append_event(
            session.session_id,
            "probe.event",
            status="complete",
            metadata={"code": index},
        )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    try:
        first_response = await client.get(
            "/v1/sessions/"
            + session.session_id
            + "/proof?after_sequence=0&event_limit=1000",
            headers=headers,
        )
        first = (await first_response.json())["proof"]
        through = first["event_range"]["through_sequence"]
        assert through == 1_003
        assert first["event_range"]["complete"] is False
        assert first["truncated"] == ["events"]

        service.store.append_event(
            session.session_id,
            "concurrent.event",
            status="complete",
        )
        second_response = await client.get(
            "/v1/sessions/"
            + session.session_id
            + "/proof?after_sequence="
            + str(first["event_range"]["next_after_sequence"])
            + "&event_limit=1000&through_sequence="
            + str(through)
            + "&snapshot_id="
            + first["snapshot_id"],
            headers=headers,
        )
        second = (await second_response.json())["proof"]
        combined = first["events"] + second["events"]
        assert len(combined) == through
        assert combined[-1]["sequence"] == through
        assert second["event_range"]["complete"] is True
        assert second["event_range"]["through_sequence"] == through
        assert (
            second["event_range"]["snapshot_digest"]
            == first["event_range"]["snapshot_digest"]
        )
        assert second["snapshot_id"] == first["snapshot_id"]
        assert second["snapshot_digest"] == first["snapshot_digest"]
        assert second["state_stable"] is True
        assert second["complete"] is True
    finally:
        await client.close()
        service.close()


@pytest.mark.asyncio
async def test_api_idempotency_survives_response_loss_and_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "state"),
        worker_manager=Workers(),
    )
    session = service.create_session({"workspace": str(workspace)})
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    try:
        real_json_response = api_module.web.json_response
        response_was_lost = False

        def lose_first_successful_response(
            payload: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            nonlocal response_was_lost
            if (
                not response_was_lost
                and isinstance(payload, dict)
                and "safety" in payload
            ):
                response_was_lost = True
                raise ConnectionResetError("simulated response loss")
            return real_json_response(payload, *args, **kwargs)

        monkeypatch.setattr(
            api_module.web,
            "json_response",
            lose_first_successful_response,
        )
        lost = await client.post(
            "/v1/sessions/" + session.session_id + "/budget-extensions",
            headers={
                **headers,
                "Idempotency-Key": "lost-budget-response",
            },
            json={
                "reason": "retain the exact response receipt",
                "additional_seconds": 30,
            },
        )
        assert lost.status == 500
        monkeypatch.setattr(api_module.web, "json_response", real_json_response)

        replayed = await client.post(
            "/v1/sessions/" + session.session_id + "/budget-extensions",
            headers={
                **headers,
                "Idempotency-Key": "lost-budget-response",
            },
            json={
                "reason": "retain the exact response receipt",
                "additional_seconds": 30,
            },
        )
        assert replayed.status == 201
        assert (await replayed.json())["safety"]["xhigh_authorizations"] == 0
        budget_events = [
            event
            for event in service.store.all_events(session.session_id)
            if event.event_type == "budget.extended"
        ]
        assert len(budget_events) == 1

        service.store.record_usage(
            "codex",
            25.0,
            False,
            {"payload": {}, "error": ""},
        )

        async def create_lease() -> object:
            return await client.post(
                "/v1/leases",
                headers={
                    **headers,
                    "Idempotency-Key": "concurrent-lease",
                },
                json={
                    "provider": "codex",
                    "session_id": session.session_id,
                    "execution_profile": "unattended",
                },
            )

        lease_responses = await asyncio.gather(create_lease(), create_lease())
        lease_payloads = [await response.json() for response in lease_responses]
        assert all(response.status == 201 for response in lease_responses)
        assert lease_payloads[0] == lease_payloads[1]
        assert len(service.store.active_process_leases()) == 1

        async def create_checkpoint() -> object:
            return await client.post(
                "/v1/sessions/" + session.session_id + "/checkpoints",
                headers={
                    **headers,
                    "Idempotency-Key": "concurrent-checkpoint",
                },
                json={},
            )

        checkpoint_responses = await asyncio.gather(
            create_checkpoint(),
            create_checkpoint(),
        )
        checkpoint_payloads = [
            await response.json() for response in checkpoint_responses
        ]
        assert all(response.status == 201 for response in checkpoint_responses)
        assert checkpoint_payloads[0] == checkpoint_payloads[1]
        assert len(service.store.checkpoints(session.session_id)) == 1
    finally:
        await client.close()
        service.close()


@pytest.mark.asyncio
async def test_proof_snapshot_fails_closed_above_total_event_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "state"),
        worker_manager=Workers(),
    )
    monkeypatch.setattr(proof_module, "MAX_PROOF_TOTAL_EVENTS", 10)
    session = service.create_session({"workspace": str(workspace)})
    for index in range(10):
        service.store.append_event(
            session.session_id,
            "probe.event",
            status="complete",
            metadata={"code": index},
        )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    try:
        response = await client.get(
            "/v1/sessions/" + session.session_id + "/proof",
            headers={"Authorization": "Bearer test-token"},
        )
        payload = await response.json()
        assert response.status == 413
        assert payload["error"]["code"] == "E_PROOF_EVENT_LIMIT"
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

        def supervise_workers(self) -> None:
            calls.append("workers-supervised")

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

    async def supervision_loop(unused_service: object) -> None:
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
        lambda unused_path, unused_mode: calls.append("socket-chmod"),
    )
    monkeypatch.setattr(api_module, "_sync_loop", sync_loop)
    monkeypatch.setattr(
        api_module,
        "_supervision_loop",
        supervision_loop,
    )
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


def _ambiguous_service_reconciliation(
    tmp_path: Path,
) -> tuple[HarnessService, object, object, str]:
    workspace = tmp_path / "reconcile-repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "reconcile-state"),
        worker_manager=Workers(),
    )
    session = service.create_session(
        {
            "workspace": str(workspace),
            "direct": True,
            "permission_mode": "full",
        }
    )
    command = service.store.enqueue_command(
        session.session_id,
        "message",
        {"text": "Make one bounded change."},
        "ambiguous-message",
    )
    assert service.store.claim_command(session.session_id) is not None
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=session.session_id,
        provider="codex",
        native_session_id="codex-native-session",
        model="default",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    service.store.create_attempt(attempt)
    turn_id = service.store.start_turn(session.session_id, attempt.attempt_id)
    checkpoint = checkpoint_workspace(
        session,
        service.blobs,
        sequence=service.store.last_sequence(session.session_id),
        provider="codex",
        native_session_id="",
        context_text="",
    )
    service.store.add_checkpoint(checkpoint)
    service.store.record_dispatch_checkpoint(
        command.command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    service.store.mark_provider_boundary(attempt.attempt_id)
    digest, summary = inspect_workspace(workspace)
    recovery = service.store.recover_interrupted_commands(
        session.session_id,
        digest,
        summary,
    )
    return service, session, recovery.reconciliations[0], digest


@pytest.mark.asyncio
async def test_reconciliation_resolution_without_a_key_records_its_event(
    tmp_path: Path,
) -> None:
    service, session, record, digest = _ambiguous_service_reconciliation(tmp_path)
    try:
        with pytest.raises(ValueError, match="requires key and digest"):
            await service.resolve_reconciliation(
                record.reconciliation_id,
                {"decision": "accept-current", "observed_workspace_digest": digest},
                idempotency_key="resolve",
            )

        resolved = await service.resolve_reconciliation(
            record.reconciliation_id,
            {
                "decision": "accept-current",
                "observed_workspace_digest": digest,
            },
        )

        assert resolved["status"] == "resolved"
        assert service.workers.started == [session.session_id]
        events = [
            item
            for item in service.store.events(session.session_id)
            if item.event_type == "reconciliation.resolved"
        ]
        assert len(events) == 1
        assert events[0].metadata["decision"] == "accept-current"
    finally:
        service.close()


def _promotion_fixture(tmp_path: Path) -> tuple[HarnessService, object, object]:
    workspace = tmp_path / "promotion-repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "promotion-state"),
        worker_manager=Workers(),
    )
    first_predicate = {
        "type": "report",
        "subject": "tier-8h",
        "outcome": "passed",
    }
    session = service.create_session(
        {
            "workspace": str(workspace),
            "direct": True,
            "goal": "Prove the first tier.",
            "goal_kind": "finite",
            "constraints": ["Use retained evidence."],
            "predicates": [first_predicate],
            "milestones": [
                {
                    "milestone_id": "tier-8h",
                    "title": "Prove tier 8h",
                    "dependencies": [],
                    "predicates": [first_predicate],
                }
            ],
            "budgets": {"turns": 2},
            "permitted_providers": ["claude", "codex"],
            "permitted_efforts": ["low", "medium"],
        }
    )
    goal = service.store.goal_for_session(session.session_id)
    assert goal is not None
    return service, session, goal


def test_goal_promotion_rejects_widened_or_malformed_contracts(
    tmp_path: Path,
) -> None:
    service, session, goal = _promotion_fixture(tmp_path)
    first_predicate = dict(goal.predicates[0])
    later_predicate = {
        "type": "report",
        "subject": "tier-24h",
        "outcome": "passed",
    }
    base = {
        "from_goal_id": goal.goal_id,
        "stage": "tier-24h",
        "objective": "Prove the promoted run.",
        "goal_kind": "finite",
        "constraints": ["Use retained evidence."],
        "predicates": [first_predicate, later_predicate],
        "milestones": [
            {
                "milestone_id": "tier-8h",
                "title": "Prove tier 8h",
                "dependencies": [],
                "predicates": [first_predicate],
            },
            {
                "milestone_id": "tier-24h",
                "title": "Prove tier 24h",
                "dependencies": [],
                "predicates": [later_predicate],
            },
        ],
        "budgets": {"turns": 3},
        "permitted_providers": ["claude", "codex"],
        "permitted_efforts": ["low", "medium"],
        "max_concurrency": 1,
        "authorization": {"schema": "unverified-placeholder"},
    }

    def rejected(
        message: str,
        error: type[Exception] = ValueError,
        *,
        key: str = "promote",
        **overrides: object,
    ) -> None:
        payload = copy.deepcopy(base)
        payload.update(overrides)
        with pytest.raises(error, match=message):
            service.promote_goal(
                session.session_id,
                payload,
                idempotency_key=key,
            )

    try:
        with pytest.raises(ValueError, match="requires an idempotency key"):
            service.promote_goal(
                session.session_id,
                copy.deepcopy(base),
                idempotency_key="",
            )
        rejected("stage must contain 1 to 128 characters", stage="  ")
        rejected("requires explicit authorization", authorization={})
        rejected("goal constraints must be a list", constraints="one")
        rejected("goal predicates must be a list", predicates="one")
        rejected("goal milestones must be a list", milestones="one")
        rejected("goal budgets must be an object", budgets=[])
        rejected("permitted providers must be a list", permitted_providers="claude")
        rejected("permitted efforts must be a list", permitted_efforts="low")
        rejected("cannot remove constraints", constraints=["Other."])
        rejected(
            "requires finite goals",
            goal_kind="invariant",
            completion_policy="never",
        )
        rejected(
            "cannot widen permitted providers",
            permitted_providers=["claude", "codex", "kimi"],
        )
        rejected(
            "cannot widen permitted efforts",
            permitted_efforts=["low", "medium", "high"],
        )
        rejected("cannot increase concurrency", max_concurrency=2)
        rejected("cannot remove a budget", budgets={"attempts": 3})
        rejected("cannot reduce a budget", budgets={"turns": 1})
        rejected("requires an additive budget", budgets={"turns": 2})

        foreign = service.create_session(
            {
                "workspace": str(Path(session.workspace)),
                "direct": True,
                "goal": "Prove another run.",
                "predicates": [first_predicate],
            }
        )
        foreign_goal = service.store.goal_for_session(foreign.session_id)
        assert foreign_goal is not None
        rejected(
            "source belongs to another session",
            ConflictError,
            from_goal_id=foreign_goal.goal_id,
        )
    finally:
        service.close()


def test_goal_promotion_rejects_drifted_stored_policies(tmp_path: Path) -> None:
    service, session, goal = _promotion_fixture(tmp_path)
    first_predicate = dict(goal.predicates[0])
    later_predicate = {
        "type": "report",
        "subject": "tier-24h",
        "outcome": "passed",
    }

    def promotion(previous_goal_id: str) -> dict[str, object]:
        return {
            "from_goal_id": previous_goal_id,
            "stage": "tier-24h",
            "objective": "Prove the promoted run.",
            "goal_kind": "finite",
            "constraints": ["Use retained evidence."],
            "predicates": [first_predicate, later_predicate],
            "milestones": [],
            "budgets": {"turns": 3},
            "permitted_providers": ["claude", "codex"],
            "permitted_efforts": ["low", "medium"],
            "max_concurrency": 1,
            "completion_policy": "evidence-all",
            "incident_policy": "recover-then-pause",
            "authorization": {"schema": "unverified-placeholder"},
        }

    try:
        for field, value, message in (
            ("completion_policy", "drifted", "cannot change completion policy"),
            ("incident_policy", "drifted", "cannot change incident policy"),
        ):
            drifted = service.store.create_goal(
                replace(
                    goal,
                    goal_id=new_uuid(),
                    **{field: value},
                )
            )
            with pytest.raises(ValueError, match=message):
                service.promote_goal(
                    session.session_id,
                    promotion(drifted.goal_id),
                    idempotency_key="promote-" + field,
                )
    finally:
        service.close()


def _transition_fixture(
    tmp_path: Path,
) -> tuple[HarnessService, object, dict[str, object], dict[str, str], str]:
    workspace = tmp_path / "invalidation-repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "invalidation-state"),
        worker_manager=Workers(),
    )
    external_ref = {"orchestrator": "machines", "job_id": "builder-stage"}
    session = service.create_session(
        {
            "workspace": str(workspace),
            "direct": True,
            "execution_profile": "unattended",
            "external_ref": external_ref,
        }
    )
    next_turn_ref = {"step_id": "verify", "agent_role": "verifier"}
    next_command_digest = command_envelope_digest(
        "message",
        {
            "text": "Verify the implementation.",
            "provider": "codex",
            "turn_ref": next_turn_ref,
        },
        "unattended",
    )
    policy = _single_transition_policy(
        session.session_id,
        external_ref,
        "invalidation-epoch-1",
        next_turn_ref,
        next_command_digest,
    )
    service.store.create_goal(
        create_goal(
            session.session_id,
            "Run one exact reviewed stage.",
            constraints=(
                "dispatch-generation-transition-policy-sha256:"
                + normalized_digest(policy),
                "dispatch-generation-transition-epoch:" + str(policy["epoch_id"]),
            ),
        )
    )
    prior = service.store.enqueue_command(
        session.session_id,
        "message",
        {
            "text": "Review without changes.",
            "turn_ref": {"step_id": "review", "agent_role": "reviewer"},
        },
        "prior-review",
    )
    checkpoint = checkpoint_workspace(
        session,
        service.blobs,
        sequence=service.store.last_sequence(session.session_id),
        provider="claude",
        native_session_id="claude-review",
        context_text="reviewed",
    )
    service.store.add_checkpoint(checkpoint)
    service.store.resolve_command(
        prior.command_id,
        "complete",
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "workspace_material_digest": inspect_workspace(workspace)[0],
        },
    )
    return service, session, policy, next_turn_ref, next_command_digest


def test_dispatch_invalidation_binds_every_transition_field(
    tmp_path: Path,
) -> None:
    service, session, policy, next_turn_ref, next_command_digest = (
        _transition_fixture(tmp_path)
    )
    base = _managed_transition_payload(
        service,
        session.session_id,
        policy,
        next_turn_ref,
        next_command_digest,
    )
    reason = str(base["reason"])

    def rejected(
        message: str,
        mutate: Callable[[dict[str, object]], None],
        error: type[Exception] = ValueError,
    ) -> None:
        payload = copy.deepcopy(base)
        mutate(payload)
        with pytest.raises(error, match=message):
            service.invalidate_dispatch_generation(
                session.session_id,
                payload,
                idempotency_key="invalidate-" + str(len(message)),
            )

    def authorization(payload: dict[str, object]) -> dict[str, object]:
        value = payload["authorization"]
        assert isinstance(value, dict)
        return value

    def reseal(payload: dict[str, object], **receipt_overrides: object) -> None:
        current = authorization(payload)
        receipt = current["receipt"]
        assert isinstance(receipt, dict)
        receipt.update(receipt_overrides)
        current["receipt_sha256"] = normalized_digest(receipt)

    try:
        with pytest.raises(ValueError, match="requires an idempotency key"):
            service.invalidate_dispatch_generation(
                session.session_id,
                copy.deepcopy(base),
                idempotency_key="",
            )
        rejected(
            "reason must contain 1 to 500 characters",
            lambda item: item.__setitem__("reason", "  "),
        )
        rejected(
            "requires typed authorization",
            lambda item: item.__setitem__("authorization", "signed"),
        )
        rejected(
            "authorization is unsupported",
            lambda item: authorization(item).__setitem__("schema", "other"),
        )
        rejected(
            "reason does not match",
            lambda item: authorization(item).__setitem__("reason", "other"),
        )
        rejected(
            "receipt is invalid",
            lambda item: authorization(item).__setitem__(
                "receipt_sha256",
                "short",
            ),
        )
        rejected(
            "requires a next step identifier",
            lambda item: item.__setitem__("next_turn_ref", None),
        )
        rejected(
            "prior command does not match",
            lambda item: authorization(item).__setitem__(
                "prior_command_id",
                new_uuid(),
            ),
        )
        rejected(
            "prior command type is invalid",
            lambda item: item.__setitem__("prior_command_type", ""),
        )
        rejected(
            "anchor kind is invalid",
            lambda item: item.__setitem__("prior_anchor_kind", "guess"),
        )
        rejected(
            "reconciliation resolution is invalid",
            lambda item: item.update(
                prior_anchor_kind="resolved-reconciliation",
                prior_reconciliation_id=new_uuid(),
                prior_reconciliation_resolution="ignore",
            ),
        )
        rejected(
            "reconciliation binding is unexpected",
            lambda item: item.__setitem__("prior_reconciliation_id", new_uuid()),
        )
        rejected(
            "orchestrator does not match",
            lambda item: authorization(item).__setitem__(
                "external_orchestrator",
                "other",
            ),
        )
        rejected(
            "sequence is invalid",
            lambda item: item.__setitem__("transition_sequence", 0),
        )
        rejected(
            "generation digest is invalid",
            lambda item: item.__setitem__("prior_generation_digest", "short"),
        )
        rejected(
            "material digest is invalid",
            lambda item: item.__setitem__("prior_material_digest", "short"),
        )
        rejected(
            "goal does not match",
            lambda item: authorization(item).__setitem__("goal_id", new_uuid()),
        )
        rejected(
            "epoch is invalid",
            lambda item: authorization(item).__setitem__("epoch_id", ""),
        )
        rejected(
            "requires the full policy",
            lambda item: authorization(item).pop("policy"),
        )
        rejected(
            "must not carry a policy reference",
            lambda item: authorization(item).__setitem__("policy_ref", {}),
        )
        rejected(
            "command digest is invalid",
            lambda item: item.__setitem__("next_command_digest", "short"),
        )
        rejected(
            "prior_checkpoint_id does not match",
            lambda item: authorization(item).__setitem__(
                "prior_checkpoint_id",
                new_uuid(),
            ),
        )
        rejected(
            "command digest does not match",
            lambda item: authorization(item).__setitem__(
                "next_command_digest",
                "b" * 64,
            ),
        )
        rejected(
            "source receipt does not match",
            lambda item: reseal(item, session_id=new_uuid()),
        )

        operator_receipt = {"scope": "operator"}
        operator_payload = {
            "reason": reason,
            "authorization": {
                "schema": ("p13i/agent-harness/dispatch-invalidation-authorization/v1"),
                "session_id": session.session_id,
                "reason": reason,
                "receipt": operator_receipt,
                "receipt_sha256": normalized_digest(operator_receipt),
            },
        }
        with pytest.raises(ValueError, match="requires a transition"):
            service.invalidate_dispatch_generation(
                session.session_id,
                operator_payload,
                idempotency_key="invalidate-operator",
            )
    finally:
        service.close()


def test_dispatch_invalidation_requires_a_retained_policy_reference(
    tmp_path: Path,
) -> None:
    service, session, policy, next_turn_ref, next_command_digest = (
        _transition_fixture(tmp_path)
    )
    anchor = service.store.dispatch_transition_anchor(session.session_id)
    follow_up = _managed_transition_payload_from_anchor(
        service,
        session.session_id,
        policy,
        next_turn_ref,
        next_command_digest,
        anchor,
        transition_sequence=2,
    )

    try:
        missing_reference = copy.deepcopy(follow_up)
        missing_authorization = missing_reference["authorization"]
        assert isinstance(missing_authorization, dict)
        missing_authorization.pop("policy_ref")
        with pytest.raises(ValueError, match="policy reference is required"):
            service.invalidate_dispatch_generation(
                session.session_id,
                missing_reference,
                idempotency_key="invalidate-missing-reference",
            )

        mismatched = copy.deepcopy(follow_up)
        mismatched_authorization = mismatched["authorization"]
        assert isinstance(mismatched_authorization, dict)
        mismatched_authorization["policy_ref"] = {"policy_sha256": "a" * 64}
        with pytest.raises(ValueError, match="policy reference does not match"):
            service.invalidate_dispatch_generation(
                session.session_id,
                mismatched,
                idempotency_key="invalidate-mismatched-reference",
            )
    finally:
        service.close()


def test_durable_dispatch_invalidations_bind_every_transition_field(
    tmp_path: Path,
) -> None:
    service, current, policy, next_turn_ref, next_command_digest = (
        _transition_fixture(tmp_path)
    )
    base = _managed_transition_payload(
        service,
        current.session_id,
        policy,
        next_turn_ref,
        next_command_digest,
    )

    def invalidate(payload: dict[str, object], key: str) -> dict[str, object]:
        authorization = payload["authorization"]
        assert isinstance(authorization, dict)
        return service.store.create_dispatch_invalidation(
            current.session_id,
            reason=str(payload["reason"]),
            authorization=authorization,
            request_digest=normalized_digest(payload),
            idempotency_key=key,
            prior_command_id=str(payload["prior_command_id"]),
            next_turn_ref=payload["next_turn_ref"],
            authorization_digest=normalized_digest(authorization),
        )

    counter = {"value": 0}

    def rejected(
        message: str,
        mutate: Callable[[dict[str, object]], None],
    ) -> None:
        counter["value"] += 1
        payload = copy.deepcopy(base)
        mutate(payload)
        with pytest.raises(ConflictError, match=message):
            invalidate(payload, "durable-" + str(counter["value"]))

    def authorization(payload: dict[str, object]) -> dict[str, object]:
        value = payload["authorization"]
        assert isinstance(value, dict)
        return value

    def reseal_policy(payload: dict[str, object], **changes: object) -> None:
        current_authorization = authorization(payload)
        current_policy = current_authorization["policy"]
        assert isinstance(current_policy, dict)
        current_policy.update(changes)
        current_authorization["policy_sha256"] = normalized_digest(current_policy)

    try:
        with pytest.raises(NotFoundError):
            service.store.create_dispatch_invalidation(
                new_uuid(),
                reason="absent",
                authorization={"schema": "other"},
                request_digest="a" * 64,
                idempotency_key="absent-session",
                prior_command_id="",
                next_turn_ref={},
                authorization_digest="a" * 64,
            )

        rejected(
            "prior command is required",
            lambda item: item.__setitem__("prior_command_id", ""),
        )
        rejected(
            "requires a transition",
            lambda item: authorization(item).__setitem__("schema", "operator"),
        )
        rejected(
            "prior command changed",
            lambda item: authorization(item).__setitem__(
                "prior_command_id",
                new_uuid(),
            ),
        )
        rejected(
            "next stage changed",
            lambda item: item.__setitem__(
                "next_turn_ref",
                {"step_id": "publish", "agent_role": "publisher"},
            ),
        )
        rejected(
            "prior command is unknown",
            lambda item: (
                item.__setitem__("prior_command_id", new_uuid()),
                authorization(item).__setitem__(
                    "prior_command_id",
                    item["prior_command_id"],
                ),
            ),
        )
        rejected(
            "prior_command_type changed",
            lambda item: authorization(item).__setitem__(
                "prior_command_type",
                "control",
            ),
        )
        rejected(
            "material changed",
            lambda item: authorization(item).__setitem__(
                "prior_material_digest",
                "a" * 64,
            ),
        )
        rejected(
            "generation changed",
            lambda item: authorization(item).__setitem__(
                "prior_generation_digest",
                "a" * 64,
            ),
        )
        rejected(
            "sequence is invalid",
            lambda item: authorization(item).__setitem__(
                "transition_sequence",
                True,
            ),
        )
        rejected(
            "goal changed",
            lambda item: authorization(item).__setitem__("goal_id", new_uuid()),
        )
        rejected(
            "sequence is out of order",
            lambda item: authorization(item).__setitem__("transition_sequence", 3),
        )
        durable_connection = service.store._connection

        class MissingPredecessorConnection:
            def execute(
                self,
                statement: str,
                parameters: tuple[object, ...] = (),
            ) -> object:
                if "SELECT COALESCE(MAX(transition_sequence)" in statement:
                    return SimpleNamespace(fetchone=lambda: {"count": 1})
                return durable_connection.execute(statement, parameters)

        service.store._connection = MissingPredecessorConnection()
        try:
            rejected(
                "predecessor is missing",
                lambda item: authorization(item).__setitem__(
                    "transition_sequence",
                    2,
                ),
            )
        finally:
            service.store._connection = durable_connection
        rejected(
            "orchestrator changed",
            lambda item: authorization(item).__setitem__(
                "external_orchestrator",
                "other",
            ),
        )
        rejected(
            "policy is missing",
            lambda item: authorization(item).pop("policy"),
        )
        rejected(
            "policy schema changed",
            lambda item: reseal_policy(item, schema="other"),
        )
        rejected(
            "policy digest changed",
            lambda item: authorization(item).__setitem__(
                "policy_sha256",
                "a" * 64,
            ),
        )
        rejected(
            "policy session changed",
            lambda item: reseal_policy(item, session_id=new_uuid()),
        )
        rejected(
            "policy orchestrator changed",
            lambda item: reseal_policy(item, external_ref={}),
        )
        rejected(
            "epoch changed",
            lambda item: reseal_policy(item, epoch_id="other-epoch"),
        )
        rejected(
            "has a policy reference",
            lambda item: authorization(item).__setitem__("policy_ref", {}),
        )
        rejected(
            "role is outside policy",
            lambda item: reseal_policy(item, allowed_agent_roles=["publisher"]),
        )
        rejected(
            "step is outside policy",
            lambda item: reseal_policy(item, allowed_step_prefixes=["publish"]),
        )
        rejected(
            "exceeds policy limit",
            lambda item: reseal_policy(item, max_transitions=0),
        )
        rejected(
            "policy stages changed",
            lambda item: reseal_policy(item, transitions="none"),
        )
        rejected(
            "policy limit changed",
            lambda item: reseal_policy(item, max_transitions=2),
        )
        rejected(
            "policy stage changed",
            lambda item: reseal_policy(item, transitions=["bare"]),
        )
        rejected(
            "policy order changed",
            lambda item: reseal_policy(
                item,
                transitions=[
                    {
                        "sequence": 2,
                        "next_turn_ref": next_turn_ref,
                        "next_command_digest": next_command_digest,
                    }
                ],
            ),
        )
        rejected(
            "policy stage changed",
            lambda item: reseal_policy(
                item,
                allowed_agent_roles=["verifier", "publisher"],
                allowed_step_prefixes=["verify", "publish"],
                transitions=[
                    {
                        "sequence": 1,
                        "next_turn_ref": {
                            "step_id": "publish",
                            "agent_role": "publisher",
                        },
                        "next_command_digest": next_command_digest,
                    }
                ],
            ),
        )
        rejected(
            "command policy changed",
            lambda item: reseal_policy(
                item,
                transitions=[
                    {
                        "sequence": 1,
                        "next_turn_ref": next_turn_ref,
                        "next_command_digest": "b" * 64,
                    }
                ],
            ),
        )
        rejected(
            "policy is not authorized",
            lambda item: reseal_policy(item, note="unauthorized"),
        )
        rejected(
            "source receipt changed",
            lambda item: authorization(item)["receipt"].__setitem__(
                "session_id",
                new_uuid(),
            ),
        )
        rejected(
            "receipt digest changed",
            lambda item: authorization(item).__setitem__(
                "receipt_sha256",
                "a" * 64,
            ),
        )

        receipt = invalidate(copy.deepcopy(base), "durable-accepted")
        assert receipt["prior_command_id"] == base["prior_command_id"]
        replayed = invalidate(copy.deepcopy(base), "durable-accepted")
        assert replayed["invalidation_id"] == receipt["invalidation_id"]
        durable_replay = service.store.dispatch_invalidation_replay(
            current.session_id,
            "durable-accepted",
            normalized_digest(base),
        )
        assert durable_replay is not None
        assert durable_replay["invalidation_id"] == receipt["invalidation_id"]
        with pytest.raises(ConflictError, match="idempotency key was reused"):
            invalidate(
                {**copy.deepcopy(base), "reason": "another reason"},
                "durable-accepted",
            )
        with pytest.raises(ConflictError, match="idempotency key was reused"):
            service.store.dispatch_invalidation_replay(
                current.session_id,
                "durable-accepted",
                "b" * 64,
            )
    finally:
        service.close()


def test_durable_dispatch_invalidations_require_a_quiescent_anchor(
    tmp_path: Path,
) -> None:
    service, current, policy, next_turn_ref, next_command_digest = (
        _transition_fixture(tmp_path)
    )
    base = _managed_transition_payload(
        service,
        current.session_id,
        policy,
        next_turn_ref,
        next_command_digest,
    )

    def invalidate(key: str) -> dict[str, object]:
        authorization = base["authorization"]
        assert isinstance(authorization, dict)
        return service.store.create_dispatch_invalidation(
            current.session_id,
            reason=str(base["reason"]),
            authorization=authorization,
            request_digest=normalized_digest(base),
            idempotency_key=key,
            prior_command_id=str(base["prior_command_id"]),
            next_turn_ref=base["next_turn_ref"],
            authorization_digest=normalized_digest(authorization),
        )

    try:
        service.store.update_session(current.session_id, attention="working")
        with pytest.raises(ConflictError, match="requires quiescence"):
            invalidate("quiescence-working")
        service.store.update_session(current.session_id, attention="idle")

        queued = service.store.enqueue_command(
            current.session_id,
            "message",
            {"text": "Queued after the anchor."},
            "queued-after-anchor",
        )
        with pytest.raises(ConflictError, match="has an active command"):
            invalidate("quiescence-active")

        service.store.resolve_command(queued.command_id, "cancelled", {})
        with pytest.raises(ConflictError, match="is not latest"):
            invalidate("quiescence-latest")
    finally:
        service.close()


def test_contract_adoption_rejects_untyped_or_retargeted_envelopes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "adoption-repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "adoption-state"),
        worker_manager=Workers(),
    )
    external_ref = {"orchestrator": "p13i/machines/proof", "job_id": "job-7"}
    try:
        session = service.create_session(
            {
                "workspace": str(workspace),
                "direct": True,
                "execution_profile": "unattended",
            }
        )
        base = machines_session_payload(workspace, external_ref)
        base.update(
            {
                "permission_mode": "approval",
                "authorization": {"schema": "unverified-placeholder"},
            }
        )

        def rejected(message: str, **overrides: object) -> None:
            payload = copy.deepcopy(base)
            payload.update(overrides)
            with pytest.raises(ValueError, match=message):
                service.adopt_session_contract(
                    session.session_id,
                    payload,
                    idempotency_key="adopt",
                )

        with pytest.raises(ValueError, match="requires an idempotency key"):
            service.adopt_session_contract(
                session.session_id,
                copy.deepcopy(base),
                idempotency_key="",
            )
        rejected("workspace does not match", workspace=str(tmp_path))
        rejected("direct must be boolean", direct="yes")
        rejected("cannot change worktree mode", direct=False)
        rejected(
            "reserved for p13i/machines",
            external_ref={"orchestrator": "operator", "job_id": "job-7"},
        )
        rejected("constraints must be an array", constraints="one")
        rejected("predicates must be an array", predicates="one")
        rejected("milestones must be an array", milestones="one")
        rejected("budgets must be an object", budgets=[])
        rejected("permitted_providers must be an array", permitted_providers="claude")
        rejected("permitted_efforts must be an array", permitted_efforts="low")
        rejected("unsupported permission mode", permission_mode="root")
        rejected("requires typed authorization", authorization=[])
    finally:
        service.close()


def test_session_creation_requires_typed_goal_collections(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "typed-repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "typed-state"),
        worker_manager=Workers(),
    )
    try:
        for field, message in (
            ("milestones", "milestones must be an array"),
            ("permitted_providers", "permitted_providers must be an array"),
            ("permitted_efforts", "permitted_efforts must be an array"),
        ):
            with pytest.raises(ValueError, match=message):
                service.create_session(
                    {
                        "workspace": str(workspace),
                        "direct": True,
                        "goal": "Prove the run.",
                        field: "not-an-array",
                    }
                )
    finally:
        service.close()


@pytest.mark.asyncio
async def test_daemon_background_loops_absorb_faults_between_intervals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[BaseException] = []

    class FaultyService:
        def __init__(self) -> None:
            self.paths = paths(tmp_path / "state")
            self.store = object()

        def supervise_workers(self) -> None:
            raise RuntimeError("supervision failed")

        def record_worker_supervision_failure(
            self,
            error: BaseException,
        ) -> None:
            recorded.append(error)

    def failing_publish(*unused_args: object) -> None:
        raise OSError("publish failed")

    intervals: list[float] = []

    async def interrupted_sleep(seconds: float) -> None:
        intervals.append(seconds)
        raise asyncio.CancelledError

    service = FaultyService()
    monkeypatch.setattr(api_module, "publish_all", failing_publish)
    monkeypatch.setattr(api_module.asyncio, "sleep", interrupted_sleep)

    with pytest.raises(asyncio.CancelledError):
        await api_module._sync_loop(service)
    with pytest.raises(asyncio.CancelledError):
        await api_module._supervision_loop(service)

    assert intervals == [30, 1]
    assert [str(item) for item in recorded] == ["supervision failed"]


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
        await api_module._sync_loop(SimpleNamespace(paths=object(), store=object()))


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
        "permission_mode": "full",
        "execution_profile": "unattended",
        "direct": False,
        "external_ref": {
            "orchestrator": "p13i/machines",
            "job_id": "job-42",
        },
        "goal": "Complete the bounded external job.",
        "goal_kind": "finite",
        "constraints": ["Preserve the external job receipt."],
        "predicates": [
            {"type": "machines-proof", "subject": "job-42", "outcome": "passed"}
        ],
        "milestones": [
            {
                "milestone_id": "job-42-complete",
                "title": "Complete job 42",
                "dependencies": [],
                "predicates": [
                    {
                        "type": "machines-proof",
                        "subject": "job-42",
                        "outcome": "passed",
                    }
                ],
            }
        ],
        "budgets": {
            "seconds": 300,
            "turns": 2,
            "tokens": 20_000,
            "context_tokens": 16_000,
            "output_tokens": 4_000,
            "tool_calls": 10,
            "attempts": 2,
            "child_agents": 1,
            "dollars": 0,
        },
        "permitted_providers": ["claude", "codex"],
        "permitted_efforts": ["low", "medium"],
        "max_concurrency": 1,
        "completion_policy": "evidence-all",
        "incident_policy": "recover-then-pause",
    }
    try:
        missing_constraints = await client.post(
            "/v1/sessions",
            headers={
                **headers,
                "Idempotency-Key": "invalid-job-42-constraints",
            },
            json={**create_payload, "constraints": []},
        )
        assert missing_constraints.status == 400
        unsupported_provider = await client.post(
            "/v1/sessions",
            headers={
                **headers,
                "Idempotency-Key": "invalid-job-42-provider",
            },
            json={**create_payload, "permitted_providers": ["unknown"]},
        )
        assert unsupported_provider.status == 400
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
        assert [item["session_id"] for item in values] == [session["session_id"]]
        missing_lookup_field = await client.get(
            "/v1/sessions?external_job_id=job-42",
            headers=headers,
        )
        assert missing_lookup_field.status == 400

        turn_payload = {
            "text": "Implement the bounded step.",
            "permission_mode": "read-only",
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
        service.store.update_session(
            session["session_id"],
            permission_mode="read-only",
        )
        widening_turn = await client.post(
            "/v1/sessions/" + session["session_id"] + "/messages",
            headers={
                **headers,
                "Idempotency-Key": "turn-step-8",
            },
            json={
                "text": "Widen this turn.",
                "permission_mode": "approval",
            },
        )
        assert widening_turn.status == 400
        service.store.update_session(
            session["session_id"],
            permission_mode="full",
        )

        evidence_payload = {
            "type": "machines-proof",
            "subject": "job-42",
            "outcome": "passed",
            "value": {"report_digest": "sha256:proof-42"},
        }
        missing_evidence_key = await client.post(
            "/v1/sessions/" + session["session_id"] + "/evidence",
            headers=headers,
            json=evidence_payload,
        )
        assert missing_evidence_key.status == 400
        evidence_headers = {
            **headers,
            "Idempotency-Key": "evidence-job-42",
        }
        evidence_responses = await asyncio.gather(
            client.post(
                "/v1/sessions/" + session["session_id"] + "/evidence",
                headers=evidence_headers,
                json=evidence_payload,
            ),
            client.post(
                "/v1/sessions/" + session["session_id"] + "/evidence",
                headers=evidence_headers,
                json=evidence_payload,
            ),
        )
        evidence_values = [
            (await response.json())["evidence"] for response in evidence_responses
        ]
        assert evidence_values[0]["evidence_id"] == evidence_values[1]["evidence_id"]
        current_goal = service.store.goal_for_session(session["session_id"])
        assert current_goal is not None
        assert len(service.store.evidence(current_goal.goal_id)) == 1
        evidence_events = [
            event
            for event in service.store.events(session["session_id"])
            if event.event_type == "goal.evidence"
        ]
        assert len(evidence_events) == 1
        evidence_conflict = await client.post(
            "/v1/sessions/" + session["session_id"] + "/evidence",
            headers=evidence_headers,
            json={
                **evidence_payload,
                "outcome": "failed",
            },
        )
        assert evidence_conflict.status == 409

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
async def test_managed_contract_adoption_preserves_creation_replay_and_exact_uuid(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "legacy-repo"
    repository(workspace)
    workers = Workers()
    service = HarnessService(
        paths(tmp_path / "legacy-state"),
        worker_manager=workers,
    )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    try:
        initial_external_ref = {
            "orchestrator": "p13i/machines/cs-sre",
            "job_id": "cs-sre",
        }
        initial_payload = machines_session_payload(
            workspace,
            initial_external_ref,
            name="Legacy SRE",
            permission_mode="full",
            direct=False,
            goal="Keep the legacy service healthy.",
            goal_kind="invariant",
            completion_policy="never",
        )
        creation_headers = {
            **headers,
            "Idempotency-Key": "stable-cs-sre-session",
        }
        created_response = await client.post(
            "/v1/sessions",
            headers=creation_headers,
            json=initial_payload,
        )
        created = (await created_response.json())["session"]
        session_id = created["session_id"]
        original_creation_digest = created["creation_digest"]
        previous = service.store.goal_for_session(session_id)
        assert previous is not None
        service.store.extend_session_safety(
            session_id,
            {"max_seconds": 60, "allow_xhigh_once": True},
        )
        predicates = [
            {
                "type": "sre-health",
                "subject": "cs-sre",
                "outcome": "passed",
            }
        ]
        milestones = [
            {
                "milestone_id": "sre-installed",
                "title": "Install cs-sre",
                "dependencies": [],
                "predicates": predicates,
            }
        ]
        budgets = {
            "seconds": 86_400,
            "turns": 12,
            "tokens": 100_000,
            "context_tokens": 64_000,
            "output_tokens": 16_000,
            "tool_calls": 64,
            "attempts": 6,
            "child_agents": 2,
            "dollars": 0,
        }
        transition_turn_ref = {"step_id": "sre-tick", "agent_role": "sre"}
        transition_command_payload = {
            "text": "Run the bounded SRE tick.",
            "provider": "codex",
            "turn_ref": transition_turn_ref,
        }
        transition_command_digest = command_envelope_digest(
            "message",
            transition_command_payload,
            "unattended",
        )

        def epoch_policy(epoch_id: str) -> dict[str, object]:
            return {
                "schema": (
                    "p13i/agent-harness/dispatch-generation-transition-policy/v1"
                ),
                "session_id": session_id,
                "external_ref": {
                    "orchestrator": "p13i/machines/cs-sre",
                    "job_id": "cs-sre",
                },
                "epoch_id": epoch_id,
                "allowed_agent_roles": ["sre"],
                "allowed_step_prefixes": ["sre-"],
                "max_transitions": 1,
                "transitions": [
                    {
                        "sequence": 1,
                        "next_turn_ref": transition_turn_ref,
                        "next_command_digest": transition_command_digest,
                    }
                ],
            }

        first_policy = epoch_policy("contract-epoch-1")
        goal_envelope = {
            "objective": "Keep cs-sre healthy.",
            "kind": "invariant",
            "constraints": [
                "Pause after an incident.",
                "dispatch-generation-transition-policy-sha256:"
                + normalized_digest(first_policy),
                "dispatch-generation-transition-epoch:contract-epoch-1",
            ],
            "predicates": predicates,
            "milestones": milestones,
            "budgets": budgets,
            "permitted_providers": ["claude", "codex"],
            "permitted_efforts": ["low", "medium"],
            "max_concurrency": 1,
            "completion_policy": "never",
            "incident_policy": "recover-then-pause",
        }
        source_receipt = {"actor": "test-operator", "change": "formalize-sre"}
        source_receipt_sha256 = normalized_digest(source_receipt)
        adoption_payload = {
            "workspace": str(workspace.resolve()),
            "name": "Managed cs-sre",
            "permission_mode": "full",
            "execution_profile": "unattended",
            "direct": False,
            "model": "",
            "effort": "",
            "external_ref": {
                "orchestrator": "p13i/machines/cs-sre",
                "job_id": "cs-sre",
            },
            "goal": goal_envelope["objective"],
            "goal_kind": goal_envelope["kind"],
            "constraints": goal_envelope["constraints"],
            "predicates": predicates,
            "milestones": milestones,
            "budgets": budgets,
            "permitted_providers": goal_envelope["permitted_providers"],
            "permitted_efforts": goal_envelope["permitted_efforts"],
            "max_concurrency": 1,
            "completion_policy": "never",
            "incident_policy": "recover-then-pause",
            "authorization": {
                "schema": (
                    "p13i/agent-harness/session-contract-adoption-authorization/v1"
                ),
                "session_id": session_id,
                "external_ref": {
                    "orchestrator": "p13i/machines/cs-sre",
                    "job_id": "cs-sre",
                },
                "previous_goal_digest": goal_contract_digest(previous),
                "goal_envelope_digest": normalized_digest(goal_envelope),
                "receipt": source_receipt,
                "receipt_sha256": source_receipt_sha256,
            },
        }
        adoption_headers = {
            **headers,
            "Idempotency-Key": "adopt-cs-sre-contract",
        }
        adopted_response = await client.post(
            "/v1/sessions/" + session_id + "/contract-adoptions",
            headers=adoption_headers,
            json=adoption_payload,
        )
        assert adopted_response.status == 201
        adopted = await adopted_response.json()
        assert adopted["session"]["session_id"] == session_id
        assert adopted["session"]["external_ref"] == adoption_payload["external_ref"]
        assert adopted["session"]["creation_digest"] == original_creation_digest
        assert adopted["adoption"]["creation_digest"] != original_creation_digest
        assert adopted["goal"]["completion_policy"] == "never"
        assert service.store.get_goal(previous.goal_id).status == "cancelled"
        safety = service.store.session_safety(session_id)
        assert safety["extensions"] == {}
        assert safety["xhigh_authorizations"] == 0
        creation_replay = await client.post(
            "/v1/sessions",
            headers=creation_headers,
            json=initial_payload,
        )
        assert creation_replay.status == 201
        assert (await creation_replay.json())["session"]["session_id"] == session_id
        current_session = await client.get(
            "/v1/sessions/" + session_id,
            headers=headers,
        )
        assert (await current_session.json())["session"]["external_ref"] == (
            initial_external_ref
        )
        changed_creation = await client.post(
            "/v1/sessions",
            headers=creation_headers,
            json={**initial_payload, "name": "Changed SRE"},
        )
        assert changed_creation.status == 409
        replay = await client.post(
            "/v1/sessions/" + session_id + "/contract-adoptions",
            headers=adoption_headers,
            json=adoption_payload,
        )
        assert (await replay.json())["adoption"] == adopted["adoption"]
        reused_key = await client.post(
            "/v1/sessions/" + session_id + "/contract-adoptions",
            headers=adoption_headers,
            json={**adoption_payload, "name": "Retargeted SRE"},
        )
        assert reused_key.status == 409
        assert "idempotency key was reused" in await reused_key.text()
        first_epoch_goal = service.store.goal_for_session(session_id)
        assert first_epoch_goal is not None

        def completed_epoch_command(label: str):
            current_session = service.store.get_session(session_id)
            command = service.store.enqueue_command(
                session_id,
                "message",
                {
                    "text": "Complete prior " + label + ".",
                    "turn_ref": {"step_id": label, "agent_role": "sre"},
                },
                "prior-" + label,
            )
            checkpoint = checkpoint_workspace(
                current_session,
                service.blobs,
                sequence=service.store.last_sequence(session_id),
                provider="codex",
                native_session_id="codex-" + label,
                context_text=label,
            )
            service.store.add_checkpoint(checkpoint)
            service.store.resolve_command(
                command.command_id,
                "complete",
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "workspace_material_digest": inspect_workspace(workspace)[0],
                },
            )
            return command, checkpoint

        def epoch_transition_payload(
            goal_id: str,
            policy: dict[str, object],
            prior_command,
            prior_checkpoint,
        ) -> dict[str, object]:
            anchor = service.store.dispatch_transition_anchor(session_id)
            assert anchor["eligible"] is True
            assert anchor["prior_command_id"] == prior_command.command_id
            assert anchor["prior_checkpoint_id"] == prior_checkpoint.checkpoint_id
            prior_generation = str(anchor["prior_generation_digest"])
            prior_material = str(anchor["prior_material_digest"])
            policy_sha256 = normalized_digest(policy)
            receipt = {
                "session_id": session_id,
                "external_ref": adoption_payload["external_ref"],
                "goal_id": goal_id,
                "prior_command_id": prior_command.command_id,
                "prior_command_type": anchor["prior_command_type"],
                "prior_anchor_kind": anchor["prior_anchor_kind"],
                "prior_reconciliation_id": anchor["prior_reconciliation_id"],
                "prior_reconciliation_resolution": anchor[
                    "prior_reconciliation_resolution"
                ],
                "prior_checkpoint_id": prior_checkpoint.checkpoint_id,
                "prior_generation_digest": prior_generation,
                "prior_material_digest": prior_material,
                "next_turn_ref": transition_turn_ref,
                "transition_sequence": 1,
                "epoch_id": str(policy["epoch_id"]),
                "policy_sha256": policy_sha256,
                "next_command_digest": transition_command_digest,
            }
            authorization = {
                "schema": (
                    "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
                ),
                "session_id": session_id,
                "goal_id": goal_id,
                "reason": "Advance one bounded invariant epoch tick.",
                "prior_command_id": prior_command.command_id,
                "prior_command_type": anchor["prior_command_type"],
                "prior_anchor_kind": anchor["prior_anchor_kind"],
                "prior_reconciliation_id": anchor["prior_reconciliation_id"],
                "prior_reconciliation_resolution": anchor[
                    "prior_reconciliation_resolution"
                ],
                "prior_checkpoint_id": prior_checkpoint.checkpoint_id,
                "prior_generation_digest": prior_generation,
                "prior_material_digest": prior_material,
                "next_turn_ref": transition_turn_ref,
                "transition_sequence": 1,
                "epoch_id": str(policy["epoch_id"]),
                "external_orchestrator": "p13i/machines/cs-sre",
                "external_job_id": "cs-sre",
                "policy": policy,
                "policy_sha256": policy_sha256,
                "next_command_digest": transition_command_digest,
                "receipt": receipt,
                "receipt_sha256": normalized_digest(receipt),
            }
            return {
                "reason": authorization["reason"],
                "prior_command_id": prior_command.command_id,
                "prior_command_type": anchor["prior_command_type"],
                "prior_anchor_kind": anchor["prior_anchor_kind"],
                "prior_reconciliation_id": anchor["prior_reconciliation_id"],
                "prior_reconciliation_resolution": anchor[
                    "prior_reconciliation_resolution"
                ],
                "prior_checkpoint_id": prior_checkpoint.checkpoint_id,
                "prior_generation_digest": prior_generation,
                "prior_material_digest": prior_material,
                "next_turn_ref": transition_turn_ref,
                "transition_sequence": 1,
                "next_command_digest": transition_command_digest,
                "authorization": authorization,
            }

        first_prior, first_checkpoint = completed_epoch_command("epoch-1")
        first_transition_payload = epoch_transition_payload(
            first_epoch_goal.goal_id,
            first_policy,
            first_prior,
            first_checkpoint,
        )
        first_transition_headers = {
            **headers,
            "Idempotency-Key": "contract-epoch-1-transition",
        }
        first_transition = await client.post(
            "/v1/sessions/" + session_id + "/dispatch-invalidations",
            headers=first_transition_headers,
            json=first_transition_payload,
        )
        assert first_transition.status == 201, await first_transition.text()
        next_goal_envelope = copy.deepcopy(goal_envelope)
        second_policy = epoch_policy("contract-epoch-2")
        next_goal_envelope["constraints"] = [
            "Pause after an incident.",
            "Invariant epoch 2.",
            "dispatch-generation-transition-policy-sha256:"
            + normalized_digest(second_policy),
            "dispatch-generation-transition-epoch:contract-epoch-2",
        ]
        rollover_receipt = {
            "actor": "test-operator",
            "change": "rollover-sre-epoch-2",
        }
        rollover_payload = copy.deepcopy(adoption_payload)
        rollover_payload["constraints"] = next_goal_envelope["constraints"]
        rollover_payload["authorization"] = {
            "schema": ("p13i/agent-harness/session-contract-adoption-authorization/v1"),
            "session_id": session_id,
            "external_ref": adoption_payload["external_ref"],
            "previous_goal_digest": goal_contract_digest(first_epoch_goal),
            "goal_envelope_digest": normalized_digest(next_goal_envelope),
            "receipt": rollover_receipt,
            "receipt_sha256": normalized_digest(rollover_receipt),
        }
        rollover = await client.post(
            "/v1/sessions/" + session_id + "/contract-adoptions",
            headers={**headers, "Idempotency-Key": "rollover-cs-sre-epoch-2"},
            json=rollover_payload,
        )
        assert rollover.status == 201, await rollover.text()
        rolled = await rollover.json()
        assert rolled["session"]["session_id"] == session_id
        assert rolled["goal"]["kind"] == "invariant"
        assert rolled["goal"]["completion_policy"] == "never"
        assert service.store.get_goal(first_epoch_goal.goal_id).status == "cancelled"
        old_epoch_replay = await client.post(
            "/v1/sessions/" + session_id + "/dispatch-invalidations",
            headers=first_transition_headers,
            json=first_transition_payload,
        )
        assert old_epoch_replay.status in {400, 409}
        second_epoch_goal = service.store.goal_for_session(session_id)
        assert second_epoch_goal is not None
        second_prior, second_checkpoint = completed_epoch_command("epoch-2")
        second_transition_payload = epoch_transition_payload(
            second_epoch_goal.goal_id,
            second_policy,
            second_prior,
            second_checkpoint,
        )
        second_transition = await client.post(
            "/v1/sessions/" + session_id + "/dispatch-invalidations",
            headers={
                **headers,
                "Idempotency-Key": "contract-epoch-2-transition",
            },
            json=second_transition_payload,
        )
        assert second_transition.status == 201, await second_transition.text()
        second_transition_body = await second_transition.json()
        assert second_transition_body["invalidation"]["transition_sequence"] == 1
        assert second_transition_body["invalidation"]["goal_id"] == (
            second_epoch_goal.goal_id
        )
        assert second_transition_body["invalidation"]["epoch_id"] == (
            "contract-epoch-2"
        )
        proof_response = await client.get(
            "/v1/sessions/" + session_id + "/proof",
            headers=headers,
        )
        proof = (await proof_response.json())["proof"]
        assert len(proof["goal_contract_adoptions"]) == 2
        assert all(
            item["session_id"] == session_id
            for item in proof["goal_contract_adoptions"]
        )
        assert all(
            item["receipt_digest_valid"] is True
            for item in proof["authorization_receipts"]
        )
        transition_receipts = proof["dispatch_transition_ledger"]["receipts"]
        assert proof["dispatch_transition_ledger"]["complete"] is True
        assert [item["transition_sequence"] for item in transition_receipts] == [1, 1]
        assert [item["epoch_id"] for item in transition_receipts] == [
            "contract-epoch-1",
            "contract-epoch-2",
        ]
        assert [item["goal_id"] for item in transition_receipts] == [
            first_epoch_goal.goal_id,
            second_epoch_goal.goal_id,
        ]
        exported = service.store.export_session(session_id)
        restored = StateStore(tmp_path / "legacy-restored.sqlite3")
        restored.import_session(
            exported,
            worktree=str(workspace),
            owner_host="restored-host",
            owner_epoch=2,
        )
        restored_session = restored.get_session(session_id)
        assert restored_session.session_id == session_id
        assert restored_session.creation_digest == original_creation_digest
        restored_tables = restored.portable_session(session_id)["tables"]
        assert len(restored_tables["goal_contract_adoptions"]) == 2
        assert len(restored_tables["authorization_receipts"]) == 4
        assert len(restored_tables["dispatch_transition_policies"]) == 2
        assert len(restored_tables["dispatch_transition_ledger"]) == 2
        restored.close()
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
            **unused: object,
        ) -> dict[str, object]:
            del payload, unused
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


@pytest.mark.asyncio
async def test_api_dispatch_transition_is_exact_idempotent_and_checkpoint_bound(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "transition-repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "transition-state"),
        worker_manager=Workers(),
    )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    external_ref = {"orchestrator": "machines", "job_id": "builder-stage"}
    try:
        session = service.create_session(
            {
                "workspace": str(workspace),
                "direct": True,
                "execution_profile": "unattended",
                "external_ref": external_ref,
            }
        )
        next_turn_ref = {"step_id": "verify", "agent_role": "verifier"}
        next_command_payload = {
            "text": "Verify reviewed implementation.",
            "provider": "codex",
            "turn_ref": next_turn_ref,
        }
        next_command_digest = command_envelope_digest(
            "message",
            next_command_payload,
            "unattended",
        )
        publish_turn_ref = {"step_id": "publish", "agent_role": "publisher"}
        publish_command_payload = {
            "text": "Publish the verified implementation.",
            "provider": "codex",
            "turn_ref": publish_turn_ref,
        }
        publish_command_digest = command_envelope_digest(
            "message",
            publish_command_payload,
            "unattended",
        )
        policy = {
            "schema": ("p13i/agent-harness/dispatch-generation-transition-policy/v1"),
            "session_id": session.session_id,
            "external_ref": external_ref,
            "epoch_id": "api-transition-epoch-1",
            "allowed_agent_roles": ["verifier", "publisher"],
            "allowed_step_prefixes": ["verify", "publish"],
            "max_transitions": 2,
            "transitions": [
                {
                    "sequence": 1,
                    "next_turn_ref": next_turn_ref,
                    "next_command_digest": next_command_digest,
                },
                {
                    "sequence": 2,
                    "next_turn_ref": publish_turn_ref,
                    "next_command_digest": publish_command_digest,
                },
            ],
        }
        service.store.create_goal(
            create_goal(
                session.session_id,
                "Run exact reviewed stages.",
                constraints=(
                    "dispatch-generation-transition-policy-sha256:"
                    + normalized_digest(policy),
                    "dispatch-generation-transition-epoch:" + str(policy["epoch_id"]),
                ),
            )
        )
        prior = service.store.enqueue_command(
            session.session_id,
            "message",
            {
                "text": "Review without changes.",
                "turn_ref": {"step_id": "review", "agent_role": "reviewer"},
            },
            "prior-review",
        )
        checkpoint = checkpoint_workspace(
            session,
            service.blobs,
            sequence=service.store.last_sequence(session.session_id),
            provider="claude",
            native_session_id="claude-review",
            context_text="reviewed",
        )
        service.store.add_checkpoint(checkpoint)
        service.store.resolve_command(
            prior.command_id,
            "complete",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "workspace_material_digest": inspect_workspace(workspace)[0],
            },
        )
        anchor = service.store.dispatch_transition_anchor(session.session_id)
        assert anchor["eligible"] is True
        reason = "Advance one completed orchestration stage."
        prior_generation_digest = str(anchor["prior_generation_digest"])
        policy_sha256 = normalized_digest(policy)
        retained_receipt = {
            "session_id": session.session_id,
            "external_ref": external_ref,
            "goal_id": str(service.store.goal_for_session(session.session_id).goal_id),
            "prior_command_id": prior.command_id,
            "prior_command_type": anchor["prior_command_type"],
            "prior_anchor_kind": anchor["prior_anchor_kind"],
            "prior_reconciliation_id": anchor["prior_reconciliation_id"],
            "prior_reconciliation_resolution": anchor[
                "prior_reconciliation_resolution"
            ],
            "prior_checkpoint_id": checkpoint.checkpoint_id,
            "prior_generation_digest": prior_generation_digest,
            "prior_material_digest": inspect_workspace(workspace)[0],
            "next_turn_ref": next_turn_ref,
            "transition_sequence": 1,
            "epoch_id": str(policy["epoch_id"]),
            "policy_sha256": policy_sha256,
            "next_command_digest": next_command_digest,
        }
        authorization = {
            "schema": (
                "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
            ),
            "session_id": session.session_id,
            "goal_id": str(service.store.goal_for_session(session.session_id).goal_id),
            "reason": reason,
            "prior_command_id": prior.command_id,
            "prior_command_type": anchor["prior_command_type"],
            "prior_anchor_kind": anchor["prior_anchor_kind"],
            "prior_reconciliation_id": anchor["prior_reconciliation_id"],
            "prior_reconciliation_resolution": anchor[
                "prior_reconciliation_resolution"
            ],
            "prior_checkpoint_id": checkpoint.checkpoint_id,
            "prior_generation_digest": prior_generation_digest,
            "prior_material_digest": inspect_workspace(workspace)[0],
            "next_turn_ref": next_turn_ref,
            "transition_sequence": 1,
            "epoch_id": str(policy["epoch_id"]),
            "external_orchestrator": external_ref["orchestrator"],
            "external_job_id": external_ref["job_id"],
            "policy": policy,
            "policy_sha256": policy_sha256,
            "next_command_digest": next_command_digest,
            "receipt": retained_receipt,
            "receipt_sha256": normalized_digest(retained_receipt),
        }
        payload = {
            "reason": reason,
            "prior_command_id": prior.command_id,
            "prior_command_type": anchor["prior_command_type"],
            "prior_anchor_kind": anchor["prior_anchor_kind"],
            "prior_reconciliation_id": anchor["prior_reconciliation_id"],
            "prior_reconciliation_resolution": anchor[
                "prior_reconciliation_resolution"
            ],
            "prior_checkpoint_id": checkpoint.checkpoint_id,
            "prior_generation_digest": prior_generation_digest,
            "prior_material_digest": inspect_workspace(workspace)[0],
            "next_turn_ref": next_turn_ref,
            "transition_sequence": 1,
            "next_command_digest": next_command_digest,
            "authorization": authorization,
        }
        transition_headers = {
            **headers,
            "Idempotency-Key": "review-to-verifier",
        }

        def altered_transition(
            *,
            transition_sequence: int = 1,
            prior_checkpoint_id: str = checkpoint.checkpoint_id,
            prior_generation: str = prior_generation_digest,
        ) -> dict[str, object]:
            altered = copy.deepcopy(payload)
            altered_authorization = altered["authorization"]
            assert isinstance(altered_authorization, dict)
            altered_receipt = altered_authorization["receipt"]
            assert isinstance(altered_receipt, dict)
            for target in (altered, altered_authorization, altered_receipt):
                target["transition_sequence"] = transition_sequence
                target["prior_checkpoint_id"] = prior_checkpoint_id
                target["prior_generation_digest"] = prior_generation
            altered_authorization["receipt_sha256"] = normalized_digest(altered_receipt)
            return altered

        malformed_future = copy.deepcopy(payload)
        malformed_authorization = malformed_future["authorization"]
        assert isinstance(malformed_authorization, dict)
        malformed_policy = malformed_authorization["policy"]
        assert isinstance(malformed_policy, dict)
        malformed_stages = malformed_policy["transitions"]
        assert isinstance(malformed_stages, list)
        malformed_stage = malformed_stages[1]
        assert isinstance(malformed_stage, dict)
        malformed_stage["next_command_digest"] = "bad"
        malformed_policy_sha256 = normalized_digest(malformed_policy)
        malformed_authorization["policy_sha256"] = malformed_policy_sha256
        malformed_receipt = malformed_authorization["receipt"]
        assert isinstance(malformed_receipt, dict)
        malformed_receipt["policy_sha256"] = malformed_policy_sha256
        malformed_authorization["receipt_sha256"] = normalized_digest(malformed_receipt)
        malformed_future_response = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "malformed-future-stage"},
            json=malformed_future,
        )
        assert malformed_future_response.status == 400
        assert "stage digest" in await malformed_future_response.text()

        out_of_order = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "skipped-transition"},
            json=altered_transition(transition_sequence=2),
        )
        assert out_of_order.status == 400, await out_of_order.text()
        forged_checkpoint = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "forged-checkpoint"},
            json=altered_transition(prior_checkpoint_id=new_uuid()),
        )
        assert forged_checkpoint.status == 409
        forged_generation = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "forged-generation"},
            json=altered_transition(prior_generation="b" * 64),
        )
        assert forged_generation.status == 409
        created = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers=transition_headers,
            json=payload,
        )
        assert created.status == 201, await created.text()
        transition = (await created.json())["invalidation"]
        assert transition["prior_command_id"] == prior.command_id
        assert transition["next_turn_ref"] == next_turn_ref
        assert transition["authorization_digest"] == normalized_digest(authorization)
        with service.store._lock:
            retained_authorization = service.store._connection.execute(
                """
                SELECT payload_json FROM authorization_receipts
                WHERE operation_id = ?
                """,
                (transition["invalidation_id"],),
            ).fetchone()
            retained_policies = service.store._connection.execute(
                "SELECT COUNT(*) AS count FROM dispatch_transition_policies"
            ).fetchone()
        assert retained_authorization is not None
        retained_payload = json.loads(str(retained_authorization["payload_json"]))
        assert "policy" not in retained_payload
        assert retained_payload["policy_ref"]["policy_sha256"] == policy_sha256
        assert retained_policies is not None
        assert int(retained_policies["count"]) == 1

        replay = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers=transition_headers,
            json=payload,
        )
        assert replay.status == 201
        assert (await replay.json())["invalidation"] == transition

        same_sequence = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "sequence-one-replay"},
            json=payload,
        )
        assert same_sequence.status == 409

        verify_command = service.store.enqueue_command(
            session.session_id,
            "message",
            next_command_payload,
            "execute-verify-transition",
        )
        claimed_verify = service.store.claim_command(session.session_id)
        assert claimed_verify is not None
        assert claimed_verify.command_id == verify_command.command_id
        service.store.create_command_envelope(
            verify_command.command_id,
            session.session_id,
            "unattended",
            {"max_attempts": 1},
        )
        assert (
            service.store.reserve_dispatch_generation_transition(
                session.session_id,
                verify_command.command_id,
                next_turn_ref,
                inspect_workspace(workspace)[0],
            )
            == "reserved"
        )
        service.store.register_worker(
            session.session_id,
            123,
            "api-transition-worker",
        )
        current_goal = service.store.goal_for_session(session.session_id)
        assert current_goal is not None
        admission = service.store.reserve_route_admission(
            verify_command.command_id,
            "codex",
            "unattended",
            effort="high",
            worker_incarnation="api-transition-worker",
            goal_id=current_goal.goal_id,
            max_concurrency=1,
            lease_expires_at="2099-01-01T00:00:00+00:00",
        )
        assert admission["admitted"] is True
        service.store.update_command_envelope(
            verify_command.command_id,
            state="complete",
        )
        verify_checkpoint = checkpoint_workspace(
            service.store.get_session(session.session_id),
            service.blobs,
            sequence=service.store.last_sequence(session.session_id),
            provider="codex",
            native_session_id="codex-verify",
            context_text="verified",
        )
        service.store.add_checkpoint(verify_checkpoint)
        service.store.resolve_command(
            verify_command.command_id,
            "complete",
            {
                "checkpoint_id": verify_checkpoint.checkpoint_id,
                "workspace_material_digest": inspect_workspace(workspace)[0],
            },
        )
        compact_anchor = service.store.dispatch_transition_anchor(session.session_id)
        assert compact_anchor["eligible"] is True
        compact_payload = _managed_transition_payload_from_anchor(
            service,
            session.session_id,
            policy,
            publish_turn_ref,
            publish_command_digest,
            compact_anchor,
            transition_sequence=2,
        )
        repeated_full_policy = copy.deepcopy(compact_payload)
        repeated_full_authorization = repeated_full_policy["authorization"]
        assert isinstance(repeated_full_authorization, dict)
        repeated_full_authorization.pop("policy_ref")
        repeated_full_authorization["policy"] = policy
        full_repeat = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "repeated-full-policy"},
            json=repeated_full_policy,
        )
        assert full_repeat.status == 400
        unknown_reference = copy.deepcopy(compact_payload)
        unknown_authorization = unknown_reference["authorization"]
        assert isinstance(unknown_authorization, dict)
        unknown_policy_ref = unknown_authorization["policy_ref"]
        assert isinstance(unknown_policy_ref, dict)
        unknown_policy_ref["policy_sha256"] = "f" * 64
        unknown_authorization["policy_sha256"] = "f" * 64
        unknown_reference_response = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "unknown-policy-reference"},
            json=unknown_reference,
        )
        assert unknown_reference_response.status == 409
        compact_headers = {
            **headers,
            "Idempotency-Key": "verify-to-publish",
        }
        compact_created = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers=compact_headers,
            json=compact_payload,
        )
        assert compact_created.status == 201, await compact_created.text()
        compact_transition = (await compact_created.json())["invalidation"]
        compact_authorization = compact_payload["authorization"]
        assert compact_transition["transition_sequence"] == 2
        assert compact_transition["authorization_digest"] == normalized_digest(
            compact_authorization
        )
        compact_replay = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers=compact_headers,
            json=compact_payload,
        )
        assert compact_replay.status == 201
        assert (await compact_replay.json())["invalidation"] == compact_transition

        other_external_ref = {
            "orchestrator": "machines",
            "job_id": "other-builder-stage",
        }
        other = service.create_session(
            {
                "workspace": str(workspace),
                "direct": True,
                "execution_profile": "unattended",
                "external_ref": other_external_ref,
            }
        )
        cross_session = await client.post(
            "/v1/sessions/" + other.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "cross-session-transition"},
            json=payload,
        )
        assert cross_session.status == 400

        forged = copy.deepcopy(payload)
        forged["next_turn_ref"] = {"step_id": "publish", "agent_role": "publisher"}
        forged_response = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "forged-transition"},
            json=forged,
        )
        assert forged_response.status == 400
    finally:
        await client.close()
        service.close()


@pytest.mark.asyncio
@pytest.mark.scale
async def test_api_executes_one_thousand_compact_transition_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "thousand-transition-repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "thousand-transition-state"),
        worker_manager=Workers(),
    )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    external_ref = {
        "orchestrator": "machines",
        "job_id": "thousand-transition-stage",
    }
    try:
        session = service.create_session(
            {
                "workspace": str(workspace),
                "direct": True,
                "execution_profile": "unattended",
                "external_ref": external_ref,
            }
        )
        stages: list[tuple[dict[str, str], dict[str, object], str]] = []
        policy_stages: list[dict[str, object]] = []
        for sequence in range(1, 1_001):
            turn_ref = {
                "step_id": "stage-" + str(sequence).zfill(4),
                "agent_role": "sre",
            }
            command_payload: dict[str, object] = {
                "text": "Run bounded stage " + str(sequence) + ".",
                "provider": "codex",
                "turn_ref": turn_ref,
            }
            command_digest = command_envelope_digest(
                "message",
                command_payload,
                "unattended",
            )
            stages.append((turn_ref, command_payload, command_digest))
            policy_stages.append(
                {
                    "sequence": sequence,
                    "next_turn_ref": turn_ref,
                    "next_command_digest": command_digest,
                }
            )
        policy: dict[str, object] = {
            "schema": ("p13i/agent-harness/dispatch-generation-transition-policy/v1"),
            "session_id": session.session_id,
            "external_ref": external_ref,
            "epoch_id": "api-thousand-transition-epoch",
            "allowed_agent_roles": ["sre"],
            "allowed_step_prefixes": ["stage-"],
            "max_transitions": 1_000,
            "transitions": policy_stages,
        }
        service.store.create_goal(
            create_goal(
                session.session_id,
                "Run one thousand exact provider-free orchestration stages.",
                constraints=(
                    "dispatch-generation-transition-policy-sha256:"
                    + normalized_digest(policy),
                    "dispatch-generation-transition-epoch:"
                    "api-thousand-transition-epoch",
                ),
            )
        )
        material_digest = inspect_workspace(workspace)[0]
        checkpoint = checkpoint_workspace(
            session,
            service.blobs,
            sequence=service.store.last_sequence(session.session_id),
            provider="codex",
            native_session_id="",
            context_text="provider-free transition proof",
        )
        service.store.add_checkpoint(checkpoint)
        monkeypatch.setattr(
            storage_module,
            "inspect_workspace",
            lambda unused: (material_digest, "stable scale-test workspace"),
        )
        prior = service.store.enqueue_command(
            session.session_id,
            "message",
            {"text": "Establish the transition epoch anchor."},
            "thousand-transition-prior",
        )
        service.store.resolve_command(
            prior.command_id,
            "complete",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "workspace_material_digest": material_digest,
            },
        )
        worker_incarnation = "api-thousand-transition-worker"
        service.store.register_worker(
            session.session_id,
            123,
            worker_incarnation,
        )
        goal = service.store.goal_for_session(session.session_id)
        assert goal is not None

        for sequence, stage in enumerate(stages, start=1):
            turn_ref, command_payload, command_digest = stage
            transition_payload = _managed_transition_payload_from_anchor(
                service,
                session.session_id,
                policy,
                turn_ref,
                command_digest,
                service.store.dispatch_transition_anchor(session.session_id),
                transition_sequence=sequence,
            )
            transition_response = await client.post(
                "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
                headers={
                    **headers,
                    "Idempotency-Key": "transition-" + str(sequence),
                },
                json=transition_payload,
            )
            assert transition_response.status == 201, await transition_response.text()
            message_response = await client.post(
                "/v1/sessions/" + session.session_id + "/messages",
                headers={
                    **headers,
                    "Idempotency-Key": "message-" + str(sequence),
                },
                json=command_payload,
            )
            assert message_response.status == 202
            command_value = (await message_response.json())["command"]
            claimed = service.store.claim_command(session.session_id)
            assert claimed is not None
            assert claimed.command_id == command_value["command_id"]
            service.store.create_command_envelope(
                claimed.command_id,
                session.session_id,
                "unattended",
                {"max_attempts": 1},
            )
            reservation = service.store.reserve_dispatch_generation_transition(
                session.session_id,
                claimed.command_id,
                turn_ref,
                material_digest,
            )
            assert reservation == "reserved"
            admission = service.store.reserve_route_admission(
                claimed.command_id,
                "codex",
                "unattended",
                effort="high",
                worker_incarnation=worker_incarnation,
                goal_id=goal.goal_id,
                max_concurrency=1,
                lease_expires_at="2099-01-01T00:00:00+00:00",
            )
            assert admission["admitted"] is True
            lease_id = str(admission["lease_id"])
            assert lease_id
            service.store.update_process_lease(
                lease_id,
                state="released",
            )
            service.store.update_command_envelope(
                claimed.command_id,
                state="complete",
            )
            service.store.resolve_command(
                claimed.command_id,
                "complete",
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "workspace_material_digest": material_digest,
                },
            )

        proof_response = await client.get(
            "/v1/sessions/" + session.session_id + "/proof?event_limit=1000",
            headers=headers,
        )
        assert proof_response.status == 200
        proof = (await proof_response.json())["proof"]
        transition_proof = proof["dispatch_transition_ledger"]
        assert transition_proof["complete"] is True
        assert len(transition_proof["receipts"]) == 1_000
        assert len(transition_proof["policies"]) == 1
        assert all(receipt["valid"] is True for receipt in transition_proof["receipts"])
        assert all(
            receipt["authorization"]["operation_id"] == receipt["invalidation_id"]
            and receipt["authorization"]["authorization_digest"]
            == receipt["authorization_digest"]
            and receipt["authorization"]["authorization_digest_valid"] is True
            and receipt["authorization"]["receipt_digest_valid"] is True
            and normalized_digest(receipt["authorization"]["binding"])
            == receipt["authorization"]["binding_digest"]
            and normalized_digest(receipt["safe_request_binding"])
            == receipt["safe_request_binding_digest"]
            and receipt["safe_request_binding"]["authorization_digest"]
            == receipt["authorization_digest"]
            for receipt in transition_proof["receipts"]
        )
        assert all(
            normalized_digest(
                _transition_source_receipt(receipt["authorization"]["binding"])
            )
            == receipt["authorization"]["receipt_digest"]
            == receipt["authorization"]["receipt_sha256"]
            for receipt in transition_proof["receipts"]
        )
        assert all(
            receipt["prior_command_type"] == "message"
            for receipt in transition_proof["receipts"]
        )
        assert all(
            receipt["prior_reconciliation_id"] == ""
            and receipt["prior_reconciliation_resolution"] == ""
            for receipt in transition_proof["receipts"]
        )
        all_events = list(proof["events"])
        while proof["event_range"]["complete"] is False:
            proof_response = await client.get(
                "/v1/sessions/"
                + session.session_id
                + "/proof?event_limit=1000&after_sequence="
                + str(proof["event_range"]["next_after_sequence"])
                + "&through_sequence="
                + str(proof["event_range"]["through_sequence"])
                + "&snapshot_id="
                + proof["snapshot_id"],
                headers=headers,
            )
            assert proof_response.status == 200
            proof = (await proof_response.json())["proof"]
            all_events.extend(proof["events"])
        assert proof["complete"] is True
        assert proof["truncated"] == []
        assert len(all_events) == proof["event_range"]["through_sequence"]
    finally:
        await client.close()
        service.close()


@pytest.mark.asyncio
async def test_api_exposes_and_accepts_control_command_transition_anchor(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "control-transition-repo"
    repository(workspace)
    service = HarnessService(
        paths(tmp_path / "control-transition-state"),
        worker_manager=Workers(),
    )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    external_ref = {
        "orchestrator": "p13i/machines/cs-builder",
        "job_id": "control-transition",
    }
    try:
        session = service.create_session(
            machines_session_payload(workspace, external_ref)
        )
        next_turn_ref = {
            "step_id": "resume-after-interrupt",
            "agent_role": "builder",
        }
        next_payload = {
            "text": "Resume after the bounded interrupt.",
            "turn_ref": next_turn_ref,
        }
        next_digest = command_envelope_digest(
            "message",
            next_payload,
            "unattended",
        )
        policy = _single_transition_policy(
            session.session_id,
            external_ref,
            "control-anchor-epoch",
            next_turn_ref,
            next_digest,
        )
        service.store.create_goal(
            create_goal(
                session.session_id,
                "Resume one interrupted managed stage.",
                constraints=(
                    "dispatch-generation-transition-policy-sha256:"
                    + normalized_digest(policy),
                    "dispatch-generation-transition-epoch:control-anchor-epoch",
                ),
            )
        )
        prior = service.store.enqueue_command(
            session.session_id,
            "message",
            {"text": "Complete the initial provider stage."},
            "control-anchor-provider",
        )
        checkpoint = checkpoint_workspace(
            session,
            service.blobs,
            sequence=service.store.last_sequence(session.session_id),
            provider="codex",
            native_session_id="codex-control-anchor",
            context_text="control anchor",
        )
        service.store.add_checkpoint(checkpoint)
        service.store.resolve_command(
            prior.command_id,
            "complete",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "workspace_material_digest": inspect_workspace(workspace)[0],
            },
        )
        control = service.store.enqueue_command(
            session.session_id,
            "interrupt",
            {},
            "control-anchor-interrupt",
        )
        service.store.resolve_command(control.command_id, "complete", {})

        anchor_response = await client.get(
            "/v1/sessions/" + session.session_id + "/dispatch-transition-anchor",
            headers=headers,
        )
        assert anchor_response.status == 200
        anchor = (await anchor_response.json())["transition_anchor"]
        proof_response = await client.get(
            "/v1/sessions/" + session.session_id + "/proof",
            headers=headers,
        )
        assert proof_response.status == 200
        assert (await proof_response.json())["proof"]["transition_anchor"] == anchor
        assert anchor["eligible"] is True
        assert anchor["prior_command_id"] == control.command_id
        assert anchor["prior_command_type"] == "interrupt"
        assert anchor["prior_anchor_kind"] == "control-command"
        assert anchor["prior_checkpoint_id"] == checkpoint.checkpoint_id
        assert len(anchor["prior_material_digest"]) == 64
        assert len(anchor["prior_generation_digest"]) == 64

        payload = _managed_transition_payload(
            service,
            session.session_id,
            policy,
            next_turn_ref,
            next_digest,
        )
        created = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "control-anchor-transition"},
            json=payload,
        )
        assert created.status == 201, await created.text()
        receipt = (await created.json())["invalidation"]
        assert receipt["prior_command_type"] == "interrupt"
        assert receipt["prior_anchor_kind"] == "control-command"
        assert receipt["prior_reconciliation_id"] == ""
        proof_response = await client.get(
            "/v1/sessions/" + session.session_id + "/proof",
            headers=headers,
        )
        proof_receipt = (await proof_response.json())["proof"][
            "dispatch_transition_ledger"
        ]["receipts"][0]
        assert proof_receipt["prior_command_type"] == "interrupt"
        assert proof_receipt["prior_anchor_kind"] == "control-command"
        assert proof_receipt["prior_reconciliation_id"] == ""
        assert proof_receipt["prior_reconciliation_resolution"] == ""
        assert (
            proof_receipt["authorization"]["binding"]["prior_command_id"]
            == control.command_id
        )
        assert (
            normalized_digest(proof_receipt["authorization"]["binding"])
            == (proof_receipt["authorization"]["binding_digest"])
        )
    finally:
        await client.close()
        service.close()


@pytest.mark.asyncio
async def test_api_accepts_only_exact_resolved_reconciliation_anchor(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "reconciliation-transition-repo"
    repository(workspace)
    workers = Workers()
    service = HarnessService(
        paths(tmp_path / "reconciliation-transition-state"),
        worker_manager=workers,
    )
    client = TestClient(TestServer(create_app(service, "test-token")))
    await client.start_server()
    headers = {"Authorization": "Bearer test-token"}
    external_ref = {
        "orchestrator": "p13i/machines/cs-sre",
        "job_id": "reconciliation-transition",
    }
    try:
        session = service.create_session(
            machines_session_payload(
                workspace,
                external_ref,
                permission_mode="full",
            )
        )
        next_turn_ref = {
            "step_id": "post-reconciliation",
            "agent_role": "sre",
        }
        next_payload = {
            "text": "Continue after exact reconciliation.",
            "turn_ref": next_turn_ref,
        }
        next_digest = command_envelope_digest(
            "message",
            next_payload,
            "unattended",
        )
        policy = _single_transition_policy(
            session.session_id,
            external_ref,
            "reconciliation-anchor-epoch",
            next_turn_ref,
            next_digest,
        )
        service.store.create_goal(
            create_goal(
                session.session_id,
                "Continue one reconciled managed stage.",
                constraints=(
                    "dispatch-generation-transition-policy-sha256:"
                    + normalized_digest(policy),
                    "dispatch-generation-transition-epoch:reconciliation-anchor-epoch",
                ),
            )
        )
        prior = service.store.enqueue_command(
            session.session_id,
            "message",
            {"text": "Cross one ambiguous provider boundary."},
            "reconciliation-anchor-provider",
        )
        claimed = service.store.claim_command(session.session_id)
        assert claimed is not None
        attempt = ProviderAttempt(
            attempt_id=new_uuid(),
            session_id=session.session_id,
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
            session.session_id,
            attempt.attempt_id,
        )
        checkpoint = checkpoint_workspace(
            session,
            service.blobs,
            sequence=service.store.last_sequence(session.session_id),
            provider="codex",
            native_session_id="",
            context_text="before ambiguous boundary",
        )
        service.store.add_checkpoint(checkpoint)
        service.store.record_dispatch_checkpoint(
            prior.command_id,
            attempt.attempt_id,
            turn_id,
            checkpoint.checkpoint_id,
        )
        service.store.create_command_envelope(
            prior.command_id,
            session.session_id,
            "unattended",
            {"max_attempts": 1},
        )
        service.store.register_worker(
            session.session_id,
            123,
            "reconciliation-old-worker",
        )
        admission = service.store.reserve_route_admission(
            prior.command_id,
            "codex",
            "unattended",
            effort="high",
            attempt_id=attempt.attempt_id,
            worker_incarnation="reconciliation-old-worker",
            goal_id=str(service.store.goal_for_session(session.session_id).goal_id),
            max_concurrency=1,
            lease_expires_at="2099-01-01T00:00:00+00:00",
        )
        assert admission["admitted"] is True
        service.store.register_worker(
            session.session_id,
            456,
            "reconciliation-new-worker",
        )
        recovery = await service.reconciliations.recover_after_restart(
            session.session_id
        )
        assert len(recovery.reconciliations) == 1
        record = recovery.reconciliations[0]
        unresolved_anchor = service.store.dispatch_transition_anchor(session.session_id)
        assert unresolved_anchor["eligible"] is False
        assert "unresolved" in unresolved_anchor["reason"]

        unresolved_payload = _managed_transition_payload_from_anchor(
            service,
            session.session_id,
            policy,
            next_turn_ref,
            next_digest,
            {
                "prior_command_id": prior.command_id,
                "prior_command_type": "message",
                "prior_anchor_kind": "resolved-reconciliation",
                "prior_reconciliation_id": record.reconciliation_id,
                "prior_reconciliation_resolution": "accept-current",
                "prior_checkpoint_id": str(
                    service.store.repetition_generation(session.session_id)[
                        "checkpoint_id"
                    ]
                ),
                "prior_generation_digest": str(
                    service.store.repetition_generation(session.session_id)[
                        "generation_digest"
                    ]
                ),
                "prior_material_digest": inspect_workspace(workspace)[0],
            },
        )
        unresolved = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "unresolved-transition"},
            json=unresolved_payload,
        )
        assert unresolved.status == 409

        resolved = await client.post(
            "/v1/reconciliations/" + record.reconciliation_id + "/resolution",
            headers={**headers, "Idempotency-Key": "accept-ambiguous-current"},
            json={
                "decision": "accept-current",
                "observed_workspace_digest": record.current_workspace_digest,
                "audit": {"actor": "integration-test"},
            },
        )
        assert resolved.status == 200, await resolved.text()
        anchor = service.store.dispatch_transition_anchor(session.session_id)
        assert anchor["eligible"] is True
        assert anchor["prior_anchor_kind"] == "resolved-reconciliation"
        assert anchor["prior_reconciliation_id"] == record.reconciliation_id
        assert anchor["prior_reconciliation_resolution"] == "accept-current"
        payload = _managed_transition_payload(
            service,
            session.session_id,
            policy,
            next_turn_ref,
            next_digest,
        )

        wrong_resolution = copy.deepcopy(payload)
        _rewrite_transition_binding(
            wrong_resolution,
            "prior_reconciliation_resolution",
            "restore-pre-turn",
        )
        wrong = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "wrong-reconciliation"},
            json=wrong_resolution,
        )
        assert wrong.status == 409
        stale_checkpoint = copy.deepcopy(payload)
        _rewrite_transition_binding(
            stale_checkpoint,
            "prior_checkpoint_id",
            new_uuid(),
        )
        stale = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "stale-reconciliation"},
            json=stale_checkpoint,
        )
        assert stale.status == 409

        created = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "reconciled-transition"},
            json=payload,
        )
        assert created.status == 201, await created.text()
        proof_response = await client.get(
            "/v1/sessions/" + session.session_id + "/proof",
            headers=headers,
        )
        proof_receipt = (await proof_response.json())["proof"][
            "dispatch_transition_ledger"
        ]["receipts"][0]
        assert proof_receipt["prior_command_type"] == "message"
        assert proof_receipt["prior_anchor_kind"] == "resolved-reconciliation"
        assert proof_receipt["prior_reconciliation_id"] == record.reconciliation_id
        assert proof_receipt["prior_reconciliation_resolution"] == "accept-current"
        assert (
            proof_receipt["authorization"]["binding"]["prior_reconciliation_id"]
            == record.reconciliation_id
        )
        assert (
            normalized_digest(proof_receipt["authorization"]["binding"])
            == (proof_receipt["authorization"]["binding_digest"])
        )
        replay = await client.post(
            "/v1/sessions/" + session.session_id + "/dispatch-invalidations",
            headers={**headers, "Idempotency-Key": "reconciled-transition-replay"},
            json=payload,
        )
        assert replay.status == 409
    finally:
        await client.close()
        service.close()
