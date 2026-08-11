"""End-to-end journeys through the provider-neutral execution core."""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agent_harness.reconciliation as reconciliation_module
import agent_harness.safety as safety_module
import agent_harness.worker as worker_module
import agent_harness.workspace as workspace_module
import agent_harness.workspace_state as workspace_state_module
from agent_harness import child_gate
from agent_harness import process_control as process_control_module
from agent_harness.blobs import BlobStore
from agent_harness.config import paths, prepare_paths
from agent_harness.errors import (
    ConflictError,
    HarnessError,
    ProviderExhaustedError,
    ProviderUnavailableError,
    SafetyGuardError,
    WorkerOwnershipLostError,
)
from agent_harness.goals import create_goal, make_evidence
from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import (
    CommandReceipt,
    ProviderAttempt,
    ReconciliationDecision,
    ReconciliationRecord,
)
from agent_harness.orchestration import command_envelope_digest, normalized_digest
from agent_harness.proof import proof_snapshot
from agent_harness.providers.base import (
    ApprovalHandler,
    ChildLaunchGate,
    EventHandler,
    PrePromptGate,
    ProviderAdapter,
    ProviderEvent,
    ProviderModel,
    ProviderResult,
    ProviderStatus,
)
from agent_harness.reconciliation import ReconciliationManager, inspect_workspace
from agent_harness.safety import SafetyConsumption, TurnGuard, limits_for
from agent_harness.scheduler import Scheduler
from agent_harness.service import HarnessService
from agent_harness.storage import STOPPED_SESSION_COMMANDS, StateStore
from agent_harness.usage import UsageSnapshot
from agent_harness.worker import SessionWorker
from agent_harness.workspace import checkpoint_workspace
from tests.test_support import session


@pytest.fixture(autouse=True)
def stable_state_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        safety_module.shutil,
        "disk_usage",
        lambda unused: SimpleNamespace(free=10 * 1024**3),
    )


class ScriptedAdapter(ProviderAdapter):
    def __init__(
        self,
        provider: str,
        *,
        fail_turns: int = 0,
        guard_turns: int = 0,
        usage_turns: int = 0,
        request_approval: bool = False,
        turn_delay: float = 0.0,
        cost: object | None = None,
        claims_cost_reporting: bool = False,
    ) -> None:
        self.provider_id = provider
        self.fail_turns = fail_turns
        self.guard_turns = guard_turns
        self.usage_turns = usage_turns
        self.request_approval = request_approval
        self.turn_delay = turn_delay
        self.cost = cost
        self.claims_cost_reporting = claims_cost_reporting
        self.prompts: list[str] = []
        self.native_inputs: list[str] = []
        self.efforts: list[str] = []
        self.permission_modes: list[str] = []
        self.child_agent_limits: list[int | None] = []
        self.approval_decisions: list[dict[str, Any]] = []
        self.steered: list[str] = []
        self.interruptions = 0
        self.process_running = False
        self.process_pid = 4242
        self.process_start = "proof-process-start"

    async def run_turn(
        self,
        *,
        workspace: Path,
        prompt: str,
        native_session_id: str,
        permission_mode: str,
        model: str,
        effort: str,
        event_handler: EventHandler,
        approval_handler: ApprovalHandler,
        child_launch_gate: ChildLaunchGate | None = None,
        pre_prompt_gate: PrePromptGate | None = None,
    ) -> ProviderResult:
        del workspace
        del model
        if pre_prompt_gate is not None:
            await pre_prompt_gate()
        self.prompts.append(prompt)
        self.native_inputs.append(native_session_id)
        self.efforts.append(effort)
        self.permission_modes.append(permission_mode)
        if child_launch_gate is None:
            self.child_agent_limits.append(None)
        else:
            self.child_agent_limits.append(child_launch_gate.limit)
        if self.fail_turns:
            self.fail_turns -= 1
            raise ProviderExhaustedError(self.provider_id)
        await event_handler(
            ProviderEvent(
                "provider.prompt.accepted",
                status="accepted",
                native_session_id=(self.provider_id + "-native-session"),
            )
        )
        if self.guard_turns:
            self.guard_turns -= 1
            for unused in range(3):
                del unused
                await event_handler(
                    ProviderEvent(
                        "tool.started",
                        text="Read",
                        metadata={"path": "SKILL.md"},
                        native_session_id=(self.provider_id + "-native-session"),
                    )
                )
                await event_handler(
                    ProviderEvent(
                        "tool.completed",
                        text="unchanged",
                    )
                )
            await asyncio.sleep(0.25)
        if self.usage_turns:
            self.usage_turns -= 1
            await event_handler(
                ProviderEvent(
                    "usage.updated",
                    metadata={
                        "input_tokens": 90_000,
                        "output_tokens": 20_000,
                        "total_tokens": 110_000,
                    },
                )
            )
            await asyncio.sleep(0.25)
        if self.request_approval:
            decision = await approval_handler(
                "tool.execute",
                {
                    "id": self.provider_id + "-approval",
                    "prompt": "Run the repository test command?",
                    "choices": [
                        {"id": "accept", "label": "Run"},
                        {"id": "decline", "label": "Skip"},
                    ],
                },
            )
            self.approval_decisions.append(decision)
        if self.turn_delay:
            await asyncio.sleep(self.turn_delay)
        await event_handler(
            ProviderEvent(
                "agent.message",
                text=self.provider_id + " completed the turn",
                status="complete",
                raw={"provider": self.provider_id},
            )
        )
        resolved_session_id = native_session_id
        if not resolved_session_id:
            resolved_session_id = self.provider_id + "-native-session"
        usage: dict[str, Any] = {"total_tokens": 42}
        if self.cost is not None:
            if isinstance(self.cost, dict):
                usage.update(self.cost)
            else:
                usage["total_cost_usd"] = self.cost
        return ProviderResult(
            provider=self.provider_id,
            native_session_id=resolved_session_id,
            native_turn_id=self.provider_id + "-turn",
            status="complete",
            usage=usage,
        )

    async def models(self, workspace: Path) -> tuple[ProviderModel, ...]:
        del workspace
        return (
            ProviderModel(
                self.provider_id + "-default",
                self.provider_id.title(),
                ("low", "medium", "high", "xhigh", "max"),
                200_000,
                default=True,
            ),
        )

    def status(self) -> ProviderStatus:
        capabilities = {
            "approval",
            "checkpoint",
            "hooks",
            "mcp",
            "plugins",
            "proof-fault-barrier",
            "proof-service-fault-barrier",
            "resume",
            "skills",
            "streaming",
            "subagents",
            "tools",
            "worktree",
        }
        if self.claims_cost_reporting:
            capabilities.add("cost-reporting")
        return ProviderStatus(
            provider=self.provider_id,
            ready=True,
            detail="scripted journey provider",
            capabilities=frozenset(capabilities),
        )

    async def interrupt(self) -> None:
        self.interruptions += 1

    async def steer(self, text: str) -> None:
        self.steered.append(text)

    def process_identity(self) -> tuple[int, str]:
        if not self.process_running:
            return (0, "")
        return (self.process_pid, self.process_start)


class ZeroProgressAdapter(ScriptedAdapter):
    def __init__(self, provider: str, *, write_file: bool = False) -> None:
        super().__init__(provider)
        self.write_file = write_file

    async def run_turn(
        self,
        *,
        workspace: Path,
        prompt: str,
        native_session_id: str,
        permission_mode: str,
        model: str,
        effort: str,
        event_handler: EventHandler,
        approval_handler: ApprovalHandler,
        child_launch_gate: ChildLaunchGate | None = None,
        pre_prompt_gate: PrePromptGate | None = None,
    ) -> ProviderResult:
        del permission_mode
        del model
        del approval_handler
        if pre_prompt_gate is not None:
            await pre_prompt_gate()
        self.prompts.append(prompt)
        self.efforts.append(effort)
        if child_launch_gate is None:
            self.child_agent_limits.append(None)
        else:
            self.child_agent_limits.append(child_launch_gate.limit)
        await event_handler(
            ProviderEvent(
                "provider.prompt.accepted",
                status="accepted",
                native_session_id=(self.provider_id + "-native-session"),
            )
        )
        if self.write_file:
            (workspace / "zero-progress-output.txt").write_text(
                "written without narration\n"
            )
        resolved_session_id = native_session_id
        if not resolved_session_id:
            resolved_session_id = self.provider_id + "-native-session"
        return ProviderResult(
            provider=self.provider_id,
            native_session_id=resolved_session_id,
            native_turn_id=self.provider_id + "-turn",
            status="complete",
            usage={"total_tokens": 1},
        )


class JourneyRig:
    def __init__(
        self,
        root: Path,
        *,
        claude: ScriptedAdapter | None = None,
        codex: ScriptedAdapter | None = None,
        external_ref: dict[str, str] | None = None,
        goal_constraints: tuple[str, ...] = (),
    ) -> None:
        workspace = root / "workspace"
        _repository(workspace)
        harness_paths = paths(root / "state")
        prepare_paths(harness_paths)
        self.store = StateStore(harness_paths.database)
        self.blobs = BlobStore(harness_paths.blobs)
        self.session = session(workspace)
        if external_ref is not None:
            self.session = replace(
                self.session,
                external_ref=external_ref,
            )
        self.store.create_session(self.session)
        if goal_constraints:
            self.store.create_goal(
                create_goal(
                    self.session.session_id,
                    "Complete the bounded proof fault stage.",
                    constraints=goal_constraints,
                    permitted_providers=("claude", "codex"),
                    permitted_efforts=("low", "medium", "high"),
                )
            )
        self.store.set_session_safety(
            self.session.session_id,
            "interactive",
        )
        if claude is None:
            claude = ScriptedAdapter("claude")
        if codex is None:
            codex = ScriptedAdapter("codex")
        self.adapters = {
            "claude": claude,
            "codex": codex,
        }
        self.scheduler = Scheduler(self.store, self.adapters)
        self.worker = SessionWorker(
            self.store,
            self.blobs,
            self.scheduler,
            self.adapters,
            self.session.session_id,
        )
        self.store.register_worker(
            self.session.session_id,
            123,
            self.worker.incarnation,
        )

    def prime_capacity(
        self,
        *,
        claude: float = 20.0,
        codex: float = 20.0,
    ) -> None:
        self.scheduler._usage_cache = {
            "claude": _usage("claude", claude),
            "codex": _usage("codex", codex),
        }
        self.scheduler._usage_at = asyncio.get_running_loop().time()

    async def message(
        self,
        text: str,
        **route: Any,
    ) -> CommandReceipt:
        receipt = self.enqueue_message(text, **route)
        await self.execute(receipt.command_id)
        return self.store.get_command(receipt.command_id)

    def enqueue_message(
        self,
        text: str,
        *,
        idempotency_key: str = "",
        **route: Any,
    ) -> CommandReceipt:
        if not idempotency_key:
            idempotency_key = new_uuid()
        payload = {"text": text, **route}
        receipt = self.store.enqueue_command(
            self.session.session_id,
            "message",
            payload,
            idempotency_key,
        )
        self.store.append_event(
            self.session.session_id,
            "user.message",
            role="user",
            text=text,
            status="accepted",
            metadata={"command_id": receipt.command_id},
        )
        return receipt

    async def execute(self, command_id: str) -> None:
        claimed = self.store.claim_command(self.session.session_id)
        assert claimed is not None
        assert claimed.command_id == command_id
        await self.worker._message(claimed)

    def authorize_xhigh(self, command_id: str, provider: str) -> None:
        self.store.create_xhigh_authorization(
            self.session.session_id,
            command_id,
            provider,
            authorization_request_digest="a" * 64,
            idempotency_key=new_uuid(),
            expires_at="2099-01-01T00:00:00+00:00",
        )

    def close(self) -> None:
        self.store.close()


@pytest.mark.asyncio
async def test_per_turn_permission_narrows_provider_execution_and_proof(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude")
    rig = JourneyRig(tmp_path, claude=claude)
    try:
        rig.prime_capacity(claude=10.0, codex=70.0)
        receipt = await rig.message(
            "Review without changing files.",
            provider="claude",
            permission_mode="read-only",
        )
        assert receipt.status == "complete"
        assert claude.permission_modes == ["read-only"]
        routing_events = [
            event
            for event in rig.store.events(rig.session.session_id)
            if event.event_type == "routing.selected"
        ]
        assert routing_events[0].metadata["parent_permission_mode"] == ("read-only")
        assert routing_events[0].metadata["child_permission_mode"] == ("read-only")
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_paused_session_resumes_and_stops(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.store.update_session(
        rig.session.session_id,
        lifecycle="paused",
        attention="idle",
    )
    resumed = rig.store.enqueue_command(
        rig.session.session_id,
        "resume",
        {},
        new_uuid(),
    )
    worker_task = asyncio.create_task(rig.worker.run())
    try:
        for unused in range(1000):
            del unused
            current = rig.store.get_command(resumed.command_id)
            if current.status == "complete":
                break
            await asyncio.sleep(0.01)
        assert current.status == "complete"
        assert rig.store.get_session(rig.session.session_id).lifecycle == "running"

        stopped = rig.store.enqueue_command(
            rig.session.session_id,
            "stop",
            {},
            new_uuid(),
        )
        await asyncio.wait_for(worker_task, timeout=2)

        assert rig.store.get_command(stopped.command_id).status == "complete"
        assert rig.store.get_session(rig.session.session_id).lifecycle == "stopped"
    finally:
        if not worker_task.done():
            worker_task.cancel()
        rig.close()


@pytest.mark.asyncio
async def test_e2e_provider_resume_and_cross_provider_continuity(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
        instructions = Path(rig.session.worktree) / "AGENTS.md"
        instructions.write_text(
            "UNIQUE_PROVIDER_NATIVE_INSTRUCTION",
            encoding="utf-8",
        )
        first = await rig.message("Implement the parser.", provider="codex")
        second = await rig.message("Now add tests.", provider="codex")
        third = await rig.message(
            "Review the completed change.",
            provider="claude",
            workload="review",
        )
        fourth = await rig.message(
            "Apply the review findings.",
            provider="codex",
        )
        fifth = await rig.message(
            "Certify the resulting state.",
            provider="claude",
            workload="review",
        )

        codex = rig.adapters["codex"]
        claude = rig.adapters["claude"]
        assert first.status == "complete"
        assert second.status == "complete"
        assert third.status == "complete"
        assert fourth.status == "complete"
        assert fifth.status == "complete"
        assert codex.native_inputs == [
            "",
            "codex-native-session",
            "codex-native-session",
        ]
        assert "UNIQUE_PROVIDER_NATIVE_INSTRUCTION" in codex.prompts[0]
        assert codex.prompts[1] == "Now add tests."
        assert claude.native_inputs == ["", "claude-native-session"]
        assert "# Harness session" in claude.prompts[0]
        assert "Implement the parser." in claude.prompts[0]
        assert "codex completed the turn" in claude.prompts[0]
        assert "# Next instruction" in claude.prompts[0]
        assert "UNIQUE_PROVIDER_NATIVE_INSTRUCTION" in claude.prompts[0]
        assert len(rig.store.checkpoints(rig.session.session_id)) == 10
        dispatches = rig.store._connection.execute(
            """
            SELECT crossed_boundary, state, checkpoint_id
            FROM command_dispatches ORDER BY created_at
            """
        ).fetchall()
        assert len(dispatches) == 5
        assert all(row["crossed_boundary"] == 1 for row in dispatches)
        assert all(row["state"] == "complete" for row in dispatches)
        deliveries = rig.store._connection.execute(
            """
            SELECT command_id, provider, checkpoint_id, context_digest, state,
                payload_digest, accepted_at, transport
            FROM context_deliveries ORDER BY delivered_at
            """
        ).fetchall()
        assert [row["provider"] for row in deliveries] == [
            "codex",
            "codex",
            "claude",
            "codex",
            "claude",
        ]
        assert [row["transport"] for row in deliveries] == [
            "context-package",
            "native-resume",
            "context-package",
            "context-package",
            "context-package",
        ]
        assert [row["command_id"] for row in deliveries] == [
            first.command_id,
            second.command_id,
            third.command_id,
            fourth.command_id,
            fifth.command_id,
        ]
        assert deliveries[3]["provider"] == "codex"
        assert deliveries[3]["transport"] == "context-package"
        assert len({row["context_digest"] for row in deliveries}) == 5
        assert all(row["state"] == "delivered" for row in deliveries)
        assert all(row["payload_digest"] for row in deliveries)
        assert all(row["accepted_at"] for row in deliveries)
        assert deliveries[0]["checkpoint_id"] == dispatches[0]["checkpoint_id"]
        assert deliveries[1]["checkpoint_id"] == dispatches[1]["checkpoint_id"]
        assert deliveries[2]["checkpoint_id"] == dispatches[2]["checkpoint_id"]
        assert deliveries[3]["checkpoint_id"] == dispatches[3]["checkpoint_id"]
        assert deliveries[4]["checkpoint_id"] == dispatches[4]["checkpoint_id"]
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_native_resume_delivery_identity_includes_the_command(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude")
    rig = JourneyRig(tmp_path, claude=claude)
    rig.prime_capacity()
    try:
        established = await rig.message(
            "Establish the native session.",
            provider="claude",
        )
        first = await rig.message(
            "Run the unchanged tests.",
            provider="claude",
        )
        (Path(rig.session.worktree) / "operator-change.txt").write_text(
            "new material generation\n",
            encoding="utf-8",
        )
        second = await rig.message(
            "Run the unchanged tests.",
            provider="claude",
        )

        assert established.status == "complete"
        assert first.status == "complete"
        assert second.status == "complete"
        deliveries = rig.store._connection.execute(
            """
            SELECT command_id, context_digest, transport
            FROM context_deliveries ORDER BY delivered_at
            """
        ).fetchall()
        assert len(deliveries) == 3
        assert len({row["command_id"] for row in deliveries}) == 3
        assert len({row["context_digest"] for row in deliveries}) == 3
        assert [row["transport"] for row in deliveries] == [
            "context-package",
            "native-resume",
            "native-resume",
        ]
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_missing_native_resume_does_not_accept_context_delivery(
    tmp_path: Path,
) -> None:
    class MissingNativeAdapter(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            self.prompts.append(str(values["prompt"]))
            self.native_inputs.append(str(values["native_session_id"]))
            event_handler = values["event_handler"]
            await event_handler(
                ProviderEvent(
                    "recovery.native_resume_missing",
                    status="failed",
                    metadata={
                        "provider": "claude",
                        "native_session_id": "missing-native",
                    },
                )
            )
            raise ProviderUnavailableError("claude resume")

    claude = MissingNativeAdapter("claude")
    rig = JourneyRig(tmp_path, claude=claude)
    now = utc_now()
    rig.store.create_attempt(
        ProviderAttempt(
            attempt_id=new_uuid(),
            session_id=rig.session.session_id,
            provider="claude",
            native_session_id="missing-native",
            model="claude-default",
            effort="medium",
            auth_mode="subscription",
            status="complete",
            started_at=now,
            ended_at=now,
        )
    )
    rig.store.update_session(
        rig.session.session_id,
        active_provider="codex",
    )
    rig.prime_capacity()
    try:
        receipt = await rig.message(
            "Resume Claude with transferred context.",
            provider="claude",
        )

        assert receipt.status == "failed"
        deliveries = rig.store.portable_session(rig.session.session_id)["tables"][
            "context_deliveries"
        ]
        assert len(deliveries) == 1
        assert deliveries[0]["state"] == "prepared"
        assert deliveries[0]["accepted_at"] == ""
        assert any(
            event.event_type == "recovery.native_resume_missing"
            for event in rig.store.all_events(rig.session.session_id)
        )
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_missing_native_state_falls_back_to_canonical_context(
    tmp_path: Path,
) -> None:
    class RecoveringAdapter(ScriptedAdapter):
        def __init__(self) -> None:
            super().__init__("claude")
            self.native_available = True
            self.generation = 0

        def native_session_available(
            self,
            workspace: Path,
            native_session_id: str,
        ) -> bool:
            del workspace
            del native_session_id
            return self.native_available

        async def run_turn(self, **values: Any) -> ProviderResult:
            result = await super().run_turn(**values)
            if not str(values["native_session_id"]):
                self.generation += 1
                self.native_available = True
                return replace(
                    result,
                    native_session_id="claude-native-" + str(self.generation),
                )
            return result

    claude = RecoveringAdapter()
    rig = JourneyRig(tmp_path, claude=claude)
    rig.prime_capacity()
    try:
        first = await rig.message(
            "Create the first bounded artifact.",
            provider="claude",
        )
        claude.native_available = False
        second = await rig.message(
            "Continue without replaying the first mutation.",
            provider="claude",
        )

        assert first.status == "complete"
        assert second.status == "complete"
        assert claude.native_inputs == ["", ""]
        assert "Create the first bounded artifact." in claude.prompts[1]
        assert claude.prompts[1].count("# Next instruction") == 1
        assert claude.prompts[1].endswith(
            "Continue without replaying the first mutation."
        )
        attempts = rig.store.attempts(rig.session.session_id)
        assert attempts[0].native_session_id == "claude-native-1"
        assert attempts[1].native_session_id == "claude-native-2"
        fallback = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "provider.native_resume.fallback"
        ]
        assert len(fallback) == 1
        assert fallback[0].metadata["unavailable_native_session_id"] == (
            "claude-native-1"
        )
        assert fallback[0].metadata["context_payload_digest"]
    finally:
        rig.close()


@pytest.mark.parametrize(
    "failure_stage",
    ("attempt", "checkpoint", "context-delivery"),
)
@pytest.mark.asyncio
async def test_pre_admission_setup_failure_leaves_no_capacity_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    receipt = rig.store.enqueue_command(
        rig.session.session_id,
        "message",
        {"text": "Exercise setup failure."},
        new_uuid(),
    )
    rig.store.append_event(
        rig.session.session_id,
        "user.message",
        role="user",
        text="Exercise setup failure.",
        status="accepted",
        metadata={"command_id": receipt.command_id},
    )
    claimed = rig.store.claim_command(rig.session.session_id)
    assert claimed is not None

    def fail(*unused: object, **unused_values: object) -> None:
        del unused
        del unused_values
        raise RuntimeError("injected " + failure_stage + " failure")

    if failure_stage == "attempt":
        monkeypatch.setattr(rig.store, "create_attempt", fail)
    if failure_stage == "checkpoint":
        monkeypatch.setattr(worker_module, "checkpoint_workspace", fail)
    if failure_stage == "context-delivery":
        monkeypatch.setattr(rig.store, "prepare_context_delivery", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            await rig.worker._execute_message(claimed)
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["provider"] == ""
        assert envelope["state"] == "reserved"
        assert rig.store.active_process_leases() == []
        assert rig.store.active_unattended_provider_count("claude") == 0
        assert rig.store.active_unattended_provider_count("codex") == 0
        if failure_stage == "context-delivery":
            attempts = rig.store.attempts(rig.session.session_id)
            assert [attempt.status for attempt in attempts] == ["failed"]
            transition = rig.store._connection.execute(
                """
                SELECT turns.status AS turn_status,
                    command_dispatches.state AS dispatch_state
                FROM turns JOIN command_dispatches USING(turn_id)
                WHERE command_dispatches.attempt_id = ?
                """,
                (attempts[0].attempt_id,),
            ).fetchone()
            assert transition is not None
            assert transition["turn_status"] == "failed"
            assert transition["dispatch_state"] == "failed"
        rig.store.recover_interrupted_commands(
            rig.session.session_id,
            "recovered-digest",
            "recovered summary",
        )
        assert rig.store.get_command(receipt.command_id).status == "queued"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_worker_cancellation_stops_provider_before_lease_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingAdapter(ScriptedAdapter):
        def __init__(self) -> None:
            super().__init__("claude")
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.provider_stopped = False

        async def run_turn(self, **values: Any) -> ProviderResult:
            event_handler = values["event_handler"]
            await event_handler(
                ProviderEvent(
                    "provider.prompt.accepted",
                    status="accepted",
                    native_session_id="claude-hanging-native",
                )
            )
            self.started.set()
            try:
                await self.release.wait()
            finally:
                self.provider_stopped = True
            return ProviderResult(
                provider="claude",
                native_session_id="claude-hanging-native",
                native_turn_id="turn",
                status="complete",
                usage={"total_tokens": 1},
            )

        async def interrupt(self) -> None:
            self.release.set()

        def process_identity(self) -> tuple[int, str]:
            return (123, "process-start")

    claude = HangingAdapter()
    rig = JourneyRig(tmp_path, claude=claude)
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    released_after_stop: list[bool] = []
    original_update = rig.store.update_process_lease

    def update_lease(lease_id: str, **values: Any) -> dict[str, Any]:
        if values.get("state") == "released":
            released_after_stop.append(claude.provider_stopped)
        return original_update(lease_id, **values)

    monkeypatch.setattr(rig.store, "update_process_lease", update_lease)
    receipt = rig.store.enqueue_command(
        rig.session.session_id,
        "message",
        {"text": "Wait until cancelled.", "provider": "claude"},
        new_uuid(),
    )
    rig.store.append_event(
        rig.session.session_id,
        "user.message",
        role="user",
        text="Wait until cancelled.",
        status="accepted",
        metadata={"command_id": receipt.command_id},
    )
    claimed = rig.store.claim_command(rig.session.session_id)
    assert claimed is not None
    task = asyncio.create_task(rig.worker._execute_message(claimed))
    try:
        await asyncio.wait_for(claude.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert claude.provider_stopped
        assert released_after_stop == [True]
        assert rig.store.active_process_leases() == []
    finally:
        if not task.done():
            task.cancel()
        rig.close()


@pytest.mark.asyncio
async def test_e2e_managed_turn_ref_stays_out_of_provider_prompt(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    turn_ref = {
        "step_id": "step-private-correlation",
        "agent_role": "implementer",
    }
    try:
        receipt = await rig.message(
            "Implement the bounded step.",
            provider="codex",
            turn_ref=turn_ref,
        )

        assert receipt.turn_ref == turn_ref
        codex = rig.adapters["codex"]
        assert "step-private-correlation" not in codex.prompts[0]
        provider_events = [
            item
            for item in rig.store.all_events(rig.session.session_id)
            if item.event_type == "agent.message"
        ]
        assert provider_events[0].metadata["turn_ref"] == turn_ref
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_worker_rejects_empty_messages_and_unclaimed_safety(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    try:
        empty = rig.store.enqueue_command(
            rig.session.session_id,
            "message",
            {"text": "   "},
            "empty-message",
        )
        claimed = rig.store.claim_command(rig.session.session_id)
        assert claimed is not None
        await rig.worker._execute_message(claimed)
        empty_result = rig.store.get_command(empty.command_id)
        assert empty_result.status == "failed"
        assert empty_result.result["code"] == "E_INPUT"

        with rig.store.transaction() as connection:
            connection.execute(
                "DELETE FROM session_safety WHERE session_id = ?",
                (rig.session.session_id,),
            )
        unclaimed = rig.store.enqueue_command(
            rig.session.session_id,
            "message",
            {"text": "continue"},
            "unclaimed-safety",
        )
        claimed = rig.store.claim_command(rig.session.session_id)
        assert claimed is not None
        await rig.worker._execute_message(claimed)
        unclaimed_result = rig.store.get_command(unclaimed.command_id)
        assert unclaimed_result.status == "failed"
        assert unclaimed_result.result["code"] == "E_SAFETY_PROFILE"
        current = rig.store.get_session(rig.session.session_id)
        assert current.lifecycle == "paused"
        assert current.attention == "needs-input"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_worker_controls_interrupt_steer_pause_resume_and_stop(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)

    async def control(
        command_type: str,
        payload: dict[str, object] | None = None,
    ) -> CommandReceipt:
        value = payload
        if value is None:
            value = {}
        receipt = rig.store.enqueue_command(
            rig.session.session_id,
            command_type,
            value,
            "control-" + command_type + "-" + new_uuid(),
        )
        await rig.worker._control(receipt)
        return rig.store.get_command(receipt.command_id)

    try:
        adapter = rig.adapters["codex"]
        rig.worker._active_adapter = adapter
        assert (await control("interrupt")).status == "complete"
        assert adapter.interruptions == 1

        inactive = rig.store.enqueue_command(
            rig.session.session_id,
            "message",
            {"text": "not active"},
            "inactive-interrupt-target",
        )
        invalid_interrupt = await control(
            "interrupt",
            {"target_command_id": inactive.command_id},
        )
        assert invalid_interrupt.status == "failed"
        assert invalid_interrupt.result["code"] == "E_CONTROL_TARGET"
        rig.store.resolve_command(inactive.command_id, "cancelled", {})

        missing_interrupt = await control(
            "interrupt",
            {"target_command_id": new_uuid()},
        )
        assert missing_interrupt.status == "failed"
        assert missing_interrupt.result["code"] == "E_CONTROL_TARGET"

        rig.worker._active_adapter = None
        steer = await control("steer", {"text": "change direction"})
        assert steer.status == "failed"
        assert steer.result["code"] == "E_NO_ACTIVE_TURN"

        rig.worker._active_adapter = adapter
        assert (await control("steer", {"text": "new plan"})).status == ("complete")
        assert adapter.steered == ["new plan"]

        assert (await control("pause")).status == "complete"
        assert rig.store.get_session(rig.session.session_id).lifecycle == "paused"
        assert (await control("resume")).status == "complete"
        assert rig.store.get_session(rig.session.session_id).lifecycle == "running"

        rig.worker._active_adapter = adapter
        assert (await control("stop")).status == "complete"
        assert adapter.interruptions == 2
        assert rig.worker._stopping
    finally:
        rig.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("implementation_status", ["queued", "dispatching"])
async def test_stop_releases_an_unaccepted_command_and_frees_the_route(
    tmp_path: Path,
    implementation_status: str,
) -> None:
    workspace = tmp_path / "workspace"
    _repository(workspace)
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    store = StateStore(harness_paths.database)
    blobs = BlobStore(harness_paths.blobs)
    created = session(workspace)
    store.create_session(created)
    store.set_session_safety(created.session_id, "unattended")
    adapters: dict[str, ProviderAdapter] = {"kimi": ScriptedAdapter("kimi")}
    scheduler = Scheduler(store, adapters)
    scheduler._usage_cache = {"kimi": _usage("kimi", 20.0)}
    scheduler._usage_at = asyncio.get_running_loop().time()
    worker = SessionWorker(
        store,
        blobs,
        scheduler,
        adapters,
        created.session_id,
    )
    store.register_worker(created.session_id, 123, worker.incarnation)
    checkpoint = checkpoint_workspace(
        created,
        blobs,
        sequence=store.last_sequence(created.session_id),
        provider="kimi",
        native_session_id="kimi-native",
        context_text="context",
    )
    store.add_checkpoint(checkpoint)
    try:
        implementation = store.enqueue_command(
            created.session_id,
            "message",
            {"text": "implement the change"},
            "kimi-implementation",
        )
        store.create_command_envelope(
            implementation.command_id,
            created.session_id,
            "unattended",
            {"max_seconds": 900},
        )
        store.update_command_envelope(
            implementation.command_id,
            provider="kimi",
        )
        claimed_implementation = store.claim_command(created.session_id)
        assert claimed_implementation is not None
        assert claimed_implementation.command_id == implementation.command_id
        attempt = ProviderAttempt(
            attempt_id=new_uuid(),
            session_id=created.session_id,
            provider="kimi",
            native_session_id="kimi-native",
            model="kimi-code/k3",
            effort="medium",
            auth_mode="subscription",
            status="running",
            started_at=utc_now(),
            ended_at="",
        )
        store.create_attempt(attempt)
        turn_id = store.start_turn(
            created.session_id,
            attempt.attempt_id,
            turn_ref=implementation.turn_ref,
        )
        store.record_dispatch_checkpoint(
            implementation.command_id,
            attempt.attempt_id,
            turn_id,
            checkpoint.checkpoint_id,
        )
        store.mark_provider_boundary(attempt.attempt_id)
        if implementation_status == "queued":
            store.requeue_command(implementation.command_id)
        assert store.active_unattended_provider_count("kimi") == 1
        follower = session(workspace)
        store.create_session(follower)
        store.set_session_safety(follower.session_id, "unattended")
        next_command = store.enqueue_command(
            follower.session_id,
            "message",
            {"text": "implement the next change"},
            "kimi-next-implementation",
        )
        store.create_command_envelope(
            next_command.command_id,
            follower.session_id,
            "unattended",
            {"max_seconds": 900},
        )

        async def route_next() -> Any:
            return await scheduler.choose(
                store.get_session(follower.session_id),
                workload="implementation",
                required_capabilities=frozenset(),
                execution_profile="unattended",
                enforce_concurrency=True,
                command_id=next_command.command_id,
            )

        with pytest.raises(ProviderUnavailableError):
            await route_next()

        stop = store.enqueue_command(
            created.session_id,
            "stop",
            {},
            "kimi-stop",
        )
        claimed = store.claim_command(
            created.session_id,
            frozenset({"stop"}),
        )
        assert claimed is not None
        await worker._control(claimed)

        released = store.get_command(implementation.command_id)
        assert released.status == "cancelled"
        assert released.result["code"] == "E_SESSION_STOPPED"
        assert released.result["accepted"] is False
        assert released.result["prior_status"] == implementation_status
        envelope = store.command_envelope(implementation.command_id)
        assert envelope["state"] == "released"
        assert envelope["guard_reason"] == "session-stopped"
        assert store.active_unattended_provider_count("kimi") == 0
        assert store.get_session(created.session_id).lifecycle == "stopped"
        stop_receipt = store.get_command(stop.command_id)
        assert stop_receipt.status == "complete"
        assert [
            entry["command_id"] for entry in stop_receipt.result["released_commands"]
        ] == [implementation.command_id]
        event_types = [
            event.event_type for event in store.events(created.session_id)
        ]
        assert "command.released" in event_types
        assert "session.stopped" in event_types
        assert [
            record.checkpoint_id for record in store.checkpoints(created.session_id)
        ] == [checkpoint.checkpoint_id]

        decision = await route_next()
        assert decision.provider == "kimi"
        assert store.get_command(next_command.command_id).status == "queued"

        with pytest.raises(ConflictError, match="admits only a resume command"):
            store.enqueue_command(
                created.session_id,
                "stop",
                {},
                "kimi-stop-again",
            )
        assert not store.queued_command_exists(
            created.session_id,
            STOPPED_SESSION_COMMANDS,
        )
        assert store.claim_command(created.session_id) is None
        assert store.get_session(created.session_id).lifecycle == "stopped"
        assert store.get_command(implementation.command_id).result == (
            released.result
        )

        resume = store.enqueue_command(
            created.session_id,
            "resume",
            {},
            "kimi-resume",
        )
        replacement = SessionWorker(
            store,
            blobs,
            scheduler,
            adapters,
            created.session_id,
        )
        store.register_worker(
            created.session_id,
            124,
            replacement.incarnation,
        )
        loop = asyncio.create_task(replacement._loop())
        await _await_command(store, resume.command_id)
        assert store.get_session(created.session_id).lifecycle == "running"

        pending = store.enqueue_command(
            created.session_id,
            "interrupt",
            {},
            "kimi-interrupt-after-resume",
        )
        assert (await _await_command(store, pending.command_id)).status == (
            "complete"
        )
        store.remove_worker(created.session_id, replacement.incarnation)
        await asyncio.wait_for(loop, timeout=10)
        assert store.claim_command(created.session_id) is None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_worker_run_surfaces_restart_recovery_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        SimpleNamespace(
            reconciliations=(object(),),
            requeued_command_ids=(),
            as_dict=lambda: {"reconciliations": ["one"]},
        ),
        SimpleNamespace(
            reconciliations=(),
            requeued_command_ids=("command-1",),
            as_dict=lambda: {"requeued_command_ids": ["command-1"]},
        ),
    ]

    class Manager:
        def __init__(self, store: object, blobs: object) -> None:
            del store
            del blobs

        async def recover_after_restart(
            self,
            session_id: str,
        ) -> object:
            del session_id
            return results.pop(0)

    monkeypatch.setattr(worker_module, "ReconciliationManager", Manager)

    for index in range(2):
        root = tmp_path / ("recovery-" + str(index))
        root.mkdir()
        rig = JourneyRig(root)

        async def no_loop() -> None:
            return

        monkeypatch.setattr(rig.worker, "_loop", no_loop)
        try:
            await rig.worker.run()
            events = rig.store.all_events(rig.session.session_id)
            assert events[-1].event_type == "worker.recovered"
            if index == 0:
                current = rig.store.get_session(rig.session.session_id)
                assert current.attention == "needs-reconciliation"
            assert rig.store.worker_registrations() == []
        finally:
            rig.close()


@pytest.mark.parametrize(
    ("termination", "error_type"),
    (
        ("identity-invalid", ""),
        ("termination-error", "RuntimeError"),
    ),
)
@pytest.mark.asyncio
async def test_worker_restart_retains_unresolved_process_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
    error_type: str,
) -> None:
    rig = JourneyRig(tmp_path)
    lease = rig.store.create_process_lease(
        rig.session.session_id,
        "codex",
        "unattended",
        "2099-01-01T00:00:00+00:00",
    )
    rig.store.update_process_lease(
        str(lease["lease_id"]),
        pid=123,
        pid_start="recorded-start",
        state="active",
    )

    async def unresolved_termination(
        unused_pid: int,
        unused_pid_start: str,
    ) -> str:
        del unused_pid
        del unused_pid_start
        if error_type:
            raise RuntimeError("process identity could not be established")
        return termination

    monkeypatch.setattr(
        worker_module,
        "terminate_recorded_process_group",
        unresolved_termination,
    )
    try:
        await rig.worker.run()
        session = rig.store.get_session(rig.session.session_id)
        assert session.lifecycle == "paused"
        assert session.attention == "needs-reconciliation"
        active = rig.store.active_process_leases()
        assert len(active) == 1
        assert active[0]["lease_id"] == lease["lease_id"]
        assert active[0]["state"] == "recovery-blocked"
        events = rig.store.all_events(rig.session.session_id)
        blocked = [
            event for event in events if event.event_type == "lease.recovery.blocked"
        ]
        assert len(blocked) == 1
        assert blocked[0].metadata["termination"] == termination
        assert blocked[0].metadata["error_type"] == error_type
        assert rig.store.worker_registrations() == []
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_recorded_dead_leader_with_live_group_blocks_recovery() -> None:
    leader = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(60)']); "
            "sys.stdin.buffer.read(1); "
            "sys.exit(0)"
        ),
        stdin=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    leader_pid = leader.pid
    identity = process_control_module.process_group_identity(leader_pid)
    assert leader.stdin is not None
    leader.stdin.write(b"x")
    await leader.stdin.drain()
    leader.stdin.close()
    await leader.wait()
    try:
        termination = await process_control_module.terminate_recorded_process_group(
            leader_pid,
            identity.pid_start,
            terminate_timeout=0.1,
            kill_timeout=0.1,
        )
        assert termination == "identity-invalid"
        assert process_control_module._group_exists(leader_pid) is True
    finally:
        os.killpg(leader_pid, signal.SIGKILL)
        for unused in range(100):
            if not process_control_module._group_exists(leader_pid):
                break
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_worker_restart_releases_pid_reuse_without_signaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    lease = rig.store.create_process_lease(
        rig.session.session_id,
        "claude",
        "unattended",
        "2099-01-01T00:00:00+00:00",
    )
    rig.store.update_process_lease(
        str(lease["lease_id"]),
        pid=456,
        pid_start="prior-process-start",
        state="active",
    )
    loop_ran = False

    async def reused_pid(
        unused_pid: int,
        unused_pid_start: str,
    ) -> str:
        del unused_pid
        del unused_pid_start
        return "identity-changed"

    async def no_loop() -> None:
        nonlocal loop_ran
        loop_ran = True

    monkeypatch.setattr(
        worker_module,
        "terminate_recorded_process_group",
        reused_pid,
    )
    monkeypatch.setattr(rig.worker, "_loop", no_loop)
    try:
        await rig.worker.run()
        assert loop_ran is True
        assert rig.store.active_process_leases() == []
        released = rig.store.process_lease(str(lease["lease_id"]))
        assert released["state"] == "released"
        events = rig.store.all_events(rig.session.session_id)
        recovered = [event for event in events if event.event_type == "lease.recovered"]
        assert len(recovered) == 1
        assert recovered[0].metadata["termination"] == "identity-changed"
        assert rig.store.worker_registrations() == []
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_worker_loop_handles_idle_control_barriers_and_unknown_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    try:
        with monkeypatch.context() as context:
            context.setattr(
                rig.store,
                "heartbeat_worker",
                lambda unused_session, unused_incarnation: False,
            )
            await rig.worker._loop()

        rig.store.update_session(
            rig.session.session_id,
            lifecycle="stopped",
        )
        with monkeypatch.context() as context:
            context.setattr(
                rig.store,
                "heartbeat_worker",
                lambda unused_session, unused_incarnation: True,
            )
            await rig.worker._loop()
        assert rig.store.worker_registered(rig.session.session_id) is False
        rig.store.register_worker(
            rig.session.session_id,
            123,
            rig.worker.incarnation,
        )

        rig.store.update_session(
            rig.session.session_id,
            lifecycle="paused",
            attention="idle",
        )
        rig.worker._stopping = False

        async def stop_after_sleep(unused: float) -> None:
            del unused
            rig.worker._stopping = True

        with monkeypatch.context() as context:
            context.setattr(
                rig.store,
                "heartbeat_worker",
                lambda unused_session, unused_incarnation: True,
            )
            context.setattr(
                worker_module.asyncio,
                "sleep",
                stop_after_sleep,
            )
            await rig.worker._loop()

        rig.store.update_session(
            rig.session.session_id,
            lifecycle="starting",
            attention="needs-input",
        )
        rig.worker._stopping = False
        with monkeypatch.context() as context:
            context.setattr(
                rig.store,
                "heartbeat_worker",
                lambda unused_session, unused_incarnation: True,
            )
            context.setattr(
                rig.store,
                "pending_reconciliations",
                lambda unused_session: [],
            )
            context.setattr(
                worker_module.asyncio,
                "sleep",
                stop_after_sleep,
            )
            await rig.worker._loop()
        normalized = rig.store.get_session(rig.session.session_id)
        assert normalized.lifecycle == "running"
        assert normalized.attention == "idle"

        rig.store.update_session(
            rig.session.session_id,
            lifecycle="running",
            attention="idle",
        )
        rig.worker._stopping = False
        with monkeypatch.context() as context:
            context.setattr(
                rig.store,
                "heartbeat_worker",
                lambda unused_session, unused_incarnation: True,
            )
            context.setattr(
                rig.store,
                "pending_reconciliations",
                lambda unused_session: [object()],
            )
            context.setattr(
                worker_module.asyncio,
                "sleep",
                stop_after_sleep,
            )
            await rig.worker._loop()
        assert (
            rig.store.get_session(rig.session.session_id).attention
            == "needs-reconciliation"
        )

        rig.store.update_session(
            rig.session.session_id,
            attention="idle",
        )
        unknown = rig.store.enqueue_command(
            rig.session.session_id,
            "unknown",
            {},
            "unknown-work",
        )
        rig.worker._stopping = False
        with monkeypatch.context() as context:
            context.setattr(
                rig.store,
                "heartbeat_worker",
                lambda unused_session, unused_incarnation: True,
            )
            context.setattr(
                rig.store,
                "pending_reconciliations",
                lambda unused_session: [],
            )
            context.setattr(
                worker_module.asyncio,
                "sleep",
                stop_after_sleep,
            )
            await rig.worker._loop()
        failed = rig.store.get_command(unknown.command_id)
        assert failed.status == "failed"
        assert failed.result["code"] == "E_COMMAND"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_worker_approval_defaults_and_user_event_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    try:
        monkeypatch.setattr(
            rig.store,
            "approval_decision",
            lambda unused_approval: {"decision": "accept"},
        )
        decision = await rig.worker._approval(
            "",
            "tool.execute",
            {
                "id": "request-1",
                "prompt": "Run the tool?",
                "choices": "invalid",
            },
        )
        assert decision == {"decision": "accept"}
        assert rig.store.get_session(rig.session.session_id).attention == "working"
        monkeypatch.setattr(
            rig.store,
            "turn_ref",
            lambda unused_turn: {},
        )
        await rig.worker._provider_event(
            "",
            ProviderEvent(
                "user.message",
                text="steered",
                raw={"provider": "scripted"},
            ),
        )
        projected = rig.store.all_events(rig.session.session_id)[-1]
        assert projected.role == "user"
        assert projected.blob_digest == ""
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_worker_guard_interrupt_cleanup_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)

    class FailingInterrupt:
        async def interrupt(self) -> None:
            raise RuntimeError("already stopped")

    class Interrupt:
        async def interrupt(self) -> None:
            return

    async def complete() -> None:
        return

    async def fail() -> None:
        raise RuntimeError("provider failed")

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    try:
        completed = asyncio.create_task(complete())
        await asyncio.sleep(0)
        await rig.worker._interrupt_guarded_turn(
            FailingInterrupt(),  # type: ignore[arg-type]
            completed,
        )

        failed = asyncio.create_task(fail())
        await asyncio.sleep(0)
        await rig.worker._interrupt_guarded_turn(
            Interrupt(),  # type: ignore[arg-type]
            failed,
        )

        waiting = asyncio.create_task(wait_forever())

        async def timeout(
            awaitable: object,
            *,
            timeout: float,
        ) -> None:
            del awaitable
            del timeout
            raise asyncio.TimeoutError

        with monkeypatch.context() as context:
            context.setattr(
                worker_module.asyncio,
                "wait_for",
                timeout,
            )
            await rig.worker._interrupt_guarded_turn(
                Interrupt(),  # type: ignore[arg-type]
                waiting,
            )
        assert waiting.cancelled()
        assert worker_module._optional_number(True) is None
        assert worker_module._optional_number(7) == 7.0
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_worker_message_publish_and_failover_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    try:
        queued = rig.store.enqueue_command(
            rig.session.session_id,
            "message",
            {"text": "queued"},
            new_uuid(),
        )

        async def stop_after_message(command: CommandReceipt) -> None:
            assert command.command_id == queued.command_id
            rig.worker._stopping = True

        monkeypatch.setattr(rig.worker, "_message", stop_after_message)
        monkeypatch.setattr(
            rig.store,
            "heartbeat_worker",
            lambda unused_session, unused_incarnation: True,
        )
        await rig.worker._loop()
        assert rig.worker._stopping

        rig.worker._stopping = False
        rig.store.enqueue_command(
            rig.session.session_id,
            "message",
            {"text": "publish"},
            new_uuid(),
        )
        claimed = rig.store.claim_command(rig.session.session_id)
        assert claimed is not None
        published: list[str] = []

        async def execute(unused: CommandReceipt) -> None:
            return

        monkeypatch.setattr(rig.worker, "_execute_message", execute)
        monkeypatch.setattr(
            worker_module,
            "publish_session",
            lambda unused_paths, unused_store, session_id: published.append(session_id),
        )
        rig.worker.paths = paths(tmp_path / "publish-state")
        await SessionWorker._message(rig.worker, claimed)
        assert published == [rig.session.session_id]
        failover_command_id = claimed.command_id
        rig.store.create_command_envelope(
            failover_command_id,
            rig.session.session_id,
            "interactive",
            limits_for("interactive", "implementation").as_dict(),
        )

        guard = TurnGuard(limits_for("interactive", "implementation"))

        async def safety_failure(*unused: object, **values: object) -> object:
            del unused, values
            raise SafetyGuardError(
                "stagnation",
                "codex",
                recoverable=True,
            )

        monkeypatch.setattr(rig.worker, "_execute_attempt", safety_failure)
        with pytest.raises(SafetyGuardError):
            await rig.worker._execute_with_failover(
                failover_command_id,
                {"provider": "codex", "effort": "low"},
                "message",
                guard,
            )

        async def unavailable(*unused: object, **values: object) -> object:
            del unused, values
            raise ProviderUnavailableError("codex")

        monkeypatch.setattr(rig.worker, "_execute_attempt", unavailable)
        with pytest.raises(ProviderUnavailableError):
            await rig.worker._execute_with_failover(
                failover_command_id,
                {"provider": "codex"},
                "message",
                TurnGuard(limits_for("interactive", "implementation")),
            )

        attempted_exclusions: list[frozenset[str]] = []

        async def closed_codex_transport(
            *unused: object,
            **values: object,
        ) -> dict[str, str]:
            del values
            excluded = unused[3]
            attempted_exclusions.append(excluded)
            if not excluded:
                raise ProviderUnavailableError(
                    "codex",
                    detail="Codex app-server connection closed",
                )
            return {"provider": "claude"}

        monkeypatch.setattr(
            rig.worker,
            "_execute_attempt",
            closed_codex_transport,
        )
        failover_result = await rig.worker._execute_with_failover(
            failover_command_id,
            {},
            "message",
            TurnGuard(limits_for("interactive", "implementation")),
        )
        assert failover_result == {"provider": "claude"}
        assert attempted_exclusions == [frozenset(), frozenset({"codex"})]

        empty_guard = TurnGuard(
            replace(
                limits_for("interactive", "implementation"),
                max_attempts=0,
            )
        )
        with pytest.raises(ProviderUnavailableError, match="all providers"):
            await rig.worker._execute_with_failover(
                failover_command_id,
                {},
                "message",
                empty_guard,
            )
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_worker_attempt_guard_event_lease_and_failure_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FileAdapter(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            event_handler = values["event_handler"]
            await event_handler(ProviderEvent("file.change.completed"))
            return await super().run_turn(**values)

    file_adapter = FileAdapter("codex")
    file_root = tmp_path / "file"
    file_root.mkdir()
    rig = JourneyRig(file_root, codex=file_adapter)
    try:
        rig.prime_capacity(claude=95, codex=10)
        result = await rig.message("change a file", provider="codex")
        assert result.status == "complete"

        limited = SimpleNamespace(
            limits=limits_for("interactive", "implementation"),
            begin_attempt=lambda unused, *, charge_reported_cost=True: (
                "context-tokens"
            ),
        )
        with pytest.raises(SafetyGuardError, match="context-tokens"):
            await rig.worker._execute_attempt(
                result.command_id,
                {"provider": "codex"},
                "message",
                frozenset(),
                limited,  # type: ignore[arg-type]
                0,
                enforce_concurrency=True,
            )
    finally:
        rig.close()

    class LeaseAdapter(ScriptedAdapter):
        def process_identity(self) -> tuple[int, str]:
            return 123, "start"

    lease_adapter = LeaseAdapter("codex", turn_delay=0.2)
    lease_root = tmp_path / "lease"
    lease_root.mkdir()
    lease_rig = JourneyRig(lease_root, codex=lease_adapter)
    try:
        lease_rig.store.set_session_safety(
            lease_rig.session.session_id,
            "unattended",
        )
        lease_rig.prime_capacity(claude=95, codex=10)
        tick = [0.0]

        def monotonic() -> float:
            tick[0] += 20.0
            return tick[0]

        monkeypatch.setattr(
            worker_module,
            "time",
            SimpleNamespace(monotonic=monotonic),
        )
        result = await lease_rig.message("leased turn", provider="codex")
        assert result.status == "complete"
        event_types = {
            event.event_type
            for event in lease_rig.store.all_events(lease_rig.session.session_id)
        }
        assert "lease.attached" in event_types
        assert "lease.released" in event_types
    finally:
        lease_rig.close()

    class ResultUsageAdapter(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            del values
            return ProviderResult(
                provider="codex",
                native_session_id="native",
                native_turn_id="turn",
                status="complete",
                usage={"output_tokens": 1_000_000},
            )

    usage_root = tmp_path / "result-usage"
    usage_root.mkdir()
    usage_rig = JourneyRig(
        usage_root,
        codex=ResultUsageAdapter("codex"),
    )
    try:
        usage_rig.prime_capacity(claude=95, codex=10)
        result = await usage_rig.message("large result", provider="codex")
        assert result.status == "failed"
    finally:
        usage_rig.close()

    class RuntimeAdapter(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            del values
            raise RuntimeError("adapter failure")

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    runtime_rig = JourneyRig(
        runtime_root,
        codex=RuntimeAdapter("codex"),
    )
    try:
        runtime_rig.prime_capacity(claude=95, codex=10)
        with pytest.raises(RuntimeError, match="adapter failure"):
            await runtime_rig.message("fail", provider="codex")
    finally:
        runtime_rig.close()

    gate_root = tmp_path / "child-gate"
    gate_root.mkdir()
    gate_rig = JourneyRig(gate_root, codex=ScriptedAdapter("codex"))
    try:
        gate_rig.prime_capacity(claude=95, codex=10)

        def fail_child_gate(**unused: object) -> None:
            del unused
            raise RuntimeError("child gate construction failed")

        monkeypatch.setattr(
            worker_module,
            "ChildLaunchGate",
            fail_child_gate,
        )
        with pytest.raises(RuntimeError, match="child gate construction failed"):
            await gate_rig.message("fail before provider start", provider="codex")
        tables = gate_rig.store.portable_session(gate_rig.session.session_id)[
            "tables"
        ]
        command_id = str(tables["commands"][0]["command_id"])
        dispatch = next(
            item
            for item in tables["command_dispatches"]
            if item["command_id"] == command_id
        )
        attempt = next(
            item
            for item in tables["provider_attempts"]
            if item["attempt_id"] == dispatch["attempt_id"]
        )
        turn = next(
            item
            for item in tables["turns"]
            if item["turn_id"] == dispatch["turn_id"]
        )
        assert dispatch["state"] == "failed"
        assert attempt["status"] == "failed"
        assert turn["status"] == "failed"
        assert gate_rig.worker._active_command_id == ""
        assert gate_rig.worker._active_turn_id == ""
    finally:
        gate_rig.close()


@pytest.mark.asyncio
async def test_stagnation_after_acceptance_requires_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StagnationGuard:
        def __init__(
            self,
            limits: object,
            consumption: SafetyConsumption | None = None,
        ) -> None:
            self.limits = limits
            if consumption is None:
                consumption = SafetyConsumption()
            self.consumption = consumption

        def begin_attempt(
            self,
            context_tokens: int,
            *,
            charge_reported_cost: bool = True,
        ) -> str:
            del charge_reported_cost
            self.consumption.attempts += 1
            self.consumption.context_tokens += context_tokens
            return ""

        def establish_material_state(self, digest: str) -> None:
            assert digest

        def note_child_admissions(self, consumed: int) -> None:
            self.consumption.child_agents = consumed

        def observe(self, event: ProviderEvent) -> str:
            del event
            return ""

        def violation(self) -> str:
            return "stagnation"

        def note_provider_terminal(self) -> str:
            return self.violation()

        def live_violation(self) -> str:
            return self.violation()

        def terminal_violation(self) -> str:
            return self.violation()

        def warning_due(self) -> bool:
            return False

        def snapshot(self) -> dict[str, object]:
            return {"consumption": self.consumption.as_dict()}

        def recover(self) -> None:
            return

    monkeypatch.setattr(
        worker_module,
        "TurnGuard",
        StagnationGuard,
    )
    rig = JourneyRig(
        tmp_path,
        codex=ScriptedAdapter("codex"),
    )
    rig.prime_capacity(claude=80, codex=10)
    try:
        receipt = await rig.message("Recover a stagnant provider turn.")
        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_SAFETY_GUARD"
        assert receipt.result["reconciliation_id"]
        incidents = rig.store.guard_incidents(rig.session.session_id)
        assert [item["action"] for item in incidents] == ["pause"]
        recoveries = [
            item
            for item in rig.store.all_events(rig.session.session_id)
            if item.event_type == "recovery.started"
        ]
        assert recoveries == []
        reconciliations = rig.store.pending_reconciliations(
            rig.session.session_id
        )
        assert len(reconciliations) == 1
    finally:
        rig.close()

    class PreAcceptanceAdapter(ScriptedAdapter):
        def __init__(self, provider: str) -> None:
            super().__init__(provider)
            self.release: asyncio.Event | None = None

        async def run_turn(self, **values: Any) -> ProviderResult:
            del values
            self.release = asyncio.Event()
            await self.release.wait()
            return ProviderResult(
                provider=self.provider_id,
                native_session_id="",
                native_turn_id="",
                status="cancelled",
            )

        async def interrupt(self) -> None:
            self.interruptions += 1
            assert self.release is not None
            self.release.set()

    pre_acceptance_root = tmp_path / "pre-acceptance"
    pre_acceptance_root.mkdir()
    pre_acceptance_rig = JourneyRig(
        pre_acceptance_root,
        claude=PreAcceptanceAdapter("claude"),
        codex=PreAcceptanceAdapter("codex"),
    )
    pre_acceptance_rig.prime_capacity(claude=80, codex=10)
    try:
        receipt = await pre_acceptance_rig.message(
            "Recover only before provider acceptance."
        )
        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_SAFETY_GUARD", receipt.result
        incidents = pre_acceptance_rig.store.guard_incidents(
            pre_acceptance_rig.session.session_id
        )
        assert [item["action"] for item in incidents] == [
            "downgrade",
            "failover",
            "pause",
        ]
        recoveries = [
            item
            for item in pre_acceptance_rig.store.all_events(
                pre_acceptance_rig.session.session_id
            )
            if item.event_type == "recovery.started"
        ]
        assert [item.metadata["stage"] for item in recoveries] == [1, 2]
        attempts = pre_acceptance_rig.store.attempts(
            pre_acceptance_rig.session.session_id
        )
        assert [item.provider for item in attempts] == [
            "codex",
            "codex",
            "claude",
        ]
        assert [item.effort for item in attempts] == [
            "high",
            "medium",
            "low",
        ]
        assert not pre_acceptance_rig.store.pending_reconciliations(
            pre_acceptance_rig.session.session_id
        )
    finally:
        pre_acceptance_rig.close()


def test_goal_limit_and_workspace_digest_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    goal = create_goal(
        rig.session.session_id,
        "Exercise invalid durable budget values.",
    )
    invalid_goal = replace(
        goal,
        budgets={
            "tool_calls": True,
            "attempts": True,
            "child_agents": True,
        },
    )
    monkeypatch.setattr(
        rig.store,
        "goal_for_session",
        lambda unused_session: invalid_goal,
    )
    limits = limits_for("interactive", "implementation")
    assert (
        rig.worker._goal_limited_limits(
            limits,
            metered_budget=None,
        )
        == limits
    )
    rig.close()

    workspace = tmp_path / "digest"
    workspace.mkdir()
    (workspace / "directory").mkdir()
    (workspace / "link").symlink_to("directory", target_is_directory=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    assert (
        worker_module._workspace_artifact_digest(
            workspace,
            ("directory", "link", "../outside.txt"),
        )
        == ""
    )
    (workspace / "a-large.txt").write_bytes(b"a" * 200_000)
    (workspace / "z-after.txt").write_text("after", encoding="utf-8")
    assert worker_module._workspace_artifact_digest(
        workspace,
        ("*",),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_status", "terminal_status"),
    [
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("unknown", "failed"),
    ],
)
async def test_noncomplete_provider_result_fails_closed(
    tmp_path: Path,
    provider_status: str,
    terminal_status: str,
) -> None:
    class NoncompleteAdapter(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            del values
            return ProviderResult(
                provider="codex",
                native_session_id="native-noncomplete",
                native_turn_id="turn-noncomplete",
                status=provider_status,
                usage={"total_tokens": 1},
            )

    root = tmp_path / provider_status
    root.mkdir()
    rig = JourneyRig(root, codex=NoncompleteAdapter("codex"))
    rig.prime_capacity(claude=95, codex=10)
    try:
        receipt = await rig.message("Reject a non-complete result.")
        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_PROVIDER_RESULT"
        tables = rig.store.portable_session(rig.session.session_id)["tables"]
        assert tables["provider_attempts"][-1]["status"] == terminal_status
        assert tables["turns"][-1]["status"] == terminal_status
        assert tables["command_dispatches"][-1]["state"] == "failed"
        assert len(tables["checkpoints"]) == 1
        assert (
            tables["command_dispatches"][-1]["checkpoint_id"]
            == (tables["checkpoints"][0]["checkpoint_id"])
        )
        assert not [
            event
            for event in tables["events"]
            if event["event_type"] == "checkpoint.created"
            and event["status"] == "complete"
        ]
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_worker_approval_timeout_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    try:
        monkeypatch.setattr(worker_module, "APPROVAL_POLL_LIMIT", 0)
        assert await rig.worker._approval("", "tool", {}) == {"decision": "decline"}
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_idle_worker_does_not_reorder_session_history(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    before = rig.store.get_session(rig.session.session_id).updated_at
    worker_task = asyncio.create_task(rig.worker.run())
    try:
        await asyncio.sleep(0.35)
        first_idle = rig.store.get_session(rig.session.session_id).updated_at
        await asyncio.sleep(0.35)
        second_idle = rig.store.get_session(rig.session.session_id).updated_at

        assert first_idle == before
        assert second_idle == first_idle

        stopped = rig.store.enqueue_command(
            rig.session.session_id,
            "stop",
            {},
            new_uuid(),
        )
        await asyncio.wait_for(worker_task, timeout=2)
        assert rig.store.get_command(stopped.command_id).status == "complete"
    finally:
        if not worker_task.done():
            worker_task.cancel()
        rig.close()


@pytest.mark.asyncio
async def test_e2e_capacity_exhaustion_fails_over_once(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(
        tmp_path,
        claude=ScriptedAdapter("claude", fail_turns=1),
    )
    rig.prime_capacity()
    try:
        receipt = await rig.message("Build the feature.")

        attempts = rig.store.attempts(rig.session.session_id)
        events = rig.store.all_events(rig.session.session_id)
        assert receipt.status == "complete", receipt.result
        assert [item.provider for item in attempts] == ["claude", "codex"]
        assert [item.status for item in attempts] == [
            "exhausted",
            "complete",
        ]
        assert any(item.event_type == "routing.failover" for item in events)
        current = rig.store.get_session(rig.session.session_id)
        assert current.active_provider == "codex"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_provider_switch_prepends_handoff_envelope(
    tmp_path: Path,
) -> None:
    codex = ScriptedAdapter("codex", fail_turns=1)
    claude = ScriptedAdapter("claude")
    rig = JourneyRig(tmp_path, claude=claude, codex=codex)
    rig.prime_capacity(claude=60.0, codex=10.0)
    try:
        receipt = await rig.message("Build the feature.")

        assert receipt.status == "complete", receipt.result
        attempts = rig.store.attempts(rig.session.session_id)
        assert [item.provider for item in attempts] == ["codex", "claude"]
        assert claude.native_inputs == [""]
        prompt = claude.prompts[0]
        assert prompt.startswith("# Session handoff")
        assert "session-handoff/v1" in prompt
        assert "Harness-generated context" in prompt
        assert "- Source provider: `codex`" in prompt
        assert "- Target provider: `claude`" in prompt
        assert "Build the feature." in prompt
        handoff_events = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "session.handoff"
        ]
        assert len(handoff_events) == 1
        metadata = handoff_events[0].metadata
        assert metadata["schema"] == "session-handoff/v1"
        assert metadata["origin"] == "provider-switch"
        assert metadata["source_provider"] == "codex"
        assert metadata["target_provider"] == "claude"
        blob_text = rig.blobs.get_text(metadata["blob_digest"])
        assert prompt.startswith(blob_text + "\n\n")
        assert (len(blob_text) + 3) // 4 == metadata["rendered_tokens"]
        assert metadata["rendered_tokens"] <= metadata["token_budget"]
        # The ScriptedAdapter claude model reports a 200_000 token window.
        assert metadata["token_budget"] <= 200_000
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_provider_switch_compacts_to_the_command_context_limit(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude")
    codex = ScriptedAdapter("codex")
    rig = JourneyRig(tmp_path, claude=claude, codex=codex)
    rig.prime_capacity()
    try:
        first = await rig.message("Implement the bounded change.", provider="codex")
        assert first.status == "complete", first.result
        rig.store.append_event(
            rig.session.session_id,
            "agent.message",
            role="assistant",
            text="implementation evidence " * 2_000,
            status="complete",
        )

        receipt = await rig.message(
            "Review the bounded change.",
            provider="claude",
            safety_limits={
                "max_context_tokens": 2_000,
                "max_output_tokens": 512,
                "max_total_tokens": 3_000,
            },
        )

        assert receipt.status == "complete", receipt.result
        prompt = claude.prompts[-1]
        assert prompt.startswith("# Session handoff")
        assert (len(prompt) + 3) // 4 <= 2_000
        events = rig.store.all_events(rig.session.session_id)
        assert not [event for event in events if event.event_type == "guard.tripped"]
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_claude_to_kimi_switch_carries_handoff_envelope(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude", fail_turns=1)
    kimi = ScriptedAdapter("kimi")
    rig = JourneyRig(tmp_path, claude=claude)
    del rig.adapters["codex"]
    rig.adapters["kimi"] = kimi
    rig.scheduler._usage_cache = {
        "claude": _usage("claude", 20.0),
        "kimi": _usage("kimi", 20.0),
    }
    rig.scheduler._usage_at = asyncio.get_running_loop().time()
    try:
        receipt = await rig.message("Port the parser.")

        assert receipt.status == "complete", receipt.result
        attempts = rig.store.attempts(rig.session.session_id)
        assert [item.provider for item in attempts] == ["claude", "kimi"]
        assert kimi.native_inputs == [""]
        prompt = kimi.prompts[0]
        assert prompt.startswith("# Session handoff")
        assert "- Source provider: `claude`" in prompt
        assert "- Target provider: `kimi`" in prompt
        assert "Port the parser." in prompt
        metadata = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "session.handoff"
        ][0].metadata
        assert metadata["origin"] == "provider-switch"
        assert metadata["target_provider"] == "kimi"
        assert metadata["rendered_tokens"] <= metadata["token_budget"]
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_same_provider_dispatch_carries_no_handoff_envelope(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude")
    rig = JourneyRig(tmp_path, claude=claude)
    rig.prime_capacity()
    try:
        first = await rig.message("Build the feature.", provider="claude")
        second = await rig.message("Refine the feature.", provider="claude")

        assert first.status == "complete", first.result
        assert second.status == "complete", second.result
        assert claude.native_inputs == ["", "claude-native-session"]
        assert not claude.prompts[0].startswith("# Session handoff")
        assert "session-handoff/v1" not in claude.prompts[0]
        assert claude.prompts[1] == "Refine the feature."
        events = rig.store.all_events(rig.session.session_id)
        assert not [
            event for event in events if event.event_type == "session.handoff"
        ]
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_handoff_envelope_stays_outside_guard_digests(
    tmp_path: Path,
) -> None:
    codex = ScriptedAdapter("codex", fail_turns=1)
    rig = JourneyRig(tmp_path, codex=codex)
    rig.prime_capacity(claude=60.0, codex=10.0)
    try:
        receipt = await rig.message("Build the feature.")

        assert receipt.status == "complete", receipt.result
        events = rig.store.all_events(rig.session.session_id)
        # The failover re-dispatch of the same command, envelope
        # included, did not trip the repeated-dispatch guard.
        assert not [event for event in events if event.event_type == "guard.tripped"]
        fingerprints = [
            event for event in events if event.event_type == "dispatch.fingerprint"
        ]
        assert len(fingerprints) == 1
        # The envelope is harness-generated context: the guard's
        # instruction digest still covers only the operator text.
        assert fingerprints[0].metadata["instruction_digest"] == (
            worker_module._text_digest("Build the feature.")
        )
        handoffs = [
            event for event in events if event.event_type == "session.handoff"
        ]
        assert len(handoffs) == 1
    finally:
        rig.close()


class _ForkWorkers:
    def __init__(self) -> None:
        self.ensured: list[str] = []

    def ensure(self, session_id: str, *, force: bool = False) -> None:
        del force
        self.ensured.append(session_id)

    def stop_all(self) -> None:
        return


@pytest.mark.asyncio
async def test_e2e_fork_to_provider_seeds_and_delivers_handoff(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _repository(workspace)
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    service = HarnessService(harness_paths, worker_manager=_ForkWorkers())
    try:
        source = service.create_session(
            {
                "workspace": str(workspace),
                "name": "implementation",
                "goal": "Ship the reviewed change.",
            }
        )
        attempt = ProviderAttempt(
            attempt_id=new_uuid(),
            session_id=source.session_id,
            provider="codex",
            native_session_id="codex-native-session",
            model="codex-default",
            effort="high",
            auth_mode="subscription",
            status="complete",
            started_at=utc_now(),
            ended_at=utc_now(),
        )
        service.store.create_attempt(attempt)
        turn_id = service.store.start_turn(
            source.session_id,
            attempt.attempt_id,
        )
        service.store.append_event(
            source.session_id,
            "user.message",
            role="user",
            text="implement the parser port",
            status="complete",
            turn_id=turn_id,
        )
        service.store.append_event(
            source.session_id,
            "agent.message",
            role="assistant",
            text="parser ported with tests",
            status="complete",
            turn_id=turn_id,
        )
        service.store.finish_turn(turn_id, "complete")
        service.store.update_session(
            source.session_id,
            active_provider="codex",
        )
        with pytest.raises(ValueError, match="target provider"):
            service.fork_session(
                source.session_id,
                {"target_provider": "unregistered"},
            )

        forked = service.fork_session(
            source.session_id,
            {"name": "review fork", "target_provider": "claude"},
        )

        assert forked.name == "review fork"
        forked_events = service.store.all_events(forked.session_id)
        fork_event = [
            event for event in forked_events if event.event_type == "session.forked"
        ][0]
        assert fork_event.metadata["source_session_id"] == source.session_id
        assert fork_event.metadata["target_provider"] == "claude"
        lineage = service.store.fork_lineage(forked.session_id)
        assert lineage["source_session_id"] == source.session_id
        assert lineage["source_context_digest"]
        seed = [
            event for event in forked_events if event.event_type == "session.handoff"
        ][0]
        assert seed.metadata["origin"] == "fork-seed"
        assert seed.metadata["target_provider"] == "claude"
        seeded_text = service.blobs.get_text(seed.metadata["blob_digest"])
        assert seeded_text.startswith("# Session handoff")
        assert "session-handoff/v1" in seeded_text
        assert "- Source provider: `codex`" in seeded_text
        assert "- Target provider: `claude`" in seeded_text
        assert "implement the parser port" in seeded_text
        assert "parser ported with tests" in seeded_text
        oversized_seed_text = seeded_text + "\n" + "review evidence " * 5_000
        oversized_seed_blob = service.blobs.put_text(oversized_seed_text)
        oversized_seed = dict(seed.metadata)
        oversized_seed["blob_digest"] = oversized_seed_blob
        oversized_seed["handoff_digest"] = worker_module._text_digest(
            oversized_seed_text
        )
        service.store.append_event(
            forked.session_id,
            "session.handoff",
            status="complete",
            metadata=oversized_seed,
        )

        claude = ScriptedAdapter("claude")
        adapters = {"claude": claude}
        scheduler = Scheduler(service.store, adapters)
        scheduler._usage_cache = {"claude": _usage("claude", 20.0)}
        scheduler._usage_at = asyncio.get_running_loop().time()
        worker = SessionWorker(
            service.store,
            service.blobs,
            scheduler,
            adapters,
            forked.session_id,
        )
        service.store.register_worker(
            forked.session_id,
            123,
            worker.incarnation,
        )
        service.store.set_session_safety(forked.session_id, "interactive")
        receipt = service.store.enqueue_command(
            forked.session_id,
            "message",
            {
                "text": "Review the carried state.",
                "safety_limits": {
                    "max_context_tokens": 2_000,
                    "max_output_tokens": 512,
                    "max_total_tokens": 3_000,
                },
            },
            new_uuid(),
        )
        service.store.append_event(
            forked.session_id,
            "user.message",
            role="user",
            text="Review the carried state.",
            status="accepted",
            metadata={"command_id": receipt.command_id},
        )
        claimed = service.store.claim_command(forked.session_id)
        assert claimed is not None
        await worker._message(claimed)

        prompt = claude.prompts[0]
        assert claude.native_inputs == [""]
        assert prompt.startswith("# Session handoff")
        assert "Review the carried state." in prompt
        assert (len(prompt) + 3) // 4 <= 2_000
        deliveries = [
            event
            for event in service.store.all_events(forked.session_id)
            if event.event_type == "session.handoff"
        ]
        assert [event.metadata["origin"] for event in deliveries] == [
            "fork-seed",
            "fork-seed",
            "fork-seed",
        ]
        assert deliveries[2].metadata["source_blob_digest"] == (
            oversized_seed_blob
        )
        assert deliveries[2].metadata["handoff_digest"] != (
            oversized_seed["handoff_digest"]
        )
    finally:
        service.close()


@pytest.mark.asyncio
async def test_e2e_command_limit_tightening_is_immutable_and_prompt_free(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    requested = {"max_attempts": 2, "max_seconds": 300}
    try:
        receipt = await rig.message(
            "Run one bounded managed stage.",
            safety_limits=requested,
        )

        assert receipt.status == "complete", receipt.result
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["requested_limits"] == requested
        assert envelope["requested_limits_digest"] == normalized_digest(requested)
        assert envelope["limits"]["max_attempts"] == 2
        assert envelope["limits"]["max_seconds"] == 300
        proof = proof_snapshot(rig.store, rig.session.session_id)
        proof_envelope = next(
            item
            for item in proof["safety"]["envelopes"]
            if item["command_id"] == receipt.command_id
        )
        assert proof_envelope["requested_limits_digest"] == normalized_digest(requested)
        assert proof_envelope["limits"]["max_attempts"] == 2
        assert proof_envelope["limits"]["max_seconds"] == 300
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_command_limit_widening_fails_before_provider_admission(
    tmp_path: Path,
) -> None:
    codex = ScriptedAdapter("codex")
    rig = JourneyRig(tmp_path, codex=codex)
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity(claude=80, codex=1)
    try:
        receipt = await rig.message(
            "Reject a widened managed stage.",
            provider="codex",
            safety_limits={"max_attempts": 4},
        )

        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_SAFETY_BUDGET"
        assert not codex.prompts
        assert not rig.store.attempts(rig.session.session_id)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_transport_loss_after_dispatch_requires_reconciliation(
    tmp_path: Path,
) -> None:
    class ClosedCodexAdapter(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            workspace = values["workspace"]
            event_handler = values["event_handler"]
            await event_handler(ProviderEvent("provider.prompt.accepted"))
            (workspace / "ambiguous-change.txt").write_text(
                "provider may have changed the workspace\n",
                encoding="utf-8",
            )
            raise ProviderUnavailableError(
                "codex",
                detail="Codex app-server connection closed",
            )

    rig = JourneyRig(
        tmp_path,
        codex=ClosedCodexAdapter("codex"),
    )
    rig.prime_capacity(claude=80, codex=10)
    try:
        receipt = await rig.message("Recover from a closed Codex transport.")

        attempts = rig.store.attempts(rig.session.session_id)
        assert receipt.status == "failed", receipt.result
        assert receipt.result["code"] == "E_NEEDS_RECONCILIATION"
        assert receipt.result["reconciliation_id"]
        assert [item.provider for item in attempts] == ["codex"]
        assert [item.status for item in attempts] == ["ambiguous"]
        assert not rig.adapters["claude"].prompts
        reconciliations = rig.store.pending_reconciliations(rig.session.session_id)
        assert len(reconciliations) == 1
        assert reconciliations[0].command_id == receipt.command_id
        assert reconciliations[0].audit["discovery_checkpoint_id"]
        assert rig.store.get_session(rig.session.session_id).attention == (
            "needs-reconciliation"
        )
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_child_budget_is_shared_across_provider_failover(
    tmp_path: Path,
) -> None:
    class ChildThenFailAdapter(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            event_handler = values["event_handler"]
            gate = values["child_launch_gate"]
            assert child_gate.admit(
                gate.database,
                gate.command_id,
                gate.limit,
                "claude:Agent:visible-child",
            )
            del event_handler
            return await super().run_turn(**values)

    claude = ChildThenFailAdapter("claude", fail_turns=1)
    codex = ScriptedAdapter("codex")
    rig = JourneyRig(tmp_path, claude=claude, codex=codex)
    rig.prime_capacity()
    try:
        receipt = await rig.message("Use one child, then fail over.")

        assert receipt.status == "complete", receipt.result
        assert claude.child_agent_limits == [16]
        assert codex.child_agent_limits == [16]
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["consumption"]["child_agents"] == 1
        assert rig.store.child_launch_gate(receipt.command_id)["consumed"] == 1
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_child_gate_survives_transport_loss_before_event(
    tmp_path: Path,
) -> None:
    class LaunchThenLoseTransport(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            gate = values["child_launch_gate"]
            assert child_gate.admit(
                gate.database,
                gate.command_id,
                gate.limit,
                "claude:Agent:lost-transport",
            )
            raise ProviderExhaustedError(self.provider_id)

    class RetryAdapter(ScriptedAdapter):
        def __init__(self) -> None:
            super().__init__("codex")
            self.second_launch_admitted = True

        async def run_turn(self, **values: Any) -> ProviderResult:
            gate = values["child_launch_gate"]
            self.second_launch_admitted = child_gate.admit(
                gate.database,
                gate.command_id,
                gate.limit,
                "codex:Agent:retry-child",
            )
            return await super().run_turn(**values)

    claude = LaunchThenLoseTransport("claude")
    codex = RetryAdapter()
    rig = JourneyRig(tmp_path, claude=claude, codex=codex)
    rig.prime_capacity()
    try:
        goal = create_goal(
            rig.session.session_id,
            "Complete with one child launch across provider retries.",
            budgets={"attempts": 2, "child_agents": 1},
        )
        rig.store.create_goal(goal)

        receipt = await rig.message("Launch one child and fail over.")

        assert receipt.status == "complete", receipt.result
        assert not codex.second_launch_admitted
        gate = rig.store.child_launch_gate(receipt.command_id)
        assert gate["permit_limit"] == 1
        assert gate["consumed"] == 1
        assert (
            rig.store.command_envelope(receipt.command_id)["consumption"][
                "child_agents"
            ]
            == 1
        )
        proof = proof_snapshot(rig.store, rig.session.session_id)
        proof_command = proof["commands"][0]
        assert proof_command["session_id"] == rig.session.session_id
        assert len(proof_command["command_envelope_digest"]) == 64
        admissions = proof["safety"]["child_launch_admissions"]
        assert len(admissions) == 2
        assert [item["admitted"] for item in admissions].count(True) == 1
        assert all(len(item["admission_key_digest"]) == 64 for item in admissions)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_approval_blocks_and_resumes_the_turn(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(
        tmp_path,
        codex=ScriptedAdapter("codex", request_approval=True),
    )
    rig.prime_capacity()
    try:
        turn = asyncio.create_task(rig.message("Run the validation.", provider="codex"))
        approval = await _wait_for_approval(
            lambda: rig.store.pending_approvals(rig.session.session_id)
        )
        assert rig.store.resolve_approval(
            str(approval["approval_id"]),
            {"decision": "accept"},
        )
        receipt = await asyncio.wait_for(turn, timeout=3)

        codex = rig.adapters["codex"]
        assert receipt.status == "complete"
        assert codex.approval_decisions == [{"decision": "accept"}]
        assert not rig.store.pending_approvals(rig.session.session_id)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_steering_reaches_the_active_provider(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(
        tmp_path,
        codex=ScriptedAdapter("codex", turn_delay=0.35),
    )
    rig.prime_capacity()
    try:
        turn = asyncio.create_task(
            rig.message("Start the implementation.", provider="codex")
        )
        await asyncio.sleep(0.05)
        receipt = rig.store.enqueue_command(
            rig.session.session_id,
            "steer",
            {"text": "Focus only on the parser."},
            new_uuid(),
        )
        completed = await asyncio.wait_for(turn, timeout=3)

        codex = rig.adapters["codex"]
        assert completed.status == "complete"
        assert codex.steered == ["Focus only on the parser."]
        assert rig.store.get_command(receipt.command_id).status == "complete"
        events = rig.store.all_events(rig.session.session_id)
        assert any(item.event_type == "user.steer" for item in events)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_evidence_completes_a_finite_goal(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
        goal = create_goal(
            rig.session.session_id,
            "Deliver a tested change.",
            predicates=(
                {
                    "type": "command",
                    "subject": "make test",
                    "outcome": "passed",
                },
            ),
        )
        rig.store.create_goal(goal)
        rig.store.add_evidence(
            make_evidence(
                goal.goal_id,
                "command",
                "make test",
                "passed",
            )
        )

        receipt = await rig.message("Finish the work.", provider="codex")

        assert receipt.status == "complete"
        current = rig.store.get_session(rig.session.session_id)
        assert current.lifecycle == "completed"
        assert current.attention == "ready"
        assert rig.store.get_goal(goal.goal_id).status == "complete"
        events = rig.store.all_events(rig.session.session_id)
        assert any(item.event_type == "goal.completed" for item in events)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_failed_bash_result_records_goal_evidence_without_completion(
    tmp_path: Path,
) -> None:
    class FailedBashAdapter(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            event_handler = values["event_handler"]
            await event_handler(
                ProviderEvent(
                    "provider.prompt.accepted",
                    status="accepted",
                    native_session_id="claude-native-session",
                )
            )
            await event_handler(
                ProviderEvent(
                    "tool.started",
                    text="Bash",
                    metadata={
                        "id": "bash-failed",
                        "input": {"command": "make test"},
                        "name": "Bash",
                    },
                )
            )
            await event_handler(
                ProviderEvent(
                    "tool.completed",
                    text="Exit code 7\ntests failed",
                    metadata={
                        "exit_code": 7,
                        "is_error": True,
                        "tool_use_id": "bash-failed",
                    },
                )
            )
            return ProviderResult(
                provider="claude",
                native_session_id="claude-native-session",
                native_turn_id="claude-turn",
                status="complete",
                usage={"total_tokens": 42},
            )

    rig = JourneyRig(tmp_path, claude=FailedBashAdapter("claude"))
    rig.prime_capacity()
    try:
        goal = create_goal(
            rig.session.session_id,
            "Run the exact test command successfully.",
            predicates=(
                {
                    "type": "command",
                    "subject": "make test",
                    "outcome": "passed",
                },
            ),
        )
        rig.store.create_goal(goal)

        receipt = await rig.message("Run make test.", provider="claude")

        assert receipt.status == "complete"
        evidence = rig.store.evidence(goal.goal_id)
        assert len(evidence) == 1
        assert evidence[0].evidence_type == "command"
        assert evidence[0].subject == "make test"
        assert evidence[0].outcome == "failed"
        assert evidence[0].value["exit_code"] == 7
        assert len(evidence[0].value["command_digest"]) == 64
        assert evidence[0].value["schema"] == (
            "p13i/agent-harness/failed-command-evidence/v1"
        )
        assert rig.store.get_goal(goal.goal_id).status == "active"
        assert rig.store.get_session(rig.session.session_id).lifecycle == "running"
        events = rig.store.all_events(rig.session.session_id)
        goal_evidence = [
            event for event in events if event.event_type == "goal.evidence"
        ]
        assert len(goal_evidence) == 1
        assert goal_evidence[0].turn_id
        assert not any(event.event_type == "goal.completed" for event in events)
    finally:
        rig.close()


def test_failed_command_evidence_and_heartbeat_boundaries(tmp_path: Path) -> None:
    failures: list[dict[str, Any]] = []
    commands: dict[str, str] = {}
    common = {
        "provider": "claude",
        "attempt_id": "attempt",
        "turn_id": "turn",
    }
    worker_module._observe_bash_command_result(
        ProviderEvent("tool.started"),
        commands,
        failures,
        **common,
    )
    worker_module._observe_bash_command_result(
        ProviderEvent("tool.started", metadata={"id": "read", "name": "Read"}),
        commands,
        failures,
        **common,
    )
    worker_module._observe_bash_command_result(
        ProviderEvent("tool.started", metadata={"name": "Bash"}),
        commands,
        failures,
        **common,
    )
    worker_module._observe_bash_command_result(
        ProviderEvent(
            "tool.started",
            metadata={
                "id": "bad-input",
                "input": "not-an-object",
                "name": "Bash",
            },
        ),
        commands,
        failures,
        **common,
    )
    worker_module._observe_bash_command_result(
        ProviderEvent(
            "tool.started",
            metadata={"id": "empty", "input": {"command": ""}, "name": "Bash"},
        ),
        commands,
        failures,
        **common,
    )
    worker_module._observe_bash_command_result(
        ProviderEvent("tool.progress", metadata={"id": "progress"}),
        commands,
        failures,
        **common,
    )
    worker_module._observe_bash_command_result(
        ProviderEvent(
            "tool.completed",
            metadata={"exit_code": 1, "tool_use_id": "unknown"},
        ),
        commands,
        failures,
        **common,
    )
    for tool_id, exit_code in (
        ("boolean", True),
        ("missing", None),
        ("success", 0),
        ("failure", 2),
    ):
        commands[tool_id] = "make test"
        metadata: dict[str, Any] = {"tool_use_id": tool_id}
        if exit_code is not None:
            metadata["exit_code"] = exit_code
        worker_module._observe_bash_command_result(
            ProviderEvent("tool.completed", metadata=metadata),
            commands,
            failures,
            **common,
        )
    assert len(failures) == 1

    assert worker_module._failed_command_evidence(None, failures) == ()
    invariant = create_goal(
        new_uuid(),
        "Remain healthy.",
        kind="invariant",
        predicates=({"type": "command", "subject": "make test"},),
    )
    assert worker_module._failed_command_evidence(invariant, failures) == ()
    finite = create_goal(
        new_uuid(),
        "Pass tests.",
        predicates=({"type": "command", "subject": "make test"},),
    )
    assert worker_module._failed_command_evidence(finite, {}) == ()
    evidence = worker_module._failed_command_evidence(finite, [None, *failures])
    assert len(evidence) == 1
    fallback = create_goal(
        new_uuid(),
        "Pass the named build stage.",
        predicates=(
            {"type": "command", "subject": "build-stage"},
            {"type": "probe", "subject": "health"},
        ),
    )
    assert len(worker_module._failed_command_evidence(fallback, failures)) == 1
    ambiguous = create_goal(
        new_uuid(),
        "Pass both commands.",
        predicates=(
            {"type": "command", "subject": "build"},
            {"type": "command", "subject": "test"},
            {"type": "command", "subject": "ignored", "outcome": "failed"},
        ),
    )
    assert worker_module._failed_command_evidence(ambiguous, failures) == ()

    rig = JourneyRig(tmp_path)
    try:
        calls: list[str] = []
        rig.worker._worker_heartbeat_at = worker_module.time.monotonic()
        original_heartbeat = rig.store.heartbeat_worker
        rig.store.heartbeat_worker = lambda *unused: calls.append("heartbeat") or True
        assert rig.worker._maintain_worker_ownership()
        assert calls == []
        rig.worker._worker_heartbeat_at = 0.0
        assert rig.worker._maintain_worker_ownership()
        assert calls == ["heartbeat"]
        rig.store.heartbeat_worker = lambda *unused: False
        assert not rig.worker._maintain_worker_ownership(force=True)
        rig.store.heartbeat_worker = original_heartbeat
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_exhausted_goal_budget_pauses_before_provider_work(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
        goal = create_goal(
            rig.session.session_id,
            "Stay within the declared budget.",
            budgets={"turns": 0},
        )
        rig.store.create_goal(goal)

        receipt = await rig.message("Do more work.", provider="codex")

        codex = rig.adapters["codex"]
        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_GOAL_BUDGET"
        assert not codex.prompts
        current = rig.store.get_session(rig.session.session_id)
        assert current.lifecycle == "paused"
        assert current.attention == "needs-input"
        await rig.worker._evaluate_goal()
        assert rig.store.get_goal(goal.goal_id).status == "waiting"
        assert any(
            item.event_type == "goal.budget_exhausted"
            for item in rig.store.all_events(rig.session.session_id)
        )
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_goal_remainder_constrains_current_envelope(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
        goal = create_goal(
            rig.session.session_id,
            "Keep the current turn within the remaining goal budget.",
            budgets={
                "tokens": 10,
                "context_tokens": 8,
                "output_tokens": 7,
                "tool_calls": 4,
                "attempts": 1,
                "child_agents": 1,
                "seconds": 10_000,
                "dollars": 2,
            },
        )
        rig.store.create_goal(goal)

        receipt = await rig.message(
            "This context cannot fit.",
            metered_budget=1,
        )

        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_SAFETY_GUARD"
        assert not rig.adapters["claude"].prompts
        assert not rig.adapters["codex"].prompts
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["limits"]["max_context_tokens"] == 8
        assert envelope["limits"]["max_output_tokens"] == 7
        assert envelope["limits"]["max_total_tokens"] == 10
        assert envelope["limits"]["max_tool_calls"] == 4
        assert envelope["limits"]["max_attempts"] == 1
        assert envelope["limits"]["max_child_agents"] == 1
        assert envelope["limits"]["max_dollars"] == 1.0
        assert 0 < envelope["limits"]["max_seconds"] <= 10_000
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_rejects_nonpositive_metered_budget_before_provider(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
        for budget in (0, -1):
            receipt = await rig.message(
                "Do not spend metered credits.",
                metered_budget=budget,
            )
            assert receipt.status == "failed"
            assert receipt.result["code"] == "E_SAFETY_BUDGET"
        assert not rig.adapters["claude"].prompts
        assert not rig.adapters["codex"].prompts
    finally:
        rig.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_name", ["tool_calls", "child_agents", "dollars"])
async def test_e2e_zero_discretionary_goal_budget_allows_parent_turn(
    tmp_path: Path,
    budget_name: str,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
        goal = create_goal(
            rig.session.session_id,
            "Do not cross an exhausted discretionary budget.",
            budgets={budget_name: 0},
        )
        rig.store.create_goal(goal)

        receipt = await rig.message("Complete without discretionary resource use.")

        assert receipt.status == "complete", receipt.result
        envelope = rig.store.command_envelope(receipt.command_id)
        if budget_name == "tool_calls":
            assert envelope["limits"]["max_tool_calls"] == 0
        if budget_name == "child_agents":
            assert envelope["limits"]["max_child_agents"] == 0
            assert rig.store.child_launch_gate(receipt.command_id)["permit_limit"] == 0
        if budget_name == "dollars":
            assert envelope["limits"]["max_dollars"] == 0
            assert envelope["consumption"]["dollars"] == 0
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_limit_tightening_cannot_grant_spend_under_zero_dollar_goal(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    rig.store.create_goal(
        create_goal(
            rig.session.session_id,
            "Never authorize metered credits.",
            budgets={"dollars": 0},
        )
    )
    try:
        receipt = await rig.message(
            "Reject an attempted spend grant.",
            safety_limits={"max_dollars": 100},
        )

        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_SAFETY_BUDGET"
        assert not rig.adapters["claude"].prompts
        assert not rig.adapters["codex"].prompts
        assert not rig.store.attempts(rig.session.session_id)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_nonfinite_limits_and_metered_budget_never_reach_provider(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
        requests = [
            {"safety_limits": {"binding_ceiling": math.nan}},
            {"safety_limits": {"max_dollars": math.inf}},
            {"metered_budget": math.nan},
        ]
        for index, request in enumerate(requests):
            receipt = await rig.message(
                "Reject non-finite request " + str(index) + ".",
                **request,
            )
            assert receipt.status == "failed"
            assert receipt.result["code"] == "E_SAFETY_BUDGET"
        assert not rig.adapters["claude"].prompts
        assert not rig.adapters["codex"].prompts
        assert not rig.store.attempts(rig.session.session_id)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_zero_dollar_goal_rejects_metered_route_before_provider(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter(
        "claude",
        cost=0.1,
        claims_cost_reporting=True,
    )
    rig = JourneyRig(tmp_path, claude=claude)
    rig.scheduler._usage_cache = {
        "claude": _usage("claude", 20.0, credits=True),
        "codex": _usage("codex", 20.0),
    }
    rig.scheduler._usage_at = asyncio.get_running_loop().time()
    try:
        goal = create_goal(
            rig.session.session_id,
            "Use subscription capacity only.",
            budgets={"dollars": 0},
        )
        rig.store.create_goal(goal)

        receipt = await rig.message(
            "Do not enter the metered route.",
            provider="claude",
            metered_budget=1,
        )

        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_PROVIDER_UNAVAILABLE"
        assert not claude.prompts
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_subscription_cost_equivalent_is_not_metered_spend(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter(
        "claude",
        cost=0.25,
        claims_cost_reporting=True,
    )
    rig = JourneyRig(tmp_path, claude=claude)
    rig.scheduler._usage_cache = {
        "claude": _usage("claude", 20.0, credits=False),
        "codex": _usage("codex", 20.0),
    }
    rig.scheduler._usage_at = asyncio.get_running_loop().time()
    try:
        goal = create_goal(
            rig.session.session_id,
            "Use subscription capacity only.",
            budgets={"dollars": 0},
        )
        rig.store.create_goal(goal)

        receipt = await rig.message(
            "Complete on the subscription route.",
            provider="claude",
        )

        assert receipt.status == "complete"
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["consumption"]["dollars"] == 0
        assert envelope["consumption"]["exact_dollars"] is False
    finally:
        rig.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reported_cost", "expected_cost"),
    [
        (None, None),
        ({"total_cost_usd": -1, "cost_usd": 0}, None),
        (0.5, 0.5),
    ],
)
async def test_e2e_metered_work_requires_exact_bounded_cost(
    tmp_path: Path,
    reported_cost: object | None,
    expected_cost: float | None,
) -> None:
    claude = ScriptedAdapter(
        "claude",
        cost=reported_cost,
        claims_cost_reporting=True,
    )
    rig = JourneyRig(tmp_path, claude=claude)
    rig.scheduler._usage_cache = {
        "claude": _usage("claude", 20.0, credits=True),
        "codex": _usage("codex", 20.0),
    }
    rig.scheduler._usage_at = asyncio.get_running_loop().time()
    try:
        receipt = await rig.message(
            "Use the explicit metered envelope.",
            provider="claude",
            metered_budget=1,
        )

        envelope = rig.store.command_envelope(receipt.command_id)
        if expected_cost is None:
            assert receipt.status == "failed"
            assert receipt.result["code"] == "E_SAFETY_GUARD"
            assert receipt.result["message"].endswith("dollar-accounting")
            assert not rig.store.pending_reconciliations(rig.session.session_id)
            checkpoints = rig.store.checkpoints(rig.session.session_id)
            assert checkpoints[-1].provider == "claude"
            assert envelope["guard_reason"] == "dollar-accounting"
            incidents = rig.store.guard_incidents(rig.session.session_id)
            assert incidents[0]["reason"] == "dollar-accounting"
            assert incidents[0]["action"] == "pause"
            events = rig.store.all_events(rig.session.session_id)
            requested = [
                event
                for event in events
                if event.event_type == "reconciliation.requested"
            ]
            assert requested == []
        else:
            assert receipt.status == "complete"
            assert envelope["consumption"]["dollars"] == expected_cost
            assert envelope["consumption"]["exact_dollars"] is True
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_unattended_admission_requires_fresh_headroom(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    rig.scheduler._usage_cache = {
        "claude": _usage("claude", None),
        "codex": _usage("codex", None),
    }
    rig.scheduler._usage_at = asyncio.get_running_loop().time()
    try:
        receipt = await rig.message("Run unattended operations.")

        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_PROVIDER_UNAVAILABLE"
        assert not rig.adapters["claude"].prompts
        assert not rig.adapters["codex"].prompts
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_unattended_pinned_provider_dispatches_without_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rate_limited = {
        "claude": UsageSnapshot(
            provider="claude",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error="HTTP 429",
        ),
        "codex": UsageSnapshot(
            provider="codex",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error="HTTP 429",
        ),
    }

    async def probe() -> dict[str, UsageSnapshot]:
        return rate_limited

    monkeypatch.setattr("agent_harness.scheduler.probe_all", probe)
    rig = JourneyRig(tmp_path)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    rig.scheduler._usage_cache = dict(rate_limited)
    rig.scheduler._usage_at = asyncio.get_running_loop().time()
    try:
        receipt = await rig.message(
            "Run the unattended builder step.",
            provider="claude",
            model="claude-default",
            effort="medium",
        )

        assert receipt.status == "complete", receipt.result
        attempts = rig.store.attempts(rig.session.session_id)
        assert [item.provider for item in attempts] == ["claude"]
        assert attempts[0].model == "claude-default"
        assert attempts[0].effort == "medium"
        assert len(rig.adapters["claude"].prompts) == 1
        assert not rig.adapters["codex"].prompts
        routed = rig.store.routing_decisions(rig.session.session_id)
        assert routed[0]["payload"]["binding_percent"] is None
        proof = proof_snapshot(rig.store, rig.session.session_id)
        assert proof["routing"][0]["binding_percent"] is None
        assert proof["routing"][0]["admissible_at_route"] is False
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_retryable_provider_failure_keeps_session_claimable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
        original_failover = rig.worker._execute_with_failover
        attempts: list[str] = []

        async def failover(
            command_id: str,
            payload: dict[str, Any],
            text: str,
            guard: Any,
            evaluator: Any = None,
        ) -> dict[str, Any]:
            attempts.append(command_id)
            if len(attempts) == 1:
                raise ProviderUnavailableError(
                    "claude",
                    detail="fleet saturated",
                )
            return await original_failover(
                command_id,
                payload,
                text,
                guard,
                evaluator,
            )

        monkeypatch.setattr(rig.worker, "_execute_with_failover", failover)

        receipt = await rig.message("Trigger a transient capacity failure.")

        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_PROVIDER_UNAVAILABLE"
        assert receipt.result["retryable"] is True
        failed_events = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "turn.failed"
        ]
        assert len(failed_events) == 1
        assert failed_events[0].metadata["code"] == "E_PROVIDER_UNAVAILABLE"
        assert failed_events[0].metadata["retryable"] is True
        current = rig.store.get_session(rig.session.session_id)
        assert current.lifecycle == "running"
        assert current.attention == "idle"

        # The claim loop must pick up a queued follow-up without an
        # operator resume: drive the real worker loop to claim it.
        follow_up = rig.enqueue_message("Continue after the transient failure.")
        worker_task = asyncio.create_task(rig.worker.run())
        try:
            current = rig.store.get_command(follow_up.command_id)
            for unused in range(1000):
                del unused
                current = rig.store.get_command(follow_up.command_id)
                if current.status == "complete":
                    break
                await asyncio.sleep(0.01)
            assert current.status == "complete", current.result
        finally:
            rig.worker._stopping = True
            if not worker_task.done():
                worker_task.cancel()
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_identical_resubmission_retries_a_retryable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
        original_failover = rig.worker._execute_with_failover
        attempts: list[str] = []

        async def failover(
            command_id: str,
            payload: dict[str, Any],
            text: str,
            guard: Any,
            evaluator: Any = None,
        ) -> dict[str, Any]:
            attempts.append(command_id)
            if len(attempts) == 1:
                raise ProviderUnavailableError(
                    "grok",
                    detail="fleet saturated",
                )
            return await original_failover(
                command_id,
                payload,
                text,
                guard,
                evaluator,
            )

        monkeypatch.setattr(rig.worker, "_execute_with_failover", failover)
        payload = {"text": "Review the cs-builder change."}
        submitted, was_created = rig.store.ensure_message_command(
            rig.session.session_id,
            payload,
            "cs-builder-review",
        )
        assert was_created

        await rig.execute(submitted.command_id)
        failed = rig.store.get_command(submitted.command_id)
        assert failed.status == "failed"
        assert failed.result["code"] == "E_PROVIDER_UNAVAILABLE"
        assert failed.result["retryable"] is True

        requeued, created_again = rig.store.ensure_message_command(
            rig.session.session_id,
            payload,
            "cs-builder-review",
        )
        assert not created_again
        assert requeued.command_id == submitted.command_id
        assert requeued.status == "queued"

        await rig.execute(submitted.command_id)
        completed = rig.store.get_command(submitted.command_id)

        assert completed.status == "complete", completed.result
        assert attempts == [submitted.command_id, submitted.command_id]
        instructions = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "user.message"
        ]
        assert [event.metadata["command_id"] for event in instructions] == [
            submitted.command_id
        ]
        assert len(rig.store.attempts(rig.session.session_id)) == 1
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_non_retryable_failure_still_pauses_for_operator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:

        async def defect(
            unused_command_id: str,
            unused_payload: dict[str, Any],
            unused_text: str,
            unused_guard: Any,
            unused_evaluator: Any = None,
        ) -> dict[str, Any]:
            del unused_command_id, unused_payload, unused_text
            del unused_guard, unused_evaluator
            raise HarnessError("E_DEFECT", "a genuine non-retryable defect")

        monkeypatch.setattr(rig.worker, "_execute_with_failover", defect)

        receipt = await rig.message("Trigger a genuine defect.")

        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_DEFECT"
        assert receipt.result["retryable"] is False
        failed_events = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "turn.failed"
        ]
        assert len(failed_events) == 1
        current = rig.store.get_session(rig.session.session_id)
        assert current.lifecycle == "paused"
        assert current.attention == "needs-input"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_hard_usage_limit_interrupts_without_recovery(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude", usage_turns=1)
    rig = JourneyRig(tmp_path, claude=claude)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    rig.prime_capacity()
    try:
        receipt = await rig.message(
            "Run bounded operations.",
            provider="claude",
            workload="operations",
        )

        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_SAFETY_GUARD"
        assert receipt.result["reconciliation_id"]
        assert claude.interruptions == 1
        assert len(rig.store.attempts(rig.session.session_id)) == 1
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["state"] == "paused"
        assert envelope["guard_reason"] == "output-tokens"
        incidents = rig.store.guard_incidents(rig.session.session_id)
        assert incidents[0]["reason"] == "output-tokens"
        assert incidents[0]["action"] == "pause"
        events = rig.store.all_events(rig.session.session_id)
        requested = [
            event
            for event in events
            if event.event_type == "reconciliation.requested"
        ]
        assert len(requested) == 1
        assert requested[0].metadata["reason"] == "output-tokens"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_repetition_pauses_without_another_provider_attempt(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude", guard_turns=2)
    rig = JourneyRig(tmp_path, claude=claude)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    rig.prime_capacity()
    try:
        receipt = await rig.message("Implement without rereading context.")

        assert receipt.status == "failed", receipt.result
        attempts = rig.store.attempts(rig.session.session_id)
        assert [item.provider for item in attempts] == ["claude"]
        assert claude.efforts == ["high"]
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["recovery_stage"] == 0
        assert envelope["consumption"]["attempts"] == 1
        assert len(rig.store.guard_incidents(rig.session.session_id)) == 1
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_explicit_effort_pin_is_preserved_during_failover(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude", fail_turns=1)
    codex = ScriptedAdapter("codex")
    rig = JourneyRig(tmp_path, claude=claude, codex=codex)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    rig.prime_capacity()
    try:
        receipt = await rig.message(
            "Preserve the exact effort pin.",
            effort="high",
        )

        assert receipt.status == "complete", receipt.result
        attempts = rig.store.attempts(rig.session.session_id)
        assert [item.provider for item in attempts] == ["claude", "codex"]
        assert claude.efforts == ["high"]
        assert codex.efforts == ["high"]
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_repeated_failed_dispatch_pauses_before_provider(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude", fail_turns=1)
    rig = JourneyRig(tmp_path, claude=claude)
    rig.prime_capacity()
    try:
        first = await rig.message(
            "Repeat-safe instruction.",
            provider="claude",
        )
        rig.store.update_session(
            rig.session.session_id,
            lifecycle="running",
            attention="idle",
        )
        second = await rig.message(
            "Repeat-safe instruction.",
            provider="claude",
        )

        assert first.status == "failed"
        assert second.status == "failed"
        assert second.result["code"] == "E_SAFETY_GUARD"
        assert len(claude.prompts) == 1
        envelope = rig.store.command_envelope(second.command_id)
        assert envelope["guard_reason"] == "repeated-instruction"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_repeated_completed_dispatch_pauses_before_provider(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude")
    rig = JourneyRig(tmp_path, claude=claude)
    rig.prime_capacity()
    try:
        first = await rig.message(
            "Repeat-safe completed instruction.",
            provider="claude",
        )
        second = await rig.message(
            "Repeat-safe completed instruction.",
            provider="claude",
        )
        (Path(rig.session.worktree) / "material.txt").write_text(
            "new material state\n",
            encoding="utf-8",
        )
        material_checkpoint = checkpoint_workspace(
            rig.store.get_session(rig.session.session_id),
            rig.blobs,
            sequence=rig.store.last_sequence(rig.session.session_id),
            provider="claude",
            native_session_id="claude-native-session",
            context_text="material generation changed",
        )
        rig.store.add_checkpoint(material_checkpoint)
        third = await rig.message(
            "Repeat-safe completed instruction.",
            provider="claude",
        )
        fourth = await rig.message(
            "Repeat-safe completed instruction.",
            provider="claude",
        )
        reason = "Operator confirmed a new bounded dispatch generation."
        retained_receipt = {"actor": "test-operator", "scope": "retry"}
        authorization = {
            "schema": ("p13i/agent-harness/dispatch-invalidation-authorization/v1"),
            "session_id": rig.session.session_id,
            "reason": reason,
            "receipt": retained_receipt,
            "receipt_sha256": normalized_digest(retained_receipt),
        }
        rig.store.create_dispatch_invalidation(
            rig.session.session_id,
            reason=reason,
            authorization=authorization,
            request_digest=normalized_digest(
                {"reason": reason, "authorization": authorization}
            ),
            idempotency_key="new-dispatch-generation",
        )
        fifth = await rig.message(
            "Repeat-safe completed instruction.",
            provider="claude",
        )

        assert first.status == "complete"
        assert second.status == "failed"
        assert second.result["code"] == "E_SAFETY_GUARD"
        assert third.status == "complete"
        assert fourth.status == "failed"
        assert fourth.result["code"] == "E_SAFETY_GUARD"
        assert fifth.status == "complete"
        assert len(claude.prompts) == 3
        envelope = rig.store.command_envelope(second.command_id)
        assert envelope["guard_reason"] == "repeated-instruction"
        proof_events = rig.store.all_events(rig.session.session_id)
        assert any(
            event.event_type == "dispatch.invalidation" for event in proof_events
        )
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_distinct_managed_steps_reuse_governing_artifacts(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    workspace = Path(rig.session.worktree)
    (workspace / "AGENTS.md").write_text("stable guidance\n", encoding="utf-8")
    (workspace / "plans").mkdir()
    (workspace / "plans" / "stable.gpt.md").write_text(
        "stable plan\n",
        encoding="utf-8",
    )
    skill = workspace / ".agents" / "skills" / "stable"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("stable skill\n", encoding="utf-8")
    rig.prime_capacity()
    try:
        first = await rig.message(
            "Complete managed stage one.",
            provider="claude",
            turn_ref={"step_id": "stage-1", "agent_role": "builder"},
        )
        second = await rig.message(
            "Complete managed stage two.",
            provider="claude",
            turn_ref={"step_id": "stage-2", "agent_role": "builder"},
        )

        assert first.status == "complete", first.result
        assert second.status == "complete", second.result
        assert len(rig.adapters["claude"].prompts) == 2
    finally:
        rig.close()


def _advance_orchestration_generation(
    rig: JourneyRig,
    prior_command_id: str,
    next_turn_ref: dict[str, str],
    idempotency_key: str,
    policy: dict[str, Any],
    transition_sequence: int,
    next_command_payload: dict[str, Any],
) -> dict[str, Any]:
    reason = "Advance one completed orchestration stage."
    anchor = rig.store.dispatch_transition_anchor(rig.session.session_id)
    assert anchor["eligible"] is True
    assert anchor["prior_command_id"] == prior_command_id
    prior_checkpoint_id = str(anchor["prior_checkpoint_id"])
    prior_material_digest = str(anchor["prior_material_digest"])
    prior_generation_digest = str(anchor["prior_generation_digest"])
    policy_sha256 = normalized_digest(policy)
    execution_profile = str(rig.store.session_safety(rig.session.session_id)["profile"])
    next_command_digest = command_envelope_digest(
        "message",
        next_command_payload,
        execution_profile,
    )
    retained_receipt = {
        "session_id": rig.session.session_id,
        "external_ref": rig.session.external_ref,
        "goal_id": str(rig.store.goal_for_session(rig.session.session_id).goal_id),
        "prior_command_id": prior_command_id,
        "prior_command_type": anchor["prior_command_type"],
        "prior_anchor_kind": anchor["prior_anchor_kind"],
        "prior_reconciliation_id": anchor["prior_reconciliation_id"],
        "prior_reconciliation_resolution": anchor["prior_reconciliation_resolution"],
        "prior_checkpoint_id": prior_checkpoint_id,
        "prior_generation_digest": prior_generation_digest,
        "prior_material_digest": prior_material_digest,
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
        "session_id": rig.session.session_id,
        "goal_id": str(rig.store.goal_for_session(rig.session.session_id).goal_id),
        "reason": reason,
        "prior_command_id": prior_command_id,
        "prior_command_type": anchor["prior_command_type"],
        "prior_anchor_kind": anchor["prior_anchor_kind"],
        "prior_reconciliation_id": anchor["prior_reconciliation_id"],
        "prior_reconciliation_resolution": anchor["prior_reconciliation_resolution"],
        "prior_checkpoint_id": prior_checkpoint_id,
        "prior_generation_digest": prior_generation_digest,
        "prior_material_digest": prior_material_digest,
        "next_turn_ref": next_turn_ref,
        "transition_sequence": transition_sequence,
        "epoch_id": str(policy["epoch_id"]),
        "external_orchestrator": rig.session.external_ref["orchestrator"],
        "external_job_id": rig.session.external_ref["job_id"],
        "policy_sha256": policy_sha256,
        "next_command_digest": next_command_digest,
        "receipt": retained_receipt,
        "receipt_sha256": normalized_digest(retained_receipt),
    }
    wire_authorization = authorization
    if transition_sequence == 1:
        wire_authorization["policy"] = policy
    else:
        wire_authorization["policy_ref"] = {
            "policy_sha256": policy_sha256,
            "session_id": rig.session.session_id,
            "goal_id": str(rig.store.goal_for_session(rig.session.session_id).goal_id),
            "epoch_id": str(policy["epoch_id"]),
        }
    stored_authorization = wire_authorization
    if transition_sequence > 1:
        stored_authorization = dict(wire_authorization)
        stored_authorization["policy"] = policy
    payload = {
        "reason": reason,
        "prior_command_id": prior_command_id,
        "prior_command_type": anchor["prior_command_type"],
        "prior_anchor_kind": anchor["prior_anchor_kind"],
        "prior_reconciliation_id": anchor["prior_reconciliation_id"],
        "prior_reconciliation_resolution": anchor["prior_reconciliation_resolution"],
        "prior_checkpoint_id": prior_checkpoint_id,
        "prior_generation_digest": prior_generation_digest,
        "prior_material_digest": prior_material_digest,
        "next_turn_ref": next_turn_ref,
        "transition_sequence": transition_sequence,
        "next_command_digest": next_command_digest,
        "authorization": wire_authorization,
    }
    return rig.store.create_dispatch_invalidation(
        rig.session.session_id,
        reason=reason,
        authorization=stored_authorization,
        request_digest=normalized_digest(payload),
        idempotency_key=idempotency_key,
        prior_command_id=prior_command_id,
        next_turn_ref=next_turn_ref,
        authorization_digest=normalized_digest(wire_authorization),
    )


def _install_transition_policy(
    rig: JourneyRig,
    *,
    allowed_agent_roles: list[str],
    allowed_step_prefixes: list[str],
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    policy = {
        "schema": ("p13i/agent-harness/dispatch-generation-transition-policy/v1"),
        "session_id": rig.session.session_id,
        "external_ref": rig.session.external_ref,
        "epoch_id": "journey-epoch-1",
        "allowed_agent_roles": allowed_agent_roles,
        "allowed_step_prefixes": allowed_step_prefixes,
        "max_transitions": len(transitions),
        "transitions": transitions,
    }
    rig.store.create_goal(
        create_goal(
            rig.session.session_id,
            "Complete only declared orchestration stages.",
            constraints=(
                "dispatch-generation-transition-policy-sha256:"
                + normalized_digest(policy),
                "dispatch-generation-transition-epoch:" + str(policy["epoch_id"]),
            ),
            permitted_providers=("claude", "codex"),
            permitted_efforts=("low", "medium", "high"),
        )
    )
    return policy


@pytest.mark.asyncio
async def test_e2e_provider_switch_accepts_an_exact_checkpoint_collapse(
    tmp_path: Path,
) -> None:
    external_ref = {
        "orchestrator": "p13i/machines/cs-builder",
        "job_id": "provider-switch-checkpoint-collapse",
    }
    rig = JourneyRig(tmp_path, external_ref=external_ref)
    workspace = Path(rig.session.worktree)
    tracked = workspace / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(workspace), "add", "tracked.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "-qm", "base"],
        check=True,
    )
    tracked.write_text("accepted\n", encoding="utf-8")
    next_turn_ref = {"step_id": "review", "agent_role": "reviewer"}
    next_payload = {
        "text": "Review the exact collapsed checkpoint.",
        "provider": "codex",
        "turn_ref": next_turn_ref,
    }
    policy = _install_transition_policy(
        rig,
        allowed_agent_roles=["reviewer"],
        allowed_step_prefixes=["review"],
        transitions=[
            {
                "sequence": 1,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    next_payload,
                    "unattended",
                ),
            }
        ],
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    try:
        prior = await rig.message(
            "Complete implementation before the provider switch.",
            provider="claude",
            turn_ref={"step_id": "implement", "agent_role": "builder"},
        )
        assert prior.status == "complete"
        transition = _advance_orchestration_generation(
            rig,
            prior.command_id,
            next_turn_ref,
            "provider-switch-checkpoint-collapse",
            policy,
            1,
            next_payload,
        )
        authorized_digest = inspect_workspace(workspace)[0]
        subprocess.run(
            ["git", "-C", str(workspace), "add", "tracked.txt"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(workspace), "commit", "-qm", "collapse"],
            check=True,
        )
        assert inspect_workspace(workspace)[0] != authorized_digest

        reviewed = await rig.message(**next_payload)

        assert reviewed.status == "complete", reviewed.result
        transition_events = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.metadata.get("invalidation_id") == transition["invalidation_id"]
            and event.event_type.startswith("dispatch.generation.transition.")
        ]
        assert [event.status for event in transition_events] == [
            "reserved",
            "consumed",
        ]
        assert all(
            event.metadata["material_binding"] == "checkpoint-collapse"
            for event in transition_events
        )
        assert len(rig.adapters["claude"].prompts) == 1
        assert len(rig.adapters["codex"].prompts) == 1
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_control_anchor_consumes_exact_next_message(
    tmp_path: Path,
) -> None:
    class InterruptibleAdapter(ScriptedAdapter):
        def __init__(self) -> None:
            super().__init__("codex")
            self.started = asyncio.Event()
            self.interrupted = asyncio.Event()

        async def run_turn(self, **values: Any) -> ProviderResult:
            if self.interrupted.is_set():
                return await super().run_turn(**values)
            pre_prompt_gate = values["pre_prompt_gate"]
            if pre_prompt_gate is not None:
                await pre_prompt_gate()
            event_handler = values["event_handler"]
            await event_handler(
                ProviderEvent(
                    "provider.prompt.accepted",
                    status="accepted",
                    native_session_id="codex-interrupted",
                )
            )
            self.started.set()
            await self.interrupted.wait()
            return ProviderResult(
                provider="codex",
                native_session_id="codex-interrupted",
                native_turn_id="codex-interrupted-turn",
                status="cancelled",
                usage={"total_tokens": 1},
            )

        async def interrupt(self) -> None:
            self.interruptions += 1
            self.interrupted.set()

    external_ref = {
        "orchestrator": "p13i/machines/cs-builder",
        "job_id": "control-anchor-consumption",
    }
    codex = InterruptibleAdapter()
    rig = JourneyRig(
        tmp_path,
        codex=codex,
        external_ref=external_ref,
    )
    next_turn_ref = {
        "step_id": "resume-after-interrupt",
        "agent_role": "builder",
    }
    next_command_payload = {
        "text": "Resume the exact stage after interruption.",
        "provider": "codex",
        "turn_ref": next_turn_ref,
    }
    first_turn_ref = {
        "step_id": "post-provider-transition",
        "agent_role": "builder",
    }
    first_command_payload = {
        "text": "Consume the first exact transition.",
        "provider": "codex",
        "turn_ref": first_turn_ref,
    }
    policy = _install_transition_policy(
        rig,
        allowed_agent_roles=["builder"],
        allowed_step_prefixes=[
            "post-provider-transition",
            "resume-after-interrupt",
        ],
        transitions=[
            {
                "sequence": 1,
                "next_turn_ref": first_turn_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    first_command_payload,
                    "unattended",
                ),
            },
            {
                "sequence": 2,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    next_command_payload,
                    "unattended",
                ),
            },
        ],
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    try:
        prior = await rig.message(
            "Complete the provider stage before interruption.",
            provider="claude",
            turn_ref={"step_id": "implement", "agent_role": "builder"},
        )
        assert prior.status == "complete"
        _advance_orchestration_generation(
            rig,
            prior.command_id,
            first_turn_ref,
            "provider-to-first-control-lineage-message",
            policy,
            1,
            first_command_payload,
        )
        first = rig.enqueue_message(**first_command_payload)
        claimed_first = rig.store.claim_command(rig.session.session_id)
        assert claimed_first is not None
        assert claimed_first.command_id == first.command_id
        message_task = asyncio.create_task(rig.worker._message(claimed_first))
        await asyncio.wait_for(codex.started.wait(), timeout=2)
        assert rig.store.get_command(first.command_id).status == "dispatching"
        interrupt = rig.store.enqueue_command(
            rig.session.session_id,
            "interrupt",
            {"target_command_id": first.command_id},
            "bounded-control-interrupt",
        )
        await asyncio.wait_for(message_task, timeout=2)
        assert rig.store.get_command(interrupt.command_id).status == "complete"
        assert codex.interruptions == 1
        assert rig.store.get_command(first.command_id).status == "failed"
        attempts = rig.store.attempts(rig.session.session_id)
        assert attempts[-1].status == "cancelled"
        assert attempts[-1].ended_at
        interrupted_events = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "turn.interrupted"
        ]
        assert len(interrupted_events) == 1
        assert interrupted_events[0].metadata["control_command_id"] == (
            interrupt.command_id
        )
        assert interrupted_events[0].metadata["target_command_id"] == (first.command_id)
        control_proof = proof_snapshot(rig.store, rig.session.session_id)
        proof_control = next(
            command
            for command in control_proof["commands"]
            if command["command_id"] == interrupt.command_id
        )
        assert proof_control["result"]["target_command_id"] == first.command_id
        assert (
            proof_control["result"]["checkpoint_id"]
            == (interrupted_events[0].metadata["checkpoint_id"])
        )
        assert (
            proof_control["result"]["workspace_material_digest"]
            == (interrupted_events[0].metadata["workspace_material_digest"])
        )
        proof_interrupt = next(
            event
            for event in control_proof["events"]
            if event["event_type"] == "turn.interrupted"
        )
        assert proof_interrupt["metadata"] == interrupted_events[0].metadata
        anchor = rig.store.dispatch_transition_anchor(rig.session.session_id)
        assert anchor["prior_command_id"] == interrupt.command_id
        assert anchor["prior_anchor_kind"] == "control-command"

        transition = _advance_orchestration_generation(
            rig,
            interrupt.command_id,
            next_turn_ref,
            "control-anchor-next-message",
            policy,
            2,
            next_command_payload,
        )
        resumed = await rig.message(**next_command_payload)

        assert resumed.status == "complete"
        consumed = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "dispatch.generation.transition.consumed"
        ]
        assert len(consumed) == 2
        assert (
            consumed[1].metadata["invalidation_id"] == (transition["invalidation_id"])
        )
        assert consumed[1].metadata["command_id"] == resumed.command_id
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_noop_review_transition_allows_one_verifier_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_ref = {
        "orchestrator": "p13i/machines/agent-harness-proof-builder",
        "job_id": "builder-proof-noop-review",
    }
    rig = JourneyRig(tmp_path, external_ref=external_ref)
    next_turn_ref = {"step_id": "verify", "agent_role": "verifier"}
    next_command_payload = {
        "text": "Verify the reviewed implementation.",
        "provider": "codex",
        "turn_ref": next_turn_ref,
    }
    policy = _install_transition_policy(
        rig,
        allowed_agent_roles=["verifier"],
        allowed_step_prefixes=["verify"],
        transitions=[
            {
                "sequence": 1,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    next_command_payload,
                    "unattended",
                ),
            }
        ],
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    workspace = Path(rig.session.worktree)
    (workspace / "AGENTS.md").write_text("stable guidance\n", encoding="utf-8")
    (workspace / "plans").mkdir()
    (workspace / "plans" / "proof.gpt.md").write_text(
        "stable plan\n",
        encoding="utf-8",
    )
    rig.prime_capacity()
    try:
        reviewed = await rig.message(
            "Review the completed implementation without editing it.",
            provider="claude",
            turn_ref={"step_id": "review", "agent_role": "reviewer"},
        )
        transition = _advance_orchestration_generation(
            rig,
            reviewed.command_id,
            next_turn_ref,
            "review-to-verify",
            policy,
            1,
            next_command_payload,
        )
        forged_payloads = [
            {
                **next_command_payload,
                "text": "Run a different instruction under the verifier stage.",
            },
            {**next_command_payload, "effort": "medium"},
            {
                **next_command_payload,
                "proof_fault_probe": {
                    "provider": "codex",
                    "stage": "before-prompt",
                },
            },
        ]
        for forged_payload in forged_payloads:
            forged = await rig.message(**forged_payload)
            assert forged.status == "failed"
            assert forged.result["code"] == "E_SAFETY_GUARD"
            assert (
                rig.store.command_envelope(forged.command_id)["guard_reason"]
                == "dispatch-transition-command-mismatch"
            )
            rig.store.update_session(
                rig.session.session_id,
                lifecycle="running",
                attention="idle",
            )
        out_of_band = workspace / "out-of-band.txt"
        out_of_band.write_text("changed after authorization\n", encoding="utf-8")
        changed = await rig.message(**next_command_payload)
        assert changed.status == "failed"
        assert (
            rig.store.command_envelope(changed.command_id)["guard_reason"]
            == "dispatch-transition-material-mismatch"
        )
        out_of_band.unlink()
        rig.store.update_session(
            rig.session.session_id,
            lifecycle="running",
            attention="idle",
        )
        prompts_before_race = sum(
            len(adapter.prompts) for adapter in rig.adapters.values()
        )
        original_admission = rig.store.reserve_route_admission

        def mutate_before_admission(*args: Any, **kwargs: Any):
            out_of_band.write_text(
                "changed after transition reservation\n",
                encoding="utf-8",
            )
            return original_admission(*args, **kwargs)

        monkeypatch.setattr(
            rig.store,
            "reserve_route_admission",
            mutate_before_admission,
        )
        raced = await rig.message(**next_command_payload)
        assert raced.status == "failed"
        assert raced.result["code"] == "E_CONFLICT"
        assert (
            sum(len(adapter.prompts) for adapter in rig.adapters.values())
            == prompts_before_race
        )
        monkeypatch.setattr(
            rig.store,
            "reserve_route_admission",
            original_admission,
        )
        out_of_band.unlink()
        rig.store.update_session(
            rig.session.session_id,
            lifecycle="running",
            attention="idle",
        )
        verified = await rig.message(**next_command_payload)
        replay = await rig.message(
            "Run another verification without a new boundary.",
            provider="codex",
            turn_ref=next_turn_ref,
        )

        assert reviewed.status == "complete", reviewed.result
        assert transition["prior_command_id"] == reviewed.command_id
        assert verified.status == "complete", verified.result
        assert replay.status == "failed"
        assert replay.result["code"] == "E_SAFETY_GUARD"
        assert (
            rig.store.command_envelope(replay.command_id)["guard_reason"]
            == "dispatch-transition-already-consumed"
        )
        consumed = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "dispatch.generation.transition.consumed"
        ]
        assert len(consumed) == 1
        assert consumed[0].metadata["invalidation_id"] == transition["invalidation_id"]
        assert consumed[0].metadata["command_id"] == verified.command_id
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_terminal_guard_checkpoint_carries_one_review_stage(
    tmp_path: Path,
) -> None:
    external_ref = {
        "orchestrator": "p13i/machines/cs-builder",
        "job_id": "builder-terminal-checkpoint",
    }
    claude = ScriptedAdapter("claude", claims_cost_reporting=True)
    rig = JourneyRig(tmp_path, claude=claude, external_ref=external_ref)
    next_turn_ref = {"step_id": "review", "agent_role": "reviewer"}
    next_command_payload = {
        "text": "Review the terminal builder checkpoint.",
        "provider": "codex",
        "turn_ref": next_turn_ref,
    }
    policy = _install_transition_policy(
        rig,
        allowed_agent_roles=["reviewer"],
        allowed_step_prefixes=["review"],
        transitions=[
            {
                "sequence": 1,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    next_command_payload,
                    "unattended",
                ),
            }
        ],
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.scheduler._usage_cache = {
        "claude": _usage("claude", 20.0, credits=True),
        "codex": _usage("codex", 20.0),
    }
    rig.scheduler._usage_at = asyncio.get_running_loop().time()
    workspace = Path(rig.session.worktree)
    try:
        stopped = await rig.message(
            "Implement the stage under a metered envelope.",
            provider="claude",
            metered_budget=1,
            turn_ref={"step_id": "implement", "agent_role": "builder"},
        )

        assert stopped.status == "failed", stopped.result
        assert stopped.result["code"] == "E_SAFETY_GUARD"
        assert stopped.result["message"].endswith("dollar-accounting")
        assert stopped.result["provider_terminal"] is True
        certified = rig.store.checkpoints(rig.session.session_id)[-1]
        material_digest = inspect_workspace(workspace)[0]
        assert stopped.result["checkpoint_id"] == certified.checkpoint_id
        assert stopped.result["workspace_material_digest"] == material_digest
        assert not rig.store.pending_reconciliations(rig.session.session_id)
        guard_events = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "guard.tripped"
        ]
        assert len(guard_events) == 1
        assert guard_events[0].metadata["provider_terminal"] is True
        assert guard_events[0].metadata["checkpoint_id"] == (certified.checkpoint_id)
        anchor = rig.store.dispatch_transition_anchor(rig.session.session_id)
        assert anchor["eligible"] is True, anchor["reason"]
        assert anchor["prior_anchor_kind"] == "terminal-checkpoint"
        assert anchor["prior_command_id"] == stopped.command_id
        assert anchor["prior_command_type"] == "message"
        assert anchor["prior_checkpoint_id"] == certified.checkpoint_id
        assert anchor["prior_material_digest"] == material_digest
        assert anchor["prior_reconciliation_id"] == ""
        assert anchor["prior_reconciliation_resolution"] == ""
        assert proof_snapshot(rig.store, rig.session.session_id)[
            "transition_anchor"
        ] == anchor

        transition = _advance_orchestration_generation(
            rig,
            stopped.command_id,
            next_turn_ref,
            "terminal-checkpoint-to-review",
            policy,
            1,
            next_command_payload,
        )
        assert transition["prior_anchor_kind"] == "terminal-checkpoint"
        rig.store.update_session(
            rig.session.session_id,
            lifecycle="running",
            attention="idle",
        )
        reviewed = await rig.message(**next_command_payload)
        replay = await rig.message(
            "Review again without a new boundary.",
            provider="codex",
            turn_ref=next_turn_ref,
        )

        assert reviewed.status == "complete", reviewed.result
        assert replay.status == "failed"
        assert replay.result["code"] == "E_SAFETY_GUARD"
        assert (
            rig.store.command_envelope(replay.command_id)["guard_reason"]
            == "dispatch-transition-already-consumed"
        )
        consumed = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "dispatch.generation.transition.consumed"
        ]
        assert len(consumed) == 1
        assert consumed[0].metadata["invalidation_id"] == transition["invalidation_id"]
        assert consumed[0].metadata["command_id"] == reviewed.command_id
    finally:
        rig.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_status", ["failed", "cancelled"])
async def test_e2e_noncomplete_terminal_guard_never_anchors_a_transition(
    tmp_path: Path,
    provider_status: str,
) -> None:
    class NoncompleteGuardAdapter(ScriptedAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            pre_prompt_gate = values["pre_prompt_gate"]
            if pre_prompt_gate is not None:
                await pre_prompt_gate()
            event_handler = values["event_handler"]
            await event_handler(
                ProviderEvent(
                    "provider.prompt.accepted",
                    status="accepted",
                    native_session_id="claude-native-session",
                )
            )
            return ProviderResult(
                provider="claude",
                native_session_id="claude-native-session",
                native_turn_id="claude-turn",
                status=provider_status,
                usage={"total_tokens": 7},
            )

    external_ref = {
        "orchestrator": "p13i/machines/cs-builder",
        "job_id": "builder-noncomplete-guard",
    }
    root = tmp_path / provider_status
    root.mkdir()
    claude = NoncompleteGuardAdapter("claude", claims_cost_reporting=True)
    rig = JourneyRig(root, claude=claude, external_ref=external_ref)
    next_turn_ref = {"step_id": "review", "agent_role": "reviewer"}
    next_command_payload = {
        "text": "Review a stage that never certified a checkpoint.",
        "provider": "codex",
        "turn_ref": next_turn_ref,
    }
    _install_transition_policy(
        rig,
        allowed_agent_roles=["reviewer"],
        allowed_step_prefixes=["review"],
        transitions=[
            {
                "sequence": 1,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    next_command_payload,
                    "unattended",
                ),
            }
        ],
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.scheduler._usage_cache = {
        "claude": _usage("claude", 20.0, credits=True),
        "codex": _usage("codex", 20.0),
    }
    rig.scheduler._usage_at = asyncio.get_running_loop().time()
    try:
        stopped = await rig.message(
            "Implement the stage under a metered envelope.",
            provider="claude",
            metered_budget=1,
            turn_ref={"step_id": "implement", "agent_role": "builder"},
        )

        assert stopped.status == "failed", stopped.result
        assert stopped.result["code"] == "E_SAFETY_GUARD"
        assert "provider_terminal" not in stopped.result
        assert "checkpoint_id" not in stopped.result
        assert "workspace_material_digest" not in stopped.result
        guard_events = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "guard.tripped"
        ]
        assert len(guard_events) == 1
        assert "provider_terminal" not in guard_events[0].metadata
        assert "checkpoint_id" not in guard_events[0].metadata
        anchor = rig.store.dispatch_transition_anchor(rig.session.session_id)
        assert anchor["eligible"] is False
        assert anchor["reason"] == (
            "dispatch transition requires exactly one reconciliation"
        )
    finally:
        rig.close()


class TerminalUsageAdapter(ScriptedAdapter):
    """Reports the whole turn's usage on the terminal turn event."""

    async def run_turn(self, **values: Any) -> ProviderResult:
        pre_prompt_gate = values["pre_prompt_gate"]
        if pre_prompt_gate is not None:
            await pre_prompt_gate()
        event_handler = values["event_handler"]
        await event_handler(
            ProviderEvent(
                "provider.prompt.accepted",
                status="accepted",
                native_session_id="claude-native-session",
            )
        )
        await event_handler(
            ProviderEvent(
                "agent.message",
                text="claude implemented the stage",
                status="complete",
            )
        )
        usage = {
            "input_tokens": 110_000,
            "output_tokens": 10_000,
            "total_tokens": 120_000,
        }
        await event_handler(
            ProviderEvent(
                "turn.completed",
                status="complete",
                metadata={"usage": usage},
                native_session_id="claude-native-session",
            )
        )
        return ProviderResult(
            provider="claude",
            native_session_id="claude-native-session",
            native_turn_id="claude-turn",
            status="complete",
            usage=usage,
        )


@pytest.mark.asyncio
async def test_e2e_terminal_turn_usage_never_fails_a_completed_result(
    tmp_path: Path,
) -> None:
    claude = TerminalUsageAdapter("claude")
    rig = JourneyRig(tmp_path, claude=claude)
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    try:
        receipt = await rig.message(
            "Implement the stage and report usage at the end.",
            provider="claude",
            safety_limits={"max_total_tokens": 2_000},
        )

        assert receipt.status == "complete", receipt.result
        assert receipt.result["checkpoint_id"]
        # The turn the provider finished keeps its result, and the
        # overage that only its terminal event reported is still
        # charged to the command envelope.
        safety = receipt.result["safety"]
        assert safety["violation"] == "total-tokens"
        assert safety["consumption"]["total_tokens"] == 120_000
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["state"] == "complete"
        assert envelope["consumption"]["total_tokens"] == 120_000
        events = rig.store.all_events(rig.session.session_id)
        assert not [event for event in events if event.event_type == "guard.tripped"]
        assert not [event for event in events if event.event_type == "turn.failed"]
        assert not rig.store.guard_incidents(rig.session.session_id)
        assert not rig.store.pending_reconciliations(rig.session.session_id)
        assert claude.interruptions == 0
        assert len(rig.store.attempts(rig.session.session_id)) == 1
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_usage_before_the_terminal_turn_still_stops_the_turn(
    tmp_path: Path,
) -> None:
    class EarlyUsageAdapter(TerminalUsageAdapter):
        async def run_turn(self, **values: Any) -> ProviderResult:
            event_handler = values["event_handler"]
            await event_handler(
                ProviderEvent(
                    "provider.prompt.accepted",
                    status="accepted",
                    native_session_id="claude-native-session",
                )
            )
            await event_handler(
                ProviderEvent(
                    "usage.updated",
                    metadata={
                        "input_tokens": 110_000,
                        "output_tokens": 10_000,
                        "total_tokens": 120_000,
                    },
                )
            )
            await asyncio.sleep(0.25)
            return await super().run_turn(**values)

    claude = EarlyUsageAdapter("claude")
    rig = JourneyRig(tmp_path, claude=claude)
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    try:
        receipt = await rig.message(
            "Implement the stage and report usage early.",
            provider="claude",
            safety_limits={"max_total_tokens": 2_000},
        )

        assert receipt.status == "failed", receipt.result
        assert receipt.result["code"] == "E_SAFETY_GUARD"
        assert receipt.result["reconciliation_id"]
        assert claude.interruptions == 1
        incidents = rig.store.guard_incidents(rig.session.session_id)
        assert incidents[0]["reason"] == "total-tokens"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_compact_followup_transition_consumes_exact_second_stage(
    tmp_path: Path,
) -> None:
    external_ref = {
        "orchestrator": "p13i/machines/cs-sre",
        "job_id": "compact-followup",
    }
    rig = JourneyRig(tmp_path, external_ref=external_ref)
    verify_ref = {"step_id": "verify", "agent_role": "sre"}
    publish_ref = {"step_id": "publish", "agent_role": "sre"}
    verify_payload = {
        "text": "Verify the reviewed implementation.",
        "provider": "codex",
        "turn_ref": verify_ref,
    }
    publish_payload = {
        "text": "Publish the verified implementation.",
        "provider": "codex",
        "turn_ref": publish_ref,
    }
    policy = _install_transition_policy(
        rig,
        allowed_agent_roles=["sre"],
        allowed_step_prefixes=["verify", "publish"],
        transitions=[
            {
                "sequence": 1,
                "next_turn_ref": verify_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    verify_payload,
                    "unattended",
                ),
            },
            {
                "sequence": 2,
                "next_turn_ref": publish_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    publish_payload,
                    "unattended",
                ),
            },
        ],
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    try:
        reviewed = await rig.message(
            "Review the implementation.",
            provider="claude",
            turn_ref={"step_id": "review", "agent_role": "reviewer"},
        )
        first_transition = _advance_orchestration_generation(
            rig,
            reviewed.command_id,
            verify_ref,
            "compact-review-to-verify",
            policy,
            1,
            verify_payload,
        )
        verified = await rig.message(**verify_payload)
        second_transition = _advance_orchestration_generation(
            rig,
            verified.command_id,
            publish_ref,
            "compact-verify-to-publish",
            policy,
            2,
            publish_payload,
        )
        published = await rig.message(**publish_payload)

        assert reviewed.status == "complete", reviewed.result
        assert verified.status == "complete", verified.result
        assert published.status == "complete", published.result
        assert first_transition["transition_sequence"] == 1
        assert second_transition["transition_sequence"] == 2
        proof = proof_snapshot(rig.store, rig.session.session_id)
        ledger = proof["dispatch_transition_ledger"]
        assert ledger["complete"] is True
        assert ledger["policy_count"] == 1
        assert [item["state"] for item in ledger["receipts"]] == [
            "consumed",
            "consumed",
        ]
        with rig.store._lock:
            retained = rig.store._connection.execute(
                """
                SELECT payload_json FROM authorization_receipts
                WHERE schema = ? ORDER BY created_at, operation_id
                """,
                ("p13i/agent-harness/dispatch-generation-transition-authorization/v1",),
            ).fetchall()
        retained_payloads = [json.loads(str(row["payload_json"])) for row in retained]
        assert len(retained_payloads) == 2
        assert all("policy" not in item for item in retained_payloads)
        assert retained_payloads[1]["policy_ref"]["policy_sha256"] == (
            normalized_digest(policy)
        )
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_transition_rebinds_after_preboundary_route_failure(
    tmp_path: Path,
) -> None:
    external_ref = {
        "orchestrator": "p13i/machines/agent-harness-proof-builder",
        "job_id": "builder-proof-route-retry",
    }
    rig = JourneyRig(tmp_path, external_ref=external_ref)
    next_turn_ref = {"step_id": "verify", "agent_role": "verifier"}
    next_command_payload = {
        "text": "Verify the reviewed implementation.",
        "provider": "codex",
        "turn_ref": next_turn_ref,
    }
    policy = _install_transition_policy(
        rig,
        allowed_agent_roles=["verifier"],
        allowed_step_prefixes=["verify"],
        transitions=[
            {
                "sequence": 1,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    next_command_payload,
                    "unattended",
                ),
            }
        ],
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    try:
        reviewed = await rig.message(
            "Review the completed implementation.",
            provider="claude",
            turn_ref={"step_id": "review", "agent_role": "reviewer"},
        )
        _advance_orchestration_generation(
            rig,
            reviewed.command_id,
            next_turn_ref,
            "review-to-verify-route-retry",
            policy,
            1,
            next_command_payload,
        )
        rig.prime_capacity(claude=95.0, codex=95.0)
        unavailable = await rig.message(**next_command_payload)
        assert unavailable.status == "failed"
        assert unavailable.result["code"] == "E_PROVIDER_UNAVAILABLE"

        rig.store.update_session(
            rig.session.session_id,
            lifecycle="running",
            attention="idle",
        )
        rig.prime_capacity()
        retried = await rig.message(**next_command_payload)

        assert retried.status == "complete", retried.result
        transition_events = [
            event.event_type
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type.startswith("dispatch.generation.transition.")
        ]
        assert transition_events == [
            "dispatch.generation.transition.reserved",
            "dispatch.generation.transition.released",
            "dispatch.generation.transition.reserved",
            "dispatch.generation.transition.consumed",
        ]
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_exact_sre_tick_repeats_only_after_bound_transition(
    tmp_path: Path,
) -> None:
    external_ref = {
        "orchestrator": "p13i/machines/agent-harness-proof",
        "job_id": "cs-sre-stable-ticks",
    }
    rig = JourneyRig(tmp_path, external_ref=external_ref)
    next_turn_ref = {"step_id": "tick-2", "agent_role": "cs-sre"}
    next_command_payload = {
        "text": "sre()",
        "provider": "codex",
        "turn_ref": next_turn_ref,
    }
    policy = _install_transition_policy(
        rig,
        allowed_agent_roles=["cs-sre"],
        allowed_step_prefixes=["tick-"],
        transitions=[
            {
                "sequence": 1,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    next_command_payload,
                    "unattended",
                ),
            }
        ],
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity()
    try:
        first = await rig.message(
            "sre()",
            provider="codex",
            turn_ref={"step_id": "tick-1", "agent_role": "cs-sre"},
        )
        _advance_orchestration_generation(
            rig,
            first.command_id,
            next_turn_ref,
            "sre-tick-1-to-2",
            policy,
            1,
            next_command_payload,
        )
        second = await rig.message(**next_command_payload)

        assert first.status == "complete", first.result
        assert second.status == "complete", second.result
        assert len(rig.adapters["codex"].prompts) == 2
    finally:
        rig.close()


@pytest.mark.parametrize(
    ("component", "expected_reason"),
    (
        ("instruction", "repeated-instruction"),
        ("context", "repeated-context"),
        ("workspace_instruction", "repeated-workspace_instruction"),
        ("plan", "repeated-plan"),
        ("skill", "repeated-skill"),
    ),
)
def test_distinct_managed_steps_allow_each_repeated_component(
    tmp_path: Path,
    component: str,
    expected_reason: str,
) -> None:
    rig = JourneyRig(tmp_path)
    workspace = Path(rig.session.worktree)
    if component == "workspace_instruction":
        (workspace / "AGENTS.md").write_text(
            "stable guidance\n",
            encoding="utf-8",
        )
    if component == "plan":
        (workspace / "plans").mkdir()
        (workspace / "plans" / "stable.gpt.md").write_text(
            "stable plan\n",
            encoding="utf-8",
        )
    if component == "skill":
        skill = workspace / ".agents" / "skills" / "stable"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "stable skill\n",
            encoding="utf-8",
        )
    first_instruction = "instruction-a"
    second_instruction = "instruction-b"
    first_context = "context-a"
    second_context = "context-b"
    if component == "instruction":
        second_instruction = first_instruction
    if component == "context":
        second_context = first_context
    first_turn_ref = {"step_id": "stage-1", "agent_role": "builder"}
    second_turn_ref = {"step_id": "stage-2", "agent_role": "builder"}
    try:
        rig.worker._guard_repeated_dispatch(
            new_uuid(),
            first_instruction,
            first_context,
            workspace,
            first_turn_ref,
            "unattended",
        )
        rig.worker._guard_repeated_dispatch(
            new_uuid(),
            second_instruction,
            second_context,
            workspace,
            second_turn_ref,
            "unattended",
        )
        assert expected_reason not in {
            event.metadata.get("reason", "")
            for event in rig.store.all_events(rig.session.session_id)
        }
    finally:
        rig.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ("xhigh", "max"))
async def test_e2e_high_tier_requires_and_consumes_one_authorization(
    tmp_path: Path,
    effort: str,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    rig.prime_capacity()
    try:
        waiting = await rig.message(
            "Use maximum effort.",
            provider="claude",
            effort=effort,
        )
        assert waiting.status == "awaiting-xhigh-authorization"

        rig.authorize_xhigh(waiting.command_id, "claude")
        await rig.execute(waiting.command_id)
        accepted = rig.store.get_command(waiting.command_id)

        assert accepted.status == "complete"
        claude = rig.adapters["claude"]
        assert claude.efforts == [effort]
        safety = rig.store.session_safety(rig.session.session_id)
        assert safety["xhigh_authorizations"] == 0
    finally:
        rig.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ("xhigh", "max"))
async def test_e2e_one_high_tier_authorization_never_funds_a_retry(
    tmp_path: Path,
    effort: str,
) -> None:
    claude = ScriptedAdapter("claude", fail_turns=1)
    codex = ScriptedAdapter("codex")
    rig = JourneyRig(tmp_path, claude=claude, codex=codex)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    rig.prime_capacity()
    try:
        receipt = rig.enqueue_message(
            "Use only one high-tier attempt.",
            effort=effort,
        )
        rig.authorize_xhigh(receipt.command_id, "claude")
        await rig.execute(receipt.command_id)
        receipt = rig.store.get_command(receipt.command_id)

        assert receipt.status == "failed"
        assert claude.efforts == [effort]
        assert codex.efforts == []
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["limits"]["max_attempts"] == 1
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_provider_fault_barrier_precedes_prompt_and_fails_over_once(
    tmp_path: Path,
) -> None:
    external_ref = {
        "orchestrator": "p13i/machines/agent-harness-proof-builder",
        "job_id": "provider-fault-job",
    }
    idempotency_key = "provider-fault-message"
    authorization = {
        "schema": "p13i/machines/provider-fault-authorization/v1",
        "external_ref": external_ref,
        "idempotency_key": idempotency_key,
        "stage": "after-lease-before-acceptance",
        "provider": "codex",
        "agent_role": "proof-fault-probe",
    }
    authorization_digest = normalized_digest(authorization)
    codex = ScriptedAdapter("codex")
    codex.process_running = True
    claude = ScriptedAdapter("claude")
    rig = JourneyRig(
        tmp_path,
        claude=claude,
        codex=codex,
        external_ref=external_ref,
        goal_constraints=("proof-fault-authorization-sha256:" + authorization_digest,),
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity(claude=20.0, codex=40.0)
    receipt = rig.enqueue_message(
        "Exercise one externally injected provider fault.",
        idempotency_key=idempotency_key,
        required_capabilities=["proof-fault-barrier"],
        turn_ref={
            "step_id": "provider-fault-stage",
            "agent_role": "proof-fault-probe",
        },
        proof_fault_probe={
            "stage": "after-lease-before-acceptance",
            "provider": "codex",
            "authorization": authorization,
            "authorization_digest": authorization_digest,
        },
    )
    execute = asyncio.create_task(rig.execute(receipt.command_id))
    try:
        ready = None
        for unused_attempt in range(1_000):
            del unused_attempt
            ready_events = [
                event
                for event in rig.store.all_events(rig.session.session_id)
                if event.event_type == "proof.fault.ready"
            ]
            if ready_events:
                ready = ready_events[0]
                break
            await asyncio.sleep(0.01)
        assert ready is not None, {
            "command": rig.store.command_envelope(receipt.command_id),
            "events": [
                event.event_type
                for event in rig.store.all_events(rig.session.session_id)
            ],
        }
        assert ready.status == "waiting"
        assert ready.metadata["provider"] == "codex"
        assert ready.metadata["pid"] == codex.process_pid
        assert ready.metadata["pid_start"] == codex.process_start
        assert codex.prompts == []
        assert not any(
            event.event_type == "provider.prompt.accepted"
            for event in rig.store.all_events(rig.session.session_id)
        )
        assert rig.store.command_envelope(receipt.command_id)["state"] == (
            "fault-ready"
        )

        codex.process_running = False
        await asyncio.wait_for(execute, timeout=5)
        completed = rig.store.get_command(receipt.command_id)
        assert completed.status == "complete"
        assert codex.prompts == []
        assert len(claude.prompts) == 1
        attempts = rig.store.attempts(rig.session.session_id)
        assert [attempt.provider for attempt in attempts] == ["codex", "claude"]
        leases = rig.store.process_leases(rig.session.session_id)
        assert len(leases) == 2
        assert leases[0]["state"] == "released"
        assert leases[0]["command_id"] == receipt.command_id
        assert leases[0]["attempt_id"] == attempts[0].attempt_id
        observed = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "proof.fault.observed"
        ]
        assert len(observed) == 1
        assert observed[0].metadata["termination"] == ("external-process-termination")
    finally:
        if not execute.done():
            execute.cancel()
            await asyncio.gather(execute, return_exceptions=True)
        rig.close()


@pytest.mark.asyncio
async def test_e2e_service_fault_barrier_precedes_restart_reconciliation(
    tmp_path: Path,
) -> None:
    external_ref = {
        "orchestrator": "p13i/machines/agent-harness-proof-builder",
        "job_id": "service-fault-job",
    }
    idempotency_key = "service-fault-message"
    authorization = {
        "schema": "p13i/machines/service-fault-authorization/v1",
        "external_ref": external_ref,
        "idempotency_key": idempotency_key,
        "stage": "after-acceptance-before-terminal",
        "provider": "codex",
        "agent_role": "proof-service-fault-probe",
    }
    authorization_digest = normalized_digest(authorization)
    codex = ScriptedAdapter("codex")
    codex.process_running = True
    rig = JourneyRig(
        tmp_path,
        codex=codex,
        external_ref=external_ref,
    )
    next_turn_ref = {
        "step_id": "post-service-reconciliation",
        "agent_role": "sre",
    }
    next_command_payload = {
        "text": "Continue after the exact service-fault reconciliation.",
        "provider": "codex",
        "turn_ref": next_turn_ref,
    }
    policy = {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": rig.session.session_id,
        "external_ref": external_ref,
        "epoch_id": "service-fault-reconciliation-epoch",
        "allowed_agent_roles": ["sre"],
        "allowed_step_prefixes": ["post-service-reconciliation"],
        "max_transitions": 1,
        "transitions": [
            {
                "sequence": 1,
                "next_turn_ref": next_turn_ref,
                "next_command_digest": command_envelope_digest(
                    "message",
                    next_command_payload,
                    "unattended",
                ),
            }
        ],
    }
    rig.store.create_goal(
        create_goal(
            rig.session.session_id,
            "Recover and continue one exact service-fault stage.",
            constraints=(
                "proof-service-fault-authorization-sha256:" + authorization_digest,
                "dispatch-generation-transition-policy-sha256:"
                + normalized_digest(policy),
                "dispatch-generation-transition-epoch:"
                "service-fault-reconciliation-epoch",
            ),
            permitted_providers=("claude", "codex"),
            permitted_efforts=("low", "medium", "high"),
        )
    )
    rig.store.set_session_safety(rig.session.session_id, "unattended")
    rig.prime_capacity(claude=20.0, codex=40.0)
    receipt = rig.enqueue_message(
        "Hold after acceptance for one service restart.",
        idempotency_key=idempotency_key,
        required_capabilities=["proof-service-fault-barrier"],
        turn_ref={
            "step_id": "service-fault-stage",
            "agent_role": "proof-service-fault-probe",
        },
        proof_service_fault_probe={
            "stage": "after-acceptance-before-terminal",
            "provider": "codex",
            "authorization": authorization,
            "authorization_digest": authorization_digest,
        },
    )
    execute = asyncio.create_task(rig.execute(receipt.command_id))
    try:
        ready = None
        for unused_attempt in range(1_000):
            del unused_attempt
            events = rig.store.all_events(rig.session.session_id)
            ready_events = [
                event
                for event in events
                if event.event_type == "proof.service-fault.ready"
            ]
            if ready_events:
                ready = ready_events[0]
                break
            await asyncio.sleep(0.01)
        assert ready is not None, {
            "command": rig.store.command_envelope(receipt.command_id),
            "events": [event.event_type for event in events],
        }
        accepted = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "provider.prompt.accepted"
        ]
        assert len(accepted) == 1
        assert accepted[0].sequence < ready.sequence
        assert ready.metadata["provider"] == "codex"
        assert ready.metadata["context_digest"]

        replacement = SessionWorker(
            rig.store,
            rig.blobs,
            rig.scheduler,
            rig.adapters,
            rig.session.session_id,
        )
        rig.store.register_worker(
            rig.session.session_id,
            456,
            replacement.incarnation,
        )
        manager = ReconciliationManager(
            rig.store,
            rig.blobs,
        )
        recovery = await manager.recover_after_restart(rig.session.session_id)
        assert len(recovery.reconciliations) == 1
        reconciliation = recovery.reconciliations[0]
        assert reconciliation.command_id == receipt.command_id

        with pytest.raises(WorkerOwnershipLostError):
            await asyncio.wait_for(execute, timeout=7)
        assert len(rig.store.attempts(rig.session.session_id)) == 1
        original_attempt = rig.store.attempts(rig.session.session_id)[0]
        interrupted = rig.store.get_command(receipt.command_id)
        assert interrupted.status == "failed"
        assert interrupted.result["code"] == "E_NEEDS_RECONCILIATION"
        assert len(rig.store.pending_reconciliations(rig.session.session_id)) == 1
        original_leases = rig.store.process_leases(rig.session.session_id)
        assert len(original_leases) == 1
        assert original_leases[0]["state"] == "active"
        assert original_leases[0]["attempt_id"] == original_attempt.attempt_id
        assert original_leases[0]["worker_incarnation"] == rig.worker.incarnation
        resolved = await manager.resolve(
            reconciliation.reconciliation_id,
            ReconciliationDecision.ACCEPT_CURRENT,
            reconciliation.current_workspace_digest,
            audit={"actor": "journey"},
        )
        assert resolved.status == "resolved"
        resolution_workspace_digest = inspect_workspace(Path(rig.session.worktree))[0]
        assert resolved.audit["resolution_workspace_digest"] == (
            resolution_workspace_digest
        )
        proof_reconciliation = proof_snapshot(
            rig.store,
            rig.session.session_id,
        )["reconciliations"][0]
        assert proof_reconciliation["resolution_workspace_digest"] == (
            resolution_workspace_digest
        )
        original_audit = dict(resolved.audit)
        tampered_audit = dict(original_audit)
        tampered_audit["resolution_workspace_digest"] = "0" * 64
        with rig.store.transaction() as connection:
            connection.execute(
                """
                UPDATE reconciliations SET audit_json = ?
                WHERE reconciliation_id = ?
                """,
                (
                    json.dumps(
                        tampered_audit,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    resolved.reconciliation_id,
                ),
            )
        tampered_anchor = rig.store.dispatch_transition_anchor(rig.session.session_id)
        assert tampered_anchor["eligible"] is False
        assert tampered_anchor["reason"] == (
            "dispatch transition reconciliation material is not current"
        )
        with rig.store.transaction() as connection:
            connection.execute(
                """
                UPDATE reconciliations SET audit_json = ?
                WHERE reconciliation_id = ?
                """,
                (
                    json.dumps(
                        original_audit,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    resolved.reconciliation_id,
                ),
            )
        topology = resolved.audit["topology_receipt"]
        assert topology["schema"] == (
            "p13i/agent-harness/reconciliation-topology-receipt/v1"
        )
        assert topology["command_id"] == receipt.command_id
        assert topology["attempt_id"] == original_attempt.attempt_id
        assert topology["attempt_state"] == "ambiguous"
        assert topology["turn_state"] == "ambiguous"
        assert topology["dispatch_state"] == "ambiguous"
        assert topology["envelope_state"] == "paused"
        assert topology["guard_reason"] == "ambiguous-provider-dispatch"
        assert len(topology["leases"]) == 1
        assert topology["leases"][0]["prior_state"] == "active"
        assert topology["leases"][0]["state"] == "released"
        assert topology["leases"][0]["worker_incarnation"] == (rig.worker.incarnation)
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["state"] == "paused"
        assert envelope["guard_reason"] == "ambiguous-provider-dispatch"
        normalized_attempt = rig.store.attempts(rig.session.session_id)[0]
        assert normalized_attempt.status == "ambiguous"
        topology_rows = rig.store._connection.execute(
            """
            SELECT turns.status AS turn_state,
                command_dispatches.state AS dispatch_state
            FROM turns JOIN command_dispatches USING(turn_id)
            WHERE command_dispatches.attempt_id = ?
            """,
            (original_attempt.attempt_id,),
        ).fetchone()
        assert topology_rows is not None
        assert topology_rows["turn_state"] == "ambiguous"
        assert topology_rows["dispatch_state"] == "ambiguous"
        released = rig.store.process_leases(rig.session.session_id)
        assert len(released) == 1
        assert released[0]["state"] == "released"
        assert rig.store.active_process_leases() == []
        anchor = rig.store.dispatch_transition_anchor(rig.session.session_id)
        assert anchor["eligible"] is True
        assert anchor["prior_anchor_kind"] == "resolved-reconciliation"
        transition = _advance_orchestration_generation(
            rig,
            receipt.command_id,
            next_turn_ref,
            "service-fault-reconciled-next-message",
            policy,
            1,
            next_command_payload,
        )
        rig.worker = replacement
        continued = await rig.message(**next_command_payload)
        assert continued.status == "complete"
        consumed = [
            event
            for event in rig.store.all_events(rig.session.session_id)
            if event.event_type == "dispatch.generation.transition.consumed"
        ]
        assert len(consumed) == 1
        assert (
            consumed[0].metadata["invalidation_id"] == (transition["invalidation_id"])
        )
        assert consumed[0].metadata["command_id"] == continued.command_id
    finally:
        if not execute.done():
            execute.cancel()
            await asyncio.gather(execute, return_exceptions=True)
        rig.close()


@pytest.mark.asyncio
async def test_e2e_reconciliation_restores_pre_dispatch_checkpoint(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    workspace = Path(rig.session.worktree)
    (workspace / "pre-turn-target.txt").write_text(
        "pre-turn material\n",
        encoding="utf-8",
    )
    (workspace / "pre-turn-link").symlink_to("pre-turn-target.txt")
    script = workspace / "pre-turn-script.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o644)
    pre_turn_workspace_digest = inspect_workspace(workspace)[0]

    def make_script_executable(unused_workspace: Path) -> None:
        del unused_workspace
        script.chmod(0o755)

    manager, record, command = await _ambiguous_reconciliation(
        rig,
        make_script_executable,
    )
    assert script.stat().st_mode & 0o111
    assert record.current_workspace_digest != pre_turn_workspace_digest
    queued = rig.store.enqueue_command(
        rig.session.session_id,
        "message",
        {"text": "wait behind the barrier"},
        "queued-after-ambiguous",
    )
    try:
        assert manager.inspect(record.reconciliation_id) == record
        assert rig.store.claim_command(rig.session.session_id) is None
        with pytest.raises(ConflictError):
            await manager.resolve(
                record.reconciliation_id,
                ReconciliationDecision.RESTORE_PRE_TURN,
                "stale-digest",
                approved=True,
            )
        with pytest.raises(HarnessError) as approval:
            await manager.resolve(
                record.reconciliation_id,
                ReconciliationDecision.RESTORE_PRE_TURN,
                record.current_workspace_digest,
            )
        assert approval.value.detail.code == "E_APPROVAL_REQUIRED"

        resolved = await manager.resolve(
            record.reconciliation_id,
            ReconciliationDecision.RESTORE_PRE_TURN,
            record.current_workspace_digest,
            audit={"actor": "journey"},
            approved=True,
        )
        replay = await manager.resolve(
            record.reconciliation_id,
            ReconciliationDecision.RESTORE_PRE_TURN,
            record.current_workspace_digest,
            approved=True,
        )

        assert resolved == replay
        assert resolved.status == "resolved"
        assert resolved.resolution == "restore-pre-turn"
        assert "checkpoint_id" in resolved.audit
        resolution_workspace_digest = inspect_workspace(Path(rig.session.worktree))[0]
        assert (
            resolved.audit["resolution_workspace_digest"] == resolution_workspace_digest
        )
        proof = proof_snapshot(rig.store, rig.session.session_id)
        proof_reconciliation = proof["reconciliations"][0]
        assert proof_reconciliation["resolution_workspace_digest"] == (
            resolution_workspace_digest
        )
        assert proof_reconciliation["resolution_workspace_digest_valid"] is True
        assert proof_reconciliation["resolution_material_certified"] is True
        assert resolution_workspace_digest == pre_turn_workspace_digest
        assert (workspace / "pre-turn-link").is_symlink()
        assert os.readlink(workspace / "pre-turn-link") == ("pre-turn-target.txt")
        assert script.stat().st_mode & 0o111 == 0
        assert (Path(rig.session.worktree) / "file.txt").read_text(
            encoding="utf-8"
        ) == "base\n"
        assert (
            rig.store.get_command(command.command_id).result["code"]
            == "E_NEEDS_RECONCILIATION"
        )
        claimed = rig.store.claim_command(rig.session.session_id)
        assert claimed is not None
        assert claimed.command_id == queued.command_id
        tampered_audit = dict(resolved.audit)
        tampered_audit["resolution_workspace_digest"] = "tampered"
        with rig.store.transaction() as connection:
            connection.execute(
                """
                UPDATE reconciliations SET audit_json = ?
                WHERE reconciliation_id = ?
                """,
                (
                    json.dumps(
                        tampered_audit,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    resolved.reconciliation_id,
                ),
            )
        with pytest.raises(ConflictError, match="certified workspace material"):
            await manager.resolve(
                record.reconciliation_id,
                ReconciliationDecision.RESTORE_PRE_TURN,
                record.current_workspace_digest,
                approved=True,
            )
        tampered_proof = proof_snapshot(rig.store, rig.session.session_id)
        tampered_reconciliation = tampered_proof["reconciliations"][0]
        assert tampered_reconciliation["resolution_workspace_digest_valid"] is False
        assert tampered_reconciliation["resolution_material_certified"] is False
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_reconciliation_accepts_current_or_stops(
    tmp_path: Path,
) -> None:
    accept_root = tmp_path / "accept"
    accept_root.mkdir()
    accept_rig = JourneyRig(accept_root)
    accept_script = Path(accept_rig.session.worktree) / "accept-script.sh"
    accept_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    accept_script.chmod(0o644)
    accept_before_digest = inspect_workspace(Path(accept_rig.session.worktree))[0]

    def accept_executable(unused_workspace: Path) -> None:
        del unused_workspace
        accept_script.chmod(0o755)

    accept_manager, accept_record, unused_command = await _ambiguous_reconciliation(
        accept_rig,
        accept_executable,
    )
    del unused_command
    try:
        accept_rig.store.begin_reconciliation_resolution(
            accept_record.reconciliation_id,
            ReconciliationDecision.ACCEPT_CURRENT,
            accept_record.current_workspace_digest,
        )
        accepted = await accept_manager.resolve(
            accept_record.reconciliation_id,
            ReconciliationDecision.ACCEPT_CURRENT,
            accept_record.current_workspace_digest,
        )
        assert accepted.resolution == "accept-current"
        assert accepted.current_workspace_digest != accept_before_digest
        assert accept_script.stat().st_mode & 0o111
        assert "checkpoint_id" in accepted.audit
        assert "discovery_checkpoint_id" in accepted.audit
        assert "resolution_checkpoint_id" in accepted.audit
        accepted_workspace_digest = inspect_workspace(
            Path(accept_rig.session.worktree)
        )[0]
        assert accepted.audit["resolution_workspace_digest"] == (
            accepted_workspace_digest
        )
        proof = proof_snapshot(
            accept_rig.store,
            accept_rig.session.session_id,
        )
        reconciliation = proof["reconciliations"][0]
        assert reconciliation["pre_dispatch_checkpoint_bound"] is True
        assert reconciliation["discovery_checkpoint_bound"] is True
        assert reconciliation["resolution_checkpoint_bound"] is True
        assert reconciliation["resolution_workspace_digest"] == (
            accepted_workspace_digest
        )
        assert reconciliation["resolution_workspace_digest_valid"] is True
        assert reconciliation["resolution_material_certified"] is True
        assert reconciliation["discovery_resolution_material_equal"] is True
        assert reconciliation["recovery_material_certified"] is True
        checkpoints = {item["checkpoint_id"]: item for item in proof["checkpoints"]}
        discovery = checkpoints[reconciliation["discovery_checkpoint_id"]]
        resolution = checkpoints[reconciliation["resolution_checkpoint_id"]]
        for field in (
            "base_commit",
            "patch_digest",
            "untracked_digest",
            "context_digest",
        ):
            assert discovery[field] == resolution[field]
        assert (Path(accept_rig.session.worktree) / "file.txt").read_text(
            encoding="utf-8"
        ) == "ambiguous effect\n"
        assert (
            accept_rig.store.get_session(accept_rig.session.session_id).attention
            == "idle"
        )
        with pytest.raises(ConflictError):
            await accept_manager.resolve(
                accept_record.reconciliation_id,
                ReconciliationDecision.STOP,
                accept_record.current_workspace_digest,
            )
        tampered_audit = dict(accepted.audit)
        tampered_audit["resolution_checkpoint_id"] = "missing-checkpoint"
        tampered_audit["resolution_workspace_digest"] = "tampered"
        with accept_rig.store.transaction() as connection:
            connection.execute(
                """
                UPDATE reconciliations SET audit_json = ?
                WHERE reconciliation_id = ?
                """,
                (
                    json.dumps(
                        tampered_audit,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    accepted.reconciliation_id,
                ),
            )
        tampered_proof = proof_snapshot(
            accept_rig.store,
            accept_rig.session.session_id,
        )
        tampered_reconciliation = tampered_proof["reconciliations"][0]
        assert tampered_reconciliation["resolution_checkpoint_bound"] is False
        assert tampered_reconciliation["resolution_workspace_digest_valid"] is False
        assert tampered_reconciliation["resolution_material_certified"] is False
        assert tampered_reconciliation["discovery_resolution_material_equal"] is False
        assert tampered_reconciliation["recovery_material_certified"] is False
        with pytest.raises(ConflictError, match="certified workspace material"):
            await accept_manager.resolve(
                accept_record.reconciliation_id,
                ReconciliationDecision.ACCEPT_CURRENT,
                accept_record.current_workspace_digest,
            )
    finally:
        accept_rig.close()

    stop_root = tmp_path / "stop"
    stop_root.mkdir()
    stop_rig = JourneyRig(stop_root)
    stop_manager, stop_record, unused_command = await _ambiguous_reconciliation(
        stop_rig
    )
    del unused_command
    try:
        stop_workspace = Path(stop_rig.session.worktree)
        (stop_workspace / "file.txt").write_text(
            "changed after inspection\n",
            encoding="utf-8",
        )
        (stop_workspace / "changed-after-inspection.txt").write_text(
            "retain this material\n",
            encoding="utf-8",
        )
        changed_digest = inspect_workspace(stop_workspace)[0]
        assert changed_digest != stop_record.current_workspace_digest
        stopped = await stop_manager.resolve(
            stop_record.reconciliation_id,
            ReconciliationDecision.STOP,
            stop_record.current_workspace_digest,
        )
        stopped_replay = await stop_manager.resolve(
            stop_record.reconciliation_id,
            ReconciliationDecision.STOP,
            stop_record.current_workspace_digest,
        )
        assert stopped_replay == stopped
        assert stopped.resolution == "stop"
        assert (
            stopped.current_workspace_digest == stop_record.current_workspace_digest
        )
        assert stopped.audit["observed_workspace_digest"] == (
            stop_record.current_workspace_digest
        )
        assert (
            stop_rig.store.get_session(stop_rig.session.session_id).lifecycle
            == "stopped"
        )
        assert inspect_workspace(stop_workspace)[0] == changed_digest
        assert (stop_workspace / "file.txt").read_text(encoding="utf-8") == (
            "changed after inspection\n"
        )
        assert (stop_workspace / "changed-after-inspection.txt").read_text(
            encoding="utf-8"
        ) == "retain this material\n"
        with pytest.raises(ValueError):
            await stop_manager.resolve(
                stop_record.reconciliation_id,
                "unsupported",
                stop_record.current_workspace_digest,
            )
    finally:
        stop_rig.close()


@pytest.mark.asyncio
async def test_e2e_reconciliation_rejects_changes_and_conflicting_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    changed_rig = JourneyRig(changed_root)
    manager, record, unused_command = await _ambiguous_reconciliation(changed_rig)
    del unused_command
    try:
        (Path(changed_rig.session.worktree) / "file.txt").write_text(
            "changed after inspection\n",
            encoding="utf-8",
        )
        with pytest.raises(ConflictError):
            await manager.resolve(
                record.reconciliation_id,
                ReconciliationDecision.ACCEPT_CURRENT,
                record.current_workspace_digest,
            )
    finally:
        changed_rig.close()

    race_root = tmp_path / "race"
    race_root.mkdir()
    race_rig = JourneyRig(race_root)
    manager, record, unused_command = await _ambiguous_reconciliation(race_rig)
    del unused_command
    race_rig.store.begin_reconciliation_resolution(
        record.reconciliation_id,
        ReconciliationDecision.ACCEPT_CURRENT,
        record.current_workspace_digest,
    )
    monkeypatch.setattr(
        reconciliation_module,
        "inspect_workspace",
        lambda unused: ("different-digest", "second"),
    )
    try:
        with pytest.raises(ConflictError, match="during"):
            await manager.resolve(
                record.reconciliation_id,
                ReconciliationDecision.ACCEPT_CURRENT,
                record.current_workspace_digest,
            )
    finally:
        race_rig.close()

    intent_root = tmp_path / "intent"
    intent_root.mkdir()
    intent_rig = JourneyRig(intent_root)
    manager, record, unused_command = await _ambiguous_reconciliation(intent_rig)
    del unused_command
    try:
        intent_rig.store.begin_reconciliation_resolution(
            record.reconciliation_id,
            ReconciliationDecision.STOP,
            record.current_workspace_digest,
        )
        with pytest.raises(ConflictError):
            await manager.resolve(
                record.reconciliation_id,
                ReconciliationDecision.ACCEPT_CURRENT,
                record.current_workspace_digest,
            )
    finally:
        intent_rig.close()


@pytest.mark.asyncio
async def test_e2e_reconciliation_rejects_read_only_restore(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.store.update_session(
        rig.session.session_id,
        permission_mode="read-only",
    )
    manager, record, unused_command = await _ambiguous_reconciliation(rig)
    del unused_command
    try:
        with pytest.raises(HarnessError) as denied:
            await manager.resolve(
                record.reconciliation_id,
                ReconciliationDecision.RESTORE_PRE_TURN,
                record.current_workspace_digest,
                approved=True,
            )
        assert denied.value.detail.code == "E_PERMISSION"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_special_object_effect_requires_reconciliation(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    fifo = Path(rig.session.worktree) / "provider-effect.fifo"

    def create_fifo(unused_workspace: Path) -> None:
        del unused_workspace
        os.mkfifo(fifo)

    manager, record, command = await _ambiguous_reconciliation(
        rig,
        create_fifo,
    )
    del manager
    try:
        assert fifo.exists()
        current_digest = inspect_workspace(Path(rig.session.worktree))[0]
        assert record.current_workspace_digest == current_digest
        failed = rig.store.get_command(command.command_id)
        assert failed.status == "failed"
        assert failed.result["code"] == "E_NEEDS_RECONCILIATION"
        queued = rig.store.enqueue_command(
            rig.session.session_id,
            "message",
            {"text": "must remain behind reconciliation"},
            "special-object-blocked-command",
        )
        assert rig.store.claim_command(rig.session.session_id) is None
        assert rig.store.get_command(queued.command_id).status == "queued"
    finally:
        rig.close()


def test_reconciliation_workspace_inspection_defenses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _repository(workspace)
    (workspace / "untracked.txt").write_text(
        "untracked\n",
        encoding="utf-8",
    )
    (workspace / "alternate.txt").write_text(
        "alternate\n",
        encoding="utf-8",
    )
    link = workspace / "untracked-link"
    without_link_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    link.symlink_to("untracked.txt")

    digest, summary = inspect_workspace(workspace)

    assert len(digest) == 64
    assert digest != without_link_digest
    assert '"commit"' in summary
    link.unlink()
    link.symlink_to("alternate.txt")
    retargeted_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    assert retargeted_digest != digest
    link.unlink()
    removed_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    assert removed_digest == without_link_digest
    script = workspace / "mode-effect.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o644)
    nonexecutable_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    script.chmod(0o755)
    executable_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    assert executable_digest != nonexecutable_digest
    fifo = workspace / "provider-effect.fifo"
    os.mkfifo(fifo)
    fifo_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    assert fifo_digest != executable_digest
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("outside\n", encoding="utf-8")
    (workspace / "outside-directory").symlink_to(outside)
    original_git = workspace_state_module._git

    def unsafe_paths(path: Path, *arguments: str) -> bytes:
        if arguments[0] == "ls-files":
            return (
                b"../escape\0missing\0provider-effect.fifo\0"
                b"outside-directory/file\0"
            )
        return original_git(path, *arguments)

    monkeypatch.setattr(workspace_state_module, "_git", unsafe_paths)
    second_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    assert len(second_digest) == 64

    monkeypatch.setattr(workspace_state_module, "_git", original_git)
    monkeypatch.setattr(
        reconciliation_module.subprocess,
        "run",
        lambda *unused_args, **unused_kwargs: SimpleNamespace(
            returncode=1,
            stdout=b"",
        ),
    )
    with pytest.raises(HarnessError):
        reconciliation_module._git(workspace, "status")
    reconciliation_module._require_restore_permission(
        "full",
        approved=False,
    )


def test_workspace_inspection_rejects_material_that_moves_underneath_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _repository(workspace)

    with pytest.raises(HarnessError, match="Git operation failed"):
        workspace_state_module._git(tmp_path / "not-a-repository", "status")

    with pytest.raises(HarnessError, match="changed during inspection"):
        workspace_module.special_workspace_objects(tmp_path / "absent")

    class UnstableEntry:
        def __init__(self, path: Path) -> None:
            self.name = path.name
            self.path = str(path)

        def stat(self, *, follow_symlinks: bool = True) -> object:
            del follow_symlinks
            raise OSError("workspace object disappeared")

    monkeypatch.setattr(
        workspace_state_module.os,
        "scandir",
        lambda directory: [UnstableEntry(Path(directory) / "vanished")],
    )
    with pytest.raises(HarnessError, match="changed during inspection"):
        workspace_module.special_workspace_objects(workspace)


@pytest.mark.asyncio
async def test_reconciliation_discovery_reuses_and_guards_its_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = JourneyRig(tmp_path)
    try:
        manager, record, unused_command = await _ambiguous_reconciliation(rig)
        del unused_command
        assert record.audit["discovery_checkpoint_id"]

        reused = await manager._ensure_discovery_checkpoint(record)

        assert reused.audit["discovery_checkpoint_id"] == (
            record.audit["discovery_checkpoint_id"]
        )
    finally:
        rig.close()

    for message, digests in (
        ("changed before reconciliation discovery", ["moved"]),
        ("changed during reconciliation discovery", [None, "moved"]),
    ):
        root = tmp_path / ("discovery-" + str(len(digests)))
        root.mkdir()
        discovery_rig = JourneyRig(root)
        try:
            manager, record, unused_command = await _ambiguous_reconciliation(
                discovery_rig
            )
            del unused_command
            stripped = replace(
                record,
                audit={
                    key: value
                    for key, value in record.audit.items()
                    if key != "discovery_checkpoint_id"
                },
            )
            original_inspect = reconciliation_module.inspect_workspace
            remaining = list(digests)

            def drifting_inspect(workspace: Path) -> tuple[str, str]:
                digest, summary = original_inspect(workspace)
                if remaining:
                    replacement = remaining.pop(0)
                    if replacement is not None:
                        return replacement, summary
                return digest, summary

            monkeypatch.setattr(
                reconciliation_module,
                "inspect_workspace",
                drifting_inspect,
            )
            with pytest.raises(ConflictError, match=message):
                await manager._ensure_discovery_checkpoint(stripped)
            monkeypatch.undo()
        finally:
            discovery_rig.close()


@pytest.mark.asyncio
async def test_reconciliation_resolution_rejects_material_that_moves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for decision, message, drift_index in (
        (
            ReconciliationDecision.ACCEPT_CURRENT,
            "during reconciliation resolution checkpoint",
            1,
        ),
        (
            ReconciliationDecision.RESTORE_PRE_TURN,
            "during restored resolution checkpoint",
            2,
        ),
    ):
        root = tmp_path / ("resolution-" + str(drift_index))
        root.mkdir()
        rig = JourneyRig(root)
        try:
            manager, record, unused_command = await _ambiguous_reconciliation(rig)
            del unused_command
            original_inspect = reconciliation_module.inspect_workspace
            observed = {"count": 0}

            def drifting_inspect(workspace: Path) -> tuple[str, str]:
                digest, summary = original_inspect(workspace)
                observed["count"] += 1
                if observed["count"] > drift_index:
                    return "moved-material-digest", summary
                return digest, summary

            monkeypatch.setattr(
                reconciliation_module,
                "inspect_workspace",
                drifting_inspect,
            )
            with pytest.raises(ConflictError, match=message):
                await manager.resolve(
                    record.reconciliation_id,
                    decision,
                    record.current_workspace_digest,
                    approved=True,
                )
            monkeypatch.undo()
        finally:
            rig.close()


async def _ambiguous_reconciliation(
    rig: JourneyRig,
    ambiguous_mutation: Callable[[Path], None] | None = None,
) -> tuple[ReconciliationManager, ReconciliationRecord, CommandReceipt]:
    command = rig.store.enqueue_command(
        rig.session.session_id,
        "message",
        {
            "text": "perform one effect",
            "turn_ref": {
                "step_id": "step-ambiguous",
                "agent_role": "implementer",
            },
        },
        new_uuid(),
    )
    assert rig.store.claim_command(rig.session.session_id) is not None
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=rig.session.session_id,
        provider="claude",
        native_session_id="claude-native",
        model="claude-default",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    rig.store.create_attempt(attempt)
    turn_id = rig.store.start_turn(
        rig.session.session_id,
        attempt.attempt_id,
        turn_ref=command.turn_ref,
    )
    checkpoint = checkpoint_workspace(
        rig.session,
        rig.blobs,
        sequence=rig.store.last_sequence(rig.session.session_id),
        provider="claude",
        native_session_id="claude-native",
        context_text="context",
    )
    rig.store.add_checkpoint(checkpoint)
    rig.store.record_dispatch_checkpoint(
        command.command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    rig.store.create_command_envelope(
        command.command_id,
        rig.session.session_id,
        "unattended",
        {"max_seconds": 900},
    )
    rig.store.update_command_envelope(
        command.command_id,
        consumption={"total_tokens": 73},
    )
    rig.store.mark_provider_boundary(attempt.attempt_id)
    workspace = Path(rig.session.worktree)
    before_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    (workspace / "file.txt").write_text(
        "ambiguous effect\n",
        encoding="utf-8",
    )
    (workspace / "untracked-effect.txt").write_text(
        "untracked effect\n",
        encoding="utf-8",
    )
    (workspace / "untracked-effect-link").symlink_to("untracked-effect.txt")
    if ambiguous_mutation is not None:
        ambiguous_mutation(workspace)
    after_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    assert before_digest != after_digest
    manager = ReconciliationManager(rig.store, rig.blobs)
    recovery = await manager.recover_after_restart(rig.session.session_id)
    assert len(recovery.reconciliations) == 1
    record = recovery.reconciliations[0]
    assert record.safety_consumption["total_tokens"] == 73
    return manager, record, command


async def _await_command(
    store: StateStore,
    command_id: str,
) -> CommandReceipt:
    for unused in range(1000):
        del unused
        receipt = store.get_command(command_id)
        if receipt.status in {"complete", "failed", "cancelled"}:
            return receipt
        await asyncio.sleep(0.01)
    raise AssertionError("command did not settle: " + command_id)


async def _wait_for_approval(
    reader: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any]:
    for unused in range(100):
        del unused
        approvals = reader()
        if approvals:
            return approvals[0]
        await asyncio.sleep(0.01)
    raise AssertionError("provider approval was not published")


def _usage(
    provider: str,
    binding: float | None,
    *,
    credits: bool = False,
) -> UsageSnapshot:
    return UsageSnapshot(
        provider=provider,
        binding_percent=binding,
        credits_engaged=credits,
        payload={},
    )


def _repository(path: Path) -> None:
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
async def test_zero_output_zero_change_turn_is_no_progress(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path, claude=ZeroProgressAdapter("claude"))
    try:
        rig.store.create_goal(
            create_goal(
                rig.session.session_id,
                "Survive a dead turn.",
                budgets={"turns": 1},
            )
        )
        rig.prime_capacity()
        receipt = await rig.message("Work silently.", provider="claude")

        assert receipt.status == "failed"
        assert receipt.result["code"] == "E_PROVIDER_NO_PROGRESS"
        turn_rows = rig.store.presentation_turn_rows(rig.session.session_id)
        assert [row["turn_status"] for row in turn_rows] == ["no-progress"]
        assert rig.store.turn_count(rig.session.session_id) == 1
        assert rig.store.countable_turn_count(rig.session.session_id) == 0
        assert rig.worker._exhausted_budget() == ""
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_output_without_workspace_change_completes_and_counts(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path, claude=ScriptedAdapter("claude"))
    try:
        rig.prime_capacity()
        receipt = await rig.message(
            "Decline with an explanation.",
            provider="claude",
        )

        assert receipt.status == "complete"
        turn_rows = rig.store.presentation_turn_rows(rig.session.session_id)
        assert [row["turn_status"] for row in turn_rows] == ["complete"]
        assert rig.store.countable_turn_count(rig.session.session_id) == 1
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_workspace_change_without_output_completes_and_counts(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(
        tmp_path,
        claude=ZeroProgressAdapter("claude", write_file=True),
    )
    try:
        rig.prime_capacity()
        receipt = await rig.message("Write without narration.", provider="claude")

        assert receipt.status == "complete"
        turn_rows = rig.store.presentation_turn_rows(rig.session.session_id)
        assert [row["turn_status"] for row in turn_rows] == ["complete"]
        assert rig.store.countable_turn_count(rig.session.session_id) == 1
    finally:
        rig.close()
