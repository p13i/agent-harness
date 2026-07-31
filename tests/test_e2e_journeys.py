"""End-to-end journeys through the provider-neutral execution core."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

import agent_harness.reconciliation as reconciliation_module
import agent_harness.safety as safety_module
import agent_harness.worker as worker_module
from agent_harness.blobs import BlobStore
from agent_harness.config import paths
from agent_harness.config import prepare_paths
from agent_harness.errors import ProviderExhaustedError
from agent_harness.errors import ConflictError
from agent_harness.errors import HarnessError
from agent_harness.goals import create_goal
from agent_harness.goals import make_evidence
from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import CommandReceipt
from agent_harness.models import ProviderAttempt
from agent_harness.models import ReconciliationDecision
from agent_harness.models import ReconciliationRecord
from agent_harness.providers.base import ApprovalHandler
from agent_harness.providers.base import EventHandler
from agent_harness.providers.base import ProviderAdapter
from agent_harness.providers.base import ProviderEvent
from agent_harness.providers.base import ProviderModel
from agent_harness.providers.base import ProviderResult
from agent_harness.providers.base import ProviderStatus
from agent_harness.scheduler import Scheduler
from agent_harness.reconciliation import ReconciliationManager
from agent_harness.reconciliation import inspect_workspace
from agent_harness.storage import StateStore
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
    ) -> None:
        self.provider_id = provider
        self.fail_turns = fail_turns
        self.guard_turns = guard_turns
        self.usage_turns = usage_turns
        self.request_approval = request_approval
        self.turn_delay = turn_delay
        self.prompts: list[str] = []
        self.native_inputs: list[str] = []
        self.efforts: list[str] = []
        self.approval_decisions: list[dict[str, Any]] = []
        self.steered: list[str] = []
        self.interruptions = 0

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
    ) -> ProviderResult:
        del workspace
        del permission_mode
        del model
        self.prompts.append(prompt)
        self.native_inputs.append(native_session_id)
        self.efforts.append(effort)
        if self.fail_turns:
            self.fail_turns -= 1
            raise ProviderExhaustedError(self.provider_id)
        if self.guard_turns:
            self.guard_turns -= 1
            for unused in range(3):
                del unused
                await event_handler(
                    ProviderEvent(
                        "tool.started",
                        text="Read",
                        metadata={"path": "SKILL.md"},
                        native_session_id=(
                            self.provider_id + "-native-session"
                        ),
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
        return ProviderResult(
            provider=self.provider_id,
            native_session_id=resolved_session_id,
            native_turn_id=self.provider_id + "-turn",
            status="complete",
            usage={"total_tokens": 42},
        )

    async def models(self, workspace: Path) -> tuple[ProviderModel, ...]:
        del workspace
        return (
            ProviderModel(
                self.provider_id + "-default",
                self.provider_id.title(),
                ("low", "medium", "high", "xhigh"),
                200_000,
                default=True,
            ),
        )

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.provider_id,
            ready=True,
            detail="scripted journey provider",
            capabilities=frozenset(
                {
                    "approval",
                    "checkpoint",
                    "hooks",
                    "mcp",
                    "plugins",
                    "resume",
                    "skills",
                    "streaming",
                    "subagents",
                    "tools",
                    "worktree",
                }
            ),
        )

    async def interrupt(self) -> None:
        self.interruptions += 1

    async def steer(self, text: str) -> None:
        self.steered.append(text)


class JourneyRig:
    def __init__(
        self,
        root: Path,
        *,
        claude: ScriptedAdapter | None = None,
        codex: ScriptedAdapter | None = None,
    ) -> None:
        workspace = root / "workspace"
        _repository(workspace)
        harness_paths = paths(root / "state")
        prepare_paths(harness_paths)
        self.store = StateStore(harness_paths.database)
        self.blobs = BlobStore(harness_paths.blobs)
        self.session = session(workspace)
        self.store.create_session(self.session)
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
        payload = {"text": text, **route}
        receipt = self.store.enqueue_command(
            self.session.session_id,
            "message",
            payload,
            new_uuid(),
        )
        self.store.append_event(
            self.session.session_id,
            "user.message",
            role="user",
            text=text,
            status="accepted",
            metadata={"command_id": receipt.command_id},
        )
        claimed = self.store.claim_command(self.session.session_id)
        assert claimed is not None
        await self.worker._message(claimed)
        return self.store.get_command(receipt.command_id)

    def close(self) -> None:
        self.store.close()


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
        for unused in range(100):
            del unused
            current = rig.store.get_command(resumed.command_id)
            if current.status == "complete":
                break
            await asyncio.sleep(0.01)
        assert current.status == "complete"
        assert (
            rig.store.get_session(rig.session.session_id).lifecycle
            == "running"
        )

        stopped = rig.store.enqueue_command(
            rig.session.session_id,
            "stop",
            {},
            new_uuid(),
        )
        await asyncio.wait_for(worker_task, timeout=2)

        assert rig.store.get_command(stopped.command_id).status == "complete"
        assert (
            rig.store.get_session(rig.session.session_id).lifecycle
            == "stopped"
        )
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

        codex = rig.adapters["codex"]
        claude = rig.adapters["claude"]
        assert first.status == "complete"
        assert second.status == "complete"
        assert third.status == "complete"
        assert codex.native_inputs == ["", "codex-native-session"]
        assert "UNIQUE_PROVIDER_NATIVE_INSTRUCTION" not in codex.prompts[0]
        assert codex.prompts[1] == "Now add tests."
        assert claude.native_inputs == [""]
        assert "# Harness session" in claude.prompts[0]
        assert "Implement the parser." in claude.prompts[0]
        assert "codex completed the turn" in claude.prompts[0]
        assert "# Next instruction" in claude.prompts[0]
        assert "UNIQUE_PROVIDER_NATIVE_INSTRUCTION" not in claude.prompts[0]
        assert len(rig.store.checkpoints(rig.session.session_id)) == 6
        dispatches = rig.store._connection.execute(
            """
            SELECT crossed_boundary, state, checkpoint_id
            FROM command_dispatches ORDER BY created_at
            """
        ).fetchall()
        assert len(dispatches) == 3
        assert all(row["crossed_boundary"] == 1 for row in dispatches)
        assert all(row["state"] == "complete" for row in dispatches)
    finally:
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

        rig.worker._active_adapter = None
        steer = await control("steer", {"text": "change direction"})
        assert steer.status == "failed"
        assert steer.result["code"] == "E_NO_ACTIVE_TURN"

        rig.worker._active_adapter = adapter
        assert (await control("steer", {"text": "new plan"})).status == (
            "complete"
        )
        assert adapter.steered == ["new plan"]

        assert (await control("pause")).status == "complete"
        assert (
            rig.store.get_session(rig.session.session_id).lifecycle
            == "paused"
        )
        assert (await control("resume")).status == "complete"
        assert (
            rig.store.get_session(rig.session.session_id).lifecycle
            == "running"
        )

        rig.worker._active_adapter = adapter
        assert (await control("stop")).status == "complete"
        assert adapter.interruptions == 2
        assert rig.worker._stopping
    finally:
        rig.close()


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
        assert (
            rig.store.get_session(rig.session.session_id).attention
            == "working"
        )
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
        assert projected.blob_digest
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
async def test_idle_worker_does_not_reorder_session_history(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    before = rig.store.get_session(rig.session.session_id).updated_at
    worker_task = asyncio.create_task(rig.worker.run())
    try:
        await asyncio.sleep(0.35)
        first_idle = rig.store.get_session(
            rig.session.session_id
        ).updated_at
        await asyncio.sleep(0.35)
        second_idle = rig.store.get_session(
            rig.session.session_id
        ).updated_at

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
        assert any(
            item.event_type == "routing.failover" for item in events
        )
        current = rig.store.get_session(rig.session.session_id)
        assert current.active_provider == "codex"
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
        turn = asyncio.create_task(
            rig.message("Run the validation.", provider="codex")
        )
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
        assert claude.interruptions == 1
        assert len(rig.store.attempts(rig.session.session_id)) == 1
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["state"] == "paused"
        assert envelope["guard_reason"] == "output-tokens"
        incidents = rig.store.guard_incidents(rig.session.session_id)
        assert incidents[0]["action"] == "pause"
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_repetition_downgrades_then_fails_over_in_one_envelope(
    tmp_path: Path,
) -> None:
    claude = ScriptedAdapter("claude", guard_turns=2)
    codex = ScriptedAdapter("codex")
    rig = JourneyRig(tmp_path, claude=claude, codex=codex)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    rig.prime_capacity()
    try:
        receipt = await rig.message("Implement without rereading context.")

        assert receipt.status == "complete", receipt.result
        attempts = rig.store.attempts(rig.session.session_id)
        assert [item.provider for item in attempts] == [
            "claude",
            "claude",
            "codex",
        ]
        assert claude.efforts == ["high", "medium"]
        assert codex.efforts == ["low"]
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["recovery_stage"] == 2
        assert envelope["consumption"]["attempts"] == 3
        assert len(rig.store.guard_incidents(rig.session.session_id)) == 2
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_xhigh_requires_and_consumes_one_authorization(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.store.set_session_safety(
        rig.session.session_id,
        "unattended",
    )
    rig.prime_capacity()
    try:
        rejected = await rig.message(
            "Use maximum effort.",
            provider="codex",
            effort="xhigh",
        )
        assert rejected.status == "failed"
        assert rejected.result["code"] == "E_SAFETY_EFFORT"

        rig.store.extend_session_safety(
            rig.session.session_id,
            {
                "reason": "operator approved one bounded attempt",
                "allow_xhigh_once": True,
            },
        )
        accepted = await rig.message(
            "Use maximum effort once.",
            provider="codex",
            effort="xhigh",
        )

        assert accepted.status == "complete"
        codex = rig.adapters["codex"]
        assert codex.efforts == ["xhigh"]
        safety = rig.store.session_safety(rig.session.session_id)
        assert safety["xhigh_authorizations"] == 0
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_reconciliation_restores_pre_dispatch_checkpoint(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    manager, record, command = await _ambiguous_reconciliation(rig)
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
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_e2e_reconciliation_accepts_current_or_stops(
    tmp_path: Path,
) -> None:
    accept_root = tmp_path / "accept"
    accept_root.mkdir()
    accept_rig = JourneyRig(accept_root)
    accept_manager, accept_record, unused_command = (
        await _ambiguous_reconciliation(accept_rig)
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
        assert "checkpoint_id" in accepted.audit
        assert (Path(accept_rig.session.worktree) / "file.txt").read_text(
            encoding="utf-8"
        ) == "ambiguous effect\n"
        assert (
            accept_rig.store.get_session(
                accept_rig.session.session_id
            ).attention
            == "idle"
        )
        with pytest.raises(ConflictError):
            await accept_manager.resolve(
                accept_record.reconciliation_id,
                ReconciliationDecision.STOP,
                accept_record.current_workspace_digest,
            )
    finally:
        accept_rig.close()

    stop_root = tmp_path / "stop"
    stop_root.mkdir()
    stop_rig = JourneyRig(stop_root)
    stop_manager, stop_record, unused_command = (
        await _ambiguous_reconciliation(stop_rig)
    )
    del unused_command
    try:
        stopped = await stop_manager.resolve(
            stop_record.reconciliation_id,
            ReconciliationDecision.STOP,
            stop_record.current_workspace_digest,
        )
        assert stopped.resolution == "stop"
        assert (
            stop_rig.store.get_session(stop_rig.session.session_id).lifecycle
            == "stopped"
        )
        assert (Path(stop_rig.session.worktree) / "file.txt").read_text(
            encoding="utf-8"
        ) == "ambiguous effect\n"
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
    manager, record, unused_command = await _ambiguous_reconciliation(
        changed_rig
    )
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
    manager, record, unused_command = await _ambiguous_reconciliation(
        race_rig
    )
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
    manager, record, unused_command = await _ambiguous_reconciliation(
        intent_rig
    )
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
    (workspace / "untracked-link").symlink_to("untracked.txt")

    digest, summary = inspect_workspace(workspace)

    assert len(digest) == 64
    assert '"commit"' in summary
    original_git = reconciliation_module._git

    def unsafe_paths(path: Path, *arguments: str) -> bytes:
        if arguments[0] == "ls-files":
            return b"../escape\0missing\0"
        return original_git(path, *arguments)

    monkeypatch.setattr(reconciliation_module, "_git", unsafe_paths)
    second_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    assert len(second_digest) == 64

    monkeypatch.setattr(reconciliation_module, "_git", original_git)
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


async def _ambiguous_reconciliation(
    rig: JourneyRig,
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
    (workspace / "untracked-effect-link").symlink_to(
        "untracked-effect.txt"
    )
    after_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    assert before_digest != after_digest
    manager = ReconciliationManager(rig.store, rig.blobs)
    recovery = await manager.recover_after_restart(
        rig.session.session_id
    )
    assert len(recovery.reconciliations) == 1
    record = recovery.reconciliations[0]
    assert record.safety_consumption["total_tokens"] == 73
    return manager, record, command


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
) -> UsageSnapshot:
    return UsageSnapshot(
        provider=provider,
        binding_percent=binding,
        credits_engaged=False,
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
