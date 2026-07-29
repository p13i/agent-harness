"""End-to-end journeys through the provider-neutral execution core."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import subprocess
from typing import Any

import pytest

from agent_harness.blobs import BlobStore
from agent_harness.config import paths
from agent_harness.config import prepare_paths
from agent_harness.errors import ProviderExhaustedError
from agent_harness.goals import create_goal
from agent_harness.goals import make_evidence
from agent_harness.ids import new_uuid
from agent_harness.models import CommandReceipt
from agent_harness.providers.base import ApprovalHandler
from agent_harness.providers.base import EventHandler
from agent_harness.providers.base import ProviderAdapter
from agent_harness.providers.base import ProviderEvent
from agent_harness.providers.base import ProviderModel
from agent_harness.providers.base import ProviderResult
from agent_harness.providers.base import ProviderStatus
from agent_harness.scheduler import Scheduler
from agent_harness.storage import StateStore
from agent_harness.usage import UsageSnapshot
from agent_harness.worker import SessionWorker
from tests.test_support import session


class ScriptedAdapter(ProviderAdapter):
    def __init__(
        self,
        provider: str,
        *,
        fail_turns: int = 0,
        request_approval: bool = False,
        turn_delay: float = 0.0,
    ) -> None:
        self.provider_id = provider
        self.fail_turns = fail_turns
        self.request_approval = request_approval
        self.turn_delay = turn_delay
        self.prompts: list[str] = []
        self.native_inputs: list[str] = []
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
        del effort
        self.prompts.append(prompt)
        self.native_inputs.append(native_session_id)
        if self.fail_turns:
            self.fail_turns -= 1
            raise ProviderExhaustedError(self.provider_id)
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
                ("medium", "high", "xhigh"),
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
async def test_e2e_provider_resume_and_cross_provider_continuity(
    tmp_path: Path,
) -> None:
    rig = JourneyRig(tmp_path)
    rig.prime_capacity()
    try:
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
        assert codex.prompts[1] == "Now add tests."
        assert claude.native_inputs == [""]
        assert "# Harness session" in claude.prompts[0]
        assert "Implement the parser." in claude.prompts[0]
        assert "codex completed the turn" in claude.prompts[0]
        assert "# Next instruction" in claude.prompts[0]
        assert len(rig.store.checkpoints(rig.session.session_id)) == 3
    finally:
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
        assert receipt.status == "complete"
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
    finally:
        rig.close()


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


def _usage(provider: str, binding: float) -> UsageSnapshot:
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
