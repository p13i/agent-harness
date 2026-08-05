"""Deterministic three-level enterprise policy evaluation tests."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_support import session

from agent_harness.blobs import BlobStore
from agent_harness.config import paths, prepare_paths
from agent_harness.errors import (
    PolicyDeferredError,
    ProviderUnavailableError,
)
from agent_harness.goals import create_goal
from agent_harness.ids import new_uuid
from agent_harness.models import CommandStatus, ProviderAttempt
from agent_harness.policy import (
    ALLOW,
    BLOCK,
    DEFER,
    LEVEL_COMMAND,
    LEVEL_GOAL,
    LEVEL_SERVER,
    RULE_APPROVAL_REQUIRED,
    RULE_BINDING_CEILING,
    RULE_PERMITTED_PROVIDERS,
    RULE_REQUIRED_CAPABILITIES,
    RULE_REVIEW_PROVIDER_MUST_DIFFER,
    RULE_SAFETY_LIMITS,
    SERVER_POLICY_SCHEMA,
    PolicyEvaluator,
    implementation_providers,
    limit_decisions,
    load_server_policy,
)
from agent_harness.providers.base import (
    ProviderAdapter,
    ProviderEvent,
    ProviderModel,
    ProviderResult,
    ProviderStatus,
)
from agent_harness.safety import limits_for, tighten_limits
from agent_harness.scheduler import Scheduler
from agent_harness.storage import StateStore
from agent_harness.usage import UsageSnapshot
from agent_harness.worker import SessionWorker


class PolicyAdapter(ProviderAdapter):
    def __init__(
        self,
        provider: str,
        *,
        capabilities: tuple[str, ...] = ("tools", "resume", "approval"),
    ) -> None:
        self.provider_id = provider
        self._capabilities = frozenset(capabilities)
        self.prompts: list[str] = []

    async def run_turn(self, **kwargs: Any) -> ProviderResult:
        event_handler = kwargs["event_handler"]
        self.prompts.append(str(kwargs["prompt"]))
        await event_handler(
            ProviderEvent(
                "provider.prompt.accepted",
                status="accepted",
                native_session_id=self.provider_id + "-native",
            )
        )
        await event_handler(
            ProviderEvent(
                "agent.message",
                text=self.provider_id + " completed the turn",
                status="complete",
            )
        )
        return ProviderResult(
            provider=self.provider_id,
            native_session_id=self.provider_id + "-native",
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
                None,
                default=True,
            ),
        )

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.provider_id,
            ready=True,
            detail="policy test provider",
            capabilities=self._capabilities,
        )


def _policy_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema": SERVER_POLICY_SCHEMA}
    payload.update(overrides)
    return payload


def _decisions(store: StateStore, session_id: str) -> list[dict[str, Any]]:
    events = store.events(session_id, limit=5000)
    return [
        {**event.metadata, "event_status": event.status}
        for event in events
        if event.event_type == "policy.decision"
    ]


def _usage(provider: str, binding: float | None) -> UsageSnapshot:
    return UsageSnapshot(
        provider=provider,
        binding_percent=binding,
        credits_engaged=False,
        payload={},
    )


class SchedulerRig:
    def __init__(
        self,
        root: Path,
        adapters: dict[str, PolicyAdapter],
        usage: dict[str, float | None],
    ) -> None:
        workspace = root / "workspace"
        workspace.mkdir()
        self.store = StateStore(root / "state.sqlite3")
        self.session = session(workspace)
        self.store.create_session(self.session)
        self.adapters = adapters
        self.scheduler = Scheduler(self.store, self.adapters)
        self._usage = usage

    async def choose(self, **kwargs: Any) -> Any:
        self.scheduler._usage_cache = {
            provider: _usage(provider, binding)
            for provider, binding in self._usage.items()
        }
        self.scheduler._usage_at = asyncio.get_running_loop().time()
        return await self.scheduler.choose(self.session, **kwargs)

    def close(self) -> None:
        self.store.close()


def test_server_policy_file_is_optional_and_validated(tmp_path: Path) -> None:
    assert load_server_policy(tmp_path) == {}
    assert load_server_policy(None) == {}
    (tmp_path / "policy.json").write_text(
        json.dumps(_policy_payload(require_approval=True)),
        encoding="utf-8",
    )
    loaded = load_server_policy(tmp_path)
    assert loaded["require_approval"] is True
    (tmp_path / "policy.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_server_policy(tmp_path)
    (tmp_path / "policy.json").write_text(
        json.dumps({"schema": "other"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema is unsupported"):
        load_server_policy(tmp_path)
    (tmp_path / "policy.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_server_policy(tmp_path)


def test_limit_stacking_emits_a_decision_per_level() -> None:
    evaluator = PolicyEvaluator(
        _policy_payload(safety_limits={"max_seconds": 1800})
    )
    base = limits_for("interactive", "implementation")
    limits, server_decisions = evaluator.apply_server_limits(base)
    goal_base = limits
    limits = tighten_limits(limits, {"max_total_tokens": 100_000})
    goal_decisions = limit_decisions(LEVEL_GOAL, goal_base, limits)
    command_base = limits
    limits = tighten_limits(limits, {"max_total_tokens": 50_000})
    command_decisions = limit_decisions(LEVEL_COMMAND, command_base, limits)

    assert limits.max_seconds == 1800
    assert limits.max_total_tokens == 50_000
    assert len(server_decisions) == 1
    assert server_decisions[0].rule == RULE_SAFETY_LIMITS
    assert server_decisions[0].level == LEVEL_SERVER
    assert server_decisions[0].outcome == ALLOW
    assert "max_seconds" in server_decisions[0].reason
    assert [item.metadata["field"] for item in goal_decisions] == [
        "max_total_tokens"
    ]
    assert goal_decisions[0].level == LEVEL_GOAL
    assert [item.metadata["field"] for item in command_decisions] == [
        "max_total_tokens"
    ]
    assert command_decisions[0].level == LEVEL_COMMAND


def test_gate_decisions_map_rejections_to_uniform_rules() -> None:
    evaluator = PolicyEvaluator()
    rejected = [
        {
            "provider": "kimi",
            "model": "default",
            "reason": "required capabilities are unavailable",
        },
        {
            "provider": "codex",
            "model": "default",
            "reason": "provider is not ready",
        },
    ]
    decisions = evaluator.gate_decisions(
        rejected,
        {
            "codex": (
                RULE_BINDING_CEILING,
                "binding usage 95.0 reached the policy binding ceiling 90.0",
            )
        },
        command_id="command-1",
    )
    assert decisions[0].outcome == BLOCK
    assert decisions[0].rule == RULE_REQUIRED_CAPABILITIES
    assert decisions[0].provider == "kimi"
    assert decisions[0].command_id == "command-1"
    assert decisions[1].rule == RULE_BINDING_CEILING
    assert "binding ceiling" in decisions[1].reason


def test_implementation_providers_skip_review_only_turns() -> None:
    attempts = [
        ProviderAttempt(
            attempt_id="a1",
            session_id="s",
            provider="claude",
            native_session_id="n1",
            model="opus",
            effort="high",
            auth_mode="subscription",
            status="complete",
            started_at="",
            ended_at="",
        ),
        ProviderAttempt(
            attempt_id="a2",
            session_id="s",
            provider="codex",
            native_session_id="n2",
            model="default",
            effort="high",
            auth_mode="subscription",
            status="complete",
            started_at="",
            ended_at="",
        ),
    ]
    routings = [
        {"provider": "claude", "payload": {"workload": "implementation"}},
        {"provider": "codex", "payload": {"workload": "review"}},
    ]
    assert implementation_providers(attempts, routings) == frozenset({"claude"})
    assert implementation_providers(attempts, []) == frozenset(
        {"claude", "codex"}
    )


def test_binding_ceiling_rejection_emits_a_uniform_event(
    tmp_path: Path,
) -> None:
    rig = SchedulerRig(
        tmp_path,
        {
            "claude": PolicyAdapter("claude"),
            "codex": PolicyAdapter("codex"),
        },
        {"claude": 95.0, "codex": 20.0},
    )
    try:
        evaluator = PolicyEvaluator()

        async def scenario() -> Any:
            return await rig.choose(
                workload="implementation",
                required_capabilities=frozenset(),
                binding_ceiling=90.0,
                execution_profile="interactive",
                policy=evaluator,
                command_id="command-1",
            )

        decision = asyncio.run(scenario())
        assert decision.provider == "codex"
        decisions = _decisions(rig.store, rig.session.session_id)
        block = [
            item
            for item in decisions
            if item["outcome"] == BLOCK and item["rule"] == RULE_BINDING_CEILING
        ]
        assert len(block) == 1
        assert block[0]["provider"] == "claude"
        assert block[0]["command_id"] == "command-1"
        assert "binding ceiling" in block[0]["reason"]
        allow = [item for item in decisions if item["outcome"] == ALLOW]
        assert len(allow) == 1
        assert allow[0]["provider"] == "codex"
    finally:
        rig.close()


def test_capability_rejection_emits_a_uniform_event(tmp_path: Path) -> None:
    rig = SchedulerRig(
        tmp_path,
        {"claude": PolicyAdapter("claude")},
        {"claude": 20.0},
    )
    try:
        evaluator = PolicyEvaluator()

        async def scenario() -> None:
            await rig.choose(
                workload="implementation",
                required_capabilities=frozenset({"quantum"}),
                execution_profile="interactive",
                policy=evaluator,
            )

        with pytest.raises(ProviderUnavailableError):
            asyncio.run(scenario())
        decisions = _decisions(rig.store, rig.session.session_id)
        assert len(decisions) == 1
        assert decisions[0]["outcome"] == BLOCK
        assert decisions[0]["rule"] == RULE_REQUIRED_CAPABILITIES
        assert decisions[0]["provider"] == "claude"
    finally:
        rig.close()


def test_permitted_provider_rejection_emits_a_uniform_event(
    tmp_path: Path,
) -> None:
    rig = SchedulerRig(
        tmp_path,
        {
            "claude": PolicyAdapter("claude"),
            "codex": PolicyAdapter("codex"),
        },
        {"claude": 20.0, "codex": 20.0},
    )
    try:
        evaluator = PolicyEvaluator()

        async def scenario() -> Any:
            return await rig.choose(
                workload="implementation",
                required_capabilities=frozenset(),
                execution_profile="interactive",
                permitted_providers=frozenset({"claude"}),
                policy=evaluator,
            )

        decision = asyncio.run(scenario())
        assert decision.provider == "claude"
        decisions = _decisions(rig.store, rig.session.session_id)
        block = [
            item
            for item in decisions
            if item["rule"] == RULE_PERMITTED_PROVIDERS
        ]
        assert len(block) == 1
        assert block[0]["outcome"] == BLOCK
        assert block[0]["level"] == LEVEL_GOAL
        assert block[0]["provider"] == "codex"
    finally:
        rig.close()


def test_review_must_differ_routes_to_an_independent_provider(
    tmp_path: Path,
) -> None:
    rig = SchedulerRig(
        tmp_path,
        {
            "claude": PolicyAdapter("claude"),
            "codex": PolicyAdapter("codex"),
        },
        {"claude": 10.0, "codex": 20.0},
    )
    try:
        evaluator = PolicyEvaluator(
            _policy_payload(review_provider_must_differ=True)
        )

        async def scenario() -> Any:
            return await rig.choose(
                workload="review",
                required_capabilities=frozenset(),
                execution_profile="interactive",
                policy=evaluator,
                implementation_providers=frozenset({"claude"}),
            )

        decision = asyncio.run(scenario())
        assert decision.provider == "codex"
        decisions = _decisions(rig.store, rig.session.session_id)
        block = [
            item
            for item in decisions
            if item["rule"] == RULE_REVIEW_PROVIDER_MUST_DIFFER
        ]
        assert len(block) == 1
        assert block[0]["outcome"] == BLOCK
        assert block[0]["provider"] == "claude"
        assert block[0]["level"] == LEVEL_SERVER
    finally:
        rig.close()


def test_review_must_differ_defers_when_only_the_implementer_is_admissible(
    tmp_path: Path,
) -> None:
    rig = SchedulerRig(
        tmp_path,
        {"claude": PolicyAdapter("claude")},
        {"claude": 10.0},
    )
    try:
        evaluator = PolicyEvaluator(
            _policy_payload(review_provider_must_differ=True)
        )

        async def scenario() -> None:
            await rig.choose(
                workload="review",
                required_capabilities=frozenset(),
                execution_profile="interactive",
                policy=evaluator,
                implementation_providers=frozenset({"claude"}),
            )

        with pytest.raises(PolicyDeferredError) as captured:
            asyncio.run(scenario())
        assert captured.value.rule == RULE_REVIEW_PROVIDER_MUST_DIFFER
        decisions = _decisions(rig.store, rig.session.session_id)
        defer = [
            item
            for item in decisions
            if item["rule"] == RULE_REVIEW_PROVIDER_MUST_DIFFER
            and item["outcome"] == DEFER
        ]
        assert len(defer) == 1
        assert "independent provider" in defer[0]["reason"]
    finally:
        rig.close()


class WorkerRig:
    def __init__(
        self,
        root: Path,
        *,
        adapters: dict[str, PolicyAdapter] | None = None,
        profile: str = "interactive",
        policy: dict[str, Any] | None = None,
        goal_budgets: dict[str, Any] | None = None,
        approval_handler: Any = None,
    ) -> None:
        workspace = root / "workspace"
        _repository(workspace)
        harness_paths = paths(root / "state")
        prepare_paths(harness_paths)
        if policy is not None:
            (harness_paths.state_dir / "policy.json").write_text(
                json.dumps(policy),
                encoding="utf-8",
            )
        self.store = StateStore(harness_paths.database)
        self.blobs = BlobStore(harness_paths.blobs)
        self.session = session(workspace)
        self.store.create_session(self.session)
        if goal_budgets is not None:
            self.store.create_goal(
                create_goal(
                    self.session.session_id,
                    "Complete the bounded objective.",
                    budgets=goal_budgets,
                )
            )
        self.store.set_session_safety(self.session.session_id, profile)
        if adapters is None:
            adapters = {"claude": PolicyAdapter("claude")}
        self.adapters = adapters
        self.scheduler = Scheduler(self.store, self.adapters)
        self.worker = SessionWorker(
            self.store,
            self.blobs,
            self.scheduler,
            self.adapters,
            self.session.session_id,
            paths=harness_paths,
            policy_approval_handler=approval_handler,
        )
        self.store.register_worker(
            self.session.session_id,
            123,
            self.worker.incarnation,
        )

    def prime_capacity(self, binding: float = 20.0) -> None:
        self.scheduler._usage_cache = {
            provider: _usage(provider, binding) for provider in self.adapters
        }
        self.scheduler._usage_at = asyncio.get_running_loop().time()

    async def message(self, text: str, **route: Any) -> Any:
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


def test_dispatch_stacking_emits_levelled_decisions(tmp_path: Path) -> None:
    rig = WorkerRig(
        tmp_path,
        policy=_policy_payload(safety_limits={"max_seconds": 1800}),
        goal_budgets={"tokens": 100_000},
    )
    try:
        async def scenario() -> Any:
            rig.prime_capacity()
            return await rig.message(
                "Summarize the repository.",
                safety_limits={"max_total_tokens": 50_000},
            )

        receipt = asyncio.run(scenario())
        assert receipt.status == CommandStatus.COMPLETE
        envelope = rig.store.command_envelope(receipt.command_id)
        assert envelope["limits"]["max_seconds"] == 1800
        assert envelope["limits"]["max_total_tokens"] == 50_000
        decisions = _decisions(rig.store, rig.session.session_id)
        stacked = [
            (item["level"], item["metadata"]["field"])
            for item in decisions
            if item["rule"] == RULE_SAFETY_LIMITS
        ]
        assert (LEVEL_SERVER, "max_seconds") in stacked
        assert (LEVEL_GOAL, "max_total_tokens") in stacked
        assert (LEVEL_COMMAND, "max_total_tokens") in stacked
    finally:
        rig.close()


def test_interactive_approval_round_trip(tmp_path: Path) -> None:
    observed: list[str] = []

    async def approve(decision: Any) -> bool:
        observed.append(decision.provider)
        return True

    rig = WorkerRig(
        tmp_path,
        policy=_policy_payload(require_approval=True),
        approval_handler=approve,
    )
    try:
        async def scenario() -> Any:
            rig.prime_capacity()
            return await rig.message("Implement the change.")

        receipt = asyncio.run(scenario())
        assert receipt.status == CommandStatus.COMPLETE
        assert observed == ["claude"]
        assert rig.adapters["claude"].prompts
        decisions = _decisions(rig.store, rig.session.session_id)
        approval = [
            item
            for item in decisions
            if item["rule"] == RULE_APPROVAL_REQUIRED
        ]
        assert len(approval) == 1
        assert approval[0]["outcome"] == ALLOW
        assert approval[0]["provider"] == "claude"
    finally:
        rig.close()


def test_interactive_approval_decline_blocks_the_dispatch(
    tmp_path: Path,
) -> None:
    async def decline(decision: Any) -> bool:
        del decision
        return False

    rig = WorkerRig(
        tmp_path,
        policy=_policy_payload(require_approval=True),
        approval_handler=decline,
    )
    try:
        async def scenario() -> Any:
            rig.prime_capacity()
            return await rig.message("Implement the change.")

        receipt = asyncio.run(scenario())
        assert receipt.status == CommandStatus.FAILED
        assert receipt.result["code"] == "E_POLICY_BLOCKED"
        assert not rig.adapters["claude"].prompts
        decisions = _decisions(rig.store, rig.session.session_id)
        approval = [
            item
            for item in decisions
            if item["rule"] == RULE_APPROVAL_REQUIRED
        ]
        assert len(approval) == 1
        assert approval[0]["outcome"] == BLOCK
    finally:
        rig.close()


def test_kimi_blocks_when_approval_is_required(tmp_path: Path) -> None:
    kimi = PolicyAdapter("kimi", capabilities=("tools", "resume"))
    rig = WorkerRig(
        tmp_path,
        adapters={"kimi": kimi},
        policy=_policy_payload(require_approval=True),
    )
    try:
        async def scenario() -> Any:
            rig.prime_capacity()
            return await rig.message("Implement the change.", provider="kimi")

        receipt = asyncio.run(scenario())
        assert receipt.status == CommandStatus.FAILED
        assert receipt.result["code"] == "E_POLICY_BLOCKED"
        assert not kimi.prompts
        decisions = _decisions(rig.store, rig.session.session_id)
        approval = [
            item
            for item in decisions
            if item["rule"] == RULE_APPROVAL_REQUIRED
        ]
        assert len(approval) == 1
        assert approval[0]["outcome"] == BLOCK
        assert approval[0]["provider"] == "kimi"
        assert "cannot prompt" in approval[0]["reason"]
    finally:
        rig.close()


def test_unattended_approval_requirement_defers_with_policy_paused(
    tmp_path: Path,
) -> None:
    rig = WorkerRig(
        tmp_path,
        profile="unattended",
        policy=_policy_payload(require_approval=True),
    )
    try:
        async def scenario() -> Any:
            rig.prime_capacity()
            return await rig.message("Implement the change.")

        receipt = asyncio.run(scenario())
        assert receipt.status == CommandStatus.QUEUED
        assert not rig.adapters["claude"].prompts
        session_row = rig.store.get_session(rig.session.session_id)
        assert session_row.lifecycle == "paused"
        decisions = _decisions(rig.store, rig.session.session_id)
        approval = [
            item
            for item in decisions
            if item["rule"] == RULE_APPROVAL_REQUIRED
        ]
        assert len(approval) == 1
        assert approval[0]["outcome"] == DEFER
        paused = [
            event
            for event in rig.store.events(rig.session.session_id, limit=5000)
            if event.event_type == "policy.paused"
        ]
        assert len(paused) == 1
        assert paused[0].metadata["rule"] == RULE_APPROVAL_REQUIRED
    finally:
        rig.close()


def test_worker_review_routing_records_the_exclusion(tmp_path: Path) -> None:
    rig = WorkerRig(
        tmp_path,
        adapters={
            "claude": PolicyAdapter("claude"),
            "codex": PolicyAdapter("codex"),
        },
        policy=_policy_payload(review_provider_must_differ=True),
    )
    try:
        async def scenario() -> Any:
            rig.prime_capacity()
            first = await rig.message(
                "Implement the change.",
                provider="claude",
            )
            assert first.status == CommandStatus.COMPLETE
            return await rig.message(
                "Review the change.",
                workload="review",
            )

        receipt = asyncio.run(scenario())
        assert receipt.status == CommandStatus.COMPLETE
        assert rig.adapters["codex"].prompts
        assert len(rig.adapters["claude"].prompts) == 1
        decisions = _decisions(rig.store, rig.session.session_id)
        exclusion = [
            item
            for item in decisions
            if item["rule"] == RULE_REVIEW_PROVIDER_MUST_DIFFER
        ]
        assert len(exclusion) == 1
        assert exclusion[0]["outcome"] == BLOCK
        assert exclusion[0]["provider"] == "claude"
        routing = [
            event
            for event in rig.store.events(rig.session.session_id, limit=5000)
            if event.event_type == "routing.selected"
        ]
        assert routing[-1].metadata["provider"] == "codex"
    finally:
        rig.close()


def test_invalid_server_policy_fails_closed(tmp_path: Path) -> None:
    rig = WorkerRig(tmp_path)
    try:
        (tmp_path / "state" / "policy.json").write_text(
            json.dumps({"schema": "other"}),
            encoding="utf-8",
        )

        async def scenario() -> Any:
            rig.prime_capacity()
            return await rig.message("Implement the change.")

        receipt = asyncio.run(scenario())
        assert receipt.status == CommandStatus.FAILED
        assert receipt.result["code"] == "E_POLICY_INVALID"
        assert not rig.adapters["claude"].prompts
        decisions = _decisions(rig.store, rig.session.session_id)
        assert len(decisions) == 1
        assert decisions[0]["outcome"] == BLOCK
        assert decisions[0]["rule"] == "server_policy"
    finally:
        rig.close()
