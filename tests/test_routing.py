import asyncio
import datetime
import math
from pathlib import Path

import pytest
from test_support import session

from agent_harness.errors import ProviderUnavailableError
from agent_harness.models import RoutingCandidate
from agent_harness.providers.base import (
    ProviderAdapter,
    ProviderModel,
    ProviderResult,
    ProviderStatus,
)
from agent_harness.routing import route
from agent_harness.scheduler import (
    Scheduler,
    _durable_usage,
    _fallback_models,
    _optional_number,
    _select_effort,
    _select_model,
    _usage_is_fresh,
)
from agent_harness.storage import StateStore
from agent_harness.usage import UsageSnapshot


def candidate(
    provider: str,
    *,
    binding: float | None,
    credits: bool = False,
    cost_reporting: bool = False,
    queue: int = 0,
    affinity: bool = False,
) -> RoutingCandidate:
    capabilities = {"tools", "resume"}
    if cost_reporting:
        capabilities.add("cost-reporting")
    return RoutingCandidate(
        provider=provider,
        model="frontier",
        effort="xhigh",
        ready=True,
        capabilities=frozenset(capabilities),
        quality=100.0,
        binding_percent=binding,
        credits_engaged=credits,
        queue_count=queue,
        affinity=affinity,
        context_transfer_tokens=0,
    )


def test_routing_prefers_implementation_headroom_with_role_bias() -> None:
    decision = route(
        [
            candidate("codex", binding=50),
            candidate("claude", binding=50),
        ],
        workload="implementation",
    )
    assert decision.provider == "claude"


def test_routing_drops_ninety_percent_capacity() -> None:
    decision = route(
        [
            candidate("codex", binding=90),
            candidate("claude", binding=70),
        ]
    )
    assert decision.provider == "claude"
    assert decision.rejected[0]["provider"] == "codex"


@pytest.mark.parametrize("binding", [math.nan, math.inf, -1.0])
def test_routing_rejects_malformed_binding_usage(binding: float) -> None:
    with pytest.raises(ProviderUnavailableError):
        route([candidate("codex", binding=binding)])


def test_review_routing_uses_provider_neutral_ordering() -> None:
    first = route(
        [
            candidate("codex", binding=50),
            candidate("kimi", binding=50),
        ],
        workload="code review",
    )
    second = route(
        [
            candidate("kimi", binding=50),
            candidate("codex", binding=50),
        ],
        workload="code review",
    )

    assert first.provider == "codex"
    assert second.provider == "codex"
    assert (
        "deterministic quality, then provider/model tie-breaking"
        in first.reason
    )


def test_routing_normalizes_workload_whitespace() -> None:
    decision = route(
        [
            candidate("codex", binding=50),
            candidate("claude", binding=50),
        ],
        workload="  implementation  ",
    )

    assert decision.provider == "claude"


def test_review_routing_prefers_an_independent_provider() -> None:
    for workload in ("code review", "code-review", "code_review", "review"):
        decision = route(
            [
                candidate("codex", binding=50, affinity=True),
                candidate("claude", binding=50),
            ],
            workload=workload,
        )

        assert decision.provider == "claude"


@pytest.mark.parametrize("budget", [math.nan, math.inf, -math.inf])
def test_routing_rejects_nonfinite_metered_budget(budget: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        route(
            [candidate("codex", binding=10, credits=True, cost_reporting=True)],
            metered_budget=budget,
        )


@pytest.mark.parametrize("ceiling", [math.nan, math.inf, -math.inf])
def test_routing_rejects_a_nonfinite_binding_ceiling(ceiling: float) -> None:
    with pytest.raises(ValueError, match="binding ceiling must be finite"):
        route(
            [candidate("codex", binding=10)],
            binding_ceiling=ceiling,
        )


def test_metered_capacity_requires_explicit_budget() -> None:
    with pytest.raises(ProviderUnavailableError):
        route([candidate("codex", binding=10, credits=True)])
    with pytest.raises(ProviderUnavailableError):
        route(
            [candidate("codex", binding=10, credits=True)],
            metered_budget=0,
        )
    with pytest.raises(ProviderUnavailableError):
        route(
            [candidate("codex", binding=10, credits=True)],
            metered_budget=-1,
        )
    with pytest.raises(ProviderUnavailableError):
        route(
            [candidate("codex", binding=10, credits=True)],
            metered_budget=1.0,
        )
    decision = route(
        [
            candidate(
                "claude",
                binding=10,
                credits=True,
                cost_reporting=True,
            )
        ],
        metered_budget=1.0,
    )
    assert decision.provider == "claude"
    assert decision.credits_engaged is True


def test_explicit_model_and_effort_never_fall_back() -> None:
    models = (
        ProviderModel(
            "frontier",
            "Frontier",
            ("high", "xhigh"),
            None,
            default=True,
        ),
    )

    assert _select_model(models, "missing") is None
    selected = _select_model(models, "frontier")
    assert selected is not None
    assert _select_effort(selected, "low") is None
    assert _select_effort(selected, "xhigh") == "xhigh"


def test_routing_rejects_capability_and_quality_outliers() -> None:
    unavailable = candidate("claude", binding=20)
    unavailable = RoutingCandidate(
        provider=unavailable.provider,
        model=unavailable.model,
        effort=unavailable.effort,
        ready=unavailable.ready,
        capabilities=frozenset(),
        quality=unavailable.quality,
        binding_percent=unavailable.binding_percent,
        credits_engaged=unavailable.credits_engaged,
        queue_count=unavailable.queue_count,
        affinity=unavailable.affinity,
        context_transfer_tokens=unavailable.context_transfer_tokens,
    )
    low_quality = candidate("other", binding=20)
    low_quality = RoutingCandidate(
        provider=low_quality.provider,
        model=low_quality.model,
        effort=low_quality.effort,
        ready=low_quality.ready,
        capabilities=low_quality.capabilities,
        quality=70,
        binding_percent=low_quality.binding_percent,
        credits_engaged=low_quality.credits_engaged,
        queue_count=low_quality.queue_count,
        affinity=low_quality.affinity,
        context_transfer_tokens=low_quality.context_transfer_tokens,
    )
    decision = route(
        [
            candidate("codex", binding=20),
            unavailable,
            low_quality,
        ],
        required_capabilities=frozenset({"tools"}),
        workload="planning",
    )

    assert decision.provider == "codex"
    assert {item["reason"] for item in decision.rejected} == {
        "required capabilities are unavailable",
        "quality is outside the eligible window",
    }


class SlowAdapter(ProviderAdapter):
    provider_id = "claude"

    async def run_turn(self, **kwargs) -> ProviderResult:
        del kwargs
        raise AssertionError("turn execution was not requested")

    async def models(self, workspace: Path) -> tuple[ProviderModel, ...]:
        del workspace
        await asyncio.sleep(60)
        return ()

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider="claude",
            ready=True,
            detail="ready",
            capabilities=frozenset({"tools"}),
        )


class FailingModelAdapter(ProviderAdapter):
    provider_id = "codex"

    async def run_turn(self, **kwargs) -> ProviderResult:
        del kwargs
        raise AssertionError("turn execution was not requested")

    async def models(self, workspace: Path) -> tuple[ProviderModel, ...]:
        del workspace
        raise RuntimeError("model discovery failed")

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider="codex",
            ready=True,
            detail="ready",
            capabilities=frozenset({"tools"}),
        )


def test_status_returns_durable_usage_before_slow_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_usage() -> dict[str, UsageSnapshot]:
        await asyncio.sleep(60)
        return {}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        store.record_usage(
            "claude",
            37.0,
            False,
            {"payload": {"five_hour": {}}, "error": ""},
        )
        monkeypatch.setattr(
            "agent_harness.scheduler.probe_all",
            slow_usage,
        )
        scheduler = Scheduler(store, {"claude": SlowAdapter()})
        started = asyncio.get_running_loop().time()
        status = await scheduler.status(tmp_path)
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < 0.1
        assert status["claude"]["usage"]["binding_percent"] == 37.0
        assert status["claude"]["usage"]["sample_id"]
        assert status["claude"]["usage"]["observed_at"]
        assert status["claude"]["usage"]["fresh"] is True
        assert status["claude"]["usage"]["admissible"] is True
        assert status["claude"]["usage_refreshing"]
        assert status["claude"]["models"][0]["id"] == "opus"
        refresh = scheduler._status_refresh
        assert refresh is not None
        refresh.cancel()
        await asyncio.gather(refresh, return_exceptions=True)
        store.close()

    asyncio.run(scenario())


def test_status_reports_active_sessions_and_last_probe_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_usage() -> dict[str, UsageSnapshot]:
        await asyncio.sleep(60)
        return {}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        running = session(tmp_path)
        store.create_session(running)
        store.update_session(
            running.session_id,
            active_provider="claude",
        )
        paused = session(tmp_path)
        store.create_session(paused)
        store.update_session(
            paused.session_id,
            active_provider="claude",
            lifecycle="paused",
        )
        store.record_usage(
            "claude",
            None,
            False,
            {"payload": {}, "error": "probe timed out"},
        )
        monkeypatch.setattr(
            "agent_harness.scheduler.probe_all",
            slow_usage,
        )
        scheduler = Scheduler(store, {"claude": SlowAdapter()})
        status = await scheduler.status(tmp_path)

        # Paused sessions do not count toward the active total.
        assert status["claude"]["active_sessions"] == 1
        assert status["claude"]["last_error"] == "probe timed out"
        assert status["claude"]["ready"] is True
        assert "usage" in status["claude"]

        failed = UsageSnapshot(
            provider="claude",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error="HTTP 401",
        )

        async def failing_probe() -> dict[str, UsageSnapshot]:
            return {"claude": failed}

        monkeypatch.setattr(
            "agent_harness.scheduler.probe_all",
            failing_probe,
        )
        await scheduler.refresh_usage()
        refreshed = await scheduler.status(tmp_path)
        assert refreshed["claude"]["last_error"] == "HTTP 401"
        refresh = scheduler._status_refresh
        assert refresh is not None
        refresh.cancel()
        await asyncio.gather(refresh, return_exceptions=True)
        store.close()

    asyncio.run(scenario())


def test_scheduler_refreshes_usage_and_falls_back_on_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = UsageSnapshot(
        provider="codex",
        binding_percent=25.0,
        credits_engaged=False,
        payload={"weekly": {}},
    )

    async def probe() -> dict[str, UsageSnapshot]:
        return {"codex": snapshot}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        monkeypatch.setattr("agent_harness.scheduler.probe_all", probe)
        scheduler = Scheduler(store, {"codex": FailingModelAdapter()})

        assert await scheduler.usage() == {"codex": snapshot}
        assert store.latest_usage()["codex"]["binding_percent"] == 25.0
        models = await scheduler.models(tmp_path)
        assert models["codex"][0].model_id == "default"
        status = await scheduler.status(tmp_path)
        assert not status["codex"]["usage_refreshing"]
        store.close()

    asyncio.run(scenario())


def test_scheduler_uses_a_bounded_operator_usage_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = UsageSnapshot(
        provider="claude",
        binding_percent=None,
        credits_engaged=False,
        payload={},
        error="HTTP 429",
    )

    async def probe() -> dict[str, UsageSnapshot]:
        return {"claude": failed}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        scheduler = Scheduler(store, {"claude": SlowAdapter()})
        assert scheduler._operator_usage_fallback(failed) is failed
        unchanged = UsageSnapshot(
            provider="claude",
            binding_percent=20.0,
            credits_engaged=False,
            payload={},
        )
        assert scheduler._operator_usage_fallback(unchanged) is unchanged
        with pytest.raises(ValueError, match="attestation is invalid"):
            scheduler.attest_operator_usage(
                "claude",
                {
                    "binding_percent": True,
                    "credits_engaged": False,
                    "valid_seconds": True,
                    "evidence_sha256": "invalid",
                },
            )
        receipt = scheduler.attest_operator_usage(
            "claude",
            {
                "binding_percent": 44.0,
                "credits_engaged": False,
                "valid_seconds": 3600,
                "evidence_sha256": "a" * 64,
            },
        )
        assert receipt["source"] == "operator-attestation"
        assert receipt["sample_id"]
        assert receipt["binding_percent"] == 44.0
        assert receipt["credits_engaged"] is False
        assert receipt["observed_at"]
        assert receipt["provider"] == "claude"
        monkeypatch.setattr("agent_harness.scheduler.probe_all", probe)
        selected = await scheduler.refresh_usage()
        assert selected["claude"].binding_percent == 44.0
        assert selected["claude"].error == ""
        assert (
            selected["claude"].payload["source"]
            == "operator-attestation-fallback"
        )
        assert selected["claude"].payload["attestation_sample_id"]
        assert len(selected["claude"].payload["live_probe_error_sha256"]) == 64
        latest = store.latest_usage()["claude"]
        assert latest["payload"]["error"] == ""
        attestation = store.latest_operator_usage_attestation("claude")
        assert attestation is not None
        assert attestation["sample_id"] == receipt["sample_id"]
        store.close()

    asyncio.run(scenario())


def test_operator_usage_attestation_fails_closed_when_invalid_or_expired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    scheduler = Scheduler(store, {"claude": SlowAdapter()})
    failed = UsageSnapshot(
        provider="claude",
        binding_percent=None,
        credits_engaged=False,
        payload={},
        error="HTTP 429",
    )
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "latest_operator_usage_attestation",
            lambda unused: {"payload": "invalid"},
        )
        assert scheduler._operator_usage_fallback(failed) is failed
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "latest_operator_usage_attestation",
            lambda unused: {"payload": {"payload": "invalid"}},
        )
        assert scheduler._operator_usage_fallback(failed) is failed
    base = {
        "schema": "p13i/agent-harness/operator-usage-attestation/v1",
        "source": "operator-attestation",
        "evidence_sha256": "b" * 64,
        "attested_at": "2026-08-03T00:00:00+00:00",
    }
    store.record_usage(
        "claude",
        44.0,
        False,
        {"payload": {**base, "expires_at": "invalid"}, "error": ""},
    )
    assert scheduler._operator_usage_fallback(failed) is failed
    store.record_usage(
        "claude",
        44.0,
        False,
        {
            "payload": {
                **base,
                "expires_at": "2099-01-01T00:00:00",
            },
            "error": "",
        },
    )
    assert scheduler._operator_usage_fallback(failed).error == ""
    store.record_usage(
        "claude",
        44.0,
        False,
        {
            "payload": {
                **base,
                "expires_at": "2020-01-01T00:00:00+00:00",
            },
            "error": "",
        },
    )
    assert scheduler._operator_usage_fallback(failed) is failed
    store.record_usage(
        "claude",
        95.0,
        False,
        {
            "payload": {
                **base,
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
            "error": "",
        },
    )
    assert scheduler._operator_usage_fallback(failed) is failed
    store.record_usage(
        "claude",
        44.0,
        True,
        {
            "payload": {
                **base,
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
            "error": "",
        },
    )
    assert scheduler._operator_usage_fallback(failed) is failed
    store.close()


def test_scheduler_enforces_capacity_concurrency_model_and_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = UsageSnapshot(
        provider="codex",
        binding_percent=95.0,
        credits_engaged=False,
        payload={},
    )

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        scheduler = Scheduler(store, {"codex": FailingModelAdapter()})
        scheduler._usage_cache = {"codex": snapshot}
        scheduler._usage_at = asyncio.get_running_loop().time()
        scheduler._model_cache = {
            "codex": (
                ProviderModel(
                    "frontier",
                    "Frontier",
                    ("low",),
                    100,
                    default=True,
                ),
            )
        }
        current = session(tmp_path)
        with pytest.raises(ProviderUnavailableError):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                binding_ceiling=90,
            )

        scheduler._usage_cache = {
            "codex": UsageSnapshot(
                provider="codex",
                binding_percent=10.0,
                credits_engaged=False,
                payload={},
            )
        }
        with pytest.raises(ProviderUnavailableError):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                provider="codex",
                permitted_providers=frozenset({"claude"}),
            )
        policy_route = await scheduler.choose(
            current,
            workload="implementation",
            required_capabilities=frozenset(),
            permitted_providers=frozenset({"codex"}),
            permitted_efforts=frozenset({"low"}),
        )
        assert policy_route.provider == "codex"
        assert policy_route.effort == "low"
        with pytest.raises(ProviderUnavailableError):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                effort="high",
                permitted_efforts=frozenset({"low"}),
            )
        monkeypatch.setattr(
            store,
            "active_unattended_provider_count",
            lambda provider: 1,
        )
        with pytest.raises(ProviderUnavailableError):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                execution_profile="unattended",
                enforce_concurrency=True,
            )
        monkeypatch.setattr(
            store,
            "command_envelope",
            lambda command_id: {
                "command_id": command_id,
                "provider": "codex",
                "state": "recovering",
            },
        )
        resumed = await scheduler.choose(
            current,
            workload="implementation",
            required_capabilities=frozenset(),
            execution_profile="unattended",
            enforce_concurrency=True,
            command_id="command-1",
        )
        assert resumed.provider == "codex"
        monkeypatch.setattr(
            store,
            "command_envelope",
            lambda command_id: {
                "command_id": command_id,
                "provider": "codex",
                "state": "complete",
            },
        )
        with pytest.raises(ProviderUnavailableError):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                execution_profile="unattended",
                enforce_concurrency=True,
                command_id="command-2",
            )
        monkeypatch.setattr(
            store,
            "active_unattended_provider_count",
            lambda provider: 0,
        )
        monkeypatch.setattr(store, "active_goal_command_count", lambda goal_id: 2)
        with pytest.raises(ProviderUnavailableError):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                enforce_concurrency=True,
                goal_id="goal-1",
                max_concurrency=2,
            )
        monkeypatch.setattr(store, "active_goal_command_count", lambda goal_id: 0)
        with pytest.raises(ProviderUnavailableError):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                model="missing",
            )
        with pytest.raises(ProviderUnavailableError):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                model="frontier",
                effort="high",
            )
        store.close()

    asyncio.run(scenario())


def test_scheduler_fails_closed_on_unusable_budgets_and_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        scheduler = Scheduler(store, {"codex": FailingModelAdapter()})
        scheduler._usage_cache = {
            "codex": UsageSnapshot(
                provider="codex",
                binding_percent=math.nan,
                credits_engaged=False,
                payload={},
            )
        }
        scheduler._usage_at = asyncio.get_running_loop().time()
        current = session(tmp_path)
        with pytest.raises(ValueError, match="metered budget must be finite"):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                metered_budget=math.nan,
            )
        with pytest.raises(ValueError, match="binding ceiling must be finite"):
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                binding_ceiling=math.inf,
            )
        unbound = await scheduler.choose(
            current,
            workload="implementation",
            required_capabilities=frozenset(),
        )
        assert unbound.provider == "codex"
        assert unbound.binding_percent is None

        monkeypatch.setattr(store, "active_goal_command_count", lambda goal_id: 3)
        monkeypatch.setattr(
            store,
            "command_envelope",
            lambda command_id: {
                "command_id": command_id,
                "provider": "codex",
                "state": "complete",
            },
        )
        assert scheduler._active_other_goal_count("goal-1", "command-1") == 3
        store.close()

    asyncio.run(scenario())


def test_unattended_dispatch_probes_once_on_a_missing_usage_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy = UsageSnapshot(
        provider="codex",
        binding_percent=12.0,
        credits_engaged=False,
        payload={},
    )
    probes = 0

    async def probe() -> dict[str, UsageSnapshot]:
        nonlocal probes
        probes += 1
        return {"codex": healthy}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        monkeypatch.setattr("agent_harness.scheduler.probe_all", probe)
        scheduler = Scheduler(store, {"codex": FailingModelAdapter()})
        # Post-restart cold cache: a startup probe already failed, so the
        # cache serves a missing sample inside the 60-second reuse window.
        scheduler._usage_cache = {
            "codex": UsageSnapshot(
                provider="codex",
                binding_percent=None,
                credits_engaged=False,
                payload={},
                error="probe timed out",
            )
        }
        scheduler._usage_at = asyncio.get_running_loop().time()
        scheduler._model_cache = {
            "codex": (
                ProviderModel(
                    "frontier",
                    "Frontier",
                    ("low",),
                    100,
                    default=True,
                ),
            )
        }
        current = session(tmp_path)
        decision = await scheduler.choose(
            current,
            workload="implementation",
            required_capabilities=frozenset(),
            binding_ceiling=90,
            execution_profile="unattended",
        )
        assert decision.provider == "codex"
        assert decision.binding_percent == 12.0
        assert probes == 1
        store.close()

    asyncio.run(scenario())


def test_unattended_dispatch_routes_on_an_empty_usage_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy = UsageSnapshot(
        provider="codex",
        binding_percent=12.0,
        credits_engaged=False,
        payload={},
    )
    probes = 0

    async def probe() -> dict[str, UsageSnapshot]:
        nonlocal probes
        probes += 1
        return {"codex": healthy}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        monkeypatch.setattr("agent_harness.scheduler.probe_all", probe)
        scheduler = Scheduler(store, {"codex": FailingModelAdapter()})
        scheduler._model_cache = {
            "codex": (
                ProviderModel(
                    "frontier",
                    "Frontier",
                    ("low",),
                    100,
                    default=True,
                ),
            )
        }
        current = session(tmp_path)
        decision = await scheduler.choose(
            current,
            workload="implementation",
            required_capabilities=frozenset(),
            binding_ceiling=90,
            execution_profile="unattended",
        )
        assert decision.provider == "codex"
        # The initial probe already refreshed; admission must not probe twice.
        assert probes == 1
        store.close()

    asyncio.run(scenario())


def test_unattended_dispatch_rejects_a_saturated_fleet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saturated = UsageSnapshot(
        provider="codex",
        binding_percent=95.0,
        credits_engaged=False,
        payload={},
    )
    probes = 0

    async def probe() -> dict[str, UsageSnapshot]:
        nonlocal probes
        probes += 1
        return {"codex": saturated}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        monkeypatch.setattr("agent_harness.scheduler.probe_all", probe)
        scheduler = Scheduler(store, {"codex": FailingModelAdapter()})
        scheduler._model_cache = {
            "codex": (
                ProviderModel(
                    "frontier",
                    "Frontier",
                    ("low",),
                    100,
                    default=True,
                ),
            )
        }
        current = session(tmp_path)
        with pytest.raises(ProviderUnavailableError) as captured:
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                binding_ceiling=90,
                execution_profile="unattended",
            )
        assert captured.value.detail.code == "E_PROVIDER_UNAVAILABLE"
        assert "automatic routing is unavailable" in str(captured.value)
        assert probes == 1
        store.close()

    asyncio.run(scenario())


def test_unattended_dispatch_rejects_when_the_probe_keeps_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = UsageSnapshot(
        provider="codex",
        binding_percent=None,
        credits_engaged=False,
        payload={},
        error="HTTP 503",
    )
    probes = 0

    async def probe() -> dict[str, UsageSnapshot]:
        nonlocal probes
        probes += 1
        return {"codex": failed}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        monkeypatch.setattr("agent_harness.scheduler.probe_all", probe)
        scheduler = Scheduler(store, {"codex": FailingModelAdapter()})
        scheduler._usage_cache = {"codex": failed}
        scheduler._usage_at = asyncio.get_running_loop().time()
        scheduler._model_cache = {
            "codex": (
                ProviderModel(
                    "frontier",
                    "Frontier",
                    ("low",),
                    100,
                    default=True,
                ),
            )
        }
        current = session(tmp_path)
        with pytest.raises(ProviderUnavailableError) as captured:
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                binding_ceiling=90,
                execution_profile="unattended",
            )
        assert captured.value.detail.code == "E_PROVIDER_UNAVAILABLE"
        # One bounded refresh per admission check, never a tight loop.
        assert probes == 1
        store.close()

    asyncio.run(scenario())


def test_unattended_pinned_provider_proceeds_when_binding_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sibling of the automatic-routing case above: the same missing
    # sample must keep failing closed there and admit the pin here.
    failed = UsageSnapshot(
        provider="codex",
        binding_percent=None,
        credits_engaged=False,
        payload={},
        error="HTTP 429",
    )
    probes = 0

    async def probe() -> dict[str, UsageSnapshot]:
        nonlocal probes
        probes += 1
        return {"codex": failed}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        monkeypatch.setattr("agent_harness.scheduler.probe_all", probe)
        scheduler = Scheduler(store, {"codex": FailingModelAdapter()})
        scheduler._usage_cache = {"codex": failed}
        scheduler._usage_at = asyncio.get_running_loop().time()
        scheduler._model_cache = {
            "codex": (
                ProviderModel(
                    "frontier",
                    "Frontier",
                    ("low", "medium"),
                    100,
                    default=True,
                ),
            )
        }
        current = session(tmp_path)
        decision = await scheduler.choose(
            current,
            workload="implementation",
            required_capabilities=frozenset({"tools"}),
            provider="codex",
            model="frontier",
            effort="medium",
            binding_ceiling=90,
            execution_profile="unattended",
        )
        assert decision.provider == "codex"
        assert decision.model == "frontier"
        assert decision.effort == "medium"
        # Missing telemetry stays missing in the routing record: it is
        # never rewritten as headroom the harness did not observe.
        assert decision.binding_percent is None
        assert decision.ranked[0]["binding_percent"] is None
        # One bounded refresh per admission check, never a tight loop.
        assert probes == 1
        # Every other gate still decides on its own evidence.
        with pytest.raises(ProviderUnavailableError) as captured:
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset({"checkpoint"}),
                provider="codex",
                binding_ceiling=90,
                execution_profile="unattended",
            )
        assert captured.value.rejected[0]["reason"] == (
            "required capabilities are unavailable"
        )
        store.close()

    asyncio.run(scenario())


def test_unattended_pinned_provider_rejects_known_saturation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saturated = UsageSnapshot(
        provider="codex",
        binding_percent=95.0,
        credits_engaged=False,
        payload={},
    )

    async def probe() -> dict[str, UsageSnapshot]:
        return {"codex": saturated}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        monkeypatch.setattr("agent_harness.scheduler.probe_all", probe)
        scheduler = Scheduler(store, {"codex": FailingModelAdapter()})
        scheduler._model_cache = {
            "codex": (
                ProviderModel(
                    "frontier",
                    "Frontier",
                    ("low",),
                    100,
                    default=True,
                ),
            )
        }
        current = session(tmp_path)
        with pytest.raises(ProviderUnavailableError) as captured:
            await scheduler.choose(
                current,
                workload="implementation",
                required_capabilities=frozenset(),
                provider="codex",
                binding_ceiling=90,
                execution_profile="unattended",
            )
        # Same adapter, same models, same pin as the case above: only
        # the known saturated sample differs, and the ceiling gate
        # withholds readiness for it.
        assert captured.value.rejected[0]["reason"] == "provider is not ready"
        store.close()

    asyncio.run(scenario())


def test_scheduler_reuses_active_status_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_usage() -> dict[str, UsageSnapshot]:
        await asyncio.sleep(60)
        return {}

    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        monkeypatch.setattr(
            "agent_harness.scheduler.probe_all",
            slow_usage,
        )
        scheduler = Scheduler(store, {"claude": SlowAdapter()})
        await scheduler.status(tmp_path)
        first = scheduler._status_refresh
        await scheduler.status(tmp_path)
        assert scheduler._status_refresh is first
        assert first is not None
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        store.close()

    asyncio.run(scenario())


def test_scheduler_helpers_normalize_fallback_values() -> None:
    durable = _durable_usage(
        {
            "codex": {
                "binding_percent": True,
                "credits_engaged": True,
                "payload": "invalid",
            },
            "claude": {
                "binding_percent": "unknown",
                "payload": {"payload": "invalid", "error": "offline"},
            },
        }
    )
    assert durable["codex"].binding_percent is None
    assert durable["codex"].payload == {}
    assert durable["claude"].binding_percent is None
    assert durable["claude"].payload == {}
    assert durable["claude"].error == "offline"
    assert _optional_number(False) is None
    assert _optional_number("unknown") is None
    assert _optional_number(math.nan) is None
    assert _optional_number(-1) is None

    without_default = (ProviderModel("first", "First", ("low",), None),)
    assert _select_model(without_default, "").model_id == "first"
    assert _select_model((), "").model_id == "default"
    assert _select_effort(without_default[0], "") == "low"
    assert (
        _select_effort(
            without_default[0],
            "",
            frozenset({"high"}),
        )
        is None
    )
    medium = ProviderModel("medium", "Medium", ("medium",), None)
    assert _select_effort(medium, "") == "medium"
    no_efforts = ProviderModel("none", "None", (), None)
    assert _select_effort(no_efforts, "") == ""
    assert _select_effort(no_efforts, "", frozenset({"high"})) is None
    # Without a config.toml pin, codex still falls back to "default".
    assert _fallback_models("codex")[0].model_id == "default"

    stale = datetime.datetime.now(datetime.UTC)
    stale -= datetime.timedelta(minutes=5)
    assert _usage_is_fresh(stale.isoformat()) is False
    assert _usage_is_fresh("not-a-timestamp") is False
    assert _usage_is_fresh(datetime.datetime.now().isoformat()) is True


def test_status_marks_high_binding_and_probe_errors_inadmissible(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = StateStore(tmp_path / "state.sqlite3")
        snapshot = UsageSnapshot(
            provider="claude",
            binding_percent=95.0,
            credits_engaged=False,
            payload={},
            error="probe failed",
        )
        store.record_usage(
            "claude",
            95.0,
            False,
            {"error": "probe failed"},
        )
        scheduler = Scheduler(store, {"claude": SlowAdapter()})
        scheduler._usage_cache = {"claude": snapshot}
        scheduler._usage_at = asyncio.get_running_loop().time()
        status = await scheduler.status(tmp_path)
        assert status["claude"]["usage"]["admissible"] is False
        refresh = scheduler._status_refresh
        assert refresh is not None
        refresh.cancel()
        await asyncio.gather(refresh, return_exceptions=True)
        store.close()

    asyncio.run(scenario())


def test_codex_config_model_and_merge(tmp_path: Path, monkeypatch) -> None:
    from agent_harness.providers import codex as codex_mod
    from agent_harness.providers.base import ProviderModel

    monkeypatch.setattr(codex_mod.Path, "home", lambda: tmp_path)
    assert codex_mod.codex_config_model() == ""
    assert codex_mod.resolve_codex_model("default") == "default"
    assert codex_mod.resolve_codex_model("") == ""

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text(
        'model = "gpt-5.6-sol"\nmodel_reasoning_effort = "xhigh"\n',
        encoding="utf-8",
    )
    assert codex_mod.codex_config_model() == "gpt-5.6-sol"
    assert codex_mod.resolve_codex_model("default") == "gpt-5.6-sol"
    assert codex_mod.resolve_codex_model("") == "gpt-5.6-sol"
    assert codex_mod.resolve_codex_model("o3") == "o3"

    only_default = (
        ProviderModel("default", "Account default", ("high",), None, default=True),
    )
    merged = codex_mod._merge_codex_config_models(only_default)
    assert merged[0].model_id == "gpt-5.6-sol"
    assert merged[0].default is True
    assert any(item.model_id == "default" for item in merged)

    empty = codex_mod._merge_codex_config_models(())
    assert empty[0].model_id == "gpt-5.6-sol"

    assert _fallback_models("codex")[0].model_id == "gpt-5.6-sol"
