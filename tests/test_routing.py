import asyncio
from pathlib import Path

import pytest

from agent_harness.errors import ProviderUnavailableError
from agent_harness.models import RoutingCandidate
from agent_harness.providers.base import ProviderAdapter
from agent_harness.providers.base import ProviderModel
from agent_harness.providers.base import ProviderResult
from agent_harness.providers.base import ProviderStatus
from agent_harness.routing import route
from agent_harness.scheduler import Scheduler
from agent_harness.scheduler import _durable_usage
from agent_harness.scheduler import _fallback_models
from agent_harness.scheduler import _optional_number
from agent_harness.scheduler import _select_effort
from agent_harness.scheduler import _select_model
from agent_harness.storage import StateStore
from agent_harness.usage import UsageSnapshot


def candidate(
    provider: str,
    *,
    binding: float | None,
    credits: bool = False,
    queue: int = 0,
) -> RoutingCandidate:
    return RoutingCandidate(
        provider=provider,
        model="frontier",
        effort="xhigh",
        ready=True,
        capabilities=frozenset({"tools", "resume"}),
        quality=100.0,
        binding_percent=binding,
        credits_engaged=credits,
        queue_count=queue,
        affinity=False,
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


def test_metered_capacity_requires_explicit_budget() -> None:
    with pytest.raises(ProviderUnavailableError):
        route([candidate("codex", binding=10, credits=True)])
    decision = route(
        [candidate("codex", binding=10, credits=True)],
        metered_budget=1.0,
    )
    assert decision.provider == "codex"


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
        assert status["claude"]["usage_refreshing"]
        assert status["claude"]["models"][0]["id"] == "opus"
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

    without_default = (
        ProviderModel("first", "First", ("low",), None),
    )
    assert _select_model(without_default, "").model_id == "first"
    assert _select_model((), "").model_id == "default"
    assert _select_effort(without_default[0], "") == "low"
    medium = ProviderModel("medium", "Medium", ("medium",), None)
    assert _select_effort(medium, "") == "medium"
    no_efforts = ProviderModel("none", "None", (), None)
    assert _select_effort(no_efforts, "") == ""
    assert _fallback_models("codex")[0].model_id == "default"
