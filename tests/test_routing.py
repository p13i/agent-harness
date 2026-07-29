import pytest

from agent_harness.errors import ProviderUnavailableError
from agent_harness.models import RoutingCandidate
from agent_harness.providers.base import ProviderModel
from agent_harness.routing import route
from agent_harness.scheduler import _select_effort
from agent_harness.scheduler import _select_model


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
