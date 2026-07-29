"""Explainable provider and model routing."""

from __future__ import annotations

from typing import Iterable

from agent_harness.errors import ProviderUnavailableError
from agent_harness.models import RoutingCandidate
from agent_harness.models import RoutingDecision


CAPACITY_LIMIT_PERCENT = 90.0
QUALITY_WINDOW = 10.0


def route(
    candidates: Iterable[RoutingCandidate],
    *,
    required_capabilities: frozenset[str] = frozenset(),
    workload: str = "implementation",
    manual_provider: str = "",
    metered_budget: float | None = None,
) -> RoutingDecision:
    accepted: list[tuple[RoutingCandidate, float]] = []
    rejected: list[dict[str, str]] = []
    selected_candidates = list(candidates)
    for candidate in selected_candidates:
        reason = _rejection_reason(
            candidate,
            required_capabilities,
            manual_provider,
            metered_budget,
        )
        if reason:
            rejected.append(
                {
                    "provider": candidate.provider,
                    "model": candidate.model,
                    "reason": reason,
                }
            )
            continue
        accepted.append((candidate, 0.0))
    if not accepted:
        provider = manual_provider
        if not provider:
            provider = "automatic routing"
        raise ProviderUnavailableError(provider)

    max_quality = max(item[0].quality for item in accepted)
    quality_floor = max_quality - QUALITY_WINDOW
    qualified: list[tuple[RoutingCandidate, float]] = []
    for candidate, unused in accepted:
        del unused
        if candidate.quality < quality_floor:
            rejected.append(
                {
                    "provider": candidate.provider,
                    "model": candidate.model,
                    "reason": "quality is outside the eligible window",
                }
            )
            continue
        fairness = _fairness_score(candidate, workload)
        qualified.append((candidate, fairness))
    qualified.sort(
        key=lambda item: (
            item[1],
            -item[0].quality,
            item[0].provider,
            item[0].model,
        )
    )
    selected = qualified[0][0]
    ranked: list[dict[str, object]] = []
    for candidate, fairness in qualified:
        ranked.append(
            {
                "provider": candidate.provider,
                "model": candidate.model,
                "effort": candidate.effort,
                "quality": candidate.quality,
                "fairness": round(fairness, 6),
                "binding_percent": candidate.binding_percent,
                "affinity": candidate.affinity,
            }
        )
    reason = (
        selected.provider
        + "/"
        + selected.model
        + " passed capability and budget gates, stayed within the quality "
        + "window, and had the best headroom-adjusted fair-share score"
    )
    if selected.affinity:
        reason += " with provider-session affinity"
    return RoutingDecision(
        provider=selected.provider,
        model=selected.model,
        effort=selected.effort,
        reason=reason,
        ranked=tuple(ranked),
        rejected=tuple(rejected),
    )


def _rejection_reason(
    candidate: RoutingCandidate,
    required: frozenset[str],
    manual_provider: str,
    metered_budget: float | None,
) -> str:
    if manual_provider and candidate.provider != manual_provider:
        return "another provider was pinned"
    if not candidate.ready:
        return "provider is not ready"
    if not required.issubset(candidate.capabilities):
        return "required capabilities are unavailable"
    if candidate.credits_engaged and metered_budget is None:
        return "metered credits require an explicit budget"
    if candidate.binding_percent is not None:
        if candidate.binding_percent >= CAPACITY_LIMIT_PERCENT:
            return "binding usage is at or above 90 percent"
    return ""


def _fairness_score(candidate: RoutingCandidate, workload: str) -> float:
    headroom = 50.0
    if candidate.binding_percent is not None:
        headroom = max(1.0, 100.0 - candidate.binding_percent)
    role_weight = 1.0
    normalized = workload.casefold()
    if normalized in {"planning", "architecture"}:
        if candidate.provider == "codex":
            role_weight = 1.25
    if normalized in {"implementation", "debugging"}:
        if candidate.provider == "claude":
            role_weight = 1.25
    weight = headroom * role_weight
    fairness = (candidate.queue_count + 1.0) / weight
    if candidate.affinity:
        fairness -= 0.05
    fairness += candidate.context_transfer_tokens / 10_000_000.0
    return fairness

