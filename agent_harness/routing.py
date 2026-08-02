"""Explainable provider and model routing."""

from __future__ import annotations

import math
from typing import Iterable

from agent_harness.errors import ProviderUnavailableError
from agent_harness.models import RoutingCandidate, RoutingDecision

CAPACITY_LIMIT_PERCENT = 90.0
QUALITY_WINDOW = 10.0


def route(
    candidates: Iterable[RoutingCandidate],
    *,
    required_capabilities: frozenset[str] = frozenset(),
    workload: str = "implementation",
    manual_provider: str = "",
    metered_budget: float | None = None,
    binding_ceiling: float | None = None,
    execution_profile: str = "",
) -> RoutingDecision:
    if metered_budget is not None and not math.isfinite(metered_budget):
        raise ValueError("metered budget must be finite")
    if binding_ceiling is not None and not math.isfinite(binding_ceiling):
        raise ValueError("binding ceiling must be finite")
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
        + " with deterministic quality, then provider/model tie-breaking"
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
        credits_engaged=selected.credits_engaged,
        binding_percent=selected.binding_percent,
        usage_sample_id=selected.usage_sample_id,
        usage_observed_at=selected.usage_observed_at,
        required_capabilities=tuple(sorted(required_capabilities)),
        metered_budget=metered_budget,
        binding_ceiling=binding_ceiling,
        execution_profile=execution_profile,
        workload=workload,
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
    if candidate.credits_engaged:
        if metered_budget is None or metered_budget <= 0:
            return "metered credits require a positive explicit budget"
        if "cost-reporting" not in candidate.capabilities:
            return "metered credits require exact cost reporting"
    if candidate.binding_percent is not None:
        if not math.isfinite(candidate.binding_percent):
            return "binding usage is not finite"
        if candidate.binding_percent < 0:
            return "binding usage is negative"
        if candidate.binding_percent >= CAPACITY_LIMIT_PERCENT:
            return "binding usage is at or above 90 percent"
    return ""


def _fairness_score(candidate: RoutingCandidate, workload: str) -> float:
    headroom = 50.0
    if candidate.binding_percent is not None:
        if math.isfinite(candidate.binding_percent):
            headroom = max(1.0, 100.0 - candidate.binding_percent)
    role_weight = 1.0
    normalized = workload.strip().casefold()
    if normalized in {"planning", "architecture"}:
        if candidate.provider == "codex":
            role_weight = 1.25
    if normalized in {"implementation", "debugging"}:
        if candidate.provider == "claude":
            role_weight = 1.25
    independent_review = normalized in {
        "code review",
        "code-review",
        "code_review",
        "review",
    }
    if independent_review and not candidate.affinity:
        role_weight = 1.25
    weight = headroom * role_weight
    fairness = (candidate.queue_count + 1.0) / weight
    if candidate.affinity and not independent_review:
        fairness -= 0.05
    fairness += candidate.context_transfer_tokens / 10_000_000.0
    return fairness
