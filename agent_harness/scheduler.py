"""Provider discovery and turn-boundary routing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agent_harness.models import RoutingCandidate
from agent_harness.models import RoutingDecision
from agent_harness.models import Session
from agent_harness.providers.base import ProviderAdapter
from agent_harness.providers.base import ProviderModel
from agent_harness.routing import route
from agent_harness.safety import INTERACTIVE
from agent_harness.storage import StateStore
from agent_harness.usage import UsageSnapshot
from agent_harness.usage import probe_all


QUALITY = {
    "codex": 100.0,
    "claude": 100.0,
}


class Scheduler:
    def __init__(
        self,
        store: StateStore,
        adapters: dict[str, ProviderAdapter],
    ) -> None:
        self.store = store
        self.adapters = adapters
        self._model_cache: dict[str, tuple[ProviderModel, ...]] = {}
        self._usage_cache = _durable_usage(store.latest_usage())
        self._usage_at = 0.0
        self._status_refresh: asyncio.Task[None] | None = None

    async def refresh_usage(self) -> dict[str, UsageSnapshot]:
        snapshots = await probe_all()
        for snapshot in snapshots.values():
            self.store.record_usage(
                snapshot.provider,
                snapshot.binding_percent,
                snapshot.credits_engaged,
                {
                    "payload": snapshot.payload,
                    "error": snapshot.error,
                },
            )
        self._usage_cache = snapshots
        self._usage_at = asyncio.get_running_loop().time()
        return snapshots

    async def usage(self) -> dict[str, UsageSnapshot]:
        now = asyncio.get_running_loop().time()
        if not self._usage_cache or now - self._usage_at > 60:
            return await self.refresh_usage()
        return self._usage_cache

    async def models(
        self,
        workspace: Path,
        *,
        refresh: bool = False,
    ) -> dict[str, tuple[ProviderModel, ...]]:
        for provider, adapter in self.adapters.items():
            if not refresh and provider in self._model_cache:
                continue
            try:
                models = await adapter.models(workspace)
            except BaseException:
                models = ()
            if not models:
                models = _fallback_models(provider)
            self._model_cache[provider] = models
        return dict(self._model_cache)

    async def choose(
        self,
        session: Session,
        *,
        workload: str,
        required_capabilities: frozenset[str],
        provider: str = "",
        model: str = "",
        effort: str = "",
        metered_budget: float | None = None,
        excluded: frozenset[str] = frozenset(),
        context_transfer_tokens: int = 0,
        binding_ceiling: float | None = None,
        execution_profile: str = INTERACTIVE,
        enforce_concurrency: bool = False,
    ) -> RoutingDecision:
        usage = await self.usage()
        models = await self.models(Path(session.worktree))
        counts = self.store.active_provider_counts()
        candidates: list[RoutingCandidate] = []
        for provider_id, adapter in self.adapters.items():
            if provider_id in excluded:
                continue
            status = adapter.status()
            provider_usage = usage.get(provider_id)
            binding: float | None = None
            credits = False
            if provider_usage is not None:
                binding = provider_usage.binding_percent
                credits = provider_usage.credits_engaged
            safety_ready = status.ready
            if binding_ceiling is not None:
                if binding is None and execution_profile != INTERACTIVE:
                    safety_ready = False
                if binding is not None and binding >= binding_ceiling:
                    safety_ready = False
            if enforce_concurrency and execution_profile == "unattended":
                active = self.store.active_unattended_provider_count(
                    provider_id
                )
                if active >= 1:
                    safety_ready = False
            chosen_model = _select_model(models.get(provider_id, ()), model)
            if chosen_model is None:
                continue
            chosen_effort = _select_effort(chosen_model, effort)
            if chosen_effort is None:
                continue
            candidates.append(
                RoutingCandidate(
                    provider=provider_id,
                    model=chosen_model.model_id,
                    effort=chosen_effort,
                    ready=safety_ready,
                    capabilities=status.capabilities,
                    quality=QUALITY.get(provider_id, 90.0),
                    binding_percent=binding,
                    credits_engaged=credits,
                    queue_count=counts.get(provider_id, 0),
                    affinity=session.active_provider == provider_id,
                    context_transfer_tokens=context_transfer_tokens,
                )
            )
        return route(
            candidates,
            required_capabilities=required_capabilities,
            workload=workload,
            manual_provider=provider,
            metered_budget=metered_budget,
        )

    async def status(self, workspace: Path) -> dict[str, Any]:
        self._start_status_refresh(workspace)
        usage = dict(self._usage_cache)
        models = dict(self._model_cache)
        refreshing = self._status_refresh is not None
        if self._status_refresh is not None:
            refreshing = not self._status_refresh.done()
        result: dict[str, Any] = {}
        for provider, adapter in self.adapters.items():
            provider_status = adapter.status()
            snapshot = usage.get(provider)
            usage_value: dict[str, Any] = {}
            if snapshot is not None:
                usage_value = snapshot.as_dict()
            result[provider] = {
                "ready": provider_status.ready,
                "detail": provider_status.detail,
                "capabilities": sorted(provider_status.capabilities),
                "usage": usage_value,
                "usage_refreshing": refreshing,
                "models": [
                    {
                        "id": item.model_id,
                        "display_name": item.display_name,
                        "efforts": list(item.efforts),
                        "context_window": item.context_window,
                        "default": item.default,
                    }
                    for item in models.get(
                        provider,
                        _fallback_models(provider),
                    )
                ],
            }
        return result

    def _start_status_refresh(self, workspace: Path) -> None:
        if self._status_refresh is not None:
            if not self._status_refresh.done():
                return
        now = asyncio.get_running_loop().time()
        models_complete = all(
            provider in self._model_cache for provider in self.adapters
        )
        usage_fresh = self._usage_at > 0
        if usage_fresh:
            usage_fresh = now - self._usage_at <= 60
        if usage_fresh and models_complete:
            return
        self._status_refresh = asyncio.create_task(
            self._refresh_status(workspace)
        )

    async def _refresh_status(self, workspace: Path) -> None:
        await asyncio.gather(
            self.refresh_usage(),
            self.models(workspace, refresh=True),
        )


def _durable_usage(
    values: dict[str, dict[str, Any]],
) -> dict[str, UsageSnapshot]:
    snapshots: dict[str, UsageSnapshot] = {}
    for provider, value in values.items():
        stored = value.get("payload", {})
        if not isinstance(stored, dict):
            stored = {}
        payload = stored.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        snapshots[provider] = UsageSnapshot(
            provider=provider,
            binding_percent=_optional_number(
                value.get("binding_percent")
            ),
            credits_engaged=bool(value.get("credits_engaged", False)),
            payload=payload,
            error=str(stored.get("error", "")),
        )
    return snapshots


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _select_model(
    models: tuple[ProviderModel, ...],
    pinned: str,
) -> ProviderModel | None:
    if pinned:
        for item in models:
            if item.model_id == pinned:
                return item
        return None
    for item in models:
        if item.default:
            return item
    if models:
        return models[0]
    return ProviderModel("default", "Default", ("high",), None, default=True)


def _select_effort(model: ProviderModel, pinned: str) -> str | None:
    if pinned:
        if not model.efforts or pinned in model.efforts:
            return pinned
        return None
    for candidate in ("xhigh", "high", "medium"):
        if candidate in model.efforts:
            return candidate
    if model.efforts:
        return model.efforts[-1]
    return ""


def _fallback_models(provider: str) -> tuple[ProviderModel, ...]:
    efforts = ("low", "medium", "high", "xhigh")
    if provider == "claude":
        return (ProviderModel("opus", "Opus", efforts, None, default=True),)
    return (
        ProviderModel("default", "Account default", efforts, None, default=True),
    )
