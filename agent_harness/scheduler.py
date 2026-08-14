"""Provider discovery and turn-boundary routing."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import math
from pathlib import Path
from typing import Any

from agent_harness.errors import PolicyDeferredError, ProviderUnavailableError
from agent_harness.models import RoutingCandidate, RoutingDecision, Session
from agent_harness.policy import (
    BLOCK,
    DEFER,
    LEVEL_GOAL,
    LEVEL_SERVER,
    RULE_APPROVAL_REQUIRED,
    RULE_BINDING_CEILING,
    RULE_GOAL_CONCURRENCY,
    RULE_PERMITTED_EFFORTS,
    RULE_PERMITTED_PROVIDERS,
    RULE_REVIEW_PROVIDER_MUST_DIFFER,
    RULE_UNATTENDED_CONCURRENCY,
    PolicyDecision,
    PolicyEvaluator,
    is_review_workload,
)
from agent_harness.providers.base import ProviderAdapter, ProviderModel
from agent_harness.routing import route
from agent_harness.safety import INTERACTIVE
from agent_harness.storage import StateStore
from agent_harness.usage import UsageSnapshot, probe_all

QUALITY = {
    "codex": 100.0,
    "claude": 100.0,
    # Every provider scores the same here on purpose: capacity headroom,
    # not quality, is meant to decide between providers.
    "kimi": 90.0,
    "grok": 100.0,
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
        self._usage_at: float | None = None
        self._status_refresh: asyncio.Task[None] | None = None

    async def refresh_usage(self) -> dict[str, UsageSnapshot]:
        snapshots = await probe_all()
        effective: dict[str, UsageSnapshot] = {}
        for provider, snapshot in snapshots.items():
            selected = self._operator_usage_fallback(snapshot)
            effective[provider] = selected
            self.store.record_usage(
                selected.provider,
                selected.binding_percent,
                selected.credits_engaged,
                {
                    "payload": selected.payload,
                    "error": selected.error,
                },
            )
        self._usage_cache = effective
        self._usage_at = asyncio.get_running_loop().time()
        return effective

    def attest_operator_usage(
        self,
        provider: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        binding = _optional_number(payload.get("binding_percent"))
        valid_seconds = payload.get("valid_seconds")
        evidence_sha256 = str(payload.get("evidence_sha256", ""))
        valid_duration = isinstance(valid_seconds, int)
        if isinstance(valid_seconds, bool):
            valid_duration = False
        valid_evidence = len(evidence_sha256) == 64
        if valid_evidence:
            valid_evidence = all(
                character in "0123456789abcdef"
                for character in evidence_sha256
            )
        if (
            provider not in self.adapters
            or binding is None
            or binding >= 90.0
            or payload.get("credits_engaged") is not False
            or not valid_duration
            or int(valid_seconds) < 1
            or int(valid_seconds) > 7 * 24 * 60 * 60
            or not valid_evidence
        ):
            raise ValueError("operator usage attestation is invalid")
        attested_at = datetime.datetime.now(datetime.UTC)
        expires_at = attested_at + datetime.timedelta(seconds=int(valid_seconds))
        public_payload = {
            "schema": "p13i/agent-harness/operator-usage-attestation/v1",
            "source": "operator-attestation",
            "evidence_sha256": evidence_sha256,
            "attested_at": attested_at.isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        sample_id = self.store.record_usage(
            provider,
            binding,
            False,
            {"payload": public_payload, "error": ""},
        )
        snapshot = UsageSnapshot(
            provider=provider,
            binding_percent=binding,
            credits_engaged=False,
            payload=public_payload,
        )
        self._usage_cache[provider] = snapshot
        self._usage_at = asyncio.get_running_loop().time()
        durable = self.store.latest_usage()[provider]
        return {
            "sample_id": sample_id,
            "provider": provider,
            "binding_percent": binding,
            "credits_engaged": False,
            "observed_at": str(durable["observed_at"]),
            **public_payload,
        }

    def _operator_usage_fallback(
        self,
        snapshot: UsageSnapshot,
    ) -> UsageSnapshot:
        if not snapshot.error:
            return snapshot
        attestation = self.store.latest_operator_usage_attestation(
            snapshot.provider
        )
        if attestation is None:
            return snapshot
        stored = attestation.get("payload", {})
        if not isinstance(stored, dict):
            return snapshot
        public = stored.get("payload", {})
        if not isinstance(public, dict):
            return snapshot
        expires_at = str(public.get("expires_at", ""))
        try:
            expiration = datetime.datetime.fromisoformat(
                expires_at.replace("Z", "+00:00")
            )
        except ValueError:
            return snapshot
        if expiration.tzinfo is None:
            expiration = expiration.replace(tzinfo=datetime.UTC)
        if datetime.datetime.now(datetime.UTC) >= expiration:
            return snapshot
        binding = _optional_number(attestation.get("binding_percent"))
        if binding is None or binding >= 90.0:
            return snapshot
        if attestation.get("credits_engaged") is not False:
            return snapshot
        fallback_payload = dict(public)
        fallback_payload["source"] = "operator-attestation-fallback"
        fallback_payload["attestation_sample_id"] = str(
            attestation.get("sample_id", "")
        )
        fallback_payload["live_probe_error_sha256"] = hashlib.sha256(
            snapshot.error.encode("utf-8")
        ).hexdigest()
        return UsageSnapshot(
            provider=snapshot.provider,
            binding_percent=binding,
            credits_engaged=False,
            payload=fallback_payload,
        )

    async def usage(self) -> dict[str, UsageSnapshot]:
        now = asyncio.get_running_loop().time()
        if not self._usage_cache or self._usage_at is None or now - self._usage_at > 60:
            return await self.refresh_usage()
        return self._usage_cache

    async def _admission_usage(
        self,
        *,
        binding_ceiling: float | None,
        execution_profile: str,
        excluded: frozenset[str],
    ) -> dict[str, UsageSnapshot]:
        # Dispatch admission probe-on-miss: when the cache serves a
        # sample whose binding is missing, refresh once and let the
        # caller re-evaluate. Claude may then proceed with an explicitly
        # unknown binding; other automatic providers still fail closed.
        # Bounded to one refresh per admission check: when the cache
        # lookup already probed, its result stands even when the probe
        # errored.
        usage_at = self._usage_at
        usage = await self.usage()
        if self._usage_at != usage_at:
            return usage
        if binding_ceiling is None or execution_profile == INTERACTIVE:
            return usage
        for provider in self.adapters:
            if provider in excluded:
                continue
            snapshot = usage.get(provider)
            if snapshot is None or snapshot.binding_percent is None:
                return await self.refresh_usage()
        return usage

    def cached_models(self, provider: str) -> tuple[ProviderModel, ...]:
        """Last models listing for one provider, empty when never probed."""
        return self._model_cache.get(provider, ())

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
        permission_mode: str = "approval",
        provider: str = "",
        model: str = "",
        effort: str = "",
        metered_budget: float | None = None,
        excluded: frozenset[str] = frozenset(),
        context_transfer_tokens: int = 0,
        binding_ceiling: float | None = None,
        execution_profile: str = INTERACTIVE,
        enforce_concurrency: bool = False,
        command_id: str = "",
        goal_id: str = "",
        permitted_providers: frozenset[str] = frozenset(),
        permitted_efforts: frozenset[str] = frozenset(),
        max_concurrency: int = 1,
        policy: PolicyEvaluator | None = None,
        implementation_providers: frozenset[str] = frozenset(),
    ) -> RoutingDecision:
        if metered_budget is not None and not math.isfinite(metered_budget):
            raise ValueError("metered budget must be finite")
        if binding_ceiling is not None and not math.isfinite(binding_ceiling):
            raise ValueError("binding ceiling must be finite")
        usage = await self._admission_usage(
            binding_ceiling=binding_ceiling,
            execution_profile=execution_profile,
            excluded=excluded,
        )
        durable_usage = self.store.latest_usage()
        models = await self.models(Path(session.worktree))
        counts = self.store.active_provider_counts()
        goal_capacity_available = True
        if enforce_concurrency and goal_id:
            active_goal_commands = self._active_other_goal_count(
                goal_id,
                command_id,
            )
            goal_capacity_available = active_goal_commands < max_concurrency
        review_differ_active = False
        if policy is not None:
            review_differ_active = (
                policy.review_provider_must_differ
                and is_review_workload(workload)
                and bool(implementation_providers)
            )
        approval_required = False
        if policy is not None:
            approval_required = policy.require_approval and not provider
        pre_route_decisions: list[PolicyDecision] = []
        gate_notes: dict[str, tuple[str, str]] = {}
        review_excluded = False
        candidates: list[RoutingCandidate] = []
        for provider_id, adapter in self.adapters.items():
            if provider_id in excluded:
                continue
            status = adapter.status()
            if review_differ_active and provider_id in implementation_providers:
                review_excluded = True
                pre_route_decisions.append(
                    PolicyDecision(
                        outcome=BLOCK,
                        rule=RULE_REVIEW_PROVIDER_MUST_DIFFER,
                        reason=(
                            "provider ran implementation turns; the review "
                            "workload requires an independent provider"
                        ),
                        level=LEVEL_SERVER,
                        provider=provider_id,
                        command_id=command_id,
                    )
                )
                continue
            if approval_required and "approval" not in status.capabilities:
                pre_route_decisions.append(
                    PolicyDecision(
                        outcome=BLOCK,
                        rule=RULE_APPROVAL_REQUIRED,
                        reason=(
                            "provider cannot prompt for the policy-required "
                            "dispatch approval"
                        ),
                        level=LEVEL_SERVER,
                        provider=provider_id,
                        command_id=command_id,
                    )
                )
                continue
            if permitted_providers and provider_id not in permitted_providers:
                pre_route_decisions.append(
                    PolicyDecision(
                        outcome=BLOCK,
                        rule=RULE_PERMITTED_PROVIDERS,
                        reason="provider is outside the goal-permitted providers",
                        level=LEVEL_GOAL,
                        provider=provider_id,
                        command_id=command_id,
                    )
                )
                continue
            provider_usage = usage.get(provider_id)
            binding: float | None = None
            credits = False
            if provider_usage is not None:
                binding = provider_usage.binding_percent
                credits = provider_usage.credits_engaged
            if binding is not None:
                if not math.isfinite(binding) or binding < 0:
                    binding = None
            usage_sample = durable_usage.get(provider_id, {})
            safety_ready = status.ready
            unavailable_reason = ""
            if not adapter.supports_permission_mode(permission_mode):
                safety_ready = False
                unavailable_reason = (
                    "provider cannot map the requested permission mode"
                )
            if binding_ceiling is not None:
                # Automatic routing stays fail-closed when binding
                # telemetry is missing, except for Claude. Claude's
                # subscription endpoint is not a provider-execution
                # prerequisite, and its own local usage command is a
                # best-effort fallback. Missing telemetry remains
                # missing in the routing proof rather than being
                # rewritten as invented headroom. Known saturation
                # still blocks every provider below.
                pinned = provider == provider_id
                if (
                    binding is None
                    and execution_profile != INTERACTIVE
                    and not pinned
                    and provider_id != "claude"
                ):
                    safety_ready = False
                    gate_notes[provider_id] = (
                        RULE_BINDING_CEILING,
                        "binding usage is unavailable for the policy "
                        "binding ceiling",
                    )
                if binding is not None and binding >= binding_ceiling:
                    safety_ready = False
                    gate_notes[provider_id] = (
                        RULE_BINDING_CEILING,
                        "binding usage "
                        + str(binding)
                        + " reached the policy binding ceiling "
                        + str(binding_ceiling),
                    )
            if enforce_concurrency and execution_profile == "unattended":
                active = self._active_other_unattended_count(
                    provider_id,
                    command_id,
                )
                if active >= 1:
                    safety_ready = False
                    gate_notes[provider_id] = (
                        RULE_UNATTENDED_CONCURRENCY,
                        "another unattended command is active on this provider",
                    )
            if not goal_capacity_available:
                safety_ready = False
                gate_notes[provider_id] = (
                    RULE_GOAL_CONCURRENCY,
                    "the goal concurrency limit is already consumed",
                )
            chosen_model = _select_model(models.get(provider_id, ()), model)
            if chosen_model is None:
                continue
            chosen_effort = _select_effort(
                chosen_model,
                effort,
                permitted_efforts,
            )
            if chosen_effort is None:
                if permitted_efforts:
                    pre_route_decisions.append(
                        PolicyDecision(
                            outcome=BLOCK,
                            rule=RULE_PERMITTED_EFFORTS,
                            reason=(
                                "no model effort satisfies the goal-permitted "
                                "efforts"
                            ),
                            level=LEVEL_GOAL,
                            provider=provider_id,
                            command_id=command_id,
                        )
                    )
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
                    usage_sample_id=str(usage_sample.get("sample_id", "")),
                    usage_observed_at=str(usage_sample.get("observed_at", "")),
                    unavailable_reason=unavailable_reason,
                )
            )
        if review_excluded and not candidates:
            decision = PolicyDecision(
                outcome=DEFER,
                rule=RULE_REVIEW_PROVIDER_MUST_DIFFER,
                reason=(
                    "only the implementing provider is admissible; deferring "
                    "until an independent provider is available"
                ),
                level=LEVEL_SERVER,
                command_id=command_id,
            )
            self._emit_policy_decisions(
                session.session_id,
                policy,
                [*pre_route_decisions, decision],
            )
            raise PolicyDeferredError(
                RULE_REVIEW_PROVIDER_MUST_DIFFER,
                decision.reason,
            )
        try:
            decision = route(
                candidates,
                required_capabilities=required_capabilities,
                workload=workload,
                manual_provider=provider,
                metered_budget=metered_budget,
                binding_ceiling=binding_ceiling,
                execution_profile=execution_profile,
            )
        except ProviderUnavailableError as error:
            decisions = [
                *pre_route_decisions,
                *self._gate_decisions(
                    policy,
                    error.rejected,
                    gate_notes,
                    command_id=command_id,
                ),
            ]
            if review_excluded:
                deferred = PolicyDecision(
                    outcome=DEFER,
                    rule=RULE_REVIEW_PROVIDER_MUST_DIFFER,
                    reason=(
                        "only the implementing provider remains admissible; "
                        "deferring until an independent provider is available"
                    ),
                    level=LEVEL_SERVER,
                    command_id=command_id,
                )
                self._emit_policy_decisions(
                    session.session_id,
                    policy,
                    [*decisions, deferred],
                )
                raise PolicyDeferredError(
                    RULE_REVIEW_PROVIDER_MUST_DIFFER,
                    deferred.reason,
                ) from error
            self._emit_policy_decisions(
                session.session_id,
                policy,
                decisions,
            )
            raise
        self._emit_policy_decisions(
            session.session_id,
            policy,
            [
                *pre_route_decisions,
                *self._gate_decisions(
                    policy,
                    decision.rejected,
                    gate_notes,
                    command_id=command_id,
                ),
                *self._allow_decision(
                    policy,
                    decision,
                    command_id=command_id,
                ),
            ],
        )
        return decision

    def _gate_decisions(
        self,
        policy: PolicyEvaluator | None,
        rejected: Any,
        gate_notes: dict[str, tuple[str, str]],
        *,
        command_id: str,
    ) -> list[PolicyDecision]:
        if policy is None:
            return []
        return policy.gate_decisions(
            rejected,
            gate_notes,
            command_id=command_id,
        )

    def _allow_decision(
        self,
        policy: PolicyEvaluator | None,
        decision: RoutingDecision,
        *,
        command_id: str,
    ) -> list[PolicyDecision]:
        if policy is None:
            return []
        return [
            policy.allow_decision(
                decision.provider,
                decision.reason,
                command_id=command_id,
            )
        ]

    def _emit_policy_decisions(
        self,
        session_id: str,
        policy: PolicyEvaluator | None,
        decisions: list[PolicyDecision],
    ) -> None:
        if policy is None:
            return
        for decision in decisions:
            self.store.append_event(
                session_id,
                "policy.decision",
                status=decision.outcome,
                metadata=decision.as_dict(),
            )

    def _active_other_unattended_count(
        self,
        provider: str,
        command_id: str,
    ) -> int:
        active = self.store.active_unattended_provider_count(provider)
        if not command_id:
            return active
        envelope = self.store.command_envelope(command_id)
        if envelope["provider"] != provider:
            return active
        if envelope["state"] not in {"reserved", "running", "recovering"}:
            return active
        return max(0, active - 1)

    def _active_other_goal_count(
        self,
        goal_id: str,
        command_id: str,
    ) -> int:
        active = self.store.active_goal_command_count(goal_id)
        if not command_id:
            return active
        envelope = self.store.command_envelope(command_id)
        if envelope["state"] not in {"reserved", "running", "recovering"}:
            return active
        return max(0, active - 1)

    async def status(self, workspace: Path) -> dict[str, Any]:
        self._start_status_refresh(workspace)
        usage = dict(self._usage_cache)
        durable_usage = self.store.latest_usage()
        active_counts = self.store.active_provider_counts()
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
                durable = durable_usage.get(provider, {})
                observed_at = str(durable.get("observed_at", ""))
                fresh = _usage_is_fresh(observed_at)
                binding = snapshot.binding_percent
                admissible = fresh and not snapshot.credits_engaged
                if provider == "claude" and binding is None:
                    admissible = not snapshot.credits_engaged
                if binding is None and provider != "claude":
                    admissible = False
                if binding is not None and binding >= 90.0:
                    admissible = False
                if snapshot.error and provider != "claude":
                    admissible = False
                admission_basis = "measured-usage"
                if provider == "claude" and binding is None:
                    admission_basis = "usage-unavailable-fallback"
                usage_value.update(
                    {
                        "admission_basis": admission_basis,
                        "sample_id": str(durable.get("sample_id", "")),
                        "observed_at": observed_at,
                        "fresh": fresh,
                        "admissible": admissible,
                    }
                )
            result[provider] = {
                "ready": provider_status.ready,
                "detail": provider_status.detail,
                "capabilities": sorted(provider_status.capabilities),
                "active_sessions": active_counts.get(provider, 0),
                "last_error": _last_probe_error(
                    snapshot,
                    durable_usage.get(provider, {}),
                ),
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
        usage_fresh = self._usage_at is not None
        if usage_fresh and self._usage_at is not None:
            usage_fresh = now - self._usage_at <= 60
        if usage_fresh and models_complete:
            return
        self._status_refresh = asyncio.create_task(self._refresh_status(workspace))

    async def _refresh_status(self, workspace: Path) -> None:
        await asyncio.gather(
            self.refresh_usage(),
            self.models(workspace, refresh=True),
        )


def _last_probe_error(
    snapshot: UsageSnapshot | None,
    durable: dict[str, Any],
) -> str:
    if snapshot is not None and snapshot.error:
        return snapshot.error
    stored = durable.get("payload", {})
    if not isinstance(stored, dict):
        return ""
    return str(stored.get("error", ""))


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
            binding_percent=_optional_number(value.get("binding_percent")),
            credits_engaged=bool(value.get("credits_engaged", False)),
            payload=payload,
            error=str(stored.get("error", "")),
        )
    return snapshots


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        normalized = float(value)
        if math.isfinite(normalized) and normalized >= 0:
            return normalized
    return None


def _usage_is_fresh(value: str) -> bool:
    try:
        observed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=datetime.UTC)
    age = datetime.datetime.now(datetime.UTC) - observed.astimezone(datetime.UTC)
    return datetime.timedelta(0) <= age <= datetime.timedelta(seconds=90)


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


def _select_effort(
    model: ProviderModel,
    pinned: str,
    permitted: frozenset[str] = frozenset(),
) -> str | None:
    if pinned:
        if permitted and pinned not in permitted:
            return None
        if not model.efforts or pinned in model.efforts:
            return pinned
        return None
    for candidate in ("xhigh", "high", "medium"):
        if permitted and candidate not in permitted:
            continue
        if candidate in model.efforts:
            return candidate
    if model.efforts:
        for candidate in reversed(model.efforts):
            if not permitted or candidate in permitted:
                return candidate
        return None
    if permitted:
        return None
    return ""


def _fallback_models(provider: str) -> tuple[ProviderModel, ...]:
    efforts = ("low", "medium", "high", "xhigh")
    if provider == "claude":
        return (ProviderModel("opus", "Opus", efforts, None, default=True),)
    if provider == "codex":
        # Avoid synthetic "default" when config.toml pins a real model;
        # ChatGPT-backed Codex rejects model id default.
        from agent_harness.providers.codex import codex_config_model

        configured = codex_config_model()
        if configured:
            return (
                ProviderModel(
                    configured,
                    configured + " (config.toml)",
                    efforts,
                    None,
                    default=True,
                ),
            )
    return (ProviderModel("default", "Account default", efforts, None, default=True),)
