"""Deterministic three-level enterprise policy evaluation.

Policies stack from least to most specific: the optional server-wide
policy file in the harness state directory, per-goal constraints from
the goal envelope, and per-command ``safety_limits``. Every evaluation
emits a uniform ``policy.decision`` event (allow, block, or defer,
with the rule and the reason) into the session event log. The existing
enforcement gates stay in place; this module is the single auditable
decision path their outcomes flow through.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agent_harness.config import HarnessPaths
from agent_harness.safety import SafetyLimits, tighten_limits

ALLOW = "allow"
BLOCK = "block"
DEFER = "defer"

LEVEL_SERVER = "server"
LEVEL_GOAL = "goal"
LEVEL_COMMAND = "command"

RULE_SAFETY_LIMITS = "safety_limits"
RULE_SERVER_POLICY = "server_policy"
RULE_BINDING_CEILING = "binding_ceiling"
RULE_REQUIRED_CAPABILITIES = "required_capabilities"
RULE_PERMITTED_PROVIDERS = "permitted_providers"
RULE_PERMITTED_EFFORTS = "permitted_efforts"
RULE_METERED_BUDGET = "metered_budget"
RULE_CAPACITY_LIMIT = "capacity_limit"
RULE_QUALITY_WINDOW = "quality_window"
RULE_PINNED_PROVIDER = "pinned_provider"
RULE_PROVIDER_READINESS = "provider_readiness"
RULE_REVIEW_PROVIDER_MUST_DIFFER = "review_provider_must_differ"
RULE_APPROVAL_REQUIRED = "approval_required"
RULE_GOAL_CONCURRENCY = "goal_concurrency"
RULE_UNATTENDED_CONCURRENCY = "unattended_concurrency"
RULE_ROUTING = "routing"

SERVER_POLICY_SCHEMA = "p13i/agent-harness/server-policy/v1"
SERVER_POLICY_FILE = "policy.json"
REVIEW_WORKLOADS = frozenset({"review", "code review", "code-review", "code_review"})


@dataclass(frozen=True)
class PolicyDecision:
    """One uniform, auditable policy outcome."""

    outcome: str
    rule: str
    reason: str
    level: str = ""
    provider: str = ""
    command_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_review_workload(workload: str) -> bool:
    return workload.strip().casefold() in REVIEW_WORKLOADS


def server_policy_path(paths: HarnessPaths) -> Path:
    """Server-wide policy file beside the harness-global state records."""

    return paths.state_dir / SERVER_POLICY_FILE


def load_server_policy(state_dir: Path | None) -> dict[str, Any]:
    """Load the optional server-wide policy file.

    An absent file (or an absent state directory) means no server
    policy. A present but malformed file raises ``ValueError`` so the
    dispatch path can fail closed with a recorded decision.
    """

    if state_dir is None:
        return {}
    path = state_dir.expanduser() / SERVER_POLICY_FILE
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("server policy file is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("server policy must be a JSON object")
    if payload.get("schema") != SERVER_POLICY_SCHEMA:
        raise ValueError("server policy schema is unsupported")
    safety_limits = payload.get("safety_limits", {})
    if not isinstance(safety_limits, dict):
        raise ValueError("server policy safety_limits must be an object")
    for name in ("review_provider_must_differ", "require_approval"):
        value = payload.get(name, False)
        if not isinstance(value, bool):
            raise ValueError("server policy " + name + " must be boolean")
    return payload


def implementation_providers(
    attempts: Iterable[Any],
    routings: Iterable[dict[str, Any]],
) -> frozenset[str]:
    """Providers that ran the session's non-review turns.

    Derived deterministically from the provider attempts, with the
    recorded routing decisions distinguishing review-only providers.
    """

    review_only: set[str] = set()
    implemented: set[str] = set()
    for row in routings:
        provider = str(row.get("provider", ""))
        if not provider:
            continue
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        workload = str(payload.get("workload", ""))
        if is_review_workload(workload):
            if provider not in implemented:
                review_only.add(provider)
            continue
        implemented.add(provider)
        review_only.discard(provider)
    for attempt in attempts:
        provider = str(getattr(attempt, "provider", ""))
        if not provider:
            continue
        if provider in review_only:
            continue
        implemented.add(provider)
    return frozenset(implemented)


def limit_decisions(
    level: str,
    before: SafetyLimits,
    after: SafetyLimits,
    *,
    command_id: str = "",
) -> list[PolicyDecision]:
    """One allow decision per limit field a stacking level changed."""

    decisions: list[PolicyDecision] = []
    for name, previous in asdict(before).items():
        current = asdict(after)[name]
        if current == previous:
            continue
        decisions.append(
            PolicyDecision(
                outcome=ALLOW,
                rule=RULE_SAFETY_LIMITS,
                reason=(
                    level
                    + " level set "
                    + name
                    + " from "
                    + str(previous)
                    + " to "
                    + str(current)
                ),
                level=level,
                command_id=command_id,
                metadata={
                    "field": name,
                    "before": previous,
                    "after": current,
                },
            )
        )
    return decisions


_REJECTION_RULES = {
    "another provider was pinned": RULE_PINNED_PROVIDER,
    "required capabilities are unavailable": RULE_REQUIRED_CAPABILITIES,
    "metered credits require a positive explicit budget": RULE_METERED_BUDGET,
    "metered credits require exact cost reporting": RULE_METERED_BUDGET,
    "binding usage is at or above 90 percent": RULE_CAPACITY_LIMIT,
    "quality is outside the eligible window": RULE_QUALITY_WINDOW,
    "binding usage is not finite": RULE_BINDING_CEILING,
    "binding usage is negative": RULE_BINDING_CEILING,
}

_GATE_LEVELS = {
    RULE_PERMITTED_PROVIDERS: LEVEL_GOAL,
    RULE_PERMITTED_EFFORTS: LEVEL_GOAL,
}


class PolicyEvaluator:
    """Server policy rules plus uniform gate-decision mapping."""

    def __init__(self, server_policy: dict[str, Any] | None = None) -> None:
        if server_policy is None:
            server_policy = {}
        self._server_policy = server_policy

    @property
    def review_provider_must_differ(self) -> bool:
        return bool(self._server_policy.get("review_provider_must_differ", False))

    @property
    def require_approval(self) -> bool:
        return bool(self._server_policy.get("require_approval", False))

    @property
    def server_safety_limits(self) -> dict[str, Any]:
        value = self._server_policy.get("safety_limits", {})
        if not isinstance(value, dict):
            return {}
        return dict(value)

    def apply_server_limits(
        self,
        limits: SafetyLimits,
        *,
        command_id: str = "",
    ) -> tuple[SafetyLimits, list[PolicyDecision]]:
        """Tighten the profile limits with the server-wide policy file."""

        requested = self.server_safety_limits
        if not requested:
            return (limits, [])
        tightened = tighten_limits(limits, requested)
        decisions = limit_decisions(
            LEVEL_SERVER,
            limits,
            tightened,
            command_id=command_id,
        )
        return (tightened, decisions)

    def gate_decisions(
        self,
        rejected: Iterable[dict[str, str]],
        gate_notes: dict[str, tuple[str, str]],
        *,
        command_id: str = "",
    ) -> list[PolicyDecision]:
        """Map routing gate rejections to uniform block decisions.

        ``gate_notes`` refines bare readiness rejections with the
        specific gate the scheduler enforced, keyed by provider.
        """

        decisions: list[PolicyDecision] = []
        for entry in rejected:
            provider = str(entry.get("provider", ""))
            reason = str(entry.get("reason", ""))
            rule = _REJECTION_RULES.get(reason, "")
            level = _GATE_LEVELS.get(rule, "")
            if reason == "provider is not ready":
                note = gate_notes.get(provider)
                if note is not None:
                    rule, reason = note
                    level = _GATE_LEVELS.get(rule, "")
            if not rule:
                rule = RULE_ROUTING
            decisions.append(
                PolicyDecision(
                    outcome=BLOCK,
                    rule=rule,
                    reason=reason,
                    level=level,
                    provider=provider,
                    command_id=command_id,
                )
            )
        return decisions

    def allow_decision(
        self,
        provider: str,
        reason: str,
        *,
        rule: str = RULE_ROUTING,
        level: str = "",
        command_id: str = "",
    ) -> PolicyDecision:
        return PolicyDecision(
            outcome=ALLOW,
            rule=rule,
            reason=reason,
            level=level,
            provider=provider,
            command_id=command_id,
        )
