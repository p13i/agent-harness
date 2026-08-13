"""Canonical provider-neutral data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Lifecycle(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    ORPHANED = "orphaned"
    TRANSFERRING = "transferring"


class Attention(StrEnum):
    WORKING = "working"
    NEEDS_INPUT = "needs-input"
    NEEDS_RECONCILIATION = "needs-reconciliation"
    READY = "ready"
    IDLE = "idle"
    FAILED = "failed"


class PermissionMode(StrEnum):
    FULL = "full"
    APPROVAL = "approval"
    READ_ONLY = "read-only"
    PLAN = "plan"


class GoalKind(StrEnum):
    FINITE = "finite"
    INVARIANT = "invariant"


class GoalStatus(StrEnum):
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class CommandStatus(StrEnum):
    QUEUED = "queued"
    AWAITING_XHIGH_AUTHORIZATION = "awaiting-xhigh-authorization"
    DISPATCHING = "dispatching"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReconciliationStatus(StrEnum):
    PENDING = "pending"
    RESOLVING = "resolving"
    RESOLVED = "resolved"


class ReconciliationDecision(StrEnum):
    ACCEPT_CURRENT = "accept-current"
    RESTORE_PRE_TURN = "restore-pre-turn"
    STOP = "stop"


@dataclass(frozen=True)
class Session:
    session_id: str
    name: str
    workspace: str
    worktree: str
    lifecycle: str
    attention: str
    permission_mode: str
    active_provider: str
    model: str
    effort: str
    goal_id: str
    owner_host: str
    owner_epoch: int
    created_at: str
    updated_at: str
    archived: bool = False
    external_ref: dict[str, str] = field(default_factory=dict)
    creation_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderAttempt:
    attempt_id: str
    session_id: str
    provider: str
    native_session_id: str
    model: str
    effort: str
    auth_mode: str
    status: str
    started_at: str
    ended_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionEvent:
    session_id: str
    sequence: int
    event_id: str
    event_type: str
    role: str
    text: str
    status: str
    metadata: dict[str, Any]
    blob_digest: str
    turn_id: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommandReceipt:
    command_id: str
    idempotency_key: str
    session_id: str
    command_type: str
    status: str
    result: dict[str, Any]
    created_at: str
    updated_at: str
    turn_ref: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Milestone:
    milestone_id: str
    title: str
    status: str
    dependencies: tuple[str, ...]
    predicates: tuple[dict[str, Any], ...]
    position: int

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dependencies"] = list(self.dependencies)
        value["predicates"] = list(self.predicates)
        return value


@dataclass(frozen=True)
class Goal:
    goal_id: str
    session_id: str
    kind: str
    objective: str
    status: str
    constraints: tuple[str, ...]
    predicates: tuple[dict[str, Any], ...]
    budgets: dict[str, Any]
    created_at: str
    updated_at: str
    milestones: tuple[Milestone, ...] = ()
    permitted_providers: tuple[str, ...] = ()
    permitted_efforts: tuple[str, ...] = ()
    max_concurrency: int = 1
    completion_policy: str = "evidence-all"
    incident_policy: str = "recover-then-pause"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["constraints"] = list(self.constraints)
        value["predicates"] = list(self.predicates)
        value["milestones"] = [item.as_dict() for item in self.milestones]
        value["permitted_providers"] = list(self.permitted_providers)
        value["permitted_efforts"] = list(self.permitted_efforts)
        return value


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    goal_id: str
    evidence_type: str
    subject: str
    outcome: str
    value: dict[str, Any]
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutingCandidate:
    provider: str
    model: str
    effort: str
    ready: bool
    capabilities: frozenset[str]
    quality: float
    binding_percent: float | None
    credits_engaged: bool
    queue_count: int
    affinity: bool
    context_transfer_tokens: int
    usage_sample_id: str = ""
    usage_observed_at: str = ""
    unavailable_reason: str = ""


@dataclass(frozen=True)
class RoutingDecision:
    provider: str
    model: str
    effort: str
    reason: str
    ranked: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, str], ...]
    credits_engaged: bool = False
    binding_percent: float | None = None
    usage_sample_id: str = ""
    usage_observed_at: str = ""
    required_capabilities: tuple[str, ...] = ()
    metered_budget: float | None = None
    binding_ceiling: float | None = None
    execution_profile: str = ""
    workload: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "effort": self.effort,
            "reason": self.reason,
            "credits_engaged": self.credits_engaged,
            "binding_percent": self.binding_percent,
            "usage_sample_id": self.usage_sample_id,
            "usage_observed_at": self.usage_observed_at,
            "required_capabilities": list(self.required_capabilities),
            "metered_budget": self.metered_budget,
            "binding_ceiling": self.binding_ceiling,
            "execution_profile": self.execution_profile,
            "workload": self.workload,
            "ranked": list(self.ranked),
            "rejected": list(self.rejected),
        }


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    session_id: str
    sequence: int
    provider: str
    native_session_id: str
    base_commit: str
    patch_digest: str
    untracked_digest: str
    context_digest: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationRecord:
    reconciliation_id: str
    session_id: str
    command_id: str
    pre_dispatch_checkpoint_id: str
    current_workspace_digest: str
    current_workspace_summary: str
    provider_attempts: tuple[dict[str, Any], ...]
    safety_consumption: dict[str, Any]
    status: str
    resolution: str
    audit: dict[str, Any]
    created_at: str
    resolved_at: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["provider_attempts"] = list(self.provider_attempts)
        return value


@dataclass(frozen=True)
class RestartRecovery:
    requeued_command_ids: tuple[str, ...]
    reconciliations: tuple[ReconciliationRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "requeued_command_ids": list(self.requeued_command_ids),
            "reconciliations": [item.as_dict() for item in self.reconciliations],
        }
