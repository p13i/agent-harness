"""Canonical provider-neutral data models."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
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
    DISPATCHING = "dispatching"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["constraints"] = list(self.constraints)
        value["predicates"] = list(self.predicates)
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


@dataclass(frozen=True)
class RoutingDecision:
    provider: str
    model: str
    effort: str
    reason: str
    ranked: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "effort": self.effort,
            "reason": self.reason,
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

