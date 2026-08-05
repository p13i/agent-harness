"""Stable harness errors shared by the CLI and HTTP API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorDetail:
    code: str
    message: str
    retryable: bool = False
    status: int = 400
    correlation_id: str = ""


class HarnessError(RuntimeError):
    """A sanitized failure safe to expose through a client contract."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status: int = 400,
        correlation_id: str = "",
    ) -> None:
        super().__init__(message)
        self.detail = ErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
            status=status,
            correlation_id=correlation_id,
        )


class NotFoundError(HarnessError):
    def __init__(self, resource: str) -> None:
        super().__init__(
            "E_NOT_FOUND",
            resource + " was not found",
            status=404,
        )


class ConflictError(HarnessError):
    def __init__(self, message: str) -> None:
        super().__init__("E_CONFLICT", message, status=409)


class WorkerOwnershipLostError(RuntimeError):
    """The worker incarnation no longer owns session mutation rights."""


class ProviderUnavailableError(HarnessError):
    def __init__(
        self,
        provider: str,
        *,
        detail: str = "",
        rejected: tuple[dict[str, str], ...] = (),
    ) -> None:
        self.provider = provider
        self.rejected = rejected
        message = provider + " is unavailable"
        if detail:
            message = message + ": " + detail
        super().__init__(
            "E_PROVIDER_UNAVAILABLE",
            message,
            retryable=True,
            status=503,
        )


class ProviderExhaustedError(HarnessError):
    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            "E_PROVIDER_EXHAUSTED",
            provider + " has no eligible capacity",
            retryable=True,
            status=429,
        )


class ReconciliationRequiredError(HarnessError):
    def __init__(
        self,
        reconciliation_id: str,
        *,
        reason: str = "ambiguous-provider-dispatch",
    ) -> None:
        self.reconciliation_id = reconciliation_id
        self.reason = reason
        super().__init__(
            "E_NEEDS_RECONCILIATION",
            "provider dispatch effect is ambiguous; reconciliation is required",
            status=409,
        )


class SafetyGuardError(HarnessError):
    def __init__(
        self,
        reason: str,
        provider: str,
        *,
        recoverable: bool = False,
    ) -> None:
        super().__init__(
            "E_SAFETY_GUARD",
            "execution safety guard stopped " + provider + ": " + reason,
            retryable=recoverable,
            status=429,
        )
        self.reason = reason
        self.provider = provider
        self.recoverable = recoverable


class PolicyBlockedError(HarnessError):
    """A policy rule blocked the dispatch outright."""

    def __init__(self, rule: str, reason: str, *, provider: str = "") -> None:
        self.rule = rule
        self.reason = reason
        self.provider = provider
        super().__init__(
            "E_POLICY_BLOCKED",
            "policy rule " + rule + " blocked the dispatch: " + reason,
            status=403,
        )


class PolicyDeferredError(HarnessError):
    """A policy rule deferred the dispatch until conditions change."""

    def __init__(self, rule: str, reason: str, *, provider: str = "") -> None:
        self.rule = rule
        self.reason = reason
        self.provider = provider
        super().__init__(
            "E_POLICY_DEFERRED",
            "policy rule " + rule + " deferred the dispatch: " + reason,
            retryable=True,
            status=202,
        )


class NeedsReconciliationError(HarnessError):
    def __init__(self) -> None:
        super().__init__(
            "E_NEEDS_RECONCILIATION",
            "a mutating provider action has an ambiguous outcome",
            status=423,
        )
