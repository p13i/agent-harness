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


class ProviderUnavailableError(HarnessError):
    def __init__(self, provider: str) -> None:
        super().__init__(
            "E_PROVIDER_UNAVAILABLE",
            provider + " is unavailable",
            retryable=True,
            status=503,
        )


class ProviderExhaustedError(HarnessError):
    def __init__(self, provider: str) -> None:
        super().__init__(
            "E_PROVIDER_EXHAUSTED",
            provider + " has no eligible capacity",
            retryable=True,
            status=429,
        )


class SafetyGuardError(HarnessError):
    def __init__(
        self,
        reason: str,
        provider: str,
        *,
        recoverable: bool,
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


class NeedsReconciliationError(HarnessError):
    def __init__(self) -> None:
        super().__init__(
            "E_NEEDS_RECONCILIATION",
            "a mutating provider action has an ambiguous outcome",
            status=423,
        )
