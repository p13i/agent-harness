"""Stable provider adapter contract."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Any


EventHandler = Callable[["ProviderEvent"], Awaitable[None]]
ApprovalHandler = Callable[
    [str, dict[str, Any]],
    Awaitable[dict[str, Any]],
]


@dataclass(frozen=True)
class ProviderEvent:
    event_type: str
    text: str = ""
    status: str = ""
    metadata: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    native_session_id: str = ""
    native_turn_id: str = ""


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    native_session_id: str
    native_turn_id: str
    status: str
    usage: dict[str, Any]
    ambiguous_mutation: bool = False


@dataclass(frozen=True)
class ProviderModel:
    model_id: str
    display_name: str
    efforts: tuple[str, ...]
    context_window: int | None
    default: bool = False
    service_tiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    ready: bool
    detail: str
    capabilities: frozenset[str]


class ProviderAdapter(ABC):
    provider_id: str

    @abstractmethod
    async def run_turn(
        self,
        *,
        workspace: Path,
        prompt: str,
        native_session_id: str,
        permission_mode: str,
        model: str,
        effort: str,
        event_handler: EventHandler,
        approval_handler: ApprovalHandler,
    ) -> ProviderResult:
        raise NotImplementedError

    @abstractmethod
    async def models(self, workspace: Path) -> tuple[ProviderModel, ...]:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> ProviderStatus:
        raise NotImplementedError

    async def interrupt(self) -> None:
        return

    async def steer(self, text: str) -> None:
        del text
        return

    def process_identity(self) -> tuple[int, str]:
        return (0, "")


_SENSITIVE_NAME = re.compile(
    r"(?:TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY)",
    re.IGNORECASE,
)


def provider_environment(provider: str, auth_mode: str = "subscription") -> dict[str, str]:
    environment: dict[str, str] = {}
    for name, value in os.environ.items():
        if name == "npm_config_package":
            continue
        if _SENSITIVE_NAME.search(name):
            keep = False
            if auth_mode == "api":
                if provider == "claude" and name == "ANTHROPIC_API_KEY":
                    keep = True
                if provider == "codex" and name == "OPENAI_API_KEY":
                    keep = True
            if not keep:
                continue
        environment[name] = value
    return environment
