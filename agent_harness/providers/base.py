"""Stable provider adapter contract."""

from __future__ import annotations

import os
import re
import shutil
import stat
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EventHandler = Callable[["ProviderEvent"], Awaitable[None]]
PrePromptGate = Callable[[], Awaitable[None]]
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


@dataclass(frozen=True)
class ChildLaunchGate:
    database: Path
    command_id: str
    limit: int


class ProviderAdapter(ABC):
    provider_id: str
    supported_permission_modes: frozenset[str] = frozenset()

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
        child_launch_gate: ChildLaunchGate | None = None,
        pre_prompt_gate: PrePromptGate | None = None,
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

    def native_session_available(
        self,
        workspace: Path,
        native_session_id: str,
    ) -> bool:
        del workspace
        del native_session_id
        return True

    def supports_permission_mode(self, permission_mode: str) -> bool:
        """Report whether the adapter can map one harness permission mode."""
        return permission_mode in self.supported_permission_modes


_SENSITIVE_NAME = re.compile(
    r"(?:TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY|API_KEY)",
    re.IGNORECASE,
)
_SAFE_ENVIRONMENT_NAMES = frozenset(
    {
        "COLORTERM",
        "HOME",
        "LANG",
        "LOGNAME",
        "NO_COLOR",
        "PYTHONDONTWRITEBYTECODE",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "USER",
    }
)
_TRUSTED_EXECUTABLE_SEARCH = os.pathsep.join(
    (
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    )
)


def trusted_executable(name: str) -> str:
    candidate = shutil.which(name, path=_TRUSTED_EXECUTABLE_SEARCH)
    if candidate is None:
        raise RuntimeError("trusted executable was not found: " + name)
    path = Path(candidate)
    resolved = path.resolve()
    details = resolved.stat()
    if not resolved.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError("trusted executable is not runnable: " + name)
    if stat.S_IMODE(details.st_mode) & 0o022:
        raise RuntimeError("trusted executable is writable by another user: " + name)
    return str(path)


def trusted_provider_path() -> str:
    directories = []
    for name in ("node", "npx"):
        parent = str(Path(trusted_executable(name)).parent)
        if parent not in directories:
            directories.append(parent)
    for parent in ("/usr/bin", "/bin"):
        if parent not in directories:
            directories.append(parent)
    return os.pathsep.join(directories)


def provider_npm_cache() -> Path:
    """Return the durable npm cache beneath the private runtime root."""
    return (
        Path.home()
        / "my"
        / "chats"
        / ".runtime"
        / "provider-cache"
        / "npm"
    )


def provider_environment(
    provider: str, auth_mode: str = "subscription"
) -> dict[str, str]:
    environment: dict[str, str] = {}
    for name, value in os.environ.items():
        if name not in _SAFE_ENVIRONMENT_NAMES and not name.startswith("LC_"):
            if name != "CLAUDE_CODE_OAUTH_TOKEN":
                continue
        if name == "CLAUDE_CODE_OAUTH_TOKEN":
            if provider != "claude" or auth_mode != "subscription":
                continue
        if _SENSITIVE_NAME.search(name) and name != "CLAUDE_CODE_OAUTH_TOKEN":
            continue
        environment[name] = value
    if auth_mode == "api":
        credential_name = ""
        if provider == "claude":
            credential_name = "ANTHROPIC_API_KEY"
        if provider == "codex":
            credential_name = "OPENAI_API_KEY"
        credential = os.environ.get(credential_name, "")
        if credential_name and credential:
            environment[credential_name] = credential
    environment["PATH"] = trusted_provider_path()
    environment["npm_config_cache"] = str(provider_npm_cache())
    return environment
