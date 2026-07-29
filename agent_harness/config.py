"""Local service paths and security-sensitive configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import socket


@dataclass(frozen=True)
class HarnessPaths:
    state_dir: Path
    database: Path
    blobs: Path
    worktrees: Path
    exports: Path
    logs: Path
    socket: Path
    token: Path
    machine_keys: Path


def default_state_dir() -> Path:
    configured = os.environ.get("XDG_STATE_HOME", "")
    if configured:
        return Path(configured) / "p13i-agent-harness"
    return Path.home() / ".local" / "state" / "p13i-agent-harness"


def paths(state_dir: Path | None = None) -> HarnessPaths:
    root = state_dir
    if root is None:
        root = default_state_dir()
    root = root.expanduser().resolve()
    return HarnessPaths(
        state_dir=root,
        database=root / "state.sqlite3",
        blobs=root / "blobs",
        worktrees=root / "worktrees",
        exports=root / "exports",
        logs=root / "logs",
        socket=root / "control.sock",
        token=root / "secrets" / "api-token",
        machine_keys=root / "secrets" / "machine-keys.json",
    )


def prepare_paths(value: HarnessPaths) -> None:
    directories = (
        value.state_dir,
        value.blobs,
        value.worktrees,
        value.exports,
        value.logs,
        value.token.parent,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)


def host_id() -> str:
    return socket.gethostname().casefold()


def api_token(value: HarnessPaths) -> str:
    prepare_paths(value)
    if value.token.exists():
        token = value.token.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(48)
    descriptor = os.open(
        value.token,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return token
