"""Local service paths and security-sensitive configuration."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
from dataclasses import dataclass
from pathlib import Path

API_VERSION = "1.11.0"
CONTROL_PROTOCOL_VERSION = 3


def _control_build_id() -> str:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


CONTROL_BUILD_ID = _control_build_id()


def runtime_build_id(executable: Path | None = None) -> str:
    """Return the selected verified-bundle identifier, if installed."""
    selected = executable
    if selected is None:
        selected = Path(__file__)
    resolved = selected.expanduser().resolve()
    for parent in resolved.parents:
        if parent.name != "bin":
            continue
        return _manifest_build_id(
            parent.parent / "bundle-manifest.json"
        )
    return ""


def _manifest_build_id(manifest: Path) -> str:
    if manifest.is_symlink():
        return ""
    try:
        payload = json.loads(
            manifest.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    if payload.get("schema") != "p13i/agent-harness/install-bundle/v1":
        return ""
    build_id = payload.get("build_id")
    if not isinstance(build_id, str):
        return ""
    if manifest.parent.name != build_id:
        return ""
    return build_id


@dataclass(frozen=True)
class HarnessPaths:
    state_dir: Path
    runtime: Path
    database: Path
    blobs: Path
    sessions: Path
    worktrees: Path
    exports: Path
    logs: Path
    socket: Path
    daemon_pid: Path
    token: Path
    machine_keys: Path
    sync_lock: Path
    sync_status: Path


def default_state_dir() -> Path:
    return Path.home() / "my" / "chats"


def paths(state_dir: Path | None = None) -> HarnessPaths:
    root = state_dir
    if root is None:
        root = default_state_dir()
    root = root.expanduser().resolve()
    runtime = root / ".runtime"
    return HarnessPaths(
        state_dir=root,
        runtime=runtime,
        database=runtime / "state.sqlite3",
        blobs=root / "blobs",
        sessions=root / "sessions",
        worktrees=runtime / "worktrees",
        exports=root / "exports",
        logs=runtime / "logs",
        socket=runtime / "control.sock",
        daemon_pid=runtime / "daemon.pid",
        token=runtime / "secrets" / "api-token",
        machine_keys=runtime / "secrets" / "machine-keys.json",
        sync_lock=runtime / "sync.lock",
        sync_status=runtime / "sync-status.json",
    )


def public_paths(value: HarnessPaths) -> dict[str, str]:
    """Return discoverable locations without secret contents."""

    return {
        "state_root": str(value.state_dir),
        "runtime": str(value.runtime),
        "database": str(value.database),
        "socket": str(value.socket),
        "token_path": str(value.token),
        "sessions": str(value.sessions),
        "blobs": str(value.blobs),
        "worktrees": str(value.worktrees),
        "exports": str(value.exports),
        "logs": str(value.logs),
    }


def prepare_paths(value: HarnessPaths) -> None:
    directories = (
        value.state_dir,
        value.runtime,
        value.blobs,
        value.sessions,
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
