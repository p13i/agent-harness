"""Bounded deterministic synchronization for portable chat records."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from agent_harness.config import HarnessPaths
from agent_harness.ids import utc_now
from agent_harness.records import materialize_all
from agent_harness.records import materialize_session
from agent_harness.storage import StateStore


MANAGED_PATHS = (
    "sessions",
    "blobs",
    "exports",
    "global.gpt.json",
)


def publish_session(
    paths: HarnessPaths,
    store: StateStore,
    session_id: str,
) -> dict[str, Any]:
    try:
        materialized = materialize_session(paths, store, session_id)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ):
        return _pending_status(paths, "record-materialization")
    try:
        result = sync_repository(paths)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ):
        return _pending_status(paths, "repository-synchronization")
    result["materialized"] = materialized
    return result


def publish_all(
    paths: HarnessPaths,
    store: StateStore,
) -> dict[str, Any]:
    try:
        materialized = materialize_all(paths, store)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ):
        return _pending_status(paths, "record-materialization")
    try:
        result = sync_repository(paths)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ):
        return _pending_status(paths, "repository-synchronization")
    result["materialized_sessions"] = len(materialized)
    return result


def sync_repository(
    paths: HarnessPaths,
    *,
    attempts: int = 3,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("sync attempts must be positive")
    if not _is_repository(paths.state_dir):
        return _write_status(
            paths,
            {
                "state": "not-configured",
                "pending": False,
                "detail": "data root is not a Git repository",
            },
        )
    paths.sync_lock.parent.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o700,
    )
    descriptor = os.open(
        paths.sync_lock,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
    with os.fdopen(descriptor, "r+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _sync_locked(paths, attempts)


def read_sync_status(paths: HarnessPaths) -> dict[str, Any]:
    if not paths.sync_status.is_file():
        return {
            "state": "unknown",
            "pending": False,
            "updated_at": "",
        }
    try:
        value = json.loads(
            paths.sync_status.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {
            "state": "invalid",
            "pending": True,
            "updated_at": "",
        }
    if not isinstance(value, dict):
        return {
            "state": "invalid",
            "pending": True,
            "updated_at": "",
        }
    return value


def _sync_locked(
    paths: HarnessPaths,
    attempts: int,
) -> dict[str, Any]:
    _git(paths.state_dir, "add", "--", *MANAGED_PATHS)
    staged = _git(
        paths.state_dir,
        "diff",
        "--cached",
        "--quiet",
        check=False,
    )
    if staged.returncode not in {0, 1}:
        return _pending_status(paths, "git-index")
    if staged.returncode == 1:
        commit = _git(
            paths.state_dir,
            "commit",
            "-m",
            "Snapshot agent chats",
            check=False,
        )
        if commit.returncode != 0:
            return _pending_status(paths, "git-commit")
    for index in range(attempts):
        fetched = _git(
            paths.state_dir,
            "fetch",
            "origin",
            "main",
            check=False,
        )
        if fetched.returncode != 0:
            if index + 1 < attempts:
                time.sleep(2**index)
                continue
            return _pending_status(paths, "git-fetch")
        rebased = _git(
            paths.state_dir,
            "rebase",
            "origin/main",
            check=False,
        )
        if rebased.returncode != 0:
            _git(
                paths.state_dir,
                "rebase",
                "--abort",
                check=False,
            )
            return _write_status(
                paths,
                {
                    "state": "conflict",
                    "pending": True,
                    "detail": "remote history conflicts with local records",
                },
            )
        pushed = _git(
            paths.state_dir,
            "push",
            "origin",
            "HEAD:main",
            check=False,
            timeout=180,
        )
        if pushed.returncode == 0:
            return _write_status(
                paths,
                {
                    "state": "synced",
                    "pending": False,
                    "detail": "",
                    "commit": _head(paths.state_dir),
                },
            )
        if index + 1 < attempts:
            time.sleep(2**index)
    return _pending_status(paths, "git-push")


def _pending_status(
    paths: HarnessPaths,
    detail: str,
) -> dict[str, Any]:
    return _write_status(
        paths,
        {
            "state": "pending",
            "pending": True,
            "detail": detail,
            "commit": _head(paths.state_dir),
        },
    )


def _write_status(
    paths: HarnessPaths,
    value: dict[str, Any],
) -> dict[str, Any]:
    result = dict(value)
    result["updated_at"] = utc_now()
    paths.sync_status.parent.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o700,
    )
    temporary = paths.sync_status.with_name(
        paths.sync_status.name + ".tmp"
    )
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, paths.sync_status)
    return result


def _head(root: Path) -> str:
    result = _git(root, "rev-parse", "HEAD", check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _is_repository(root: Path) -> bool:
    result = _git(
        root,
        "rev-parse",
        "--show-toplevel",
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        return Path(result.stdout.strip()).resolve() == root.resolve()
    except OSError:
        return False


def _git(
    root: Path,
    *arguments: str,
    check: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "chat repository Git operation timed out"
        ) from error
    if check and completed.returncode != 0:
        raise RuntimeError("chat repository Git operation failed")
    return completed
