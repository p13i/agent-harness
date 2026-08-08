"""Bounded process-group identity and termination."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessGroupIdentity:
    pid: int
    pgid: int
    pid_start: str


def process_group_identity(pid: int) -> ProcessGroupIdentity:
    if pid <= 0:
        raise ValueError("process id must be positive")
    pgid = os.getpgid(pid)
    if pgid != pid:
        raise RuntimeError("process is not the leader of its isolated process group")
    return ProcessGroupIdentity(
        pid=pid,
        pgid=pgid,
        pid_start=_process_start(pid),
    )


async def terminate_process_group(
    process: asyncio.subprocess.Process,
    identity: ProcessGroupIdentity,
    *,
    grace_timeout: float = 0.0,
    terminate_timeout: float = 2.0,
    kill_timeout: float = 2.0,
) -> None:
    if process.pid != identity.pid:
        raise RuntimeError("process identity changed before termination")
    if process.returncode is None:
        try:
            current = process_group_identity(identity.pid)
        except ProcessLookupError:
            # Leader already gone; wait for the asyncio handle only.
            await _bounded_wait(process, kill_timeout)
            return
        if current != identity:
            # PID was reused after the original leader exited. Signaling
            # the new process would be unsafe; treat as already-exited
            # (same outcome as terminate_recorded_process_group's
            # identity-changed path).
            await _bounded_wait(process, kill_timeout)
            return
    if grace_timeout > 0 and await _wait_group_exit(identity.pgid, grace_timeout):
        await _bounded_wait(process, kill_timeout)
        return
    _signal_group(identity.pgid, signal.SIGTERM)
    if await _wait_group_exit(identity.pgid, terminate_timeout):
        await _bounded_wait(process, kill_timeout)
        return
    _signal_group(identity.pgid, signal.SIGKILL)
    if not await _wait_group_exit(identity.pgid, kill_timeout):
        raise RuntimeError("process group did not exit after SIGKILL")
    await _bounded_wait(process, kill_timeout)


async def terminate_recorded_process_group(
    pid: int,
    pid_start: str,
    *,
    terminate_timeout: float = 2.0,
    kill_timeout: float = 2.0,
) -> str:
    if pid <= 0 or not pid_start:
        return "unattached"
    try:
        identity = process_group_identity(pid)
    except ProcessLookupError:
        if _group_exists(pid):
            return "identity-invalid"
        return "already-exited"
    except (OSError, RuntimeError, ValueError):
        return "identity-invalid"
    if identity.pid_start != pid_start:
        return "identity-changed"
    _signal_group(identity.pgid, signal.SIGTERM)
    if await _wait_group_exit(identity.pgid, terminate_timeout):
        return "terminated"
    _signal_group(identity.pgid, signal.SIGKILL)
    if not await _wait_group_exit(identity.pgid, kill_timeout):
        raise RuntimeError("recorded process group did not exit after SIGKILL")
    return "killed"


async def _bounded_wait(
    process: asyncio.subprocess.Process,
    timeout: float,
) -> None:
    if process.returncode is not None:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError as error:
        raise RuntimeError(
            "process leader did not exit within the kill deadline"
        ) from error


async def _wait_group_exit(pgid: int, timeout: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if not _group_exists(pgid):
            return True
        await asyncio.sleep(0.05)
    return not _group_exists(pgid)


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, selected_signal: signal.Signals) -> None:
    try:
        os.killpg(pgid, selected_signal)
    except ProcessLookupError:
        return


def _process_start(pid: int) -> str:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except OSError:
        fields = []
    if len(fields) > 21:
        return fields[21]
    completed = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode == 0:
        value = completed.stdout.strip()
        if value:
            return value
    return str(pid)
