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


@dataclass(frozen=True)
class ProcessGroupMemberIdentity:
    pid: int
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
            if not _group_exists(identity.pgid):
                # The leader and every group member are already gone.
                await _bounded_wait(process, kill_timeout)
                return
        else:
            if current != identity:
                # PID was reused after the original leader exited.
                # Signaling the new process would be unsafe.
                await _bounded_wait(process, kill_timeout)
                return
    members = _process_group_members(identity.pgid)
    if grace_timeout > 0 and await _wait_group_exit(identity.pgid, grace_timeout):
        _signal_group_members(identity.pgid, members, signal.SIGKILL)
        await _bounded_wait(process, kill_timeout)
        return
    _signal_group(identity.pgid, signal.SIGTERM)
    if await _wait_group_exit(identity.pgid, terminate_timeout):
        _signal_group_members(identity.pgid, members, signal.SIGKILL)
        await _bounded_wait(process, kill_timeout)
        return
    _signal_group(identity.pgid, signal.SIGKILL)
    group_exited = await _wait_group_exit(identity.pgid, kill_timeout)
    _signal_group_members(identity.pgid, members, signal.SIGKILL)
    if not group_exited:
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


def _process_group_members(pgid: int) -> tuple[ProcessGroupMemberIdentity, ...]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,pgid="],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return ()
    members: list[ProcessGroupMemberIdentity] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
            observed_pgid = int(fields[1])
        except ValueError:
            continue
        if pid <= 0 or observed_pgid != pgid:
            continue
        members.append(
            ProcessGroupMemberIdentity(
                pid=pid,
                pid_start=_process_start(pid),
            )
        )
    return tuple(members)


def _signal_group_members(
    pgid: int,
    members: tuple[ProcessGroupMemberIdentity, ...],
    selected_signal: signal.Signals,
) -> None:
    del pgid
    for member in members:
        if _process_start(member.pid) != member.pid_start:
            continue
        try:
            os.kill(member.pid, selected_signal)
        except ProcessLookupError:
            continue


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
