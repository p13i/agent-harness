"""Bounded process-group identity and termination."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Linux /proc/<pid>/stat numbers its fields from one. Field 3 (state)
# is the first field after the parenthesized comm, so field N sits at
# index N - 3 of the fields that follow comm.
_STAT_STATE_INDEX = 0
_STAT_STARTTIME_INDEX = 19


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
                await _reap_terminated_process(process, kill_timeout)
                return
        except ValueError:
            # An identity that cannot be established must never be
            # signaled as though it still belonged to this process.
            await _reap_terminated_process(process, kill_timeout)
            return
        else:
            if current != identity:
                # PID was reused after the original leader exited.
                # Signaling the new process would be unsafe.
                await _reap_terminated_process(process, kill_timeout)
                return
    members = _process_group_members(identity.pgid)
    if grace_timeout > 0 and await _wait_group_exit(identity.pgid, grace_timeout):
        _signal_group_members(identity.pgid, members, signal.SIGKILL)
        if not await _wait_members_exit(members, kill_timeout):
            raise RuntimeError(
                "identity-bound process did not exit after SIGKILL"
            )
        await _reap_terminated_process(process, kill_timeout)
        return
    _signal_group(identity.pgid, signal.SIGTERM)
    if await _wait_group_exit(identity.pgid, terminate_timeout):
        _signal_group_members(identity.pgid, members, signal.SIGKILL)
        if not await _wait_members_exit(members, kill_timeout):
            raise RuntimeError(
                "identity-bound process did not exit after SIGKILL"
            )
        await _reap_terminated_process(process, kill_timeout)
        return
    _signal_group(identity.pgid, signal.SIGKILL)
    group_exited = await _wait_group_exit(identity.pgid, kill_timeout)
    _signal_group_members(identity.pgid, members, signal.SIGKILL)
    members_exited = await _wait_members_exit(members, kill_timeout)
    if not group_exited:
        if not await _wait_group_exit(identity.pgid, kill_timeout):
            raise RuntimeError("process group did not exit after SIGKILL")
    if not members_exited:
        raise RuntimeError("identity-bound process did not exit after SIGKILL")
    await _reap_terminated_process(process, kill_timeout)


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


async def _reap_terminated_process(
    process: asyncio.subprocess.Process,
    timeout: float,
) -> None:
    try:
        await _bounded_wait(process, timeout)
    except RuntimeError:
        # The isolated group and its captured members have already been
        # terminated at every call site. A delayed asyncio child watcher must
        # not turn successful containment into a worker crash and redispatch.
        return


async def _wait_group_exit(pgid: int, timeout: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if not _group_exists(pgid):
            return True
        await asyncio.sleep(0.05)
    return not _group_exists(pgid)


async def _wait_members_exit(
    members: tuple[ProcessGroupMemberIdentity, ...],
    timeout: float,
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if not any(_member_is_alive(member) for member in members):
            return True
        await asyncio.sleep(0.05)
    return not any(_member_is_alive(member) for member in members)


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
        try:
            pid_start = _process_start(pid)
        except ValueError:
            # A member whose identity cannot be established is never
            # recorded, so it is never signaled.
            continue
        members.append(
            ProcessGroupMemberIdentity(
                pid=pid,
                pid_start=pid_start,
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
        try:
            current_start = _process_start(member.pid)
        except ValueError:
            continue
        if current_start != member.pid_start:
            continue
        try:
            os.kill(member.pid, selected_signal)
        except ProcessLookupError:
            continue


def _member_is_alive(member: ProcessGroupMemberIdentity) -> bool:
    try:
        current_start = _process_start(member.pid)
    except ValueError:
        # Identity cannot be established, so this member cannot be
        # claimed as the recorded one.
        return False
    if current_start != member.pid_start:
        return False
    stat_text = _read_proc_stat(member.pid)
    if stat_text is not None:
        fields = _proc_stat_fields_after_comm(stat_text)
        if fields and fields[_STAT_STATE_INDEX] in {"X", "Z"}:
            return False
    try:
        os.kill(member.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_proc_stat(pid: int) -> str | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        return stat_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _proc_stat_fields_after_comm(stat_text: str) -> tuple[str, ...] | None:
    """Return the /proc/<pid>/stat fields that follow field 2 (comm).

    Field 2 is parenthesized and may hold spaces and closing
    parentheses, so the line cannot be split as a whole. Every field
    after comm is the single state character or a number, so the
    final ')' always closes comm. Data without a parenthesized comm
    is malformed and yields no fields.
    """
    prefix, separator, suffix = stat_text.rpartition(")")
    if not separator:
        return None
    if "(" not in prefix:
        return None
    return tuple(suffix.split())


def _process_start(pid: int) -> str:
    stat_text = _read_proc_stat(pid)
    if stat_text is not None:
        fields = _proc_stat_fields_after_comm(stat_text)
        if fields is None or len(fields) <= _STAT_STARTTIME_INDEX:
            # Readable but unparsable stat data cannot identify the
            # process, and a fallback value would let an unrelated
            # process match a recorded identity.
            raise ValueError("process stat data is malformed")
        return fields[_STAT_STARTTIME_INDEX]
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
