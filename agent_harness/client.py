"""Authenticated local control-plane client."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
from typing import Any

from aiohttp import ClientConnectorError
from aiohttp import ClientSession
from aiohttp import UnixConnector

from agent_harness.config import HarnessPaths
from agent_harness.config import CONTROL_BUILD_ID
from agent_harness.config import CONTROL_PROTOCOL_VERSION
from agent_harness.config import api_token
from agent_harness.config import prepare_paths
from agent_harness.errors import HarnessError
from agent_harness.runtime import launcher_command


class HarnessClient:
    def __init__(self, paths: HarnessPaths) -> None:
        self.paths = paths
        self.token = api_token(paths)

    async def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        connector = UnixConnector(path=str(self.paths.socket))
        headers = {"Authorization": "Bearer " + self.token}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        async with ClientSession(
            connector=connector,
            base_url="http://localhost",
            headers=headers,
        ) as session:
            async with session.request(
                method,
                path,
                json=payload,
            ) as response:
                content = await response.text()
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as error:
                    raise HarnessError(
                        "E_PROTOCOL",
                        "harness returned invalid JSON",
                        status=502,
                    ) from error
                if response.status >= 400:
                    error_value = result.get("error", {})
                    if not isinstance(error_value, dict):
                        error_value = {}
                    raise HarnessError(
                        str(error_value.get("code", "E_REMOTE")),
                        str(error_value.get("message", "request failed")),
                        retryable=bool(
                            error_value.get("retryable", False)
                        ),
                        status=response.status,
                        correlation_id=str(
                            error_value.get("correlation_id", "")
                        ),
                    )
                if not isinstance(result, dict):
                    raise HarnessError(
                        "E_PROTOCOL",
                        "harness returned a non-object response",
                        status=502,
                    )
                return result

    async def health(self) -> bool:
        result = await self._health_payload()
        return _compatible_health(result)

    async def _health_payload(self) -> dict[str, Any]:
        try:
            result = await self.request("GET", "/healthz")
        except (HarnessError, ClientConnectorError, OSError):
            return {}
        return result


async def ensure_daemon(paths: HarnessPaths) -> HarnessClient:
    prepare_paths(paths)
    client = HarnessClient(paths)
    health = await client._health_payload()
    if _compatible_health(health):
        return client
    if health.get("status") == "ok":
        await _stop_incompatible_daemon(paths, client)
    command = [
        *launcher_command(),
        "--state-dir",
        str(paths.state_dir),
        "daemon",
    ]
    log_path = paths.logs / "daemon.log"
    with log_path.open("ab", buffering=0) as log:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    for unused in range(100):
        del unused
        if await client.health():
            return client
        await asyncio.sleep(0.1)
    raise HarnessError(
        "E_DAEMON",
        "harness daemon did not become ready; inspect " + str(log_path),
        retryable=True,
        status=503,
    )


async def stop_daemon(paths: HarnessPaths) -> bool:
    client = HarnessClient(paths)
    if not await client.health():
        return False
    pids = _managed_daemon_pids(paths)
    if not pids:
        raise HarnessError(
            "E_DAEMON_STOP",
            "the running harness daemon could not be identified",
            retryable=True,
            status=503,
        )
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    for unused in range(150):
        del unused
        if not await client.health():
            return True
        await asyncio.sleep(0.1)
    raise HarnessError(
        "E_DAEMON_STOP",
        "the harness daemon did not stop cleanly",
        retryable=True,
        status=503,
    )


def _compatible_health(value: dict[str, Any]) -> bool:
    if value.get("status") != "ok":
        return False
    if (
        value.get("control_protocol_version")
        != CONTROL_PROTOCOL_VERSION
    ):
        return False
    return value.get("control_build_id") == CONTROL_BUILD_ID


async def _stop_incompatible_daemon(
    paths: HarnessPaths,
    client: HarnessClient,
) -> None:
    pids = _managed_daemon_pids(paths)
    if not pids:
        raise HarnessError(
            "E_DAEMON_UPGRADE",
            "an incompatible harness daemon is running, but its managed "
            "process could not be identified",
            retryable=True,
            status=503,
        )
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    for unused in range(50):
        del unused
        health = await client._health_payload()
        if health.get("status") != "ok":
            return
        await asyncio.sleep(0.1)
    raise HarnessError(
        "E_DAEMON_UPGRADE",
        "the incompatible harness daemon did not stop cleanly",
        retryable=True,
        status=503,
    )


def _managed_daemon_pids(paths: HarnessPaths) -> tuple[int, ...]:
    candidates: set[int] = set()
    if paths.daemon_pid.exists():
        try:
            candidates.add(
                int(paths.daemon_pid.read_text(encoding="utf-8").strip())
            )
        except ValueError:
            pass
    managed = _filter_managed_daemon_pids(candidates)
    if managed:
        return managed
    completed = subprocess.run(
        ["lsof", "-t", str(paths.socket)],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        try:
            candidates.add(int(line))
        except ValueError:
            continue
    return _filter_managed_daemon_pids(candidates)


def _filter_managed_daemon_pids(
    candidates: set[int],
) -> tuple[int, ...]:
    managed: list[int] = []
    for pid in sorted(candidates):
        if pid <= 1 or pid == os.getpid():
            continue
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
        command = completed.stdout.casefold()
        if "agent-harness" not in command:
            continue
        if " daemon" not in command:
            continue
        managed.append(pid)
    return tuple(managed)


async def wait_command(
    client: HarnessClient,
    command_id: str,
    *,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = await client.request(
            "GET",
            "/v1/commands/" + command_id,
        )
        command = result.get("command", {})
        if not isinstance(command, dict):
            command = {}
        if command.get("status") in {
            "complete",
            "failed",
            "cancelled",
        }:
            return command
        if asyncio.get_running_loop().time() >= deadline:
            return command
        await asyncio.sleep(0.2)


def read_projection(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("projection must contain an object")
    return value
