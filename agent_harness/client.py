"""Authenticated local control-plane client."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from typing import Any

from aiohttp import ClientConnectorError
from aiohttp import ClientSession
from aiohttp import UnixConnector

from agent_harness.config import HarnessPaths
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
        try:
            result = await self.request("GET", "/healthz")
        except (HarnessError, ClientConnectorError, OSError):
            return False
        return result.get("status") == "ok"


async def ensure_daemon(paths: HarnessPaths) -> HarnessClient:
    prepare_paths(paths)
    client = HarnessClient(paths)
    if await client.health():
        return client
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
