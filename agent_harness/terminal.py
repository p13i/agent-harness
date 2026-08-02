"""Authenticated provider passthrough over a WebSocket."""

from __future__ import annotations

import asyncio
import os
import pty
import termios
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from agent_harness.process_control import (
    process_group_identity,
    terminate_process_group,
)
from agent_harness.providers.base import trusted_executable


async def terminal_socket(
    request: web.Request,
    workspace: Path,
) -> web.WebSocketResponse:
    socket = web.WebSocketResponse(heartbeat=20.0, max_msg_size=1024 * 1024)
    await socket.prepare(request)
    initial = await socket.receive_json()
    provider = str(initial.get("provider", ""))
    permission_mode = str(initial.get("permission_mode", "approval"))
    arguments = initial.get("arguments", [])
    if not isinstance(arguments, list):
        arguments = []
    command = _command(
        provider,
        permission_mode,
        [str(item) for item in arguments],
    )
    primary, secondary = pty.openpty()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=workspace,
        stdin=secondary,
        stdout=secondary,
        stderr=secondary,
        start_new_session=True,
    )
    process_group = process_group_identity(process.pid)
    os.close(secondary)
    reader = asyncio.create_task(_pty_to_socket(primary, socket))
    try:
        async for message in socket:
            if message.type == WSMsgType.BINARY:
                await asyncio.to_thread(os.write, primary, message.data)
            elif message.type == WSMsgType.TEXT:
                await _text_message(primary, message.json())
            elif message.type == WSMsgType.ERROR:
                break
    finally:
        await terminate_process_group(process, process_group)
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        os.close(primary)
    return socket


async def _pty_to_socket(
    descriptor: int,
    socket: web.WebSocketResponse,
) -> None:
    while not socket.closed:
        try:
            content = await asyncio.to_thread(os.read, descriptor, 65536)
        except OSError:
            return
        if not content:
            return
        await socket.send_bytes(content)


async def _text_message(descriptor: int, payload: dict[str, Any]) -> None:
    message_type = str(payload.get("type", ""))
    if message_type == "input":
        content = str(payload.get("data", "")).encode("utf-8")
        await asyncio.to_thread(os.write, descriptor, content)
    if message_type == "resize":
        rows = int(payload.get("rows", 24))
        columns = int(payload.get("columns", 80))
        termios.tcsetwinsize(descriptor, (rows, columns))


def _command(
    provider: str,
    permission_mode: str,
    arguments: list[str],
) -> list[str]:
    return native_provider_command(provider, permission_mode, arguments)


def native_provider_command(
    provider: str,
    permission_mode: str,
    arguments: list[str],
) -> list[str]:
    _validate_permission_arguments(arguments)
    if provider == "codex":
        command = [trusted_executable("npx"), "-y", "@openai/codex@0.146.0"]
        if permission_mode == "full":
            command.append("--yolo")
        elif permission_mode == "approval":
            command.extend(
                ["--sandbox", "workspace-write", "--ask-for-approval", "on-request"]
            )
        elif permission_mode in {"plan", "read-only"}:
            command.extend(
                ["--sandbox", "read-only", "--ask-for-approval", "on-request"]
            )
        else:
            raise ValueError("unsupported terminal permission mode")
        command.extend(arguments)
        return command
    if provider == "claude":
        command = [trusted_executable("npx"), "@anthropic-ai/claude-code@2.1.220"]
        if permission_mode == "full":
            command.append("--dangerously-skip-permissions")
        elif permission_mode == "approval":
            command.extend(["--permission-mode", "default"])
        elif permission_mode in {"plan", "read-only"}:
            command.extend(["--permission-mode", "plan"])
        else:
            raise ValueError("unsupported terminal permission mode")
        command.extend(arguments)
        return command
    raise ValueError("unsupported terminal provider")


def _validate_permission_arguments(arguments: list[str]) -> None:
    protected = {
        "--approval-policy",
        "--ask-for-approval",
        "--dangerously-bypass-approvals-and-sandbox",
        "--dangerously-skip-permissions",
        "--permission-mode",
        "--sandbox",
        "--sandbox-mode",
        "--yolo",
        "-a",
        "-s",
    }
    for argument in arguments:
        name = argument.split("=", 1)[0]
        if name in protected:
            raise ValueError("terminal arguments cannot override permission mode")
