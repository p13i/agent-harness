"""Authenticated provider passthrough over a WebSocket."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import pty
import signal
import termios
from typing import Any

from aiohttp import WSMsgType
from aiohttp import web


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
        if process.returncode is None:
            process.send_signal(signal.SIGTERM)
        await process.wait()
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
    if provider == "codex":
        command = ["npx", "-y", "@openai/codex@0.146.0"]
        if permission_mode == "full":
            command.append("--yolo")
        command.extend(arguments)
        return command
    if provider == "claude":
        command = ["npx", "@anthropic-ai/claude-code@2.1.220"]
        if permission_mode == "full":
            command.append("--dangerously-skip-permissions")
        command.extend(arguments)
        return command
    raise ValueError("unsupported terminal provider")
