"""Explicitly spend one bounded call on each supported provider."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from agent_harness.client import ensure_daemon
from agent_harness.client import wait_command
from agent_harness.config import paths
from agent_harness.errors import HarnessError
from agent_harness.sdk import AgentHarnessClient


PROVIDER_PROMPTS = {
    "claude": (
        "Reply with exactly LIVE-SMOKE-CLAUDE. "
        "Do not use tools or modify files."
    ),
    "codex": (
        "Reply with exactly LIVE-SMOKE-CODEX. "
        "Do not use tools or modify files."
    ),
}


def default_workspace() -> Path:
    value = os.environ.get("BUILD_WORKING_DIRECTORY", "").strip()
    if value:
        return Path(value)
    return Path.cwd()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run one Claude and one Codex live provider turn "
            "inside the strict live-smoke safety profile."
        )
    )
    value.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace(),
    )
    value.add_argument("--state-dir", type=Path)
    value.add_argument(
        "--confirm-spend",
        action="store_true",
        help="Acknowledge that this invokes a live subscription provider.",
    )
    return value


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    if not arguments.confirm_spend:
        raise ValueError(
            "--confirm-spend is required; no provider was invoked"
        )
    harness_paths = paths(arguments.state_dir)
    await ensure_daemon(harness_paths)
    client = AgentHarnessClient(harness_paths)
    workspace = arguments.workspace.expanduser().resolve()
    session = await client.create_session(
        workspace,
        name="Live cross-provider smoke",
        goal="Validate one bounded read-only turn per provider.",
        permission_mode="read-only",
        direct=True,
        execution_profile="live-smoke",
    )
    session_id = str(session["session_id"])
    command_ids: list[str] = []
    native_sessions: dict[str, str] = {}
    try:
        for provider in ("claude", "codex"):
            receipt = await client.send_message(
                session_id,
                PROVIDER_PROMPTS[provider],
                provider=provider,
                effort="low",
                workload="operations",
            )
            command_ids.append(receipt.command_id)
            result = await wait_command(
                client.raw,
                receipt.command_id,
                timeout=330,
            )
            _require_complete(result, provider)
            native_sessions[provider] = _native_session(result)

        safety = await client.usage(session_id)
        envelopes = safety.get("envelopes", [])
        if not isinstance(envelopes, list) or len(envelopes) != 2:
            raise RuntimeError(
                "live smoke did not record exactly two envelopes"
            )
        for envelope in envelopes:
            if not isinstance(envelope, dict):
                raise RuntimeError("invalid live-smoke envelope")
            if envelope.get("profile") != "live-smoke":
                raise RuntimeError("live-smoke profile was not enforced")
            limits = envelope.get("limits", {})
            if not isinstance(limits, dict):
                raise RuntimeError("live-smoke limits are absent")
            if limits.get("max_attempts") != 1:
                raise RuntimeError("live smoke permitted provider recovery")
        return {
            "session_id": session_id,
            "providers": ["claude", "codex"],
            "commands": command_ids,
            "native_sessions": native_sessions,
            "profile": "live-smoke",
        }
    finally:
        try:
            await client.command(session_id, "stop")
        except HarnessError:
            pass


def _require_complete(
    command: dict[str, object],
    label: str,
) -> None:
    if command.get("status") == "complete":
        return
    result = command.get("result", {})
    raise RuntimeError(
        label
        + " provider turn did not complete: "
        + json.dumps(result, sort_keys=True)
    )


def _native_session(command: dict[str, object]) -> str:
    result = command.get("result", {})
    if not isinstance(result, dict):
        raise RuntimeError("provider result is absent")
    value = str(result.get("native_session_id", ""))
    if not value:
        raise RuntimeError("provider did not return a native session id")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = asyncio.run(run(arguments))
    except (HarnessError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
