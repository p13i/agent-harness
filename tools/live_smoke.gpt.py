"""Explicitly spend at most two provider calls on resume validation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from agent_harness.client import ensure_daemon
from agent_harness.client import wait_command
from agent_harness.config import paths
from agent_harness.errors import HarnessError
from agent_harness.sdk import AgentHarnessClient


INITIAL_PROMPT = (
    "Reply with exactly LIVE-SMOKE-INITIAL. "
    "Do not use tools or modify files."
)
RESUME_PROMPT = (
    "Reply with exactly LIVE-SMOKE-RESUME. "
    "Do not use tools or modify files."
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run one initial and one resumed live provider turn "
            "inside the strict live-smoke safety profile."
        )
    )
    value.add_argument("provider", choices=("claude", "codex"))
    value.add_argument("--workspace", type=Path, default=Path.cwd())
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
        name="Live resume smoke",
        goal="Validate one bounded initial turn and native resume.",
        permission_mode="read-only",
        direct=True,
        execution_profile="live-smoke",
    )
    session_id = str(session["session_id"])
    command_ids: list[str] = []
    try:
        first = await client.send_message(
            session_id,
            INITIAL_PROMPT,
            provider=arguments.provider,
            effort="low",
            workload="operations",
        )
        command_ids.append(first.command_id)
        first_result = await wait_command(
            client.raw,
            first.command_id,
            timeout=330,
        )
        _require_complete(first_result, "initial")
        first_native = _native_session(first_result)

        second = await client.send_message(
            session_id,
            RESUME_PROMPT,
            provider=arguments.provider,
            effort="low",
            workload="operations",
        )
        command_ids.append(second.command_id)
        second_result = await wait_command(
            client.raw,
            second.command_id,
            timeout=330,
        )
        _require_complete(second_result, "resume")
        second_native = _native_session(second_result)
        if first_native != second_native:
            raise RuntimeError(
                "provider-native session changed across resume"
            )

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
            "provider": arguments.provider,
            "commands": command_ids,
            "native_resume": True,
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
