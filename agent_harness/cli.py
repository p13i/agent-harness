"""Command-line entry point for p13i/agent-harness."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from agent_harness.api import run_daemon
from agent_harness.blobs import BlobStore
from agent_harness.client import HarnessClient
from agent_harness.client import ensure_daemon
from agent_harness.client import wait_command
from agent_harness.config import paths
from agent_harness.config import prepare_paths
from agent_harness.errors import HarnessError
from agent_harness.ids import new_uuid
from agent_harness.providers.claude import ClaudeAdapter
from agent_harness.providers.codex import CodexAdapter
from agent_harness.scheduler import Scheduler
from agent_harness.storage import StateStore
from agent_harness.tui import HarnessApp
from agent_harness.worker import SessionWorker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="agent-harness",
        description="Durable provider-neutral agent chat workspace.",
    )
    value.add_argument("--state-dir", type=Path)
    value.add_argument("--cwd", type=Path, default=Path.cwd())
    subcommands = value.add_subparsers(dest="command", required=True)

    chat = subcommands.add_parser("chat")
    chat.add_argument("chat_action", nargs="?", choices=("resume", "new"))
    chat.add_argument("session_id", nargs="?")
    chat.add_argument(
        "--permission-mode",
        choices=("approval", "full", "read-only", "plan"),
        default="approval",
    )

    new = subcommands.add_parser("new")
    new.add_argument("--name", default="")
    new.add_argument("--goal", default="")
    new.add_argument(
        "--goal-kind",
        choices=("finite", "invariant"),
        default="finite",
    )
    new.add_argument("--constraint", action="append", default=[])
    new.add_argument("--predicate", action="append", default=[])
    new.add_argument("--budgets", default="{}")
    new.add_argument("--direct", action="store_true")
    new.add_argument(
        "--permission-mode",
        choices=("approval", "full", "read-only", "plan"),
        default="approval",
    )

    resume = subcommands.add_parser("resume")
    resume.add_argument("session_id")

    send = subcommands.add_parser("send")
    send.add_argument("session_id")
    send.add_argument("text")
    send.add_argument("--provider", choices=("claude", "codex"))
    send.add_argument("--model", default="")
    send.add_argument("--effort", default="")
    send.add_argument("--workload", default="implementation")
    send.add_argument("--wait", action="store_true")

    status = subcommands.add_parser("status")
    status.add_argument("session_id", nargs="?")
    subcommands.add_parser("providers")

    events = subcommands.add_parser("events")
    events.add_argument("session_id")
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--limit", type=int, default=1000)

    fork = subcommands.add_parser("fork")
    fork.add_argument("session_id")
    fork.add_argument("--name", default="")

    checkpoint = subcommands.add_parser("checkpoint")
    checkpoint.add_argument("session_id")

    route = subcommands.add_parser("route")
    route.add_argument("session_id")
    route.add_argument("--provider", choices=("claude", "codex"))
    route.add_argument("--model", default="")
    route.add_argument("--effort", default="")
    route.add_argument("--workload", default="implementation")
    route.add_argument(
        "--required-capability",
        action="append",
        default=[],
    )
    route.add_argument("--metered-budget", type=float)

    action = subcommands.add_parser("action")
    action.add_argument("session_id")
    action.add_argument(
        "action",
        choices=("interrupt", "pause", "resume", "stop"),
    )

    evidence = subcommands.add_parser("evidence")
    evidence.add_argument("session_id")
    evidence.add_argument("type")
    evidence.add_argument("subject")
    evidence.add_argument("outcome")
    evidence.add_argument("--value", default="{}")

    export = subcommands.add_parser("export")
    export.add_argument("session_id")

    transfer = subcommands.add_parser("transfer")
    transfer_commands = transfer.add_subparsers(
        dest="transfer_action",
        required=True,
    )
    transfer_create = transfer_commands.add_parser("create")
    transfer_create.add_argument("session_id")
    transfer_create.add_argument("destination_host")
    transfer_create.add_argument("destination_encryption_public")
    transfer_import = transfer_commands.add_parser("import")
    transfer_import.add_argument("envelope_file", type=Path)
    transfer_import.add_argument("source_signing_public")
    transfer_finalize = transfer_commands.add_parser("finalize")
    transfer_finalize.add_argument("session_id")
    transfer_finalize.add_argument("destination_host")
    transfer_finalize.add_argument("owner_epoch", type=int)

    service = subcommands.add_parser("service")
    service.add_argument(
        "service_action",
        choices=("run", "status"),
    )
    service.add_argument("--tcp-host", default="")
    service.add_argument("--tcp-port", type=int, default=0)

    daemon = subcommands.add_parser("daemon")
    daemon.add_argument("--tcp-host", default="")
    daemon.add_argument("--tcp-port", type=int, default=0)

    worker = subcommands.add_parser("worker")
    worker.add_argument("session_id")

    subcommands.add_parser("doctor")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        return asyncio.run(_run(arguments))
    except KeyboardInterrupt:
        return 130
    except HarnessError as error:
        message = error.detail.code + ": " + error.detail.message
        if error.detail.correlation_id:
            message += (
                " [correlation "
                + error.detail.correlation_id
                + "]"
            )
        print(message, file=sys.stderr)
        return 1
    except ValueError as error:
        print("E_INPUT: " + str(error), file=sys.stderr)
        return 2


async def _run(arguments: argparse.Namespace) -> int:
    harness_paths = paths(arguments.state_dir)
    prepare_paths(harness_paths)
    if arguments.command == "daemon":
        await run_daemon(
            harness_paths,
            tcp_host=arguments.tcp_host,
            tcp_port=arguments.tcp_port,
        )
        return 0
    if arguments.command == "service":
        if arguments.service_action == "run":
            await run_daemon(
                harness_paths,
                tcp_host=arguments.tcp_host,
                tcp_port=arguments.tcp_port,
            )
            return 0
        client = HarnessClient(harness_paths)
        healthy = await client.health()
        _print_json({"running": healthy})
        if healthy:
            return 0
        return 1
    if arguments.command == "worker":
        await _worker(harness_paths, arguments.session_id)
        return 0
    if arguments.command == "doctor":
        return await _doctor(harness_paths)
    client = await ensure_daemon(harness_paths)
    if arguments.command == "chat":
        session_id = ""
        if arguments.chat_action == "resume":
            if not arguments.session_id:
                raise ValueError("chat resume requires a session UUID")
            session_id = arguments.session_id
        app = HarnessApp(
            client,
            arguments.cwd.expanduser().resolve(),
            session_id=session_id,
            permission_mode=arguments.permission_mode,
        )
        await asyncio.to_thread(app.run)
        return 0
    if arguments.command == "resume":
        app = HarnessApp(
            client,
            arguments.cwd.expanduser().resolve(),
            session_id=arguments.session_id,
        )
        await asyncio.to_thread(app.run)
        return 0
    if arguments.command == "new":
        predicates = [
            _json_object(value, "--predicate")
            for value in arguments.predicate
        ]
        budgets = _json_object(arguments.budgets, "--budgets")
        result = await client.request(
            "POST",
            "/v1/sessions",
            payload={
                "workspace": str(arguments.cwd.expanduser().resolve()),
                "name": arguments.name,
                "goal": arguments.goal,
                "goal_kind": arguments.goal_kind,
                "constraints": arguments.constraint,
                "predicates": predicates,
                "budgets": budgets,
                "direct": arguments.direct,
                "permission_mode": arguments.permission_mode,
            },
        )
        _print_json(result)
        return 0
    if arguments.command == "send":
        payload = {
            "text": arguments.text,
            "model": arguments.model,
            "effort": arguments.effort,
            "workload": arguments.workload,
        }
        if arguments.provider:
            payload["provider"] = arguments.provider
        result = await client.request(
            "POST",
            "/v1/sessions/" + arguments.session_id + "/messages",
            payload=payload,
            idempotency_key=new_uuid(),
        )
        command = _object(result.get("command"))
        if arguments.wait:
            command_id = str(command.get("command_id", ""))
            command = await wait_command(client, command_id)
        _print_json({"command": command})
        return _command_exit(command)
    if arguments.command == "status":
        path = "/v1/sessions"
        if arguments.session_id:
            path += "/" + arguments.session_id
        _print_json(await client.request("GET", path))
        return 0
    if arguments.command == "providers":
        workspace = str(arguments.cwd.expanduser().resolve())
        result = await client.request(
            "GET",
            "/v1/providers?workspace=" + workspace,
        )
        _print_json(result)
        return 0
    if arguments.command == "events":
        if arguments.after < 0:
            raise ValueError("--after cannot be negative")
        if arguments.limit < 1 or arguments.limit > 5000:
            raise ValueError("--limit must be between 1 and 5000")
        path = (
            "/v1/sessions/"
            + arguments.session_id
            + "/events?after="
            + str(arguments.after)
            + "&limit="
            + str(arguments.limit)
        )
        _print_json(await client.request("GET", path))
        return 0
    if arguments.command == "fork":
        result = await client.request(
            "POST",
            "/v1/sessions/" + arguments.session_id + "/fork",
            payload={"name": arguments.name},
        )
        _print_json(result)
        return 0
    if arguments.command == "checkpoint":
        result = await client.request(
            "POST",
            "/v1/sessions/"
            + arguments.session_id
            + "/checkpoints",
            payload={},
        )
        _print_json(result)
        return 0
    if arguments.command == "route":
        payload = {
            "model": arguments.model,
            "effort": arguments.effort,
            "workload": arguments.workload,
            "required_capabilities": arguments.required_capability,
        }
        if arguments.provider:
            payload["provider"] = arguments.provider
        if arguments.metered_budget is not None:
            payload["metered_budget"] = arguments.metered_budget
        result = await client.request(
            "POST",
            "/v1/sessions/" + arguments.session_id + "/route",
            payload=payload,
        )
        _print_json(result)
        return 0
    if arguments.command == "action":
        result = await client.request(
            "POST",
            "/v1/sessions/"
            + arguments.session_id
            + "/commands/"
            + arguments.action,
            payload={},
            idempotency_key=new_uuid(),
        )
        _print_json(result)
        return 0
    if arguments.command == "evidence":
        raw_value = json.loads(arguments.value)
        if not isinstance(raw_value, dict):
            raise ValueError("--value must be a JSON object")
        result = await client.request(
            "POST",
            "/v1/sessions/" + arguments.session_id + "/evidence",
            payload={
                "type": arguments.type,
                "subject": arguments.subject,
                "outcome": arguments.outcome,
                "value": raw_value,
            },
        )
        _print_json(result)
        return 0
    if arguments.command == "export":
        result = await client.request(
            "POST",
            "/v1/sessions/" + arguments.session_id + "/export",
            payload={},
        )
        _print_json(result)
        return 0
    if arguments.command == "transfer":
        if arguments.transfer_action == "create":
            result = await client.request(
                "POST",
                "/v1/sessions/"
                + arguments.session_id
                + "/transfers",
                payload={
                    "destination_host": arguments.destination_host,
                    "destination_encryption_public": (
                        arguments.destination_encryption_public
                    ),
                },
            )
            _print_json(result)
            return 0
        if arguments.transfer_action == "import":
            envelope = arguments.envelope_file.read_text(
                encoding="utf-8"
            ).strip()
            result = await client.request(
                "POST",
                "/v1/transfers/import",
                payload={
                    "envelope": envelope,
                    "source_signing_public": (
                        arguments.source_signing_public
                    ),
                },
            )
            _print_json(result)
            return 0
        if arguments.transfer_action == "finalize":
            result = await client.request(
                "POST",
                "/v1/sessions/"
                + arguments.session_id
                + "/transfers/finalize",
                payload={
                    "destination_host": arguments.destination_host,
                    "owner_epoch": arguments.owner_epoch,
                },
            )
            _print_json(result)
            return 0
    raise ValueError("unsupported command")


async def _worker(harness_paths: Any, session_id: str) -> None:
    store = StateStore(harness_paths.database)
    blobs = BlobStore(harness_paths.blobs)
    adapters = {
        "claude": ClaudeAdapter(),
        "codex": CodexAdapter(),
    }
    scheduler = Scheduler(store, adapters)
    worker = SessionWorker(
        store,
        blobs,
        scheduler,
        adapters,
        session_id,
    )
    try:
        await worker.run()
    finally:
        store.close()


async def _doctor(harness_paths: Any) -> int:
    checks = {
        "npx": shutil.which("npx") is not None,
        "git": shutil.which("git") is not None,
        "state_directory": harness_paths.state_dir.is_dir(),
        "private_state_mode": (
            harness_paths.state_dir.stat().st_mode & 0o077
        )
        == 0,
    }
    store = StateStore(harness_paths.database)
    try:
        store.list_sessions()
        checks["sqlite"] = True
    finally:
        store.close()
    _print_json({"checks": checks, "ok": all(checks.values())})
    if all(checks.values()):
        return 0
    return 1


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _json_object(value: str, option: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(option + " must be a JSON object")
    return parsed


def _command_exit(command: dict[str, Any]) -> int:
    if command.get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
