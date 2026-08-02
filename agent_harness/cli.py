"""Command-line entry point for p13i/agent-harness."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import shutil
import stat
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_harness.api import run_daemon
from agent_harness.blobs import BlobStore
from agent_harness.client import HarnessClient, ensure_daemon, stop_daemon, wait_command
from agent_harness.config import (
    CONTROL_BUILD_ID,
    CONTROL_PROTOCOL_VERSION,
    paths,
    prepare_paths,
    public_paths,
)
from agent_harness.errors import HarnessError
from agent_harness.ids import new_uuid
from agent_harness.migration import migrate_state
from agent_harness.providers.base import trusted_executable
from agent_harness.providers.claude import ClaudeAdapter
from agent_harness.providers.codex import CodexAdapter
from agent_harness.scheduler import Scheduler
from agent_harness.service_manager import SystemdUserService, UnitConfiguration
from agent_harness.storage import StateStore
from agent_harness.sync import publish_all, read_sync_status
from agent_harness.tui import run_tui
from agent_harness.usage import provider_auth_ready
from agent_harness.worker import SessionWorker
from tools.bundle import BundleError, verify_bundle
from tools.install import InstallError, default_launcher, read_selection


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
    new.add_argument(
        "--execution-profile",
        choices=("interactive", "unattended", "live-smoke"),
        default="interactive",
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
    subcommands.add_parser("quiescence")
    usage = subcommands.add_parser("usage")
    usage.add_argument("session_id")
    extend = subcommands.add_parser("extend-budget")
    extend.add_argument("session_id")
    extend.add_argument("--seconds", type=int)
    extend.add_argument("--tokens", type=int)
    extend.add_argument("--allow-xhigh-once", action="store_true")
    extend.add_argument("--command-id", default="")
    extend.add_argument("--provider", choices=("claude", "codex"), default="")
    extend.add_argument("--reason", required=True)
    subcommands.add_parser("providers")
    subcommands.add_parser("capabilities")

    paths_command = subcommands.add_parser("paths")
    paths_command.add_argument("--json", action="store_true")

    events = subcommands.add_parser("events")
    events.add_argument("session_id")
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--limit", type=int, default=1000)

    fork = subcommands.add_parser("fork")
    fork.add_argument("session_id")
    fork.add_argument("--name", default="")

    checkpoint = subcommands.add_parser("checkpoint")
    checkpoint.add_argument("session_id")

    archive = subcommands.add_parser("archive")
    archive.add_argument("session_id")
    unarchive = subcommands.add_parser("unarchive")
    unarchive.add_argument("session_id")

    reconcile = subcommands.add_parser("reconcile")
    reconcile_commands = reconcile.add_subparsers(
        dest="reconcile_action",
        required=True,
    )
    reconcile_list = reconcile_commands.add_parser("list")
    reconcile_list.add_argument("session_id")
    reconcile_inspect = reconcile_commands.add_parser("inspect")
    reconcile_inspect.add_argument("reconciliation_id")
    reconcile_resolve = reconcile_commands.add_parser("resolve")
    reconcile_resolve.add_argument("reconciliation_id")
    reconcile_resolve.add_argument(
        "decision",
        choices=("accept-current", "restore-pre-turn", "stop"),
    )
    reconcile_resolve.add_argument(
        "--observed-workspace-digest",
        required=True,
    )
    reconcile_resolve.add_argument("--approval-id", default="")
    reconcile_resolve.add_argument("--audit", default="{}")

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
        choices=(
            "install",
            "start",
            "restart",
            "status",
            "stop",
            "uninstall",
            "run",
        ),
    )
    service.add_argument("--tcp-host", default="")
    service.add_argument("--tcp-port", type=int, default=0)

    daemon = subcommands.add_parser("daemon")
    daemon.add_argument("--tcp-host", default="")
    daemon.add_argument("--tcp-port", type=int, default=0)

    worker = subcommands.add_parser("worker")
    worker.add_argument("session_id")

    subcommands.add_parser("sync")
    subcommands.add_parser("sync-status")
    migrate = subcommands.add_parser("migrate-state")
    migrate.add_argument(
        "--from",
        dest="source_root",
        type=Path,
        required=True,
    )
    migrate.add_argument(
        "--to",
        dest="destination_root",
        type=Path,
        required=True,
    )
    migrate.add_argument("--trash-source", action="store_true")

    subcommands.add_parser("doctor")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command in {"chat", "resume"}:
            return _run_tui_command(arguments)
        return asyncio.run(_run(arguments))
    except KeyboardInterrupt:
        return 130
    except HarnessError as error:
        message = error.detail.code + ": " + error.detail.message
        if error.detail.correlation_id:
            message += " [correlation " + error.detail.correlation_id + "]"
        print(message, file=sys.stderr)
        return 1
    except ValueError as error:
        print("E_INPUT: " + str(error), file=sys.stderr)
        return 2
    except RuntimeError as error:
        print("E_RUNTIME: " + str(error), file=sys.stderr)
        return 1


def _run_tui_command(arguments: argparse.Namespace) -> int:
    harness_paths = paths(arguments.state_dir)
    prepare_paths(harness_paths)
    client = asyncio.run(ensure_daemon(harness_paths))
    session_id = ""
    permission_mode = "approval"
    if arguments.command == "chat":
        permission_mode = arguments.permission_mode
        if arguments.chat_action == "resume":
            if not arguments.session_id:
                raise ValueError("chat resume requires a session UUID")
            session_id = arguments.session_id
    else:
        session_id = arguments.session_id
    run_tui(
        client,
        arguments.cwd.expanduser().resolve(),
        session_id=session_id,
        permission_mode=permission_mode,
    )
    return 0


async def _run(arguments: argparse.Namespace) -> int:
    if arguments.command == "migrate-state":
        result = await asyncio.to_thread(
            migrate_state,
            arguments.source_root,
            arguments.destination_root,
            trash_source=arguments.trash_source,
        )
        _print_json(result)
        return 0
    harness_paths = paths(arguments.state_dir)
    prepare_paths(harness_paths)
    if arguments.command == "paths":
        _print_json(public_paths(harness_paths))
        return 0
    if arguments.command == "quiescence":
        return await _quiescence(harness_paths)
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
        manager = _service_manager()
        if arguments.service_action == "install":
            selection = _installed_selection()
            manager.install(
                UnitConfiguration(
                    executable=selection.executable,
                    state_dir=harness_paths.state_dir,
                    build_id=selection.build_id,
                )
            )
            _print_json(
                {
                    "installed": True,
                    "build_id": selection.build_id,
                    "unit": str(manager.unit_path),
                }
            )
            return 0
        if arguments.service_action == "start":
            manager.start()
            _print_json({"started": True})
            return 0
        if arguments.service_action == "restart":
            manager.restart()
            _print_json({"restarted": True})
            return 0
        if arguments.service_action == "stop":
            status = manager.status()
            if status.installed:
                manager.stop()
                _print_json({"stopped": True, "service": True})
                return 0
            stopped = await stop_daemon(harness_paths)
            _print_json(
                {
                    "running": False,
                    "stopped": stopped,
                    "service": False,
                }
            )
            return 0
        if arguments.service_action == "uninstall":
            manager.uninstall()
            _print_json(
                {
                    "installed": False,
                    "state_preserved": str(harness_paths.state_dir),
                }
            )
            return 0
        status = manager.status()
        if not status.installed:
            client = HarnessClient(harness_paths)
            healthy = await client.health()
            _print_json(
                {
                    "active": healthy,
                    "installed": False,
                    "detail": "local daemon",
                    "control_build_id": CONTROL_BUILD_ID,
                    "control_protocol_version": (CONTROL_PROTOCOL_VERSION),
                }
            )
            if healthy:
                return 0
            return 1
        _print_json(
            {
                **asdict(status),
                "control_build_id": CONTROL_BUILD_ID,
                "control_protocol_version": (CONTROL_PROTOCOL_VERSION),
            }
        )
        if status.active:
            return 0
        return 1
    if arguments.command == "worker":
        await _worker(harness_paths, arguments.session_id)
        return 0
    if arguments.command == "sync":
        store = StateStore(harness_paths.database)
        try:
            result = await asyncio.to_thread(
                publish_all,
                harness_paths,
                store,
            )
        finally:
            store.close()
        _print_json(
            {
                "state_root": str(harness_paths.state_dir),
                "sync": result,
            }
        )
        if result.get("state") == "synced":
            return 0
        return 1
    if arguments.command == "sync-status":
        _print_json(
            {
                "state_root": str(harness_paths.state_dir),
                "sync": read_sync_status(harness_paths),
            }
        )
        return 0
    if arguments.command == "doctor":
        return await _doctor(harness_paths)
    client = await ensure_daemon(harness_paths)
    if arguments.command == "new":
        predicates = [
            _json_object(value, "--predicate") for value in arguments.predicate
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
                "execution_profile": arguments.execution_profile,
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
    if arguments.command == "capabilities":
        _print_json(await client.request("GET", "/v1/capabilities"))
        return 0
    if arguments.command == "usage":
        result = await client.request(
            "GET",
            "/v1/sessions/" + arguments.session_id + "/usage",
        )
        _print_json(result)
        return 0
    if arguments.command == "extend-budget":
        payload = {
            "reason": arguments.reason,
            "allow_xhigh_once": arguments.allow_xhigh_once,
        }
        if arguments.seconds is not None:
            payload["additional_seconds"] = arguments.seconds
        if arguments.tokens is not None:
            payload["additional_tokens"] = arguments.tokens
        if arguments.command_id:
            payload["command_id"] = arguments.command_id
        if arguments.provider:
            payload["provider"] = arguments.provider
        result = await client.request(
            "POST",
            "/v1/sessions/" + arguments.session_id + "/budget-extensions",
            payload=payload,
            idempotency_key=new_uuid(),
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
            "/v1/sessions/" + arguments.session_id + "/checkpoints",
            payload={},
        )
        _print_json(result)
        return 0
    if arguments.command in {"archive", "unarchive"}:
        result = await client.request(
            "POST",
            "/v1/sessions/" + arguments.session_id + "/" + arguments.command,
            payload={},
            idempotency_key=new_uuid(),
        )
        _print_json(result)
        return 0
    if arguments.command == "reconcile":
        if arguments.reconcile_action == "list":
            result = await client.request(
                "GET",
                "/v1/sessions/" + arguments.session_id + "/reconciliations",
            )
            _print_json(result)
            return 0
        if arguments.reconcile_action == "inspect":
            result = await client.request(
                "GET",
                "/v1/reconciliations/" + arguments.reconciliation_id,
            )
            _print_json(result)
            return 0
        audit = _json_object(arguments.audit, "--audit")
        result = await client.request(
            "POST",
            "/v1/reconciliations/" + arguments.reconciliation_id + "/resolution",
            payload={
                "decision": arguments.decision,
                "observed_workspace_digest": (arguments.observed_workspace_digest),
                "approval_id": arguments.approval_id,
                "audit": audit,
            },
            idempotency_key=new_uuid(),
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
            "/v1/sessions/" + arguments.session_id + "/commands/" + arguments.action,
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
                "/v1/sessions/" + arguments.session_id + "/transfers",
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
            envelope = arguments.envelope_file.read_text(encoding="utf-8").strip()
            result = await client.request(
                "POST",
                "/v1/transfers/import",
                payload={
                    "envelope": envelope,
                    "source_signing_public": (arguments.source_signing_public),
                },
            )
            _print_json(result)
            return 0
        if arguments.transfer_action == "finalize":
            result = await client.request(
                "POST",
                "/v1/sessions/" + arguments.session_id + "/transfers/finalize",
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
        paths=harness_paths,
    )
    try:
        await worker.run()
    finally:
        store.close()


async def _doctor(harness_paths: Any) -> int:
    state_mode = stat.S_IMODE(harness_paths.state_dir.stat().st_mode)
    runtime_private = stat.S_IMODE(harness_paths.runtime.stat().st_mode) == 0o700
    socket_mode: int | None = None
    if harness_paths.socket.exists():
        socket_mode = stat.S_IMODE(harness_paths.socket.stat().st_mode)
    trusted_node_tools = True
    try:
        trusted_executable("node")
        trusted_executable("npx")
    except (OSError, RuntimeError):
        trusted_node_tools = False
    checks = {
        "npx": trusted_node_tools,
        "git": shutil.which("git") is not None,
        "state_directory": harness_paths.state_dir.is_dir(),
        "git_state_repository": (harness_paths.state_dir / ".git").exists(),
        "private_state_mode": state_mode == 0o700,
        "private_runtime_mode": runtime_private,
        "private_socket_mode": (socket_mode is None or socket_mode == 0o600),
    }
    details: dict[str, Any] = {}
    creation_intent_root = harness_paths.runtime / "creation-intents"
    pending_creation_intents = []
    if creation_intent_root.is_dir():
        pending_creation_intents = sorted(
            path.name for path in creation_intent_root.glob("*.json")
        )
    checks["pending_creation_intents"] = not pending_creation_intents
    details["pending_creation_intents"] = pending_creation_intents
    store = StateStore(harness_paths.database)
    try:
        checks["sqlite"] = store.integrity_check() == "ok"
        workers = store.worker_registrations()
        active_leases = store.active_process_leases()
        stale_workers = [
            worker
            for worker in workers
            if _timestamp_is_stale(
                str(worker.get("heartbeat_at", "")),
                maximum_age_seconds=90,
            )
        ]
        stale_leases = [
            lease
            for lease in active_leases
            if _timestamp_is_expired(str(lease.get("expires_at", "")))
        ]
        checks["stale_workers"] = not stale_workers
        checks["stale_process_leases"] = not stale_leases
        details["workers"] = workers
        details["active_process_leases"] = active_leases
        details["stale_workers"] = stale_workers
        details["stale_process_leases"] = stale_leases
        socket_mode_detail = "absent"
        if socket_mode is not None:
            socket_mode_detail = oct(socket_mode)
        details["permissions"] = {
            "state_mode": oct(state_mode),
            "runtime_mode": oct(stat.S_IMODE(harness_paths.runtime.stat().st_mode)),
            "socket_mode": socket_mode_detail,
        }
    finally:
        store.close()
    bundle_status: dict[str, Any] = {
        "status": "fail",
        "detail": "no installed bundle is selected",
    }
    try:
        selection = _installed_selection()
        bundle = verify_bundle(selection.executable.parent.parent)
        bundle_status = {
            "status": "pass",
            "build_id": selection.build_id,
            "content_digest": bundle.content_digest,
            "executable": str(selection.executable),
        }
    except (BundleError, InstallError, ValueError):
        bundle_status = {
            "status": "fail",
            "detail": "installed bundle is absent or invalid",
        }
    service_probes = [asdict(item) for item in _service_manager().diagnostics()]
    daemon = HarnessClient(harness_paths)
    daemon_health = await daemon._health_payload()
    daemon_status = "not-running"
    if daemon_health:
        daemon_status = "incompatible"
        if (
            daemon_health.get("control_build_id") == CONTROL_BUILD_ID
            and daemon_health.get("control_protocol_version")
            == CONTROL_PROTOCOL_VERSION
        ):
            daemon_status = "compatible"
    runtime_build_id = str(daemon_health.get("runtime_build_id", ""))
    selected_build_id = str(bundle_status.get("build_id", ""))
    runtime_binding_required = bool(daemon_health and selected_build_id)
    runtime_build_matches = True
    if runtime_binding_required:
        runtime_build_matches = runtime_build_id == selected_build_id
    checks["installed_bundle"] = bundle_status.get("status") == "pass"
    checks["daemon_compatible"] = daemon_status == "compatible"
    checks["runtime_build_matches_selection"] = runtime_build_matches
    checks["systemd_recovery"] = bool(service_probes) and all(
        probe.get("status") == "pass" for probe in service_probes
    )
    disk = shutil.disk_usage(harness_paths.state_dir)
    checks["disk_headroom"] = disk.free >= 2 * 1024**3
    checks["claude_auth"] = provider_auth_ready("claude")
    checks["codex_auth"] = provider_auth_ready("codex")
    details.update(
        {
            "bundle": bundle_status,
            "daemon": {
                "status": daemon_status,
                "control_build_id": daemon_health.get(
                    "control_build_id",
                    "",
                ),
                "control_protocol_version": daemon_health.get(
                    "control_protocol_version",
                    0,
                ),
                "runtime_build_id": runtime_build_id,
                "selected_build_id": selected_build_id,
                "runtime_binding_required": runtime_binding_required,
                "quiescence": daemon_health.get("quiescence", {}),
            },
            "disk": {
                "free_bytes": disk.free,
                "headroom": disk.free >= 2 * 1024**3,
            },
            "provider_launchers": {
                "claude": ("npx @anthropic-ai/claude-code@2.1.220"),
                "codex": "npx -y @openai/codex@0.146.0",
            },
            "service": service_probes,
        }
    )
    sync_status = read_sync_status(harness_paths)
    sync_state = str(sync_status.get("state", "unknown"))
    sync_pending = bool(sync_status.get("pending", False))
    checks["sync_clean"] = not sync_pending and sync_state not in {
        "conflict",
        "invalid",
    }
    details["sync_lag_seconds"] = _sync_lag_seconds(sync_status)
    _print_json(
        {
            "checks": checks,
            "details": details,
            "ok": all(checks.values()),
            "state_root": str(harness_paths.state_dir),
            "sync": sync_status,
        }
    )
    if all(checks.values()):
        return 0
    return 1


def _service_manager() -> SystemdUserService:
    return SystemdUserService()


async def _quiescence(harness_paths: Any) -> int:
    manager_status = _service_manager().status()
    health = await HarnessClient(harness_paths)._health_payload()
    remote = health.get("quiescence")
    if isinstance(remote, dict):
        restart_safe = remote.get("restart_safe") is True
        _print_json(
            {
                "runtime_build_id": str(health.get("runtime_build_id", "")),
                "installed_build_id": manager_status.build_id,
                "service_active": manager_status.active,
                "daemon_reachable": True,
                "proof_state_known": True,
                "quiescence": remote,
            }
        )
        if restart_safe:
            return 0
        return 1

    store = StateStore(harness_paths.database)
    try:
        commands = store.active_command_summaries()
    finally:
        store.close()
    proof_state_known = not manager_status.active
    restart_safe = not commands and proof_state_known
    _print_json(
        {
            "runtime_build_id": "",
            "installed_build_id": manager_status.build_id,
            "service_active": manager_status.active,
            "daemon_reachable": False,
            "proof_state_known": proof_state_known,
            "quiescence": {
                "restart_safe": restart_safe,
                "active_commands": len(commands),
                "active_command_details": commands,
                "active_unattended_commands": [
                    item for item in commands if item.get("profile") == "unattended"
                ],
                "active_proofs": 0,
                "active_proof_sessions": [],
            },
        }
    )
    if restart_safe:
        return 0
    return 1


def _installed_selection():
    selection = read_selection(default_launcher())
    if selection is None:
        raise ValueError("install a verified bundle before installing the service")
    bundle = verify_bundle(selection.executable.parent.parent)
    if bundle.build_id != selection.build_id:
        raise ValueError("installed bundle selection is inconsistent")
    return selection


def _sync_lag_seconds(value: dict[str, Any]) -> float | None:
    updated_at = str(value.get("updated_at", ""))
    if not updated_at:
        return None
    try:
        observed = datetime.datetime.fromisoformat(updated_at)
    except ValueError:
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=datetime.UTC)
    now = datetime.datetime.now(datetime.UTC)
    return max(0.0, (now - observed).total_seconds())


def _timestamp_is_stale(
    value: str,
    *,
    maximum_age_seconds: float,
) -> bool:
    observed = _timestamp(value)
    if observed is None:
        return True
    age = datetime.datetime.now(datetime.UTC) - observed
    return age.total_seconds() > maximum_age_seconds


def _timestamp_is_expired(value: str) -> bool:
    observed = _timestamp(value)
    if observed is None:
        return True
    return observed <= datetime.datetime.now(datetime.UTC)


def _timestamp(value: str) -> datetime.datetime | None:
    try:
        observed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=datetime.UTC)
    return observed.astimezone(datetime.UTC)


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
