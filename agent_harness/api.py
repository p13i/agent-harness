"""Versioned HTTP, SSE, and WebSocket control plane."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import signal
import socket
from typing import Any

from aiohttp import web

from agent_harness.config import HarnessPaths
from agent_harness.config import CONTROL_BUILD_ID
from agent_harness.config import CONTROL_PROTOCOL_VERSION
from agent_harness.config import api_token
from agent_harness.errors import HarnessError
from agent_harness.ids import new_uuid
from agent_harness.service import HarnessService
from agent_harness.terminal import terminal_socket


def create_app(
    service: HarnessService,
    token: str,
) -> web.Application:
    @web.middleware
    async def correlate(
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        correlation_id = request.headers.get(
            "X-Correlation-ID",
            "",
        ).strip()
        if not correlation_id:
            correlation_id = new_uuid()
        request["correlation_id"] = correlation_id[:128]
        response = await handler(request)
        if not response.prepared:
            response.headers["X-Correlation-ID"] = request[
                "correlation_id"
            ]
        return response

    @web.middleware
    async def errors(
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        try:
            return await handler(request)
        except HarnessError as error:
            return web.json_response(
                {
                    "error": {
                        "code": error.detail.code,
                        "message": error.detail.message,
                        "retryable": error.detail.retryable,
                        "correlation_id": _correlation_id(request),
                    }
                },
                status=error.detail.status,
            )
        except ValueError as error:
            return web.json_response(
                {
                    "error": {
                        "code": "E_INPUT",
                        "message": str(error),
                        "retryable": False,
                        "correlation_id": _correlation_id(request),
                    }
                },
                status=400,
            )
        except Exception:
            return web.json_response(
                {
                    "error": {
                        "code": "E_INTERNAL",
                        "message": "internal harness failure",
                        "retryable": True,
                        "correlation_id": _correlation_id(request),
                    }
                },
                status=500,
            )

    @web.middleware
    async def authenticate(
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        authorization = request.headers.get("Authorization", "")
        expected = "Bearer " + token
        if not hmac.compare_digest(authorization, expected):
            return web.json_response(
                {
                    "error": {
                        "code": "E_AUTH",
                        "message": "valid bearer authentication is required",
                        "retryable": False,
                        "correlation_id": _correlation_id(request),
                    }
                },
                status=401,
            )
        return await handler(request)

    app = web.Application(
        middlewares=[correlate, errors, authenticate],
        client_max_size=8 * 1024 * 1024,
    )
    app["service"] = service
    app.router.add_get("/healthz", _health)
    app.router.add_get("/readyz", _ready)
    app.router.add_get("/v1/sessions", _sessions)
    app.router.add_post("/v1/sessions", _create_session)
    app.router.add_get("/v1/sessions/{session_id}", _session)
    app.router.add_patch("/v1/sessions/{session_id}", _configure_session)
    app.router.add_get(
        "/v1/sessions/{session_id}/ui-state",
        _ui_state,
    )
    app.router.add_put(
        "/v1/sessions/{session_id}/ui-state",
        _set_ui_state,
    )
    app.router.add_get("/v1/sessions/{session_id}/events", _events)
    app.router.add_get("/v1/sessions/{session_id}/stream", _stream)
    app.router.add_post("/v1/sessions/{session_id}/messages", _message)
    app.router.add_post(
        "/v1/sessions/{session_id}/commands/{command_type}",
        _command,
    )
    app.router.add_get("/v1/commands/{command_id}", _command_status)
    app.router.add_get(
        "/v1/sessions/{session_id}/approvals",
        _approvals,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/approvals/{approval_id}",
        _resolve_approval,
    )
    app.router.add_get("/v1/sessions/{session_id}/goal", _goal)
    app.router.add_get(
        "/v1/sessions/{session_id}/usage",
        _session_usage,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/budget-extensions",
        _budget_extension,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/evidence",
        _evidence,
    )
    app.router.add_post("/v1/sessions/{session_id}/export", _export)
    app.router.add_post(
        "/v1/sessions/{session_id}/checkpoints",
        _checkpoint,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/fork",
        _fork,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/route",
        _route_preview,
    )
    app.router.add_get("/v1/providers", _providers)
    app.router.add_get("/v1/leases", _leases)
    app.router.add_post("/v1/leases", _create_lease)
    app.router.add_patch("/v1/leases/{lease_id}", _update_lease)
    app.router.add_get("/v1/registry", _registry)
    app.router.add_get("/v1/fleet/keys", _fleet_keys)
    app.router.add_post("/v1/transfers/import", _import_transfer)
    app.router.add_post(
        "/v1/sessions/{session_id}/transfers",
        _create_transfer,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/transfers/finalize",
        _finalize_transfer,
    )
    app.router.add_get(
        "/v1/sessions/{session_id}/terminal",
        _terminal,
    )
    return app


async def run_daemon(
    paths: HarnessPaths,
    *,
    tcp_host: str = "",
    tcp_port: int = 0,
) -> None:
    service = HarnessService(paths)
    token = api_token(paths)
    app = create_app(service, token)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await _prepare_socket(paths.socket)
    unix_site = web.UnixSite(runner, str(paths.socket))
    await unix_site.start()
    _write_daemon_pid(paths.daemon_pid)
    try:
        tcp_site: web.TCPSite | None = None
        if tcp_host:
            _validate_tcp_host(tcp_host)
            if tcp_port <= 0 or tcp_port > 65535:
                raise ValueError("TCP port must be between 1 and 65535")
            tcp_site = web.TCPSite(runner, tcp_host, tcp_port)
            await tcp_site.start()
        service.recover_workers()
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signal_value in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signal_value, stopped.set)
            except NotImplementedError:
                continue
        await stopped.wait()
    finally:
        service.workers.stop_all()
        await runner.cleanup()
        service.close()
        if paths.socket.exists():
            paths.socket.unlink()
        _remove_daemon_pid(paths.daemon_pid)


async def _health(request: web.Request) -> web.Response:
    del request
    return web.json_response(
        {
            "status": "ok",
            "control_build_id": CONTROL_BUILD_ID,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
        }
    )


async def _ready(request: web.Request) -> web.Response:
    service = _service(request)
    service.store.list_sessions()
    return web.json_response({"status": "ready"})


async def _sessions(request: web.Request) -> web.Response:
    service = _service(request)
    include_archived = request.query.get("archived") == "1"
    sessions = service.store.list_sessions(include_archived)
    return web.json_response(
        {"sessions": [item.as_dict() for item in sessions]}
    )


async def _create_session(request: web.Request) -> web.Response:
    service = _service(request)
    session = service.create_session(await _body(request))
    return web.json_response({"session": session.as_dict()}, status=201)


async def _session(request: web.Request) -> web.Response:
    service = _service(request)
    session_id = request.match_info["session_id"]
    session = service.store.get_session(session_id)
    goal = service.store.goal_for_session(session_id)
    goal_value: dict[str, Any] | None = None
    if goal is not None:
        goal_value = goal.as_dict()
    return web.json_response(
        {
            "session": session.as_dict(),
            "goal": goal_value,
            "approvals": service.store.pending_approvals(session_id),
            "last_sequence": service.store.last_sequence(session_id),
            "safety": service.safety_state(session_id),
        }
    )


async def _configure_session(request: web.Request) -> web.Response:
    session = _service(request).configure_session(
        request.match_info["session_id"],
        await _body(request),
    )
    return web.json_response({"session": session.as_dict()})


async def _ui_state(request: web.Request) -> web.Response:
    state = _service(request).ui_state(
        request.match_info["session_id"]
    )
    return web.json_response({"ui_state": state})


async def _set_ui_state(request: web.Request) -> web.Response:
    state = _service(request).set_ui_state(
        request.match_info["session_id"],
        await _body(request),
    )
    return web.json_response({"ui_state": state})


async def _events(request: web.Request) -> web.Response:
    service = _service(request)
    after = _integer(request.query.get("after", "0"))
    limit = _integer(request.query.get("limit", "1000"))
    events = service.store.events(
        request.match_info["session_id"],
        after=after,
        limit=limit,
    )
    return web.json_response(
        {"events": [item.as_dict() for item in events]}
    )


async def _stream(request: web.Request) -> web.StreamResponse:
    service = _service(request)
    session_id = request.match_info["session_id"]
    after_text = request.headers.get("Last-Event-ID", "")
    if not after_text:
        after_text = request.query.get("after", "0")
    after = _integer(after_text)
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Correlation-ID": _correlation_id(request),
        },
    )
    await response.prepare(request)
    for unused in range(3600):
        del unused
        events = service.store.events(session_id, after=after, limit=500)
        for event in events:
            payload = json.dumps(event.as_dict(), separators=(",", ":"))
            content = (
                "id: "
                + str(event.sequence)
                + "\nevent: "
                + event.event_type
                + "\ndata: "
                + payload
                + "\n\n"
            )
            await response.write(content.encode("utf-8"))
            after = event.sequence
        await response.write(b": heartbeat\n\n")
        await asyncio.sleep(1.0)
    return response


async def _message(request: web.Request) -> web.Response:
    service = _service(request)
    payload = await _body(request)
    key = _idempotency_key(request)
    receipt = service.submit_message(
        request.match_info["session_id"],
        payload,
        key,
    )
    return web.json_response({"command": receipt}, status=202)


async def _command(request: web.Request) -> web.Response:
    service = _service(request)
    receipt = service.command(
        request.match_info["session_id"],
        request.match_info["command_type"],
        await _body(request),
        _idempotency_key(request),
    )
    return web.json_response({"command": receipt}, status=202)


async def _command_status(request: web.Request) -> web.Response:
    service = _service(request)
    command = service.store.get_command(request.match_info["command_id"])
    return web.json_response({"command": command.as_dict()})


async def _approvals(request: web.Request) -> web.Response:
    service = _service(request)
    approvals = service.store.pending_approvals(
        request.match_info["session_id"]
    )
    return web.json_response({"approvals": approvals})


async def _resolve_approval(request: web.Request) -> web.Response:
    service = _service(request)
    result = service.resolve_approval(
        request.match_info["session_id"],
        request.match_info["approval_id"],
        await _body(request),
    )
    return web.json_response(result)


async def _goal(request: web.Request) -> web.Response:
    service = _service(request)
    goal = service.store.goal_for_session(request.match_info["session_id"])
    if goal is None:
        return web.json_response({"goal": None, "evidence": []})
    evidence = service.store.evidence(goal.goal_id)
    return web.json_response(
        {
            "goal": goal.as_dict(),
            "evidence": [item.as_dict() for item in evidence],
        }
    )


async def _session_usage(request: web.Request) -> web.Response:
    safety = _service(request).safety_state(
        request.match_info["session_id"]
    )
    return web.json_response({"safety": safety})


async def _budget_extension(request: web.Request) -> web.Response:
    payload = await _body(request)
    return _idempotent_response(
        request,
        "budget-extension:"
        + request.match_info["session_id"],
        payload,
        201,
        lambda: {
            "safety": _service(request).extend_budget(
                request.match_info["session_id"],
                payload,
            )
        },
    )


async def _evidence(request: web.Request) -> web.Response:
    service = _service(request)
    evidence = service.add_evidence(
        request.match_info["session_id"],
        await _body(request),
    )
    return web.json_response({"evidence": evidence}, status=201)


async def _export(request: web.Request) -> web.Response:
    service = _service(request)
    path = service.export(request.match_info["session_id"])
    return web.json_response({"path": str(path)})


async def _checkpoint(request: web.Request) -> web.Response:
    checkpoint = _service(request).checkpoint(
        request.match_info["session_id"]
    )
    return web.json_response(
        {"checkpoint": checkpoint},
        status=201,
    )


async def _fork(request: web.Request) -> web.Response:
    session = _service(request).fork_session(
        request.match_info["session_id"],
        await _body(request),
    )
    return web.json_response(
        {"session": session.as_dict()},
        status=201,
    )


async def _route_preview(request: web.Request) -> web.Response:
    decision = await _service(request).preview_route(
        request.match_info["session_id"],
        await _body(request),
    )
    return web.json_response({"route": decision})


async def _providers(request: web.Request) -> web.Response:
    service = _service(request)
    workspace_text = request.query.get("workspace", ".")
    result = await service.scheduler.status(Path(workspace_text).resolve())
    return web.json_response({"providers": result})


async def _leases(request: web.Request) -> web.Response:
    return web.json_response(
        {"leases": _service(request).process_leases()}
    )


async def _create_lease(request: web.Request) -> web.Response:
    payload = await _body(request)
    return _idempotent_response(
        request,
        "lease-create",
        payload,
        201,
        lambda: {
            "lease": _service(request).create_process_lease(
                payload
            )
        },
    )


async def _update_lease(request: web.Request) -> web.Response:
    payload = await _body(request)
    return _idempotent_response(
        request,
        "lease-update:" + request.match_info["lease_id"],
        payload,
        200,
        lambda: {
            "lease": _service(request).update_process_lease(
                request.match_info["lease_id"],
                payload,
            )
        },
    )


async def _registry(request: web.Request) -> web.Response:
    service = _service(request)
    return web.json_response(
        {"entries": service.store.registry_entries()}
    )


async def _fleet_keys(request: web.Request) -> web.Response:
    return web.json_response({"keys": _service(request).public_keys()})


async def _create_transfer(request: web.Request) -> web.Response:
    result = _service(request).create_transfer(
        request.match_info["session_id"],
        await _body(request),
    )
    return web.json_response({"transfer": result}, status=201)


async def _import_transfer(request: web.Request) -> web.Response:
    result = _service(request).import_transfer(await _body(request))
    return web.json_response({"transfer": result}, status=201)


async def _finalize_transfer(request: web.Request) -> web.Response:
    payload = await _body(request)
    result = _service(request).finalize_transfer(
        request.match_info["session_id"],
        str(payload.get("destination_host", "")),
        int(payload.get("owner_epoch", 0)),
    )
    return web.json_response({"session": result})


async def _terminal(request: web.Request) -> web.WebSocketResponse:
    service = _service(request)
    session = service.store.get_session(request.match_info["session_id"])
    return await terminal_socket(request, Path(session.worktree))


async def _body(request: web.Request) -> dict[str, Any]:
    if not request.can_read_body:
        return {}
    value = await request.json()
    if not isinstance(value, dict):
        raise ValueError("JSON object is required")
    return value


def _service(request: web.Request) -> HarnessService:
    service = request.app["service"]
    if not isinstance(service, HarnessService):
        raise RuntimeError("service is unavailable")
    return service


def _idempotency_key(request: web.Request) -> str:
    value = request.headers.get("Idempotency-Key", "")
    if not value:
        raise ValueError("Idempotency-Key header is required")
    if len(value) > 128:
        raise ValueError("Idempotency-Key header is too long")
    return value


def _idempotent_response(
    request: web.Request,
    operation: str,
    payload: dict[str, Any],
    status: int,
    mutate: Any,
) -> web.Response:
    key = _idempotency_key(request)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    store = _service(request).store
    receipt = store.mutation_receipt(key, operation, digest)
    if receipt is not None:
        return web.json_response(
            receipt["response"],
            status=receipt["status_code"],
        )
    response = mutate()
    receipt = store.record_mutation_receipt(
        key,
        operation,
        digest,
        response,
        status,
    )
    return web.json_response(
        receipt["response"],
        status=receipt["status_code"],
    )


def _integer(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError("integer query value required") from error


def _correlation_id(request: web.Request) -> str:
    value = request.get("correlation_id", "")
    if isinstance(value, str):
        return value
    return ""


async def _prepare_socket(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        return
    try:
        reader, writer = await asyncio.open_unix_connection(path)
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        path.unlink()
        return
    writer.close()
    await writer.wait_closed()
    del reader
    raise RuntimeError("daemon is already running")


def _write_daemon_pid(path: Path) -> None:
    path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _remove_daemon_pid(path: Path) -> None:
    if not path.exists():
        return
    try:
        recorded = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    if recorded != os.getpid():
        return
    path.unlink()


def _validate_tcp_host(host: str) -> None:
    disallowed = {"0.0.0.0", "::", ""}
    if host in disallowed:
        raise ValueError("TCP listener must use an explicit host address")
    try:
        socket.getaddrinfo(host, None)
    except socket.gaierror as error:
        raise ValueError("TCP host cannot be resolved") from error
