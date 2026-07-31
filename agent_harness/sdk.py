"""Typed async client for the agent harness control plane."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from aiohttp import ClientSession
from aiohttp import UnixConnector

from agent_harness.client import HarnessClient
from agent_harness.config import HarnessPaths
from agent_harness.ids import new_uuid


@dataclass(frozen=True)
class SessionView:
    session: dict[str, Any]
    goal: dict[str, Any] | None
    approvals: tuple[dict[str, Any], ...]
    safety: dict[str, Any]
    last_sequence: int


@dataclass(frozen=True)
class EventPage:
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CommandView:
    command_id: str
    status: str
    value: dict[str, Any]


@dataclass(frozen=True)
class RouteView:
    provider: str
    model: str
    effort: str
    reason: str
    value: dict[str, Any]


class AgentHarnessClient:
    """Typed high-level API over the private local harness transport."""

    def __init__(self, paths: HarnessPaths) -> None:
        self.paths = paths
        self.raw = HarnessClient(paths)

    async def capabilities(self) -> dict[str, Any]:
        return await self.raw.request("GET", "/v1/capabilities")

    async def list_sessions(
        self,
        *,
        include_archived: bool = False,
        external_orchestrator: str = "",
        external_job_id: str = "",
    ) -> tuple[dict[str, Any], ...]:
        path = "/v1/sessions"
        query: dict[str, str] = {}
        if include_archived:
            query["archived"] = "1"
        if external_orchestrator or external_job_id:
            if not external_orchestrator or not external_job_id:
                raise ValueError(
                    "external orchestrator and job ID are both required"
                )
            query["external_orchestrator"] = external_orchestrator
            query["external_job_id"] = external_job_id
        if query:
            path += "?" + urlencode(query)
        result = await self.raw.request("GET", path)
        return _object_tuple(result.get("sessions"))

    async def session_by_external_ref(
        self,
        orchestrator: str,
        job_id: str,
    ) -> dict[str, Any] | None:
        sessions = await self.list_sessions(
            include_archived=True,
            external_orchestrator=orchestrator,
            external_job_id=job_id,
        )
        if not sessions:
            return None
        if len(sessions) != 1:
            raise RuntimeError("external reference is not unique")
        return sessions[0]

    async def create_session(
        self,
        workspace: Path,
        *,
        name: str = "",
        goal: str = "",
        goal_kind: str = "finite",
        constraints: tuple[str, ...] = (),
        predicates: tuple[dict[str, Any], ...] = (),
        budgets: dict[str, Any] | None = None,
        permission_mode: str = "approval",
        direct: bool = False,
        execution_profile: str = "unattended",
        external_ref: dict[str, str] | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        if budgets is None:
            budgets = {}
        payload: dict[str, Any] = {
            "workspace": str(workspace.expanduser().resolve()),
            "name": name,
            "goal": goal,
            "goal_kind": goal_kind,
            "constraints": list(constraints),
            "predicates": list(predicates),
            "budgets": budgets,
            "permission_mode": permission_mode,
            "direct": direct,
            "execution_profile": execution_profile,
        }
        if external_ref is not None:
            payload["external_ref"] = external_ref
        result = await self.raw.request(
            "POST",
            "/v1/sessions",
            payload=payload,
            idempotency_key=idempotency_key,
        )
        return _object(result.get("session"))

    async def ensure_session(
        self,
        workspace: Path,
        *,
        orchestrator: str,
        job_id: str,
        idempotency_key: str,
        name: str = "",
        goal: str = "",
        permission_mode: str = "approval",
        direct: bool = False,
        execution_profile: str = "unattended",
    ) -> dict[str, Any]:
        _require_managed_key(idempotency_key)
        return await self.create_session(
            workspace,
            name=name,
            goal=goal,
            permission_mode=permission_mode,
            direct=direct,
            execution_profile=execution_profile,
            external_ref={
                "orchestrator": orchestrator,
                "job_id": job_id,
            },
            idempotency_key=idempotency_key,
        )

    async def session(self, session_id: str) -> SessionView:
        result = await self.raw.request(
            "GET",
            "/v1/sessions/" + session_id,
        )
        goal_value = result.get("goal")
        goal: dict[str, Any] | None = None
        if isinstance(goal_value, dict):
            goal = goal_value
        return SessionView(
            session=_object(result.get("session")),
            goal=goal,
            approvals=_object_tuple(result.get("approvals")),
            safety=_object(result.get("safety")),
            last_sequence=int(result.get("last_sequence", 0)),
        )

    async def configure_session(
        self,
        session_id: str,
        *,
        name: str | None = None,
        permission_mode: str | None = None,
        execution_profile: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if permission_mode is not None:
            payload["permission_mode"] = permission_mode
        if execution_profile is not None:
            payload["execution_profile"] = execution_profile
        result = await self.raw.request(
            "PATCH",
            "/v1/sessions/" + session_id,
            payload=payload,
        )
        return _object(result.get("session"))

    async def set_archived(
        self,
        session_id: str,
        archived: bool,
        *,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        action = "archive"
        if not archived:
            action = "unarchive"
        key = idempotency_key
        if not key:
            key = new_uuid()
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/" + action,
            payload={},
            idempotency_key=key,
        )
        return _object(result.get("session"))

    async def ui_state(self, session_id: str) -> dict[str, Any]:
        result = await self.raw.request(
            "GET",
            "/v1/sessions/" + session_id + "/ui-state",
        )
        return _object(result.get("ui_state"))

    async def update_ui_state(
        self,
        session_id: str,
        state: dict[str, str],
    ) -> dict[str, Any]:
        result = await self.raw.request(
            "PUT",
            "/v1/sessions/" + session_id + "/ui-state",
            payload=state,
        )
        return _object(result.get("ui_state"))

    async def events(
        self,
        session_id: str,
        *,
        after: int = 0,
        limit: int = 1000,
    ) -> EventPage:
        result = await self.raw.request(
            "GET",
            "/v1/sessions/"
            + session_id
            + "/events?after="
            + str(after)
            + "&limit="
            + str(limit),
        )
        return EventPage(_object_tuple(result.get("events")))

    async def turns(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        query = urlencode(
            {
                "after_sequence": str(after_sequence),
                "limit": str(limit),
            }
        )
        return await self.raw.request(
            "GET",
            "/v1/sessions/" + session_id + "/turns?" + query,
        )

    async def turn(
        self,
        session_id: str,
        turn_id: str,
    ) -> dict[str, Any]:
        return await self.raw.request(
            "GET",
            "/v1/sessions/" + session_id + "/turns/" + turn_id,
        )

    async def checkpoint_diff(
        self,
        session_id: str,
        checkpoint_id: str,
        *,
        start_line: int = 0,
        limit: int = 400,
    ) -> dict[str, Any]:
        query = urlencode(
            {
                "start_line": str(start_line),
                "limit": str(limit),
            }
        )
        result = await self.raw.request(
            "GET",
            "/v1/sessions/"
            + session_id
            + "/checkpoints/"
            + checkpoint_id
            + "/diff?"
            + query,
        )
        return _object(result.get("diff"))

    async def stream_events(
        self,
        session_id: str,
        *,
        after: int = 0,
    ) -> AsyncIterator[dict[str, Any]]:
        connector = UnixConnector(path=str(self.paths.socket))
        headers = {
            "Authorization": "Bearer " + self.raw.token,
            "Last-Event-ID": str(after),
        }
        async with ClientSession(
            connector=connector,
            base_url="http://localhost",
            headers=headers,
        ) as session:
            async with session.get(
                "/v1/sessions/" + session_id + "/stream"
            ) as response:
                response.raise_for_status()
                data_lines: list[str] = []
                async for content in response.content:
                    line = content.decode("utf-8").rstrip("\r\n")
                    if not line:
                        if data_lines:
                            yield _event_data(data_lines)
                            data_lines = []
                        continue
                    if line.startswith("data: "):
                        data_lines.append(line[6:])

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        provider: str = "",
        model: str = "",
        effort: str = "",
        workload: str = "implementation",
        idempotency_key: str = "",
        turn_ref: dict[str, str] | None = None,
    ) -> CommandView:
        payload = {
            "text": text,
            "model": model,
            "effort": effort,
            "workload": workload,
        }
        if provider:
            payload["provider"] = provider
        if turn_ref is not None:
            payload["turn_ref"] = turn_ref
        key = idempotency_key
        if not key:
            key = new_uuid()
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/messages",
            payload=payload,
            idempotency_key=key,
        )
        return _command_view(result.get("command"))

    async def submit_managed_turn(
        self,
        session_id: str,
        text: str,
        *,
        step_id: str,
        agent_role: str,
        idempotency_key: str,
        provider: str = "",
        model: str = "",
        effort: str = "",
        workload: str = "implementation",
    ) -> CommandView:
        _require_managed_key(idempotency_key)
        return await self.send_message(
            session_id,
            text,
            provider=provider,
            model=model,
            effort=effort,
            workload=workload,
            idempotency_key=idempotency_key,
            turn_ref={
                "step_id": step_id,
                "agent_role": agent_role,
            },
        )

    async def command(
        self,
        session_id: str,
        command_type: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> CommandView:
        if payload is None:
            payload = {}
        key = idempotency_key
        if not key:
            key = new_uuid()
        result = await self.raw.request(
            "POST",
            "/v1/sessions/"
            + session_id
            + "/commands/"
            + command_type,
            payload=payload,
            idempotency_key=key,
        )
        return _command_view(result.get("command"))

    async def command_status(self, command_id: str) -> CommandView:
        result = await self.raw.request(
            "GET",
            "/v1/commands/" + command_id,
        )
        return _command_view(result.get("command"))

    async def wait_command(
        self,
        command_id: str,
        *,
        timeout: float,
        poll_interval: float = 0.2,
    ) -> CommandView:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll interval must be positive")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            command = await self.command_status(command_id)
            if command.status in {"complete", "failed", "cancelled"}:
                return command
            if loop.time() >= deadline:
                return command
            await asyncio.sleep(poll_interval)

    async def approvals(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any], ...]:
        result = await self.raw.request(
            "GET",
            "/v1/sessions/" + session_id + "/approvals",
        )
        return _object_tuple(result.get("approvals"))

    async def resolve_approval(
        self,
        session_id: str,
        approval_id: str,
        decision: str,
        *,
        idempotency_key: str = "",
    ) -> bool:
        key = idempotency_key
        if not key:
            key = new_uuid()
        result = await self.raw.request(
            "POST",
            "/v1/sessions/"
            + session_id
            + "/approvals/"
            + approval_id,
            payload={"decision": decision},
            idempotency_key=key,
        )
        return bool(result.get("resolved", False))

    async def goal(self, session_id: str) -> dict[str, Any]:
        return await self.raw.request(
            "GET",
            "/v1/sessions/" + session_id + "/goal",
        )

    async def add_evidence(
        self,
        session_id: str,
        *,
        evidence_type: str,
        subject: str,
        outcome: str,
        value: dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        if value is None:
            value = {}
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/evidence",
            payload={
                "type": evidence_type,
                "subject": subject,
                "outcome": outcome,
                "value": value,
            },
            idempotency_key=idempotency_key,
        )
        return _object(result.get("evidence"))

    async def checkpoint(
        self,
        session_id: str,
        *,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/checkpoints",
            payload={},
            idempotency_key=idempotency_key,
        )
        return _object(result.get("checkpoint"))

    async def fork(
        self,
        session_id: str,
        *,
        name: str = "",
        external_ref: dict[str, str] | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name}
        if external_ref is not None:
            payload["external_ref"] = external_ref
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/fork",
            payload=payload,
            idempotency_key=idempotency_key,
        )
        return _object(result.get("session"))

    async def preview_route(
        self,
        session_id: str,
        *,
        provider: str = "",
        model: str = "",
        effort: str = "",
        workload: str = "implementation",
        required_capabilities: tuple[str, ...] = (),
        metered_budget: float | None = None,
    ) -> RouteView:
        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "effort": effort,
            "workload": workload,
            "required_capabilities": list(required_capabilities),
        }
        if metered_budget is not None:
            payload["metered_budget"] = metered_budget
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/route",
            payload=payload,
        )
        value = _object(result.get("route"))
        return RouteView(
            provider=str(value.get("provider", "")),
            model=str(value.get("model", "")),
            effort=str(value.get("effort", "")),
            reason=str(value.get("reason", "")),
            value=value,
        )

    async def providers(self, workspace: Path) -> dict[str, Any]:
        return await self.raw.request(
            "GET",
            "/v1/providers?workspace="
            + str(workspace.expanduser().resolve()),
        )

    async def usage(self, session_id: str) -> dict[str, Any]:
        result = await self.raw.request(
            "GET",
            "/v1/sessions/" + session_id + "/usage",
        )
        return _object(result.get("safety"))

    async def reconciliations(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any], ...]:
        result = await self.raw.request(
            "GET",
            "/v1/sessions/" + session_id + "/reconciliations",
        )
        return _object_tuple(result.get("reconciliations"))

    async def reconciliation(
        self,
        reconciliation_id: str,
    ) -> dict[str, Any]:
        result = await self.raw.request(
            "GET",
            "/v1/reconciliations/" + reconciliation_id,
        )
        return _object(result.get("reconciliation"))

    async def resolve_reconciliation(
        self,
        reconciliation_id: str,
        *,
        decision: str,
        observed_workspace_digest: str,
        idempotency_key: str,
        approval_id: str = "",
        audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _require_managed_key(idempotency_key)
        payload: dict[str, Any] = {
            "decision": decision,
            "observed_workspace_digest": observed_workspace_digest,
        }
        if approval_id:
            payload["approval_id"] = approval_id
        if audit is not None:
            payload["audit"] = audit
        result = await self.raw.request(
            "POST",
            "/v1/reconciliations/"
            + reconciliation_id
            + "/resolution",
            payload=payload,
            idempotency_key=idempotency_key,
        )
        return _object(result.get("reconciliation"))

    async def extend_budget(
        self,
        session_id: str,
        *,
        reason: str,
        additional_seconds: int | None = None,
        additional_tokens: int | None = None,
        allow_xhigh_once: bool = False,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "reason": reason,
            "allow_xhigh_once": allow_xhigh_once,
        }
        if additional_seconds is not None:
            payload["additional_seconds"] = additional_seconds
        if additional_tokens is not None:
            payload["additional_tokens"] = additional_tokens
        key = idempotency_key
        if not key:
            key = new_uuid()
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/budget-extensions",
            payload=payload,
            idempotency_key=key,
        )
        return _object(result.get("safety"))

    async def create_process_lease(
        self,
        provider: str,
        *,
        session_id: str = "",
        execution_profile: str = "unattended",
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        key = idempotency_key
        if not key:
            key = new_uuid()
        result = await self.raw.request(
            "POST",
            "/v1/leases",
            payload={
                "provider": provider,
                "session_id": session_id,
                "execution_profile": execution_profile,
            },
            idempotency_key=key,
        )
        return _object(result.get("lease"))

    async def update_process_lease(
        self,
        lease_id: str,
        *,
        action: str,
        pid: int | None = None,
        pid_start: str = "",
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": action}
        if pid is not None:
            payload["pid"] = pid
        if pid_start:
            payload["pid_start"] = pid_start
        key = idempotency_key
        if not key:
            key = new_uuid()
        result = await self.raw.request(
            "PATCH",
            "/v1/leases/" + lease_id,
            payload=payload,
            idempotency_key=key,
        )
        return _object(result.get("lease"))

    async def process_leases(self) -> tuple[dict[str, Any], ...]:
        result = await self.raw.request("GET", "/v1/leases")
        return _object_tuple(result.get("leases"))

    async def sync_status(self) -> dict[str, Any]:
        return await self.raw.request("GET", "/v1/sync")

    async def sync(
        self,
        *,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        return await self.raw.request(
            "POST",
            "/v1/sync",
            payload={},
            idempotency_key=idempotency_key,
        )

    async def export(
        self,
        session_id: str,
        *,
        idempotency_key: str = "",
    ) -> Path:
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/export",
            payload={},
            idempotency_key=idempotency_key,
        )
        return Path(str(result.get("path", "")))


def _command_view(value: object) -> CommandView:
    command = _object(value)
    return CommandView(
        command_id=str(command.get("command_id", "")),
        status=str(command.get("status", "")),
        value=command,
    )


def _object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _object_tuple(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _event_data(lines: list[str]) -> dict[str, Any]:
    value = json.loads("\n".join(lines))
    if not isinstance(value, dict):
        raise ValueError("SSE event data must be an object")
    return value


def _require_managed_key(value: str) -> None:
    if not value:
        raise ValueError(
            "managed operations require a caller idempotency key"
        )
