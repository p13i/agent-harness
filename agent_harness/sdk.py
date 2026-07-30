"""Typed async client for the agent harness control plane."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

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

    async def list_sessions(
        self,
        *,
        include_archived: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        path = "/v1/sessions"
        if include_archived:
            path += "?archived=1"
        result = await self.raw.request("GET", path)
        return _object_tuple(result.get("sessions"))

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
    ) -> dict[str, Any]:
        if budgets is None:
            budgets = {}
        result = await self.raw.request(
            "POST",
            "/v1/sessions",
            payload={
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
            },
        )
        return _object(result.get("session"))

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
    ) -> CommandView:
        payload = {
            "text": text,
            "model": model,
            "effort": effort,
            "workload": workload,
        }
        if provider:
            payload["provider"] = provider
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
    ) -> bool:
        result = await self.raw.request(
            "POST",
            "/v1/sessions/"
            + session_id
            + "/approvals/"
            + approval_id,
            payload={"decision": decision},
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
        )
        return _object(result.get("evidence"))

    async def checkpoint(self, session_id: str) -> dict[str, Any]:
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/checkpoints",
            payload={},
        )
        return _object(result.get("checkpoint"))

    async def fork(
        self,
        session_id: str,
        *,
        name: str = "",
    ) -> dict[str, Any]:
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/fork",
            payload={"name": name},
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

    async def extend_budget(
        self,
        session_id: str,
        *,
        reason: str,
        additional_seconds: int | None = None,
        additional_tokens: int | None = None,
        allow_xhigh_once: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "reason": reason,
            "allow_xhigh_once": allow_xhigh_once,
        }
        if additional_seconds is not None:
            payload["additional_seconds"] = additional_seconds
        if additional_tokens is not None:
            payload["additional_tokens"] = additional_tokens
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/budget-extensions",
            payload=payload,
            idempotency_key=new_uuid(),
        )
        return _object(result.get("safety"))

    async def create_process_lease(
        self,
        provider: str,
        *,
        session_id: str = "",
        execution_profile: str = "unattended",
    ) -> dict[str, Any]:
        result = await self.raw.request(
            "POST",
            "/v1/leases",
            payload={
                "provider": provider,
                "session_id": session_id,
                "execution_profile": execution_profile,
            },
            idempotency_key=new_uuid(),
        )
        return _object(result.get("lease"))

    async def update_process_lease(
        self,
        lease_id: str,
        *,
        action: str,
        pid: int | None = None,
        pid_start: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": action}
        if pid is not None:
            payload["pid"] = pid
        if pid_start:
            payload["pid_start"] = pid_start
        result = await self.raw.request(
            "PATCH",
            "/v1/leases/" + lease_id,
            payload=payload,
            idempotency_key=new_uuid(),
        )
        return _object(result.get("lease"))

    async def process_leases(self) -> tuple[dict[str, Any], ...]:
        result = await self.raw.request("GET", "/v1/leases")
        return _object_tuple(result.get("leases"))

    async def sync_status(self) -> dict[str, Any]:
        return await self.raw.request("GET", "/v1/sync")

    async def sync(self) -> dict[str, Any]:
        return await self.raw.request(
            "POST",
            "/v1/sync",
            payload={},
        )

    async def export(self, session_id: str) -> Path:
        result = await self.raw.request(
            "POST",
            "/v1/sessions/" + session_id + "/export",
            payload={},
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
