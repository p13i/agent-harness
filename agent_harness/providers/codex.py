"""Codex app-server adapter."""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_harness.errors import (
    HarnessError,
    ProviderExhaustedError,
    ProviderUnavailableError,
)
from agent_harness.process_control import (
    ProcessGroupIdentity,
    process_group_identity,
    terminate_process_group,
)
from agent_harness.providers.base import (
    ApprovalHandler,
    ChildLaunchGate,
    EventHandler,
    PrePromptGate,
    ProviderAdapter,
    ProviderEvent,
    ProviderModel,
    ProviderResult,
    ProviderStatus,
    provider_environment,
    trusted_executable,
)
from agent_harness.providers.normalize import codex_notification

NotificationHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
RequestHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class CodexAppServer:
    def __init__(
        self,
        workspace: Path,
        *,
        notification_handler: NotificationHandler,
        request_handler: RequestHandler,
        child_launch_gate: ChildLaunchGate | None = None,
    ) -> None:
        self.workspace = workspace
        self.notification_handler = notification_handler
        self.request_handler = request_handler
        self.child_launch_gate = child_launch_gate
        self.process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 0
        self.stderr_tail: list[str] = []
        self._process_group: ProcessGroupIdentity | None = None
        self._closing = False

    async def start(self) -> None:
        if self.process is not None:
            return
        self._closing = False
        command = [
            trusted_executable("npx"),
            "-y",
            "@openai/codex@0.146.0",
        ]
        if self.child_launch_gate is not None:
            command.extend(self._child_gate_arguments())
        else:
            command.extend(["-c", "agents.enabled=false"])
        command.extend(["app-server", "--listen", "stdio://"])
        self.process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.workspace,
            env=provider_environment("codex"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=16 * 1024 * 1024,
        )
        self._process_group = process_group_identity(self.process.pid)
        self._reader = asyncio.create_task(self._read_messages())
        self._stderr = asyncio.create_task(self._read_stderr())
        await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "p13i_agent_harness",
                    "title": "p13i agent harness",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": False},
            },
        )
        await self.notify("initialized", {})

    def _child_gate_arguments(self) -> list[str]:
        gate = self.child_launch_gate
        if gate is None:
            return ["-c", "agents.enabled=false"]
        limit = max(0, int(gate.limit))
        script_path = Path(__file__).resolve().parents[1] / "child_gate.py"
        hook_command = " ".join(
            (
                shlex.quote(str(Path(sys.executable).resolve())),
                "-B",
                shlex.quote(str(script_path)),
                shlex.quote(str(gate.database)),
                shlex.quote(gate.command_id),
                str(limit),
                "codex",
            )
        )
        hook = (
            "[{matcher='^(Agent|spawn_agent)$',hooks=[{type='command',command="
            + json.dumps(hook_command)
            + ",timeout=5}]}]"
        )
        arguments = [
            "--dangerously-bypass-hook-trust",
            "-c",
            "hooks.PreToolUse=" + hook,
        ]
        if limit == 0:
            arguments.extend(["-c", "agents.enabled=false"])
        else:
            arguments.extend(
                [
                    "-c",
                    "agents.max_concurrent_threads_per_session=" + str(limit),
                ]
            )
        return arguments

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        self._closing = True
        if process.stdin is not None:
            process.stdin.close()
        identity = self._process_group
        if identity is None:
            raise RuntimeError("Codex process group identity is missing")
        await terminate_process_group(process, identity, grace_timeout=2.0)
        tasks: list[asyncio.Task[None]] = []
        for task in (self._reader, self._stderr):
            if task is None:
                continue
            if not task.done():
                task.cancel()
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.process = None
        self._process_group = None
        self._closing = False

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        await self._send({"method": method, "id": request_id, "params": params})
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as error:
            raise HarnessError(
                "E_PROVIDER_TIMEOUT",
                "Codex app-server request timed out",
                retryable=True,
                status=504,
            ) from error
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"method": method, "params": params})

    async def _send(self, payload: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise ProviderUnavailableError("codex")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        process.stdin.write((serialized + "\n").encode("utf-8"))
        await process.stdin.drain()

    async def _read_messages(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        failure: BaseException | None = None
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                try:
                    payload = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                await self._route(payload)
        except BaseException as error:
            failure = error
        if failure is None and self._closing:
            return
        if failure is None:
            detail = "Codex app-server connection closed"
            if self.stderr_tail:
                detail += ": " + self.stderr_tail[-1]
            failure = ProviderUnavailableError("codex", detail=detail)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(failure)
        raise failure

    async def _route(self, payload: dict[str, Any]) -> None:
        request_id = payload.get("id")
        method = payload.get("method")
        if isinstance(request_id, int) and isinstance(method, str):
            params = _object(payload.get("params"))
            result = await self.request_handler(method, params)
            await self._send({"id": request_id, "result": result})
            return
        if isinstance(request_id, int):
            future = self._pending.get(request_id)
            if future is None or future.done():
                return
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message", "Codex request failed"))
                if _looks_exhausted(message):
                    future.set_exception(ProviderExhaustedError("codex"))
                else:
                    future.set_exception(
                        HarnessError(
                            "E_PROVIDER",
                            message,
                            retryable=True,
                            status=502,
                        )
                    )
                return
            future.set_result(_object(payload.get("result")))
            return
        if isinstance(method, str):
            await self.notification_handler(
                method,
                _object(payload.get("params")),
            )

    async def _read_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                self.stderr_tail.append(text)
                self.stderr_tail = self.stderr_tail[-40:]


class CodexAdapter(ProviderAdapter):
    provider_id = "codex"

    def __init__(self) -> None:
        self._active_server: CodexAppServer | None = None
        self._active_thread = ""
        self._active_turn = ""

    def status(self) -> ProviderStatus:
        ready = True
        detail = "trusted node and npx are available"
        try:
            trusted_executable("node")
            trusted_executable("npx")
        except (OSError, RuntimeError) as error:
            ready = False
            detail = str(error)
        return ProviderStatus(
            provider="codex",
            ready=ready,
            detail=detail,
            capabilities=frozenset(
                {
                    "approval",
                    "apps",
                    "checkpoint",
                    "fork",
                    "hooks",
                    "images",
                    "mcp",
                    "plugins",
                    "proof-fault-barrier",
                    "proof-service-fault-barrier",
                    "resume",
                    "skills",
                    "steer",
                    "streaming",
                    "subagents",
                    "tools",
                    "worktree",
                }
            ),
        )

    async def models(self, workspace: Path) -> tuple[ProviderModel, ...]:
        models: list[ProviderModel] = []

        async def ignore_notification(
            method: str,
            params: dict[str, Any],
        ) -> None:
            del method
            del params

        async def decline(
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            del method
            del params
            return {"decision": "decline"}

        server = CodexAppServer(
            workspace,
            notification_handler=ignore_notification,
            request_handler=decline,
        )
        try:
            await server.start()
            result = await server.request("model/list", {"limit": 100})
            raw_models = result.get("data")
            if not isinstance(raw_models, list):
                raw_models = result.get("models")
            if isinstance(raw_models, list):
                for item in raw_models:
                    if not isinstance(item, dict):
                        continue
                    model_id = str(item.get("id", item.get("model", "")))
                    if not model_id:
                        continue
                    efforts = _efforts(item)
                    context_window = _optional_int(
                        item.get("contextWindow") or item.get("context_window")
                    )
                    tiers = _strings(item.get("serviceTiers"))
                    models.append(
                        ProviderModel(
                            model_id=model_id,
                            display_name=str(item.get("displayName", model_id)),
                            efforts=efforts,
                            context_window=context_window,
                            default=bool(item.get("isDefault", False)),
                            service_tiers=tiers,
                        )
                    )
        finally:
            await server.close()
        return _merge_codex_config_models(tuple(models))

    async def run_turn(
        self,
        *,
        workspace: Path,
        prompt: str,
        native_session_id: str,
        permission_mode: str,
        model: str,
        effort: str,
        event_handler: EventHandler,
        approval_handler: ApprovalHandler,
        child_launch_gate: ChildLaunchGate | None = None,
        pre_prompt_gate: PrePromptGate | None = None,
    ) -> ProviderResult:
        completed = asyncio.Event()
        terminal_status = {"value": "failed"}
        usage: dict[str, Any] = {}

        async def notification(
            method: str,
            params: dict[str, Any],
        ) -> None:
            nonlocal native_session_id
            for event in codex_notification(method, params):
                if event.native_session_id:
                    native_session_id = event.native_session_id
                    self._active_thread = native_session_id
                if event.native_turn_id:
                    self._active_turn = event.native_turn_id
                if event.event_type == "usage.updated":
                    if event.metadata is not None:
                        usage.update(event.metadata)
                if event.event_type == "turn.completed":
                    terminal_status["value"] = event.status
                    completed.set()
                if event.event_type == "provider.error":
                    terminal_status["value"] = "failed"
                    completed.set()
                await event_handler(event)

        async def provider_request(
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            return await approval_handler(method, params)

        server = CodexAppServer(
            workspace,
            notification_handler=notification,
            request_handler=provider_request,
            child_launch_gate=child_launch_gate,
        )
        self._active_server = server
        try:
            await server.start()
            if pre_prompt_gate is not None:
                await pre_prompt_gate()
            # ChatGPT-backed Codex rejects model id "default"
            # (invalid_request_error). Resolve empty/default to the
            # operator's ~/.codex/config.toml model when present.
            model = resolve_codex_model(model)
            thread_params = {
                "cwd": str(workspace),
                "approvalPolicy": _approval_policy(permission_mode),
                "sandbox": _sandbox(permission_mode),
            }
            if model:
                thread_params["model"] = model
            if native_session_id:
                thread_params["threadId"] = native_session_id
                result = await server.request("thread/resume", thread_params)
            else:
                result = await server.request("thread/start", thread_params)
            thread_id = _nested_id(result, "thread")
            if not thread_id:
                raise HarnessError(
                    "E_PROVIDER_PROTOCOL",
                    "Codex did not return a thread identifier",
                    status=502,
                )
            native_session_id = thread_id
            self._active_thread = thread_id
            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "cwd": str(workspace),
                "input": [{"type": "text", "text": prompt}],
                "summary": "auto",
            }
            if model:
                turn_params["model"] = model
            if effort:
                turn_params["effort"] = effort
            turn_result = await server.request("turn/start", turn_params)
            turn_id = _nested_id(turn_result, "turn")
            if not turn_id:
                raise HarnessError(
                    "E_PROVIDER_PROTOCOL",
                    "Codex did not return a turn identifier",
                    status=502,
                )
            self._active_turn = turn_id
            await event_handler(
                ProviderEvent(
                    "provider.prompt.accepted",
                    status="accepted",
                    native_session_id=native_session_id,
                    native_turn_id=turn_id,
                )
            )
            completion_task = asyncio.create_task(completed.wait())
            reader_task = server._reader
            if reader_task is None:
                raise RuntimeError("Codex reader task is missing")
            try:
                finished, unused_pending = await asyncio.wait(
                    {completion_task, reader_task},
                    timeout=3600.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                del unused_pending
                if not finished:
                    raise TimeoutError("Codex turn timed out")
                if reader_task in finished:
                    await reader_task
                await completion_task
            finally:
                if not completion_task.done():
                    completion_task.cancel()
                await asyncio.gather(completion_task, return_exceptions=True)
            return ProviderResult(
                provider="codex",
                native_session_id=native_session_id,
                native_turn_id=turn_id,
                status=terminal_status["value"],
                usage=usage,
            )
        finally:
            await server.close()
            self._active_server = None
            self._active_thread = ""
            self._active_turn = ""

    async def interrupt(self) -> None:
        server = self._active_server
        if server is None:
            return
        if not self._active_thread or not self._active_turn:
            return
        await server.request(
            "turn/interrupt",
            {
                "threadId": self._active_thread,
                "turnId": self._active_turn,
            },
        )

    async def steer(self, text: str) -> None:
        server = self._active_server
        if server is None:
            raise ProviderUnavailableError("codex active turn")
        await server.request(
            "turn/steer",
            {
                "threadId": self._active_thread,
                "expectedTurnId": self._active_turn,
                "input": [{"type": "text", "text": text}],
            },
        )

    def process_identity(self) -> tuple[int, str]:
        server = self._active_server
        if server is None or server.process is None:
            return (0, "")
        if server.process.returncode is not None:
            return (0, "")
        identity = server._process_group
        if identity is None or identity.pid != server.process.pid:
            return (0, "")
        return (identity.pid, identity.pid_start)

    def native_session_available(
        self,
        workspace: Path,
        native_session_id: str,
    ) -> bool:
        del workspace
        return _has_native_session(native_session_id)


def codex_config_model(home: Path | None = None) -> str:
    """Return the non-default model pin from ~/.codex/config.toml.

    ChatGPT-auth Codex rejects the synthetic model id ``default``.
    Operators pin a real model in config.toml; the harness must
    advertise and dispatch that id.
    """
    base = Path.home() if home is None else home
    path = base / ".codex" / "config.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("model"):
            continue
        if "=" not in line:
            continue
        _, _, rest = line.partition("=")
        value = rest.strip().strip('"').strip("'")
        if value and value != "default":
            return value
    return ""


def resolve_codex_model(model: str) -> str:
    """Map empty/default model pins to the config.toml model when set."""
    if model and model != "default":
        return model
    configured = codex_config_model()
    if configured:
        return configured
    return model


def _merge_codex_config_models(
    listed: tuple[ProviderModel, ...],
) -> tuple[ProviderModel, ...]:
    """Ensure the config.toml model is advertised and preferred.

    When app-server only returns ``default`` (or nothing), ChatGPT
    accounts cannot run turns. Prefer the operator pin from config.toml
    as the default entry while keeping any other listed models.
    """
    efforts = ("low", "medium", "high", "xhigh")
    configured = codex_config_model()
    if not listed:
        if not configured:
            return ()
        return (
            ProviderModel(
                configured,
                configured + " (config.toml)",
                efforts,
                None,
                default=True,
            ),
        )
    ids = {item.model_id for item in listed}
    only_placeholder = ids <= {"default"}
    if not configured:
        return listed
    if configured in ids and not only_placeholder:
        return listed
    prepend = ProviderModel(
        configured,
        configured + " (config.toml)",
        efforts,
        None,
        default=True,
    )
    rest: list[ProviderModel] = []
    for item in listed:
        if item.model_id == configured:
            continue
        rest.append(
            ProviderModel(
                item.model_id,
                item.display_name,
                item.efforts,
                item.context_window,
                default=False if only_placeholder else item.default,
                service_tiers=item.service_tiers,
            )
        )
    return (prepend, *rest)


def _approval_policy(permission_mode: str) -> str:
    if permission_mode == "approval":
        return "on-request"
    if permission_mode in {"full", "read-only", "plan"}:
        return "never"
    raise ValueError("unsupported permission mode")


def _sandbox(permission_mode: str) -> str:
    if permission_mode == "full":
        return "danger-full-access"
    if permission_mode in {"read-only", "plan"}:
        return "read-only"
    if permission_mode == "approval":
        return "workspace-write"
    raise ValueError("unsupported permission mode")


def _object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _nested_id(value: dict[str, Any], field: str) -> str:
    nested = value.get(field)
    if isinstance(nested, dict):
        identifier = nested.get("id")
        if isinstance(identifier, str):
            return identifier
    return ""


def _has_native_session(native_session_id: str) -> bool:
    if not native_session_id:
        return False
    if not all(
        character.isalnum() or character in {"-", "_"}
        for character in native_session_id
    ):
        return False
    root = Path.home() / ".codex" / "sessions"
    if not root.is_dir():
        return False
    pattern = "*" + native_session_id + "*.jsonl"
    return next(root.rglob(pattern), None) is not None


def _looks_exhausted(message: str) -> bool:
    lowered = message.casefold()
    terms = ("rate limit", "usage limit", "quota", "capacity")
    return any(term in lowered for term in terms)


def _efforts(value: dict[str, Any]) -> tuple[str, ...]:
    raw = value.get("supportedReasoningEfforts")
    if not isinstance(raw, list):
        raw = value.get("efforts")
    if not isinstance(raw, list):
        return ()
    efforts: list[str] = []
    for item in raw:
        if isinstance(item, str):
            efforts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        effort = item.get("reasoningEffort")
        if isinstance(effort, str):
            efforts.append(effort)
    return tuple(efforts)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    strings: list[str] = []
    for item in value:
        if isinstance(item, str):
            strings.append(item)
            continue
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        if isinstance(identifier, str):
            strings.append(identifier)
    return tuple(strings)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
