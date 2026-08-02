"""Deterministic coverage for provider process boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import RateLimitInfo

from agent_harness.errors import HarnessError
from agent_harness.errors import ProviderExhaustedError
from agent_harness.errors import ProviderUnavailableError
from agent_harness.providers import claude
from agent_harness.providers import kimi
from agent_harness.providers import normalize
from agent_harness.providers import codex
from agent_harness.providers import base
from agent_harness import terminal
from agent_harness.providers.base import ProviderAdapter
from agent_harness.providers.base import ProviderEvent


class Stream:
    def __init__(self, values: list[bytes]) -> None:
        self.values = list(values)
        self.writes: list[bytes] = []
        self.closed = False

    async def readline(self) -> bytes:
        if self.values:
            return self.values.pop(0)
        return b""

    def write(self, value: bytes) -> None:
        self.writes.append(value)

    async def drain(self) -> None:
        return

    def close(self) -> None:
        self.closed = True


class Process:
    def __init__(
        self,
        *,
        stdout: Stream | None = None,
        stderr: Stream | None = None,
    ) -> None:
        self.stdin = Stream([])
        self.stdout = stdout
        self.stderr = stderr
        self.pid = 42
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


async def _ignore_event(event: ProviderEvent) -> None:
    del event


async def _ignore_notification(
    method: str,
    params: dict[str, Any],
) -> None:
    del method
    del params


async def _decline(
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    del method
    del params
    return {"decision": "decline"}


def _server(tmp_path: Path) -> codex.CodexAppServer:
    return codex.CodexAppServer(
        tmp_path,
        notification_handler=_ignore_notification,
        request_handler=_decline,
    )


def test_provider_base_contract_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Adapter(ProviderAdapter):
        provider_id = "test"

        async def run_turn(self, **kwargs: Any):
            del kwargs
            return await super().run_turn(
                workspace=Path("."),
                prompt="",
                native_session_id="",
                permission_mode="approval",
                model="",
                effort="",
                event_handler=_ignore_event,
                approval_handler=_decline,
            )

        async def models(self, workspace: Path):
            return await super().models(workspace)

        def status(self):
            return super().status()

    Adapter.__abstractmethods__ = frozenset()
    adapter = Adapter()
    with pytest.raises(NotImplementedError):
        asyncio.run(adapter.run_turn())
    with pytest.raises(NotImplementedError):
        asyncio.run(adapter.models(Path(".")))
    with pytest.raises(NotImplementedError):
        adapter.status()
    asyncio.run(adapter.interrupt())
    asyncio.run(adapter.steer("continue"))
    assert adapter.process_identity() == (0, "")

    monkeypatch.setattr(
        base.os,
        "environ",
        {
            "PATH": "/bin",
            "npm_config_package": "ignored",
            "OTHER_SECRET": "ignored",
            "ANTHROPIC_API_KEY": "anthropic",
            "OPENAI_API_KEY": "openai",
        },
    )
    assert codex.provider_environment("codex") == {"PATH": "/bin"}
    assert codex.provider_environment("codex", "api") == {
        "PATH": "/bin",
        "OPENAI_API_KEY": "openai",
    }
    assert codex.provider_environment("claude", "api") == {
        "PATH": "/bin",
        "ANTHROPIC_API_KEY": "anthropic",
    }


def test_codex_protocol_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert codex._approval_policy("approval") == "on-request"
    assert codex._approval_policy("full") == "never"
    assert codex._approval_policy("read-only") == "never"
    assert codex._approval_policy("plan") == "never"
    with pytest.raises(ValueError):
        codex._approval_policy("invalid")

    assert codex._sandbox("full") == "danger-full-access"
    assert codex._sandbox("read-only") == "read-only"
    assert codex._sandbox("plan") == "read-only"
    assert codex._sandbox("approval") == "workspace-write"
    with pytest.raises(ValueError):
        codex._sandbox("invalid")

    assert codex._object({"value": 1}) == {"value": 1}
    assert codex._object("not-an-object") == {}
    assert codex._nested_id({"thread": {"id": "t"}}, "thread") == "t"
    assert codex._nested_id({"thread": {"id": 1}}, "thread") == ""
    assert codex._nested_id({}, "thread") == ""
    assert codex._looks_exhausted("Usage limit reached")
    assert not codex._looks_exhausted("ordinary error")
    assert codex._optional_int(4) == 4
    assert codex._optional_int(True) is None
    assert codex._optional_int("4") is None
    assert codex._efforts({}) == ()
    assert codex._efforts({"efforts": "high"}) == ()
    assert codex._efforts(
        {
            "efforts": [
                "low",
                1,
                {"reasoningEffort": "high"},
                {"reasoningEffort": 2},
            ]
        }
    ) == ("low", "high")
    assert codex._strings("fast") == ()
    assert codex._strings(
        ["fast", 1, {"id": "flex"}, {"id": 2}]
    ) == ("fast", "flex")

    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "")
    assert codex._process_start(42) == "42"
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: " ".join(str(index) for index in range(24)),
    )
    assert codex._process_start(42) == "21"


def test_codex_server_send_request_and_routes(tmp_path: Path) -> None:
    async def scenario() -> None:
        notifications: list[tuple[str, dict[str, Any]]] = []
        requests: list[tuple[str, dict[str, Any]]] = []

        async def notification(
            method: str,
            params: dict[str, Any],
        ) -> None:
            notifications.append((method, params))

        async def request(
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            requests.append((method, params))
            return {"decision": "accept"}

        server = codex.CodexAppServer(
            tmp_path,
            notification_handler=notification,
            request_handler=request,
        )
        with pytest.raises(ProviderUnavailableError):
            await server._send({})

        process = Process()
        server.process = process  # type: ignore[assignment]
        await server.notify("ready", {"value": 1})
        assert process.stdin.writes

        async def reply(payload: dict[str, Any]) -> None:
            await server._route(
                {"id": payload["id"], "result": {"answer": 7}}
            )

        server._send = reply  # type: ignore[method-assign]
        assert await server.request("question", {}) == {"answer": 7}
        assert not server._pending

        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        completed.set_result({})
        server._pending[10] = completed
        await server._route({"id": 10, "result": {}})
        await server._route({"id": 99, "result": {}})

        exhausted = loop.create_future()
        server._pending[11] = exhausted
        await server._route(
            {"id": 11, "error": {"message": "quota reached"}}
        )
        assert isinstance(exhausted.exception(), ProviderExhaustedError)

        failed = loop.create_future()
        server._pending[12] = failed
        await server._route({"id": 12, "error": {}})
        assert isinstance(failed.exception(), HarnessError)

        successful = loop.create_future()
        server._pending[13] = successful
        await server._route({"id": 13, "result": "invalid"})
        assert successful.result() == {}

        await server._route({"method": "event", "params": {"x": 1}})
        assert notifications == [("event", {"x": 1})]
        await server._route(
            {"id": 14, "method": "approval", "params": {"x": 2}}
        )
        assert requests == [("approval", {"x": 2})]

    asyncio.run(scenario())


def test_codex_server_readers_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        server = _server(tmp_path)
        server.process = Process(
            stdout=Stream(
                [
                    b"not-json\n",
                    b"[]\n",
                    b'{"method":"notice","params":{}}\n',
                ]
            ),
            stderr=Stream([b"first\n", b"\xffsecond\n"]),
        )  # type: ignore[assignment]
        server.stderr_tail = ["last stderr"]
        pending = asyncio.get_running_loop().create_future()
        server._pending[1] = pending
        await server._read_messages()
        assert isinstance(pending.exception(), ProviderUnavailableError)
        await server._read_stderr()
        assert server.stderr_tail == [
            "last stderr",
            "first",
            "�second",
        ]

        server.process = None
        await server._read_messages()
        await server._read_stderr()
        await server.close()

        async def send(payload: dict[str, Any]) -> None:
            del payload

        async def timeout(
            future: asyncio.Future[dict[str, Any]],
            *,
            timeout: float,
        ) -> dict[str, Any]:
            del future
            del timeout
            raise TimeoutError

        server._send = send  # type: ignore[method-assign]
        monkeypatch.setattr(asyncio, "wait_for", timeout)
        with pytest.raises(HarnessError, match="timed out"):
            await server.request("slow", {}, timeout=0.01)
        assert not server._pending

    asyncio.run(scenario())


def test_codex_server_start_and_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = Process(stdout=Stream([]), stderr=Stream([]))

        async def create(*args: Any, **kwargs: Any) -> Process:
            assert args[:3] == ("npx", "-y", "@openai/codex@0.146.0")
            assert kwargs["cwd"] == tmp_path
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        server = _server(tmp_path)
        calls: list[str] = []

        async def request(
            method: str,
            params: dict[str, Any],
            *,
            timeout: float = 30.0,
        ) -> dict[str, Any]:
            del params
            del timeout
            calls.append(method)
            return {}

        async def notify(method: str, params: dict[str, Any]) -> None:
            del params
            calls.append(method)

        server.request = request  # type: ignore[method-assign]
        server.notify = notify  # type: ignore[method-assign]
        await server.start()
        await server.start()
        assert calls == ["initialize", "initialized"]
        await server.close()
        assert process.stdin.closed
        assert server.process is None

    asyncio.run(scenario())


def test_codex_close_escalates_and_reader_preserves_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = Process(
            stdout=Stream([b'{"method":"event"}\n']),
            stderr=Stream([]),
        )
        server = _server(tmp_path)
        server.process = process  # type: ignore[assignment]

        async def broken_route(payload: dict[str, Any]) -> None:
            del payload
            raise RuntimeError("reader failed")

        server._route = broken_route  # type: ignore[method-assign]
        pending = asyncio.get_running_loop().create_future()
        server._pending[1] = pending
        await server._read_messages()
        assert isinstance(pending.exception(), RuntimeError)

        calls = 0

        async def wait_for(
            future: Any,
            *,
            timeout: float,
        ) -> int:
            nonlocal calls
            del future
            del timeout
            calls += 1
            if calls <= 2:
                raise TimeoutError
            return 0

        monkeypatch.setattr(asyncio, "wait_for", wait_for)
        await server.close()
        assert process.terminated
        assert process.killed

    asyncio.run(scenario())


def test_codex_adapter_models_and_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Server:
        instances: list["Server"] = []

        def __init__(
            self,
            workspace: Path,
            *,
            notification_handler: Any,
            request_handler: Any,
        ) -> None:
            self.workspace = workspace
            self.notification_handler = notification_handler
            self.request_handler = request_handler
            self.process = SimpleNamespace(pid=44)
            self.closed = False
            self.instances.append(self)

        async def start(self) -> None:
            if self.notification_handler.__name__ == (
                "ignore_notification"
            ):
                await self.notification_handler("notice", {})
            await self.request_handler("approval", {})

        async def request(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            if method == "model/list":
                return {
                    "models": [
                        "invalid",
                        {},
                        {
                            "model": "gpt-test",
                            "displayName": "Test",
                            "efforts": ["low"],
                            "context_window": 100,
                            "serviceTiers": ["fast"],
                            "isDefault": True,
                        },
                    ]
                }
            if method in {"thread/start", "thread/resume"}:
                return {"thread": {"id": "thread-1"}}
            if method == "turn/start":
                for event in (
                    ProviderEvent(
                        "agent.message",
                        text="working",
                        native_session_id="thread-1",
                        native_turn_id="turn-1",
                    ),
                    ProviderEvent(
                        "usage.updated",
                        metadata={"total_tokens": 9},
                    ),
                    ProviderEvent("turn.completed", status="complete"),
                ):
                    await self.notification_handler("event", {})
                return {"turn": {"id": "turn-1"}}
            return {}

        async def close(self) -> None:
            self.closed = True

    provider_events = iter(
        [
            [
                ProviderEvent(
                    "agent.message",
                    text="working",
                    native_session_id="thread-1",
                    native_turn_id="turn-1",
                )
            ],
            [
                ProviderEvent(
                    "usage.updated",
                    metadata={"total_tokens": 9},
                )
            ],
            [ProviderEvent("turn.completed", status="complete")],
        ]
    )
    monkeypatch.setattr(codex, "CodexAppServer", Server)
    monkeypatch.setattr(
        codex,
        "codex_notification",
        lambda method, params: next(provider_events),
    )

    async def scenario() -> None:
        adapter = codex.CodexAdapter()
        models = await adapter.models(tmp_path)
        assert models[0].model_id == "gpt-test"
        assert models[0].default

        events: list[ProviderEvent] = []

        async def collect(event: ProviderEvent) -> None:
            events.append(event)

        result = await adapter.run_turn(
            workspace=tmp_path,
            prompt="work",
            native_session_id="",
            permission_mode="approval",
            model="gpt-test",
            effort="high",
            event_handler=collect,
            approval_handler=_decline,
        )
        assert result.status == "complete"
        assert result.usage == {"total_tokens": 9}
        assert len(events) == 3
        assert adapter.process_identity() == (0, "")
        await adapter.interrupt()
        with pytest.raises(ProviderUnavailableError):
            await adapter.steer("more")

        active = Server(
            tmp_path,
            notification_handler=_ignore_notification,
            request_handler=_decline,
        )
        adapter._active_server = active  # type: ignore[assignment]
        adapter._active_thread = "thread"
        adapter._active_turn = "turn"
        await adapter.interrupt()
        await adapter.steer("more")
        assert adapter.process_identity()[0] == 44

    asyncio.run(scenario())


def test_codex_adapter_status_resume_failure_and_interrupt_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex.shutil, "which", lambda command: None)
    assert not codex.CodexAdapter().status().ready
    monkeypatch.setattr(
        codex.shutil,
        "which",
        lambda command: "/usr/bin/npx",
    )
    assert codex.CodexAdapter().status().ready

    class Server:
        mode = "failed"

        def __init__(
            self,
            workspace: Path,
            *,
            notification_handler: Any,
            request_handler: Any,
        ) -> None:
            del workspace
            self.notification_handler = notification_handler
            self.request_handler = request_handler
            self.process = None

        async def start(self) -> None:
            return

        async def request(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            if method == "thread/resume":
                if self.mode == "missing":
                    return {}
                return {"thread": {"id": "thread-resumed"}}
            if method == "turn/start":
                await self.notification_handler("failed", {})
                return {"turn": {"id": "turn-failed"}}
            return {}

        async def close(self) -> None:
            return

    monkeypatch.setattr(codex, "CodexAppServer", Server)
    monkeypatch.setattr(
        codex,
        "codex_notification",
        lambda method, params: [
            ProviderEvent("provider.error", status="failed")
        ],
    )

    async def scenario() -> None:
        async def collect(event: ProviderEvent) -> None:
            del event

        adapter = codex.CodexAdapter()
        result = await adapter.run_turn(
            workspace=tmp_path,
            prompt="resume",
            native_session_id="native",
            permission_mode="plan",
            model="",
            effort="",
            event_handler=collect,
            approval_handler=_decline,
        )
        assert result.status == "failed"

        Server.mode = "missing"
        with pytest.raises(HarnessError, match="thread identifier"):
            await adapter.run_turn(
                workspace=tmp_path,
                prompt="resume",
                native_session_id="native",
                permission_mode="read-only",
                model="",
                effort="",
                event_handler=collect,
                approval_handler=_decline,
            )

        active = Server(
            tmp_path,
            notification_handler=_ignore_notification,
            request_handler=_decline,
        )
        adapter._active_server = active  # type: ignore[assignment]
        adapter._active_thread = ""
        adapter._active_turn = ""
        await adapter.interrupt()

    asyncio.run(scenario())


def test_claude_helpers_and_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claude.shutil, "which", lambda command: None)
    transport = object.__new__(claude.NpxClaudeTransport)
    with pytest.raises(claude.CLINotFoundError):
        transport._find_cli()
    monkeypatch.setattr(
        claude.shutil,
        "which",
        lambda command: "/usr/bin/npx",
    )
    assert transport._find_cli() == "/usr/bin/npx"
    asyncio.run(transport._check_claude_version())

    assert claude._permission_mode("full") == "bypassPermissions"
    assert claude._permission_mode("plan") == "plan"
    assert claude._permission_mode("read-only") == "plan"
    assert claude._permission_mode("approval") == "default"
    with pytest.raises(ValueError):
        claude._permission_mode("invalid")

    assert claude._message_session_id(
        SimpleNamespace(session_id="session")
    ) == "session"
    assert claude._message_session_id(object()) == ""
    assert claude._content_payload(object()) == {"type": "unknown"}
    assert claude._message_events(object())[0].event_type == "provider.event"
    assert claude._looks_exhausted("Capacity reached")
    assert not claude._looks_exhausted("healthy")
    assert claude._looks_like_spend_limit(
        "Monthly spend limit; use /usage-credits"
    )
    assert not claude._looks_like_spend_limit("monthly spend limit")
    assert claude._bounded("x" * 5000) == "x" * 4096
    assert not claude._has_native_session(tmp_path, "missing")

    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "")
    assert claude._process_start(7) == "7"
    def unavailable(*args: Any, **kwargs: Any) -> str:
        del args
        del kwargs
        raise OSError

    monkeypatch.setattr(Path, "read_text", unavailable)
    assert claude._process_start(7) == "7"
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: " ".join(str(index) for index in range(24)),
    )
    assert claude._process_start(7) == "21"

    async def prompt() -> None:
        values = [
            value async for value in claude._prompt_stream("hello", "session")
        ]
        assert values[0]["message"]["content"][0]["text"] == "hello"
        assert [value async for value in claude._empty_stream()] == []

    asyncio.run(prompt())


def test_claude_message_normalization_boundaries() -> None:
    blocks = [
        claude.TextBlock("text"),
        claude.ThinkingBlock("thought", "signature"),
        claude.ToolUseBlock("tool-1", "Bash", {"command": "true"}),
        claude.ToolResultBlock("tool-1", "done", False),
        claude.ServerToolUseBlock("server-1", "advisor", {}),
        claude.ServerToolResultBlock("server-1", {"type": "done"}),
        object(),
    ]
    payloads = [claude._content_payload(block) for block in blocks]
    assert [payload["type"] for payload in payloads] == [
        "text",
        "thinking",
        "tool_use",
        "tool_result",
        "server_tool_use",
        "advisor_tool_result",
        "unknown",
    ]

    messages = [
        claude.SystemMessage("init", {"session_id": "system-session"}),
        claude.UserMessage("hello", uuid="user-1"),
        claude.UserMessage(blocks[:1], uuid="user-2"),
        claude.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="result-session",
        ),
        claude.StreamEvent(
            uuid="stream-1",
            session_id="stream-session",
            event={"type": "delta"},
        ),
        claude.RateLimitEvent(
                rate_limit_info=RateLimitInfo(
                status="allowed_warning",
                utilization=0.8,
            ),
            uuid="rate-1",
            session_id="rate-session",
        ),
    ]
    for message in messages:
        assert isinstance(claude._message_events(message), list)
    assert claude._message_session_id(messages[0]) == "system-session"

    rejected = claude.RateLimitEvent(
        rate_limit_info=RateLimitInfo(status="rejected"),
        uuid="rate-2",
        session_id="rate-session",
    )
    assert claude._message_exhausted(rejected)
    assert claude._message_exhausted(
        claude.AssistantMessage(
            content=[],
            model="opus",
            error="rate_limit",
        )
    )
    assert not claude._message_exhausted(
        claude.AssistantMessage(
            content=[claude.ThinkingBlock("safe", "")],
            model="opus",
        )
    )
    assert claude._message_exhausted(
        claude.ResultMessage(
            subtype="failed",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="result-session",
            api_error_status=429,
        )
    )
    assert claude._message_exhausted(
        claude.ResultMessage(
            subtype="quota exceeded",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="result-session",
        )
    )
    assert not claude._message_exhausted(object())


def test_claude_run_turn_success_exhaustion_and_sdk_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Transport:
        def __init__(self, stream: object, options: object) -> None:
            del stream
            self.options = options

    class Client:
        mode = "success"
        messages: list[object] = []

        def __init__(self, *, options: Any, transport: Any) -> None:
            self.options = options
            self.transport = transport
            self.disconnected = False
            if self.mode != "empty-failure":
                for index in range(101):
                    options.stderr("line-" + str(index))
            if self.mode == "failure":
                options.stderr("ordinary provider failure")
            if self.mode == "quota":
                options.stderr("quota reached")

        async def connect(
            self,
            stream: AsyncIterator[dict[str, Any]],
        ) -> None:
            assert [value async for value in stream]
            callback = self.options.can_use_tool
            if callback is not None:
                result = await callback(
                    "Bash",
                    {"command": "true"},
                    claude.ToolPermissionContext(
                        tool_use_id="tool-1",
                    ),
                )
                assert isinstance(result, claude.PermissionResultDeny)

        async def receive_response(self) -> AsyncIterator[object]:
            if self.mode in {"failure", "quota", "empty-failure"}:
                raise claude.ClaudeSDKError("sdk failed")
            for message in self.messages:
                yield message

        async def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(claude, "NpxClaudeTransport", Transport)
    monkeypatch.setattr(claude, "ClaudeSDKClient", Client)
    monkeypatch.setattr(claude, "_has_native_session", lambda *args: False)
    monkeypatch.setattr(claude.uuid, "uuid4", lambda: "new-session")

    event_map = {
        "message": [
            ProviderEvent(
                "agent.message",
                metadata={"usage": {"total_tokens": 3}},
            ),
            ProviderEvent("turn.completed"),
        ],
        "failed": [
            ProviderEvent(
                "turn.failed",
                native_session_id="reported-session",
            )
        ],
        "exhausted": [ProviderEvent("agent.message")],
    }
    monkeypatch.setattr(
        claude,
        "_message_events",
        lambda message: event_map[str(message)],
    )
    monkeypatch.setattr(
        claude,
        "_message_exhausted",
        lambda message: message == "exhausted",
    )

    async def scenario() -> None:
        events: list[ProviderEvent] = []

        async def collect(event: ProviderEvent) -> None:
            events.append(event)

        Client.mode = "success"
        Client.messages = ["message"]
        adapter = claude.ClaudeAdapter()
        result = await adapter.run_turn(
            workspace=tmp_path,
            prompt="work",
            native_session_id="",
            permission_mode="approval",
            model="opus",
            effort="high",
            event_handler=collect,
            approval_handler=_decline,
        )
        assert result.status == "complete"
        assert result.native_session_id == "new-session"
        assert result.usage == {"total_tokens": 3}
        assert adapter._client is None
        assert adapter._transport is None

        Client.messages = [
            SimpleNamespace(
                session_id="reported-session",
                __str__=lambda self: "failed",
            )
        ]
        monkeypatch.setattr(
            claude,
            "_message_events",
            lambda message: event_map["failed"],
        )
        monkeypatch.setattr(
            claude,
            "_message_exhausted",
            lambda message: False,
        )
        result = await adapter.run_turn(
            workspace=tmp_path,
            prompt="work",
            native_session_id="native",
            permission_mode="full",
            model="",
            effort="",
            event_handler=collect,
            approval_handler=_decline,
        )
        assert result.status == "failed"
        assert result.native_session_id == "reported-session"

        monkeypatch.setattr(
            claude,
            "_message_events",
            lambda message: event_map[str(message)],
        )
        monkeypatch.setattr(
            claude,
            "_message_exhausted",
            lambda message: message == "exhausted",
        )

        Client.messages = ["exhausted"]
        with pytest.raises(ProviderExhaustedError):
            await adapter.run_turn(
                workspace=tmp_path,
                prompt="work",
                native_session_id="native",
                permission_mode="full",
                model="",
                effort="",
                event_handler=collect,
                approval_handler=_decline,
            )

        Client.mode = "failure"
        with pytest.raises(ProviderUnavailableError):
            await adapter.run_turn(
                workspace=tmp_path,
                prompt="work",
                native_session_id="native",
                permission_mode="plan",
                model="",
                effort="",
                event_handler=collect,
                approval_handler=_decline,
            )
        assert any(event.event_type == "provider.error" for event in events)

        Client.mode = "empty-failure"
        with pytest.raises(ProviderUnavailableError):
            await adapter.run_turn(
                workspace=tmp_path,
                prompt="work",
                native_session_id="native",
                permission_mode="plan",
                model="",
                effort="",
                event_handler=collect,
                approval_handler=_decline,
            )

        Client.mode = "quota"
        with pytest.raises(ProviderExhaustedError):
            await adapter.run_turn(
                workspace=tmp_path,
                prompt="work",
                native_session_id="native",
                permission_mode="read-only",
                model="",
                effort="",
                event_handler=collect,
                approval_handler=_decline,
            )

    asyncio.run(scenario())


def test_claude_adapter_status_models_and_active_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = claude.ClaudeAdapter()
    monkeypatch.setattr(
        claude.shutil,
        "which",
        lambda command: "/usr/bin/npx",
    )
    assert adapter.status().ready
    monkeypatch.setattr(claude.shutil, "which", lambda command: None)
    assert not adapter.status().ready
    assert len(asyncio.run(adapter.models(tmp_path))) == 3
    asyncio.run(adapter.interrupt())
    asyncio.run(adapter.steer("continue"))
    assert adapter.process_identity() == (0, "")

    class Client:
        async def interrupt(self) -> None:
            self.interrupted = True

        async def query(self, text: str) -> None:
            self.text = text

    client = Client()
    adapter._client = client  # type: ignore[assignment]
    asyncio.run(adapter.interrupt())
    asyncio.run(adapter.steer("continue"))
    assert client.interrupted
    assert client.text == "continue"

    adapter._transport = SimpleNamespace(_process=None)
    assert adapter.process_identity() == (0, "")
    adapter._transport = SimpleNamespace(_process=SimpleNamespace(pid=0))
    assert adapter.process_identity() == (0, "")
    monkeypatch.setattr(claude, "_process_start", lambda pid: "started")
    adapter._transport = SimpleNamespace(_process=SimpleNamespace(pid=55))
    assert adapter.process_identity() == (55, "started")


def test_terminal_command_text_and_pty_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert terminal._command("codex", "approval", ["resume"]) == [
        "npx",
        "-y",
        "@openai/codex@0.146.0",
        "resume",
    ]
    assert "--yolo" in terminal._command("codex", "full", [])
    assert "--dangerously-skip-permissions" in terminal._command(
        "claude",
        "full",
        [],
    )
    assert terminal._command("claude", "approval", [])[:2] == [
        "npx",
        "@anthropic-ai/claude-code@2.1.220",
    ]
    with pytest.raises(ValueError):
        terminal._command("other", "approval", [])

    writes: list[tuple[int, bytes]] = []
    sizes: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(
        terminal.os,
        "write",
        lambda descriptor, value: writes.append((descriptor, value)),
    )
    monkeypatch.setattr(
        terminal.termios,
        "tcsetwinsize",
        lambda descriptor, size: sizes.append((descriptor, size)),
    )

    async def scenario() -> None:
        await terminal._text_message(3, {"type": "input", "data": "hi"})
        await terminal._text_message(
            3,
            {"type": "resize", "rows": 30, "columns": 100},
        )
        await terminal._text_message(3, {"type": "ignored"})
        assert writes == [(3, b"hi")]
        assert sizes == [(3, (30, 100))]

        class Socket:
            closed = False

            def __init__(self) -> None:
                self.values = [b"content", b""]
                self.sent: list[bytes] = []

            async def send_bytes(self, value: bytes) -> None:
                self.sent.append(value)

        socket = Socket()
        monkeypatch.setattr(
            terminal.os,
            "read",
            lambda descriptor, size: socket.values.pop(0),
        )
        await terminal._pty_to_socket(3, socket)  # type: ignore[arg-type]
        assert socket.sent == [b"content"]

        def failed_read(descriptor: int, size: int) -> bytes:
            del descriptor
            del size
            raise OSError

        monkeypatch.setattr(terminal.os, "read", failed_read)
        await terminal._pty_to_socket(3, socket)  # type: ignore[arg-type]
        socket.closed = True
        await terminal._pty_to_socket(3, socket)  # type: ignore[arg-type]

    asyncio.run(scenario())


def test_terminal_socket_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Socket:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.closed = False
            self.messages = [
                SimpleNamespace(
                    type=terminal.WSMsgType.BINARY,
                    data=b"binary",
                ),
                SimpleNamespace(
                    type=terminal.WSMsgType.TEXT,
                    json=lambda: {"type": "input", "data": "text"},
                ),
                SimpleNamespace(type=terminal.WSMsgType.ERROR),
            ]

        async def prepare(self, request: object) -> None:
            self.request = request

        async def receive_json(self) -> dict[str, Any]:
            return {
                "provider": "codex",
                "permission_mode": "approval",
                "arguments": "invalid",
            }

        def __aiter__(self) -> "Socket":
            return self

        async def __anext__(self) -> object:
            if not self.messages:
                raise StopAsyncIteration
            return self.messages.pop(0)

    class Child:
        returncode = None

        def __init__(self) -> None:
            self.signals: list[int] = []

        def send_signal(self, value: int) -> None:
            self.signals.append(value)

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    child = Child()
    closed: list[int] = []
    writes: list[bytes] = []

    async def create(*args: Any, **kwargs: Any) -> Child:
        assert args[:3] == ("npx", "-y", "@openai/codex@0.146.0")
        assert kwargs["cwd"] == tmp_path
        return child

    async def relay(descriptor: int, socket: object) -> None:
        del descriptor
        del socket
        await asyncio.sleep(0)

    async def text(descriptor: int, payload: dict[str, Any]) -> None:
        del descriptor
        assert payload["type"] == "input"

    monkeypatch.setattr(terminal.web, "WebSocketResponse", Socket)
    monkeypatch.setattr(terminal.pty, "openpty", lambda: (10, 11))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(terminal, "_pty_to_socket", relay)
    monkeypatch.setattr(terminal, "_text_message", text)
    monkeypatch.setattr(terminal.os, "close", closed.append)
    monkeypatch.setattr(
        terminal.os,
        "write",
        lambda descriptor, value: writes.append(value),
    )

    async def scenario() -> None:
        result = await terminal.terminal_socket(
            SimpleNamespace(),  # type: ignore[arg-type]
            tmp_path,
        )
        assert isinstance(result, Socket)

    asyncio.run(scenario())
    assert writes == [b"binary"]
    assert child.signals == [terminal.signal.SIGTERM]
    assert closed == [11, 10]


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "--import-mode=importlib",
            ]
        )
    )


# Kimi's stream-json envelope is flat {"role", "content"}, not Claude's
# {"type": "assistant", "message": {"content": [blocks]}}. Reusing the
# Claude normalizer here silently yields nothing.
def test_kimi_payload_normalizes_the_flat_envelope() -> None:
    events = normalize.kimi_payload({"role": "assistant", "content": "ok"})
    assert [(e.event_type, e.text) for e in events] == [("agent.message", "ok")]

    events = normalize.kimi_payload({"role": "user", "content": "hi"})
    assert events[0].event_type == "user.message"

    events = normalize.kimi_payload(
        {
            "role": "meta",
            "type": "session.resume_hint",
            "session_id": "session_abc",
        }
    )
    assert events[0].event_type == "turn.completed"
    assert events[0].native_session_id == "session_abc"

    assert normalize.kimi_payload({"role": "meta", "type": "other"})[0].event_type == (
        "provider.event"
    )


# --yolo is rejected together with --prompt, so the launcher must not
# pass it; tool permissions come from ~/.kimi-code/config.toml.
def test_kimi_launch_argv_omits_yolo_and_pins_the_package() -> None:
    argv = kimi._launch_argv("do it", "kimi-code/k3", "session_abc")

    assert "--yolo" not in argv
    assert argv[:6] == [
        "npx",
        "--yes",
        "--package",
        "@moonshot-ai/kimi-code",
        "kimi",
        "--prompt",
    ]
    assert argv[-4:] == [
        "--model",
        "kimi-code/k3",
        "--session",
        "session_abc",
    ]
    assert "--session" not in kimi._launch_argv("do it", "", "")


def test_kimi_status_and_models(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(kimi.shutil, "which", lambda _name: "/usr/bin/npx")
    status = kimi.KimiAdapter().status()
    assert status.ready
    assert status.provider == "kimi"
    # A one-shot run has no live turn to steer and cannot prompt for
    # approval, so neither capability is claimed.
    assert "approval" not in status.capabilities
    assert "streaming" in status.capabilities

    monkeypatch.setattr(kimi.shutil, "which", lambda _name: None)
    assert not kimi.KimiAdapter().status().ready

    async def scenario() -> None:
        models = await kimi.KimiAdapter().models(tmp_path)
        # `--model` resolves a config.toml alias key, not the upstream
        # API model id, so every id here must carry its provider
        # namespace. A bare "k3" fails with config.invalid.
        assert models[0].model_id == "kimi-code/k3"
        assert models[0].default
        for model in models:
            assert model.model_id.startswith("kimi-code/")

    asyncio.run(scenario())


def test_kimi_run_turn_streams_and_reports_the_session(monkeypatch, tmp_path) -> None:
    lines = [
        b'{"role":"assistant","content":"ok"}\n',
        b'{"role":"meta","type":"session.resume_hint","session_id":"session_z"}\n',
        b"not json\n",
    ]

    class Stdout:
        def __aiter__(self):
            async def gen():
                for line in lines:
                    yield line

            return gen()

    class Stderr:
        async def read(self) -> bytes:
            return b""

    class Process:
        stdout = Stdout()
        stderr = Stderr()

        async def wait(self) -> int:
            return 0

    async def fake_exec(*_argv, **_kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    seen: list[ProviderEvent] = []

    async def handler(event: ProviderEvent) -> None:
        seen.append(event)

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def scenario() -> None:
        result = await kimi.KimiAdapter().run_turn(
            workspace=tmp_path,
            prompt="do it",
            native_session_id="",
            permission_mode="auto",
            model="kimi-code/k3",
            effort="",
            event_handler=handler,
            approval_handler=approvals,
        )
        assert result.status == "complete"
        # The session id arrives on the trailing meta line, not up front.
        assert result.native_session_id == "session_z"

    asyncio.run(scenario())

    assert [e.event_type for e in seen] == ["agent.message", "turn.completed"]
