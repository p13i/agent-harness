"""Deterministic coverage for provider process boundaries."""

from __future__ import annotations

import asyncio
import io
import runpy
import sqlite3
import sys
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import RateLimitInfo

from agent_harness import child_gate, terminal
from agent_harness.errors import (
    HarnessError,
    ProviderExhaustedError,
    ProviderUnavailableError,
    SafetyGuardError,
)
from agent_harness.providers import base, claude, codex, grok, kimi, normalize
from agent_harness.providers.base import (
    ChildLaunchGate,
    ProviderAdapter,
    ProviderEvent,
)


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


def _durable_child_gate(tmp_path: Path, limit: int) -> ChildLaunchGate:
    database = tmp_path / "child-gate.sqlite3"
    command_id = "00000000-0000-4000-8000-000000000001"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS child_launch_gates (
                command_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                permit_limit INTEGER NOT NULL,
                consumed INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS child_launch_admissions (
                command_id TEXT NOT NULL,
                admission_key TEXT NOT NULL,
                admitted INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(command_id, admission_key)
            )
            """
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO child_launch_gates VALUES (
                ?, 'session', ?, 0, 'created', 'updated'
            )
            """,
            (command_id, limit),
        )
        connection.commit()
    finally:
        connection.close()
    return ChildLaunchGate(database, command_id, limit)


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
    tmp_path: Path,
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
            "npm_config_cache": "/actions-runner/_work/_temp/npm-cache",
            "npm_config_package": "ignored",
            "OTHER_SECRET": "ignored",
            "LC_ACCESS_KEY": "ignored",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CLAUDE_CODE_OAUTH_TOKEN": "claude-oauth",
            "ANTHROPIC_API_KEY": "anthropic",
            "OPENAI_API_KEY": "openai",
        },
    )
    monkeypatch.setattr(base, "trusted_provider_path", lambda: "/bin")
    expected_cache = str(
        tmp_path
        / "my"
        / "chats"
        / ".runtime"
        / "provider-cache"
        / "npm"
    )
    monkeypatch.setattr(
        base,
        "provider_npm_cache",
        lambda: Path(expected_cache),
    )
    assert codex.provider_environment("codex") == {
        "PATH": "/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "npm_config_cache": expected_cache,
    }
    assert codex.provider_environment("claude") == {
        "PATH": "/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CLAUDE_CODE_OAUTH_TOKEN": "claude-oauth",
        "npm_config_cache": expected_cache,
    }
    assert codex.provider_environment("codex", "api") == {
        "PATH": "/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENAI_API_KEY": "openai",
        "npm_config_cache": expected_cache,
    }
    assert codex.provider_environment("claude", "api") == {
        "PATH": "/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ANTHROPIC_API_KEY": "anthropic",
        "npm_config_cache": expected_cache,
    }

    monkeypatch.setenv("HOME", str(tmp_path))
    cache = base.provider_npm_cache()
    assert cache == (
        tmp_path
        / "my"
        / "chats"
        / ".runtime"
        / "provider-cache"
        / "npm"
    )
    assert not cache.exists()


def test_trusted_provider_executable_ignores_hostile_path_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "npx"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    searches: list[str] = []

    def find_executable(name: str, *, path: str) -> str:
        assert name == "npx"
        searches.append(path)
        return str(executable)

    monkeypatch.setattr(base.shutil, "which", find_executable)
    monkeypatch.setenv("PATH", str(tmp_path / "hostile"))
    assert base.trusted_executable("npx") == str(executable)
    assert searches == [base._TRUSTED_EXECUTABLE_SEARCH]

    executable.chmod(0o775)
    with pytest.raises(RuntimeError, match="writable"):
        base.trusted_executable("npx")

    executable.chmod(0o644)
    with pytest.raises(RuntimeError, match="not runnable"):
        base.trusted_executable("npx")

    monkeypatch.setattr(base.shutil, "which", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="not found"):
        base.trusted_executable("npx")


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
    assert codex._strings(["fast", 1, {"id": "flex"}, {"id": 2}]) == ("fast", "flex")

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
            await server._route({"id": payload["id"], "result": {"answer": 7}})

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
        await server._route({"id": 11, "error": {"message": "quota reached"}})
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
        await server._route({"id": 14, "method": "approval", "params": {"x": 2}})
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
        with pytest.raises(ProviderUnavailableError):
            await server._read_messages()
        failure = pending.exception()
        assert isinstance(failure, ProviderUnavailableError)
        assert failure.provider == "codex"
        assert "Codex app-server connection closed" in failure.detail.message
        await server._read_stderr()
        assert server.stderr_tail == [
            "last stderr",
            "first",
            "�second",
        ]

        closing = _server(tmp_path)
        closing.process = Process(  # type: ignore[assignment]
            stdout=Stream([]),
            stderr=Stream([]),
        )
        closing._closing = True
        await closing._read_messages()

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
        identity = SimpleNamespace(pid=42, pgid=42, pid_start="start")

        async def create(*args: Any, **kwargs: Any) -> Process:
            assert Path(str(args[0])).name == "npx"
            assert args[1:3] == ("-y", "@openai/codex@0.146.0")
            assert kwargs["cwd"] == tmp_path
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        monkeypatch.setattr(
            codex,
            "process_group_identity",
            lambda unused: identity,
        )

        async def terminate(
            selected_process: Process,
            selected_identity: object,
            **unused: object,
        ) -> None:
            assert selected_process is process
            assert selected_identity is identity

        monkeypatch.setattr(codex, "terminate_process_group", terminate)
        server = codex.CodexAppServer(
            tmp_path,
            notification_handler=_ignore_notification,
            request_handler=_decline,
            child_launch_gate=_durable_child_gate(tmp_path, 2),
        )
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


def test_codex_server_disables_child_agents_without_a_durable_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = Process(stdout=Stream([]), stderr=Stream([]))
        identity = SimpleNamespace(pid=42, pgid=42, pid_start="start")
        argv: list[str] = []

        async def create(*args: Any, **kwargs: Any) -> Process:
            del kwargs
            argv.extend(str(item) for item in args)
            return process

        async def terminate(*unused: object, **options: object) -> None:
            del options

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        monkeypatch.setattr(
            codex,
            "process_group_identity",
            lambda unused: identity,
        )
        monkeypatch.setattr(codex, "terminate_process_group", terminate)
        server = _server(tmp_path)

        async def request(
            method: str,
            params: dict[str, Any],
            *,
            timeout: float = 30.0,
        ) -> dict[str, Any]:
            del method
            del params
            del timeout
            return {}

        async def notify(method: str, params: dict[str, Any]) -> None:
            del method
            del params

        server.request = request  # type: ignore[method-assign]
        server.notify = notify  # type: ignore[method-assign]
        await server.start()

        assert argv[3:5] == ["-c", "agents.enabled=false"]
        assert server._child_gate_arguments() == ["-c", "agents.enabled=false"]

        server._process_group = None
        with pytest.raises(RuntimeError, match="process group identity is missing"):
            await server.close()
        server._process_group = identity
        await server.close()

    asyncio.run(scenario())


def test_codex_native_sessions_are_recognized_by_their_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex.Path, "home", lambda: tmp_path)

    assert not codex._has_native_session("")
    assert not codex._has_native_session("thread 1")
    assert not codex._has_native_session("thread-1")

    sessions = tmp_path / ".codex" / "sessions" / "2026" / "08"
    sessions.mkdir(parents=True)
    assert not codex._has_native_session("thread-1")

    (sessions / "rollout-thread-1-2026.jsonl").write_text("{}\n", encoding="utf-8")
    assert codex._has_native_session("thread-1")
    assert codex.CodexAdapter().native_session_available(tmp_path, "thread-1")


def test_codex_child_gate_argv_is_durable(tmp_path: Path) -> None:
    durable_gate = _durable_child_gate(tmp_path, 2)
    server = codex.CodexAppServer(
        tmp_path,
        notification_handler=_ignore_notification,
        request_handler=_decline,
        child_launch_gate=durable_gate,
    )
    arguments = server._child_gate_arguments()
    assert arguments[0] == "--dangerously-bypass-hook-trust"
    assert "agents.max_concurrent_threads_per_session=2" in arguments
    hook = next(value for value in arguments if value.startswith("hooks.PreToolUse="))
    assert "^(Agent|spawn_agent)$" in hook
    assert "child_gate.py" in hook
    assert " -B " in hook
    assert hook.index(" -B ") < hook.index("child_gate.py")
    assert str(durable_gate.database) in hook
    assert durable_gate.command_id in hook
    asyncio.run(server.close())
    assert durable_gate.database.exists()

    inactive = codex.CodexAppServer(
        tmp_path,
        notification_handler=_ignore_notification,
        request_handler=_decline,
        child_launch_gate=_durable_child_gate(tmp_path, 0),
    )
    assert "agents.enabled=false" in inactive._child_gate_arguments()
    asyncio.run(inactive.close())


def test_child_gate_is_atomic_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _durable_child_gate(tmp_path, 2)
    assert child_gate.admit(gate.database, gate.command_id, 2, "claude:Agent:one")
    assert child_gate.admit(gate.database, gate.command_id, 2, "claude:Agent:one")
    assert child_gate.admit(gate.database, gate.command_id, 2, "claude:Agent:two")
    assert not child_gate.admit(
        gate.database,
        gate.command_id,
        2,
        "claude:Agent:three",
    )
    assert not child_gate.admit(
        gate.database,
        gate.command_id,
        3,
        "claude:Agent:four",
    )
    assert not child_gate.admit(
        gate.database,
        "missing",
        2,
        "claude:Agent:five",
    )
    assert not child_gate.admit(gate.database, gate.command_id, 2, "")

    concurrent_gate = _durable_child_gate(tmp_path, 2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(
            pool.map(
                lambda unused: child_gate.admit(
                    concurrent_gate.database,
                    concurrent_gate.command_id,
                    concurrent_gate.limit,
                    "codex:Agent:" + str(unused),
                ),
                range(20),
            )
        )
    assert sum(admitted) == 2
    child_gate._rollback(None)
    child_gate._rollback(SimpleNamespace(in_transaction=False))  # type: ignore[arg-type]

    class BrokenRollback:
        in_transaction = True

        def execute(self, statement: str) -> None:
            del statement
            raise sqlite3.OperationalError("rollback failed")

    child_gate._rollback(BrokenRollback())  # type: ignore[arg-type]

    monkeypatch.setattr(
        child_gate.sys,
        "stdin",
        io.StringIO('{"tool_name":"Bash"}'),
    )
    assert child_gate.main([str(gate.database), gate.command_id, "2", "codex"]) == 0
    assert child_gate.main([]) == 0
    assert '"permissionDecision":"deny"' in capsys.readouterr().out
    monkeypatch.setattr(
        child_gate.sys,
        "stdin",
        io.StringIO('{"tool_name":"Agent"}'),
    )
    assert child_gate.main([str(gate.database), gate.command_id, "2", "codex"]) == 0
    assert '"permissionDecision":"deny"' in capsys.readouterr().out

    for payload, limit in (("not-json", "2"), ("[]", "2"), ("{}", "invalid")):
        monkeypatch.setattr(child_gate.sys, "stdin", io.StringIO(payload))
        assert (
            child_gate.main([str(gate.database), gate.command_id, limit, "codex"]) == 0
        )
        assert '"permissionDecision":"deny"' in capsys.readouterr().out
    unwritable = tmp_path / "directory-state"
    unwritable.mkdir()
    monkeypatch.setattr(
        child_gate.sys,
        "stdin",
        io.StringIO('{"tool_name":"Agent"}'),
    )
    assert child_gate.main([str(unwritable), gate.command_id, "2", "codex"]) == 0

    # An unopenable database has to deny rather than propagate, so the
    # payload here carries a tool id and reaches the durable gate.
    monkeypatch.setattr(
        child_gate.sys,
        "stdin",
        io.StringIO('{"tool_name":"Agent","tool_use_id":"call-1"}'),
    )
    assert child_gate.main([str(unwritable), gate.command_id, "2", "codex"]) == 0
    assert '"permissionDecision":"deny"' in capsys.readouterr().out

    gate = _durable_child_gate(tmp_path, 2)
    monkeypatch.setattr(
        child_gate.sys,
        "argv",
        ["child_gate.py", str(gate.database), gate.command_id, "2", "codex"],
    )
    monkeypatch.setattr(
        child_gate.sys,
        "stdin",
        io.StringIO('{"tool_name":"spawn_agent","tool_use_id":"spawn-one"}'),
    )
    assert child_gate.main() == 0
    monkeypatch.setattr(
        child_gate.sys,
        "stdin",
        io.StringIO('{"tool_name":"spawn_agent","tool_use_id":"spawn-one"}'),
    )
    assert child_gate.main() == 0
    assert child_gate.admit(
        gate.database,
        gate.command_id,
        2,
        "codex:spawn_agent:spawn-one",
    )
    assert child_gate.admit(
        gate.database,
        gate.command_id,
        2,
        "codex:spawn_agent:spawn-two",
    )
    assert not child_gate.admit(
        gate.database,
        gate.command_id,
        2,
        "codex:spawn_agent:spawn-three",
    )

    monkeypatch.setattr(
        child_gate.sys,
        "stdin",
        io.StringIO('{"tool_name":"Bash"}'),
    )
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(Path(child_gate.__file__)), run_name="__main__")
    assert exited.value.code == 0


# A permit update that matches no row means the durable gate moved under
# the transaction, so the launch must be denied rather than admitted on
# the strength of the earlier read.
def test_child_gate_denies_when_the_permit_update_matches_no_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _durable_child_gate(tmp_path, 2)
    real_connect = child_gate.sqlite3.connect

    class LosingUpdate:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        @property
        def in_transaction(self) -> bool:
            return self._inner.in_transaction

        def execute(self, statement: str, *arguments: object) -> object:
            cursor = self._inner.execute(statement, *arguments)
            if "UPDATE child_launch_gates" in statement:
                return SimpleNamespace(rowcount=0)
            return cursor

        def close(self) -> None:
            self._inner.close()

    monkeypatch.setattr(
        child_gate.sqlite3,
        "connect",
        lambda *arguments, **options: LosingUpdate(
            real_connect(*arguments, **options)
        ),
    )

    assert not child_gate.admit(
        gate.database,
        gate.command_id,
        2,
        "claude:Agent:lost",
    )

    monkeypatch.undo()
    connection = real_connect(gate.database)
    try:
        consumed = connection.execute(
            "SELECT consumed FROM child_launch_gates WHERE command_id = ?",
            (gate.command_id,),
        ).fetchone()
    finally:
        connection.close()
    assert int(consumed[0]) == 0


def test_claude_child_gate_denies_third_before_start(tmp_path: Path) -> None:
    async def scenario() -> None:
        durable_gate = _durable_child_gate(tmp_path, 2)
        gate = claude._child_gate(durable_gate)
        inputs = [
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Agent",
                "tool_input": {},
                "tool_use_id": "tool-" + str(index),
            }
            for index in range(3)
        ]
        results = []
        for value in inputs:
            results.append(await gate(value, None, {}))
        allowed = [value for value in results if not value.get("hookSpecificOutput")]
        starts = [ProviderEvent("agent.child.started") for unused in allowed]
        assert len(starts) == 2
        assert results[2]["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert await gate(inputs[0], None, {}) == {}
        missing_identity = dict(inputs[0])
        missing_identity.pop("tool_use_id")
        assert (await gate(missing_identity, None, {}))["hookSpecificOutput"][
            "permissionDecision"
        ] == "deny"
        assert (
            await gate(
                {**inputs[0], "tool_name": "Bash"},
                None,
                {},
            )
            == {}
        )
        task = {**inputs[0], "tool_name": "Task", "tool_use_id": "task-tool"}
        assert (await gate(task, None, {}))["hookSpecificOutput"][
            "permissionDecision"
        ] == "deny"

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
        identity = SimpleNamespace(pid=42, pgid=42, pid_start="start")
        server._process_group = identity  # type: ignore[assignment]

        async def broken_route(payload: dict[str, Any]) -> None:
            del payload
            raise RuntimeError("reader failed")

        server._route = broken_route  # type: ignore[method-assign]
        pending = asyncio.get_running_loop().create_future()
        server._pending[1] = pending
        with pytest.raises(RuntimeError, match="reader failed"):
            await server._read_messages()
        assert isinstance(pending.exception(), RuntimeError)

        async def terminate(
            selected_process: Process,
            selected_identity: object,
            **unused: object,
        ) -> None:
            assert selected_process is process
            assert selected_identity is identity
            process.terminated = True
            process.killed = True

        monkeypatch.setattr(codex, "terminate_process_group", terminate)
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
            child_launch_gate: ChildLaunchGate | None = None,
        ) -> None:
            self.workspace = workspace
            self.notification_handler = notification_handler
            self.request_handler = request_handler
            self.child_launch_gate = child_launch_gate
            self.process = SimpleNamespace(pid=44, returncode=None)
            self._reader: asyncio.Task[None] | None = None
            self.closed = False
            self.instances.append(self)

        async def start(self) -> None:
            self._reader = asyncio.create_task(asyncio.sleep(3600))
            if self.notification_handler.__name__ == ("ignore_notification"):
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
            if self._reader is not None:
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)

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
        assert len(events) == 4
        assert events[-1].event_type == "provider.prompt.accepted"
        assert adapter.process_identity() == (0, "")
        await adapter.interrupt()
        with pytest.raises(ProviderUnavailableError):
            await adapter.steer("more")

        active = Server(
            tmp_path,
            notification_handler=_ignore_notification,
            request_handler=_decline,
        )
        active._process_group = SimpleNamespace(
            pid=44,
            pid_start="canonical-start",
        )
        adapter._active_server = active  # type: ignore[assignment]
        adapter._active_thread = "thread"
        adapter._active_turn = "turn"
        await adapter.interrupt()
        await adapter.steer("more")
        assert adapter.process_identity()[0] == 44

    asyncio.run(scenario())


class _StubCodexServer:
    """Codex app server double whose turn never reports completion."""

    def __init__(
        self,
        workspace: Path,
        *,
        notification_handler: Any,
        request_handler: Any,
        child_launch_gate: ChildLaunchGate | None = None,
    ) -> None:
        del notification_handler
        del request_handler
        del child_launch_gate
        self.workspace = workspace
        self.process = SimpleNamespace(pid=44, returncode=None)
        self._reader: asyncio.Task[None] | None = None
        self.turn_id = "turn-1"
        self.reader_factory: Any = None

    async def start(self) -> None:
        if self.reader_factory is not None:
            self._reader = asyncio.create_task(self.reader_factory())
            return
        self._reader = asyncio.create_task(asyncio.sleep(3600))

    async def request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        del params
        if method in {"thread/start", "thread/resume"}:
            return {"thread": {"id": "thread-1"}}
        if method == "turn/start":
            if not self.turn_id:
                return {}
            return {"turn": {"id": self.turn_id}}
        return {}

    async def close(self) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)


def test_codex_turn_fails_closed_without_a_turn_or_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers: list[_StubCodexServer] = []

    def build(*args: Any, **kwargs: Any) -> _StubCodexServer:
        server = _StubCodexServer(*args, **kwargs)
        servers.append(server)
        return server

    monkeypatch.setattr(codex, "CodexAppServer", build)
    monkeypatch.setattr(codex, "codex_notification", lambda method, params: [])

    async def collect(event: ProviderEvent) -> None:
        del event

    async def turn(**overrides: Any) -> Any:
        adapter = codex.CodexAdapter()
        return await adapter.run_turn(
            workspace=tmp_path,
            prompt="work",
            native_session_id="",
            permission_mode="approval",
            model="",
            effort="",
            event_handler=collect,
            approval_handler=_decline,
            **overrides,
        )

    async def scenario() -> None:
        opened: list[str] = []

        async def gate() -> None:
            opened.append("opened")
            servers[-1].turn_id = ""

        with pytest.raises(HarnessError, match="turn identifier"):
            await turn(pre_prompt_gate=gate)
        assert opened == ["opened"]

        async def missing_reader() -> None:
            servers[-1]._reader = None

        with pytest.raises(RuntimeError, match="reader task is missing"):
            await turn(pre_prompt_gate=missing_reader)

        async def failing_reader() -> None:
            raise ProviderUnavailableError("codex", detail="transport closed")

        async def use_failing_reader() -> None:
            servers[-1]._reader = asyncio.create_task(failing_reader())

        with pytest.raises(ProviderUnavailableError):
            await turn(pre_prompt_gate=use_failing_reader)

        async def never_finishes(*unused: object, **options: object) -> Any:
            del options
            return set(), set()

        monkeypatch.setattr(codex.asyncio, "wait", never_finishes)
        with pytest.raises(TimeoutError, match="turn timed out"):
            await turn()

    asyncio.run(scenario())


def test_codex_process_identity_ignores_an_exited_server(tmp_path: Path) -> None:
    adapter = codex.CodexAdapter()
    adapter._active_server = SimpleNamespace(  # type: ignore[assignment]
        process=SimpleNamespace(pid=44, returncode=0)
    )
    del tmp_path

    assert adapter.process_identity() == (0, "")


def test_codex_adapter_status_resume_failure_and_interrupt_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(unused_name: str) -> str:
        raise RuntimeError("not found")

    monkeypatch.setattr(codex, "trusted_executable", unavailable)
    assert not codex.CodexAdapter().status().ready
    monkeypatch.setattr(
        codex,
        "trusted_executable",
        lambda name: "/usr/bin/" + name,
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
            child_launch_gate: ChildLaunchGate | None = None,
        ) -> None:
            del workspace
            self.notification_handler = notification_handler
            self.request_handler = request_handler
            del child_launch_gate
            self.process = None
            self._reader: asyncio.Task[None] | None = None

        async def start(self) -> None:
            self._reader = asyncio.create_task(asyncio.sleep(3600))

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
            if self._reader is not None:
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)

    monkeypatch.setattr(codex, "CodexAppServer", Server)
    monkeypatch.setattr(
        codex,
        "codex_notification",
        lambda method, params: [ProviderEvent("provider.error", status="failed")],
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
    def unavailable(unused_name: str) -> str:
        raise RuntimeError("not found")

    monkeypatch.setattr(claude, "trusted_executable", unavailable)
    transport = object.__new__(claude.NpxClaudeTransport)
    with pytest.raises(claude.CLINotFoundError):
        transport._find_cli()
    monkeypatch.setattr(
        claude,
        "trusted_executable",
        lambda name: "/usr/bin/" + name,
    )
    assert transport._find_cli() == "/usr/bin/npx"
    asyncio.run(transport._check_claude_version())

    monkeypatch.setattr(
        claude.SubprocessCLITransport,
        "_build_command",
        lambda unused: ["/usr/bin/npx", "--output-format", "stream-json"],
    )
    monkeypatch.setattr(
        claude,
        "provider_npm_cache",
        lambda: tmp_path / "npm-cache",
    )
    assert transport._build_command() == [
        sys.executable,
        "-I",
        "-B",
        "-S",
        "-c",
        "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])",
        "/usr/bin/npx",
        "--cache",
        str(tmp_path / "npm-cache"),
        "@anthropic-ai/claude-code@2.1.220",
        "--output-format",
        "stream-json",
    ]

    assert claude._permission_mode("full") == "bypassPermissions"
    assert claude._permission_mode("plan") == "plan"
    assert claude._permission_mode("read-only") == "plan"
    assert claude._permission_mode("approval") == "default"
    with pytest.raises(ValueError):
        claude._permission_mode("invalid")

    assert (
        claude._message_session_id(SimpleNamespace(session_id="session")) == "session"
    )
    assert claude._message_session_id(object()) == ""
    assert claude._content_payload(object()) == {"type": "unknown"}
    assert claude._message_events(object())[0].event_type == "provider.event"
    child_ids: set[str] = set()
    started = claude._normalized_child_event(
        ProviderEvent(
            "agent.child.started",
            metadata={"id": "child-1"},
        ),
        child_ids,
    )
    assert started.event_type == "agent.child.started"
    assert child_ids == {"child-1"}
    bash_task = claude._normalized_child_event(
        ProviderEvent(
            "agent.child.started",
            metadata={
                "child_id": "bash-process",
                "tool_use_id": "bash-tool",
            },
        ),
        child_ids,
    )
    assert bash_task.event_type == "tool.progress"
    assert child_ids == {"child-1"}
    bash_task_completed = claude._normalized_child_event(
        ProviderEvent(
            "agent.child.completed",
            metadata={
                "child_id": "bash-process",
                "tool_use_id": "bash-tool",
            },
        ),
        child_ids,
    )
    assert bash_task_completed.event_type == "tool.completed"
    assert child_ids == {"child-1"}
    unrelated = ProviderEvent(
        "tool.completed",
        metadata={"tool_use_id": "other"},
    )
    assert (
        claude._normalized_child_event(
            unrelated,
            child_ids,
        )
        is unrelated
    )
    completed = claude._normalized_child_event(
        ProviderEvent(
            "tool.completed",
            metadata={"tool_use_id": "child-1"},
        ),
        child_ids,
    )
    assert completed.event_type == "agent.child.completed"
    assert not child_ids
    child_ids.add("child-2")
    failed = claude._normalized_child_event(
        ProviderEvent(
            "tool.completed",
            metadata={"tool_use_id": "child-2", "is_error": True},
        ),
        child_ids,
    )
    assert failed.event_type == "agent.child.failed"
    assert failed.status == "failed"
    assert not child_ids
    no_metadata = ProviderEvent("provider.event")
    assert (
        claude._normalized_child_event(
            no_metadata,
            child_ids,
        )
        is no_metadata
    )
    assert claude._looks_exhausted("Capacity reached")
    assert not claude._looks_exhausted("healthy")
    assert claude._looks_like_spend_limit("Monthly spend limit; use /usage-credits")
    assert not claude._looks_like_spend_limit("monthly spend limit")
    assert claude._bounded("x" * 5000) == "x" * 4096
    assert not claude._has_native_session(tmp_path, "missing")

    async def prompt() -> None:
        values = [value async for value in claude._prompt_stream("hello", "session")]
        assert values[0]["message"]["content"][0]["text"] == "hello"
        assert [value async for value in claude._empty_stream()] == []

    asyncio.run(prompt())


def test_claude_bash_exit_codes_are_structured_only_when_proven() -> None:
    bash_tool_ids: set[str] = set()
    without_metadata = ProviderEvent("tool.started")
    assert (
        claude._normalized_bash_event(without_metadata, bash_tool_ids)
        is without_metadata
    )
    read = ProviderEvent(
        "tool.started",
        metadata={"id": "read-1", "name": "Read"},
    )
    assert claude._normalized_bash_event(read, bash_tool_ids) is read
    missing_id = ProviderEvent(
        "tool.started",
        metadata={"name": "Bash"},
    )
    assert claude._normalized_bash_event(missing_id, bash_tool_ids) is missing_id
    started = ProviderEvent(
        "tool.started",
        metadata={"id": "bash-1", "name": "Bash"},
    )
    assert claude._normalized_bash_event(started, bash_tool_ids) is started
    progress = ProviderEvent("tool.progress", metadata={"id": "bash-1"})
    assert claude._normalized_bash_event(progress, bash_tool_ids) is progress
    unrelated = ProviderEvent(
        "tool.completed",
        metadata={"tool_use_id": "other", "is_error": False},
    )
    assert claude._normalized_bash_event(unrelated, bash_tool_ids) is unrelated
    succeeded = claude._normalized_bash_event(
        ProviderEvent(
            "tool.completed",
            text="tests passed",
            metadata={"tool_use_id": "bash-1", "is_error": False},
        ),
        bash_tool_ids,
    )
    assert succeeded.metadata == {
        "exit_code": 0,
        "is_error": False,
        "tool_use_id": "bash-1",
    }
    assert not bash_tool_ids

    for tool_id, text, expected in (
        ("bash-2", "Exit code 124\nbuild timed out", 124),
        ("bash-3", "Exit code 7", 7),
    ):
        claude._normalized_bash_event(
            ProviderEvent(
                "tool.started",
                metadata={"id": tool_id, "name": "bash"},
            ),
            bash_tool_ids,
        )
        failed = claude._normalized_bash_event(
            ProviderEvent(
                "tool.completed",
                text=text,
                metadata={"tool_use_id": tool_id, "is_error": True},
            ),
            bash_tool_ids,
        )
        assert failed.metadata is not None
        assert failed.metadata["exit_code"] == expected

    claude._normalized_bash_event(
        ProviderEvent(
            "tool.started",
            metadata={"id": "bash-4", "name": "Bash"},
        ),
        bash_tool_ids,
    )
    unproved = claude._normalized_bash_event(
        ProviderEvent(
            "tool.completed",
            text="command failed without an explicit process status",
            metadata={"tool_use_id": "bash-4", "is_error": True},
        ),
        bash_tool_ids,
    )
    assert unproved.metadata is not None
    assert "exit_code" not in unproved.metadata


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
    monkeypatch.setattr(
        claude,
        "_has_native_session",
        lambda unused_workspace, session_id: session_id == "native",
    )
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


def test_claude_resume_fails_when_transcript_is_missing_or_misbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    other_workspace = tmp_path / "other-workspace"
    workspace.mkdir()
    other_workspace.mkdir()
    monkeypatch.setattr(
        claude.Path,
        "home",
        classmethod(lambda unused_class: home),
    )
    transcript = claude._native_session_path(workspace, "native")
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")
    assert claude._has_native_session(workspace, "native")
    assert not claude._has_native_session(other_workspace, "native")
    transcript.unlink()
    assert not claude._has_native_session(workspace, "native")

    class NeverClient:
        def __init__(self, **unused: Any) -> None:
            raise AssertionError("a missing resume must not start Claude")

    monkeypatch.setattr(claude, "ClaudeSDKClient", NeverClient)
    events: list[ProviderEvent] = []

    async def collect(event: ProviderEvent) -> None:
        events.append(event)

    async def scenario() -> None:
        adapter = claude.ClaudeAdapter()
        with pytest.raises(ProviderUnavailableError):
            await adapter.run_turn(
                workspace=workspace,
                prompt="resume",
                native_session_id="native",
                permission_mode="plan",
                model="",
                effort="",
                event_handler=collect,
                approval_handler=_decline,
            )

    asyncio.run(scenario())
    assert [event.event_type for event in events] == ["recovery.native_resume_missing"]
    assert events[0].status == "failed"
    assert events[0].metadata == {
        "provider": "claude",
        "native_session_id": "native",
    }


def test_claude_adapter_status_models_and_active_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(unused_name: str) -> str:
        raise RuntimeError("not found")

    adapter = claude.ClaudeAdapter()
    monkeypatch.setattr(
        claude,
        "trusted_executable",
        lambda name: "/usr/bin/" + name,
    )
    assert adapter.status().ready
    monkeypatch.setattr(claude, "trusted_executable", unavailable)
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
    adapter._transport = SimpleNamespace(
        _process=SimpleNamespace(pid=0, returncode=None)
    )
    assert adapter.process_identity() == (0, "")
    adapter._transport = SimpleNamespace(
        _process=SimpleNamespace(pid=55, returncode=None),
        _process_group=SimpleNamespace(pid=55, pid_start="started"),
    )
    assert adapter.process_identity() == (55, "started")
    adapter._transport = SimpleNamespace(
        _process=SimpleNamespace(pid=55, returncode=0)
    )
    assert adapter.process_identity() == (0, "")
    monkeypatch.setattr(claude, "_has_native_session", lambda *unused: True)
    assert adapter.native_session_available(tmp_path, "native")


# Without a durable gate there is no way to bound child launches, so the
# hook denies instead of falling through to the provider default.
def test_claude_child_gate_denies_without_a_durable_gate() -> None:
    async def scenario() -> None:
        gate = claude._child_gate(None)
        decision = await gate(
            {"tool_name": "Agent", "tool_use_id": "tool-1"},
            "tool-1",
            {},
        )
        assert decision["hookSpecificOutput"]["permissionDecisionReason"] == (
            "The harness child-agent launch gate is unavailable."
        )

    asyncio.run(scenario())


def test_claude_transport_isolates_and_reaps_its_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_calls: list[str] = []

    async def base_connect(unused_self: object) -> None:
        base_calls.append("connect")

    async def base_close(unused_self: object) -> None:
        base_calls.append("close")

    monkeypatch.setattr(
        claude.SubprocessCLITransport,
        "connect",
        base_connect,
        raising=False,
    )
    monkeypatch.setattr(
        claude.SubprocessCLITransport,
        "close",
        base_close,
        raising=False,
    )
    transport = object.__new__(claude.NpxClaudeTransport)
    transport._process_group = None

    async def missing_process() -> None:
        with pytest.raises(RuntimeError, match="process is missing"):
            await transport.connect()

    asyncio.run(missing_process())

    transport._process = SimpleNamespace(pid=4242)
    attempts = {"count": 0}

    def slow_identity(pid: int) -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("process group is not isolated yet")
        return "group-" + str(pid)

    monkeypatch.setattr(claude, "process_group_identity", slow_identity)
    asyncio.run(transport.connect())
    assert transport._process_group == "group-4242"
    assert attempts["count"] == 3

    def never_isolated(unused_pid: int) -> str:
        raise RuntimeError("process group is not isolated")

    monkeypatch.setattr(claude, "process_group_identity", never_isolated)

    async def never_ready() -> None:
        with pytest.raises(RuntimeError, match="did not isolate"):
            await transport.connect()

    asyncio.run(never_ready())

    terminated: list[object] = []

    async def terminate(process: object, identity: object, **options: object) -> None:
        del options
        terminated.append((process, identity))

    monkeypatch.setattr(claude, "terminate_process_group", terminate)
    transport._process_group = "group-4242"
    asyncio.run(transport.close())
    assert terminated == [(transport._process, "group-4242")]
    assert transport._process_group is None

    asyncio.run(transport.close())
    assert len(terminated) == 1
    assert base_calls == ["connect", "connect", "connect", "close", "close"]


def test_claude_prompt_stream_awaits_its_pre_prompt_gate() -> None:
    opened: list[str] = []

    async def gate() -> None:
        opened.append("opened")

    async def scenario() -> None:
        values = [
            value
            async for value in claude._prompt_stream(
                "hello",
                "session",
                pre_prompt_gate=gate,
            )
        ]
        assert values[0]["message"]["content"][0]["text"] == "hello"

    asyncio.run(scenario())

    assert opened == ["opened"]


def test_terminal_command_text_and_pty_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_command = terminal._command("codex", "approval", ["resume"])
    assert Path(codex_command[0]).name == "npx"
    assert codex_command[1:] == [
        "-y",
        "@openai/codex@0.146.0",
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "on-request",
        "resume",
    ]
    assert "--yolo" in terminal._command("codex", "full", [])
    assert "--dangerously-skip-permissions" in terminal._command(
        "claude",
        "full",
        [],
    )
    claude_command = terminal._command("claude", "approval", [])
    assert Path(claude_command[0]).name == "npx"
    assert claude_command[1] == "@anthropic-ai/claude-code@2.1.220"
    assert terminal._command("codex", "read-only", ["resume"])[3:] == [
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "on-request",
        "resume",
    ]
    assert terminal._command("claude", "plan", [])[-2:] == [
        "--permission-mode",
        "plan",
    ]
    for protected in (
        "--yolo",
        "--sandbox=danger-full-access",
        "--ask-for-approval=never",
        "--dangerously-skip-permissions",
        "--permission-mode=bypassPermissions",
        "-a",
        "-s",
    ):
        with pytest.raises(ValueError, match="override"):
            terminal._command("codex", "approval", [protected])
    for provider in ("codex", "claude"):
        with pytest.raises(ValueError, match="permission mode"):
            terminal._command(provider, "unsupported", [])
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
        pid = 42

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
        assert Path(str(args[0])).name == "npx"
        assert args[1:3] == ("-y", "@openai/codex@0.146.0")
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
    identity = SimpleNamespace(pid=42, pgid=42, pid_start="start")
    monkeypatch.setattr(
        terminal,
        "process_group_identity",
        lambda unused: identity,
    )
    terminated: list[tuple[object, object]] = []

    async def terminate(
        selected_process: object,
        selected_identity: object,
    ) -> None:
        terminated.append((selected_process, selected_identity))

    monkeypatch.setattr(terminal, "terminate_process_group", terminate)
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
    assert terminated == [(child, identity)]
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
    assert "child agents contained by host config denies" in status.detail
    # A one-shot run has no live turn to steer and cannot prompt for
    # approval, so neither capability is claimed.
    assert "approval" not in status.capabilities
    assert "streaming" in status.capabilities
    assert "tools" in status.capabilities
    assert "subagents" not in status.capabilities

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
        b'{"role":"assistant","content":"","tool_calls":[{"id":"call_1",'
        b'"type":"function","function":{"name":"Read",'
        b'"arguments":"{\\"file_path\\":\\"README.md\\"}"}}]}\n',
        b'{"role":"tool","tool_call_id":"call_1","content":"file contents"}\n',
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
            permission_mode="full",
            model="kimi-code/k3",
            effort="",
            event_handler=handler,
            approval_handler=approvals,
        )
        assert result.status == "complete"
        # The session id arrives on the trailing meta line, not up front.
        assert result.native_session_id == "session_z"

    asyncio.run(scenario())

    assert [e.event_type for e in seen] == [
        "provider.prompt.accepted",
        "agent.message",
        "tool.started",
        "tool.completed",
        "turn.completed",
    ]


def test_kimi_acknowledges_prompt_acceptance_only_on_turn_output(
    monkeypatch,
    tmp_path,
) -> None:
    parsed = [
        b"not json\n",
        b"[1, 2]\n",
        b"\n",
        b'{"role":"assistant","content":"first"}\n',
        b'{"role":"assistant","content":"second"}\n',
        b'{"role":"meta","type":"session.resume_hint","session_id":"session_z"}\n',
    ]
    unparsable = [b"not json\n", b"[1, 2]\n"]
    no_turn = [
        b'{"role":"meta","type":"session.started","session_id":"session_a"}\n',
        b'{"role":"assistant","content":""}\n',
        b'{"role":"error","content":"config.invalid"}\n',
    ]
    resume_only = [
        b'{"role":"meta","type":"session.resume_hint","session_id":"session_z"}\n',
    ]
    tools_only = [
        b'{"role":"assistant","content":"","tool_calls":[{"id":"call_1",'
        b'"type":"function","function":{"name":"Read",'
        b'"arguments":"{\\"file_path\\":\\"README.md\\"}"}}]}\n',
    ]

    def process_for(lines: list[bytes]):
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

        return Process()

    pending = [parsed, unparsable, no_turn, resume_only, tools_only]

    async def fake_exec(*_argv, **_kwargs):
        return process_for(pending.pop(0))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    seen: list[ProviderEvent] = []

    async def handler(event: ProviderEvent) -> None:
        seen.append(event)

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def run() -> None:
        await kimi.KimiAdapter().run_turn(
            workspace=tmp_path,
            prompt="do it",
            native_session_id="",
            permission_mode="full",
            model="kimi-code/k3",
            effort="",
            event_handler=handler,
            approval_handler=approvals,
        )

    asyncio.run(run())

    # Acceptance is emitted once, ahead of the first normalized event,
    # and is not repeated on later lines. Without it the worker treats
    # every Kimi guard stop as pre-acceptance and may resend the
    # instruction to another provider after Kimi already ran tools.
    accepted = [e for e in seen if e.event_type == "provider.prompt.accepted"]
    assert len(accepted) == 1
    assert accepted[0].status == "accepted"
    assert [e.event_type for e in seen] == [
        "provider.prompt.accepted",
        "agent.message",
        "agent.message",
        "turn.completed",
    ]

    seen.clear()
    asyncio.run(run())

    # A stream that never parses never acknowledges: exec success alone
    # is not proof Kimi consumed the prompt.
    assert seen == []

    seen.clear()
    asyncio.run(run())

    # Neither is a startup meta line, an empty message frame, nor a
    # line reporting a failure. Acknowledging any of those would mark a
    # turn that never ran as delivered and strand the command with an
    # operator instead of failing it over.
    assert [e.event_type for e in seen] == [
        "provider.event",
        "provider.event",
        "provider.event",
    ]

    seen.clear()
    asyncio.run(run())

    # The closing resume hint only a run session emits does acknowledge,
    # so a turn that produced no message is still not mistaken for one
    # that never started.
    assert [e.event_type for e in seen] == [
        "provider.prompt.accepted",
        "turn.completed",
    ]

    seen.clear()
    asyncio.run(run())

    # Kimi announces a tool call on an assistant line whose content is
    # empty, so the frame that carries no text still acknowledges. A
    # turn stopped by a guard right after running a tool must not be
    # resent to another provider.
    assert [e.event_type for e in seen] == [
        "provider.prompt.accepted",
        "tool.started",
    ]


def test_kimi_run_turn_gates_on_a_positive_child_limit(
    monkeypatch,
    tmp_path,
) -> None:
    lines = [b'{"role":"assistant","content":"ok"}\n']

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

    async def handler(_event: ProviderEvent) -> None:
        return None

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def scenario() -> None:
        adapter = kimi.KimiAdapter()
        # A zero limit is accepted: containment comes from the host
        # config's Agent/AgentSwarm deny rules, not the harness gate.
        result = await adapter.run_turn(
            workspace=tmp_path,
            prompt="do it",
            native_session_id="",
            permission_mode="full",
            model="kimi-code/k3",
            effort="",
            event_handler=handler,
            approval_handler=approvals,
            child_launch_gate=ChildLaunchGate(
                database=tmp_path,
                command_id="cmd",
                limit=0,
            ),
        )
        assert result.status == "complete"
        with pytest.raises(RuntimeError, match="positive child-agent limit"):
            await adapter.run_turn(
                workspace=tmp_path,
                prompt="do it",
                native_session_id="",
                permission_mode="full",
                model="kimi-code/k3",
                effort="",
                event_handler=handler,
                approval_handler=approvals,
                child_launch_gate=ChildLaunchGate(
                    database=tmp_path,
                    command_id="cmd",
                    limit=2,
                ),
            )

    asyncio.run(scenario())


# A nonzero exit is the only failure signal a one-shot run emits, so the
# adapter has to turn it into a terminal event instead of a silent
# "complete".
def test_kimi_run_turn_reports_a_nonzero_exit_as_a_failed_turn(
    monkeypatch,
    tmp_path,
) -> None:
    lines = [b"\n", b"[1, 2]\n"]

    class Stdout:
        def __aiter__(self):
            async def gen():
                for line in lines:
                    yield line

            return gen()

    class Stderr:
        async def read(self) -> bytes:
            return b"kimi: config.invalid\n"

    class Process:
        stdout = Stdout()
        stderr = Stderr()

        async def wait(self) -> int:
            return 2

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
            native_session_id="session_prior",
            permission_mode="full",
            model="kimi-code/k3",
            effort="",
            event_handler=handler,
            approval_handler=approvals,
        )
        assert result.status == "failed"
        assert result.native_session_id == "session_prior"

    asyncio.run(scenario())

    assert [(item.event_type, item.text) for item in seen] == [
        ("turn.failed", "kimi: config.invalid")
    ]


def test_kimi_run_turn_terminates_the_process_when_the_handler_rejects_an_event(
    monkeypatch,
    tmp_path,
) -> None:
    lines = [b'{"role":"assistant","content":"guard trip"}\n']

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
        pid = 42
        stdout = Stdout()
        stderr = Stderr()

        async def wait(self) -> int:
            return 0

    process = Process()
    identity = SimpleNamespace(pid=42, pgid=42, pid_start="start")

    async def fake_exec(*_argv, **_kwargs):
        return process

    terminated: list[tuple[object, object]] = []

    async def terminate(selected_process, selected_identity) -> None:
        terminated.append((selected_process, selected_identity))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        kimi,
        "process_group_identity",
        lambda unused: identity,
    )
    monkeypatch.setattr(kimi, "terminate_process_group", terminate)

    async def handler(_event: ProviderEvent) -> None:
        raise SafetyGuardError("stagnation", "kimi")

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def scenario() -> None:
        adapter = kimi.KimiAdapter()
        with pytest.raises(SafetyGuardError, match="stagnation"):
            await adapter.run_turn(
                workspace=tmp_path,
                prompt="do it",
                native_session_id="",
                permission_mode="full",
                model="kimi-code/k3",
                effort="",
                event_handler=handler,
                approval_handler=approvals,
            )
        assert adapter.process_identity() == (0, "")

    asyncio.run(scenario())

    assert terminated == [(process, identity)]


def test_kimi_run_turn_terminates_detached_children_after_normal_completion(
    monkeypatch,
    tmp_path,
) -> None:
    class Stdout:
        def __aiter__(self):
            async def gen():
                if False:
                    yield b""

            return gen()

    class Stderr:
        async def read(self) -> bytes:
            return b""

    class Process:
        pid = 42
        returncode: int | None = None
        stdout = Stdout()
        stderr = Stderr()

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    process = Process()
    identity = SimpleNamespace(pid=42, pgid=42, pid_start="start")

    async def fake_exec(*_argv, **_kwargs):
        return process

    terminated: list[tuple[object, object]] = []

    async def terminate(selected_process, selected_identity) -> None:
        terminated.append((selected_process, selected_identity))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        kimi,
        "process_group_identity",
        lambda unused: identity,
    )
    monkeypatch.setattr(kimi, "terminate_process_group", terminate)

    async def handler(_event: ProviderEvent) -> None:
        return

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def scenario() -> None:
        adapter = kimi.KimiAdapter()
        result = await adapter.run_turn(
            workspace=tmp_path,
            prompt="do it",
            native_session_id="",
            permission_mode="full",
            model="kimi-code/k3",
            effort="",
            event_handler=handler,
            approval_handler=approvals,
        )
        assert result.status == "complete"
        assert adapter.process_identity() == (0, "")

    asyncio.run(scenario())

    assert terminated == [(process, identity)]


def test_kimi_run_turn_waits_for_cleanup_when_cancelled(
    monkeypatch,
    tmp_path,
) -> None:
    process_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    class Stdout:
        def __aiter__(self):
            async def gen():
                await asyncio.Event().wait()
                if False:
                    yield b""

            return gen()

    class Stderr:
        async def read(self) -> bytes:
            return b""

    class Process:
        pid = 42
        returncode: int | None = None
        stdout = Stdout()
        stderr = Stderr()

        async def wait(self) -> int:
            return 0

    process = Process()
    identity = SimpleNamespace(pid=42, pgid=42, pid_start="start")

    async def fake_exec(*_argv, **_kwargs):
        process_started.set()
        return process

    async def terminate(selected_process, selected_identity) -> None:
        assert selected_process is process
        assert selected_identity is identity
        cleanup_started.set()
        await allow_cleanup.wait()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        kimi,
        "process_group_identity",
        lambda unused: identity,
    )
    monkeypatch.setattr(kimi, "terminate_process_group", terminate)

    async def handler(_event: ProviderEvent) -> None:
        return

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def scenario() -> None:
        adapter = kimi.KimiAdapter()
        run_task = asyncio.create_task(
            adapter.run_turn(
                workspace=tmp_path,
                prompt="do it",
                native_session_id="",
                permission_mode="full",
                model="kimi-code/k3",
                effort="",
                event_handler=handler,
                approval_handler=approvals,
            )
        )
        await process_started.wait()
        run_task.cancel()
        await cleanup_started.wait()
        assert not run_task.done()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert adapter.process_identity() == (0, "")

    asyncio.run(scenario())


def test_grok_payload_normalizes_streaming_json() -> None:
    events = normalize.grok_payload({"type": "text", "data": "ok"})
    assert [(e.event_type, e.text) for e in events] == [("agent.message", "ok")]

    events = normalize.grok_payload(
        {
            "type": "end",
            "stopReason": "end_turn",
            "sessionId": "session_g",
            "usage": {"input_tokens": 1},
        }
    )
    assert events[0].event_type == "turn.completed"
    assert events[0].native_session_id == "session_g"
    assert events[0].metadata is not None
    assert events[0].metadata["usage"]["input_tokens"] == 1

    events = normalize.grok_payload(
        {"type": "error", "message": "not signed in"}
    )
    assert events[0].event_type == "turn.failed"
    assert events[0].text == "not signed in"

    assert (
        normalize.grok_payload({"type": "available_commands", "tools": []})[
            0
        ].event_type
        == "provider.event"
    )


def test_grok_launch_argv_is_headless_streaming_json() -> None:
    argv = grok._launch_argv(
        "/home/pramo/.grok/bin/grok",
        prompt="do it",
        model="grok-4.5",
        effort="medium",
        session_id="session_abc",
    )
    assert argv[0] == "/home/pramo/.grok/bin/grok"
    assert "--always-approve" in argv
    assert "--no-subagents" in argv
    assert argv[argv.index("--output-format") + 1] == "streaming-json"
    assert argv[argv.index("-m") + 1] == "grok-4.5"
    assert argv[argv.index("--reasoning-effort") + 1] == "medium"
    assert argv[argv.index("-r") + 1] == "session_abc"
    assert argv[argv.index("-p") + 1] == "do it"
    assert "-r" not in grok._launch_argv(
        "/bin/grok",
        prompt="x",
        model="",
        effort="",
        session_id="",
    )


def test_grok_status_and_models(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        grok, "resolve_grok_binary", lambda: "/home/pramo/.grok/bin/grok"
    )
    status = grok.GrokAdapter().status()
    assert status.ready
    assert status.provider == "grok"
    assert "auth-readiness" in status.detail
    assert "approval" not in status.capabilities
    assert "streaming" in status.capabilities
    assert "tools" not in status.capabilities

    monkeypatch.setattr(grok, "resolve_grok_binary", lambda: None)
    assert not grok.GrokAdapter().status().ready

    async def scenario() -> None:
        models = await grok.GrokAdapter().models(tmp_path)
        assert models[0].model_id == "grok-4.5"
        assert models[0].default
        assert "medium" in models[0].efforts

    asyncio.run(scenario())


def test_grok_run_turn_streams_and_reports_the_session(
    monkeypatch, tmp_path
) -> None:
    lines = [
        b'{"type":"text","data":"ok"}\n',
        b'{"type":"end","stopReason":"end_turn","sessionId":"session_z"}\n',
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

    captured: list[list[str]] = []
    launch_options: list[dict[str, Any]] = []

    async def fake_exec(*argv, **kwargs):
        captured.append(list(argv))
        launch_options.append(dict(kwargs))
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        grok, "resolve_grok_binary", lambda: "/usr/local/bin/grok"
    )

    seen: list[ProviderEvent] = []

    async def handler(event: ProviderEvent) -> None:
        seen.append(event)

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def scenario() -> None:
        result = await grok.GrokAdapter().run_turn(
            workspace=tmp_path,
            prompt="do it",
            native_session_id="",
            permission_mode="full",
            model="grok-4.5",
            effort="medium",
            event_handler=handler,
            approval_handler=approvals,
        )
        assert result.status == "complete"
        assert result.native_session_id == "session_z"

    asyncio.run(scenario())

    assert [e.event_type for e in seen] == [
        "provider.prompt.accepted",
        "agent.message",
        "turn.completed",
    ]
    assert captured[0][0] == "/usr/local/bin/grok"
    assert "--always-approve" in captured[0]
    assert launch_options[0]["limit"] == 16 * 1024 * 1024


def test_grok_acknowledges_prompt_acceptance_only_on_turn_output(
    monkeypatch,
    tmp_path,
) -> None:
    parsed = [
        b"not json\n",
        b"[1, 2]\n",
        b'{"type":"text","data":"first"}\n',
        b'{"type":"text","data":"second"}\n',
        b'{"type":"end","stopReason":"end_turn","sessionId":"session_z"}\n',
    ]
    unparsable = [b"not json\n", b"[1, 2]\n"]
    no_turn = [
        b'{"type":"session_start","sessionId":"session_a"}\n',
        b'{"type":"error","message":"model not found","sessionId":"session_a"}\n',
    ]
    tools_only = [
        b'{"type":"tool_call","toolName":"Read","toolCallId":"call_1"}\n',
    ]

    def process_for(lines: list[bytes]):
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

        return Process()

    pending = [parsed, unparsable, no_turn, tools_only]

    async def fake_exec(*_argv, **_kwargs):
        return process_for(pending.pop(0))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(grok, "resolve_grok_binary", lambda: "/usr/local/bin/grok")

    seen: list[ProviderEvent] = []

    async def handler(event: ProviderEvent) -> None:
        seen.append(event)

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def run() -> None:
        await grok.GrokAdapter().run_turn(
            workspace=tmp_path,
            prompt="do it",
            native_session_id="",
            permission_mode="full",
            model="grok-4.5",
            effort="medium",
            event_handler=handler,
            approval_handler=approvals,
        )

    asyncio.run(run())

    accepted = [e for e in seen if e.event_type == "provider.prompt.accepted"]
    assert len(accepted) == 1
    assert accepted[0].status == "accepted"
    assert [e.event_type for e in seen] == [
        "provider.prompt.accepted",
        "agent.message",
        "agent.message",
        "turn.completed",
    ]

    seen.clear()
    asyncio.run(run())

    assert seen == []

    seen.clear()
    asyncio.run(run())

    # Startup metadata and the structured error payload are both
    # unrecognized as turn output, so neither acknowledges. The error
    # still reaches the worker as the terminal failure it is.
    assert [e.event_type for e in seen] == [
        "provider.event",
        "turn.failed",
    ]

    seen.clear()
    asyncio.run(run())

    # A tool call is turn output, and it is the line that matters most:
    # a turn stopped by a guard right after running a tool must not be
    # resent to another provider. Acceptance is decided on the wire type
    # rather than on what the line normalizes to, so it holds whether or
    # not the normalizer gives tool activity its own event.
    assert [e.event_type for e in seen] == [
        "provider.prompt.accepted",
        "tool.started",
    ]


def test_grok_run_turn_gates_on_a_positive_child_limit(
    monkeypatch,
    tmp_path,
) -> None:
    lines = [b'{"type":"text","data":"ok"}\n']

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
    monkeypatch.setattr(
        grok, "resolve_grok_binary", lambda: "/usr/local/bin/grok"
    )

    async def handler(_event: ProviderEvent) -> None:
        return None

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def scenario() -> None:
        adapter = grok.GrokAdapter()
        result = await adapter.run_turn(
            workspace=tmp_path,
            prompt="do it",
            native_session_id="",
            permission_mode="full",
            model="grok-4.5",
            effort="low",
            event_handler=handler,
            approval_handler=approvals,
            child_launch_gate=ChildLaunchGate(
                database=tmp_path,
                command_id="cmd",
                limit=0,
            ),
        )
        assert result.status == "complete"
        with pytest.raises(RuntimeError, match="positive child-agent limit"):
            await adapter.run_turn(
                workspace=tmp_path,
                prompt="do it",
                native_session_id="",
                permission_mode="full",
                model="grok-4.5",
                effort="low",
                event_handler=handler,
                approval_handler=approvals,
                child_launch_gate=ChildLaunchGate(
                    database=tmp_path,
                    command_id="cmd",
                    limit=2,
                ),
            )

    asyncio.run(scenario())


def test_grok_run_turn_reports_a_nonzero_exit_as_a_failed_turn(
    monkeypatch,
    tmp_path,
) -> None:
    lines = [b"\n"]

    class Stdout:
        def __aiter__(self):
            async def gen():
                for line in lines:
                    yield line

            return gen()

    class Stderr:
        async def read(self) -> bytes:
            return b"Not signed in.\n"

    class Process:
        stdout = Stdout()
        stderr = Stderr()

        async def wait(self) -> int:
            return 2

    async def fake_exec(*_argv, **_kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        grok, "resolve_grok_binary", lambda: "/usr/local/bin/grok"
    )

    seen: list[ProviderEvent] = []

    async def handler(event: ProviderEvent) -> None:
        seen.append(event)

    async def approvals(_name: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def scenario() -> None:
        result = await grok.GrokAdapter().run_turn(
            workspace=tmp_path,
            prompt="do it",
            native_session_id="session_prior",
            permission_mode="full",
            model="grok-4.5",
            effort="low",
            event_handler=handler,
            approval_handler=approvals,
        )
        assert result.status == "failed"
        assert result.native_session_id == "session_prior"

    asyncio.run(scenario())

    assert [(item.event_type, item.text) for item in seen] == [
        ("turn.failed", "Not signed in.")
    ]


def test_selected_redaction_fails_closed_on_a_non_mapping_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        normalize,
        "redact_observable",
        lambda unused: ["unexpected"],
    )

    assert normalize._selected_redacted({"safe": "value"}, {"safe"}) == {}
