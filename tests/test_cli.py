from __future__ import annotations

import asyncio
from pathlib import Path
import signal
import subprocess
import threading
from types import SimpleNamespace

import pytest

from agent_harness import cli
from agent_harness import client as client_module
from agent_harness.config import CONTROL_BUILD_ID
from agent_harness.config import CONTROL_PROTOCOL_VERSION
from agent_harness.config import prepare_paths
from agent_harness.config import paths as harness_paths_for
from agent_harness.errors import HarnessError
from agent_harness.service_manager import ServiceStatus
from agent_harness.service_manager import DiagnosticProbe
from agent_harness import runtime


def test_launcher_prefers_current_source_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "agent-harness"
    package = source_root / "agent_harness"
    package.mkdir(parents=True)
    launcher = source_root / "bazel-bin" / "cmd" / "agent-harness"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o755)
    stale_runfiles = tmp_path / "stale.runfiles"
    stale_launcher = (
        stale_runfiles / "_main" / "cmd" / "agent-harness"
    )
    stale_launcher.parent.mkdir(parents=True)
    stale_launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    stale_launcher.chmod(0o755)

    monkeypatch.setattr(
        runtime,
        "__file__",
        str(package / "runtime.py"),
    )
    monkeypatch.setenv("RUNFILES_DIR", str(stale_runfiles))

    assert runtime.launcher_command() == [str(launcher)]


def test_launcher_discovers_runfiles_sys_path_invocation_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "source" / "agent_harness"
    package.mkdir(parents=True)
    monkeypatch.setattr(
        runtime,
        "__file__",
        str(package / "runtime.py"),
    )

    runfiles = tmp_path / "runfiles"
    runfiles_launcher = (
        runfiles / "workspace" / "cmd" / "agent-harness"
    )
    runfiles_launcher.parent.mkdir(parents=True)
    runfiles_launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    runfiles_launcher.chmod(0o755)
    monkeypatch.setenv("RUNFILES_DIR", str(runfiles))
    monkeypatch.setenv("TEST_WORKSPACE", "workspace")
    assert runtime.launcher_command() == [str(runfiles_launcher)]

    monkeypatch.delenv("RUNFILES_DIR")
    sys_root = tmp_path / "sys-path"
    sys_launcher = sys_root / "cmd" / "agent-harness"
    sys_launcher.parent.mkdir(parents=True)
    sys_launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    sys_launcher.chmod(0o755)
    monkeypatch.setattr(runtime.sys, "path", [str(sys_root)])
    assert runtime.launcher_command() == [str(sys_launcher)]

    monkeypatch.setattr(runtime.sys, "path", [])
    invoked = tmp_path / "invoked" / "agent-harness"
    invoked.parent.mkdir(parents=True)
    invoked.write_text("#!/bin/sh\n", encoding="utf-8")
    invoked.chmod(0o755)
    monkeypatch.setattr(runtime.sys, "argv", [str(invoked)])
    assert runtime.launcher_command() == [str(invoked.resolve())]

    invoked.unlink()
    assert runtime.launcher_command() == [
        runtime.sys.executable,
        "-m",
        "agent_harness.cli",
    ]


def test_chat_runs_textual_on_the_main_thread(
    monkeypatch,
    tmp_path,
) -> None:
    harness_paths = object()
    client = object()
    captured: dict[str, object] = {}

    def fake_paths(state_dir):
        captured["state_dir"] = state_dir
        return harness_paths

    def fake_prepare_paths(value) -> None:
        captured["prepared"] = value

    async def fake_ensure_daemon(value):
        captured["daemon_paths"] = value
        return client

    def fake_run_tui(
        value,
        workspace,
        *,
        session_id,
        permission_mode,
    ) -> None:
        captured["client"] = value
        captured["workspace"] = workspace
        captured["session_id"] = session_id
        captured["permission_mode"] = permission_mode
        captured["thread"] = threading.current_thread()

    monkeypatch.setattr(cli, "paths", fake_paths)
    monkeypatch.setattr(cli, "prepare_paths", fake_prepare_paths)
    monkeypatch.setattr(cli, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(cli, "run_tui", fake_run_tui)

    status = cli.main(["--cwd", str(tmp_path), "chat"])

    assert status == 0
    assert captured["prepared"] is harness_paths
    assert captured["daemon_paths"] is harness_paths
    assert captured["client"] is client
    assert captured["workspace"] == tmp_path.resolve()
    assert captured["session_id"] == ""
    assert captured["permission_mode"] == "approval"
    assert captured["thread"] is threading.main_thread()


def test_client_replaces_an_incompatible_daemon_before_chat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness_paths = harness_paths_for(tmp_path / "state")
    health_values = [
        {"status": "ok"},
        {
            "status": "ok",
            "control_build_id": CONTROL_BUILD_ID,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
        },
    ]
    stopped: list[object] = []
    launched: list[list[str]] = []

    async def fake_health(unused) -> dict[str, object]:
        del unused
        return health_values.pop(0)

    async def fake_stop(value, client) -> None:
        stopped.extend([value, client])

    class Process:
        def __init__(self, command, **kwargs) -> None:
            del kwargs
            launched.append(command)

    monkeypatch.setattr(
        client_module.HarnessClient,
        "_health_payload",
        fake_health,
    )
    monkeypatch.setattr(
        client_module,
        "_stop_incompatible_daemon",
        fake_stop,
    )
    monkeypatch.setattr(
        client_module,
        "launcher_command",
        lambda: ["agent-harness"],
    )
    monkeypatch.setattr(client_module.subprocess, "Popen", Process)

    client = asyncio.run(client_module.ensure_daemon(harness_paths))

    assert isinstance(client, client_module.HarnessClient)
    assert stopped[0] is harness_paths
    assert launched[0][-1] == "daemon"


def test_client_transport_validates_success_error_and_json_shapes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness_paths = harness_paths_for(tmp_path / "state")
    responses = [
        (200, '{"value":1}'),
        (409, '{"error":[]}'),
        (200, "invalid"),
        (200, "[]"),
    ]
    captured: dict[str, object] = {}

    class Response:
        def __init__(self, status: int, content: str) -> None:
            self.status = status
            self.content = content

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> None:
            del exception_type
            del exception
            del traceback

        async def text(self) -> str:
            return self.content

    class Session:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            if "first_headers" not in captured:
                captured["first_headers"] = kwargs["headers"]

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> None:
            del exception_type
            del exception
            del traceback

        def request(
            self,
            method: str,
            path: str,
            *,
            json: object,
        ) -> Response:
            captured["request"] = (method, path, json)
            status, content = responses.pop(0)
            return Response(status, content)

    monkeypatch.setattr(
        client_module,
        "UnixConnector",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(client_module, "ClientSession", Session)
    client = client_module.HarnessClient(harness_paths)

    async def scenario() -> None:
        assert await client.request(
            "POST",
            "/v1/test",
            payload={"one": 1},
            idempotency_key="request-1",
        ) == {"value": 1}
        with pytest.raises(HarnessError) as remote:
            await client.request("GET", "/v1/error")
        assert remote.value.detail.code == "E_REMOTE"
        assert remote.value.detail.status == 409
        with pytest.raises(HarnessError) as invalid:
            await client.request("GET", "/v1/invalid")
        assert invalid.value.detail.code == "E_PROTOCOL"
        with pytest.raises(HarnessError) as non_object:
            await client.request("GET", "/v1/list")
        assert non_object.value.detail.code == "E_PROTOCOL"

    asyncio.run(scenario())
    headers = captured["first_headers"]
    assert isinstance(headers, dict)
    assert headers["Idempotency-Key"] == "request-1"


def test_client_health_wait_and_projection_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness_paths = harness_paths_for(tmp_path / "state")
    client = client_module.HarnessClient(harness_paths)

    async def healthy(
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ) -> dict[str, object]:
        del method
        del path
        del payload
        del idempotency_key
        return {
            "status": "ok",
            "control_build_id": CONTROL_BUILD_ID,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
        }

    monkeypatch.setattr(client, "request", healthy)
    assert asyncio.run(client.health())

    async def unavailable(
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ) -> dict[str, object]:
        del method
        del path
        del payload
        del idempotency_key
        raise OSError("unavailable")

    monkeypatch.setattr(client, "request", unavailable)
    assert asyncio.run(client._health_payload()) == {}
    assert not asyncio.run(client.health())

    class Commands:
        def __init__(self) -> None:
            self.values = [
                {"command": "invalid"},
                {"command": {"status": "complete", "value": 1}},
            ]

        async def request(
            self,
            method: str,
            path: str,
        ) -> dict[str, object]:
            del method
            del path
            return self.values.pop(0)

    commands = Commands()

    async def scenario() -> None:
        first = await client_module.wait_command(
            commands,  # type: ignore[arg-type]
            "command-1",
            timeout=0,
        )
        assert first == {}
        second = await client_module.wait_command(
            commands,  # type: ignore[arg-type]
            "command-1",
        )
        assert second["status"] == "complete"

    asyncio.run(scenario())
    projection = tmp_path / "projection.json"
    projection.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        client_module.read_projection(projection)
    projection.write_text('{"value":1}', encoding="utf-8")
    assert client_module.read_projection(projection) == {"value": 1}


def test_client_reuses_compatible_daemon_and_rejects_unmanaged_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness_paths = harness_paths_for(tmp_path / "state")

    async def compatible(unused) -> dict[str, object]:
        del unused
        return {
            "status": "ok",
            "control_build_id": CONTROL_BUILD_ID,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
        }

    monkeypatch.setattr(
        client_module.HarnessClient,
        "_health_payload",
        compatible,
    )
    assert isinstance(
        asyncio.run(client_module.ensure_daemon(harness_paths)),
        client_module.HarnessClient,
    )

    async def running(unused) -> bool:
        del unused
        return True

    monkeypatch.setattr(client_module.HarnessClient, "health", running)
    monkeypatch.setattr(
        client_module,
        "_managed_daemon_pids",
        lambda unused: (),
    )
    with pytest.raises(HarnessError, match="could not be identified"):
        asyncio.run(client_module.stop_daemon(harness_paths))

    async def stopped(unused) -> bool:
        del unused
        return False

    monkeypatch.setattr(client_module.HarnessClient, "health", stopped)
    assert not asyncio.run(client_module.stop_daemon(harness_paths))


def test_client_stops_only_managed_incompatible_daemons(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness_paths = harness_paths_for(tmp_path / "state")
    client = client_module.HarnessClient(harness_paths)
    health_values = [{"status": "ok"}, {}]
    signals: list[tuple[int, signal.Signals]] = []

    async def fake_health() -> dict[str, object]:
        return health_values.pop(0)

    def terminate(pid: int, value: signal.Signals) -> None:
        signals.append((pid, value))
        if pid == 456:
            raise ProcessLookupError

    monkeypatch.setattr(client, "_health_payload", fake_health)
    monkeypatch.setattr(
        client_module,
        "_managed_daemon_pids",
        lambda unused: (123, 456),
    )
    monkeypatch.setattr(client_module.os, "kill", terminate)

    asyncio.run(
        client_module._stop_incompatible_daemon(
            harness_paths,
            client,
        )
    )

    assert signals == [
        (123, signal.SIGTERM),
        (456, signal.SIGTERM),
    ]

    monkeypatch.setattr(
        client_module,
        "_managed_daemon_pids",
        lambda unused: (),
    )
    with pytest.raises(HarnessError, match="could not be identified"):
        asyncio.run(
            client_module._stop_incompatible_daemon(
                harness_paths,
                client,
            )
        )


def test_client_discovers_only_daemon_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness_paths = harness_paths_for(tmp_path / "state")
    harness_paths.daemon_pid.parent.mkdir(parents=True)
    harness_paths.daemon_pid.write_text("123\n", encoding="utf-8")

    def run(command, **kwargs):
        del kwargs
        if command[0] == "ps":
            return subprocess.CompletedProcess(
                command,
                0,
                "agent-harness --state-dir state daemon\n",
                "",
            )
        raise AssertionError("lsof fallback was not expected")

    monkeypatch.setattr(client_module.subprocess, "run", run)
    monkeypatch.setattr(client_module.os, "getpid", lambda: 999)
    assert client_module._managed_daemon_pids(harness_paths) == (123,)

    harness_paths.daemon_pid.write_text("invalid\n", encoding="utf-8")

    def fallback(command, **kwargs):
        del kwargs
        if command[0] == "lsof":
            return subprocess.CompletedProcess(
                command,
                0,
                "bad\n456\n789\n",
                "",
            )
        if command[2] == "456":
            return subprocess.CompletedProcess(
                command,
                0,
                "agent-harness worker session\n",
                "",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            "agent-harness daemon\n",
            "",
        )

    monkeypatch.setattr(client_module.subprocess, "run", fallback)
    assert client_module._managed_daemon_pids(harness_paths) == (789,)
    assert client_module._filter_managed_daemon_pids(
        {0, 999, 456}
    ) == ()


def test_client_stops_a_managed_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness_paths = harness_paths_for(tmp_path / "state")
    health = [True, False]
    signals = []

    async def fake_health(unused) -> bool:
        del unused
        return health.pop(0)

    monkeypatch.setattr(
        client_module.HarnessClient,
        "health",
        fake_health,
    )
    monkeypatch.setattr(
        client_module,
        "_managed_daemon_pids",
        lambda unused: (321, 322),
    )

    def terminate(pid: int, value: signal.Signals) -> None:
        signals.append((pid, value))
        if pid == 322:
            raise ProcessLookupError

    monkeypatch.setattr(
        client_module.os,
        "kill",
        terminate,
    )

    assert asyncio.run(client_module.stop_daemon(harness_paths))
    assert signals == [
        (321, signal.SIGTERM),
        (322, signal.SIGTERM),
    ]


def test_cli_manages_sync_migration_and_service_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    migrated = []

    def fake_publish(unused_paths, unused_store):
        del unused_paths
        del unused_store
        return {"state": "synced"}

    def fake_migrate(
        source,
        destination,
        *,
        trash_source,
    ):
        migrated.append((source, destination, trash_source))
        return {"sessions": 2}

    class Manager:
        def __init__(self) -> None:
            self.unit_path = tmp_path / "service"
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

        def status(self) -> ServiceStatus:
            return ServiceStatus(
                active=True,
                installed=True,
                unit_version=1,
                build_id="build",
                detail="active",
            )

    manager = Manager()

    monkeypatch.setattr(cli, "publish_all", fake_publish)
    monkeypatch.setattr(cli, "migrate_state", fake_migrate)
    monkeypatch.setattr(cli, "_service_manager", lambda: manager)
    state = tmp_path / "state"
    source = tmp_path / "legacy"
    destination = tmp_path / "chats"

    commands = [
        ["--state-dir", str(state), "sync"],
        ["--state-dir", str(state), "sync-status"],
        ["--state-dir", str(state), "service", "stop"],
        [
            "migrate-state",
            "--from",
            str(source),
            "--to",
            str(destination),
            "--trash-source",
        ],
    ]
    for arguments in commands:
        parsed = cli.parser().parse_args(arguments)
        assert asyncio.run(cli._run(parsed)) == 0

    assert migrated == [(source, destination, True)]
    assert manager.stopped
    output = capsys.readouterr().out
    assert '"state": "synced"' in output
    assert '"stopped": true' in output


def test_cli_dispatches_every_session_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str, object, str]] = []

    class Client:
        async def request(
            self,
            method,
            path,
            *,
            payload=None,
            idempotency_key="",
        ):
            requests.append(
                (method, path, payload, idempotency_key)
            )
            if path.endswith("/messages"):
                return {
                    "command": {
                        "command_id": "command-1",
                        "status": "queued",
                    }
                }
            return {"ok": True}

    client = Client()

    async def fake_ensure_daemon(unused):
        del unused
        return client

    async def fake_wait_command(unused_client, command_id):
        del unused_client
        return {
            "command_id": command_id,
            "status": "complete",
        }

    monkeypatch.setattr(cli, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(cli, "wait_command", fake_wait_command)
    transfer_file = tmp_path / "transfer.json"
    transfer_file.write_text('{"sealed":"value"}\n', encoding="utf-8")
    base = ["--state-dir", str(tmp_path / "state")]
    commands = [
        [
            *base,
            "--cwd",
            str(tmp_path),
            "new",
            "--name",
            "session",
            "--goal",
            "finish",
            "--constraint",
            "bounded",
            "--predicate",
            '{"type":"command"}',
            "--budgets",
            '{"turns":2}',
            "--execution-profile",
            "unattended",
        ],
        [
            *base,
            "send",
            "session-1",
            "continue",
            "--provider",
            "codex",
            "--wait",
        ],
        [*base, "status"],
        [*base, "status", "session-1"],
        [*base, "--cwd", str(tmp_path), "providers"],
        [*base, "usage", "session-1"],
        [
            *base,
            "extend-budget",
            "session-1",
            "--seconds",
            "60",
            "--tokens",
            "1000",
            "--allow-xhigh-once",
            "--reason",
            "bounded",
        ],
        [
            *base,
            "events",
            "session-1",
            "--after",
            "2",
            "--limit",
            "10",
        ],
        [*base, "fork", "session-1", "--name", "fork"],
        [*base, "checkpoint", "session-1"],
        [*base, "archive", "session-1"],
        [*base, "unarchive", "session-1"],
        [*base, "reconcile", "list", "session-1"],
        [
            *base,
            "reconcile",
            "inspect",
            "reconciliation-1",
        ],
        [
            *base,
            "reconcile",
            "resolve",
            "reconciliation-1",
            "accept-current",
            "--observed-workspace-digest",
            "digest-1",
            "--audit",
            '{"actor":"test"}',
        ],
        [
            *base,
            "route",
            "session-1",
            "--provider",
            "claude",
            "--model",
            "opus",
            "--effort",
            "high",
            "--required-capability",
            "tools",
            "--metered-budget",
            "1",
        ],
        [*base, "action", "session-1", "pause"],
        [
            *base,
            "evidence",
            "session-1",
            "command",
            "make test",
            "passed",
            "--value",
            '{"exit_code":0}',
        ],
        [*base, "export", "session-1"],
        [
            *base,
            "transfer",
            "create",
            "session-1",
            "host-2",
            "public-key",
        ],
        [
            *base,
            "transfer",
            "import",
            str(transfer_file),
            "signing-key",
        ],
        [
            *base,
            "transfer",
            "finalize",
            "session-1",
            "host-2",
            "2",
        ],
    ]

    for arguments in commands:
        parsed = cli.parser().parse_args(arguments)
        assert asyncio.run(cli._run(parsed)) == 0

    capsys.readouterr()
    paths = [item[1] for item in requests]
    assert "/v1/sessions" in paths
    assert "/v1/sessions/session-1/messages" in paths
    assert "/v1/transfers/import" in paths
    assert (
        "/v1/reconciliations/reconciliation-1/resolution"
        in paths
    )
    assert any(
        item[1].endswith("/budget-extensions")
        and bool(item[3])
        for item in requests
    )


def test_cli_rejects_invalid_event_and_evidence_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Client:
        async def request(self, *args, **kwargs):
            del args
            del kwargs
            return {}

    async def fake_ensure_daemon(unused):
        del unused
        return Client()

    monkeypatch.setattr(cli, "ensure_daemon", fake_ensure_daemon)
    base = ["--state-dir", str(tmp_path / "state")]
    invalid = [
        [*base, "events", "session-1", "--after", "-1"],
        [*base, "events", "session-1", "--limit", "5001"],
        [
            *base,
            "evidence",
            "session-1",
            "command",
            "make test",
            "passed",
            "--value",
            "[]",
        ],
    ]

    for arguments in invalid:
        parsed = cli.parser().parse_args(arguments)
        with pytest.raises(ValueError):
            asyncio.run(cli._run(parsed))

    assert cli._json_object('{"value":1}', "--value") == {
        "value": 1
    }
    with pytest.raises(ValueError, match="JSON object"):
        cli._json_object("[]", "--value")
    assert cli._object({"value": 1}) == {"value": 1}
    assert cli._object(None) == {}
    assert cli._command_exit({"status": "failed"}) == 1
    assert cli._command_exit({"status": "complete"}) == 0


def test_cli_manages_installed_user_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    configurations = []

    class Manager:
        unit_path = tmp_path / "agent-harness.service"

        def install(self, configuration) -> None:
            calls.append("install")
            configurations.append(configuration)

        def start(self) -> None:
            calls.append("start")

        def restart(self) -> None:
            calls.append("restart")

        def status(self):
            calls.append("status")
            return ServiceStatus(
                active=True,
                installed=True,
                unit_version=1,
                build_id="build-1",
                detail="active",
            )

        def stop(self) -> None:
            calls.append("stop")

        def uninstall(self) -> None:
            calls.append("uninstall")

    manager = Manager()
    selection = SimpleNamespace(
        executable=tmp_path / "bundle" / "bin" / "agent-harness",
        build_id="build-1",
    )
    monkeypatch.setattr(cli, "_service_manager", lambda: manager)
    monkeypatch.setattr(cli, "_installed_selection", lambda: selection)
    base = ["--state-dir", str(tmp_path / "state"), "service"]
    for action in (
        "install",
        "start",
        "restart",
        "status",
        "stop",
        "uninstall",
    ):
        parsed = cli.parser().parse_args([*base, action])
        assert asyncio.run(cli._run(parsed)) == 0

    assert calls == [
        "install",
        "start",
        "restart",
        "status",
        "status",
        "stop",
        "uninstall",
    ]
    assert configurations[0].executable == selection.executable
    assert configurations[0].state_dir == (tmp_path / "state")
    assert configurations[0].build_id == "build-1"
    assert '"state_preserved"' in capsys.readouterr().out


def test_service_status_and_stop_preserve_local_daemon_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Manager:
        def status(self) -> ServiceStatus:
            return ServiceStatus(
                active=False,
                installed=False,
                unit_version=None,
                build_id="",
                detail="not installed",
            )

    class Client:
        def __init__(self, unused_paths: object) -> None:
            del unused_paths

        async def health(self) -> bool:
            return True

    stopped: list[object] = []

    async def local_stop(harness_paths: object) -> bool:
        stopped.append(harness_paths)
        return True

    monkeypatch.setattr(cli, "_service_manager", lambda: Manager())
    monkeypatch.setattr(cli, "HarnessClient", Client)
    monkeypatch.setattr(cli, "stop_daemon", local_stop)
    base = ["--state-dir", str(tmp_path / "state"), "service"]

    parsed = cli.parser().parse_args([*base, "status"])
    assert asyncio.run(cli._run(parsed)) == 0
    parsed = cli.parser().parse_args([*base, "stop"])
    assert asyncio.run(cli._run(parsed)) == 0

    output = capsys.readouterr().out
    assert '"detail": "local daemon"' in output
    assert '"service": false' in output
    assert len(stopped) == 1


def test_doctor_reports_bundle_service_daemon_and_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness_paths = harness_paths_for(tmp_path / "state")
    prepare_paths(harness_paths)
    (harness_paths.state_dir / ".git").mkdir()

    class Manager:
        def diagnostics(self):
            return (
                DiagnosticProbe(
                    "user-systemd",
                    "warning",
                    "not available in this fixture",
                ),
            )

    async def health(unused):
        del unused
        return {
            "status": "ok",
            "control_build_id": CONTROL_BUILD_ID,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
        }

    def no_installed_selection():
        raise ValueError("not installed")

    monkeypatch.setattr(cli, "_service_manager", lambda: Manager())
    monkeypatch.setattr(
        cli,
        "_installed_selection",
        no_installed_selection,
    )
    monkeypatch.setattr(
        cli.HarnessClient,
        "_health_payload",
        health,
    )
    monkeypatch.setattr(cli.shutil, "which", lambda unused: "/bin/tool")
    monkeypatch.setattr(
        cli.shutil,
        "disk_usage",
        lambda unused: SimpleNamespace(free=10 * 1024**3),
    )
    monkeypatch.setattr(
        cli,
        "read_sync_status",
        lambda unused: {
            "state": "synced",
            "updated_at": "2099-01-01T00:00:00+00:00",
        },
    )

    assert asyncio.run(cli._doctor(harness_paths)) == 0
    output = capsys.readouterr().out
    assert '"sqlite": true' in output
    assert '"status": "compatible"' in output
    assert '"installed bundle is absent or invalid"' in output
    assert '"private_socket_mode": true' in output
    assert '"stale_process_leases": []' in output
    assert '"stale_workers": []' in output
    assert cli._sync_lag_seconds({"updated_at": ""}) is None
    assert cli._sync_lag_seconds({"updated_at": "invalid"}) is None

    harness_paths.state_dir.chmod(0o755)
    harness_paths.socket.write_text("not a socket", encoding="utf-8")
    harness_paths.socket.chmod(0o644)
    assert asyncio.run(cli._doctor(harness_paths)) == 1
    unsafe_output = capsys.readouterr().out
    assert '"private_socket_mode": false' in unsafe_output
    assert '"private_state_mode": false' in unsafe_output


def test_doctor_stale_timestamp_classification() -> None:
    assert cli._timestamp_is_stale(
        "2000-01-01T00:00:00+00:00",
        maximum_age_seconds=90,
    )
    assert not cli._timestamp_is_stale(
        "2099-01-01T00:00:00+00:00",
        maximum_age_seconds=90,
    )
    assert cli._timestamp_is_stale(
        "invalid",
        maximum_age_seconds=90,
    )
    assert cli._timestamp_is_expired("2000-01-01T00:00:00+00:00")
    assert not cli._timestamp_is_expired(
        "2099-01-01T00:00:00+00:00"
    )
    assert cli._timestamp_is_expired("invalid")
