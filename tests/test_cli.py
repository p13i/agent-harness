from __future__ import annotations

import asyncio
from pathlib import Path
import signal
import subprocess
import threading

import pytest

from agent_harness import cli
from agent_harness import client as client_module
from agent_harness.config import CONTROL_BUILD_ID
from agent_harness.config import CONTROL_PROTOCOL_VERSION
from agent_harness.config import paths as harness_paths_for
from agent_harness.errors import HarnessError
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
        lambda unused: (321,),
    )
    monkeypatch.setattr(
        client_module.os,
        "kill",
        lambda pid, value: signals.append((pid, value)),
    )

    assert asyncio.run(client_module.stop_daemon(harness_paths))
    assert signals == [(321, signal.SIGTERM)]


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

    async def fake_stop(unused) -> bool:
        del unused
        return True

    monkeypatch.setattr(cli, "publish_all", fake_publish)
    monkeypatch.setattr(cli, "migrate_state", fake_migrate)
    monkeypatch.setattr(cli, "stop_daemon", fake_stop)
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
