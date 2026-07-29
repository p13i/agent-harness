from __future__ import annotations

import threading

from agent_harness import cli


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
