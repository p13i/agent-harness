"""No-spend validation for the opt-in live smoke entry point."""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_live_smoke_packages_daemon_launcher() -> None:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "_main")
    launcher = runfiles / workspace / "cmd" / "agent-harness"

    assert launcher.is_file()
    assert os.access(launcher, os.X_OK)


def test_live_smoke_uses_bazel_callers_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module()
    monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(tmp_path))

    assert module.default_workspace() == tmp_path

    monkeypatch.delenv("BUILD_WORKING_DIRECTORY")
    monkeypatch.chdir(tmp_path)

    assert module.default_workspace() == tmp_path


def test_live_smoke_refuses_to_invoke_without_confirmation() -> None:
    result = subprocess.run(
        [str(_launcher())],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--confirm-spend is required" in result.stderr
    assert "no provider was invoked" in result.stderr


def test_live_smoke_uses_exactly_one_turn_per_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module()
    providers: list[str] = []
    stopped: list[str] = []

    class Client:
        raw = object()

        def __init__(self, unused_paths: object) -> None:
            del unused_paths

        async def create_session(
            self,
            workspace: Path,
            **values: object,
        ) -> dict[str, str]:
            assert workspace == tmp_path
            assert values["permission_mode"] == "read-only"
            assert values["execution_profile"] == "live-smoke"
            return {"session_id": "session-1"}

        async def send_message(
            self,
            session_id: str,
            prompt: str,
            *,
            provider: str,
            effort: str,
            workload: str,
        ) -> SimpleNamespace:
            assert session_id == "session-1"
            assert prompt == module.PROVIDER_PROMPTS[provider]
            assert effort == "low"
            assert workload == "operations"
            providers.append(provider)
            return SimpleNamespace(command_id="command-" + provider)

        async def usage(self, session_id: str) -> dict[str, object]:
            assert session_id == "session-1"
            return {
                "envelopes": [
                    {
                        "profile": "live-smoke",
                        "limits": {"max_attempts": 1},
                    },
                    {
                        "profile": "live-smoke",
                        "limits": {"max_attempts": 1},
                    },
                ]
            }

        async def command(self, session_id: str, action: str) -> None:
            assert action == "stop"
            stopped.append(session_id)

    async def ready(unused_paths: object) -> None:
        del unused_paths

    async def complete(
        unused_raw: object,
        command_id: str,
        *,
        timeout: float,
    ) -> dict[str, object]:
        del unused_raw
        assert timeout == 330
        provider = command_id.removeprefix("command-")
        return {
            "status": "complete",
            "result": {"native_session_id": "native-" + provider},
        }

    monkeypatch.setattr(module, "ensure_daemon", ready)
    monkeypatch.setattr(module, "AgentHarnessClient", Client)
    monkeypatch.setattr(module, "wait_command", complete)
    result = asyncio.run(
        module.run(
            SimpleNamespace(
                confirm_spend=True,
                state_dir=tmp_path / "state",
                workspace=tmp_path,
            )
        )
    )

    assert providers == ["claude", "codex"]
    assert result["commands"] == ["command-claude", "command-codex"]
    assert result["native_sessions"] == {
        "claude": "native-claude",
        "codex": "native-codex",
    }
    assert stopped == ["session-1"]


def test_live_smoke_rejects_unconfirmed_direct_run(
    tmp_path: Path,
) -> None:
    module = _smoke_module()

    with pytest.raises(ValueError, match="confirm-spend"):
        asyncio.run(
            module.run(
                SimpleNamespace(
                    confirm_spend=False,
                    state_dir=tmp_path / "state",
                    workspace=tmp_path,
                )
            )
        )


@pytest.mark.parametrize(
    ("envelopes", "message"),
    [
        ([], "exactly two"),
        (["invalid", {}], "invalid live-smoke envelope"),
        (
            [
                {"profile": "other", "limits": {}},
                {"profile": "live-smoke", "limits": {}},
            ],
            "profile was not enforced",
        ),
        (
            [
                {"profile": "live-smoke", "limits": []},
                {"profile": "live-smoke", "limits": {}},
            ],
            "limits are absent",
        ),
        (
            [
                {
                    "profile": "live-smoke",
                    "limits": {"max_attempts": 2},
                },
                {
                    "profile": "live-smoke",
                    "limits": {"max_attempts": 1},
                },
            ],
            "permitted provider recovery",
        ),
    ],
)
def test_live_smoke_rejects_invalid_safety_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    envelopes: list[object],
    message: str,
) -> None:
    module = _smoke_module()

    class Client:
        raw = object()

        def __init__(self, unused_paths: object) -> None:
            del unused_paths

        async def create_session(
            self,
            unused_workspace: Path,
            **unused_values: object,
        ) -> dict[str, str]:
            del unused_workspace, unused_values
            return {"session_id": "session-1"}

        async def send_message(
            self,
            unused_session_id: str,
            unused_prompt: str,
            **unused_values: object,
        ) -> SimpleNamespace:
            del unused_session_id, unused_prompt, unused_values
            return SimpleNamespace(command_id="command")

        async def usage(
            self,
            unused_session_id: str,
        ) -> dict[str, object]:
            del unused_session_id
            return {"envelopes": envelopes}

        async def command(
            self,
            unused_session_id: str,
            unused_action: str,
        ) -> None:
            del unused_session_id, unused_action
            raise module.HarnessError("E_STOP", "stop failed")

    async def ready(unused_paths: object) -> None:
        del unused_paths

    async def complete(
        unused_raw: object,
        unused_command_id: str,
        *,
        timeout: float,
    ) -> dict[str, object]:
        del unused_raw, unused_command_id
        assert timeout == 330
        return {
            "status": "complete",
            "result": {"native_session_id": "native"},
        }

    monkeypatch.setattr(module, "ensure_daemon", ready)
    monkeypatch.setattr(module, "AgentHarnessClient", Client)
    monkeypatch.setattr(module, "wait_command", complete)

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(
            module.run(
                SimpleNamespace(
                    confirm_spend=True,
                    state_dir=tmp_path / "state",
                    workspace=tmp_path,
                )
            )
        )


def test_live_smoke_result_validation_boundaries() -> None:
    module = _smoke_module()

    with pytest.raises(RuntimeError, match="did not complete"):
        module._require_complete(
            {"status": "failed", "result": {"reason": "test"}},
            "codex",
        )
    with pytest.raises(RuntimeError, match="result is absent"):
        module._native_session({"result": []})
    with pytest.raises(RuntimeError, match="native session id"):
        module._native_session({"result": {}})


def test_live_smoke_main_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _smoke_module()

    async def success(
        unused_arguments: object,
    ) -> dict[str, object]:
        del unused_arguments
        return {"status": "complete"}

    monkeypatch.setattr(module, "run", success)
    assert module.main(["--confirm-spend"]) == 0
    assert '"status": "complete"' in capsys.readouterr().out

    async def failure(unused_arguments: object) -> dict[str, object]:
        del unused_arguments
        raise RuntimeError("bounded failure")

    monkeypatch.setattr(module, "run", failure)
    assert module.main(["--confirm-spend"]) == 2
    assert "bounded failure" in capsys.readouterr().err


def test_live_smoke_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _smoke_source()
    monkeypatch.setattr(sys, "argv", [str(source)])

    with pytest.raises(SystemExit, match="2"):
        runpy.run_path(str(source), run_name="__main__")


def _launcher() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "_main")
    candidate = runfiles / workspace / "tools" / "live_smoke"
    if candidate.is_file():
        return candidate
    return runfiles / "_main" / "tools" / "live_smoke"


def _smoke_module():
    source = _smoke_source()
    spec = importlib.util.spec_from_file_location(
        "live_smoke_module",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load live smoke module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smoke_source() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "_main")
    return runfiles / workspace / "tools" / "live_smoke.gpt.py"


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "--import-mode=importlib",
            ]
        )
    )
