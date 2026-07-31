from pathlib import Path
import runpy
from types import SimpleNamespace
import sys

import pytest

from agent_harness import wsl_e2e


def test_wsl_detection_and_pending_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    osrelease = tmp_path / "osrelease"
    osrelease.write_text("Darwin\n", encoding="utf-8")
    assert not wsl_e2e.is_wsl(osrelease)
    osrelease.write_text(
        "5.15.0-microsoft-standard-WSL2\n",
        encoding="utf-8",
    )
    assert wsl_e2e.is_wsl(osrelease)
    monkeypatch.setattr(wsl_e2e, "is_wsl", lambda: False)
    assert wsl_e2e.run_wsl_e2e(confirm=False)["status"] == "pending"


def test_wsl_requires_explicit_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wsl_e2e, "is_wsl", lambda: True)
    result = wsl_e2e.run_wsl_e2e(confirm=False)
    assert result["status"] == "pending"
    assert "--confirm" in str(result["reason"])


def test_wsl_e2e_requires_bundle_and_unused_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wsl_e2e, "is_wsl", lambda: True)
    monkeypatch.setattr(wsl_e2e, "default_launcher", lambda: tmp_path)
    monkeypatch.setattr(wsl_e2e, "read_selection", lambda path: None)
    result = wsl_e2e.run_wsl_e2e(confirm=True)
    assert result["status"] == "failed"
    assert "verified bundle" in str(result["reason"])

    executable = tmp_path / "bundle" / "bin" / "agent-harness"
    executable.parent.mkdir(parents=True)
    executable.write_text("")
    selection = SimpleNamespace(executable=executable)
    monkeypatch.setattr(
        wsl_e2e,
        "read_selection",
        lambda path: selection,
    )
    monkeypatch.setattr(
        wsl_e2e,
        "verify_bundle",
        lambda path: SimpleNamespace(
            executable=executable,
            build_id="build-1",
        ),
    )
    unit = (
        tmp_path
        / ".config"
        / "systemd"
        / "user"
        / wsl_e2e.E2E_SERVICE_NAME
    )
    unit.parent.mkdir(parents=True)
    unit.write_text("occupied")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = wsl_e2e.run_wsl_e2e(confirm=True)
    assert result["status"] == "failed"
    assert "already exists" in str(result["reason"])


def test_wsl_e2e_service_lifecycle_and_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "bundle" / "bin" / "agent-harness"
    executable.parent.mkdir(parents=True)
    executable.write_text("")
    bundle = SimpleNamespace(
        executable=executable,
        build_id="build-1",
    )
    monkeypatch.setattr(wsl_e2e, "is_wsl", lambda: True)
    monkeypatch.setattr(
        wsl_e2e,
        "read_selection",
        lambda path: SimpleNamespace(executable=executable),
    )
    monkeypatch.setattr(wsl_e2e, "verify_bundle", lambda path: bundle)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        wsl_e2e.tempfile,
        "mkdtemp",
        lambda prefix: str(tmp_path / "runtime"),
    )
    removed: list[Path] = []
    monkeypatch.setattr(
        wsl_e2e.shutil,
        "rmtree",
        lambda path: removed.append(path),
    )

    class Manager:
        active = [True, True]
        instances: list["Manager"] = []

        def __init__(self, unit_path: Path, *, service_name: str) -> None:
            self.unit_path = unit_path
            self.service_name = service_name
            self.actions: list[str] = []
            self.instances.append(self)

        def install(self, configuration: object) -> None:
            self.configuration = configuration
            self.actions.append("install")

        def start(self) -> None:
            self.actions.append("start")

        def restart(self) -> None:
            self.actions.append("restart")

        def stop(self) -> None:
            self.actions.append("stop")

        def status(self) -> object:
            return SimpleNamespace(active=self.active.pop(0))

        def uninstall(self) -> None:
            self.actions.append("uninstall")

    monkeypatch.setattr(wsl_e2e, "SystemdUserService", Manager)
    result = wsl_e2e.run_wsl_e2e(confirm=True)
    assert result["status"] == "passed"
    assert Manager.instances[-1].actions == [
        "install",
        "start",
        "restart",
        "stop",
        "uninstall",
    ]
    assert removed

    Manager.active = [False]
    result = wsl_e2e.run_wsl_e2e(confirm=True)
    assert result["status"] == "failed"
    assert "active" in str(result["reason"])

    Manager.active = [True, False]
    result = wsl_e2e.run_wsl_e2e(confirm=True)
    assert result["status"] == "failed"
    assert "restart" in str(result["reason"])


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("passed", 0),
        ("pending", 2),
        ("failed", 1),
    ],
)
def test_wsl_e2e_main_status_codes(
    status: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        wsl_e2e,
        "run_wsl_e2e",
        lambda confirm: {"status": status},
    )
    assert wsl_e2e.main([]) == expected
    assert '"status"' in capsys.readouterr().out


def test_wsl_detection_treats_unreadable_release_as_not_wsl(
    tmp_path: Path,
) -> None:
    assert not wsl_e2e.is_wsl(tmp_path / "missing")


def test_wsl_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["wsl-e2e"])
    with pytest.raises(SystemExit, match="2"):
        runpy.run_module(
            "agent_harness.wsl_e2e",
            run_name="__main__",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
