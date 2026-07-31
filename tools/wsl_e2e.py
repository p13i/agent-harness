"""Opt-in real WSL user-systemd lifecycle evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from agent_harness.service_manager import SystemdUserService
from agent_harness.service_manager import UnitConfiguration
from tools.bundle import verify_bundle
from tools.install import default_launcher
from tools.install import read_selection


E2E_SERVICE_NAME = "p13i-agent-harness-e2e.service"


def is_wsl(osrelease: Path = Path("/proc/sys/kernel/osrelease")) -> bool:
    try:
        value = osrelease.read_text(encoding="utf-8")
    except OSError:
        return False
    return "microsoft" in value.casefold()


def run_wsl_e2e(*, confirm: bool) -> dict[str, object]:
    if not is_wsl():
        return {
            "status": "pending",
            "reason": "real WSL user-systemd evidence requires WSL",
        }
    if not confirm:
        return {
            "status": "pending",
            "reason": "pass --confirm to mutate the isolated E2E unit",
        }
    selection = read_selection(default_launcher())
    if selection is None:
        return {
            "status": "failed",
            "reason": "install a verified bundle before WSL E2E",
        }
    bundle = verify_bundle(selection.executable.parent.parent)
    unit_path = (
        Path.home()
        / ".config"
        / "systemd"
        / "user"
        / E2E_SERVICE_NAME
    )
    if unit_path.exists():
        return {
            "status": "failed",
            "reason": "isolated E2E service unit already exists",
        }
    temporary_root = Path(
        tempfile.mkdtemp(prefix="p13i-agent-harness-wsl-e2e-")
    )
    state_dir = temporary_root / "chats"
    manager = SystemdUserService(
        unit_path,
        service_name=E2E_SERVICE_NAME,
    )
    installed = False
    try:
        manager.install(
            UnitConfiguration(
                executable=bundle.executable,
                state_dir=state_dir,
                build_id=bundle.build_id,
            )
        )
        installed = True
        manager.start()
        started = manager.status()
        if not started.active:
            return {
                "status": "failed",
                "reason": "isolated user service did not become active",
            }
        manager.restart()
        restarted = manager.status()
        if not restarted.active:
            return {
                "status": "failed",
                "reason": "isolated user service did not restart",
            }
        manager.stop()
        return {
            "status": "passed",
            "build_id": bundle.build_id,
            "service": E2E_SERVICE_NAME,
        }
    finally:
        if installed:
            manager.uninstall()
        shutil.rmtree(temporary_root)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    values = parser.parse_args(arguments)
    result = run_wsl_e2e(confirm=values.confirm)
    print(json.dumps(result, sort_keys=True))
    if result["status"] == "passed":
        return 0
    if result["status"] == "pending":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
