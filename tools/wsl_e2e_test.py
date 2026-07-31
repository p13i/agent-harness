from pathlib import Path

import pytest

from tools import wsl_e2e


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
