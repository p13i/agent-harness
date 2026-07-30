from pathlib import Path

import pytest

from tools.install import install
from tools.install import launcher


def test_launcher_is_pinned_to_bazel_and_repo(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo with spaces"
    destination = tmp_path / "bin" / "agent-harness"

    install(repo, destination)

    assert destination.stat().st_mode & 0o111
    content = destination.read_text(encoding="utf-8")
    assert content == launcher(repo)
    assert "@bazel/bazelisk" in content
    assert "//cmd:agent-harness" in content


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
