"""Bazel pytest runner for pseudo-terminal tests."""

from pathlib import Path

import pytest


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                str(Path(__file__).resolve().parent / "test_chat_pty.py"),
                "-q",
            ]
        )
    )
