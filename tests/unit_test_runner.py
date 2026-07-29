"""Bazel pytest runner for unit tests."""

from pathlib import Path

import pytest


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    raise SystemExit(
        pytest.main(
            [
                str(directory),
                "-q",
                "--ignore=" + str(directory / "test_api_integration.py"),
                "--ignore=" + str(directory / "parity_test.py"),
                "--ignore=" + str(directory / "style_test.py"),
            ]
        )
    )
