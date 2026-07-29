"""Bazel pytest runner for integration tests."""

from pathlib import Path

import pytest


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                str(Path(__file__).resolve().parent / "test_api_integration.py"),
                "-q",
            ]
        )
    )
