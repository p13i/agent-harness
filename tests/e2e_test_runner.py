"""Bazel pytest runner for product journeys."""

from pathlib import Path

import pytest


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                str(Path(__file__).resolve().parent / "test_e2e_journeys.py"),
                "-q",
            ]
        )
    )
