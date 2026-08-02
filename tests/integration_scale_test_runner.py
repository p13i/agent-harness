"""Bazel pytest runner for provider-free integration scale proofs."""

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                str(Path(__file__).resolve().parent / "test_api_integration.py"),
                "-q",
                "-m",
                "scale",
                "--override-ini",
                "markers=scale: provider-free bounded scale proof",
                *sys.argv[1:],
            ]
        )
    )
