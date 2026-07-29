from pathlib import Path

import pytest

from tools.coverage_gate import main
from tools.coverage_gate import parse_group
from tools.coverage_gate import read_lcov


def report(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "SF:agent_harness/core.py",
                "LF:100",
                "LH:98",
                "end_of_record",
                "SF:agent_harness/adapter.py",
                "LF:100",
                "LH:62",
                "end_of_record",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_coverage_gate_enforces_overall_and_group_thresholds(
    tmp_path: Path,
) -> None:
    lcov = tmp_path / "coverage.dat"
    report(lcov)

    assert len(read_lcov(lcov)) == 2
    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "80",
                "--group",
                "core=95:agent_harness/core.py",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "81",
                "--group",
                "core=99:agent_harness/core.py",
            ]
        )
        == 1
    )


def test_coverage_gate_rejects_invalid_or_missing_groups(
    tmp_path: Path,
) -> None:
    lcov = tmp_path / "coverage.dat"
    report(lcov)

    with pytest.raises(ValueError, match="NAME"):
        parse_group("invalid")
    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "0",
                "--group",
                "missing=1:agent_harness/missing.py",
            ]
        )
        == 1
    )
