from pathlib import Path
import runpy
import sys

import pytest

from tools.coverage_gate import main
from tools.coverage_gate import discover_product_files
from tools.coverage_gate import FileCoverage
from tools.coverage_gate import percentage
from tools.coverage_gate import parse_group
from tools.coverage_gate import read_lcov
from tools.coverage_gate import _is_product_file


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
                "SF:tools/bundle.py",
                "LF:100",
                "LH:100",
                "end_of_record",
                "SF:tools/unrelated_test.py",
                "LF:100",
                "LH:0",
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

    assert len(read_lcov(lcov)) == 4
    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "80",
                "--group",
                "core=95:agent_harness/core.py",
                "--group",
                "bundle=95:tools/bundle.py",
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


def test_coverage_gate_excludes_declared_process_boundaries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lcov = tmp_path / "coverage.dat"
    report(lcov)

    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "98",
                "--exclude",
                "agent_harness/adapter.py",
            ]
        )
        == 0
    )
    assert "exclusions: agent_harness/adapter.py" in (
        capsys.readouterr().out
    )
    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "0",
                "--exclude",
                "agent_harness/missing.py",
            ]
        )
        == 1
    )


def test_coverage_gate_requires_every_discovered_product_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "agent_harness").mkdir()
    (tmp_path / "cmd").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "agent_harness" / "core.py").write_text("")
    (tmp_path / "cmd" / "launcher.py").write_text("")
    (tmp_path / "tools" / "worker.gpt.py").write_text("")
    (tmp_path / "tools" / "worker_test.py").write_text("")
    lcov = tmp_path / "coverage.dat"
    lcov.write_text(
        "\n".join(
            [
                "SF:agent_harness/core.py",
                "LF:1",
                "LH:1",
                "end_of_record",
                "SF:cmd/launcher.py",
                "LF:1",
                "LH:1",
                "end_of_record",
            ]
        )
    )

    assert discover_product_files(tmp_path) == frozenset(
        {
            "agent_harness/core.py",
            "cmd/launcher.py",
            "tools/worker.gpt.py",
        }
    )
    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "100",
                "--per-file-minimum",
                "100",
                "--source-root",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert "source discovery: missing tools/worker.gpt.py" in (
        capsys.readouterr().out
    )


def test_coverage_gate_enforces_threshold_for_each_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lcov = tmp_path / "coverage.dat"
    report(lcov)

    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "0",
                "--per-file-minimum",
                "99",
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "agent_harness/core.py: 98.00%" in output
    assert "agent_harness/adapter.py: 62.00%" in output


def test_coverage_gate_helper_boundaries(tmp_path: Path) -> None:
    assert FileCoverage("empty.py", 0, 0).percent == 100
    assert percentage(()) == 0
    assert not _is_product_file("notes.txt")
    assert not _is_product_file("examples/sample.py")
    for value in ("=1:path.py", "name=:path.py", "name=1:"):
        with pytest.raises(ValueError):
            parse_group(value)

    root = tmp_path / "source"
    root.mkdir()
    (root / "tools").mkdir()
    (root / "tools" / "conftest.py").write_text("")
    (root / "tools" / "test_value.py").write_text("")
    (root / "tools" / "value_test.gpt.py").write_text("")
    assert discover_product_files(root) == frozenset()

    lcov = tmp_path / "coverage.dat"
    lcov.write_text(
        "\n".join(
            [
                "SF:tools/unexpected.py",
                "LF:1",
                "LH:1",
                "end_of_record",
            ]
        )
    )
    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "101",
                "--source-root",
                str(root),
            ]
        )
        == 1
    )

    product = root / "tools" / "worker.gpt.py"
    product.write_text("value = 1\n")
    lcov.write_text(
        "\n".join(
            [
                "SF:tools/worker.gpt.py",
                "LF:1",
                "LH:1",
                "end_of_record",
            ]
        )
    )
    assert (
        main(
            [
                "--lcov",
                str(lcov),
                "--minimum",
                "100",
                "--per-file-minimum",
                "100",
                "--source-root",
                str(root),
            ]
        )
        == 0
    )


def test_coverage_gate_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lcov = tmp_path / "coverage.dat"
    lcov.write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage-gate",
            "--lcov",
            str(lcov),
            "--minimum",
            "0",
        ],
    )
    with pytest.raises(SystemExit, match="0"):
        runpy.run_module("tools.coverage_gate", run_name="__main__")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
