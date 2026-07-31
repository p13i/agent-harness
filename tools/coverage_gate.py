"""Enforce complete product line coverage from an LCOV report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath


@dataclass(frozen=True)
class FileCoverage:
    path: str
    found: int
    hit: int

    @property
    def percent(self) -> float:
        if self.found == 0:
            return 100.0
        return self.hit * 100.0 / self.found


@dataclass(frozen=True)
class CoverageGroup:
    name: str
    minimum: float
    paths: frozenset[str]


def read_lcov(path: Path) -> tuple[FileCoverage, ...]:
    files: list[FileCoverage] = []
    source = ""
    found = 0
    hit = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("SF:"):
            source = line[3:]
            found = 0
            hit = 0
            continue
        if line.startswith("LF:"):
            found = int(line[3:])
            continue
        if line.startswith("LH:"):
            hit = int(line[3:])
            continue
        if line == "end_of_record" and source:
            files.append(FileCoverage(source, found, hit))
            source = ""
    return tuple(files)


def parse_group(value: str) -> CoverageGroup:
    name_and_minimum, separator, paths = value.partition(":")
    if not separator:
        raise ValueError(
            "coverage group must use NAME=MINIMUM:path,path"
        )
    name, equals, minimum = name_and_minimum.partition("=")
    if not equals or not name or not minimum:
        raise ValueError(
            "coverage group must use NAME=MINIMUM:path,path"
        )
    selected = frozenset(
        item.strip() for item in paths.split(",") if item.strip()
    )
    if not selected:
        raise ValueError("coverage group must name at least one path")
    return CoverageGroup(name, float(minimum), selected)


def percentage(files: tuple[FileCoverage, ...]) -> float:
    found = sum(item.found for item in files)
    hit = sum(item.hit for item in files)
    if found == 0:
        return 0.0
    return hit * 100.0 / found


def select(
    files: tuple[FileCoverage, ...],
    paths: frozenset[str],
) -> tuple[FileCoverage, ...]:
    return tuple(item for item in files if item.path in paths)


def discover_product_files(source_root: Path) -> frozenset[str]:
    discovered: set[str] = set()
    for directory in ("agent_harness", "cmd", "tools"):
        root = source_root / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            relative = path.relative_to(source_root).as_posix()
            if _is_product_file(relative):
                discovered.add(relative)
    return frozenset(discovered)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcov", type=Path, required=True)
    parser.add_argument("--minimum", type=float, required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--per-file-minimum", type=float)
    parser.add_argument("--source-root", type=Path)
    values = parser.parse_args(arguments)

    all_files = tuple(
        item
        for item in read_lcov(values.lcov)
        if _is_product_file(item.path)
    )
    exclusions = frozenset(values.exclude)
    files = tuple(
        item for item in all_files if item.path not in exclusions
    )
    failures: list[str] = []
    missing_exclusions = exclusions - {
        item.path for item in all_files
    }
    if missing_exclusions:
        names = ", ".join(sorted(missing_exclusions))
        print("exclusions: missing " + names)
        failures.append("exclusions")
    elif exclusions:
        print("exclusions: " + ", ".join(sorted(exclusions)))

    if values.source_root is not None:
        expected = discover_product_files(values.source_root)
        expected = expected - exclusions
        reported = {item.path for item in files}
        missing_files = expected - reported
        unexpected_files = reported - expected
        if missing_files:
            names = ", ".join(sorted(missing_files))
            print("source discovery: missing " + names)
            failures.append("source discovery")
        if unexpected_files:
            names = ", ".join(sorted(unexpected_files))
            print("source discovery: unexpected " + names)
            failures.append("source discovery")
        if not missing_files and not unexpected_files:
            print(
                "source discovery: "
                + str(len(expected))
                + " production files"
            )

    overall = percentage(files)
    print("overall: " + _result(overall, values.minimum))
    if overall < values.minimum:
        failures.append("overall")

    if values.per_file_minimum is not None:
        per_file_failures = tuple(
            item
            for item in files
            if item.percent < values.per_file_minimum
        )
        for item in sorted(
            per_file_failures,
            key=lambda coverage: coverage.path,
        ):
            print(
                item.path
                + ": "
                + _result(item.percent, values.per_file_minimum)
            )
        if per_file_failures:
            failures.append("per-file")
        else:
            print(
                "per-file: "
                + str(len(files))
                + " files at "
                + f"{values.per_file_minimum:.2f}%"
            )

    for raw_group in values.group:
        group = parse_group(raw_group)
        chosen = select(files, group.paths)
        missing = group.paths - {item.path for item in chosen}
        if missing:
            names = ", ".join(sorted(missing))
            print(group.name + ": missing " + names)
            failures.append(group.name)
            continue
        measured = percentage(chosen)
        print(group.name + ": " + _result(measured, group.minimum))
        if measured < group.minimum:
            failures.append(group.name)

    if failures:
        print("coverage gate failed: " + ", ".join(failures))
        return 1
    return 0


def _is_product_file(path: str) -> bool:
    pure_path = PurePosixPath(path)
    if pure_path.suffix != ".py":
        return False
    name = pure_path.name
    if name == "conftest.py" or name.startswith("test_"):
        return False
    if name.endswith("_test.py") or name.endswith("_test.gpt.py"):
        return False
    if pure_path.parts[0] in {"agent_harness", "cmd"}:
        return True
    if pure_path.parts[0] != "tools":
        return False
    return True


def _result(measured: float, minimum: float) -> str:
    return (
        f"{measured:.2f}% "
        + "(minimum "
        + f"{minimum:.2f}"
        + "%)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
