"""Enforce layered line-coverage thresholds from an LCOV report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


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


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcov", type=Path, required=True)
    parser.add_argument("--minimum", type=float, required=True)
    parser.add_argument("--group", action="append", default=[])
    values = parser.parse_args(arguments)

    files = tuple(
        item
        for item in read_lcov(values.lcov)
        if item.path.startswith("agent_harness/")
    )
    failures: list[str] = []
    overall = percentage(files)
    print("overall: " + _result(overall, values.minimum))
    if overall < values.minimum:
        failures.append("overall")

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


def _result(measured: float, minimum: float) -> str:
    return (
        f"{measured:.2f}% "
        + "(minimum "
        + f"{minimum:.2f}"
        + "%)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
