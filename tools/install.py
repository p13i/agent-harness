"""Install a stable launcher for a checked-out harness repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex


def launcher(repo: Path) -> str:
    quoted = shlex.quote(str(repo.resolve()))
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        "cd "
        + quoted
        + "\n"
        "exec npx --yes @bazel/bazelisk run "
        "--ui_event_filters=-info --noshow_progress "
        "//cmd:agent-harness -- \"$@\"\n"
    )


def install(repo: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    temporary = destination.with_name(destination.name + ".new")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
        0o755,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(launcher(repo))
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o755)
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args(argv)
    install(arguments.repo, arguments.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
