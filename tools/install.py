"""Install and atomically select a self-contained agent-harness bundle."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import uuid

from tools.bundle import Bundle
from tools.bundle import BundleError
from tools.bundle import create_bundle
from tools.bundle import validate_build_id
from tools.bundle import verify_bundle


SELECTOR_SCHEMA = "p13i/agent-harness/bundle-selector/v1"


class InstallError(RuntimeError):
    """Raised when bundle validation or atomic selection fails."""


@dataclass(frozen=True)
class Selection:
    build_id: str
    previous_build_id: str
    executable: Path


@dataclass(frozen=True)
class InstallResult:
    bundle: Bundle
    launcher: Path
    previous_build_id: str


Validator = Callable[[Path], None]


def default_bundle_root(home: Path | None = None) -> Path:
    """Return the versioned per-user installation root."""
    selected_home = home
    if selected_home is None:
        selected_home = Path.home()
    return selected_home / ".local" / "lib" / "p13i-agent-harness"


def default_launcher(home: Path | None = None) -> Path:
    """Return the stable per-user command path."""
    selected_home = home
    if selected_home is None:
        selected_home = Path.home()
    return selected_home / ".local" / "bin" / "agent-harness"


def launcher(
    executable: Path,
    build_id: str,
    *,
    previous_build_id: str = "",
) -> str:
    """Render a stable selector that executes one verified bundle."""
    validate_build_id(build_id)
    if previous_build_id:
        validate_build_id(previous_build_id)
    payload = json.dumps(
        {
            "schema": SELECTOR_SCHEMA,
            "build_id": build_id,
            "previous_build_id": previous_build_id,
            "executable": str(executable),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "#!/bin/sh\n"
        "# p13i-agent-harness-selector: "
        + payload
        + "\n"
        "set -eu\n"
        "unset RUNFILES_DIR RUNFILES_MANIFEST_FILE\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        "exec "
        + shlex.quote(str(executable))
        + ' "$@"\n'
    )


def read_selection(destination: Path) -> Selection | None:
    """Read selector metadata without executing the launcher."""
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_file():
        raise InstallError("installed launcher is not a regular file")
    try:
        lines = destination.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise InstallError("installed launcher cannot be read") from error
    prefix = "# p13i-agent-harness-selector: "
    metadata = [line[len(prefix) :] for line in lines if line.startswith(prefix)]
    if len(metadata) != 1:
        raise InstallError("installed launcher is not managed by this installer")
    try:
        payload = json.loads(metadata[0])
    except json.JSONDecodeError as error:
        raise InstallError("installed launcher metadata is invalid") from error
    if not isinstance(payload, dict) or payload.get("schema") != SELECTOR_SCHEMA:
        raise InstallError("installed launcher schema is unsupported")
    build_id = payload.get("build_id")
    previous_build_id = payload.get("previous_build_id")
    executable = payload.get("executable")
    if not isinstance(build_id, str):
        raise InstallError("installed launcher build identifier is invalid")
    if not isinstance(previous_build_id, str):
        raise InstallError("installed previous build identifier is invalid")
    if not isinstance(executable, str) or not executable:
        raise InstallError("installed executable path is invalid")
    try:
        validate_build_id(build_id)
        if previous_build_id:
            validate_build_id(previous_build_id)
    except BundleError as error:
        raise InstallError(str(error)) from error
    return Selection(build_id, previous_build_id, Path(executable))


def validate_executable(executable: Path) -> None:
    """Run the bundle's local help command before selecting it."""
    environment = os.environ.copy()
    environment.pop("RUNFILES_DIR", None)
    environment.pop("RUNFILES_MANIFEST_FILE", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [str(executable), "--help"],
            cwd=executable.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise InstallError("bundle executable validation failed") from error
    if result.returncode != 0:
        raise InstallError(
            "bundle executable validation failed with exit status "
            + str(result.returncode)
        )


def select_bundle(
    bundle_root: Path,
    build_id: str,
    destination: Path,
    *,
    validator: Validator = validate_executable,
) -> Selection:
    """Verify and atomically select an already installed bundle."""
    validate_build_id(build_id)
    bundle = verify_bundle(bundle_root.expanduser().resolve() / build_id)
    validator(bundle.executable)
    current = read_selection(destination)
    previous_build_id = ""
    if current is not None and current.build_id != build_id:
        previous_build_id = current.build_id
    if current is not None and current.build_id == build_id:
        previous_build_id = current.previous_build_id
    content = launcher(
        bundle.executable,
        bundle.build_id,
        previous_build_id=previous_build_id,
    )
    _atomic_write(destination, content, 0o755)
    return Selection(
        build_id=bundle.build_id,
        previous_build_id=previous_build_id,
        executable=bundle.executable,
    )


def rollback(
    bundle_root: Path,
    destination: Path,
    *,
    build_id: str = "",
    validator: Validator = validate_executable,
) -> Selection:
    """Select an explicit bundle or the prior bundle recorded in the launcher."""
    current = read_selection(destination)
    if current is None:
        raise InstallError("no installed launcher is available for rollback")
    selected_build_id = build_id
    if not selected_build_id:
        selected_build_id = current.previous_build_id
    if not selected_build_id:
        raise InstallError("the installed launcher has no prior bundle")
    return select_bundle(
        bundle_root,
        selected_build_id,
        destination,
        validator=validator,
    )


def install(
    repo: Path,
    destination: Path,
    *,
    bundle_root: Path | None = None,
    build_id: str = "",
    validator: Validator = validate_executable,
) -> InstallResult:
    """Bundle a checkout's Bazel binary and select it atomically."""
    source = repo.expanduser().resolve() / "bazel-bin" / "cmd" / "agent-harness"
    selected_root = bundle_root
    if selected_root is None:
        selected_root = (
            destination.expanduser().resolve().parent.parent
            / "lib"
            / "p13i-agent-harness"
        )
    bundle = create_bundle(source, selected_root, build_id=build_id)
    selection = select_bundle(
        selected_root,
        bundle.build_id,
        destination,
        validator=validator,
    )
    return InstallResult(
        bundle=bundle,
        launcher=destination,
        previous_build_id=selection.previous_build_id,
    )


def install_executable(
    source_executable: Path,
    bundle_root: Path,
    destination: Path,
    *,
    build_id: str = "",
    validator: Validator = validate_executable,
) -> InstallResult:
    """Bundle a specific Bazel executable and select it atomically."""
    bundle = create_bundle(
        source_executable,
        bundle_root,
        build_id=build_id,
    )
    selection = select_bundle(
        bundle_root,
        bundle.build_id,
        destination,
        validator=validator,
    )
    return InstallResult(
        bundle=bundle,
        launcher=destination,
        previous_build_id=selection.previous_build_id,
    )


def _atomic_write(destination: Path, content: str, mode: int) -> None:
    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    temporary = destination.with_name(
        "." + destination.name + ".new-" + uuid.uuid4().hex
    )
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        mode,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        temporary.replace(destination)
        _sync_directory(destination.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--source-executable", type=Path)
    parser.add_argument("--destination", type=Path, default=default_launcher())
    parser.add_argument("--bundle-root", type=Path, default=default_bundle_root())
    parser.add_argument("--build-id", default="")
    parser.add_argument("--rollback", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.rollback:
        selection = rollback(
            arguments.bundle_root,
            arguments.destination,
            build_id=arguments.build_id,
        )
        print(
            json.dumps(
                {
                    "build_id": selection.build_id,
                    "launcher": str(arguments.destination),
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.repo is not None and arguments.source_executable is not None:
        parser.error("--repo and --source-executable are mutually exclusive")
    if arguments.repo is None and arguments.source_executable is None:
        parser.error("one of --repo or --source-executable is required")
    if arguments.repo is not None:
        result = install(
            arguments.repo,
            arguments.destination,
            bundle_root=arguments.bundle_root,
            build_id=arguments.build_id,
        )
    else:
        if arguments.source_executable is None:
            raise AssertionError("source executable was checked above")
        result = install_executable(
            arguments.source_executable,
            arguments.bundle_root,
            arguments.destination,
            build_id=arguments.build_id,
        )
    print(
        json.dumps(
            {
                "build_id": result.bundle.build_id,
                "bundle": str(result.bundle.root),
                "launcher": str(result.launcher),
                "previous_build_id": result.previous_build_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
