"""Create and verify self-contained Bazel executable bundles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid


BUNDLE_SCHEMA = "p13i/agent-harness/install-bundle/v1"
BUNDLE_MANIFEST = "bundle-manifest.json"
EXECUTABLE_RELATIVE_PATH = Path("bin") / "agent-harness"
_BUILD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class BundleError(RuntimeError):
    """Raised when a bundle cannot be built or trusted."""


@dataclass(frozen=True)
class Bundle:
    build_id: str
    root: Path
    executable: Path
    content_digest: str


def safe_relative_path(value: str) -> Path:
    """Return a normalized bundle-relative path or reject it."""
    if not value or "\0" in value:
        raise BundleError("bundle path is empty or contains a null byte")
    path = Path(value)
    if path.is_absolute():
        raise BundleError("bundle path must be relative: " + value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise BundleError("bundle path escapes its root: " + value)
    return path


def validate_build_id(build_id: str) -> str:
    """Validate a build identifier used as one directory component."""
    if not _BUILD_ID.fullmatch(build_id):
        raise BundleError("invalid bundle build identifier")
    return build_id


def create_bundle(
    source_executable: Path,
    bundle_root: Path,
    *,
    build_id: str = "",
) -> Bundle:
    """Copy a Bazel executable and its runfiles into a versioned bundle."""
    source_executable = source_executable.expanduser().resolve()
    source_runfiles = Path(str(source_executable) + ".runfiles")
    if not source_executable.is_file():
        raise BundleError(
            "Bazel executable does not exist: " + str(source_executable)
        )
    if not source_runfiles.is_dir():
        raise BundleError(
            "Bazel runfiles directory does not exist: "
            + str(source_runfiles)
        )
    _validate_source_manifest(source_runfiles)

    bundle_root = bundle_root.expanduser().resolve()
    bundle_root.mkdir(parents=True, exist_ok=True, mode=0o755)
    staging = bundle_root / (".bundle-" + uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    try:
        executable = staging / EXECUTABLE_RELATIVE_PATH
        executable.parent.mkdir(parents=True, mode=0o755)
        _copy_regular_file(source_executable, executable)
        os.chmod(executable, executable.stat().st_mode | 0o111)

        destination_runfiles = Path(str(executable) + ".runfiles")
        _copy_tree(source_runfiles, destination_runfiles, frozenset())
        source_manifest = destination_runfiles / "MANIFEST"
        if source_manifest.exists():
            source_manifest.unlink()

        files = _manifest_entries(staging)
        content_digest = _content_digest(files)
        selected_build_id = build_id
        if not selected_build_id:
            selected_build_id = content_digest[:24]
        validate_build_id(selected_build_id)
        payload = {
            "schema": BUNDLE_SCHEMA,
            "build_id": selected_build_id,
            "content_digest": content_digest,
            "executable": EXECUTABLE_RELATIVE_PATH.as_posix(),
            "files": files,
        }
        _write_manifest(staging / BUNDLE_MANIFEST, payload)

        destination = bundle_root / selected_build_id
        if destination.exists():
            existing = verify_bundle(destination)
            if existing.content_digest != content_digest:
                raise BundleError(
                    "bundle build identifier already names different content"
                )
            shutil.rmtree(staging)
            return existing
        staging.replace(destination)
        _sync_directory(bundle_root)
        return verify_bundle(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_bundle(bundle_root: Path) -> Bundle:
    """Verify every installed file and reject links or undeclared content."""
    bundle_root = bundle_root.expanduser()
    if bundle_root.is_symlink():
        raise BundleError("bundle root must not be a symbolic link")
    bundle_root = bundle_root.resolve()
    if not bundle_root.is_dir():
        raise BundleError("bundle directory does not exist")
    manifest_path = bundle_root / BUNDLE_MANIFEST
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError("bundle manifest cannot be read") from error
    if not isinstance(payload, dict):
        raise BundleError("bundle manifest must be an object")
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise BundleError("bundle manifest schema is unsupported")

    build_id = payload.get("build_id")
    content_digest = payload.get("content_digest")
    executable_value = payload.get("executable")
    files = payload.get("files")
    if not isinstance(build_id, str):
        raise BundleError("bundle manifest build identifier is invalid")
    validate_build_id(build_id)
    if bundle_root.name != build_id:
        raise BundleError("bundle directory does not match its build identifier")
    if not isinstance(content_digest, str) or len(content_digest) != 64:
        raise BundleError("bundle content digest is invalid")
    if not isinstance(executable_value, str):
        raise BundleError("bundle executable path is invalid")
    executable_relative = safe_relative_path(executable_value)
    if not isinstance(files, list):
        raise BundleError("bundle file manifest is invalid")

    declared: dict[str, dict[str, object]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise BundleError("bundle file entry must be an object")
        relative_value = item.get("path")
        digest = item.get("sha256")
        size = item.get("size")
        mode = item.get("mode")
        if not isinstance(relative_value, str):
            raise BundleError("bundle file path is invalid")
        relative = safe_relative_path(relative_value)
        normalized = relative.as_posix()
        if normalized in declared:
            raise BundleError("bundle file path is duplicated: " + normalized)
        if not isinstance(digest, str) or len(digest) != 64:
            raise BundleError("bundle file digest is invalid: " + normalized)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BundleError("bundle file size is invalid: " + normalized)
        if not isinstance(mode, int) or isinstance(mode, bool):
            raise BundleError("bundle file mode is invalid: " + normalized)
        declared[normalized] = item

    actual: set[str] = set()
    for path in _walk_regular_files(bundle_root):
        relative = path.relative_to(bundle_root).as_posix()
        if relative == BUNDLE_MANIFEST:
            continue
        actual.add(relative)
        item = declared.get(relative)
        if item is None:
            raise BundleError("bundle contains undeclared file: " + relative)
        if path.stat().st_size != item["size"]:
            raise BundleError("bundle file size mismatch: " + relative)
        if stat.S_IMODE(path.stat().st_mode) != item["mode"]:
            raise BundleError("bundle file mode mismatch: " + relative)
        if _sha256(path) != item["sha256"]:
            raise BundleError("bundle file digest mismatch: " + relative)
    missing = set(declared) - actual
    if missing:
        raise BundleError(
            "bundle is missing declared file: " + sorted(missing)[0]
        )
    normalized_files = [declared[name] for name in sorted(declared)]
    if _content_digest(normalized_files) != content_digest:
        raise BundleError("bundle aggregate content digest is invalid")

    executable = bundle_root / executable_relative
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise BundleError("bundle executable is absent or not executable")
    return Bundle(build_id, bundle_root, executable, content_digest)


def _validate_source_manifest(runfiles: Path) -> None:
    manifest = runfiles / "MANIFEST"
    if not manifest.exists():
        return
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BundleError("Bazel runfiles manifest cannot be read") from error
    for line in lines:
        logical, separator, unused_target = line.partition(" ")
        del unused_target
        if not separator:
            logical = line
        safe_relative_path(logical)


def _copy_tree(
    source: Path,
    destination: Path,
    ancestors: frozenset[tuple[int, int]],
) -> None:
    try:
        source_stat = source.stat()
    except OSError as error:
        raise BundleError(
            "runfiles link cannot be resolved: " + str(source)
        ) from error
    identity = (source_stat.st_dev, source_stat.st_ino)
    if identity in ancestors:
        raise BundleError("symbolic-link cycle in Bazel runfiles")
    destination.mkdir(mode=stat.S_IMODE(source_stat.st_mode) | 0o700)
    nested_ancestors = ancestors | {identity}
    for child in sorted(source.iterdir(), key=lambda item: item.name):
        target = destination / child.name
        try:
            followed = child.stat()
        except OSError as error:
            raise BundleError(
                "runfiles link cannot be resolved: " + str(child)
            ) from error
        if stat.S_ISDIR(followed.st_mode):
            _copy_tree(child, target, nested_ancestors)
            continue
        if stat.S_ISREG(followed.st_mode):
            _copy_regular_file(child, target)
            continue
        raise BundleError(
            "unsupported runfile type: "
            + str(child.relative_to(source))
        )


def _copy_regular_file(source: Path, destination: Path) -> None:
    try:
        shutil.copyfile(source, destination, follow_symlinks=True)
        shutil.copystat(source, destination, follow_symlinks=True)
    except OSError as error:
        raise BundleError("failed to copy runfile: " + str(source)) from error


def _walk_regular_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise BundleError("bundle contains a symbolic link")
        for name in names:
            path = current_path / name
            if path.is_symlink():
                raise BundleError("bundle contains a symbolic link")
            mode = path.stat().st_mode
            if not stat.S_ISREG(mode):
                raise BundleError("bundle contains a non-regular file")
            files.append(path)
    return tuple(sorted(files))


def _manifest_entries(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in _walk_regular_files(root):
        relative = path.relative_to(root).as_posix()
        if relative == BUNDLE_MANIFEST:
            continue
        path_stat = path.stat()
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size": path_stat.st_size,
                "mode": stat.S_IMODE(path_stat.st_mode),
            }
        )
    entries.sort(key=lambda item: str(item["path"]))
    return entries


def _content_digest(files: list[dict[str, object]]) -> str:
    normalized = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o644,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--build-id", default="")
    values = parser.parse_args(arguments)
    bundle = create_bundle(
        values.executable,
        values.bundle_root,
        build_id=values.build_id,
    )
    print(
        json.dumps(
            {
                "build_id": bundle.build_id,
                "bundle": str(bundle.root),
                "executable": str(bundle.executable),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
