from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

import pytest

from agent_harness import service_manager as service_manager_module
from agent_harness.service_manager import SERVICE_NAME
from agent_harness.service_manager import UNIT_VERSION
from agent_harness.service_manager import ServiceManagerError
from agent_harness.service_manager import SystemdUserService
from agent_harness.service_manager import UnitConfiguration
from agent_harness.service_manager import default_unit_path
from agent_harness.service_manager import render_unit
from agent_harness.service_manager import unit_metadata
from tools import bundle as bundle_module
from tools import install as install_module
from tools.bundle import BUNDLE_MANIFEST
from tools.bundle import BUNDLE_SCHEMA
from tools.bundle import LEGACY_BUNDLE_SCHEMA
from tools.bundle import BundleError
from tools.bundle import create_bundle
from tools.bundle import main as bundle_main
from tools.bundle import safe_relative_path
from tools.bundle import validate_build_id
from tools.bundle import verify_bundle
from tools.install import InstallError
from tools.install import default_bundle_root
from tools.install import default_launcher
from tools.install import install
from tools.install import install_executable
from tools.install import launcher
from tools.install import main
from tools.install import read_selection
from tools.install import rollback
from tools.install import select_bundle


def _make_writable(path: Path) -> None:
    path.chmod((path.stat().st_mode & 0o777) | 0o200)


def _rewrite(path: Path, content: str) -> None:
    original_mode = path.stat().st_mode & 0o777
    _make_writable(path)
    path.write_text(content, encoding="utf-8")
    path.chmod(original_mode)


def test_bundle_dereferences_runfiles_and_is_content_addressed(
    tmp_path: Path,
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    runfiles = Path(str(source) + ".runfiles")
    cache = tmp_path / "cache"
    cache.mkdir()
    dependency = cache / "dependency.txt"
    dependency.write_text("cached\n", encoding="utf-8")
    (runfiles / "package" / "linked.txt").symlink_to(dependency)
    external_directory = cache / "directory"
    external_directory.mkdir()
    (external_directory / "nested.txt").write_text(
        "nested\n",
        encoding="utf-8",
    )
    (runfiles / "external").symlink_to(
        external_directory,
        target_is_directory=True,
    )

    first = create_bundle(source, tmp_path / "bundles")
    second = create_bundle(source, tmp_path / "bundles")

    assert first == second
    assert len(first.build_id) == 24
    assert first.executable.is_file()
    assert verify_bundle(first.root) == first
    copied_runfiles = Path(str(first.executable) + ".runfiles")
    assert (copied_runfiles / "package" / "linked.txt").read_text(
        encoding="utf-8"
    ) == "cached\n"
    assert (copied_runfiles / "external" / "nested.txt").read_text(
        encoding="utf-8"
    ) == "nested\n"
    assert not (copied_runfiles / "MANIFEST").exists()
    assert not any(path.is_symlink() for path in first.root.rglob("*"))
    assert not any(
        path.stat().st_mode & 0o222
        for path in (first.root, *first.root.rglob("*"))
    )
    assert json.loads(
        (first.root / BUNDLE_MANIFEST).read_text(encoding="utf-8")
    )["schema"] == BUNDLE_SCHEMA


def test_legacy_bundle_remains_verifiable_for_rollback(tmp_path: Path) -> None:
    source = _source_executable(tmp_path / "source", "one")
    bundle = create_bundle(source, tmp_path / "bundles")
    manifest = bundle.root / BUNDLE_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = LEGACY_BUNDLE_SCHEMA
    _rewrite(manifest, json.dumps(payload))
    for path in (bundle.root, *bundle.root.rglob("*")):
        if path.is_dir():
            path.chmod((path.stat().st_mode & 0o777) | 0o200)

    assert verify_bundle(bundle.root) == bundle


@pytest.mark.parametrize(
    "value",
    ["", "/absolute", "../escape", "inside/../escape", "bad\0path"],
)
def test_bundle_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(BundleError):
        safe_relative_path(value)


@pytest.mark.parametrize(
    "value",
    ["", ".hidden", "-leading", "contains/slash", "space value", "x" * 129],
)
def test_bundle_rejects_unsafe_build_identifiers(value: str) -> None:
    with pytest.raises(BundleError, match="build identifier"):
        validate_build_id(value)


def test_bundle_rejects_an_escaping_bazel_manifest(
    tmp_path: Path,
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    manifest = Path(str(source) + ".runfiles") / "MANIFEST"
    manifest.write_text("../escape /outside\n", encoding="utf-8")

    with pytest.raises(BundleError, match="escapes"):
        create_bundle(source, tmp_path / "bundles")

    assert not list((tmp_path / "bundles").glob(".bundle-*"))


def test_bundle_rejects_missing_inputs_and_link_cycles(
    tmp_path: Path,
) -> None:
    with pytest.raises(BundleError, match="executable"):
        create_bundle(tmp_path / "missing", tmp_path / "bundles")

    source = tmp_path / "source" / "agent-harness"
    source.parent.mkdir()
    source.write_text("#!/bin/sh\n", encoding="utf-8")
    source.chmod(0o755)
    with pytest.raises(BundleError, match="runfiles"):
        create_bundle(source, tmp_path / "bundles")

    runfiles = Path(str(source) + ".runfiles")
    runfiles.mkdir()
    (runfiles / "loop").symlink_to(runfiles, target_is_directory=True)
    with pytest.raises(BundleError, match="cycle"):
        create_bundle(source, tmp_path / "bundles")


def test_bundle_rejects_build_identifier_collisions(
    tmp_path: Path,
) -> None:
    first_source = _source_executable(tmp_path / "one", "one")
    second_source = _source_executable(tmp_path / "two", "two")
    root = tmp_path / "bundles"
    create_bundle(first_source, root, build_id="release")

    with pytest.raises(BundleError, match="different content"):
        create_bundle(second_source, root, build_id="release")

    assert not list(root.glob(".bundle-*"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("content", "size mismatch"),
        ("extra", "undeclared"),
        ("missing", "missing declared"),
        ("link", "symbolic link"),
        ("mode", "mode mismatch"),
        ("directory-mode", "writable directory"),
    ],
)
def test_bundle_integrity_detects_tree_mutations(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    bundle = create_bundle(source, tmp_path / "bundles")
    runfile = Path(str(bundle.executable) + ".runfiles") / "package" / "data.txt"
    if mutation == "content":
        original_mode = runfile.stat().st_mode & 0o777
        runfile.chmod(original_mode | 0o200)
        runfile.write_text("changed and longer\n", encoding="utf-8")
        runfile.chmod(original_mode)
    elif mutation == "extra":
        original_mode = bundle.root.stat().st_mode & 0o777
        bundle.root.chmod(original_mode | 0o200)
        (bundle.root / "extra").write_text("extra\n", encoding="utf-8")
        bundle.root.chmod(original_mode)
    elif mutation == "missing":
        original_mode = runfile.parent.stat().st_mode & 0o777
        runfile.parent.chmod(original_mode | 0o200)
        runfile.unlink()
        runfile.parent.chmod(original_mode)
    elif mutation == "link":
        original_mode = runfile.parent.stat().st_mode & 0o777
        runfile.parent.chmod(original_mode | 0o200)
        runfile.unlink()
        runfile.symlink_to(bundle.executable)
        runfile.parent.chmod(original_mode)
    elif mutation == "directory-mode":
        bundle.root.chmod(0o700)
    else:
        runfile.chmod(0o600)

    with pytest.raises(BundleError, match=message):
        verify_bundle(bundle.root)


def test_sealed_bundle_rejects_a_writable_manifest(tmp_path: Path) -> None:
    source = _source_executable(tmp_path / "source", "one")
    bundle = create_bundle(source, tmp_path / "bundles")
    _make_writable(bundle.root / BUNDLE_MANIFEST)

    with pytest.raises(BundleError, match="manifest is writable"):
        verify_bundle(bundle.root)


def test_bundle_integrity_rejects_invalid_manifests(
    tmp_path: Path,
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    bundle = create_bundle(source, tmp_path / "bundles")
    manifest = bundle.root / BUNDLE_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["path"] = "../escape"
    _rewrite(manifest, json.dumps(payload))
    with pytest.raises(BundleError, match="escapes"):
        verify_bundle(bundle.root)

    _rewrite(manifest, "not json")
    with pytest.raises(BundleError, match="cannot be read"):
        verify_bundle(bundle.root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "unsupported", "schema"),
        ("build_id", 1, "build identifier"),
        ("content_digest", "short", "content digest"),
        ("executable", 1, "executable path"),
        ("files", {}, "file manifest"),
    ],
)
def test_bundle_integrity_rejects_invalid_top_level_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    bundle = create_bundle(source, tmp_path / "bundles")
    manifest = bundle.root / BUNDLE_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = value
    _rewrite(manifest, json.dumps(payload))

    with pytest.raises(BundleError, match=message):
        verify_bundle(bundle.root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest-type", "must be an object"),
        ("file-digest", "digest mismatch"),
        ("aggregate-digest", "aggregate content digest"),
    ],
)
def test_bundle_integrity_rejects_tampered_digests_and_payload_type(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    bundle = create_bundle(source, tmp_path / "bundles")
    manifest = bundle.root / BUNDLE_MANIFEST
    if mutation == "manifest-type":
        _rewrite(manifest, "[]")
    elif mutation == "file-digest":
        runfile = (
            Path(str(bundle.executable) + ".runfiles")
            / "package"
            / "data.txt"
        )
        original_mode = runfile.stat().st_mode & 0o777
        _make_writable(runfile)
        runfile.write_text("two\n", encoding="utf-8")
        runfile.chmod(original_mode)
    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["content_digest"] = "0" * 64
        _rewrite(manifest, json.dumps(payload))

    with pytest.raises(BundleError, match=message):
        verify_bundle(bundle.root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("entry", "entry must be an object"),
        ("path", "file path"),
        ("duplicate", "duplicated"),
        ("digest", "file digest"),
        ("size", "file size"),
        ("mode", "file mode"),
        ("writable-mode", "file mode is writable"),
    ],
)
def test_bundle_integrity_rejects_invalid_file_entries(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    bundle = create_bundle(source, tmp_path / "bundles")
    manifest = bundle.root / BUNDLE_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entry = payload["files"][0]
    if mutation == "entry":
        payload["files"][0] = "invalid"
    elif mutation == "path":
        entry["path"] = 1
    elif mutation == "duplicate":
        payload["files"].append(dict(entry))
    elif mutation == "digest":
        entry["sha256"] = "short"
    elif mutation == "size":
        entry["size"] = True
    elif mutation == "writable-mode":
        entry["mode"] = 0o644
    else:
        entry["mode"] = True
    _rewrite(manifest, json.dumps(payload))

    with pytest.raises(BundleError, match=message):
        verify_bundle(bundle.root)


def test_bundle_integrity_rejects_root_and_executable_mutations(
    tmp_path: Path,
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    root = tmp_path / "bundles"
    bundle = create_bundle(source, root, build_id="original")

    link = tmp_path / "bundle-link"
    link.symlink_to(bundle.root, target_is_directory=True)
    with pytest.raises(BundleError, match="symbolic link"):
        verify_bundle(link)

    file_root = tmp_path / "not-a-directory"
    file_root.write_text("invalid", encoding="utf-8")
    with pytest.raises(BundleError, match="does not exist"):
        verify_bundle(file_root)

    renamed = root / "renamed"
    bundle.root.rename(renamed)
    with pytest.raises(BundleError, match="does not match"):
        verify_bundle(renamed)
    renamed.rename(bundle.root)

    bundle.executable.chmod(0o400)
    manifest = bundle.root / BUNDLE_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    executable_entry = next(
        entry
        for entry in payload["files"]
        if entry["path"] == "bin/agent-harness"
    )
    executable_entry["mode"] = 0o400
    payload["content_digest"] = bundle_module._content_digest(payload["files"])
    _rewrite(manifest, json.dumps(payload))
    with pytest.raises(BundleError, match="not executable"):
        verify_bundle(bundle.root)


def test_bundle_manifest_without_physical_targets_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    manifest = Path(str(source) + ".runfiles") / "MANIFEST"
    manifest.write_text("package/data.txt\n", encoding="utf-8")
    status = bundle_main(
        [
            "--executable",
            str(source),
            "--bundle-root",
            str(tmp_path / "bundles"),
            "--build-id",
            "cli-bundle",
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["build_id"] == "cli-bundle"


def test_bundle_filesystem_error_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runfiles = tmp_path / "runfiles"
    runfiles.mkdir()
    manifest = runfiles / "MANIFEST"
    manifest.write_text("logical target\n", encoding="utf-8")

    def unreadable(
        unused_path: Path,
        *,
        encoding: str,
    ) -> str:
        del unused_path, encoding
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(BundleError, match="cannot be read"):
        bundle_module._validate_source_manifest(runfiles)
    monkeypatch.undo()

    with pytest.raises(BundleError, match="cannot be resolved"):
        bundle_module._copy_tree(
            tmp_path / "missing",
            tmp_path / "destination",
            frozenset(),
        )

    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    (broken_root / "link").symlink_to(tmp_path / "absent")
    with pytest.raises(BundleError, match="cannot be resolved"):
        bundle_module._copy_tree(
            broken_root,
            tmp_path / "broken-copy",
            frozenset(),
        )

    special_root = tmp_path / "special"
    special_root.mkdir()
    os.mkfifo(special_root / "pipe")
    with pytest.raises(BundleError, match="unsupported runfile type"):
        bundle_module._copy_tree(
            special_root,
            tmp_path / "special-copy",
            frozenset(),
        )

    source = tmp_path / "source-file"
    source.write_text("content", encoding="utf-8")

    def copy_failure(
        unused_source: Path,
        unused_destination: Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        del unused_source, unused_destination, follow_symlinks
        raise OSError("copy failed")

    monkeypatch.setattr(bundle_module.shutil, "copyfile", copy_failure)
    with pytest.raises(BundleError, match="failed to copy"):
        bundle_module._copy_regular_file(
            source,
            tmp_path / "copied-file",
        )


def test_bundle_cleanup_unseals_a_failed_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_executable(tmp_path / "source", "one")
    bundle_root = tmp_path / "bundles"
    original_replace = Path.replace

    def fail_promotion(path: Path, target: Path) -> Path:
        if path.name.startswith(".bundle-"):
            raise OSError("promotion failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_promotion)

    with pytest.raises(OSError, match="promotion failed"):
        create_bundle(source, bundle_root)

    assert not list(bundle_root.glob(".bundle-*"))


def test_bundle_walk_rejects_special_files_and_skips_manifest(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.mkdir()
    (linked_root / "directory").symlink_to(
        external,
        target_is_directory=True,
    )
    with pytest.raises(BundleError, match="symbolic link"):
        bundle_module._walk_regular_files(linked_root)

    special_root = tmp_path / "special-root"
    special_root.mkdir()
    os.mkfifo(special_root / "pipe")
    with pytest.raises(BundleError, match="non-regular"):
        bundle_module._walk_regular_files(special_root)

    manifest_root = tmp_path / "manifest-root"
    manifest_root.mkdir()
    (manifest_root / BUNDLE_MANIFEST).write_text(
        "{}",
        encoding="utf-8",
    )
    assert bundle_module._manifest_entries(manifest_root) == []


def test_bundle_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_executable(tmp_path / "source", "entrypoint")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bundle",
            "--executable",
            str(source),
            "--bundle-root",
            str(tmp_path / "bundles"),
        ],
    )

    with pytest.raises(SystemExit, match="0"):
        runpy.run_module("tools.bundle", run_name="__main__")


def test_install_upgrade_and_rollback_retain_bundles(
    tmp_path: Path,
) -> None:
    destination = tmp_path / ".local" / "bin" / "agent-harness"
    root = tmp_path / ".local" / "lib" / "p13i-agent-harness"
    first_source = _source_executable(tmp_path / "one", "one")
    first = install_executable(
        first_source,
        root,
        destination,
        build_id="build-one",
    )
    first_source.parent.rename(tmp_path / "source-removed")

    completed = subprocess.run(
        [str(destination), "version"],
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout == "one\n"
    assert read_selection(destination).build_id == "build-one"

    second_source = _source_executable(tmp_path / "two", "two")
    second = install_executable(
        second_source,
        root,
        destination,
        build_id="build-two",
    )

    assert second.previous_build_id == "build-one"
    assert first.bundle.root.is_dir()
    assert second.bundle.root.is_dir()
    selected = rollback(root, destination)
    assert selected.build_id == "build-one"
    assert selected.previous_build_id == "build-two"
    explicit = rollback(root, destination, build_id="build-two")
    assert explicit.build_id == "build-two"


def test_failed_validation_never_switches_the_launcher(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundles"
    destination = tmp_path / "bin" / "agent-harness"
    first = create_bundle(
        _source_executable(tmp_path / "one", "one"),
        root,
        build_id="one",
    )
    select_bundle(root, first.build_id, destination)
    original = destination.read_bytes()
    second = create_bundle(
        _source_executable(tmp_path / "two", "two"),
        root,
        build_id="two",
    )

    def reject(unused: Path) -> None:
        del unused
        raise InstallError("rejected")

    with pytest.raises(InstallError, match="rejected"):
        select_bundle(
            root,
            second.build_id,
            destination,
            validator=reject,
        )

    assert destination.read_bytes() == original
    assert read_selection(destination).build_id == "one"
    assert second.root.is_dir()


def test_install_checkout_compatibility_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    source = _source_executable(
        repo / "bazel-bin" / "cmd",
        "repo",
    )
    destination = tmp_path / "prefix" / "bin" / "agent-harness"

    result = install(repo, destination, build_id="repo-build")

    assert result.bundle.root == (
        tmp_path
        / "prefix"
        / "lib"
        / "p13i-agent-harness"
        / "repo-build"
    )
    assert source.is_file()
    second_destination = tmp_path / "other" / "bin" / "agent-harness"
    status = main(
        [
            "--source-executable",
            str(source),
            "--bundle-root",
            str(tmp_path / "other" / "bundles"),
            "--destination",
            str(second_destination),
            "--build-id",
            "cli-build",
        ]
    )
    assert status == 0
    assert json.loads(capsys.readouterr().out)["build_id"] == "cli-build"
    status = main(
        [
            "--bundle-root",
            str(tmp_path / "other" / "bundles"),
            "--destination",
            str(second_destination),
            "--rollback",
            "--build-id",
            "cli-build",
        ]
    )
    assert status == 0


def test_install_cli_argument_and_repo_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    _source_executable(repo / "bazel-bin" / "cmd", "repo")
    destination = tmp_path / "bin" / "agent-harness"
    assert (
        main(
            [
                "--repo",
                str(repo),
                "--destination",
                str(destination),
                "--bundle-root",
                str(tmp_path / "bundles"),
                "--build-id",
                "repo-main",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["build_id"] == "repo-main"

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--repo",
                str(repo),
                "--source-executable",
                str(repo / "bazel-bin" / "cmd" / "agent-harness"),
            ]
        )
    with pytest.raises(SystemExit, match="2"):
        main([])

    class CyclingArguments:
        rollback = False
        repo = None
        destination = tmp_path / "unused"
        bundle_root = tmp_path / "bundles"
        build_id = ""

        def __init__(self) -> None:
            self.reads = 0

        @property
        def source_executable(self) -> Path | None:
            self.reads += 1
            if self.reads == 1:
                return tmp_path / "source"
            return None

    monkeypatch.setattr(
        install_module.argparse.ArgumentParser,
        "parse_args",
        lambda unused_self, unused_argv: CyclingArguments(),
    )
    with pytest.raises(AssertionError, match="checked above"):
        main([])


def test_install_selection_read_error_and_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "agent-harness"
    destination.write_text("managed", encoding="utf-8")

    def unreadable(
        unused_path: Path,
        *,
        encoding: str,
    ) -> str:
        del unused_path, encoding
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(InstallError, match="cannot be read"):
        read_selection(destination)
    monkeypatch.undo()

    monkeypatch.setattr(sys, "argv", ["install"])
    with pytest.raises(SystemExit, match="2"):
        runpy.run_module("tools.install", run_name="__main__")


def test_selector_contract_and_default_paths(tmp_path: Path) -> None:
    executable = tmp_path / "bundle with spaces" / "agent-harness"
    content = launcher(
        executable,
        "current",
        previous_build_id="previous",
    )
    assert content.startswith("#!/bin/sh\n")
    assert "unset RUNFILES_DIR RUNFILES_MANIFEST_FILE" in content
    assert "export PYTHONDONTWRITEBYTECODE=1" in content
    assert content.index("export PYTHONDONTWRITEBYTECODE=1") < content.index(
        "\nexec "
    )
    assert shlex_quote(str(executable)) in content
    assert default_bundle_root(tmp_path) == (
        tmp_path / ".local" / "lib" / "p13i-agent-harness"
    )
    assert default_launcher(tmp_path) == (
        tmp_path / ".local" / "bin" / "agent-harness"
    )
    assert default_unit_path(tmp_path) == (
        tmp_path
        / ".config"
        / "systemd"
        / "user"
        / SERVICE_NAME
    )


def test_service_defaults_follow_the_current_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        service_manager_module.Path,
        "home",
        lambda: tmp_path,
    )
    monkeypatch.setattr(service_manager_module.os, "getuid", lambda: 123)
    manager = SystemdUserService(
        runner=lambda command, **unused: subprocess.CompletedProcess(
            command,
            0,
            "",
            "",
        )
    )

    assert manager.unit_path == default_unit_path(tmp_path)
    assert manager.user_id == 123


def test_selector_rejects_unmanaged_and_absent_rollback(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "agent-harness"
    assert read_selection(destination) is None
    destination.write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(InstallError, match="not managed"):
        read_selection(destination)
    destination.unlink()
    with pytest.raises(InstallError, match="no installed launcher"):
        rollback(tmp_path / "bundles", destination)

    source = _source_executable(tmp_path / "source", "one")
    bundle = create_bundle(source, tmp_path / "bundles", build_id="one")
    select_bundle(tmp_path / "bundles", bundle.build_id, destination)
    with pytest.raises(InstallError, match="no prior bundle"):
        rollback(tmp_path / "bundles", destination)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "metadata is invalid"),
        ({"schema": "unsupported"}, "schema"),
        (
            {
                "schema": install_module.SELECTOR_SCHEMA,
                "build_id": 1,
                "previous_build_id": "",
                "executable": "/binary",
            },
            "build identifier",
        ),
        (
            {
                "schema": install_module.SELECTOR_SCHEMA,
                "build_id": "one",
                "previous_build_id": 1,
                "executable": "/binary",
            },
            "previous build identifier",
        ),
        (
            {
                "schema": install_module.SELECTOR_SCHEMA,
                "build_id": "one",
                "previous_build_id": "",
                "executable": "",
            },
            "executable path",
        ),
        (
            {
                "schema": install_module.SELECTOR_SCHEMA,
                "build_id": "../escape",
                "previous_build_id": "",
                "executable": "/binary",
            },
            "build identifier",
        ),
    ],
)
def test_selector_rejects_invalid_metadata(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    destination = tmp_path / "agent-harness"
    encoded = "not-json"
    if payload is not None:
        encoded = json.dumps(payload)
    destination.write_text(
        "# p13i-agent-harness-selector: " + encoded + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InstallError, match=message):
        read_selection(destination)


def test_selector_rejects_a_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("content", encoding="utf-8")
    destination = tmp_path / "agent-harness"
    destination.symlink_to(target)

    with pytest.raises(InstallError, match="regular file"):
        read_selection(destination)


def test_executable_validation_reports_launch_and_exit_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "agent-harness"
    executable.write_text("content", encoding="utf-8")

    def unavailable(*unused: object, **unused_named: object) -> object:
        del unused
        del unused_named
        raise OSError("unavailable")

    monkeypatch.setattr(install_module.subprocess, "run", unavailable)
    with pytest.raises(InstallError, match="validation failed"):
        install_module.validate_executable(executable)

    def rejected(
        *unused: object,
        **unused_named: object,
    ) -> subprocess.CompletedProcess[str]:
        del unused
        del unused_named
        return subprocess.CompletedProcess([], 7, "", "rejected")

    monkeypatch.setattr(install_module.subprocess, "run", rejected)
    with pytest.raises(InstallError, match="exit status 7"):
        install_module.validate_executable(executable)


def test_executable_validation_suppresses_bundle_bytecode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "agent-harness"
    executable.write_text("content", encoding="utf-8")
    captured_environment: dict[str, str] = {}

    def accepted(
        *unused: object,
        **named: object,
    ) -> subprocess.CompletedProcess[str]:
        del unused
        environment = named.get("env")
        assert isinstance(environment, dict)
        captured_environment.update(environment)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(install_module.subprocess, "run", accepted)

    install_module.validate_executable(executable)

    assert captured_environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_installed_python_bundle_remains_immutable_across_execution_surfaces(
    tmp_path: Path,
) -> None:
    source = _python_source_executable(tmp_path / "source")
    bundle = create_bundle(source, tmp_path / "bundles")
    unguarded_environment = os.environ.copy()
    unguarded_environment.pop("PYTHONDONTWRITEBYTECODE", None)
    unguarded_environment.pop("PYTHONPYCACHEPREFIX", None)

    install_module.validate_executable(bundle.executable)
    assert verify_bundle(bundle.root) == bundle

    selector = tmp_path / "bin" / "agent-harness"
    selector.parent.mkdir()
    selector.write_text(
        launcher(bundle.executable, bundle.build_id),
        encoding="utf-8",
    )
    selector.chmod(0o755)
    selector_result = subprocess.run(
        ["/bin/sh", str(selector), "--help"],
        env=unguarded_environment,
        check=False,
    )
    assert selector_result.returncode == 0
    assert verify_bundle(bundle.root) == bundle

    unit = render_unit(
        UnitConfiguration(
            executable=bundle.executable,
            state_dir=tmp_path / "state",
            build_id=bundle.build_id,
        )
    )
    environment_line = next(
        line for line in unit.splitlines() if line.startswith("Environment=")
    )
    environment_name, separator, environment_value = environment_line[
        len("Environment=") :
    ].partition("=")
    assert separator == "="
    service_environment = unguarded_environment.copy()
    service_environment[environment_name] = environment_value
    command_line = next(
        line for line in unit.splitlines() if line.startswith("ExecStart=")
    )
    service_result = subprocess.run(
        shlex.split(command_line[len("ExecStart=") :]),
        env=service_environment,
        check=False,
    )
    assert service_result.returncode == 0
    assert verify_bundle(bundle.root) == bundle

    if os.geteuid() != 0:
        control_result = subprocess.run(
            [str(bundle.executable), "--help"],
            env=unguarded_environment,
            check=False,
        )
        assert control_result.returncode == 0
        assert not any(bundle.root.rglob("*.pyc"))
        assert verify_bundle(bundle.root) == bundle


def test_installer_atomic_write_removes_a_failed_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bin" / "agent-harness"

    def failed_chmod(path: Path, mode: int) -> None:
        del path
        del mode
        raise OSError("chmod failed")

    monkeypatch.setattr(install_module.os, "chmod", failed_chmod)
    with pytest.raises(OSError, match="chmod failed"):
        install_module._atomic_write(destination, "content", 0o755)

    assert not destination.exists()
    assert not list(destination.parent.glob(".agent-harness.new-*"))


def test_systemd_unit_is_deterministic_private_and_socket_only(
    tmp_path: Path,
) -> None:
    configuration = UnitConfiguration(
        executable=tmp_path / "bundle % one" / "agent-harness",
        state_dir=tmp_path / 'chat "state"',
        build_id="build-one",
    )

    content = render_unit(configuration)

    assert content == render_unit(configuration)
    assert "Unit-Version: " + str(UNIT_VERSION) in content
    assert "Build-ID: build-one" in content
    assert " --state-dir " in content
    assert " service run" in content
    assert "%%" in content
    assert '\\"state\\"' in content
    assert "UMask=0077" in content
    assert "KillMode=control-group" in content
    assert "Restart=on-failure" in content
    assert "Environment=PYTHONDONTWRITEBYTECODE=1" in content
    assert content.index("[Service]\n") < content.index(
        "Environment=PYTHONDONTWRITEBYTECODE=1"
    ) < content.index("[Install]\n")
    assert "tcp" not in content.casefold()
    assert unit_metadata(content) == (UNIT_VERSION, "build-one")
    assert unit_metadata("# invalid\n") == (None, "")
    assert unit_metadata(
        "# X-P13I-Agent-Harness-Unit-Version: invalid\n"
    ) == (None, "")


@pytest.mark.parametrize(
    "configuration",
    [
        UnitConfiguration(Path("relative"), Path("/state"), "build"),
        UnitConfiguration(Path("/binary"), Path("relative"), "build"),
        UnitConfiguration(Path("/binary"), Path("/state"), "bad\nbuild"),
    ],
)
def test_systemd_unit_rejects_unsafe_configuration(
    configuration: UnitConfiguration,
) -> None:
    with pytest.raises(ValueError):
        render_unit(configuration)


def test_service_lifecycle_uses_only_user_systemd(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    active = {"value": False}

    def runner(
        command: list[str],
        **unused: object,
    ) -> subprocess.CompletedProcess[str]:
        del unused
        calls.append(command)
        if "is-active" in command:
            if active["value"]:
                return subprocess.CompletedProcess(command, 0, "active\n", "")
            return subprocess.CompletedProcess(command, 3, "inactive\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    state_dir = tmp_path / "chats"
    unit_path = tmp_path / "systemd" / SERVICE_NAME
    manager = SystemdUserService(unit_path, runner=runner, user_id=os.getuid())
    configuration = UnitConfiguration(
        executable=tmp_path / "bundle" / "agent-harness",
        state_dir=state_dir,
        build_id="one",
    )

    manager.install(configuration)
    assert unit_path.stat().st_mode & 0o077 == 0
    assert manager.status().installed
    assert not manager.status().active
    active["value"] = True
    assert manager.status().active
    manager.start()
    manager.restart()
    manager.stop()
    manager.uninstall()

    assert not unit_path.exists()
    assert not state_dir.exists()
    assert all(command[:2] == ["systemctl", "--user"] for command in calls)
    assert ["systemctl", "--user", "enable", SERVICE_NAME] in calls
    assert ["systemctl", "--user", "start", SERVICE_NAME] in calls
    assert ["systemctl", "--user", "restart", SERVICE_NAME] in calls
    assert ["systemctl", "--user", "stop", SERVICE_NAME] in calls
    assert [
        "systemctl",
        "--user",
        "disable",
        "--now",
        SERVICE_NAME,
    ] in calls


def test_service_lifecycle_supports_an_isolated_unit_name(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(
        command: list[str],
        **unused: object,
    ) -> subprocess.CompletedProcess[str]:
        del unused
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "inactive\n", "")

    service_name = "p13i-agent-harness-e2e.service"
    manager = SystemdUserService(
        tmp_path / "systemd" / service_name,
        runner=runner,
        service_name=service_name,
    )

    manager.start()
    manager.restart()
    manager.stop()
    manager.uninstall()

    assert all(
        service_name in command
        for command in calls
        if "daemon-reload" not in command
    )
    with pytest.raises(ValueError, match="service name"):
        SystemdUserService(
            tmp_path / "systemd" / "invalid",
            runner=runner,
            service_name="../invalid.service",
        )


def test_service_install_restores_prior_unit_on_reload_failure(
    tmp_path: Path,
) -> None:
    unit_path = tmp_path / "systemd" / SERVICE_NAME
    unit_path.parent.mkdir(parents=True)
    unit_path.write_text("prior\n", encoding="utf-8")
    calls = 0

    def runner(
        command: list[str],
        **unused: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        del unused
        calls += 1
        return_code = 1
        if calls > 1:
            return_code = 0
        return subprocess.CompletedProcess(command, return_code, "", "failed")

    manager = SystemdUserService(unit_path, runner=runner)
    with pytest.raises(ServiceManagerError, match="failed"):
        manager.install(
            UnitConfiguration(
                executable=tmp_path / "agent-harness",
                state_dir=tmp_path / "chats",
                build_id="one",
            )
        )
    assert unit_path.read_text(encoding="utf-8") == "prior\n"


def test_service_install_removes_a_new_unit_after_reload_failure(
    tmp_path: Path,
) -> None:
    unit_path = tmp_path / "systemd" / SERVICE_NAME
    calls = 0

    def runner(
        command: list[str],
        **unused: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        del unused
        calls += 1
        return_code = 1
        if calls > 1:
            return_code = 0
        return subprocess.CompletedProcess(command, return_code, "", "failed")

    manager = SystemdUserService(unit_path, runner=runner)
    with pytest.raises(ServiceManagerError):
        manager.install(
            UnitConfiguration(
                executable=tmp_path / "agent-harness",
                state_dir=tmp_path / "chats",
                build_id="one",
            )
        )
    assert not unit_path.exists()


def test_service_uninstall_restores_unit_on_reload_failure(
    tmp_path: Path,
) -> None:
    unit_path = tmp_path / "systemd" / SERVICE_NAME
    calls = 0
    fail_reload = {"value": False}

    def runner(
        command: list[str],
        **unused: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        del unused
        calls += 1
        if fail_reload["value"] and "daemon-reload" in command:
            return subprocess.CompletedProcess(command, 1, "", "failed")
        return subprocess.CompletedProcess(command, 0, "", "")

    manager = SystemdUserService(unit_path, runner=runner)
    manager.install(
        UnitConfiguration(
            executable=tmp_path / "agent-harness",
            state_dir=tmp_path / "chats",
            build_id="one",
        )
    )
    calls = 0
    fail_reload["value"] = True
    with pytest.raises(ServiceManagerError):
        manager.uninstall()
    assert unit_path.exists()


def test_service_diagnostics_are_read_only_and_actionable(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    lingering = {"value": "no\n"}

    def runner(
        command: list[str],
        **unused: object,
    ) -> subprocess.CompletedProcess[str]:
        del unused
        commands.append(command)
        if command[0] == "loginctl":
            return subprocess.CompletedProcess(
                command,
                0,
                lingering["value"],
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    manager = SystemdUserService(
        tmp_path / "systemd" / SERVICE_NAME,
        runner=runner,
        user_id=os.getuid(),
    )
    missing = manager.diagnostics()
    assert [probe.status for probe in missing] == [
        "pass",
        "warning",
        "warning",
    ]
    assert "enable-linger" in missing[1].remediation
    assert "service install" in missing[2].remediation
    assert commands == [
        ["systemctl", "--user", "show-environment"],
        [
            "loginctl",
            "show-user",
            str(os.getuid()),
            "--property=Linger",
            "--value",
        ],
    ]

    manager.unit_path.parent.mkdir(parents=True)
    manager.unit_path.write_text(
        render_unit(
            UnitConfiguration(
                executable=tmp_path / "agent-harness",
                state_dir=tmp_path / "chats",
                build_id="one",
            )
        ),
        encoding="utf-8",
    )
    lingering["value"] = "yes\n"
    installed = manager.diagnostics()
    assert [probe.status for probe in installed] == ["pass", "pass", "pass"]
    assert "selects build one" in installed[2].detail

    legacy = manager.unit_path.read_text(encoding="utf-8").replace(
        "Unit-Version: " + str(UNIT_VERSION),
        "Unit-Version: 1",
        1,
    )
    manager.unit_path.write_text(legacy, encoding="utf-8")
    legacy_probe = manager.diagnostics()[2]
    assert legacy_probe.status == "warning"
    assert "service install" in legacy_probe.remediation


def test_service_diagnostics_report_unavailable_dependencies(
    tmp_path: Path,
) -> None:
    def runner(
        command: list[str],
        **unused: object,
    ) -> subprocess.CompletedProcess[str]:
        del unused
        if command[0] == "systemctl":
            raise FileNotFoundError("missing")
        return subprocess.CompletedProcess(command, 1, "", "unavailable")

    manager = SystemdUserService(
        tmp_path / "systemd" / SERVICE_NAME,
        runner=runner,
    )
    probes = manager.diagnostics()
    assert [probe.status for probe in probes] == [
        "fail",
        "warning",
        "warning",
    ]
    assert "daemon-reload" in probes[0].remediation
    assert "Inspect it with" in probes[1].remediation


def test_service_status_and_unit_diagnostics_report_read_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unit_path = tmp_path / "systemd" / SERVICE_NAME
    unit_path.parent.mkdir(parents=True)
    unit_path.write_text("invalid\n", encoding="utf-8")

    def inactive(
        command: list[str],
        **unused: object,
    ) -> subprocess.CompletedProcess[str]:
        del unused
        return subprocess.CompletedProcess(command, 3, "", "inactive\n")

    manager = SystemdUserService(unit_path, runner=inactive)
    status = manager.status()
    assert status.detail == "inactive"
    assert not status.active
    assert manager.diagnostics()[2].status == "warning"

    original_read_text = service_manager_module.Path.read_text

    def unreadable(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if path == unit_path:
            raise OSError("unreadable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(
        service_manager_module.Path,
        "read_text",
        unreadable,
    )
    assert not manager.status().installed
    assert manager.diagnostics()[2].status == "fail"


def test_service_required_command_and_atomic_write_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unavailable(
        command: list[str],
        **unused: object,
    ) -> subprocess.CompletedProcess[str]:
        del command
        del unused
        raise FileNotFoundError("missing")

    manager = SystemdUserService(
        tmp_path / "systemd" / SERVICE_NAME,
        runner=unavailable,
    )
    with pytest.raises(ServiceManagerError, match="could not"):
        manager.start()

    target = tmp_path / "atomic" / "unit"
    original_chmod = service_manager_module.os.chmod

    def failed_chmod(path: Path, mode: int) -> None:
        del path
        del mode
        raise OSError("chmod failed")

    monkeypatch.setattr(
        service_manager_module.os,
        "chmod",
        failed_chmod,
    )
    with pytest.raises(OSError, match="chmod failed"):
        service_manager_module._atomic_write(target, "value", 0o600)
    monkeypatch.setattr(
        service_manager_module.os,
        "chmod",
        original_chmod,
    )
    assert not target.exists()
    assert not list(target.parent.glob(".unit.new-*"))

    with pytest.raises(ValueError, match="systemd argument"):
        service_manager_module._systemd_quote("")


def _source_executable(root: Path, identity: str) -> Path:
    root.mkdir(parents=True)
    executable = root / "agent-harness"
    executable.write_text(
        "#!/bin/sh\n"
        + "set -eu\n"
        + 'if [ "${1:-}" = "--help" ]; then\n'
        + '  echo "help"\n'
        + "  exit 0\n"
        + "fi\n"
        + "echo "
        + identity
        + "\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    runfiles = Path(str(executable) + ".runfiles")
    package = runfiles / "package"
    package.mkdir(parents=True)
    (package / "data.txt").write_text(identity + "\n", encoding="utf-8")
    (runfiles / "MANIFEST").write_text(
        "package/data.txt /cache/package/data.txt\n",
        encoding="utf-8",
    )
    return executable


def _python_source_executable(root: Path) -> Path:
    root.mkdir(parents=True)
    executable = root / "agent-harness"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        + "from pathlib import Path\n"
        + "import sys\n"
        + "runfiles = Path(str(Path(__file__)) + '.runfiles')\n"
        + "sys.path.insert(0, str(runfiles))\n"
        + "from package import probe\n"
        + "print(probe.MESSAGE)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    package = Path(str(executable) + ".runfiles") / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "probe.py").write_text(
        "MESSAGE = 'help'\n",
        encoding="utf-8",
    )
    return executable


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
