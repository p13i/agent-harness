"""Stable live workspace material inspection without state-store imports."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

from agent_harness.errors import HarnessError
from agent_harness.workspace import git_stderr_tail, workspace_summary


def inspect_workspace(workspace: Path) -> tuple[str, str]:
    root = workspace.resolve()
    digest = hashlib.sha256()
    digest.update(_git(root, "rev-parse", "HEAD"))
    digest.update(b"\0tracked\0")
    digest.update(_git(root, "diff", "--binary", "--no-ext-diff", "HEAD"))
    digest.update(b"\0untracked\0")
    untracked = _git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    for raw_path in sorted(item for item in untracked.split(b"\0") if item):
        relative = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        if relative.is_absolute() or ".." in relative.parts:
            digest.update(raw_path)
            digest.update(b"\0unsafe-path\0")
            continue
        candidate = root / relative
        if candidate.is_symlink():
            digest.update(raw_path)
            digest.update(b"\0symlink\0")
            target_text = os.readlink(candidate)
            digest.update(target_text.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            continue
        target = candidate.resolve()
        if not target.is_relative_to(root):
            digest.update(raw_path)
            digest.update(b"\0outside-workspace\0")
            continue
        if not target.is_file():
            digest.update(raw_path)
            digest.update(b"\0unsupported\0")
            try:
                object_type = stat.S_IFMT(candidate.lstat().st_mode)
            except OSError:
                object_type = 0
            digest.update(str(object_type).encode("ascii"))
            digest.update(b"\0")
            continue
        digest.update(raw_path)
        digest.update(b"\0file\0")
        executable = b"0"
        if target.stat().st_mode & 0o111:
            executable = b"1"
        digest.update(executable)
        digest.update(b"\0")
        digest.update(target.read_bytes())
        digest.update(b"\0")
    special_objects = _special_workspace_objects(root)
    if special_objects:
        digest.update(b"\0special-objects\0")
    for raw_path, object_type in special_objects:
        digest.update(raw_path)
        digest.update(b"\0unsupported\0")
        digest.update(str(object_type).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), workspace_summary(root)


def _special_workspace_objects(root: Path) -> list[tuple[bytes, int]]:
    pending = [root]
    records = []
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise HarnessError(
                "E_WORKSPACE_INSPECTION",
                "Workspace material changed during inspection",
                status=409,
            ) from error
        for entry in entries:
            if directory == root and entry.name == ".git":
                continue
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise HarnessError(
                    "E_WORKSPACE_INSPECTION",
                    "Workspace material changed during inspection",
                    status=409,
                ) from error
            if stat.S_ISDIR(mode):
                pending.append(Path(entry.path))
                continue
            if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                continue
            relative = Path(entry.path).relative_to(root)
            records.append((os.fsencode(str(relative)), stat.S_IFMT(mode)))
    return sorted(records)


def _git(workspace: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = git_stderr_tail(completed)
        message = "Git operation failed during workspace inspection"
        if detail:
            message = message + ": " + detail
        raise HarnessError(
            "E_GIT",
            message,
            status=409,
        )
    return completed.stdout
