"""Harness-owned Git worktrees and durable workspace checkpoints."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

from agent_harness.blobs import BlobStore
from agent_harness.errors import HarnessError
from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Checkpoint
from agent_harness.models import Session


def git_root(workspace: Path) -> Path:
    completed = _git(workspace, "rev-parse", "--show-toplevel")
    return Path(completed.stdout.strip()).resolve()


def create_worktree(
    workspace: Path,
    destination_root: Path,
    session_id: str,
    *,
    direct: bool = False,
) -> Path:
    root = git_root(workspace)
    if direct:
        return root
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = (destination_root / session_id).resolve()
    if destination.exists():
        raise HarnessError(
            "E_WORKTREE_EXISTS",
            "the session worktree already exists",
            status=409,
        )
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    _git(root, "worktree", "add", "--detach", str(destination), head)
    _git(
        root,
        "update-ref",
        "refs/agent-harness/" + session_id,
        head,
    )
    return destination


def checkpoint_workspace(
    session: Session,
    blobs: BlobStore,
    *,
    sequence: int,
    provider: str,
    native_session_id: str,
    context_text: str,
) -> Checkpoint:
    workspace = Path(session.worktree)
    base_commit = _git(workspace, "rev-parse", "HEAD").stdout.strip()
    patch = _git_bytes(
        workspace,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
    )
    untracked = _untracked_archive(workspace)
    patch_digest = blobs.put(patch)
    untracked_digest = blobs.put(untracked)
    context_digest = blobs.put_text(context_text)
    return Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=session.session_id,
        sequence=sequence,
        provider=provider,
        native_session_id=native_session_id,
        base_commit=base_commit,
        patch_digest=patch_digest,
        untracked_digest=untracked_digest,
        context_digest=context_digest,
        created_at=utc_now(),
    )


def restore_checkpoint(
    workspace: Path,
    checkpoint: Checkpoint,
    blobs: BlobStore,
) -> None:
    current = _git(workspace, "rev-parse", "HEAD").stdout.strip()
    if current != checkpoint.base_commit:
        raise HarnessError(
            "E_CHECKPOINT_BASE_MISMATCH",
            "workspace commit does not match the checkpoint base",
            status=409,
        )
    patch = blobs.get(checkpoint.patch_digest)
    if patch:
        _git_input(workspace, patch, "apply", "--index", "--binary", "-")
    archive = blobs.get(checkpoint.untracked_digest)
    if archive:
        _extract_untracked(workspace, archive)


def workspace_summary(workspace: Path) -> str:
    status = _git(workspace, "status", "--short").stdout.strip()
    diff_stat = _git(workspace, "diff", "--stat", "HEAD").stdout.strip()
    payload = {
        "commit": _git(workspace, "rev-parse", "HEAD").stdout.strip(),
        "status": status.splitlines(),
        "diff_stat": diff_stat.splitlines(),
    }
    return "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```"


def _untracked_archive(workspace: Path) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    )
    paths = [item for item in completed.stdout.split(b"\0") if item]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for raw in paths:
            relative = Path(os.fsdecode(raw))
            target = (workspace / relative).resolve()
            if not target.is_relative_to(workspace.resolve()):
                continue
            if target.is_symlink():
                continue
            if not target.is_file():
                continue
            archive.add(target, arcname=str(relative), recursive=False)
    return buffer.getvalue()


def _extract_untracked(workspace: Path, content: bytes) -> None:
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint contains a link",
                    status=400,
                )
            destination = (workspace / member.name).resolve()
            if not destination.is_relative_to(workspace.resolve()):
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint path escapes the workspace",
                    status=400,
                )
        archive.extractall(workspace, filter="data")


def _git(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise HarnessError(
            "E_GIT",
            "Git operation failed",
            status=409,
        )
    return completed


def _git_bytes(workspace: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise HarnessError("E_GIT", "Git operation failed", status=409)
    return completed.stdout


def _git_input(workspace: Path, content: bytes, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        input=content,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise HarnessError(
            "E_GIT_APPLY",
            "checkpoint patch does not apply cleanly",
            status=409,
        )

