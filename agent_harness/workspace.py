"""Harness-owned Git worktrees and durable workspace checkpoints."""

from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from agent_harness.blobs import BlobStore
from agent_harness.errors import HarnessError
from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import Checkpoint, Session


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
        return _recover_created_worktree(root, destination, session_id)
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    _git(root, "worktree", "add", "--detach", str(destination), head)
    try:
        _git(
            root,
            "update-ref",
            "refs/agent-harness/" + session_id,
            head,
        )
    except BaseException:
        _git(root, "worktree", "remove", "--force", str(destination))
        raise
    return destination


def _recover_created_worktree(
    root: Path,
    destination: Path,
    session_id: str,
) -> Path:
    try:
        recovered_root = Path(
            _git(destination, "rev-parse", "--show-toplevel").stdout.strip()
        ).resolve()
        head = _git(destination, "rev-parse", "HEAD").stdout.strip()
    except (HarnessError, OSError, subprocess.SubprocessError) as error:
        raise HarnessError(
            "E_WORKTREE_EXISTS",
            "the session worktree path is not recoverable",
            status=409,
        ) from error
    if recovered_root != destination:
        raise HarnessError(
            "E_WORKTREE_EXISTS",
            "the session worktree path belongs to another workspace",
            status=409,
        )
    reference = "refs/agent-harness/" + session_id
    existing_head = ""
    try:
        existing_head = _git(
            root,
            "rev-parse",
            "--verify",
            reference,
        ).stdout.strip()
    except HarnessError:
        existing_head = ""
    if existing_head:
        if existing_head != head:
            raise HarnessError(
                "E_WORKTREE_EXISTS",
                "the recovered session reference does not match its worktree",
                status=409,
            )
    else:
        _git(root, "update-ref", reference, head)
    return destination


def remove_worktree(
    workspace: Path,
    destination: Path,
    session_id: str,
) -> None:
    root = git_root(workspace)
    resolved = destination.resolve()
    if resolved == root:
        raise ValueError("direct workspace cannot be removed as a detached worktree")
    expected_ref = "refs/agent-harness/" + session_id
    _git(root, "worktree", "remove", "--force", str(resolved))
    _git(root, "update-ref", "-d", expected_ref)


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
    diff = _git(workspace, "diff", "--no-ext-diff", "--unified=3", "HEAD").stdout
    from agent_harness.presentation import sanitize_diff

    sanitized_diff, redactions, binary, unused_files = sanitize_diff(diff)
    del unused_files
    truncated = len(sanitized_diff) > 100_000
    if truncated:
        sanitized_diff = sanitized_diff[:100_000]
    payload = {
        "commit": _git(workspace, "rev-parse", "HEAD").stdout.strip(),
        "status": status.splitlines(),
        "diff_stat": diff_stat.splitlines(),
        "diff": sanitized_diff,
        "diff_binary_omitted": binary,
        "diff_redactions": redactions,
        "diff_truncated": truncated,
    }
    return "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```"


def _untracked_archive(workspace: Path) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        capture_output=True,
        check=True,
    )
    paths = sorted(item for item in completed.stdout.split(b"\0") if item)
    tar_buffer = io.BytesIO()
    with tarfile.open(
        fileobj=tar_buffer,
        mode="w",
        format=tarfile.PAX_FORMAT,
    ) as archive:
        for raw in paths:
            relative = Path(os.fsdecode(raw))
            if relative.is_absolute() or ".." in relative.parts:
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "untracked checkpoint path escapes the workspace",
                    status=400,
                )
            candidate = workspace / relative
            if candidate.is_symlink():
                link_target = os.readlink(candidate)
                if _safe_link_target(relative.as_posix(), link_target) is None:
                    raise HarnessError(
                        "E_CHECKPOINT_UNSAFE",
                        "untracked checkpoint link escapes the workspace",
                        status=400,
                    )
                resolved_target = candidate.resolve(strict=False)
                if not resolved_target.is_relative_to(workspace.resolve()):
                    raise HarnessError(
                        "E_CHECKPOINT_UNSAFE",
                        "untracked checkpoint link escapes the workspace",
                        status=400,
                    )
                info = tarfile.TarInfo(relative.as_posix())
                info.type = tarfile.SYMTYPE
                info.linkname = link_target
                info.mode = 0o777
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                archive.addfile(info)
                continue
            target = candidate.resolve()
            if not target.is_relative_to(workspace.resolve()):
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "untracked checkpoint path escapes the workspace",
                    status=400,
                )
            if not target.is_file():
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "untracked checkpoint object is unsupported",
                    status=400,
                )
            info = archive.gettarinfo(str(target), arcname=str(relative))
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.pax_headers = {}
            if info.mode & 0o111:
                info.mode = 0o755
            else:
                info.mode = 0o644
            with target.open("rb") as source:
                archive.addfile(info, source)
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=compressed,
        mtime=0,
    ) as stream:
        stream.write(tar_buffer.getvalue())
    return compressed.getvalue()


def _extract_untracked(workspace: Path, content: bytes) -> None:
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        members = archive.getmembers()
        member_paths: dict[str, tarfile.TarInfo] = {}
        root = workspace.resolve()
        for member in members:
            member_path = _safe_archive_path(member.name)
            if member_path is None:
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint path escapes the workspace",
                    status=400,
                )
            normalized_name = member_path.as_posix()
            if normalized_name in member_paths:
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint contains a duplicate path",
                    status=400,
                )
            if not member.isreg() and not member.issym():
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint contains an unsupported object",
                    status=400,
                )
            if member.issym() and (
                _safe_link_target(normalized_name, member.linkname) is None
            ):
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint link escapes the workspace",
                    status=400,
                )
            destination = workspace / normalized_name
            resolved_parent = destination.parent.resolve(strict=False)
            if not resolved_parent.is_relative_to(root):
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint path escapes the workspace",
                    status=400,
                )
            member_paths[normalized_name] = member
        for normalized_name in member_paths:
            path = PurePosixPath(normalized_name)
            for parent in path.parents:
                if parent == PurePosixPath("."):
                    continue
                if parent.as_posix() in member_paths:
                    raise HarnessError(
                        "E_CHECKPOINT_UNSAFE",
                        "checkpoint contains a colliding path topology",
                        status=400,
                    )
        for normalized_name, member in member_paths.items():
            if not member.isreg():
                continue
            destination = workspace / normalized_name
            if destination.exists() or destination.is_symlink():
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint path collides with workspace material",
                    status=400,
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint file payload is missing",
                    status=400,
                )
            destination.write_bytes(source.read())
            destination.chmod(member.mode & 0o777)
        for normalized_name, member in member_paths.items():
            if not member.issym():
                continue
            destination = workspace / normalized_name
            if destination.exists() or destination.is_symlink():
                raise HarnessError(
                    "E_CHECKPOINT_UNSAFE",
                    "checkpoint path collides with workspace material",
                    status=400,
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(member.linkname, destination)


def _safe_archive_path(value: str) -> PurePosixPath | None:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        return None
    if any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path


def _safe_link_target(
    member_name: str,
    link_target: str,
) -> PurePosixPath | None:
    target = PurePosixPath(link_target)
    if target.is_absolute() or not target.parts:
        return None
    combined = PurePosixPath(member_name).parent / target
    parts: list[str] = []
    for part in combined.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        return None
    return PurePosixPath(*parts)


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
