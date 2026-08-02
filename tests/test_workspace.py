import subprocess
from pathlib import Path

import pytest
from test_support import session

from agent_harness.blobs import BlobStore
from agent_harness.errors import HarnessError
from agent_harness.workspace import (
    _git,
    _git_bytes,
    _git_input,
    checkpoint_workspace,
    create_worktree,
    restore_checkpoint,
)


def git(path: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
    )


def repository(path: Path) -> None:
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(path, "add", "tracked.txt")
    git(path, "commit", "-qm", "initial")


def test_checkpoint_restores_tracked_and_untracked_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    repository(source)
    current = session(source)
    (source / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (source / "new.txt").write_text("new\n", encoding="utf-8")
    (source / "new-link").symlink_to("new.txt")
    blobs = BlobStore(tmp_path / "blobs")
    checkpoint = checkpoint_workspace(
        current,
        blobs,
        sequence=1,
        provider="codex",
        native_session_id="native",
        context_text="context",
    )
    target = create_worktree(
        source,
        tmp_path / "worktrees",
        current.session_id,
    )
    restore_checkpoint(target, checkpoint, blobs)
    assert (target / "tracked.txt").read_text(encoding="utf-8") == "changed\n"
    assert (target / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert (target / "new-link").is_symlink()
    assert (target / "new-link").readlink() == Path("new.txt")


def test_workspace_rejects_collisions_mismatches_and_git_failures(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    repository(source)
    current = session(source)
    blobs = BlobStore(tmp_path / "blobs")
    checkpoint = checkpoint_workspace(
        current,
        blobs,
        sequence=1,
        provider="codex",
        native_session_id="native",
        context_text="context",
    )
    destination_root = tmp_path / "worktrees"
    destination_root.mkdir()
    (destination_root / current.session_id).mkdir()
    with pytest.raises(HarnessError):
        create_worktree(
            source,
            destination_root,
            current.session_id,
        )

    (source / "tracked.txt").write_text("next\n", encoding="utf-8")
    git(source, "add", "tracked.txt")
    git(source, "commit", "-qm", "next")
    with pytest.raises(HarnessError):
        restore_checkpoint(source, checkpoint, blobs)

    outside = tmp_path / "not-a-repository"
    outside.mkdir()
    with pytest.raises(HarnessError):
        _git(outside, "status")
    with pytest.raises(HarnessError):
        _git_bytes(outside, "diff")
    with pytest.raises(HarnessError):
        _git_input(source, b"not a patch", "apply", "-")
