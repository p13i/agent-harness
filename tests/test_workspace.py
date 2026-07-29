from pathlib import Path
import subprocess

from agent_harness.blobs import BlobStore
from agent_harness.workspace import checkpoint_workspace
from agent_harness.workspace import create_worktree
from agent_harness.workspace import restore_checkpoint
from test_support import session


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
