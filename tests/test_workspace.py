import json
import subprocess
from pathlib import Path

import pytest
from test_support import session

import agent_harness.workspace as workspace_module
import agent_harness.workspace_state as workspace_state_module
from agent_harness.blobs import BlobStore
from agent_harness.errors import HarnessError
from agent_harness.workspace import (
    _git,
    _git_bytes,
    _git_input,
    checkpoint_workspace,
    create_worktree,
    remove_worktree,
    restore_checkpoint,
    workspace_summary,
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


def test_git_errors_surface_redacted_stderr(tmp_path: Path) -> None:
    outside = tmp_path / "not-a-repository"
    outside.mkdir()
    with pytest.raises(HarnessError) as raised:
        _git(outside, "rev-parse", "--show-toplevel")
    assert raised.value.detail.code == "E_GIT"
    assert "fatal:" in str(raised.value)

    with pytest.raises(HarnessError) as raised_bytes:
        _git_bytes(outside, "rev-parse", "HEAD")
    assert "fatal:" in str(raised_bytes.value)

    with pytest.raises(HarnessError) as raised_state:
        workspace_state_module._git(outside, "rev-parse", "HEAD")
    assert "fatal:" in str(raised_state.value)


def test_git_stderr_tail_is_bounded_and_redacted() -> None:
    completed = subprocess.CompletedProcess(
        args=["git"],
        returncode=128,
        stderr=b"fatal: padding " + b"x" * 500 + b" token=ghp_" + b"a" * 30,
    )
    tail = workspace_module.git_stderr_tail(completed)
    assert len(tail) <= 400
    assert "ghp_" not in tail
    assert "[REDACTED]" in tail


def test_create_worktree_removes_its_worktree_when_the_ref_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    repository(source)
    current = session(source)
    original_git = workspace_module._git

    def failing_update_ref(
        workspace: Path,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "update-ref":
            raise HarnessError("E_GIT", "update-ref failed", status=409)
        return original_git(workspace, *arguments)

    monkeypatch.setattr(workspace_module, "_git", failing_update_ref)
    with pytest.raises(HarnessError, match="update-ref failed"):
        create_worktree(source, tmp_path / "worktrees", current.session_id)

    monkeypatch.undo()
    assert not (tmp_path / "worktrees" / current.session_id).exists()
    worktrees = _git(source, "worktree", "list").stdout
    assert current.session_id not in worktrees


def test_worktree_recovery_binds_its_reference_to_the_recovered_head(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    repository(source)
    current = session(source)
    destination_root = tmp_path / "worktrees"
    destination = create_worktree(source, destination_root, current.session_id)
    reference = "refs/agent-harness/" + current.session_id
    head = _git(source, "rev-parse", "HEAD").stdout.strip()
    _git(source, "update-ref", "-d", reference)

    recovered = create_worktree(source, destination_root, current.session_id)

    assert recovered == destination
    assert _git(source, "rev-parse", "--verify", reference).stdout.strip() == head

    (source / "tracked.txt").write_text("second\n", encoding="utf-8")
    git(source, "add", "tracked.txt")
    git(source, "commit", "-qm", "second")
    _git(
        source,
        "update-ref",
        reference,
        _git(source, "rev-parse", "HEAD").stdout.strip(),
    )
    with pytest.raises(HarnessError, match="does not match its worktree"):
        create_worktree(source, destination_root, current.session_id)

    remove_worktree(source, destination, current.session_id)

    assert not destination.exists()
    with pytest.raises(HarnessError):
        _git(source, "rev-parse", "--verify", reference)
    with pytest.raises(ValueError, match="direct workspace"):
        remove_worktree(source, source, current.session_id)


def test_worktree_recovery_rejects_a_foreign_workspace_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    repository(source)
    current = session(source)
    nested_root = source / "nested"
    (nested_root / current.session_id).mkdir(parents=True)

    with pytest.raises(HarnessError, match="belongs to another workspace"):
        create_worktree(source, nested_root, current.session_id)


def test_workspace_summary_truncates_an_oversized_diff(tmp_path: Path) -> None:
    source = tmp_path / "source"
    repository(source)
    (source / "tracked.txt").write_text(
        "".join("line " + str(index) + "\n" for index in range(20_000)),
        encoding="utf-8",
    )

    summary = workspace_summary(source)
    payload = json.loads(summary.removeprefix("```json\n").removesuffix("\n```"))

    assert payload["diff_truncated"] is True
    assert len(payload["diff"]) == 100_000


def test_untracked_checkpoint_rejects_links_that_resolve_outside(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    repository(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("outside\n", encoding="utf-8")
    (source / "inner").symlink_to("../outside")
    git(source, "add", "inner")
    git(source, "commit", "-qm", "inner link")
    (source / "link").symlink_to("inner/file")

    with pytest.raises(HarnessError, match="link escapes the workspace"):
        checkpoint_workspace(
            session(source),
            BlobStore(tmp_path / "blobs"),
            sequence=1,
            provider="codex",
            native_session_id="native",
            context_text="context",
        )
