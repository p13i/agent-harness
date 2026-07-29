"""Boundary coverage for deterministic harness foundations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from agent_harness.blobs import BlobStore
from agent_harness.config import api_token
from agent_harness.config import default_state_dir
from agent_harness.config import paths
from agent_harness.context import compile_context
from agent_harness.context import estimate_tokens
from agent_harness.errors import ConflictError
from agent_harness.errors import HarnessError
from agent_harness.errors import NeedsReconciliationError
from agent_harness.errors import NotFoundError
from agent_harness.errors import ProviderExhaustedError
from agent_harness.goals import create_goal
from agent_harness.goals import make_evidence
from agent_harness.goals import validate_budgets
from agent_harness.ids import new_uuid
from agent_harness.ids import require_identifier
from agent_harness.ids import require_uuid
from agent_harness.models import SessionEvent
from agent_harness.projections import write_session_projections
from agent_harness.transfer import MachineKeys
from agent_harness.transfer import load_machine_keys
from agent_harness.transfer import open_transfer
from agent_harness.transfer import seal_transfer
from agent_harness.workspace import create_worktree
from agent_harness.workspace import checkpoint_workspace
from agent_harness.workspace import restore_checkpoint
from test_support import session


def test_blob_store_round_trip_and_validation(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "blobs")
    digest = store.put_text("hello")

    assert store.put(b"hello") == digest
    assert store.get_text(digest) == "hello"
    assert store.path(digest).stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="SHA-256"):
        store.path("short")
    with pytest.raises(ValueError, match="hexadecimal"):
        store.path("z" * 64)
    with pytest.raises(NotFoundError):
        store.get("0" * 64)


def test_configuration_generates_and_reuses_private_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert default_state_dir() == (
        tmp_path / "xdg" / "p13i-agent-harness"
    )
    value = paths(tmp_path / "state")
    first = api_token(value)
    second = api_token(value)

    assert first == second
    assert len(first) > 40
    assert value.token.stat().st_mode & 0o077 == 0

    monkeypatch.delenv("XDG_STATE_HOME")
    assert default_state_dir().name == "p13i-agent-harness"


def test_identifier_and_error_contracts() -> None:
    identifier = new_uuid()
    assert require_uuid(identifier) == identifier
    assert require_identifier("provider:model-1") == "provider:model-1"
    with pytest.raises(ValueError, match="UUID"):
        require_uuid("not-a-uuid", "session")
    with pytest.raises(ValueError, match="invalid"):
        require_identifier("contains a space")

    assert ConflictError("busy").detail.status == 409
    assert ProviderExhaustedError("claude").detail.retryable
    assert NeedsReconciliationError().detail.status == 423


def test_context_includes_goal_evidence_and_metadata_events(
    tmp_path: Path,
) -> None:
    current = session(tmp_path)
    goal = create_goal(
        current.session_id,
        "Finish the change.",
        constraints=("Preserve compatibility.",),
        predicates=({"type": "test", "outcome": "passed"},),
    )
    evidence = make_evidence(
        goal.goal_id,
        "test",
        "unit",
        "passed",
        {"count": 1},
    )
    event = SessionEvent(
        session_id=current.session_id,
        sequence=1,
        event_id=new_uuid(),
        event_type="checkpoint.created",
        role="",
        text="",
        status="complete",
        metadata={"checkpoint_id": "checkpoint"},
        blob_digest="",
        turn_id="",
        created_at=goal.created_at,
    )
    compiled = compile_context(
        current,
        [event],
        goal=goal,
        evidence=[evidence],
        instructions=["Rule one.", "  "],
        workspace_summary="clean",
    )

    assert "# Goal" in compiled.text
    assert "# Evidence" in compiled.text
    assert "checkpoint_id" in compiled.text
    assert estimate_tokens("") == 0
    with pytest.raises(ValueError, match="reserve"):
        compile_context(
            current,
            [],
            max_input_tokens=10,
            reserve_output_tokens=10,
        )


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        ({"unknown": 1}, "unsupported"),
        ({"turns": True}, "numeric"),
        ({"turns": "one"}, "numeric"),
        ({"turns": -1}, "negative"),
    ],
)
def test_goal_budget_validation(
    budget: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_budgets(budget)


def test_goal_and_evidence_input_validation() -> None:
    session_id = new_uuid()
    with pytest.raises(ValueError, match="empty"):
        create_goal(session_id, " ")
    with pytest.raises(ValueError, match="kind"):
        create_goal(session_id, "work", kind="unknown")
    with pytest.raises(ValueError, match="type"):
        make_evidence(new_uuid(), "", "", "passed")
    with pytest.raises(ValueError, match="outcome"):
        make_evidence(new_uuid(), "test", "", "")


def test_projection_rejects_missing_session_identity(
    tmp_path: Path,
) -> None:
    current = session(tmp_path)
    context = compile_context(current, [])
    with pytest.raises(ValueError, match="no session"):
        write_session_projections(tmp_path / "one", {}, context, [], None)
    with pytest.raises(ValueError, match="identifier"):
        write_session_projections(
            tmp_path / "two",
            {"session": {}},
            context,
            [],
            None,
        )


def test_workspace_rejects_invalid_roots_and_checkpoint_bases(
    tmp_path: Path,
) -> None:
    with pytest.raises(HarnessError, match="Git"):
        create_worktree(
            tmp_path,
            tmp_path / "worktrees",
            new_uuid(),
        )

    source = tmp_path / "source"
    _repository(source)
    current = session(source)
    assert create_worktree(
        source,
        tmp_path / "unused",
        current.session_id,
        direct=True,
    ) == source.resolve()

    blobs = BlobStore(tmp_path / "blobs")
    checkpoint = checkpoint_workspace(
        current,
        blobs,
        sequence=1,
        provider="codex",
        native_session_id="native",
        context_text="context",
    )
    (source / "other.txt").write_text("other\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(source), "add", "other.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "commit", "-qm", "other"],
        check=True,
    )
    with pytest.raises(HarnessError, match="base"):
        restore_checkpoint(source, checkpoint, blobs)


@pytest.mark.parametrize(
    "envelope",
    [
        b"not-json",
        b"[]",
        b'{"schema":"wrong"}',
        b'{"schema":"p13i/agent-harness/transfer/v1"}',
    ],
)
def test_transfer_rejects_malformed_envelopes(envelope: bytes) -> None:
    destination = MachineKeys.generate()
    source = MachineKeys.generate()
    with pytest.raises(HarnessError):
        open_transfer(
            envelope,
            destination_encryption_private=destination.encryption_private,
            source_signing_public=source.public_bundle()["signing"],
        )


def test_machine_key_loader_rejects_invalid_files(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(HarnessError, match="invalid"):
        load_machine_keys(path)

    path.write_text(json.dumps({"encryption": "bad"}), encoding="utf-8")
    with pytest.raises(HarnessError, match="invalid"):
        load_machine_keys(path)


def test_transfer_rejects_wrong_destination_key() -> None:
    source = MachineKeys.generate()
    intended = MachineKeys.generate()
    wrong = MachineKeys.generate()
    envelope = seal_transfer(
        {"session_id": "session"},
        destination_encryption_public=(
            intended.public_bundle()["encryption"]
        ),
        source_signing_private=source.signing_private,
    )
    with pytest.raises(HarnessError, match="decrypt"):
        open_transfer(
            envelope,
            destination_encryption_private=wrong.encryption_private,
            source_signing_public=source.public_bundle()["signing"],
        )


def _repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
    )
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "tracked.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "initial"],
        check=True,
    )
