"""Boundary coverage for deterministic harness foundations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_support import session

from agent_harness import config as config_module
import agent_harness.proof as proof_module
from agent_harness.blobs import BlobStore
from agent_harness.config import api_token, default_state_dir, paths, runtime_build_id
from agent_harness.context import (
    compile_context,
    estimate_tokens,
    workspace_instructions,
)
from agent_harness.errors import (
    ConflictError,
    HarnessError,
    NeedsReconciliationError,
    NotFoundError,
    ProviderExhaustedError,
)
from agent_harness.goals import create_goal, make_evidence, validate_budgets
from agent_harness.ids import new_uuid, require_identifier, require_uuid
from agent_harness.models import RestartRecovery, SessionEvent
from agent_harness.projections import write_session_projections
from agent_harness.transfer import (
    MachineKeys,
    load_machine_keys,
    open_transfer,
    seal_transfer,
)
from agent_harness.workspace import (
    checkpoint_workspace,
    create_worktree,
    restore_checkpoint,
)


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

    store.path(digest).chmod(0o600)
    store.path(digest).write_bytes(b"tampered")
    with pytest.raises(ConflictError, match="does not match its address"):
        store.get(digest)


def test_blob_store_removes_temporary_file_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = BlobStore(tmp_path / "blobs")

    def fail_replace(source: Path, destination: Path) -> None:
        del source
        del destination
        raise OSError("replace failed")

    monkeypatch.setattr("agent_harness.blobs.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.put(b"content")
    assert not list((tmp_path / "blobs").glob("blob.*"))


def test_configuration_generates_and_reuses_private_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert default_state_dir() == Path.home() / "my" / "chats"
    value = paths(tmp_path / "state")
    first = api_token(value)
    second = api_token(value)

    assert first == second
    assert len(first) > 40
    assert value.token.stat().st_mode & 0o077 == 0
    assert value.database == (tmp_path / "state" / ".runtime" / "state.sqlite3")
    assert value.sessions == tmp_path / "state" / "sessions"
    assert value.worktrees == (tmp_path / "state" / ".runtime" / "worktrees")
    assert paths().state_dir == default_state_dir().resolve()


def test_runtime_build_identity_is_read_from_verified_bundle_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "commit-sha" / "bin" / "agent-harness"
    executable.parent.mkdir(parents=True)
    outer_manifest = tmp_path / "bundle-manifest.json"
    outer_manifest.write_text(
        json.dumps(
            {
                "schema": "p13i/agent-harness/install-bundle/v1",
                "build_id": tmp_path.name,
            }
        ),
        encoding="utf-8",
    )
    assert runtime_build_id(executable) == ""
    manifest = executable.parent.parent / "bundle-manifest.json"
    manifest.symlink_to(outer_manifest)
    assert runtime_build_id(executable) == ""
    manifest.unlink()
    manifest.write_bytes(b"\xff")
    assert runtime_build_id(executable) == ""
    manifest.write_text("invalid", encoding="utf-8")
    assert runtime_build_id(executable) == ""
    manifest.write_text("[]", encoding="utf-8")
    assert runtime_build_id(executable) == ""
    manifest.write_text(
        json.dumps({"schema": "unknown", "build_id": "build-1"}),
        encoding="utf-8",
    )
    assert runtime_build_id(executable) == ""
    manifest.write_text(
        json.dumps(
            {
                "schema": "p13i/agent-harness/install-bundle/v1",
                "build_id": "different-build",
            }
        ),
        encoding="utf-8",
    )
    assert runtime_build_id(executable) == ""
    manifest.write_text(
        json.dumps(
            {
                "schema": "p13i/agent-harness/install-bundle/v1",
                "build_id": 1,
            }
        ),
        encoding="utf-8",
    )
    assert runtime_build_id(executable) == ""
    manifest.write_text(
        json.dumps(
            {
                "schema": "p13i/agent-harness/install-bundle/v1",
                "build_id": "commit-sha",
            }
        ),
        encoding="utf-8",
    )
    assert runtime_build_id(executable) == "commit-sha"

    stage_two = (
        executable.parent
        / "agent-harness.runfiles"
        / "_main"
        / "cmd"
        / "_agent-harness_stage2_bootstrap.py"
    )
    stage_two.parent.mkdir(parents=True)
    stage_two.write_text("", encoding="utf-8")
    assert runtime_build_id(stage_two) == "commit-sha"

    runfiles_config = (
        executable.parent
        / "agent-harness.runfiles"
        / "_main"
        / "agent_harness"
        / "config.py"
    )
    runfiles_config.parent.mkdir(parents=True)
    runfiles_config.write_text("", encoding="utf-8")
    monkeypatch.setattr(config_module, "__file__", str(runfiles_config))
    assert runtime_build_id() == "commit-sha"


def test_context_bounds_large_workspace_instruction(
    tmp_path: Path,
) -> None:
    content = "x" * 200_001
    (tmp_path / "AGENTS.md").write_text(content, encoding="utf-8")
    instructions = workspace_instructions(tmp_path)
    assert len(instructions) == 1
    assert len(instructions[0]) < len(content) + 20


def test_restart_recovery_serializes_empty_collections() -> None:
    assert RestartRecovery((), ()).as_dict() == {
        "requeued_command_ids": [],
        "reconciliations": [],
    }


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


def test_goal_budget_validation_accepts_turn_envelope_limits() -> None:
    assert (
        validate_budgets(
            {
                "tokens": 1,
                "context_tokens": 2,
                "output_tokens": 3,
                "tool_calls": 4,
                "attempts": 5,
                "child_agents": 6,
                "seconds": 7,
                "dollars": 8,
                "turns": 9,
            }
        )["child_agents"]
        == 6
    )


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


def test_proof_helpers_cover_fail_closed_projection_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = make_evidence(
        new_uuid(),
        "test",
        "unit",
        "passed",
        {"count": 1},
    )
    assert proof_module._child_identities({"receiverThreadIds": ["native-child"]}) == [
        ("native-child", ["native-child"])
    ]
    result = proof_module._proof_result(
        {
            "target_command_id": "target-command",
            "workspace_material_digest": "material-digest",
            "usage": {"tokens": [1, True, "redacted"]},
            "safety": {"limit": 1},
        }
    )
    assert result["target_command_id"] == "target-command"
    assert result["workspace_material_digest"] == "material-digest"
    assert result["usage"] == {"tokens": [1, True, None]}
    assert result["safety_digest"]
    assert proof_module._proof_approval({})["decision_digest"] == ""

    captured_at = "2026-07-31T12:00:00+00:00"
    assert not proof_module._usage_is_fresh("invalid", captured_at)
    assert proof_module._usage_is_fresh(
        "2026-07-31T12:00:00",
        "2026-07-31T12:01:00",
    )
    usage_rows = [
        {
            "sample_id": "sample-" + str(index),
            "provider": "provider-" + str(index),
            "observed_at": captured_at,
            "binding_percent": binding,
            "credits_engaged": credits,
            "payload_json": "{}",
        }
        for index, (binding, credits) in enumerate(
            ((None, False), (90.0, False), (10.0, True))
        )
    ]
    usage_rows.append(
        {
            **usage_rows[0],
            "sample_id": "sample-latest",
            "observed_at": "2026-07-31T12:00:01+00:00",
        }
    )
    projected = proof_module._usage_proof(
        {"tables": {"usage_samples": usage_rows}},
        set(),
        [],
        captured_at,
    )
    assert all(not item["admissible_at_90_percent"] for item in projected)

    truncated: list[str] = []
    monkeypatch.setattr(proof_module, "MAX_PROOF_RECORDS", 1)
    assert (
        len(
            proof_module._usage_proof(
                {"tables": {"usage_samples": usage_rows}},
                set(),
                truncated,
                captured_at,
            )
        )
        == 1
    )
    assert "usage" in truncated
    truncated.clear()
    assert proof_module._bounded_rows(
        {"commands": [{}, {}]},
        "commands",
        truncated,
    ) == [{}]
    assert truncated == ["commands"]
    with pytest.raises(ValueError, match="after_sequence exceeds"):
        proof_module._proof_page(
            {
                "payload": {"events": []},
                "through_sequence": 0,
                "digest": proof_module._digest({"events": []}),
            },
            after_sequence=1,
            event_limit=1,
        )
    with pytest.raises(ValueError, match="not contiguous"):
        proof_module._proof_page(
            {
                "payload": {"events": []},
                "through_sequence": 1,
                "digest": proof_module._digest({"events": []}),
            },
            after_sequence=0,
            event_limit=1,
        )
    with pytest.raises(ValueError, match="not contiguous"):
        proof_module._proof_page(
            {
                "payload": {"events": [{"sequence": 2}]},
                "through_sequence": 1,
                "digest": proof_module._digest({"events": [{"sequence": 2}]}),
            },
            after_sequence=0,
            event_limit=1,
        )

    assert proof_module._turn_commands(
        [{"command_id": "command", "result_json": '{"turn_id":"turn"}'}],
        [],
    ) == {"turn": "command"}
    assert proof_module._rows({"rows": "invalid"}, "rows") == []
    assert proof_module._object([]) == {}
    assert proof_module._json_object("[]") == {}
    assert proof_module._json_list("{}") == []
    assert proof_module._json_value(7) == 7
    assert proof_module._json_value("invalid") is None
    assert proof_module._predicate_fields(({"type": "test"},)) == {}

    assert not proof_module._evidence_matches({"type": "other"}, evidence)
    assert not proof_module._evidence_matches(
        {"type": "test", "subject": "other"}, evidence
    )
    assert not proof_module._evidence_matches(
        {"type": "test", "outcome": "failed"}, evidence
    )
    assert proof_module._evidence_matches({"type": "test"}, evidence)
    assert proof_module._evidence_matches(
        {"type": "test", "field": "count", "equals": 1},
        evidence,
    )
    assert proof_module._formal_value(None)["type"] == "null"
    assert proof_module._formal_value(True)["type"] == "boolean"
    assert proof_module._formal_value(1)["type"] == "integer"
    assert proof_module._formal_value(1.5)["type"] == "number"
    assert proof_module._formal_value("secret")["digest"]
    assert proof_module._formal_value({"value": 1})["type"] == "json"
    assert proof_module._strings("invalid") == []
    assert proof_module._numeric_tree(False) is False
    assert proof_module._numeric_tree([1, "secret"]) == [1, None]
    assert proof_module._numeric_tree("secret") is None
    assert proof_module._integer(True) == 0
    assert proof_module._integer(3) == 3
    assert proof_module._integer("3") == 0
    assert proof_module._optional_boolean(False) is False
    assert proof_module._optional_boolean(1) is True
    assert proof_module._optional_boolean("true") is None


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
    assert (
        create_worktree(
            source,
            tmp_path / "unused",
            current.session_id,
            direct=True,
        )
        == source.resolve()
    )

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
        destination_encryption_public=(intended.public_bundle()["encryption"]),
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
