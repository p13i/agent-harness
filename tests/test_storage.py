import copy
import datetime
import json
import sqlite3
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_support import session

import agent_harness.migration as migration_module
import agent_harness.proof as proof_module
import agent_harness.records as records_module
import agent_harness.storage as storage_module
import agent_harness.sync as sync_module
from agent_harness import child_gate
from agent_harness.config import paths, prepare_paths
from agent_harness.errors import ConflictError, NotFoundError
from agent_harness.goals import create_goal, make_evidence
from agent_harness.ids import new_uuid, utc_now
from agent_harness.migration import migrate_state
from agent_harness.models import Checkpoint, CommandStatus, ProviderAttempt
from agent_harness.orchestration import (
    creation_digest,
    normalize_creation_input,
    normalize_external_ref,
    normalize_turn_ref,
    normalized_digest,
)
from agent_harness.proof import proof_snapshot
from agent_harness.records import load_portable_records
from agent_harness.storage import StateStore
from agent_harness.sync import (
    publish_all,
    publish_session,
    read_sync_status,
    sync_repository,
)
from agent_harness.workspace import create_worktree


def test_proof_snapshots_retain_bound_ids_and_fail_closed_at_quota(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    snapshots = []
    for sequence in range(storage_module.PROOF_SNAPSHOT_MAX_PER_SESSION):
        snapshots.append(
            store.create_proof_snapshot(
                created.session_id,
                sequence,
                {"sequence": sequence},
                "digest-" + str(sequence),
            )
        )

    first = snapshots[0]
    assert store.proof_snapshot(first["snapshot_id"])["payload"] == {"sequence": 0}
    with pytest.raises(ConflictError, match="retention quota"):
        store.create_proof_snapshot(
            created.session_id,
            129,
            {"sequence": 129},
            "digest-129",
        )

    aggregate_age = datetime.datetime.now(datetime.UTC)
    aggregate_age -= datetime.timedelta(hours=168)
    with store.transaction() as connection:
        connection.execute(
            "UPDATE proof_snapshots SET created_at = ? WHERE snapshot_id = ?",
            (aggregate_age.isoformat(), first["snapshot_id"]),
        )
    with pytest.raises(ConflictError, match="retention quota"):
        store.create_proof_snapshot(
            created.session_id,
            130,
            {"sequence": 130},
            "digest-130",
        )
    expired = datetime.datetime.now(datetime.UTC)
    expired -= datetime.timedelta(hours=337)
    with store.transaction() as connection:
        connection.execute(
            "UPDATE proof_snapshots SET created_at = ? WHERE snapshot_id = ?",
            (expired.isoformat(), first["snapshot_id"]),
        )
    replacement = store.create_proof_snapshot(
        created.session_id,
        131,
        {"sequence": 131},
        "digest-131",
    )
    assert store.proof_snapshot(replacement["snapshot_id"])["payload"] == {
        "sequence": 131
    }
    with pytest.raises(NotFoundError):
        store.proof_snapshot(first["snapshot_id"])
    store.close()


def test_proof_source_is_atomic_across_command_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    queued = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "bounded proof"},
        "proof-command",
    )
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    store.append_event(
        created.session_id,
        "turn.started",
        status="running",
    )

    source_blocked = threading.Event()
    release_source = threading.Event()
    original_portable_session = store.portable_session

    def blocking_portable_session(
        session_id: str,
        *,
        include_events: bool = True,
    ) -> dict[str, object]:
        source_blocked.set()
        assert release_source.wait(timeout=2)
        return original_portable_session(
            session_id,
            include_events=include_events,
        )

    monkeypatch.setattr(store, "portable_session", blocking_portable_session)
    first: dict[str, object] = {}

    def capture() -> None:
        first.update(proof_snapshot(store, created.session_id))

    proof_thread = threading.Thread(target=capture)
    proof_thread.start()
    assert source_blocked.wait(timeout=2)

    def complete() -> None:
        store.resolve_command(
            queued.command_id,
            CommandStatus.COMPLETE,
            {"status": "complete"},
        )

    completion_thread = threading.Thread(target=complete)
    completion_thread.start()
    completion_thread.join(timeout=0.05)
    assert completion_thread.is_alive()
    release_source.set()
    proof_thread.join(timeout=2)
    completion_thread.join(timeout=2)
    assert not proof_thread.is_alive()
    assert not completion_thread.is_alive()

    first_commands = first["commands"]
    assert isinstance(first_commands, list)
    assert first_commands[0]["status"] == CommandStatus.DISPATCHING
    second = proof_snapshot(store, created.session_id)
    assert second["commands"][0]["status"] == CommandStatus.COMPLETE
    with pytest.raises(ValueError, match="after_sequence"):
        proof_snapshot(store, created.session_id, after_sequence=-1)
    with pytest.raises(ValueError, match="through_sequence"):
        proof_snapshot(store, created.session_id, through_sequence=-1)
    with pytest.raises(ValueError, match="exceeds"):
        proof_snapshot(store, created.session_id, through_sequence=99)
    with pytest.raises(ValueError, match="does not match"):
        proof_snapshot(
            store,
            created.session_id,
            through_sequence=99,
            snapshot_id=str(second["snapshot_id"]),
        )
    other = session(tmp_path)
    store.create_session(other)
    with pytest.raises(ValueError, match="another session"):
        proof_snapshot(
            store,
            other.session_id,
            snapshot_id=str(second["snapshot_id"]),
        )
    with pytest.raises(ValueError, match="positive"):
        store.proof_event_rows(created.session_id, 1, 0)
    with pytest.raises(ValueError, match="positive"):
        store.proof_source(created.session_id, None, 0)
    assert store.completed_command_results(created.session_id) == [
        {"status": "complete"}
    ]

    store.append_event(
        created.session_id,
        "agent.child.started",
        status="running",
        metadata={"child_id": "child-one"},
    )
    store.create_process_lease(
        created.session_id,
        "codex",
        "unattended",
        "2026-08-01T00:00:00+00:00",
    )
    store.register_worker(created.session_id, 123, "worker-one")
    monkeypatch.setattr(proof_module, "MAX_PROOF_RECORDS", 0)
    bounded = proof_snapshot(store, created.session_id)
    assert {"children", "leases", "workers"}.issubset(bounded["truncated"])
    store.close()


def test_proof_fails_closed_on_event_sequence_gap(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    store.append_event(created.session_id, "event.one")
    store.append_event(created.session_id, "event.two")
    with store.transaction() as connection:
        connection.execute(
            """
            DELETE FROM events WHERE session_id = ? AND sequence = 1
            """,
            (created.session_id,),
        )

    with pytest.raises(ValueError, match="not contiguous"):
        proof_snapshot(store, created.session_id)

    store.close()


def test_thousand_transition_ledger_is_complete_and_policy_storage_is_linear(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    transitions = [
        {
            "sequence": sequence,
            "next_turn_ref": {
                "step_id": "tick-" + str(sequence),
                "agent_role": "sre",
            },
            "next_command_digest": normalized_digest({"sequence": sequence}),
        }
        for sequence in range(1, 1_001)
    ]
    policy = {
        "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
        "session_id": created.session_id,
        "external_ref": {
            "orchestrator": "p13i/machines/cs-sre",
            "job_id": "thousand-ticks",
        },
        "epoch_id": "thousand-tick-epoch",
        "allowed_agent_roles": ["sre"],
        "allowed_step_prefixes": ["tick-"],
        "max_transitions": 1_000,
        "transitions": transitions,
    }
    policy_sha256 = normalized_digest(policy)
    goal = create_goal(
        created.session_id,
        "Keep the invariant healthy for one thousand ticks.",
        kind="invariant",
        constraints=(
            "dispatch-generation-transition-policy-sha256:" + policy_sha256,
            "dispatch-generation-transition-epoch:thousand-tick-epoch",
        ),
    )
    store.create_goal(goal)
    policy_json = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    now = utc_now()
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO dispatch_transition_policies VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                policy_sha256,
                created.session_id,
                goal.goal_id,
                "thousand-tick-epoch",
                policy["schema"],
                policy_json,
                now,
            ),
        )
        for sequence, transition in enumerate(transitions, start=1):
            invalidation_id = new_uuid()
            prior_command_id = new_uuid()
            prior_checkpoint_id = new_uuid()
            policy_ref = {
                "policy_sha256": policy_sha256,
                "session_id": created.session_id,
                "goal_id": goal.goal_id,
                "epoch_id": "thousand-tick-epoch",
            }
            receipt = {
                "session_id": created.session_id,
                "external_ref": policy["external_ref"],
                "goal_id": goal.goal_id,
                "prior_command_id": prior_command_id,
                "prior_command_type": "message",
                "prior_anchor_kind": "provider-result",
                "prior_reconciliation_id": "",
                "prior_reconciliation_resolution": "",
                "prior_checkpoint_id": prior_checkpoint_id,
                "prior_generation_digest": normalized_digest({"generation": sequence}),
                "prior_material_digest": normalized_digest({"material": sequence}),
                "next_turn_ref": transition["next_turn_ref"],
                "transition_sequence": sequence,
                "epoch_id": "thousand-tick-epoch",
                "policy_sha256": policy_sha256,
                "next_command_digest": transition["next_command_digest"],
            }
            stored_authorization = {
                "schema": (
                    "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
                ),
                **receipt,
                "reason": "Advance the bounded invariant tick.",
                "external_orchestrator": "p13i/machines/cs-sre",
                "external_job_id": "thousand-ticks",
                "policy_ref": policy_ref,
                "receipt": receipt,
                "receipt_sha256": normalized_digest(receipt),
            }
            request_authorization = stored_authorization
            if sequence == 1:
                request_authorization = dict(stored_authorization)
                request_authorization.pop("policy_ref")
                request_authorization["policy"] = policy
            authorization_digest = normalized_digest(request_authorization)
            request_digest = normalized_digest(
                {"sequence": sequence, "authorization": request_authorization}
            )
            connection.execute(
                "INSERT INTO dispatch_invalidations VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    invalidation_id,
                    created.session_id,
                    "Advance the bounded invariant tick.",
                    authorization_digest,
                    request_digest,
                    "tick-" + str(sequence),
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO authorization_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    authorization_digest,
                    created.session_id,
                    "dispatch-invalidation",
                    invalidation_id,
                    stored_authorization["schema"],
                    stored_authorization["receipt_sha256"],
                    json.dumps(
                        stored_authorization,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO dispatch_transition_ledger VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    invalidation_id,
                    created.session_id,
                    goal.goal_id,
                    "thousand-tick-epoch",
                    sequence,
                    policy_sha256,
                    authorization_digest,
                    stored_authorization["receipt_sha256"],
                    request_digest,
                    prior_command_id,
                    "message",
                    "provider-result",
                    "",
                    "",
                    prior_checkpoint_id,
                    receipt["prior_generation_digest"],
                    receipt["prior_material_digest"],
                    json.dumps(
                        transition["next_turn_ref"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    transition["next_command_digest"],
                    "authorized",
                    "",
                    "",
                    now,
                    now,
                ),
            )
    proof = proof_snapshot(store, created.session_id)
    ledger = proof["dispatch_transition_ledger"]
    assert proof["complete"] is True
    assert proof["truncated"] == []
    assert proof["authorization_receipts"] == []
    assert proof["dispatch_invalidations"] == []
    assert ledger["complete"] is True
    assert ledger["policy_count"] == 1
    assert ledger["receipt_count"] == 1_000
    assert len(ledger["receipts"]) == 1_000
    with store._lock:
        receipt_rows = store._connection.execute(
            "SELECT payload_json FROM authorization_receipts"
        ).fetchall()
    assert (
        sum(str(row["payload_json"]).count('"transitions"') for row in receipt_rows)
        == 0
    )
    assert sum(len(str(row["payload_json"])) for row in receipt_rows) < 3_000_000
    tampered_policy = copy.deepcopy(policy)
    tampered_policy["transitions"][499]["next_command_digest"] = "bad"
    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE dispatch_transition_policies SET payload_json = ?
            WHERE policy_sha256 = ?
            """,
            (
                json.dumps(
                    tampered_policy,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                policy_sha256,
            ),
        )
    tampered_proof = proof_snapshot(store, created.session_id)
    assert tampered_proof["complete"] is False
    assert "dispatch_transition_ledger" in tampered_proof["truncated"]
    assert tampered_proof["dispatch_transition_ledger"]["complete"] is False
    store.close()


def test_migration_copy_helpers_cover_conflicts_and_links(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    migration_module._copy_tree_verified(source, destination)

    (source / "nested").mkdir(parents=True)
    (source / "nested" / "one.txt").write_text("one")
    migration_module._copy_tree_verified(source, destination)
    assert (destination / "nested" / "one.txt").read_text() == "one"
    migration_module._copy_tree_verified(source, destination)

    (source / "nested" / "one.txt").write_text("changed")
    with pytest.raises(RuntimeError, match="conflicts"):
        migration_module._copy_tree_verified(source, destination)
    (source / "nested" / "one.txt").write_text("one")

    linked = source / "linked"
    linked.symlink_to(source / "nested", target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        migration_module._copy_tree_verified(source, destination)

    missing = tmp_path / "missing"
    target = tmp_path / "target" / "value"
    migration_module._copy_file_if_missing(missing, target)
    source_file = tmp_path / "source-file"
    source_file.write_text("value")
    migration_module._copy_file_if_missing(source_file, target)
    assert target.read_text() == "value"
    migration_module._copy_file_if_missing(source_file, target)


def test_migration_inventory_validation_boundaries(
    tmp_path: Path,
) -> None:
    worktrees = tmp_path / "worktrees"
    worktrees.mkdir()
    base = {
        "sessions": 0,
        "events": 0,
        "session_values": [],
        "blob_hashes": {},
    }
    migration_module._verify_merged_inventory(
        base,
        base,
        base,
        worktrees,
        worktrees,
    )

    cases = [
        ({**base, "sessions": 1}, base, base, "session count"),
        ({**base, "events": 1}, base, base, "event count"),
        ({**base, "session_values": {}}, base, base, "source inventory"),
        (base, {**base, "session_values": {}}, base, "destination inventory"),
        (base, base, {**base, "session_values": {}}, "merged inventory"),
        (
            {
                **base,
                "sessions": 1,
                "session_values": [{"session_id": "one"}],
            },
            base,
            {**base, "sessions": 1, "session_values": []},
            "session content",
        ),
        (base, base, {**base, "blob_hashes": []}, "merged blob"),
        ({**base, "blob_hashes": []}, base, base, "source blob"),
        (
            {**base, "blob_hashes": {"one": "digest"}},
            base,
            base,
            "blob content",
        ),
    ]
    for source, before, after, message in cases:
        with pytest.raises(RuntimeError, match=message):
            migration_module._verify_merged_inventory(
                source,
                before,
                after,
                worktrees,
                worktrees,
            )


def test_migration_source_and_preserved_session_validation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    class Store:
        values: dict[str, dict[str, object]] = {}

        def portable_session(self, session_id: str) -> dict[str, object]:
            return self.values[session_id]

    store = Store()
    malformed = [
        {"session_id": "one", "tables": []},
        {"session_id": "one", "tables": {"sessions": []}},
        {"session_id": "one", "tables": {"sessions": ["invalid"]}},
    ]
    for record in malformed:
        with pytest.raises(RuntimeError, match="portable source"):
            migration_module._verify_source_sessions(
                store,  # type: ignore[arg-type]
                [record],
                source,
                destination,
            )

    record = {
        "session_id": "one",
        "tables": {
            "sessions": [
                {
                    "session_id": "one",
                    "worktree": str(source / "one"),
                }
            ]
        },
    }
    expected = copy.deepcopy(record)
    expected["tables"]["sessions"][0]["worktree"] = str(destination / "one")
    store.values["one"] = expected
    migration_module._verify_source_sessions(
        store,  # type: ignore[arg-type]
        [record],
        source,
        destination,
    )
    store.values["one"] = {}
    with pytest.raises(RuntimeError, match="changed"):
        migration_module._verify_source_sessions(
            store,  # type: ignore[arg-type]
            [record],
            source,
            destination,
        )
    with pytest.raises(RuntimeError, match="destination session"):
        migration_module._verify_preserved_sessions(
            store,  # type: ignore[arg-type]
            {"one": {"expected": True}},
        )


def test_migration_process_and_pid_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "legacy"
    completed = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "invalid\n"
            "abc " + str(root) + " agent-harness daemon\n"
            "12 unrelated\n"
            "13 " + str(root) + " unrelated\n"
            "14 " + str(root) + " agent-harness daemon\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(migration_module, "_run", lambda *args, **kwargs: completed)
    assert migration_module._legacy_processes(root) == {14}
    assert migration_module._is_managed_legacy_process(14, root)
    completed.returncode = 1
    assert not migration_module._is_managed_legacy_process(14, root)

    pid_path = tmp_path / "pid"
    assert migration_module._read_pid(pid_path) is None
    pid_path.write_text("invalid")
    assert migration_module._read_pid(pid_path) is None
    pid_path.write_text("1")
    assert migration_module._read_pid(pid_path) is None
    pid_path.write_text("42")
    assert migration_module._read_pid(pid_path) == 42


def test_migration_hash_trash_and_quiescence_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    assert migration_module._file_hashes(root) == {}
    root.mkdir()
    (root / "file").write_text("content")
    (root / "link").symlink_to(root / "file")
    hashes = migration_module._file_hashes(root)
    assert set(hashes) == {"file"}
    assert migration_module._file_digest(root / "file") == hashes["file"]

    monkeypatch.setattr(
        migration_module,
        "_legacy_processes",
        lambda value: {42},
    )
    monkeypatch.setattr(
        migration_module,
        "_is_managed_legacy_process",
        lambda pid, value: True,
    )
    with pytest.raises(RuntimeError, match="still in use"):
        migration_module._require_root_quiescent(root)

    trash = tmp_path / ".Trash"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        migration_module.time,
        "strftime",
        lambda value: "stamp",
    )
    source = tmp_path / "legacy-source"
    source.mkdir()
    destination = migration_module._trash_source(source)
    assert destination == trash / "p13i-agent-harness-stamp"
    duplicate = tmp_path / "legacy-source-two"
    duplicate.mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        migration_module._trash_source(duplicate)


def test_migration_stop_processes_and_root_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    pid_path = root / "daemon.pid"
    pid_path.write_text("43")
    monkeypatch.setattr(
        migration_module,
        "_legacy_processes",
        lambda value: {41, 42},
    )
    checks = {41: False, 42: True, 43: True}

    def managed(pid: int, value: Path) -> bool:
        del value
        result = checks.get(pid, False)
        checks[pid] = False
        return result

    killed: list[int] = []

    def kill(pid: int, sig: int) -> None:
        del sig
        if pid == 43:
            raise ProcessLookupError
        killed.append(pid)

    monkeypatch.setattr(
        migration_module,
        "_is_managed_legacy_process",
        managed,
    )
    monkeypatch.setattr(migration_module.os, "kill", kill)
    migration_module._stop_managed_processes(root, pid_path)
    assert killed == [42]

    monkeypatch.setattr(
        migration_module,
        "_legacy_processes",
        lambda value: {42},
    )
    monkeypatch.setattr(
        migration_module,
        "_is_managed_legacy_process",
        lambda pid, value: True,
    )
    times = iter([0.0, 0.0, 16.0])
    monkeypatch.setattr(
        migration_module.time,
        "monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr(migration_module.time, "sleep", lambda value: None)
    with pytest.raises(RuntimeError, match="did not stop"):
        migration_module._stop_managed_processes(
            root,
            tmp_path / "missing-pid",
        )

    source = migration_module.legacy_paths(root)
    destination = paths(tmp_path / "missing-destination")
    (root / "state.sqlite3").write_text("")
    with pytest.raises(ValueError, match="does not exist"):
        migration_module._validate_roots(source, destination)


def test_migration_backup_and_worktree_failure_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = paths(tmp_path / "destination")
    destination.runtime.mkdir(parents=True)
    backup_root = destination.runtime / "migration-rollback"
    backup_root.mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        migration_module._create_destination_backup(destination)

    source = migration_module.legacy_paths(tmp_path / "legacy")
    source.worktrees.mkdir(parents=True)
    destination.worktrees.mkdir(parents=True)

    class Store:
        current: list[object] = []

        def list_sessions(self, *, include_archived: bool) -> list[object]:
            assert include_archived
            return self.current

    store = Store()
    outside = tmp_path / "outside" / "one"
    store.current = [
        SimpleNamespace(
            worktree=str(outside),
            workspace=str(tmp_path),
            session_id="outside",
        )
    ]
    moved: list[tuple[Path, Path, Path]] = []
    migration_module._move_worktrees(
        store,  # type: ignore[arg-type]
        source,
        destination,
        moved,
    )
    assert not moved

    missing = source.worktrees / "missing"
    store.current = [
        SimpleNamespace(
            worktree=str(missing),
            workspace=str(tmp_path),
            session_id="missing",
        )
    ]
    with pytest.raises(RuntimeError, match="missing"):
        migration_module._move_worktrees(
            store,  # type: ignore[arg-type]
            source,
            destination,
            moved,
        )

    current = source.worktrees / "occupied"
    current.mkdir()
    (destination.worktrees / "occupied").mkdir()
    store.current = [
        SimpleNamespace(
            worktree=str(current),
            workspace=str(tmp_path),
            session_id="occupied",
        )
    ]
    with pytest.raises(RuntimeError, match="already exists"):
        migration_module._move_worktrees(
            store,  # type: ignore[arg-type]
            source,
            destination,
            moved,
        )

    failing = source.worktrees / "repair"
    failing.mkdir()
    store.current = [
        SimpleNamespace(
            worktree=str(failing),
            workspace=str(tmp_path),
            session_id="repair",
        )
    ]
    monkeypatch.setattr(
        migration_module,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="repair failed"):
        migration_module._move_worktrees(
            store,  # type: ignore[arg-type]
            source,
            destination,
            moved,
        )
    assert failing.is_dir()

    original = source.worktrees / "rollback"
    relocated = destination.worktrees / "rollback"
    original.mkdir()
    migration_module._rollback_worktrees([(original, relocated, tmp_path)])


def test_migration_portable_round_trip_detects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = paths(tmp_path / "destination")
    destination.runtime.mkdir(parents=True)
    verification = destination.runtime / "portable-verify.sqlite3"
    verification.write_text("stale")
    session_value = SimpleNamespace(session_id="session-1")

    class SourceStore:
        global_value: dict[str, object] = {}

        def list_sessions(self, *, include_archived: bool) -> list[object]:
            assert include_archived
            return [session_value]

        def portable_session(self, session_id: str) -> dict[str, object]:
            del session_id
            return {"source": True}

        def portable_global(self) -> dict[str, object]:
            return self.global_value

    class RestoredStore:
        global_value: dict[str, object] = {}

        def __init__(self, path: Path) -> None:
            self.path = path

        def import_portable(
            self,
            records: object,
            global_record: object,
        ) -> None:
            del records
            del global_record

        def portable_session(self, session_id: str) -> dict[str, object]:
            del session_id
            return {}

        def portable_global(self) -> dict[str, object]:
            return self.global_value

        def close(self) -> None:
            return

    monkeypatch.setattr(
        migration_module,
        "load_portable_records",
        lambda value: ([], {}),
    )
    monkeypatch.setattr(migration_module, "StateStore", RestoredStore)
    with pytest.raises(RuntimeError, match="round trip changed session"):
        migration_module._verify_portable_round_trip(
            destination,
            SourceStore(),  # type: ignore[arg-type]
        )

    RestoredStore.portable_session = lambda self, session_id: {"source": True}
    SourceStore.global_value = {"source": True}
    with pytest.raises(RuntimeError, match="global state"):
        migration_module._verify_portable_round_trip(
            destination,
            SourceStore(),  # type: ignore[arg-type]
        )


def test_store_round_trip_and_idempotent_command(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    event = store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="continue",
    )
    assert event.sequence == 1
    first = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "continue"},
        "same-key",
    )
    second, was_created = store.ensure_command(
        created.session_id,
        "message",
        {"text": "continue"},
        "same-key",
    )
    assert first == second
    assert not was_created
    with pytest.raises(ConflictError):
        store.enqueue_command(
            created.session_id,
            "message",
            {"text": "different"},
            "same-key",
        )
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    assert claimed.status == CommandStatus.DISPATCHING
    assert store.recover_dispatching(created.session_id) == 0
    recovered = store.get_command(first.command_id)
    assert recovered.status == CommandStatus.QUEUED
    store.close()


def test_store_missing_state_paths_fail_closed(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        assert store.session_safety("missing") == {
            "session_id": "missing",
            "profile": "",
            "xhigh_authorizations": 0,
            "extensions": {},
            "created_at": "",
            "updated_at": "",
        }
        with pytest.raises(ConflictError, match="not claimed"):
            store.extend_session_safety("missing", {"value": 1})
        with pytest.raises(ConflictError, match="not claimed"):
            store.consume_session_extensions("missing")
        with pytest.raises(NotFoundError):
            store.command_envelope("missing")
        with pytest.raises(NotFoundError):
            store.update_command_envelope(
                "missing",
                state="running",
            )
        with pytest.raises(NotFoundError):
            store.checkpoint("missing")
        with pytest.raises(NotFoundError):
            store.complete_dispatch("missing", "complete")

        assert storage_module._object_or_empty("invalid") == {}
        assert storage_module._objects("invalid") == []
        with pytest.raises(ValueError, match="object"):
            storage_module._require_object("invalid", "value")
        with pytest.raises(ValueError, match="stored JSON"):
            storage_module._load_object("[]")
    finally:
        store.close()


def test_xhigh_authorization_parks_and_requeues_one_exact_command(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {
            "text": "one exact attempt",
            "provider": "codex",
            "effort": "xhigh",
        },
        "xhigh-command",
    )
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    assert store.xhigh_authorization_or_park(command.command_id) is None
    assert store.get_command(command.command_id).status == (
        CommandStatus.AWAITING_XHIGH_AUTHORIZATION
    )

    with pytest.raises(ConflictError, match="provider changed"):
        store.create_xhigh_authorization(
            created.session_id,
            command.command_id,
            "claude",
            authorization_request_digest="a" * 64,
            idempotency_key="wrong-provider",
            expires_at="2099-01-01T00:00:00+00:00",
        )

    authorization = store.create_xhigh_authorization(
        created.session_id,
        command.command_id,
        "codex",
        authorization_request_digest="b" * 64,
        idempotency_key="exact-authorization",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    assert store.get_command(command.command_id).status == CommandStatus.QUEUED
    assert store.xhigh_authorization_or_park(command.command_id) == authorization
    replay = store.create_xhigh_authorization(
        created.session_id,
        command.command_id,
        "codex",
        authorization_request_digest="b" * 64,
        idempotency_key="exact-authorization",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    assert replay == authorization
    with pytest.raises(ConflictError, match="already has"):
        store.create_xhigh_authorization(
            created.session_id,
            command.command_id,
            "codex",
            authorization_request_digest="c" * 64,
            idempotency_key="second-authorization",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    store.close()


def test_schema_v4_migrates_external_and_turn_columns(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_meta(version INTEGER NOT NULL);
        INSERT INTO schema_meta VALUES (2);
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            workspace TEXT NOT NULL,
            worktree TEXT NOT NULL,
            lifecycle TEXT NOT NULL,
            attention TEXT NOT NULL,
            permission_mode TEXT NOT NULL,
            active_provider TEXT NOT NULL,
            model TEXT NOT NULL,
            effort TEXT NOT NULL,
            goal_id TEXT NOT NULL,
            owner_host TEXT NOT NULL,
            owner_epoch INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE turns (
            turn_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            status TEXT NOT NULL,
            replay_of TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL
        );
        CREATE TABLE commands (
            idempotency_key TEXT PRIMARY KEY,
            command_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            command_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    connection.close()

    store = StateStore(database)

    assert (
        store._connection.execute("SELECT version FROM schema_meta").fetchone()[
            "version"
        ]
        == 4
    )
    session_columns = {
        row["name"] for row in store._connection.execute("PRAGMA table_info(sessions)")
    }
    command_columns = {
        row["name"] for row in store._connection.execute("PRAGMA table_info(commands)")
    }
    assert {
        "external_orchestrator",
        "external_job_id",
        "creation_digest",
    } <= session_columns
    assert {"turn_step_id", "turn_agent_role"} <= command_columns
    store.close()


def test_schema_v4_migrates_v3_and_forces_rollback_rejection(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = StateStore(database)
    store._connection.execute("UPDATE schema_meta SET version = 3")
    store.close()

    upgraded = StateStore(database)
    version = upgraded._connection.execute(
        "SELECT version FROM schema_meta"
    ).fetchone()["version"]
    assert version == 4
    durable_tables = {
        str(row["name"])
        for row in upgraded._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {
        "authorization_receipts",
        "dispatch_transition_ledger",
        "goal_contract_adoptions",
        "xhigh_authorization_receipts",
    } <= durable_tables
    upgraded.close()

    legacy = sqlite3.connect(database)
    try:
        legacy_version = legacy.execute("SELECT version FROM schema_meta").fetchone()[0]
        with pytest.raises(RuntimeError, match="unsupported database schema"):
            if legacy_version != 3:
                raise RuntimeError("unsupported database schema version")
    finally:
        legacy.close()


def test_external_reference_creation_lookup_fork_and_conflicts(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    external_ref = {
        "orchestrator": "p13i/machines",
        "job_id": "job-42",
    }
    created = replace(session(tmp_path), external_ref=external_ref)
    creation_input = {
        "name": created.name,
        "workspace": created.workspace,
        "permission_mode": created.permission_mode,
        "execution_profile": "unattended",
        "external_ref": external_ref,
    }

    first, was_created = store.ensure_session(
        created,
        creation_input,
        idempotency_key="create-job-42",
    )
    replay, replay_created = store.ensure_session(
        replace(created, session_id=new_uuid()),
        dict(reversed(list(creation_input.items()))),
        idempotency_key="create-job-42",
    )
    by_reference = store.get_session_by_external_ref(
        "p13i/machines",
        "job-42",
    )

    assert was_created
    assert not replay_created
    assert replay.session_id == first.session_id
    assert (
        store.existing_ensured_session(
            creation_input,
            idempotency_key="create-job-42",
        )
        == first
    )
    assert (
        store.existing_ensured_session(
            creation_input,
            external_ref=external_ref,
        )
        == first
    )
    assert (
        store.existing_ensured_session({"name": "new", "workspace": str(tmp_path)})
        is None
    )
    assert by_reference == first
    assert (
        store.find_session_by_external_ref(
            "p13i/machines",
            "job-42",
        )
        == first
    )
    assert store.list_sessions(external_ref=external_ref) == [first]
    with pytest.raises(ValueError):
        store.list_sessions(external_ref={})
    with pytest.raises(ConflictError):
        store.ensure_session(
            replace(created, session_id=new_uuid()),
            {**creation_input, "name": "different"},
            idempotency_key="create-job-42",
        )
    with pytest.raises(ConflictError):
        store.existing_ensured_session(
            {**creation_input, "name": "different"},
            idempotency_key="create-job-42",
        )
    with pytest.raises(ConflictError):
        store.ensure_session(
            replace(created, session_id=new_uuid()),
            {**creation_input, "name": "different"},
        )

    fork = store.create_fork(
        first.session_id,
        replace(
            session(tmp_path),
            session_id=new_uuid(),
            external_ref=external_ref,
        ),
    )
    assert fork.external_ref == {}
    with pytest.raises(ConflictError):
        store.create_fork(
            first.session_id,
            replace(session(tmp_path), session_id=new_uuid()),
            external_ref=external_ref,
        )
    with pytest.raises(NotFoundError):
        store.create_fork(
            "missing",
            replace(session(tmp_path), session_id=new_uuid()),
        )
    assert store.get_session_by_external_ref("other", "job") is None
    store.close()


def test_orchestration_normalization_and_turn_reference(
    tmp_path: Path,
) -> None:
    assert normalize_external_ref(None) == {}
    assert normalize_external_ref({}) == {}
    assert normalize_turn_ref(None) == {}
    assert normalize_turn_ref({}) == {}
    with pytest.raises(ValueError):
        normalize_external_ref("not-an-object")
    with pytest.raises(ValueError):
        normalize_external_ref({"orchestrator": "only"})
    with pytest.raises(ValueError):
        normalize_external_ref({"orchestrator": "has space", "job_id": "job"})
    with pytest.raises(ValueError):
        normalize_external_ref({"orchestrator": "x" * 129, "job_id": "job"})
    with pytest.raises(ValueError):
        normalize_turn_ref({"step_id": "only"})
    with pytest.raises(ValueError):
        normalize_turn_ref("not-an-object")
    with pytest.raises(ValueError):
        normalize_turn_ref({"step_id": 1, "agent_role": "reviewer"})
    assert (
        normalize_turn_ref({"step_id": "step", "agent_role": "x" * 128})["agent_role"]
        == "x" * 128
    )
    with pytest.raises(ValueError):
        normalize_turn_ref({"step_id": "step", "agent_role": "x" * 129})
    with pytest.raises(ValueError):
        normalize_creation_input({"routing": "automatic"})
    with pytest.raises(ValueError):
        normalize_creation_input({"goal": "finish"})
    with pytest.raises(ValueError):
        creation_digest({"goal": {"unsupported": object()}})
    with pytest.raises(ValueError):
        normalize_external_ref({"orchestrator": "control\0", "job_id": "job"})
    normalized = normalize_creation_input(
        {
            "routing": {"providers": ["codex", "claude"]},
            "goal": {"predicates": ({"type": "test"},)},
        }
    )
    assert normalized["routing"]["providers"] == ["codex", "claude"]
    assert normalized["goal"]["predicates"] == [{"type": "test"}]

    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    turn_ref = {"step_id": "step-1", "agent_role": "implementer"}
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "continue", "turn_ref": turn_ref},
        "managed-turn",
    )
    assert command.turn_ref == turn_ref
    assert store.claim_command(created.session_id) is not None
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="codex",
        native_session_id="",
        model="account-default",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    store.create_attempt(attempt)
    turn_id = store.start_turn(
        created.session_id,
        attempt.attempt_id,
        turn_ref=turn_ref,
    )
    assert store.turn_ref(turn_id) == turn_ref
    empty_turn = store.start_turn(
        created.session_id,
        attempt.attempt_id,
    )
    assert store.turn_ref(empty_turn) == {}
    with pytest.raises(NotFoundError):
        store.turn_ref("missing")
    store.close()


def test_store_backup_is_queryable(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    backup = tmp_path / "backup.sqlite3"
    store.backup(backup)
    copied = StateStore(backup)
    assert copied.get_session(created.session_id).name == "test"
    copied.close()
    store.close()


def test_safety_envelope_guard_and_lease_are_durable(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    safety = store.set_session_safety(created.session_id, "unattended")
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "continue"},
        "safety-key",
    )
    envelope = store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        {"max_seconds": 900},
    )
    child_gate_state = store.create_child_launch_gate(
        command.command_id,
        created.session_id,
        2,
    )

    assert safety["profile"] == "unattended"
    assert envelope["state"] == "reserved"
    assert envelope["consumption"]["child_agents"] == 0
    assert envelope["consumption"]["dollars"] == 0.0
    assert envelope["consumption"]["exact_dollars"] is False
    assert child_gate_state["permit_limit"] == 2
    assert child_gate_state["consumed"] == 0
    assert (
        store.create_child_launch_gate(
            command.command_id,
            created.session_id,
            2,
        )
        == child_gate_state
    )
    with pytest.raises(ValueError, match="must not be negative"):
        store.create_child_launch_gate(
            command.command_id,
            created.session_id,
            -1,
        )
    with pytest.raises(ConflictError, match="permit limit changed"):
        store.create_child_launch_gate(
            command.command_id,
            created.session_id,
            3,
        )
    with pytest.raises(ConflictError, match="session changed"):
        store.create_child_launch_gate(
            command.command_id,
            "other-session",
            2,
        )
    with pytest.raises(NotFoundError):
        store.child_launch_gate("missing")
    updated = store.update_command_envelope(
        command.command_id,
        provider="claude",
        state="running",
        consumption={"total_tokens": 100},
    )
    assert updated["provider"] == "claude"
    assert store.active_unattended_provider_count("claude") == 1
    incident_id = store.add_guard_incident(
        created.session_id,
        command.command_id,
        "attempt-1",
        "repeated-tool",
        "downgrade",
        {"consumption": {"tool_calls": 3}},
    )
    assert store.guard_incidents(created.session_id)[0]["incident_id"] == incident_id

    lease = store.create_process_lease(
        created.session_id,
        "claude",
        "unattended",
        "2099-01-01T00:00:00+00:00",
    )
    attached = store.update_process_lease(
        lease["lease_id"],
        pid=123,
        pid_start="456",
        state="active",
    )
    assert attached["pid"] == 123
    assert store.active_process_leases()[0]["lease_id"] == lease["lease_id"]
    assert (
        store.mutation_receipt(
            "new-key",
            "lease-create",
            "request-digest",
        )
        is None
    )
    receipt = store.record_mutation_receipt(
        "new-key",
        "lease-create",
        "request-digest",
        {"lease": lease},
        201,
    )
    assert receipt["response"]["lease"]["lease_id"] == lease["lease_id"]
    replay = store.mutation_receipt(
        "new-key",
        "lease-create",
        "request-digest",
    )
    assert replay == receipt
    store.close()


@pytest.mark.parametrize("same_provider", [True, False])
def test_route_admission_is_atomic_across_store_connections(
    tmp_path: Path,
    same_provider: bool,
) -> None:
    database = tmp_path / "route-admission.sqlite3"
    store = StateStore(database)
    created = session(tmp_path)
    store.create_session(created)
    goal = create_goal(
        created.session_id,
        "Admit only one command.",
        max_concurrency=1,
    )
    store.create_goal(goal)
    commands = []
    for index in range(2):
        command = store.enqueue_command(
            created.session_id,
            "message",
            {"text": "command " + str(index)},
            "route-" + str(index),
        )
        store.create_command_envelope(
            command.command_id,
            created.session_id,
            "unattended",
            {"max_attempts": 1},
        )
        commands.append(command)
    store.register_worker(created.session_id, 123, "reservation-worker")
    store.close()
    barrier = threading.Barrier(3)
    results: list[dict[str, object]] = []
    failures: list[BaseException] = []

    def reserve(index: int) -> None:
        connection = StateStore(database)
        provider = "claude"
        if not same_provider and index == 1:
            provider = "codex"
        barrier.wait()
        try:
            results.append(
                connection.reserve_route_admission(
                    commands[index].command_id,
                    provider,
                    "unattended",
                    worker_incarnation="reservation-worker",
                    goal_id=goal.goal_id,
                    max_concurrency=1,
                    lease_expires_at="2099-01-01T00:00:00+00:00",
                )
            )
        except BaseException as error:
            failures.append(error)
        finally:
            connection.close()

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not failures
    assert sum(bool(item["admitted"]) for item in results) == 1
    recovered = StateStore(database)
    assert len(recovered.active_process_leases()) == 1
    assert recovered.active_goal_command_count(goal.goal_id) == 1
    if same_provider:
        assert recovered.active_unattended_provider_count("claude") == 1
    recovered.close()


def test_worker_replacement_after_atomic_boundary_requires_reconciliation(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "incarnation-boundary.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "Cross one provider boundary."},
        "incarnation-boundary",
    )
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="codex",
        native_session_id="",
        model="default",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    store.create_attempt(attempt)
    turn_id = store.start_turn(created.session_id, attempt.attempt_id)
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="codex",
        native_session_id="",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=utc_now(),
    )
    store.add_checkpoint(checkpoint)
    store.record_dispatch_checkpoint(
        command.command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        {"max_attempts": 1},
    )
    store.register_worker(created.session_id, 123, "old-incarnation")

    admission = store.reserve_route_admission(
        command.command_id,
        "codex",
        "unattended",
        effort="high",
        attempt_id=attempt.attempt_id,
        worker_incarnation="old-incarnation",
        goal_id="",
        max_concurrency=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    assert admission["admitted"] is True

    store.register_worker(created.session_id, 456, "new-incarnation")
    recovery = store.recover_interrupted_commands(
        created.session_id,
        "current-workspace-digest",
        "current workspace",
    )

    assert recovery.requeued_command_ids == ()
    assert len(recovery.reconciliations) == 1
    assert recovery.reconciliations[0].command_id == command.command_id
    interrupted = store.get_command(command.command_id)
    assert interrupted.status == CommandStatus.FAILED
    assert interrupted.result["code"] == "E_NEEDS_RECONCILIATION"
    store.close()


def test_dispatch_transition_anchor_projects_fail_closed_reasons(
    tmp_path: Path,
) -> None:
    missing_workspace = tmp_path / "missing-anchor-workspace"
    _workspace_repository(missing_workspace)
    missing_store = StateStore(tmp_path / "missing-anchor.sqlite3")
    missing = session(missing_workspace)
    missing_store.create_session(missing)
    anchor = missing_store.dispatch_transition_anchor(missing.session_id)
    assert anchor["eligible"] is False
    assert anchor["reason"] == "missing-goal-epoch"
    missing_store.create_goal(
        create_goal(
            missing.session_id,
            "Require an external transition owner.",
            constraints=("dispatch-generation-transition-epoch:test-epoch",),
        )
    )
    anchor = missing_store.dispatch_transition_anchor(missing.session_id)
    assert anchor["reason"] == "missing-external-reference"
    missing_store.close()

    workspace = tmp_path / "anchor-workspace"
    _workspace_repository(workspace)
    store = StateStore(tmp_path / "anchor.sqlite3")
    created = replace(
        session(workspace),
        external_ref={"orchestrator": "machines", "job_id": "anchor"},
    )
    store.create_session(created)
    store.create_goal(
        create_goal(
            created.session_id,
            "Expose only a quiescent eligible anchor.",
            constraints=("dispatch-generation-transition-epoch:test-epoch",),
        )
    )
    assert store.dispatch_transition_anchor(created.session_id)["reason"] == (
        "missing-prior-command"
    )
    store.update_session(created.session_id, attention="working")
    assert store.dispatch_transition_anchor(created.session_id)["reason"] == (
        "session-is-working"
    )
    store.update_session(created.session_id, attention="idle")
    command = store.enqueue_command(
        created.session_id,
        "unsupported-control",
        {},
        "unsupported-anchor",
    )
    assert store.dispatch_transition_anchor(created.session_id)["reason"] == (
        "active-command"
    )
    store.resolve_command(command.command_id, CommandStatus.COMPLETE, {})
    assert store.dispatch_transition_anchor(created.session_id)["reason"] == (
        "missing-certified-checkpoint"
    )
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="codex",
        native_session_id="",
        base_commit=_git(workspace, "rev-parse", "HEAD"),
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=utc_now(),
    )
    store.add_checkpoint(checkpoint)
    anchor = store.dispatch_transition_anchor(created.session_id)
    assert anchor["eligible"] is False
    assert anchor["reason"] == ("dispatch transition prior command is not eligible")
    store.close()


def test_idempotent_mutation_serializes_and_rolls_back_nested_state(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    barrier = threading.Barrier(3)
    callback_lock = threading.Lock()
    callback_count = 0
    results: list[dict[str, object]] = []
    failures: list[BaseException] = []

    def mutate() -> dict[str, object]:
        nonlocal callback_count
        with callback_lock:
            callback_count += 1
        lease = store.create_process_lease(
            "",
            "codex",
            "unattended",
            "2099-01-01T00:00:00+00:00",
        )
        return {"lease": lease}

    def call() -> None:
        barrier.wait()
        try:
            results.append(
                store.idempotent_mutation(
                    "concurrent-key",
                    "lease-create",
                    "request-digest",
                    mutate,
                    201,
                )
            )
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not failures
    assert callback_count == 1
    assert len(results) == 2
    assert results[0] == results[1]
    assert len(store.active_process_leases()) == 1

    def failing_mutation() -> dict[str, object]:
        store.create_process_lease(
            "",
            "claude",
            "unattended",
            "2099-01-01T00:00:00+00:00",
        )
        raise RuntimeError("simulated response construction failure")

    with pytest.raises(RuntimeError, match="simulated response"):
        store.idempotent_mutation(
            "rollback-key",
            "lease-create",
            "rollback-digest",
            failing_mutation,
            201,
        )
    assert len(store.active_process_leases()) == 1
    assert (
        store.mutation_receipt(
            "rollback-key",
            "lease-create",
            "rollback-digest",
        )
        is None
    )
    store.close()


def test_dispatch_recovery_requeues_before_boundary_and_barriers_after(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    safe = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "safe retry"},
        "safe",
    )
    assert store.claim_command(created.session_id) is not None
    store.recover_interrupted_commands(
        created.session_id,
        "digest-before",
        "summary-before",
    )
    assert store.get_command(safe.command_id).status == CommandStatus.QUEUED

    claimed_safe = store.claim_command(created.session_id)
    assert claimed_safe is not None
    store.resolve_command(
        safe.command_id,
        CommandStatus.COMPLETE,
        {},
    )
    ambiguous = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "ambiguous"},
        "ambiguous",
    )
    assert store.claim_command(created.session_id) is not None
    now = utc_now()
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="claude",
        native_session_id="native",
        model="opus",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=now,
        ended_at="",
    )
    store.create_attempt(attempt)
    turn_id = store.start_turn(created.session_id, attempt.attempt_id)
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="claude",
        native_session_id="native",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=now,
    )
    store.add_checkpoint(checkpoint)
    store.record_dispatch_checkpoint(
        ambiguous.command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    with pytest.raises(ConflictError):
        store.mark_provider_boundary("missing")
    store.mark_provider_boundary(attempt.attempt_id)
    with pytest.raises(ConflictError):
        store.mark_provider_boundary(attempt.attempt_id)
    with pytest.raises(ValueError):
        store.complete_dispatch(attempt.attempt_id, "unknown")

    recovery = store.recover_interrupted_commands(
        created.session_id,
        "digest-current",
        "summary-current",
    )

    assert recovery.requeued_command_ids == ()
    assert len(recovery.reconciliations) == 1
    record = recovery.reconciliations[0]
    assert record.command_id == ambiguous.command_id
    assert record.pre_dispatch_checkpoint_id == checkpoint.checkpoint_id
    assert record.provider_attempts[0]["provider"] == "claude"
    assert record.safety_consumption == {}
    failed = store.get_command(ambiguous.command_id)
    assert failed.status == CommandStatus.FAILED
    assert failed.result["code"] == "E_NEEDS_RECONCILIATION"
    assert store.pending_reconciliations(created.session_id) == [record]
    assert store.reconciliation(record.reconciliation_id) == record
    with store.transaction() as connection:
        command_row = connection.execute(
            "SELECT * FROM commands WHERE command_id = ?",
            (ambiguous.command_id,),
        ).fetchone()
        dispatch_row = connection.execute(
            "SELECT * FROM command_dispatches WHERE command_id = ?",
            (ambiguous.command_id,),
        ).fetchone()
        assert (
            store._create_reconciliation(
                connection,
                command_row,
                dispatch_row,
                "digest-current",
                "summary-current",
                utc_now(),
            )
            == record
        )

    queued = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "must wait"},
        "barrier",
    )
    assert store.claim_command(created.session_id) is None
    control = store.enqueue_command(
        created.session_id,
        "stop",
        {},
        "control",
    )
    assert (
        store.claim_command(
            created.session_id,
            frozenset({"stop"}),
        ).command_id
        == control.command_id
    )
    assert store.get_command(queued.command_id).status == CommandStatus.QUEUED
    with pytest.raises(ConflictError):
        store.resolve_reconciliation_record(
            record.reconciliation_id,
            "stop",
            "stale",
            {},
        )
    with pytest.raises(NotFoundError):
        store.begin_reconciliation_resolution(
            "missing",
            "stop",
            "digest-current",
        )
    with pytest.raises(ConflictError):
        store.begin_reconciliation_resolution(
            record.reconciliation_id,
            "stop",
            "stale",
        )
    resolving = store.begin_reconciliation_resolution(
        record.reconciliation_id,
        "stop",
        "digest-current",
    )
    assert resolving.status == "resolving"
    assert (
        store.begin_reconciliation_resolution(
            record.reconciliation_id,
            "stop",
            "digest-current",
        )
        == resolving
    )
    with pytest.raises(ConflictError):
        store.begin_reconciliation_resolution(
            record.reconciliation_id,
            "accept-current",
            "digest-current",
        )
    with pytest.raises(ConflictError):
        store.resolve_reconciliation_record(
            record.reconciliation_id,
            "accept-current",
            "digest-current",
            {},
        )
    resolved = store.resolve_reconciliation_record(
        record.reconciliation_id,
        "stop",
        "digest-current",
        {"actor": "test"},
    )
    assert resolved.status == "resolved"
    assert (
        store.resolve_reconciliation_record(
            record.reconciliation_id,
            "stop",
            "digest-current",
            {"actor": "ignored-replay"},
        )
        == resolved
    )
    with pytest.raises(ConflictError):
        store.resolve_reconciliation_record(
            record.reconciliation_id,
            "accept-current",
            "digest-current",
            {},
        )
    with pytest.raises(ConflictError):
        store.begin_reconciliation_resolution(
            record.reconciliation_id,
            "stop",
            "digest-current",
        )
    with pytest.raises(NotFoundError):
        store.reconciliation("missing")
    with pytest.raises(NotFoundError):
        store.resolve_reconciliation_record(
            "missing",
            "stop",
            "digest-current",
            {},
        )
    store.close()


def test_session_export_import_preserves_resumable_state(
    tmp_path: Path,
) -> None:
    source = StateStore(tmp_path / "source.sqlite3")
    created = session(tmp_path)
    source.create_session(created)
    now = utc_now()
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="codex",
        native_session_id="native-session",
        model="account-default",
        effort="high",
        auth_mode="subscription",
        status="complete",
        started_at=now,
        ended_at=now,
    )
    source.create_attempt(attempt)
    source.append_event(
        created.session_id,
        "agent.message",
        role="assistant",
        text="durable response",
        status="complete",
    )
    goal = create_goal(
        created.session_id,
        "preserve the session",
        predicates=({"type": "test", "outcome": "passed"},),
    )
    source.create_goal(goal)
    source.add_evidence(make_evidence(goal.goal_id, "test", "unit", "passed"))
    checkpoint = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=1,
        provider="codex",
        native_session_id="native-session",
        base_commit="base",
        patch_digest="patch",
        untracked_digest="untracked",
        context_digest="context",
        created_at=now,
    )
    source.add_checkpoint(checkpoint)
    source.set_session_safety(created.session_id, "unattended")
    source.extend_session_safety(
        created.session_id,
        {
            "max_seconds": 120,
            "allow_xhigh_once": True,
        },
    )

    payload = source.export_session(created.session_id)
    destination = StateStore(tmp_path / "destination.sqlite3")
    imported = destination.import_session(
        payload,
        worktree=str(tmp_path / "restored"),
        owner_host="restored-host",
        owner_epoch=2,
    )

    assert imported.session_id == created.session_id
    assert imported.lifecycle == "paused"
    assert imported.owner_host == "restored-host"
    assert destination.attempts(created.session_id) == [attempt]
    assert destination.all_events(created.session_id)[0].text == ("durable response")
    assert destination.goal_for_session(created.session_id) is not None
    assert destination.evidence(goal.goal_id)[0].subject == "unit"
    assert destination.checkpoints(created.session_id) == [checkpoint]
    safety = destination.session_safety(created.session_id)
    assert safety["profile"] == "unattended"
    assert safety["xhigh_authorizations"] == 0
    assert safety["extensions"]["max_seconds"] == 120
    assert "allow_xhigh_once" not in safety["extensions"]
    with pytest.raises(ConflictError):
        destination.import_session(
            payload,
            worktree=str(tmp_path / "duplicate"),
            owner_host="restored-host",
            owner_epoch=3,
        )
    source.close()
    destination.close()


def test_registry_worker_ui_and_failure_paths(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)

    assert store.get_ui_state("theme") == {}
    store.set_ui_state("theme", {"mode": "dark"})
    assert store.get_ui_state("theme") == {"mode": "dark"}
    store.upsert_registry_entry(
        created.session_id,
        "host-a",
        "https://host-a.example",
        2,
        "running",
        4,
    )
    assert store.registry_entry(created.session_id)["owner_epoch"] == 2
    assert len(store.registry_entries()) == 1
    with pytest.raises(ConflictError):
        store.upsert_registry_entry(
            created.session_id,
            "host-b",
            "https://host-b.example",
            1,
            "paused",
            5,
        )
    store.record_routing(
        created.session_id,
        "turn",
        "codex",
        "account-default",
        "high",
        {"reason": "headroom"},
    )
    store.register_worker(created.session_id, 123, "incarnation")
    assert store.integrity_check() == "ok"
    registrations = store.worker_registrations()
    assert len(registrations) == 1
    assert registrations[0]["session_id"] == created.session_id
    assert registrations[0]["pid"] == 123
    assert registrations[0]["incarnation"] == "incarnation"
    assert registrations[0]["heartbeat_at"]
    assert store.heartbeat_worker(created.session_id, "incarnation")
    assert not store.heartbeat_worker(created.session_id, "other")
    store.remove_worker(created.session_id, "incarnation")
    assert store.worker_registrations() == []
    approval_id = store.create_approval(
        created.session_id,
        "",
        "reconciliation-1",
        "reconciliation.restore",
        "Restore?",
        [{"id": "approve", "label": "Restore"}],
    )
    assert store.approval(approval_id)["status"] == "pending"
    assert store.resolve_approval(
        approval_id,
        {"decision": "approve"},
    )
    assert store.approval(approval_id)["decision"] == {"decision": "approve"}
    with pytest.raises(NotFoundError):
        store.approval("missing")

    with pytest.raises(NotFoundError):
        store.get_session("missing")
    with pytest.raises(ValueError):
        store.update_session(created.session_id, unsupported=True)
    assert store.update_session(created.session_id) == created
    archived = store.update_session(created.session_id, archived=True)
    assert archived.archived
    assert store.list_sessions() == []
    assert store.list_sessions(include_archived=True) == [archived]
    with pytest.raises(NotFoundError):
        store.command_payload("missing")
    with pytest.raises(NotFoundError):
        store.get_command("missing")
    with pytest.raises(NotFoundError):
        store.resolve_command("missing", CommandStatus.FAILED, {})
    with pytest.raises(NotFoundError):
        store.get_goal("missing")
    with pytest.raises(NotFoundError):
        store.update_goal_status("missing", "complete")
    with pytest.raises(NotFoundError):
        store.process_lease("missing")
    with pytest.raises(NotFoundError):
        store.update_process_lease("missing", state="expired")
    with pytest.raises(NotFoundError):
        store.registry_entry("missing")

    with pytest.raises(RuntimeError):
        with store.transaction():
            raise RuntimeError("force rollback")
    store.close()


def test_portable_records_round_trip_complete_state(
    tmp_path: Path,
) -> None:
    harness_paths = paths(tmp_path / "chats")
    prepare_paths(harness_paths)
    source = StateStore(harness_paths.database)
    created = session(tmp_path)
    source.create_session(created)
    source.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="preserve this message",
    )
    source.set_ui_state(
        "session:" + created.session_id,
        {"theme": "system"},
    )
    source.set_ui_state("workspace", {"sidebar_width": "40"})
    command = source.enqueue_command(
        created.session_id,
        "message",
        {"text": "retain the child launch gate"},
        "portable-child-gate",
    )
    source.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        {"max_child_agents": 1},
    )
    source.create_child_launch_gate(
        command.command_id,
        created.session_id,
        1,
    )
    assert child_gate.admit(
        source.path,
        command.command_id,
        1,
        "codex:Agent:portable",
    )

    publish = publish_all(harness_paths, source)

    assert publish["state"] == "not-configured"
    records, global_record = load_portable_records(harness_paths)
    restored = StateStore(tmp_path / "restored.sqlite3")
    restored.import_portable(records, global_record)

    assert restored.portable_session(created.session_id) == (
        source.portable_session(created.session_id)
    )
    assert restored.portable_global() == source.portable_global()
    restored.close()
    source.close()


def test_portable_external_reference_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    external_ref = {
        "orchestrator": "p13i/autoplan",
        "job_id": "plan-7",
    }
    source = StateStore(tmp_path / "source.sqlite3")
    created = replace(session(tmp_path), external_ref=external_ref)
    source.ensure_session(
        created,
        {
            "name": created.name,
            "workspace": created.workspace,
            "external_ref": external_ref,
        },
    )
    record = source.portable_session(created.session_id)
    global_record = source.portable_global()

    destination = StateStore(tmp_path / "destination.sqlite3")
    existing = replace(
        session(tmp_path),
        session_id=new_uuid(),
        external_ref=external_ref,
    )
    destination.create_session(existing)
    with pytest.raises(ConflictError):
        destination.merge_portable([record], global_record)
    destination.close()

    duplicate = copy.deepcopy(record)
    duplicate_id = new_uuid()
    duplicate["session_id"] = duplicate_id
    duplicate["tables"]["sessions"][0]["session_id"] = duplicate_id
    empty = StateStore(tmp_path / "empty.sqlite3")
    with pytest.raises(ConflictError):
        empty.import_portable([record, duplicate], global_record)
    empty.close()
    source.close()


def test_portable_records_reject_missing_blobs_and_bad_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    source = StateStore(harness_paths.database)
    created = session(tmp_path)
    source.create_session(created)
    missing = "0" * 64
    source.add_checkpoint(
        Checkpoint(
            checkpoint_id=new_uuid(),
            session_id=created.session_id,
            sequence=0,
            provider="codex",
            native_session_id="native",
            base_commit="base",
            patch_digest=missing,
            untracked_digest=missing,
            context_digest=missing,
            created_at=utc_now(),
        )
    )

    pending = publish_session(
        harness_paths,
        source,
        created.session_id,
    )

    assert pending["state"] == "pending"
    assert pending["detail"] == "record-materialization"
    source.close()

    clean_paths = paths(tmp_path / "clean")
    prepare_paths(clean_paths)
    clean = StateStore(clean_paths.database)
    clean_session = session(tmp_path)
    clean.create_session(clean_session)
    clean.record_context_delivery(
        clean_session.session_id,
        "codex",
        "1" * 64,
        "",
    )
    published = publish_session(
        clean_paths,
        clean,
        clean_session.session_id,
    )
    assert published["state"] == "not-configured"

    def fail_sync(unused) -> dict[str, object]:
        del unused
        raise RuntimeError("Git failed")

    monkeypatch.setattr(
        sync_module,
        "sync_repository",
        fail_sync,
    )
    pending_sync = publish_session(
        clean_paths,
        clean,
        clean_session.session_id,
    )
    assert pending_sync["detail"] == "repository-synchronization"
    record_path = clean_paths.sessions / clean_session.session_id / "record.gpt.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["schema"] = "unsupported"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported portable"):
        load_portable_records(clean_paths)
    record_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        records_module._read_json(record_path)
    clean.close()


def test_context_delivery_survives_restart_and_blocks_ambiguous_resend(
    tmp_path: Path,
) -> None:
    database = tmp_path / "context-state.sqlite3"
    store = StateStore(database)
    created = session(tmp_path)
    store.create_session(created)
    prepared = store.prepare_context_delivery(
        created.session_id,
        "codex",
        "context-a",
        "checkpoint-a",
        "command-a",
        "attempt-a",
        "payload-a",
    )
    assert prepared["state"] == "prepared"
    assert prepared["accepted_at"] == ""
    retried = store.prepare_context_delivery(
        created.session_id,
        "codex",
        "context-b",
        "checkpoint-b",
        "command-a",
        "attempt-b",
        "payload-b",
    )
    assert retried["state"] == "prepared"
    assert retried["attempt_id"] == "attempt-b"
    store.close()

    recovered = StateStore(database)
    delivered = recovered.accept_context_delivery(
        created.session_id,
        "codex",
        "context-b",
        "attempt-b",
    )
    assert delivered["state"] == "delivered"
    assert delivered["accepted_at"]
    with pytest.raises(ConflictError, match="prior context delivery"):
        recovered.prepare_context_delivery(
            created.session_id,
            "codex",
            "context-c",
            "checkpoint-c",
            "command-a",
            "attempt-c",
            "payload-c",
        )
    recovered.close()


def test_portable_record_validation_boundaries(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="tables"):
        records_module._table_rows({"tables": []}, "events")
    with pytest.raises(ValueError, match="table"):
        records_module._table_rows(
            {"tables": {"events": {}}},
            "events",
        )
    with pytest.raises(ValueError, match="row"):
        records_module._table_rows(
            {"tables": {"events": ["invalid"]}},
            "events",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        records_module._require_blob_digest("short")
    with pytest.raises(ValueError, match="hexadecimal"):
        records_module._require_blob_digest("z" * 64)
    digest = "1" * 64
    assert records_module._blob_digests(
        {
            "tables": {
                "events": [{"blob_digest": digest}],
                "checkpoints": [],
            }
        }
    ) == [digest]
    with pytest.raises(ValueError, match="one session"):
        records_module._transcript(
            {"tables": {"sessions": []}},
            [],
        )

    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    (harness_paths.state_dir / "global.gpt.json").write_text('{"schema":"invalid"}')
    with pytest.raises(ValueError, match="global"):
        load_portable_records(harness_paths)


def test_sync_status_and_unreachable_remote_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "chats")
    prepare_paths(harness_paths)
    harness_paths.sync_status.write_text("{", encoding="utf-8")
    assert read_sync_status(harness_paths)["state"] == "invalid"
    harness_paths.sync_status.write_text("[]", encoding="utf-8")
    assert read_sync_status(harness_paths)["state"] == "invalid"
    with pytest.raises(ValueError, match="positive"):
        sync_repository(harness_paths, attempts=0)

    _workspace_repository(harness_paths.state_dir, create=False)
    store = StateStore(harness_paths.database)
    created = session(tmp_path)
    store.create_session(created)
    records_module.materialize_all(harness_paths, store)

    result = sync_repository(harness_paths, attempts=1)

    assert result["state"] == "pending"
    assert result["detail"] == "git-fetch"

    def materialization_fails(*unused_args, **unused_kwargs):
        del unused_args
        del unused_kwargs
        raise ValueError("invalid portable record")

    monkeypatch.setattr(
        sync_module,
        "materialize_all",
        materialization_fails,
    )
    result = sync_module.publish_all(harness_paths, store)
    assert result["pending"] is True
    assert result["detail"] == "record-materialization"

    def materialization_succeeds(*unused_args, **unused_kwargs):
        del unused_args
        del unused_kwargs
        return []

    def synchronization_fails(*unused_args, **unused_kwargs):
        del unused_args
        del unused_kwargs
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(
        sync_module,
        "materialize_all",
        materialization_succeeds,
    )
    monkeypatch.setattr(
        sync_module,
        "sync_repository",
        synchronization_fails,
    )
    result = sync_module.publish_all(harness_paths, store)
    assert result["pending"] is True
    assert result["detail"] == "repository-synchronization"
    store.close()

    def time_out(*unused_args, **unused_kwargs):
        del unused_args
        del unused_kwargs
        raise subprocess.TimeoutExpired(["git"], 1)

    monkeypatch.setattr(sync_module.subprocess, "run", time_out)
    with pytest.raises(RuntimeError, match="timed out"):
        sync_module._git(tmp_path, "status")


def test_sync_locked_failure_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_paths = paths(tmp_path / "state")
    prepare_paths(harness_paths)
    monkeypatch.setattr(sync_module, "_head", lambda root: "head")
    monkeypatch.setattr(sync_module.time, "sleep", lambda value: None)

    def completed(returncode: int) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            returncode,
            stdout="",
            stderr="",
        )

    def run_case(
        returncodes: list[int],
        attempts: int = 1,
    ) -> dict[str, object]:
        values = list(returncodes)

        def git(*args, **kwargs):
            del args
            del kwargs
            return completed(values.pop(0))

        monkeypatch.setattr(sync_module, "_git", git)
        return sync_module._sync_locked(harness_paths, attempts)

    assert run_case([0, 2])["detail"] == "git-index"
    assert run_case([0, 1, 1])["detail"] == "git-commit"
    assert run_case([0, 0, 1, 1], attempts=2)["detail"] == "git-fetch"
    conflict = run_case([0, 0, 0, 1, 0])
    assert conflict["state"] == "conflict"
    assert (
        run_case(
            [0, 0, 0, 0, 1, 0, 0, 1],
            attempts=2,
        )["detail"]
        == "git-push"
    )


def test_sync_repository_and_git_error_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sync_module,
        "_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [],
            0,
            stdout=str(tmp_path),
            stderr="",
        ),
    )

    def unavailable(self: Path, *args, **kwargs) -> Path:
        del self
        del args
        del kwargs
        raise OSError

    monkeypatch.setattr(Path, "resolve", unavailable)
    assert not sync_module._is_repository(tmp_path)

    monkeypatch.undo()
    monkeypatch.setattr(
        sync_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="failed"):
        sync_module._git(tmp_path, "status")


def test_runtime_state_is_cleared_and_worktrees_are_rewritten(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    original = str(tmp_path / "old" / created.session_id)
    created = replace(created, worktree=original)
    store.create_session(created)
    store.register_worker(created.session_id, 123, "worker")
    store.create_process_lease(
        created.session_id,
        "codex",
        "interactive",
        "2099-01-01T00:00:00+00:00",
    )

    changed = store.rewrite_worktree_prefix(
        str(tmp_path / "old"),
        str(tmp_path / "new"),
    )
    store.clear_runtime_state()

    assert changed == 1
    assert store.get_session(created.session_id).worktree == str(
        tmp_path / "new" / created.session_id
    )
    assert not store.heartbeat_worker(created.session_id, "worker")
    assert store.active_process_leases() == []
    store.close()


def test_git_sync_commits_and_pushes_portable_records(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote.git"
    chat_root = tmp_path / "chats"
    _chat_repository(chat_root, remote)
    harness_paths = paths(chat_root)
    prepare_paths(harness_paths)
    store = StateStore(harness_paths.database)
    created = session(tmp_path)
    store.create_session(created)
    store.append_event(
        created.session_id,
        "agent.message",
        role="assistant",
        text="durable",
    )

    result = publish_all(harness_paths, store)

    assert result["state"] == "synced"
    assert read_sync_status(harness_paths)["pending"] is False
    assert (chat_root / "sessions" / created.session_id / "record.gpt.json").is_file()
    remote_record = _git(
        remote,
        "show",
        "main:sessions/" + created.session_id + "/transcript.gpt.md",
    )
    assert "durable" in remote_record
    store.close()


def test_legacy_migration_preserves_resume_state_and_git_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _workspace_repository(workspace)
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    source_store = StateStore(source_root / "state.sqlite3")
    created = session(workspace)
    worktree = create_worktree(
        workspace,
        source_root / "worktrees",
        created.session_id,
    )
    created = replace(created, worktree=str(worktree))
    source_store.create_session(created)
    source_store.append_event(
        created.session_id,
        "user.message",
        role="user",
        text="resume after migration",
    )
    source_store.close()

    remote = tmp_path / "remote.git"
    destination_root = tmp_path / "chats"
    _chat_repository(destination_root, remote)
    destination_paths = paths(destination_root)
    prepare_paths(destination_paths)
    existing_store = StateStore(destination_paths.database)
    existing = session(tmp_path / "existing")
    existing_store.create_session(existing)
    existing_store.append_event(
        existing.session_id,
        "agent.message",
        role="assistant",
        text="preserve destination state",
    )
    existing = existing_store.get_session(existing.session_id)
    assert publish_all(destination_paths, existing_store)["state"] == ("synced")
    existing_store.close()

    trash_path = tmp_path / "trashed-source"
    monkeypatch.setattr(
        migration_module,
        "_trash_source",
        lambda unused: trash_path,
    )
    result = migrate_state(
        source_root,
        destination_root,
        trash_source=True,
    )

    assert result["sessions"] == 1
    assert result["events"] == 1
    assert result["worktrees"] == 1
    assert result["source_trashed"]
    assert result["trash_path"] == str(trash_path)
    destination_store = StateStore(destination_paths.database)
    migrated = destination_store.get_session(created.session_id)
    assert migrated.session_id == created.session_id
    assert migrated.worktree == str(destination_paths.worktrees / created.session_id)
    assert destination_store.all_events(created.session_id)[0].text == (
        "resume after migration"
    )
    assert destination_store.get_session(existing.session_id) == existing
    assert destination_store.all_events(existing.session_id)[0].text == (
        "preserve destination state"
    )
    assert len(destination_store.list_sessions()) == 2
    assert not worktree.exists()
    assert Path(migrated.worktree).is_dir()
    assert _git(Path(migrated.worktree), "status", "--porcelain=v1") == ""
    assert read_sync_status(destination_paths)["state"] == "synced"
    destination_store.close()


def test_migration_rejects_invalid_moved_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_paths = migration_module.legacy_paths(source_root)
    source_paths.worktrees.mkdir(parents=True)
    current = source_paths.worktrees / "session-worktree"
    current.mkdir()
    store = StateStore(source_paths.database)
    created = replace(
        session(tmp_path / "workspace"),
        worktree=str(current),
    )
    store.create_session(created)
    destination = paths(tmp_path / "destination")
    prepare_paths(destination)
    results = [
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 1, "", "invalid"),
    ]
    monkeypatch.setattr(
        migration_module,
        "_run",
        lambda *unused, **unused_values: results.pop(0),
    )

    with pytest.raises(RuntimeError, match="migrated worktree"):
        migration_module._move_worktrees(
            store,
            source_paths,
            destination,
            [],
        )
    store.close()


def test_failed_migration_restores_worktree_and_remains_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _workspace_repository(workspace)
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    source_store = StateStore(source_root / "state.sqlite3")
    created = session(workspace)
    worktree = create_worktree(
        workspace,
        source_root / "worktrees",
        created.session_id,
    )
    created = replace(created, worktree=str(worktree))
    source_store.create_session(created)
    source_store.close()
    remote = tmp_path / "remote.git"
    destination_root = tmp_path / "chats"
    _chat_repository(destination_root, remote)
    destination_paths = paths(destination_root)
    prepare_paths(destination_paths)
    destination_store = StateStore(destination_paths.database)
    existing = session(tmp_path / "existing")
    destination_store.create_session(existing)
    destination_store.append_event(
        existing.session_id,
        "agent.message",
        role="assistant",
        text="keep on rollback",
    )
    assert publish_all(destination_paths, destination_store)["state"] == ("synced")
    destination_store.close()
    monkeypatch.setattr(
        migration_module,
        "sync_repository",
        lambda unused: {"state": "pending"},
    )

    with pytest.raises(RuntimeError, match="synchronize"):
        migrate_state(
            source_root,
            destination_root,
            trash_source=False,
        )

    assert worktree.is_dir()
    assert _git(worktree, "status", "--porcelain=v1") == ""
    restored_destination = StateStore(destination_paths.database)
    assert len(restored_destination.list_sessions()) == 1
    assert restored_destination.get_session(existing.session_id).session_id == (
        existing.session_id
    )
    assert restored_destination.all_events(existing.session_id)[0].text == (
        "keep on rollback"
    )
    restored_destination.close()
    retry_store = StateStore(source_root / "state.sqlite3")
    assert retry_store.get_session(created.session_id).worktree == str(worktree)
    retry_store.close()


def test_migration_validation_rejects_unsafe_roots_and_low_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(ValueError, match="database"):
        migration_module._validate_roots(
            migration_module.legacy_paths(missing),
            paths(destination),
        )

    source = tmp_path / "source"
    source.mkdir()
    source_store = StateStore(source / "state.sqlite3")
    source_store.close()
    with pytest.raises(ValueError, match="Git repository"):
        migration_module._validate_roots(
            migration_module.legacy_paths(source),
            paths(destination),
        )

    _workspace_repository(source, create=False)
    with pytest.raises(ValueError, match="must differ"):
        migration_module._validate_roots(
            migration_module.legacy_paths(source),
            paths(source),
        )

    monkeypatch.setattr(
        migration_module.shutil,
        "disk_usage",
        lambda unused: type("Usage", (), {"free": 0})(),
    )
    with pytest.raises(RuntimeError, match="disk space"):
        migration_module._require_headroom(source, destination)


def _chat_repository(
    root: Path,
    remote: Path,
    *,
    create: bool = True,
) -> None:
    _run(["git", "init", "--bare", str(remote)])
    if create:
        root.mkdir()
    _run(["git", "-C", str(root), "init", "-b", "main"])
    _configure_identity(root)
    (root / ".gitignore").write_text(".runtime/\n", encoding="utf-8")
    _run(["git", "-C", str(root), "add", ".gitignore"])
    _run(["git", "-C", str(root), "commit", "-m", "Initialize chats"])
    _run(["git", "-C", str(root), "remote", "add", "origin", str(remote)])
    _run(["git", "-C", str(root), "push", "-u", "origin", "main"])


def _workspace_repository(
    root: Path,
    *,
    create: bool = True,
) -> None:
    if create:
        root.mkdir()
    _run(["git", "-C", str(root), "init", "-b", "main"])
    _configure_identity(root)
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _run(["git", "-C", str(root), "add", "tracked.txt"])
    _run(["git", "-C", str(root), "commit", "-m", "Initialize workspace"])


def _configure_identity(root: Path) -> None:
    _run(["git", "-C", str(root), "config", "user.name", "Test User"])
    _run(
        [
            "git",
            "-C",
            str(root),
            "config",
            "user.email",
            "test@example.com",
        ]
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )


TRANSITION_AUTHORIZATION_SCHEMA = (
    "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
)


def test_proof_snapshot_flags_each_truncated_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    original_rows = proof_module._rows

    def padded_rows(tables: dict, name: str) -> list[dict]:
        if name == "dispatch_transition_ledger":
            return [{"invalidation_id": "invalidation-1"}]
        if name == "authorization_receipts":
            return [
                {
                    "schema": TRANSITION_AUTHORIZATION_SCHEMA,
                    "operation_id": "invalidation-2",
                },
                {"schema": "p13i/agent-harness/goal-promotion-authorization/v1"},
            ]
        if name == "dispatch_invalidations":
            return [{"invalidation_id": "invalidation-3"}]
        return original_rows(tables, name)

    monkeypatch.setattr(proof_module, "_rows", padded_rows)
    monkeypatch.setattr(proof_module, "MAX_TRANSITION_LEDGER_RECORDS", 0)
    monkeypatch.setattr(proof_module, "MAX_PROOF_RECORDS", 0)

    snapshot = proof_snapshot(store, created.session_id)

    assert "dispatch_transition_ledger" in snapshot["truncated"]
    assert "authorization_receipts" in snapshot["truncated"]
    assert "dispatch_invalidations" in snapshot["truncated"]
    assert snapshot["complete"] is False
    store.close()


def test_proof_authorization_receipts_tolerate_an_absent_receipt() -> None:
    projected = proof_module._proof_authorization_receipt(
        {
            "payload_json": json.dumps({"schema": "other", "receipt": "opaque"}),
            "authorization_digest": "a" * 64,
            "receipt_sha256": "b" * 64,
        },
        {},
    )

    assert projected["receipt_digest"] == proof_module._digest({})
    assert projected["receipt_digest_valid"] is False


def test_proof_transition_policy_projection_rejects_malformed_stages() -> None:
    def policy(transitions: object) -> dict:
        return {
            "policy_sha256": "c" * 64,
            "schema": "p13i/agent-harness/dispatch-generation-transition-policy/v1",
            "session_id": "session-1",
            "epoch_id": "epoch-1",
            "payload_json": json.dumps(
                {
                    "schema": (
                        "p13i/agent-harness/"
                        "dispatch-generation-transition-policy/v1"
                    ),
                    "session_id": "session-1",
                    "epoch_id": "epoch-1",
                    "external_ref": {"orchestrator": "machines"},
                    "allowed_agent_roles": ["verifier"],
                    "allowed_step_prefixes": ["verify"],
                    "max_transitions": 1,
                    "transitions": transitions,
                }
            ),
        }

    valid_stage = {
        "sequence": 1,
        "next_turn_ref": {"step_id": "verify", "agent_role": "verifier"},
        "next_command_digest": "d" * 64,
    }

    assert proof_module._proof_dispatch_transition_policy(policy([valid_stage]))[
        "policy_contract_valid"
    ]
    assert not proof_module._proof_dispatch_transition_policy(policy(["bare"]))[
        "policy_contract_valid"
    ]
    assert not proof_module._proof_dispatch_transition_policy(
        policy([{**valid_stage, "sequence": 2}])
    )["policy_contract_valid"]
    assert not proof_module._proof_dispatch_transition_policy(
        policy([{**valid_stage, "next_turn_ref": {"agent_role": "verifier"}}])
    )["policy_contract_valid"]


def test_proof_transition_ledger_tracks_reservation_and_sequence_gaps() -> None:
    def ledger_row(sequence: int, state: str) -> dict:
        return {
            "invalidation_id": "invalidation-" + str(sequence),
            "session_id": "session-1",
            "goal_id": "goal-1",
            "epoch_id": "epoch-1",
            "policy_sha256": "c" * 64,
            "transition_sequence": sequence,
            "state": state,
            "reserved_command_id": "command-" + str(sequence),
            "consumed_command_id": "",
            "next_turn_ref_json": json.dumps(
                {"step_id": "verify", "agent_role": "verifier"}
            ),
            "created_at": "2026-08-02T00:0" + str(sequence) + ":00+00:00",
        }

    ledger = proof_module._proof_dispatch_transition_ledger(
        [ledger_row(1, "reserved"), ledger_row(3, "reserved")],
        [],
        [],
    )

    assert ledger["complete"] is False
    assert [item["state"] for item in ledger["receipts"]] == [
        "reserved",
        "reserved",
    ]


def test_proof_route_admissibility_requires_bounded_binding_usage() -> None:
    bound = {"provider": "codex", "error_present": False}

    assert not proof_module._usage_admissible_at_route({}, bound, False, True)
    assert not proof_module._usage_admissible_at_route(
        {},
        {"error_present": True},
        True,
        True,
    )
    assert not proof_module._usage_admissible_at_route(
        {"binding_percent": True},
        bound,
        True,
        True,
    )
    assert not proof_module._usage_admissible_at_route(
        {"binding_percent": 95.0},
        bound,
        True,
        True,
    )
    assert proof_module._usage_admissible_at_route(
        {"binding_percent": 10.0, "credits_engaged": False},
        bound,
        True,
        True,
    )
