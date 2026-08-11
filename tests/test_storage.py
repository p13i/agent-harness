import asyncio
import copy
import datetime
import hashlib
import json
import sqlite3
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_support import session

import agent_harness.migration as migration_module
import agent_harness.proof as proof_module
import agent_harness.records as records_module
import agent_harness.storage as storage_module
import agent_harness.sync as sync_module
from agent_harness import child_gate
from agent_harness.blobs import BlobStore
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
from agent_harness.reconciliation import ReconciliationManager
from agent_harness.records import load_portable_records
from agent_harness.storage import StateStore
from agent_harness.sync import (
    publish_all,
    publish_session,
    read_sync_status,
    sync_repository,
)
from agent_harness.workspace import create_worktree
from agent_harness.workspace_state import inspect_workspace


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


def test_schema_v5_migrates_external_and_turn_columns(
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
        == 5
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


def test_schema_v5_migrates_v3_and_forces_rollback_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = StateStore(database)
    store._connection.execute("UPDATE schema_meta SET version = 3")
    store.close()

    migrations: list[int] = []
    original_migration = StateStore._migrate_to_v4

    def track_migration(
        current: StateStore,
        connection: sqlite3.Connection,
    ) -> None:
        migrations.append(4)
        original_migration(current, connection)

    monkeypatch.setattr(StateStore, "_migrate_to_v4", track_migration)
    upgraded = StateStore(database)
    version = upgraded._connection.execute(
        "SELECT version FROM schema_meta"
    ).fetchone()["version"]
    assert version == 5
    assert migrations == [4]
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


def test_schema_v5_rebuilds_v4_context_delivery_identity(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = StateStore(database)
    created = session(tmp_path)
    store.create_session(created)
    store.close()

    legacy = sqlite3.connect(database)
    legacy.executescript(
        """
        DROP INDEX context_deliveries_context;
        ALTER TABLE context_deliveries RENAME TO context_deliveries_v5;
        CREATE TABLE context_deliveries (
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            provider TEXT NOT NULL,
            context_digest TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            delivered_at TEXT NOT NULL,
            command_id TEXT NOT NULL DEFAULT '',
            attempt_id TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL DEFAULT 'delivered',
            payload_digest TEXT NOT NULL DEFAULT '',
            accepted_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(session_id, provider, context_digest)
        );
        DROP TABLE context_deliveries_v5;
        UPDATE schema_meta SET version = 4;
        """
    )
    legacy.execute(
        """
        INSERT INTO context_deliveries VALUES (
            ?, 'codex', 'legacy-context', 'legacy-checkpoint',
            '2026-08-02T00:00:00+00:00', '', 'duplicate-attempt', 'delivered',
            'legacy-payload', '2026-08-02T00:00:00+00:00'
        )
        """,
        (created.session_id,),
    )
    legacy.execute(
        """
        INSERT INTO context_deliveries VALUES (
            ?, 'claude', 'legacy-context-two', 'legacy-checkpoint',
            '2026-08-02T00:00:01+00:00', '', 'duplicate-attempt', 'delivered',
            'legacy-payload-two', '2026-08-02T00:00:01+00:00'
        )
        """,
        (created.session_id,),
    )
    legacy.execute(
        """
        INSERT INTO context_deliveries VALUES (
            ?, 'codex', 'legacy-context-three', 'legacy-checkpoint',
            '2026-08-02T00:00:02+00:00', '', '', 'delivered',
            'legacy-payload-three', '2026-08-02T00:00:02+00:00'
        )
        """,
        (created.session_id,),
    )
    duplicate_rowid = legacy.execute(
        """
        SELECT rowid FROM context_deliveries
        WHERE provider = 'codex' AND context_digest = 'legacy-context'
        """
    ).fetchone()[0]
    colliding_attempt_id = "legacy-duplicate-" + normalized_digest(
        {
            "attempt_id": "duplicate-attempt",
            "rowid": duplicate_rowid,
            "session_id": created.session_id,
            "provider": "codex",
            "context_digest": "legacy-context",
        }
    )
    legacy.execute(
        """
        INSERT INTO context_deliveries VALUES (
            ?, 'other', 'legacy-context-four', 'legacy-checkpoint',
            '2026-08-02T00:00:03+00:00', '', ?, 'delivered',
            'legacy-payload-four', '2026-08-02T00:00:03+00:00'
        )
        """,
        (created.session_id, colliding_attempt_id),
    )
    legacy.commit()
    legacy.close()

    upgraded = StateStore(database)
    rows = upgraded._connection.execute(
        "SELECT * FROM context_deliveries ORDER BY attempt_id"
    ).fetchall()
    assert len(rows) == 4
    by_context = {str(row["context_digest"]): row for row in rows}
    first_duplicate = str(by_context["legacy-context"]["attempt_id"])
    second_duplicate = str(by_context["legacy-context-two"]["attempt_id"])
    assert first_duplicate.startswith("legacy-duplicate-")
    assert first_duplicate != colliding_attempt_id
    assert second_duplicate.startswith("legacy-duplicate-")
    assert second_duplicate != first_duplicate
    assert str(by_context["legacy-context-three"]["attempt_id"]).startswith(
        "legacy-"
    )
    assert by_context["legacy-context-four"]["attempt_id"] == colliding_attempt_id
    primary_key = [
        item["name"]
        for item in upgraded._connection.execute(
            "PRAGMA table_info(context_deliveries)"
        ).fetchall()
        if item["pk"]
    ]
    assert primary_key == ["attempt_id"]
    foreign_keys = upgraded._connection.execute(
        "PRAGMA foreign_key_list(context_deliveries)"
    ).fetchall()
    assert any(str(item["table"]) == "sessions" for item in foreign_keys)
    upgraded.close()


def test_current_schema_does_not_rerun_prior_migrations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state.sqlite3"
    StateStore(database).close()

    def fail_migration(*unused: object) -> None:
        raise AssertionError("current schema reran a prior migration")

    monkeypatch.setattr(StateStore, "_migrate_to_v4", fail_migration)
    monkeypatch.setattr(StateStore, "_migrate_to_v5", fail_migration)
    StateStore(database).close()


def test_v5_instruction_index_tolerates_malformed_legacy_metadata(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = StateStore(database)
    created = session(tmp_path)
    store.create_session(created)
    store.close()

    legacy = sqlite3.connect(database)
    legacy.executescript(
        """
        DROP INDEX events_command_instruction_v2;
        UPDATE schema_meta SET version = 4;
        """
    )
    legacy.execute(
        """
        INSERT INTO events VALUES (
            ?, 1, 'malformed-event', 'user.message', 'user',
            'legacy text', 'accepted', '', '', '',
            '2026-08-02T00:00:00+00:00'
        )
        """,
        (created.session_id,),
    )
    legacy.commit()
    legacy.close()

    upgraded = StateStore(database)
    retained = upgraded._connection.execute(
        "SELECT event_id FROM events WHERE session_id = ?",
        (created.session_id,),
    ).fetchone()
    assert retained["event_id"] == "malformed-event"
    with pytest.raises(NotFoundError, match="instruction event"):
        upgraded.command_instruction_sequence(created.session_id, "missing")
    upgraded.close()


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


def test_terminal_command_envelope_does_not_consume_concurrency(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "terminal-envelope.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    goal = create_goal(created.session_id, "Release terminal reservations.")
    store.create_goal(goal)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "attempt provider work"},
        "terminal-envelope",
    )
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        {"max_attempts": 1},
    )
    store.update_command_envelope(
        command.command_id,
        provider="kimi",
        state="reserved",
    )

    assert store.active_unattended_provider_count("kimi") == 1
    assert store.active_goal_command_count(goal.goal_id) == 1

    with store.transaction() as connection:
        connection.execute(
            "UPDATE commands SET status = ? WHERE command_id = ?",
            (CommandStatus.FAILED, command.command_id),
        )

    assert store.command_envelope(command.command_id)["state"] == "reserved"
    assert store.active_unattended_provider_count("kimi") == 0
    assert store.active_goal_command_count(goal.goal_id) == 0

    next_command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "attempt replacement provider work"},
        "replacement-envelope",
    )
    store.create_command_envelope(
        next_command.command_id,
        created.session_id,
        "unattended",
        {"max_attempts": 1},
    )
    store.register_worker(created.session_id, 123, "replacement-worker")
    admission = store.reserve_route_admission(
        next_command.command_id,
        "kimi",
        "unattended",
        worker_incarnation="replacement-worker",
        goal_id=goal.goal_id,
        max_concurrency=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )

    assert admission["admitted"] is True
    assert store.active_unattended_provider_count("kimi") == 1
    assert store.active_goal_command_count(goal.goal_id) == 1
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


class TerminalGuardState:
    """A failed unattended guard stop that left a certified checkpoint."""

    def __init__(
        self,
        store: StateStore,
        created: Any,
        workspace: Path,
        command_id: str,
        attempt_id: str,
        turn_id: str,
        checkpoint: Checkpoint,
        material_digest: str,
    ) -> None:
        self.store = store
        self.session = created
        self.workspace = workspace
        self.command_id = command_id
        self.attempt_id = attempt_id
        self.turn_id = turn_id
        self.checkpoint = checkpoint
        self.material_digest = material_digest

    def anchor(self) -> dict[str, Any]:
        return self.store.dispatch_transition_anchor(self.session.session_id)

    def reason(self) -> str:
        return str(self.anchor()["reason"])

    def set_result(self, **overrides: Any) -> None:
        result = {
            "code": "E_SAFETY_GUARD",
            "message": "execution safety guard stopped kimi: context-window",
            "retryable": False,
            "provider_terminal": True,
            "checkpoint_id": self.checkpoint.checkpoint_id,
            "workspace_material_digest": self.material_digest,
        }
        result.update(overrides)
        self.store.resolve_command(
            self.command_id,
            CommandStatus.FAILED,
            result,
        )

    def execute(self, statement: str, parameters: tuple[Any, ...]) -> None:
        with self.store.transaction() as connection:
            connection.execute(statement, parameters)


def _terminal_guard_state(root: Path, name: str) -> TerminalGuardState:
    workspace = root / (name + "-workspace")
    _workspace_repository(workspace)
    store = StateStore(root / (name + ".sqlite3"))
    created = replace(
        session(workspace),
        external_ref={
            "orchestrator": "p13i/machines/cs-builder",
            "job_id": name,
        },
    )
    store.create_session(created)
    store.create_goal(
        create_goal(
            created.session_id,
            "Hand a terminal builder checkpoint to the review stage.",
            constraints=("dispatch-generation-transition-epoch:test-epoch",),
        )
    )
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "implement the stage", "provider": "kimi"},
        name + "-implement",
    )
    assert store.claim_command(created.session_id) is not None
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="kimi",
        native_session_id="kimi-native",
        model="default",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )
    store.create_attempt(attempt)
    turn_id = store.start_turn(created.session_id, attempt.attempt_id)
    pre_dispatch = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="kimi",
        native_session_id="kimi-native",
        base_commit=_git(workspace, "rev-parse", "HEAD"),
        patch_digest="pre-patch",
        untracked_digest="pre-untracked",
        context_digest="pre-context",
        created_at=utc_now(),
    )
    store.add_checkpoint(pre_dispatch)
    store.record_dispatch_checkpoint(
        command.command_id,
        attempt.attempt_id,
        turn_id,
        pre_dispatch.checkpoint_id,
    )
    store.mark_provider_boundary(attempt.attempt_id)
    (workspace / "implemented.txt").write_text(
        "productive implementation\n",
        encoding="utf-8",
    )
    material_digest, unused_summary = inspect_workspace(workspace)
    del unused_summary
    certified = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=pre_dispatch.sequence + 1,
        provider="kimi",
        native_session_id="kimi-native",
        base_commit=_git(workspace, "rev-parse", "HEAD"),
        patch_digest="certified-patch",
        untracked_digest="certified-untracked",
        context_digest="certified-context",
        created_at=utc_now(),
    )
    store.add_checkpoint(certified)
    store.append_event(
        created.session_id,
        "checkpoint.created",
        status="complete",
        metadata=certified.as_dict(),
        turn_id=turn_id,
    )
    store.append_event(
        created.session_id,
        "guard.tripped",
        status="failed",
        metadata={
            "command_id": command.command_id,
            "attempt_id": attempt.attempt_id,
            "reason": "context-window",
            "action": "pause",
            "snapshot": {},
            "provider_terminal": True,
            "checkpoint_id": certified.checkpoint_id,
        },
        turn_id=turn_id,
    )
    store.update_attempt(attempt.attempt_id, status="failed")
    store.finish_turn(turn_id, "failed")
    store.complete_dispatch(attempt.attempt_id, "failed")
    store.update_session(
        created.session_id,
        lifecycle="paused",
        attention="needs-input",
    )
    state = TerminalGuardState(
        store,
        created,
        workspace,
        command.command_id,
        attempt.attempt_id,
        turn_id,
        certified,
        material_digest,
    )
    state.set_result()
    return state


def test_terminal_checkpoint_anchor_certifies_a_provider_terminal_guard_stop(
    tmp_path: Path,
) -> None:
    state = _terminal_guard_state(tmp_path, "terminal-anchor")
    anchor = state.anchor()

    assert anchor["eligible"] is True
    assert anchor["reason"] == ""
    assert anchor["prior_anchor_kind"] == "terminal-checkpoint"
    assert anchor["prior_command_type"] == "message"
    assert anchor["prior_command_id"] == state.command_id
    assert anchor["prior_command_status"] == CommandStatus.FAILED
    assert anchor["prior_checkpoint_id"] == state.checkpoint.checkpoint_id
    assert anchor["prior_material_digest"] == state.material_digest
    assert anchor["prior_reconciliation_id"] == ""
    assert anchor["prior_reconciliation_resolution"] == ""
    assert state.store.pending_reconciliations(state.session.session_id) == []
    state.store.close()


def test_terminal_checkpoint_anchor_rejects_every_unproven_boundary(
    tmp_path: Path,
) -> None:
    state = _terminal_guard_state(tmp_path, "terminal-boundary")

    state.set_result(checkpoint_id=new_uuid())
    assert state.reason() == ("dispatch transition terminal checkpoint is not latest")
    state.set_result(workspace_material_digest="short")
    assert state.reason() == "dispatch transition terminal material is invalid"
    state.set_result(workspace_material_digest="f" * 64)
    assert state.reason() == ("dispatch transition terminal material is not current")
    state.set_result(code="E_NEEDS_RECONCILIATION")
    assert state.reason() == "dispatch transition requires exactly one reconciliation"
    state.set_result(provider_terminal=False)
    assert state.reason() == "dispatch transition requires exactly one reconciliation"
    state.set_result(code="E_PROVIDER_NO_PROGRESS")
    assert state.reason() == "dispatch transition prior command is not eligible"
    state.set_result()
    assert state.anchor()["eligible"] is True

    (state.workspace / "drifted.txt").write_text("drift\n", encoding="utf-8")
    assert state.reason() == ("dispatch transition terminal material is not current")
    (state.workspace / "drifted.txt").unlink()
    assert state.anchor()["eligible"] is True

    state.execute(
        """
        INSERT INTO reconciliations VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            new_uuid(),
            state.session.session_id,
            state.command_id,
            state.checkpoint.checkpoint_id,
            state.material_digest,
            "pending material",
            "[]",
            "{}",
            "pending",
            "",
            "{}",
            utc_now(),
            "",
        ),
    )
    assert state.reason() == (
        "dispatch transition terminal result has a reconciliation"
    )
    state.execute(
        "DELETE FROM reconciliations WHERE command_id = ?",
        (state.command_id,),
    )

    state.execute(
        "UPDATE command_dispatches SET crossed_boundary = 0 WHERE command_id = ?",
        (state.command_id,),
    )
    assert state.reason() == "dispatch transition terminal boundary is not exact"
    state.execute(
        "UPDATE command_dispatches SET crossed_boundary = 1 WHERE command_id = ?",
        (state.command_id,),
    )
    other_session = session(state.workspace)
    state.store.create_session(other_session)
    state.execute(
        "UPDATE command_dispatches SET session_id = ? WHERE command_id = ?",
        (other_session.session_id, state.command_id),
    )
    assert state.reason() == ("dispatch transition terminal dispatch session changed")
    state.execute(
        "UPDATE command_dispatches SET session_id = ? WHERE command_id = ?",
        (state.session.session_id, state.command_id),
    )
    state.execute(
        "UPDATE command_dispatches SET state = 'interrupted' WHERE command_id = ?",
        (state.command_id,),
    )
    assert state.reason() == "dispatch transition terminal dispatch is not failed"
    state.execute(
        "UPDATE command_dispatches SET state = 'failed' WHERE command_id = ?",
        (state.command_id,),
    )

    state.execute(
        "UPDATE turns SET session_id = ? WHERE turn_id = ?",
        (other_session.session_id, state.turn_id),
    )
    assert state.reason() == "dispatch transition terminal turn is unknown"
    state.execute(
        "UPDATE turns SET session_id = ? WHERE turn_id = ?",
        (state.session.session_id, state.turn_id),
    )
    state.execute(
        "UPDATE turns SET status = 'ambiguous' WHERE turn_id = ?",
        (state.turn_id,),
    )
    assert state.reason() == "dispatch transition terminal turn is not failed"
    state.execute(
        "UPDATE turns SET status = 'failed' WHERE turn_id = ?",
        (state.turn_id,),
    )

    state.execute(
        """
        UPDATE events SET metadata_json = ?
        WHERE session_id = ? AND event_type = 'guard.tripped'
        """,
        (
            json.dumps({"command_id": state.command_id}, sort_keys=True),
            state.session.session_id,
        ),
    )
    assert state.reason() == "dispatch transition terminal guard receipt changed"
    state.execute(
        """
        UPDATE events SET metadata_json = ?
        WHERE session_id = ? AND event_type = 'guard.tripped'
        """,
        (
            json.dumps(
                {
                    "command_id": state.command_id,
                    "attempt_id": state.attempt_id,
                    "checkpoint_id": state.checkpoint.checkpoint_id,
                    "provider_terminal": True,
                },
                sort_keys=True,
            ),
            state.session.session_id,
        ),
    )
    assert state.anchor()["eligible"] is True

    state.execute(
        "DELETE FROM events WHERE session_id = ? AND event_type = 'checkpoint.created'",
        (state.session.session_id,),
    )
    assert state.reason() == ("dispatch transition terminal checkpoint receipt changed")
    state.store.close()


def test_needs_reconciliation_still_requires_one_resolved_reconciliation(
    tmp_path: Path,
) -> None:
    state = _terminal_guard_state(tmp_path, "terminal-reconciliation")
    state.set_result(
        code="E_NEEDS_RECONCILIATION",
        provider_terminal=True,
    )
    reconciliation_id = new_uuid()
    state.execute(
        """
        INSERT INTO reconciliations VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            reconciliation_id,
            state.session.session_id,
            state.command_id,
            state.checkpoint.checkpoint_id,
            state.material_digest,
            "ambiguous material",
            "[]",
            "{}",
            "pending",
            "",
            "{}",
            utc_now(),
            "",
        ),
    )

    assert state.reason() == "dispatch transition reconciliation is unresolved"
    state.execute(
        """
        UPDATE reconciliations SET status = 'resolved', resolution = 'accept-current',
            audit_json = ?, resolved_at = ?
        WHERE reconciliation_id = ?
        """,
        (
            json.dumps(
                {
                    "resolution_checkpoint_id": state.checkpoint.checkpoint_id,
                    "resolution_workspace_digest": state.material_digest,
                },
                sort_keys=True,
            ),
            utc_now(),
            reconciliation_id,
        ),
    )
    anchor = state.anchor()

    assert anchor["eligible"] is True
    assert anchor["prior_anchor_kind"] == "resolved-reconciliation"
    assert anchor["prior_reconciliation_id"] == reconciliation_id
    assert anchor["prior_reconciliation_resolution"] == "accept-current"
    state.store.close()


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
    with pytest.raises(ConflictError, match="prior context delivery"):
        store.prepare_context_delivery(
            created.session_id,
            "codex",
            "context-b",
            "checkpoint-b",
            "command-a",
            "attempt-b",
            "payload-b",
        )
    store.close()

    recovered = StateStore(database)
    delivered = recovered.accept_context_delivery(
        created.session_id,
        "codex",
        "context-a",
        "attempt-a",
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


_PRE_DISPATCH_MATERIAL = {
    "base_commit": "base",
    "patch_digest": "patch",
    "untracked_digest": "untracked",
    "context_digest": "context",
}
# The default accepted workspace carries work the dispatch did not
# start with, which is what separates a landed implementation from a
# provider that merely stopped talking.
_IMPLEMENTED_MATERIAL = {
    **_PRE_DISPATCH_MATERIAL,
    "patch_digest": "patch-implemented",
}


def _ambiguous_live_dispatch(
    tmp_path: Path,
    *,
    native_session_id: str = "native-live",
    record_completion: bool = True,
    record_message: bool = True,
    message_after_completion: bool = False,
    contradiction: tuple[str, str] | None = None,
    resolution_material: dict[str, str] | None = None,
) -> SimpleNamespace:
    """Reproduce a crossed-boundary dispatch whose turn already finished."""
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "live transport step"},
        "live-transport",
    )
    store.create_command_envelope(
        command.command_id,
        created.session_id,
        "unattended",
        {"max_dollars": 0.0},
    )
    assert store.claim_command(created.session_id) is not None
    now = utc_now()
    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="claude",
        native_session_id=native_session_id,
        model="sonnet",
        effort="low",
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
        native_session_id=native_session_id,
        created_at=now,
        **_PRE_DISPATCH_MATERIAL,
    )
    store.add_checkpoint(checkpoint)
    store.record_dispatch_checkpoint(
        command.command_id,
        attempt.attempt_id,
        turn_id,
        checkpoint.checkpoint_id,
    )
    store.mark_provider_boundary(attempt.attempt_id)
    text = "live-transport/claude-live: completed - nonce ba5e0a97418e71ad"
    if record_message and not message_after_completion:
        store.append_event(
            created.session_id,
            "agent.message",
            role="assistant",
            text=text,
            turn_id=turn_id,
        )
    if record_completion:
        store.append_event(
            created.session_id,
            "turn.completed",
            status="complete",
            text=text,
            turn_id=turn_id,
        )
    if record_message and message_after_completion:
        store.append_event(
            created.session_id,
            "agent.message",
            role="assistant",
            text=text,
            turn_id=turn_id,
        )
    if contradiction is not None:
        contradiction_type, contradiction_status = contradiction
        store.append_event(
            created.session_id,
            contradiction_type,
            status=contradiction_status,
            turn_id=turn_id,
        )
    recovery = store.recover_interrupted_commands(
        created.session_id,
        "digest-current",
        "summary-current",
    )
    assert len(recovery.reconciliations) == 1
    record = recovery.reconciliations[0]
    assert store.get_command(command.command_id).status == CommandStatus.FAILED
    material = dict(_IMPLEMENTED_MATERIAL)
    if resolution_material is not None:
        material = dict(resolution_material)
    resolution = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=created.session_id,
        sequence=store.last_sequence(created.session_id),
        provider="claude",
        native_session_id="",
        created_at=utc_now(),
        **material,
    )
    store.add_checkpoint(resolution)
    return SimpleNamespace(
        store=store,
        session=created,
        command=command,
        attempt=attempt,
        turn_id=turn_id,
        record=record,
        text=text,
        resolution_checkpoint_id=resolution.checkpoint_id,
        workspace_digest="b" * 64,
    )


def _resolution_audit(rig: SimpleNamespace) -> dict[str, Any]:
    return {
        "actor": "test",
        "checkpoint_id": rig.resolution_checkpoint_id,
        "resolution_checkpoint_id": rig.resolution_checkpoint_id,
        "resolution_workspace_digest": rig.workspace_digest,
    }


def _topology(rig: SimpleNamespace, resolved: Any) -> dict[str, Any]:
    del rig
    receipt = resolved.audit["topology_receipt"]
    assert isinstance(receipt, dict)
    return receipt


def test_accept_current_projects_the_recorded_turn_completion(
    tmp_path: Path,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store

    resolved = store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )

    completed = store.get_command(rig.command.command_id)
    assert completed.status == CommandStatus.COMPLETE
    assert completed.result["status"] == "complete"
    assert completed.result["native_session_id"] == "native-live"
    assert completed.result["turn_id"] == rig.turn_id
    assert completed.result["provider"] == "claude"
    assert completed.result["model"] == "sonnet"
    assert completed.result["effort"] == "low"
    assert completed.result["checkpoint_id"] == rig.resolution_checkpoint_id
    assert completed.result["workspace_material_digest"] == rig.workspace_digest
    assert completed.result["reconciled_resolution"] == "accept-current"
    assert completed.result["reconciliation_id"] == rig.record.reconciliation_id
    assert completed.result["final_message_sha256"] == hashlib.sha256(
        rig.text.encode("utf-8")
    ).hexdigest()

    attempts = store.attempts(rig.session.session_id)
    assert [item.status for item in attempts] == ["complete"]
    assert attempts[0].native_session_id == "native-live"
    envelope = store.command_envelope(rig.command.command_id)
    assert envelope["state"] == "complete"
    assert envelope["guard_reason"] == ""
    with store.transaction() as connection:
        row = connection.execute(
            """
            SELECT turns.status AS turn_state,
                command_dispatches.state AS dispatch_state
            FROM turns JOIN command_dispatches USING(turn_id)
            WHERE command_dispatches.attempt_id = ?
            """,
            (rig.attempt.attempt_id,),
        ).fetchone()
    assert row["turn_state"] == "complete"
    assert row["dispatch_state"] == "complete"

    receipt = _topology(rig, resolved)
    assert receipt["attempt_state"] == "complete"
    assert receipt["turn_state"] == "complete"
    assert receipt["dispatch_state"] == "complete"
    assert receipt["envelope_state"] == "complete"
    assert receipt["guard_reason"] == ""
    assert receipt["command_status"] == "complete"
    evidence = receipt["completion_evidence"]
    assert evidence["native_session_id"] == "native-live"
    assert evidence["attempt_id"] == rig.attempt.attempt_id
    assert evidence["final_message_sequence"] < evidence["turn_completed_sequence"]

    snapshot = proof_snapshot(store, rig.session.session_id)
    projected = [
        item
        for item in snapshot["commands"]
        if item["command_id"] == rig.command.command_id
    ]
    assert len(projected) == 1
    assert projected[0]["status"] == "complete"
    assert projected[0]["result"]["native_session_id"] == "native-live"
    assert projected[0]["result"]["final_message_sha256"] == (
        completed.result["final_message_sha256"]
    )
    assert [item["status"] for item in snapshot["attempts"]] == ["complete"]
    assert [item["status"] for item in snapshot["turns"]] == ["complete"]
    store.close()


def test_accept_current_projection_survives_the_idempotent_resolution_path(
    tmp_path: Path,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store

    resolved, created = store.resolve_reconciliation_once(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
        None,
        idempotency_key="resolve-once",
        operation="reconciliation-resolve",
        request_digest="request",
    )

    assert created is True
    assert resolved.status == "resolved"
    assert _topology(rig, resolved)["command_status"] == "complete"
    assert store.get_command(rig.command.command_id).status == CommandStatus.COMPLETE
    replayed, replay_created = store.resolve_reconciliation_once(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
        None,
        idempotency_key="resolve-once",
        operation="reconciliation-resolve",
        request_digest="request",
    )
    assert replay_created is False
    assert replayed == resolved
    assert store.get_command(rig.command.command_id).status == CommandStatus.COMPLETE
    store.close()


@pytest.mark.parametrize(
    "decision,kwargs",
    [
        ("accept-current", {"record_completion": False}),
        ("accept-current", {"record_message": False}),
        ("accept-current", {"native_session_id": ""}),
        ("accept-current", {"message_after_completion": True}),
        # A single matching completion is not proof when the same turn
        # also ended some other way. The last case is the observed Kimi
        # shape: a complete-looking resume hint, then turn.failed.
        ("accept-current", {"contradiction": ("turn.completed", "interrupted")}),
        ("accept-current", {"contradiction": ("turn.interrupted", "interrupted")}),
        ("accept-current", {"contradiction": ("provider.error", "failed")}),
        ("accept-current", {"contradiction": ("turn.failed", "failed")}),
        ("restore-pre-turn", {}),
        ("stop", {}),
    ],
)
def test_reconciliation_without_proven_completion_stays_ambiguous(
    tmp_path: Path,
    decision: str,
    kwargs: dict[str, Any],
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path, **kwargs)
    store = rig.store

    resolved = store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        decision,
        "digest-current",
        _resolution_audit(rig),
    )

    failed = store.get_command(rig.command.command_id)
    assert failed.status == CommandStatus.FAILED
    assert failed.result["code"] == "E_NEEDS_RECONCILIATION"
    assert [item.status for item in store.attempts(rig.session.session_id)] == [
        "ambiguous"
    ]
    envelope = store.command_envelope(rig.command.command_id)
    assert envelope["state"] == "paused"
    assert envelope["guard_reason"] == "ambiguous-provider-dispatch"
    receipt = _topology(rig, resolved)
    assert receipt["attempt_state"] == "ambiguous"
    assert receipt["turn_state"] == "ambiguous"
    assert receipt["dispatch_state"] == "ambiguous"
    assert "command_status" not in receipt
    assert "completion_evidence" not in receipt
    with store.transaction() as connection:
        dispatch = connection.execute(
            "SELECT state FROM command_dispatches WHERE command_id = ?",
            (rig.command.command_id,),
        ).fetchone()
    assert dispatch["state"] == "ambiguous"
    store.close()


@pytest.mark.parametrize(
    "audit",
    [
        {"actor": "test"},
        {"actor": "test", "resolution_workspace_digest": "b" * 64},
        {"actor": "test", "resolution_checkpoint_id": "checkpoint", "digest": ""},
    ],
)
def test_accept_current_without_certified_material_stays_ambiguous(
    tmp_path: Path,
    audit: dict[str, Any],
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store

    resolved = store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        audit,
    )

    assert store.get_command(rig.command.command_id).status == CommandStatus.FAILED
    assert _topology(rig, resolved)["attempt_state"] == "ambiguous"
    store.close()


@pytest.mark.parametrize(
    "material",
    [
        # The accepted workspace is exact-clean against the tree the
        # dispatch started from.
        _PRE_DISPATCH_MATERIAL,
        # Context and native identity move without any implementation
        # landing, so neither is material.
        {**_PRE_DISPATCH_MATERIAL, "context_digest": "context-after"},
    ],
)
def test_accept_current_without_moved_material_stays_ambiguous(
    tmp_path: Path,
    material: dict[str, str],
) -> None:
    """A clean provider turn over unchanged material is not completion."""
    rig = _ambiguous_live_dispatch(tmp_path, resolution_material=material)
    store = rig.store

    resolved = store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )

    failed = store.get_command(rig.command.command_id)
    assert failed.status == CommandStatus.FAILED
    assert failed.result["code"] == "E_NEEDS_RECONCILIATION"
    assert [item.status for item in store.attempts(rig.session.session_id)] == [
        "ambiguous"
    ]
    envelope = store.command_envelope(rig.command.command_id)
    assert envelope["state"] == "paused"
    assert envelope["guard_reason"] == "ambiguous-provider-dispatch"
    receipt = _topology(rig, resolved)
    assert receipt["attempt_state"] == "ambiguous"
    assert receipt["turn_state"] == "ambiguous"
    assert receipt["dispatch_state"] == "ambiguous"
    assert "command_status" not in receipt
    assert "completion_evidence" not in receipt
    with store.transaction() as connection:
        dispatch = connection.execute(
            "SELECT state FROM command_dispatches WHERE command_id = ?",
            (rig.command.command_id,),
        ).fetchone()
    assert dispatch["state"] == "ambiguous"
    store.close()


def test_accept_current_requires_a_bound_pre_dispatch_anchor(
    tmp_path: Path,
) -> None:
    """Material moved against a foreign anchor proves nothing here."""
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store
    other = session(tmp_path / "other")
    store.create_session(other)
    foreign = Checkpoint(
        checkpoint_id=new_uuid(),
        session_id=other.session_id,
        sequence=store.last_sequence(other.session_id),
        provider="claude",
        native_session_id="",
        created_at=utc_now(),
        **_PRE_DISPATCH_MATERIAL,
    )
    store.add_checkpoint(foreign)
    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE reconciliations SET pre_dispatch_checkpoint_id = ?
            WHERE reconciliation_id = ?
            """,
            (foreign.checkpoint_id, rig.record.reconciliation_id),
        )

    resolved = store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )

    assert store.get_command(rig.command.command_id).status == CommandStatus.FAILED
    receipt = _topology(rig, resolved)
    assert receipt["attempt_state"] == "ambiguous"
    assert "completion_evidence" not in receipt
    store.close()


@pytest.mark.parametrize(
    "field",
    ["base_commit", "patch_digest", "untracked_digest"],
)
def test_accept_current_accepts_any_moved_material_field(
    tmp_path: Path,
    field: str,
) -> None:
    rig = _ambiguous_live_dispatch(
        tmp_path,
        resolution_material={**_PRE_DISPATCH_MATERIAL, field: "moved"},
    )
    store = rig.store

    resolved = store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )

    assert store.get_command(rig.command.command_id).status == CommandStatus.COMPLETE
    assert _topology(rig, resolved)["command_status"] == "complete"
    store.close()


@pytest.mark.parametrize(
    "status,result",
    [
        (CommandStatus.FAILED, {"reconciliation_id": "other"}),
        (CommandStatus.CANCELLED, {}),
    ],
)
def test_accept_current_declines_a_command_it_does_not_own(
    tmp_path: Path,
    status: str,
    result: dict[str, Any],
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store
    payload = dict(result)
    if payload.get("reconciliation_id") == "other":
        payload["reconciliation_id"] = new_uuid()
    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE commands SET status = ?, result_json = ?
            WHERE command_id = ?
            """,
            (status, json.dumps(payload), rig.command.command_id),
        )

    resolved = store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )

    assert store.get_command(rig.command.command_id).status == status
    assert _topology(rig, resolved)["attempt_state"] == "ambiguous"
    store.close()


def _strand_resolved_reconciliation(rig: SimpleNamespace) -> None:
    """Rewrite the resolution the way the pre-projection build left it."""
    store = rig.store
    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE provider_attempts SET status = 'ambiguous'
            WHERE attempt_id = ?
            """,
            (rig.attempt.attempt_id,),
        )
        connection.execute(
            "UPDATE turns SET status = 'ambiguous' WHERE turn_id = ?",
            (rig.turn_id,),
        )
        connection.execute(
            """
            UPDATE command_dispatches SET state = 'ambiguous'
            WHERE attempt_id = ?
            """,
            (rig.attempt.attempt_id,),
        )
        connection.execute(
            """
            UPDATE command_envelopes SET state = 'paused',
                guard_reason = 'ambiguous-provider-dispatch'
            WHERE command_id = ?
            """,
            (rig.command.command_id,),
        )
        connection.execute(
            """
            UPDATE commands SET status = ?, result_json = ?
            WHERE command_id = ?
            """,
            (
                CommandStatus.FAILED,
                json.dumps(
                    {
                        "code": "E_SAFETY_GUARD",
                        "message": "execution safety guard stopped claude: dollars",
                        "reconciliation_id": rig.record.reconciliation_id,
                    }
                ),
                rig.command.command_id,
            ),
        )


def test_resolved_reconciliation_projects_a_stranded_completion(
    tmp_path: Path,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store
    store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )
    _strand_resolved_reconciliation(rig)
    assert store.get_command(rig.command.command_id).status == CommandStatus.FAILED

    projected = store.project_resolved_reconciliation(rig.record.reconciliation_id)

    assert projected is not None
    assert projected.status == CommandStatus.COMPLETE
    assert projected.result["native_session_id"] == "native-live"
    assert projected.result["final_message_sha256"] == hashlib.sha256(
        rig.text.encode("utf-8")
    ).hexdigest()
    assert [item.status for item in store.attempts(rig.session.session_id)] == [
        "complete"
    ]
    envelope = store.command_envelope(rig.command.command_id)
    assert envelope["state"] == "complete"
    assert envelope["guard_reason"] == ""
    receipt = store.reconciliation(rig.record.reconciliation_id).audit[
        "topology_receipt"
    ]
    assert receipt["command_status"] == "complete"
    assert receipt["prior_guard_reason"] == "ambiguous-provider-dispatch"

    # The prior failure is preserved as a code and a digest, never as
    # the raw guard or provider prose.
    assert projected.result["prior_code"] == "E_SAFETY_GUARD"
    assert "prior_message" not in projected.result
    assert "dollars" not in json.dumps(projected.result)
    assert projected.result["prior_result_sha256"] == hashlib.sha256(
        json.dumps(
            {
                "code": "E_SAFETY_GUARD",
                "message": "execution safety guard stopped claude: dollars",
                "reconciliation_id": rig.record.reconciliation_id,
            }
        ).encode("utf-8")
    ).hexdigest()

    # The transition is observable, explains the digest change at the
    # new through_sequence, and leaves the resolution event untouched.
    events = store.all_events(rig.session.session_id)
    projected_events = [
        item for item in events if item.event_type == "reconciliation.projected"
    ]
    assert len(projected_events) == 1
    transition = projected_events[0]
    assert transition.sequence == events[-1].sequence
    assert transition.metadata["reconciliation_id"] == rig.record.reconciliation_id
    assert transition.metadata["command_id"] == rig.command.command_id
    assert transition.metadata["decision"] == "accept-current"
    assert transition.metadata["resolution_checkpoint_id"] == (
        rig.resolution_checkpoint_id
    )
    assert transition.metadata["resolution_workspace_digest"] == rig.workspace_digest
    assert transition.metadata["topology_receipt"] == receipt
    resolved_events = [
        item for item in events if item.event_type == "reconciliation.resolved"
    ]
    assert len(resolved_events) <= 1

    assert store.project_resolved_reconciliation(rig.record.reconciliation_id) is None
    assert store.get_command(rig.command.command_id).status == CommandStatus.COMPLETE
    repeated = [
        item
        for item in store.all_events(rig.session.session_id)
        if item.event_type == "reconciliation.projected"
    ]
    assert len(repeated) == 1
    assert repeated[0].event_id == transition.event_id
    store.close()


def test_projection_declines_a_resolution_without_moved_material(
    tmp_path: Path,
) -> None:
    """A stranded command stays stranded when the workspace never moved."""
    rig = _ambiguous_live_dispatch(
        tmp_path,
        resolution_material=_PRE_DISPATCH_MATERIAL,
    )
    store = rig.store
    store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )
    _strand_resolved_reconciliation(rig)

    assert store.project_resolved_reconciliation(rig.record.reconciliation_id) is None

    stranded = store.get_command(rig.command.command_id)
    assert stranded.status == CommandStatus.FAILED
    assert stranded.result["code"] == "E_SAFETY_GUARD"
    assert [item.status for item in store.attempts(rig.session.session_id)] == [
        "ambiguous"
    ]
    envelope = store.command_envelope(rig.command.command_id)
    assert envelope["state"] == "paused"
    assert envelope["guard_reason"] == "ambiguous-provider-dispatch"
    receipt = store.reconciliation(rig.record.reconciliation_id).audit[
        "topology_receipt"
    ]
    assert "command_status" not in receipt
    assert "completion_evidence" not in receipt
    assert not [
        item
        for item in store.all_events(rig.session.session_id)
        if item.event_type == "reconciliation.projected"
    ]
    store.close()


def test_restranded_replay_reuses_the_recorded_projection_event(
    tmp_path: Path,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store
    store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )
    _strand_resolved_reconciliation(rig)
    first = store.project_resolved_reconciliation(rig.record.reconciliation_id)
    assert first is not None
    transition = [
        item
        for item in store.all_events(rig.session.session_id)
        if item.event_type == "reconciliation.projected"
    ][0]
    through_sequence = store.last_sequence(rig.session.session_id)

    # The build that stranded this command runs once more and strands
    # it again, so the replay has a real topology to settle and reaches
    # the append with its own transition already in the log.
    _strand_resolved_reconciliation(rig)
    assert store.get_command(rig.command.command_id).status == CommandStatus.FAILED

    replayed = store.project_resolved_reconciliation(rig.record.reconciliation_id)

    # The settle runs again and lands on the same answer.
    assert replayed is not None
    assert replayed.status == CommandStatus.COMPLETE
    assert replayed.result == first.result
    assert [item.status for item in store.attempts(rig.session.session_id)] == [
        "complete"
    ]
    envelope = store.command_envelope(rig.command.command_id)
    assert envelope["state"] == "complete"
    assert envelope["guard_reason"] == ""

    # The already-recorded transition is reused, not duplicated, so the
    # proof through_sequence and the event it explains both hold still.
    repeated = [
        item
        for item in store.all_events(rig.session.session_id)
        if item.event_type == "reconciliation.projected"
    ]
    assert repeated == [transition]
    assert store.last_sequence(rig.session.session_id) == through_sequence
    store.close()


def test_projection_event_moves_the_proof_through_sequence(
    tmp_path: Path,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store
    store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )
    _strand_resolved_reconciliation(rig)
    before = proof_snapshot(store, rig.session.session_id)

    store.project_resolved_reconciliation(rig.record.reconciliation_id)

    after = proof_snapshot(store, rig.session.session_id)
    before_through = before["event_range"]["through_sequence"]
    assert after["event_range"]["through_sequence"] > before_through
    appended = [
        item for item in after["events"] if item["sequence"] > before_through
    ]
    assert [item["event_type"] for item in appended] == ["reconciliation.projected"]
    assert after["snapshot_digest"] != before["snapshot_digest"]
    projected_command = [
        item
        for item in after["commands"]
        if item["command_id"] == rig.command.command_id
    ][0]
    assert projected_command["status"] == "complete"
    assert projected_command["result"]["prior_code"] == "E_SAFETY_GUARD"
    assert projected_command["result"]["prior_result_sha256"]
    assert "message" not in projected_command["result"]
    store.close()


@pytest.mark.parametrize(
    "tamper",
    ["second-boundary", "missing-receipt", "foreign-receipt", "absent-checkpoint"],
)
def test_projection_requires_its_bound_topology(
    tmp_path: Path,
    tamper: str,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store
    store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )
    _strand_resolved_reconciliation(rig)
    record = store.reconciliation(rig.record.reconciliation_id)
    audit = dict(record.audit)
    with store.transaction() as connection:
        if tamper == "second-boundary":
            second_attempt = ProviderAttempt(
                attempt_id=new_uuid(),
                session_id=rig.session.session_id,
                provider="claude",
                native_session_id="native-second",
                model="sonnet",
                effort="low",
                auth_mode="subscription",
                status="running",
                started_at=utc_now(),
                ended_at="",
            )
            connection.execute(
                """
                INSERT INTO provider_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    second_attempt.attempt_id,
                    second_attempt.session_id,
                    second_attempt.provider,
                    second_attempt.native_session_id,
                    second_attempt.model,
                    second_attempt.effort,
                    second_attempt.auth_mode,
                    second_attempt.status,
                    second_attempt.started_at,
                    second_attempt.ended_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO command_dispatches(
                    attempt_id, command_id, session_id, turn_id,
                    checkpoint_id, crossed_boundary, state,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, 'ambiguous', ?, ?)
                """,
                (
                    second_attempt.attempt_id,
                    rig.command.command_id,
                    rig.session.session_id,
                    rig.turn_id,
                    rig.resolution_checkpoint_id,
                    utc_now(),
                    utc_now(),
                ),
            )
        if tamper == "missing-receipt":
            audit.pop("topology_receipt", None)
            connection.execute(
                "UPDATE reconciliations SET audit_json = ? WHERE reconciliation_id = ?",
                (json.dumps(audit), rig.record.reconciliation_id),
            )
        if tamper == "foreign-receipt":
            receipt = dict(audit["topology_receipt"])
            receipt["attempt_id"] = new_uuid()
            audit["topology_receipt"] = receipt
            connection.execute(
                "UPDATE reconciliations SET audit_json = ? WHERE reconciliation_id = ?",
                (json.dumps(audit), rig.record.reconciliation_id),
            )
        if tamper == "absent-checkpoint":
            connection.execute(
                "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                (rig.resolution_checkpoint_id,),
            )

    assert store.project_resolved_reconciliation(rig.record.reconciliation_id) is None

    assert store.get_command(rig.command.command_id).status == CommandStatus.FAILED
    bound = [
        item
        for item in store.attempts(rig.session.session_id)
        if item.attempt_id == rig.attempt.attempt_id
    ]
    assert [item.status for item in bound] == ["ambiguous"]
    assert store.command_envelope(rig.command.command_id)["state"] == "paused"
    assert not [
        item
        for item in store.all_events(rig.session.session_id)
        if item.event_type == "reconciliation.projected"
    ]
    store.close()


def test_manager_replay_projects_a_stranded_completion(
    tmp_path: Path,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store
    blobs = BlobStore(tmp_path / "blobs")
    manager = ReconciliationManager(store, blobs)
    audit = _resolution_audit(rig)
    store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        audit,
    )
    _strand_resolved_reconciliation(rig)

    replayed = asyncio.run(
        manager.resolve(
            rig.record.reconciliation_id,
            "accept-current",
            "digest-current",
        )
    )

    assert replayed.status == "resolved"
    assert store.get_command(rig.command.command_id).status == CommandStatus.COMPLETE
    # The returned record must describe stored state, not the copy read
    # before the projection rewrote the receipt.
    returned_receipt = replayed.audit["topology_receipt"]
    assert returned_receipt["command_status"] == "complete"
    assert returned_receipt["attempt_state"] == "complete"
    assert returned_receipt["turn_state"] == "complete"
    assert returned_receipt["dispatch_state"] == "complete"
    assert returned_receipt["envelope_state"] == "complete"
    assert returned_receipt["completion_evidence"]["native_session_id"] == (
        "native-live"
    )
    assert returned_receipt == store.reconciliation(
        rig.record.reconciliation_id
    ).audit["topology_receipt"]
    store.close()


def test_conflicting_replay_key_leaves_a_stranded_command_untouched(
    tmp_path: Path,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store
    blobs = BlobStore(tmp_path / "blobs")
    manager = ReconciliationManager(store, blobs)
    store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )
    _strand_resolved_reconciliation(rig)
    store.idempotent_mutation(
        "shared-key",
        "other-operation",
        "other-digest",
        lambda: {"unrelated": True},
        200,
    )

    with pytest.raises(ConflictError):
        asyncio.run(
            manager.resolve(
                rig.record.reconciliation_id,
                "accept-current",
                "digest-current",
                idempotency_key="shared-key",
                operation="reconciliation-resolve",
                request_digest="request",
            )
        )

    stranded = store.get_command(rig.command.command_id)
    assert stranded.status == CommandStatus.FAILED
    assert stranded.result["code"] == "E_SAFETY_GUARD"
    assert [item.status for item in store.attempts(rig.session.session_id)] == [
        "ambiguous"
    ]
    assert store.command_envelope(rig.command.command_id)["state"] == "paused"

    resolved = asyncio.run(
        manager.resolve(
            rig.record.reconciliation_id,
            "accept-current",
            "digest-current",
            idempotency_key="fresh-key",
            operation="reconciliation-resolve",
            request_digest="request",
        )
    )
    assert resolved.status == "resolved"
    assert store.get_command(rig.command.command_id).status == CommandStatus.COMPLETE
    store.close()


def test_replayed_receipt_does_not_reproject_a_settled_command(
    tmp_path: Path,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path)
    store = rig.store
    blobs = BlobStore(tmp_path / "blobs")
    manager = ReconciliationManager(store, blobs)
    store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )
    first = asyncio.run(
        manager.resolve(
            rig.record.reconciliation_id,
            "accept-current",
            "digest-current",
            idempotency_key="replay-key",
            operation="reconciliation-resolve",
            request_digest="request",
        )
    )
    _strand_resolved_reconciliation(rig)

    replayed = asyncio.run(
        manager.resolve(
            rig.record.reconciliation_id,
            "accept-current",
            "digest-current",
            idempotency_key="replay-key",
            operation="reconciliation-resolve",
            request_digest="request",
        )
    )

    assert replayed == first
    assert store.get_command(rig.command.command_id).status == CommandStatus.FAILED
    store.close()


def test_resolved_reconciliation_projection_declines_unprovable_work(
    tmp_path: Path,
) -> None:
    rig = _ambiguous_live_dispatch(tmp_path, record_completion=False)
    store = rig.store
    store.resolve_reconciliation_record(
        rig.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(rig),
    )

    assert store.project_resolved_reconciliation(rig.record.reconciliation_id) is None
    assert store.get_command(rig.command.command_id).status == CommandStatus.FAILED

    stopped = _ambiguous_live_dispatch(tmp_path / "stopped")
    stopped.store.resolve_reconciliation_record(
        stopped.record.reconciliation_id,
        "stop",
        "digest-current",
        _resolution_audit(stopped),
    )
    assert (
        stopped.store.project_resolved_reconciliation(
            stopped.record.reconciliation_id
        )
        is None
    )

    missing_attempt = _ambiguous_live_dispatch(tmp_path / "detached")
    missing_attempt.store.resolve_reconciliation_record(
        missing_attempt.record.reconciliation_id,
        "accept-current",
        "digest-current",
        _resolution_audit(missing_attempt),
    )
    with missing_attempt.store.transaction() as connection:
        connection.execute(
            """
            UPDATE reconciliations SET audit_json = ?
            WHERE reconciliation_id = ?
            """,
            (
                json.dumps({"dispatch_identity": {"attempt_id": new_uuid()}}),
                missing_attempt.record.reconciliation_id,
            ),
        )
    assert (
        missing_attempt.store.project_resolved_reconciliation(
            missing_attempt.record.reconciliation_id
        )
        is None
    )

    with pytest.raises(NotFoundError):
        store.project_resolved_reconciliation(new_uuid())
    store.close()
    stopped.store.close()
    missing_attempt.store.close()


def test_active_turn_seconds_and_countable_turn_count(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
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
    completed_turn = store.start_turn(created.session_id, attempt.attempt_id)
    store.finish_turn(completed_turn, "complete")
    running_turn = store.start_turn(created.session_id, attempt.attempt_id)
    dead_turn = store.start_turn(created.session_id, attempt.attempt_id)
    store.finish_turn(dead_turn, "no-progress")
    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE turns SET started_at = ?, completed_at = ?
            WHERE turn_id = ?
            """,
            (
                "2026-08-05T10:00:00+00:00",
                "2026-08-05T10:00:30+00:00",
                completed_turn,
            ),
        )
        connection.execute(
            """
            UPDATE turns SET started_at = ?, completed_at = ''
            WHERE turn_id = ?
            """,
            ("2026-08-05T10:05:00+00:00", running_turn),
        )
        connection.execute(
            """
            UPDATE turns SET started_at = ?, completed_at = ?
            WHERE turn_id = ?
            """,
            (
                "2026-08-05T10:10:00+00:00",
                "2026-08-05T10:12:00+00:00",
                dead_turn,
            ),
        )

    now = datetime.datetime(2026, 8, 5, 10, 6, 0, tzinfo=datetime.UTC)
    assert store.turn_count(created.session_id) == 3
    assert store.countable_turn_count(created.session_id) == 2
    assert store.active_turn_seconds(created.session_id, now) == 90.0
    store.close()


def _command_rows(store: StateStore, session_id: str) -> list[tuple[str, str]]:
    with store.transaction() as connection:
        rows = connection.execute(
            """
            SELECT command_type, status FROM commands
            WHERE session_id = ? ORDER BY created_at, command_id
            """,
            (session_id,),
        ).fetchall()
    return [(str(row["command_type"]), str(row["status"])) for row in rows]


def test_terminal_sessions_admit_only_a_stopped_session_resume(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    racing = store.enqueue_command(
        created.session_id,
        "interrupt",
        {},
        "interrupt-racing-the-stop",
    )
    store.stop_session(created.session_id)
    cancelled = store.get_command(racing.command_id)
    assert cancelled.status == CommandStatus.CANCELLED
    assert cancelled.result["code"] == "E_SESSION_STOPPED"
    assert cancelled.result["accepted"] is False

    for command_type in ("interrupt", "pause", "steer", "stop"):
        with pytest.raises(ConflictError, match="admits only a resume command"):
            store.enqueue_command(
                created.session_id,
                command_type,
                {},
                "stopped-" + command_type,
            )
    with pytest.raises(ConflictError, match="admits only a resume command"):
        store.ensure_message_command(
            created.session_id,
            {"text": "after the stop"},
            "stopped-message",
        )
    assert _command_rows(store, created.session_id) == [
        ("interrupt", CommandStatus.CANCELLED),
    ]
    assert store.claim_command(created.session_id) is None

    resume = store.enqueue_command(
        created.session_id,
        "resume",
        {},
        "stopped-resume",
    )
    assert resume.status == CommandStatus.QUEUED
    repeated = store.enqueue_command(
        created.session_id,
        "resume",
        {},
        "stopped-resume",
    )
    assert repeated.command_id == resume.command_id
    assert _command_rows(store, created.session_id) == [
        ("interrupt", CommandStatus.CANCELLED),
        ("resume", CommandStatus.QUEUED),
    ]

    for lifecycle in ("completed", "failed"):
        store.update_session(created.session_id, lifecycle=lifecycle)
        with pytest.raises(ConflictError, match="admits no commands"):
            store.enqueue_command(
                created.session_id,
                "resume",
                {},
                lifecycle + "-resume",
            )
    store.close()


def test_a_terminal_session_never_requeues_a_retryable_failure(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    live = session(tmp_path)
    stopped = session(tmp_path)
    store.create_session(live)
    store.create_session(stopped)
    for created in (live, stopped):
        command = store.enqueue_command(
            created.session_id,
            "message",
            {"text": "retry me"},
            "retryable-" + created.session_id,
        )
        store.resolve_command(
            command.command_id,
            CommandStatus.FAILED,
            {"retryable": True},
        )
    store.stop_session(stopped.session_id)

    requeued = store.enqueue_command(
        live.session_id,
        "message",
        {"text": "retry me"},
        "retryable-" + live.session_id,
    )
    assert requeued.status == CommandStatus.QUEUED

    held = store.enqueue_command(
        stopped.session_id,
        "message",
        {"text": "retry me"},
        "retryable-" + stopped.session_id,
    )
    assert held.status == CommandStatus.FAILED
    assert store.claim_command(stopped.session_id) is None
    store.close()


def test_retire_worker_holds_a_worker_for_a_queued_stopped_session_control(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    store.stop_session(created.session_id)
    store.register_worker(created.session_id, 4321, "incarnation-one")
    assert store.worker_registered(created.session_id) is True
    assert (
        store.retire_worker(
            created.session_id,
            "incarnation-two",
            storage_module.STOPPED_SESSION_COMMANDS,
        )
        is True
    )
    assert store.worker_registered(created.session_id) is True

    resume = store.enqueue_command(
        created.session_id,
        "resume",
        {},
        "resume-before-retirement",
    )
    assert (
        store.queued_command_exists(
            created.session_id,
            storage_module.STOPPED_SESSION_COMMANDS,
        )
        is True
    )
    assert (
        store.retire_worker(
            created.session_id,
            "incarnation-one",
            storage_module.STOPPED_SESSION_COMMANDS,
        )
        is False
    )
    assert store.worker_registered(created.session_id) is True

    claimed = store.claim_command(
        created.session_id,
        storage_module.STOPPED_SESSION_COMMANDS,
    )
    assert claimed is not None
    assert claimed.command_id == resume.command_id
    assert (
        store.queued_command_exists(
            created.session_id,
            storage_module.STOPPED_SESSION_COMMANDS,
        )
        is False
    )
    assert (
        store.retire_worker(
            created.session_id,
            "incarnation-one",
            storage_module.STOPPED_SESSION_COMMANDS,
        )
        is True
    )
    assert store.worker_registered(created.session_id) is False
    store.close()
