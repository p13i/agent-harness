"""Deterministic coverage for durable storage boundary behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import agent_harness.storage as storage_module
from agent_harness.errors import ConflictError
from agent_harness.errors import NotFoundError
from agent_harness.ids import new_uuid
from agent_harness.models import ProviderAttempt
from agent_harness.orchestration import creation_digest
from agent_harness.storage import PORTABLE_GLOBAL_TABLES
from agent_harness.storage import PORTABLE_SESSION_TABLES
from agent_harness.storage import StateStore
from tests.test_support import session


def _creation(
    value: object,
    *,
    external_ref: dict[str, str] | None = None,
) -> dict[str, Any]:
    current = value
    return {
        "workspace": current.workspace,  # type: ignore[attr-defined]
        "name": current.name,  # type: ignore[attr-defined]
        "permission_mode": current.permission_mode,  # type: ignore[attr-defined]
        "execution_profile": "interactive",
        "direct": True,
        "external_ref": external_ref or {},
        "routing": {"model": "", "effort": ""},
    }


def _portable_empty() -> tuple[dict[str, Any], dict[str, Any]]:
    record = {
        "schema": "p13i/agent-harness/chat-record/v1",
        "tables": {table: [] for table in PORTABLE_SESSION_TABLES},
    }
    global_record = {
        "schema": "p13i/agent-harness/chat-global/v1",
        "tables": {
            "ui_state": [],
            **{table: [] for table in PORTABLE_GLOBAL_TABLES},
        },
    }
    return record, global_record


def test_schema_column_integrity_and_transaction_boundaries(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "unsupported.sqlite3"
    connection = sqlite3.connect(unsupported)
    connection.execute("CREATE TABLE schema_meta(version INTEGER NOT NULL)")
    connection.execute("INSERT INTO schema_meta VALUES (99)")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="schema version"):
        StateStore(unsupported)

    store = StateStore(tmp_path / "state.sqlite3")
    with store.transaction() as connection:
        store._add_column(
            connection,
            "sessions",
            "name",
            "TEXT NOT NULL DEFAULT ''",
        )
    store.close()

    class CursorProbe:
        def fetchone(self) -> None:
            return None

    class ConnectionProbe:
        def execute(self, unused: str) -> CursorProbe:
            return CursorProbe()

    probe = object.__new__(StateStore)
    probe._lock = threading.RLock()
    probe._connection = ConnectionProbe()  # type: ignore[assignment]
    assert probe.integrity_check() == ""


def test_ensured_session_conflict_and_orphan_boundaries(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    external_ref = {"orchestrator": "test", "job_id": "one"}
    created = replace(session(tmp_path), external_ref=external_ref)
    with pytest.raises(ValueError, match="external_ref"):
        store.ensure_session(
            created,
            _creation(
                created,
                external_ref={"orchestrator": "test", "job_id": "two"},
            ),
        )

    creation = _creation(created, external_ref=external_ref)
    ensured, is_new = store.ensure_session(created, creation)
    assert is_new
    replay, is_new = store.ensure_session(
        replace(created, session_id=new_uuid()),
        creation,
        idempotency_key="external-replay",
    )
    assert not is_new
    assert replay.session_id == ensured.session_id

    duplicate = replace(
        session(tmp_path),
        session_id=created.session_id,
    )
    with pytest.raises(ConflictError, match="identifier"):
        store.ensure_session(duplicate, _creation(duplicate))

    orphan_creation = _creation(session(tmp_path))
    orphan_digest = creation_digest(orphan_creation)
    store._connection.execute("PRAGMA foreign_keys = OFF")
    store._connection.execute(
        "INSERT INTO session_creation_receipts VALUES (?, ?, ?, ?, ?)",
        ("orphan", orphan_digest, new_uuid(), "{}", "now"),
    )
    store._connection.commit()
    store._connection.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(RuntimeError, match="has no session"):
        store.ensure_session(
            session(tmp_path),
            orphan_creation,
            idempotency_key="orphan",
        )
    with pytest.raises(RuntimeError, match="has no session"):
        store.existing_ensured_session(
            orphan_creation,
            idempotency_key="orphan",
        )

    keyed = session(tmp_path)
    keyed_creation = _creation(keyed)
    store.ensure_session(
        keyed,
        keyed_creation,
        idempotency_key="keyed",
    )
    referenced_ref = {"orchestrator": "test", "job_id": "referenced"}
    referenced = replace(
        session(tmp_path),
        external_ref=referenced_ref,
    )
    referenced_creation = _creation(
        referenced,
        external_ref=referenced_ref,
    )
    store.ensure_session(referenced, referenced_creation)
    target_digest = creation_digest(referenced_creation)
    with store.transaction() as connection:
        connection.execute(
            """
            UPDATE session_creation_receipts SET request_digest = ?
            WHERE idempotency_key = ?
            """,
            (target_digest, "keyed"),
        )
    with pytest.raises(ConflictError, match="different sessions"):
        store.existing_ensured_session(
            referenced_creation,
            idempotency_key="keyed",
            external_ref=referenced_ref,
        )
    store.close()


def test_update_claim_turn_and_receipt_failure_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    with pytest.raises(NotFoundError):
        store.update_session(new_uuid(), name="missing")

    command = store.enqueue_command(
        created.session_id,
        "message",
        {
            "text": "turn",
            "turn_ref": {
                "step_id": "step",
                "agent_role": "implementer",
            },
        },
        "turn",
    )
    store._connection.execute(
        """
        CREATE TRIGGER ignore_command_claim
        BEFORE UPDATE OF status ON commands
        WHEN NEW.status = 'dispatching'
        BEGIN
            SELECT RAISE(IGNORE);
        END
        """
    )
    assert store.claim_command(created.session_id) is None
    store._connection.execute("DROP TRIGGER ignore_command_claim")

    attempt = ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=created.session_id,
        provider="codex",
        native_session_id="native",
        model="model",
        effort="low",
        auth_mode="subscription",
        status="running",
        started_at="now",
        ended_at="",
    )
    store.create_attempt(attempt)
    turn_id = store.start_turn(
        created.session_id,
        attempt.attempt_id,
        turn_ref=command.turn_ref,
    )
    rows = store.presentation_turn_rows(created.session_id)
    assert rows[0]["turn_id"] == turn_id
    assert rows[0]["turn_ref"]["step_id"] == "step"

    monkeypatch.setattr(store, "mutation_receipt", lambda *unused: None)
    with pytest.raises(RuntimeError, match="not recorded"):
        store.record_mutation_receipt(
            "receipt",
            "operation",
            "digest",
            {"ok": True},
            200,
        )
    store.close()


def test_portable_validation_import_and_merge_boundaries(
    tmp_path: Path,
) -> None:
    record, global_record = _portable_empty()
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    with pytest.raises(ConflictError, match="empty store"):
        store.import_portable([], global_record)

    existing_record = store.portable_session(created.session_id)
    with pytest.raises(ConflictError, match="already exists"):
        store.merge_portable([existing_record], global_record)
    store.close()

    invalid_values = (
        ([{**record, "schema": "invalid"}], global_record),
        ([{**record, "tables": []}], global_record),
        (
            [
                {
                    **record,
                    "tables": {
                        **record["tables"],
                        "events": {},
                    },
                }
            ],
            global_record,
        ),
        ([record], {**global_record, "schema": "invalid"}),
        ([record], {**global_record, "tables": []}),
        (
            [record],
            {
                **global_record,
                "tables": {
                    **global_record["tables"],
                    "ui_state": {},
                },
            },
        ),
        (
            [record],
            {
                **global_record,
                "tables": {
                    **global_record["tables"],
                    "usage_samples": {},
                },
            },
        ),
    )
    for index, (records, global_value) in enumerate(invalid_values):
        invalid_store = StateStore(
            tmp_path / ("invalid-" + str(index) + ".sqlite3")
        )
        with pytest.raises(ValueError):
            invalid_store.import_portable(records, global_value)
        invalid_store.close()

    merge_store = StateStore(tmp_path / "merge.sqlite3")
    with merge_store.transaction() as connection:
        connection.execute("CREATE TABLE no_key(value TEXT)")
        with pytest.raises(RuntimeError, match="primary key"):
            merge_store._merge_portable_rows(connection, "no_key", [])
        connection.execute(
            "CREATE TABLE merge_values(id TEXT PRIMARY KEY, value TEXT)"
        )
        row = {"id": "one", "value": "first"}
        merge_store._merge_portable_rows(
            connection,
            "merge_values",
            [row],
        )
        merge_store._merge_portable_rows(
            connection,
            "merge_values",
            [row],
        )
        with pytest.raises(ConflictError, match="conflicts"):
            merge_store._merge_portable_rows(
                connection,
                "merge_values",
                [{"id": "one", "value": "other"}],
            )
    merge_store.close()


def test_session_import_and_reconciliation_decode_boundaries(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    with pytest.raises(ValueError, match="identifier"):
        store.import_session(
            {"session": {}},
            worktree=str(tmp_path),
            owner_host="host",
            owner_epoch=1,
        )

    external_ref = {"orchestrator": "test", "job_id": "job"}
    existing = replace(session(tmp_path), external_ref=external_ref)
    store.create_session(existing)
    conflict = {
        "session": {
            **session(tmp_path).as_dict(),
            "external_ref": external_ref,
        }
    }
    with pytest.raises(ConflictError, match="external reference"):
        store.import_session(
            conflict,
            worktree=str(tmp_path / "conflict"),
            owner_host="host",
            owner_epoch=2,
        )

    minimal = session(tmp_path)
    imported = store.import_session(
        {"session": minimal.as_dict()},
        worktree=str(tmp_path / "minimal"),
        owner_host="host",
        owner_epoch=2,
    )
    assert imported.session_id == minimal.session_id

    empty_safety = session(tmp_path)
    store.import_session(
        {
            "session": empty_safety.as_dict(),
            "safety": {"profile": ""},
        },
        worktree=str(tmp_path / "empty-safety"),
        owner_host="host",
        owner_epoch=2,
    )

    with pytest.raises(ValueError, match="not a list"):
        storage_module._reconciliation(
            {"provider_attempts_json": "{}"}  # type: ignore[arg-type]
        )
    store.close()


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "--import-mode=importlib",
                *sys.argv[1:],
            ]
        )
    )
