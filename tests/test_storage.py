from pathlib import Path

from agent_harness.models import CommandStatus
from agent_harness.storage import StateStore
from test_support import session


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
    second = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "different"},
        "same-key",
    )
    assert first.command_id == second.command_id
    claimed = store.claim_command(created.session_id)
    assert claimed is not None
    assert claimed.status == CommandStatus.DISPATCHING
    assert store.recover_dispatching(created.session_id) == 1
    recovered = store.get_command(first.command_id)
    assert recovered.status == CommandStatus.FAILED
    assert recovered.result["code"] == "E_NEEDS_RECONCILIATION"
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
