from pathlib import Path

import pytest

from agent_harness.blobs import BlobStore
from agent_harness.models import Checkpoint
from agent_harness.models import ReconciliationRecord
from agent_harness.models import SessionEvent
from agent_harness.notifications import NotificationPersistence
from agent_harness.notifications import NotificationSeverity
from agent_harness.notifications import NotificationState
from agent_harness.notifications import connection_notification
from agent_harness.notifications import notification_from_event
from agent_harness.notifications import project_notifications
from agent_harness.notifications import push_notification
from agent_harness.notifications import with_connection
from agent_harness.presentation import SessionSwitchCoordinator
from agent_harness.presentation import SessionViewCacheEntry
from agent_harness.presentation import _safe_result
from agent_harness.presentation import _safe_safety
from agent_harness.presentation import _safety_by_command
from agent_harness.presentation import checkpoint_diff
from agent_harness.presentation import sanitize_diff
from agent_harness.presentation import session_turn
from agent_harness.presentation import session_turns


SESSION_ID = "11111111-1111-4111-8111-111111111111"
TURN_ONE = "22222222-2222-4222-8222-222222222222"
TURN_TWO = "33333333-3333-4333-8333-333333333333"
CHECKPOINT_ID = "44444444-4444-4444-8444-444444444444"


class Store:
    def __init__(self) -> None:
        self.rows = [
            {
                "turn_id": TURN_ONE,
                "attempt_id": "attempt-1",
                "turn_status": "failed",
                "started_at": "2026-07-30T10:00:00+00:00",
                "completed_at": "2026-07-30T10:00:01+00:00",
                "turn_ref": {
                    "step_id": "build",
                    "agent_role": "implementer",
                },
                "provider": "claude",
                "model": "opus",
                "effort": "high",
                "attempt_status": "failed",
                "ended_at": "2026-07-30T10:00:01+00:00",
                "command_id": "command-1",
                "command_status": "complete",
                "request_text": "Build the bounded change.",
                "command_result": {
                    "status": "complete",
                    "provider": "codex",
                    "private": "hidden",
                },
            },
            {
                "turn_id": TURN_TWO,
                "attempt_id": "attempt-2",
                "turn_status": "complete",
                "started_at": "2026-07-30T10:00:02+00:00",
                "completed_at": "2026-07-30T10:00:03+00:00",
                "turn_ref": {
                    "step_id": "build",
                    "agent_role": "implementer",
                },
                "provider": "codex",
                "model": "default",
                "effort": "high",
                "attempt_status": "complete",
                "ended_at": "2026-07-30T10:00:03+00:00",
                "command_id": "command-1",
                "command_status": "complete",
                "request_text": "Build the bounded change.",
                "command_result": {
                    "status": "complete",
                    "provider": "codex",
                    "private": "hidden",
                },
            },
        ]
        self.events = [
            SessionEvent(
                session_id=SESSION_ID,
                sequence=1,
                event_id="event-1",
                event_type="user.message",
                role="user",
                text="Build the bounded change.",
                status="complete",
                metadata={"secret": "hidden"},
                blob_digest="private-blob",
                turn_id=TURN_ONE,
                created_at="2026-07-30T10:00:00+00:00",
            ),
            SessionEvent(
                session_id=SESSION_ID,
                sequence=2,
                event_id="event-2",
                event_type="checkpoint.created",
                role="",
                text="",
                status="complete",
                metadata={
                    "checkpoint_id": CHECKPOINT_ID,
                    "secret": "hidden",
                },
                blob_digest="private-blob",
                turn_id=TURN_TWO,
                created_at="2026-07-30T10:00:03+00:00",
            ),
            SessionEvent(
                session_id=SESSION_ID,
                sequence=3,
                event_id="event-3",
                event_type="session.archived",
                role="",
                text="",
                status="complete",
                metadata={},
                blob_digest="",
                turn_id="",
                created_at="2026-07-30T10:00:04+00:00",
            ),
        ]

    def presentation_turn_rows(
        self,
        session_id: str,
    ) -> list[dict[str, object]]:
        assert session_id == SESSION_ID
        return self.rows

    def all_events(self, session_id: str) -> list[SessionEvent]:
        assert session_id == SESSION_ID
        return self.events

    def session_safety(self, session_id: str) -> dict[str, object]:
        assert session_id == SESSION_ID
        return {
            "envelopes": [
                {
                    "command_id": "command-1",
                    "state": "complete",
                    "consumption": {
                        "total_tokens": 42,
                        "tool_calls": 2,
                    },
                    "private": "hidden",
                }
            ]
        }

    def all_reconciliations(
        self,
        session_id: str,
    ) -> list[ReconciliationRecord]:
        assert session_id == SESSION_ID
        return []

    def last_sequence(self, session_id: str) -> int:
        assert session_id == SESSION_ID
        return 3


def test_turn_projection_groups_attempts_and_excludes_private_data() -> None:
    store = Store()
    page = session_turns(store, SESSION_ID)  # type: ignore[arg-type]

    assert page["revision"] == 3
    assert page["next_after_sequence"] == 2
    assert len(page["turns"]) == 1
    turn = page["turns"][0]
    assert turn["turn_ids"] == [TURN_ONE, TURN_TWO]
    assert [item["provider"] for item in turn["attempts"]] == [
        "claude",
        "codex",
    ]
    assert turn["checkpoint_id"] == CHECKPOINT_ID
    assert "private" not in turn["result"]
    assert "private" not in turn["safety"]
    assert "secret" not in turn["activity"][0]["metadata"]
    assert "blob_digest" not in turn["activity"][0]
    assert page["session_activity"][0]["event_type"] == "session.archived"

    selected = session_turn(
        store,  # type: ignore[arg-type]
        SESSION_ID,
        TURN_TWO,
    )
    assert selected["turn"]["turn_id"] == TURN_ONE

    after = session_turns(
        store,  # type: ignore[arg-type]
        SESSION_ID,
        after_sequence=2,
        limit=999,
    )
    assert after["turns"] == []
    assert after["session_activity"][0]["sequence"] == 3

    with pytest.raises(ValueError, match="turn does not belong"):
        session_turn(
            store,  # type: ignore[arg-type]
            SESSION_ID,
            "55555555-5555-4555-8555-555555555555",
        )
    with pytest.raises(ValueError, match="after_sequence"):
        session_turns(
            store,  # type: ignore[arg-type]
            SESSION_ID,
            after_sequence=-1,
        )


def test_turn_projection_keeps_eventless_turn_on_initial_page() -> None:
    store = Store()
    store.events = []

    initial = session_turns(store, SESSION_ID)  # type: ignore[arg-type]
    assert len(initial["turns"]) == 1
    assert initial["turns"][0]["last_sequence"] == 0
    assert initial["next_after_sequence"] == 0

    later = session_turns(
        store,  # type: ignore[arg-type]
        SESSION_ID,
        after_sequence=1,
    )
    assert later["turns"] == []


def test_checkpoint_diff_is_paged_redacted_and_binary_safe(
    tmp_path: Path,
) -> None:
    blobs = BlobStore(tmp_path / "blobs")
    patch = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-token=ghp_abcdefghijklmnopqrstuvwxyz\n"
        "+token=sk-abcdefghijklmnopqrstuvwxyz\n"
        '+"password": "private value"\n'
        "diff --git a/deleted.py b/deleted.py\n"
        "--- a/deleted.py\n"
        "+++ /dev/null\n"
        "-removed = True\n"
        "diff --git a/.env b/.env\n"
        "--- a/.env\n"
        "+++ b/.env\n"
        "+PASSWORD=private\n"
        'diff --git "a/private values/.env" '
        '"b/private values/.env"\n'
        '--- "a/private values/.env"\n'
        '+++ "b/private values/.env"\n'
        '+"token": "still private"\n'
        "diff --git a/image.png b/image.png\n"
        "Binary files a/image.png and b/image.png differ\n"
        "binary payload must not appear\n"
    )
    digest = blobs.put_text(patch)
    checkpoint = Checkpoint(
        checkpoint_id=CHECKPOINT_ID,
        session_id=SESSION_ID,
        sequence=2,
        provider="codex",
        native_session_id="native",
        base_commit="base",
        patch_digest=digest,
        untracked_digest=blobs.put(b""),
        context_digest=blobs.put(b""),
        created_at="2026-07-30T10:00:00+00:00",
    )

    first = checkpoint_diff(
        checkpoint,
        blobs,
        start_line=0,
        limit=4,
    )
    assert first["truncated"]
    assert first["next_start_line"] == 4
    assert first["changed_files"] == ["app.py", "deleted.py"]
    assert first["binary"]
    assert first["redactions"] == 5
    assert "ghp_" not in str(first)
    assert "sk-" not in str(first)
    assert "private" not in str(first)
    assert ".env" not in str(first)
    assert "still private" not in str(first)
    assert "binary payload" not in str(first)

    last = checkpoint_diff(
        checkpoint,
        blobs,
        start_line=4,
        limit=1000,
    )
    assert last["next_start_line"] is None
    assert not last["truncated"]
    with pytest.raises(ValueError, match="start_line"):
        checkpoint_diff(checkpoint, blobs, start_line=-1)

    sanitized = sanitize_diff(
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
    )
    assert sanitized[1] == 1
    assert "abcdefghijklmnopqrstuvwxyz" not in sanitized[0]


def test_notifications_are_terse_coalesced_and_acknowledgeable() -> None:
    events = [
        {
            "sequence": 1,
            "event_type": "approval.requested",
            "text": "Approve the bounded action.",
            "metadata": {"approval_id": "approval-1"},
        },
        {
            "sequence": 2,
            "event_type": "checkpoint.created",
            "turn_id": TURN_ONE,
        },
        {
            "sequence": 3,
            "event_type": "checkpoint.created",
            "turn_id": TURN_ONE,
        },
    ]
    state = project_notifications(NotificationState(), events)

    assert len(state.notifications) == 2
    assert state.active_action is not None
    assert state.active_action.title == "Approval needed"
    assert "/approve" not in str(state.notifications)
    assert state.latest_transient is not None
    assert state.latest_transient.title == "Checkpoint created"
    assert state.unread_count == 1
    assert project_notifications(state, events) == state

    acknowledged = state.acknowledge(1)
    assert acknowledged.unread_count == 0
    dismissed = acknowledged.dismiss("approval:approval-1")
    assert dismissed.active_action is None

    reconnecting = with_connection(
        NotificationState(),
        "connected",
        "reconnecting",
    )
    reconnected = with_connection(
        reconnecting,
        "reconnecting",
        "connected",
    )
    assert reconnected.latest_transient is not None
    assert reconnected.latest_transient.title == "Reconnected"
    assert connection_notification("connected", "connected") is None

    local = notification_from_event(
        {
            "sequence": 4,
            "event_type": "command.failed",
            "status": "failed",
            "metadata": {"reason": "bounded failure"},
        }
    )
    assert local is not None
    assert local.title == "Turn failed"
    pushed = push_notification(NotificationState(), local)
    assert pushed.notifications == (local,)


def test_notification_projection_covers_every_lifecycle() -> None:
    cases = (
        (
            "approval.resolved",
            {"approval_id": "approval-1"},
            "Approval resolved",
        ),
        (
            "reconciliation.requested",
            {
                "reconciliation_id": "recovery-1",
                "reason": "proof-service-fault-timeout",
            },
            "Recovery needed",
        ),
        (
            "reconciliation.resolved",
            {"reconciliation_id": "recovery-1"},
            "Recovery resolved",
        ),
        (
            "guard.triggered",
            {"guard_reason": "bounded guard"},
            "Turn paused",
        ),
        (
            "guard.tripped",
            {"reason": "repeated context"},
            "Turn paused",
        ),
        (
            "safety.guarded",
            {"reason": "token budget reached"},
            "Turn paused at budget limit",
        ),
        (
            "goal.budget_exhausted",
            {"budget": "turns"},
            "Turn paused at budget limit",
        ),
        ("session.archived", {}, "Session archived"),
        ("session.unarchived", {}, "Session restored"),
    )
    for sequence, value in enumerate(cases, start=1):
        event_type, metadata, title = value
        notification = notification_from_event(
            {
                "sequence": str(sequence),
                "event_type": event_type,
                "metadata": metadata,
            }
        )
        assert notification is not None
        assert notification.title == title
        if event_type == "reconciliation.requested":
            assert notification.detail == (
                "Provider outcome is ambiguous: proof-service-fault-timeout"
            )

    failed = notification_from_event(
        {
            "sequence": True,
            "event_type": "unknown",
            "status": "failed",
            "text": "Provider stopped.",
            "metadata": None,
        }
    )
    assert failed is not None
    assert failed.detail == "Provider stopped."
    assert notification_from_event(
        {
            "sequence": "not-a-number",
            "event_type": "unknown",
        }
    ) is None
    assert notification_from_event(
        {
            "sequence": 1.5,
            "event_type": "unknown",
        }
    ) is None


def test_notification_connections_and_bounded_history() -> None:
    pending = connection_notification(
        "reconnecting",
        "send-unacknowledged",
    )
    assert pending is not None
    assert pending.title == "Send pending"
    disconnected = connection_notification("connected", "disconnected")
    assert disconnected is not None
    assert disconnected.title == "Disconnected"
    assert connection_notification("connected", "unknown") is None

    state = NotificationState(maximum_items=2)
    for sequence in range(3):
        notification = notification_from_event(
            {
                "sequence": sequence + 1,
                "event_type": "command.failed",
                "turn_id": "turn-" + str(sequence),
                "metadata": {"summary": "failure"},
            }
        )
        assert notification is not None
        state = push_notification(state, notification)
    assert len(state.notifications) == 2
    assert state.acknowledge().unread_count == 0


def test_presentation_defensive_branches_are_safe() -> None:
    row = dict(Store().rows[0])
    row["command_id"] = ""
    row["command_status"] = ""
    row["command_result"] = "private"
    store = Store()
    store.rows = [row]
    store.events = [
        SessionEvent(
            session_id=SESSION_ID,
            sequence=1,
            event_id="event-evidence",
            event_type="goal.evidence",
            role="",
            text="make test passed",
            status="complete",
            metadata={"outcome": "passed"},
            blob_digest="",
            turn_id=TURN_ONE,
            created_at="2026-07-30T10:00:00+00:00",
        )
    ]
    page = session_turns(store, SESSION_ID)  # type: ignore[arg-type]
    assert page["turns"][0]["command_id"] == "turn:" + TURN_ONE
    assert page["turns"][0]["status"] == "failed"
    assert page["turns"][0]["result"] == {}
    assert page["turns"][0]["evidence"][0]["event_type"] == (
        "goal.evidence"
    )

    assert _safe_result([]) == {}
    assert _safe_safety([]) == {}
    assert _safety_by_command({"envelopes": "private"}) == {}
    assert _safety_by_command(
        {
            "envelopes": [
                "private",
                {"command_id": "", "state": "private"},
            ]
        }
    ) == {}
    malformed = sanitize_diff("diff --git missing\nBearer abcdefghijklmnop")
    assert "Bearer [REDACTED]" in malformed[0]
    token = sanitize_diff(
        "value ghp_abcdefghijklmnopqrstuvwxyz012345"
    )
    assert "ghp_" not in token[0]
    assert token[1] == 1


def test_switch_coordinator_rejects_stale_views_and_evicts_lru() -> None:
    coordinator = SessionSwitchCoordinator(maximum_entries=2)
    first_generation = coordinator.begin("session-a")
    second_generation = coordinator.begin("session-b")

    assert not coordinator.is_current(first_generation, "session-a")
    assert coordinator.is_current(second_generation, "session-b")

    for session_id in ("session-a", "session-b", "session-c"):
        coordinator.remember(
            SessionViewCacheEntry(
                session_id=session_id,
                workspace_mode="control",
            )
        )
    assert coordinator.recall("session-a") is None
    recalled = coordinator.recall("session-b")
    assert recalled is not None
    assert recalled.workspace_mode == "control"
