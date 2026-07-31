"""Unified, terse notification state derived from canonical activity."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
from typing import Any
from typing import Iterable
from typing import Mapping


class NotificationSeverity(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"
    ACTION = "action"


class NotificationPersistence(str, Enum):
    TRANSIENT = "transient"
    ACTIVITY = "activity"
    ACTION = "action"


@dataclass(frozen=True, slots=True)
class NotificationAction:
    action_id: str
    label: str


@dataclass(frozen=True, slots=True)
class Notification:
    key: str
    title: str
    detail: str
    severity: NotificationSeverity
    persistence: NotificationPersistence
    source_sequence: int = 0
    unread: bool = True
    actions: tuple[NotificationAction, ...] = ()


@dataclass(frozen=True, slots=True)
class NotificationState:
    notifications: tuple[Notification, ...] = ()
    last_sequence: int = 0
    maximum_items: int = 50

    @property
    def unread_count(self) -> int:
        return sum(
            1
            for item in self.notifications
            if item.unread
            and item.persistence != NotificationPersistence.TRANSIENT
        )

    @property
    def active_action(self) -> Notification | None:
        for item in reversed(self.notifications):
            if item.persistence == NotificationPersistence.ACTION:
                return item
        return None

    @property
    def latest_transient(self) -> Notification | None:
        for item in reversed(self.notifications):
            if item.persistence == NotificationPersistence.TRANSIENT:
                return item
        return None

    def acknowledge(self, through_sequence: int | None = None) -> NotificationState:
        values: list[Notification] = []
        for item in self.notifications:
            acknowledge = through_sequence is None
            if through_sequence is not None:
                acknowledge = item.source_sequence <= through_sequence
            if acknowledge:
                values.append(replace(item, unread=False))
            else:
                values.append(item)
        return replace(self, notifications=tuple(values))

    def dismiss(self, key: str) -> NotificationState:
        return replace(
            self,
            notifications=tuple(
                item for item in self.notifications if item.key != key
            ),
        )


def project_notifications(
    state: NotificationState,
    events: Iterable[Mapping[str, Any]],
) -> NotificationState:
    """Coalesce canonical events into one notification stream."""

    current = state
    for event in events:
        sequence = _integer(event.get("sequence"))
        if sequence and sequence <= current.last_sequence:
            continue
        notification = notification_from_event(event)
        if notification is not None:
            current = _upsert(current, notification)
        current = replace(
            current,
            last_sequence=max(current.last_sequence, sequence),
        )
    return current


def notification_from_event(
    event: Mapping[str, Any],
) -> Notification | None:
    event_type = _text(event.get("event_type")).casefold()
    sequence = _integer(event.get("sequence"))
    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    turn_id = _text(event.get("turn_id"))
    status = _text(event.get("status")).casefold()

    if event_type == "approval.requested":
        approval_id = _text(metadata.get("approval_id"))
        return Notification(
            key="approval:" + approval_id,
            title="Approval needed",
            detail=_bounded(_text(event.get("text"))),
            severity=NotificationSeverity.ACTION,
            persistence=NotificationPersistence.ACTION,
            source_sequence=sequence,
            actions=(
                NotificationAction("review-approval", "Review"),
                NotificationAction("defer-approval", "Later"),
            ),
        )
    if event_type == "approval.resolved":
        approval_id = _text(metadata.get("approval_id"))
        return Notification(
            key="approval:" + approval_id,
            title="Approval resolved",
            detail="",
            severity=NotificationSeverity.SUCCESS,
            persistence=NotificationPersistence.TRANSIENT,
            source_sequence=sequence,
        )
    if event_type == "reconciliation.requested":
        reconciliation_id = _text(metadata.get("reconciliation_id"))
        return Notification(
            key="recovery:" + reconciliation_id,
            title="Recovery needed",
            detail="Provider outcome is ambiguous.",
            severity=NotificationSeverity.ACTION,
            persistence=NotificationPersistence.ACTION,
            source_sequence=sequence,
            actions=(
                NotificationAction("review-recovery", "Review"),
                NotificationAction("stop-session", "Stop"),
            ),
        )
    if event_type == "reconciliation.resolved":
        reconciliation_id = _text(metadata.get("reconciliation_id"))
        return Notification(
            key="recovery:" + reconciliation_id,
            title="Recovery resolved",
            detail="",
            severity=NotificationSeverity.SUCCESS,
            persistence=NotificationPersistence.TRANSIENT,
            source_sequence=sequence,
        )
    if event_type == "checkpoint.created":
        return Notification(
            key="checkpoint:" + (turn_id or str(sequence)),
            title="Checkpoint created",
            detail="",
            severity=NotificationSeverity.SUCCESS,
            persistence=NotificationPersistence.TRANSIENT,
            source_sequence=sequence,
        )
    if event_type in {
        "goal.budget_exhausted",
        "safety.guarded",
        "guard.triggered",
    }:
        reason = _text(metadata.get("reason"))
        if not reason:
            reason = _text(metadata.get("guard_reason"))
        if event_type == "goal.budget_exhausted":
            reason = _text(metadata.get("budget"))
        title = "Turn paused"
        if "budget" in reason.casefold():
            title = "Turn paused at budget limit"
        if event_type == "goal.budget_exhausted":
            title = "Turn paused at budget limit"
        return Notification(
            key="guard:" + (turn_id or str(sequence)),
            title=title,
            detail=_bounded(reason),
            severity=NotificationSeverity.ACTION,
            persistence=NotificationPersistence.ACTION,
            source_sequence=sequence,
            actions=(NotificationAction("review-usage", "Review usage"),),
        )
    if event_type == "session.archived":
        return Notification(
            key="session-archived",
            title="Session archived",
            detail="",
            severity=NotificationSeverity.INFO,
            persistence=NotificationPersistence.TRANSIENT,
            source_sequence=sequence,
        )
    if event_type == "session.unarchived":
        return Notification(
            key="session-archived",
            title="Session restored",
            detail="",
            severity=NotificationSeverity.SUCCESS,
            persistence=NotificationPersistence.TRANSIENT,
            source_sequence=sequence,
        )
    if event_type == "command.failed" or status == "failed":
        return Notification(
            key="turn-failed:" + (turn_id or str(sequence)),
            title="Turn failed",
            detail=_bounded(_event_detail(event, metadata)),
            severity=NotificationSeverity.WARNING,
            persistence=NotificationPersistence.ACTIVITY,
            source_sequence=sequence,
        )
    return None


def connection_notification(
    previous: str,
    current: str,
    *,
    sequence: int = 0,
) -> Notification | None:
    """Return one terse transition notification for connection changes."""

    if previous == current:
        return None
    if current == "connected" and previous in {
        "disconnected",
        "reconnecting",
        "send-unacknowledged",
    }:
        return Notification(
            key="connection",
            title="Reconnected",
            detail="",
            severity=NotificationSeverity.SUCCESS,
            persistence=NotificationPersistence.TRANSIENT,
            source_sequence=sequence,
        )
    if current == "reconnecting":
        return Notification(
            key="connection",
            title="Reconnecting",
            detail="",
            severity=NotificationSeverity.WARNING,
            persistence=NotificationPersistence.ACTIVITY,
            source_sequence=sequence,
        )
    if current == "send-unacknowledged":
        return Notification(
            key="connection",
            title="Send pending",
            detail="",
            severity=NotificationSeverity.ACTION,
            persistence=NotificationPersistence.ACTION,
            source_sequence=sequence,
            actions=(NotificationAction("retry-connection", "Reconnect"),),
        )
    if current == "disconnected":
        return Notification(
            key="connection",
            title="Disconnected",
            detail="",
            severity=NotificationSeverity.WARNING,
            persistence=NotificationPersistence.ACTIVITY,
            source_sequence=sequence,
        )
    return None


def with_connection(
    state: NotificationState,
    previous: str,
    current: str,
) -> NotificationState:
    notification = connection_notification(
        previous,
        current,
        sequence=state.last_sequence,
    )
    if notification is None:
        return state
    return _upsert(state, notification)


def push_notification(
    state: NotificationState,
    notification: Notification,
) -> NotificationState:
    """Insert or replace one locally-derived notification."""

    return _upsert(state, notification)


def _upsert(
    state: NotificationState,
    notification: Notification,
) -> NotificationState:
    values = [
        item
        for item in state.notifications
        if item.key != notification.key
    ]
    values.append(notification)
    if len(values) > state.maximum_items:
        values = values[-state.maximum_items :]
    return replace(state, notifications=tuple(values))


def _event_detail(
    event: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    for key in ("summary", "detail", "reason"):
        value = _text(metadata.get(key))
        if value:
            return value
    return _text(event.get("text"))


def _bounded(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized[:240]


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
