"""Shared test values."""

from __future__ import annotations

from pathlib import Path

from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Session


def session(workspace: Path) -> Session:
    now = utc_now()
    return Session(
        session_id=new_uuid(),
        name="test",
        workspace=str(workspace),
        worktree=str(workspace),
        lifecycle="running",
        attention="idle",
        permission_mode="approval",
        active_provider="",
        model="",
        effort="",
        goal_id="",
        owner_host="test-host",
        owner_epoch=1,
        created_at=now,
        updated_at=now,
    )
