"""Session timeline projection and renderer tests."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
from test_support import session

from agent_harness.errors import NotFoundError
from agent_harness.goals import create_goal
from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import ProviderAttempt
from agent_harness.storage import StateStore
from agent_harness.timeline import (
    ROUTING_DEFERRAL_CODE,
    TIMELINE_SCHEMA,
    project_timeline,
    render_timeline,
)


def _attempt(session_id: str, provider: str) -> ProviderAttempt:
    return ProviderAttempt(
        attempt_id=new_uuid(),
        session_id=session_id,
        provider=provider,
        native_session_id="native-" + provider,
        model="account-default",
        effort="high",
        auth_mode="subscription",
        status="running",
        started_at=utc_now(),
        ended_at="",
    )


def _seed_turn(store: StateStore, session_id: str, provider: str) -> str:
    attempt = _attempt(session_id, provider)
    store.create_attempt(attempt)
    turn_id = store.start_turn(session_id, attempt.attempt_id)
    store.record_routing(
        session_id,
        turn_id,
        provider,
        "account-default",
        "high",
        {
            "reason": provider + " had the best headroom score",
            "ranked": [
                {
                    "provider": provider,
                    "model": "account-default",
                    "binding_percent": 42.5,
                }
            ],
            "rejected": [],
            "credits_engaged": False,
            "binding_percent": 42.5,
            "binding_ceiling": 80.0,
            "workload": "implementation",
            "execution_profile": "unattended",
        },
    )
    return turn_id


def _seed_fixture(store: StateStore, workspace: Path) -> tuple[str, str, str]:
    created = session(workspace)
    store.create_session(created)
    goal = create_goal(
        created.session_id,
        "ship the bounded stage",
        budgets={"turns": 4, "seconds": 600},
    )
    store.create_goal(goal)
    first_turn = _seed_turn(store, created.session_id, "codex")
    second_turn = _seed_turn(store, created.session_id, "claude")
    store.finish_turn(first_turn, "complete")
    store.finish_turn(second_turn, "complete")
    command = store.enqueue_command(
        created.session_id,
        "message",
        {"text": "continue"},
        new_uuid(),
    )
    store.resolve_command(
        command.command_id,
        "failed",
        {
            "code": ROUTING_DEFERRAL_CODE,
            "message": "automatic routing has no eligible provider",
            "retryable": True,
        },
    )
    store.add_guard_incident(
        created.session_id,
        command.command_id,
        new_uuid(),
        "stagnation",
        "pause",
        {"consumption": {"total_tokens": 12}},
    )
    return created.session_id, goal.goal_id, command.command_id


def test_project_timeline_populates_all_sections(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    session_id, goal_id, command_id = _seed_fixture(store, tmp_path)
    later = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
        seconds=3600,
    )

    timeline = project_timeline(store, session_id, now=later)

    assert timeline["schema"] == TIMELINE_SCHEMA
    assert timeline["session_id"] == session_id
    assert timeline["generated_at"] == later.isoformat()

    turns = timeline["turns"]
    assert len(turns) == 2
    assert {turn["provider"] for turn in turns} == {"codex", "claude"}
    for turn in turns:
        assert turn["model"] == "account-default"
        assert turn["effort"] == "high"
        assert turn["turn_status"] == "complete"
        assert turn["attempt_id"]
        assert turn["started_at"]
        assert turn["completed_at"]
        assert turn["active_seconds"] >= 0.0

    decisions = timeline["routing_decisions"]
    assert len(decisions) == 2
    for decision in decisions:
        assert decision["binding_percent"] == 42.5
        assert decision["binding_ceiling"] == 80.0
        assert decision["reason"]
        assert decision["ranked"]
        assert decision["decision_id"]
        assert decision["turn_id"]

    budget = timeline["budget"]
    assert budget["goal_id"] == goal_id
    assert budget["budgets"] == {"turns": 4, "seconds": 600}
    consumption = budget["consumption"]
    assert consumption["turns"] == 2
    assert consumption["wall_seconds"] >= 3500.0
    assert consumption["wall_seconds"] > consumption["active_seconds"]
    assert consumption["active_seconds"] >= 0.0

    deferrals = timeline["deferrals"]
    assert len(deferrals) == 1
    assert deferrals[0]["command_id"] == command_id
    assert deferrals[0]["code"] == ROUTING_DEFERRAL_CODE
    assert deferrals[0]["retryable"] is True

    incidents = timeline["guard_incidents"]
    assert len(incidents) == 1
    assert incidents[0]["reason"] == "stagnation"
    assert incidents[0]["action"] == "pause"
    assert incidents[0]["command_id"] == command_id

    # The projection is JSON-stable: it round-trips unchanged.
    assert json.loads(json.dumps(timeline)) == timeline
    store.close()


def test_project_timeline_json_shape_is_stable(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    session_id, unused_goal, unused_command = _seed_fixture(store, tmp_path)
    del unused_goal, unused_command

    timeline = project_timeline(store, session_id)

    assert set(timeline) == {
        "schema",
        "session_id",
        "generated_at",
        "turns",
        "routing_decisions",
        "budget",
        "deferrals",
        "guard_incidents",
    }
    assert set(timeline["turns"][0]) == {
        "turn_id",
        "attempt_id",
        "provider",
        "model",
        "effort",
        "turn_status",
        "attempt_status",
        "replay_of",
        "started_at",
        "completed_at",
        "active_seconds",
        "command_id",
        "command_status",
        "usage",
    }
    assert set(timeline["routing_decisions"][0]) == {
        "decision_id",
        "turn_id",
        "provider",
        "model",
        "effort",
        "binding_percent",
        "binding_ceiling",
        "credits_engaged",
        "workload",
        "execution_profile",
        "reason",
        "ranked",
        "rejected",
        "created_at",
    }
    assert set(timeline["budget"]) == {"goal_id", "budgets", "consumption"}
    assert set(timeline["budget"]["consumption"]) == {
        "turns",
        "tokens",
        "context_tokens",
        "output_tokens",
        "tool_calls",
        "attempts",
        "child_agents",
        "dollars",
        "active_seconds",
        "wall_seconds",
    }
    assert set(timeline["deferrals"][0]) == {
        "command_id",
        "command_type",
        "code",
        "message",
        "retryable",
        "created_at",
        "updated_at",
    }
    assert set(timeline["guard_incidents"][0]) == {
        "incident_id",
        "command_id",
        "attempt_id",
        "reason",
        "action",
        "snapshot",
        "created_at",
    }
    store.close()


def test_project_timeline_requires_a_known_session(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    with pytest.raises(NotFoundError):
        project_timeline(store, new_uuid())
    with pytest.raises(ValueError):
        project_timeline(store, "not-a-uuid")
    store.close()


def test_render_timeline_lists_every_section(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    session_id, unused_goal, command_id = _seed_fixture(store, tmp_path)
    del unused_goal

    rendered = render_timeline(project_timeline(store, session_id))

    assert rendered.startswith("# Session timeline")
    assert "## Turns" in rendered
    assert "codex/account-default" in rendered
    assert "claude/account-default" in rendered
    assert "## Routing decisions" in rendered
    assert "binding 42.5%" in rendered
    assert "ceiling 80.0%" in rendered
    assert "## Budget" in rendered
    assert "Active seconds:" in rendered
    assert "wall seconds:" in rendered
    assert "## Deferrals" in rendered
    assert command_id in rendered
    assert ROUTING_DEFERRAL_CODE in rendered
    assert "## Guard incidents" in rendered
    assert "stagnation → pause" in rendered
    store.close()


def test_render_timeline_handles_an_empty_session(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)

    timeline = project_timeline(store, created.session_id)
    rendered = render_timeline(timeline)

    assert timeline["turns"] == []
    assert timeline["routing_decisions"] == []
    assert timeline["budget"] == {}
    assert timeline["deferrals"] == []
    assert timeline["guard_incidents"] == []
    assert "- none" in rendered
    assert "- no goal recorded" in rendered
    store.close()
