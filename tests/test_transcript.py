"""Canonical transcript projection and renderer tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_support import session

from agent_harness.blobs import BlobStore
from agent_harness.goals import create_goal
from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import ProviderAttempt
from agent_harness.storage import StateStore
from agent_harness.transcript import (
    TRANSCRIPT_SCHEMA,
    RenderPolicy,
    Transcript,
    TranscriptEntry,
    project_transcript,
    render,
    validate_render_policy,
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


def _seed_session(
    store: StateStore,
    workspace: Path,
    provider: str,
) -> tuple[str, str]:
    created = session(workspace)
    store.create_session(created)
    attempt = _attempt(created.session_id, provider)
    store.create_attempt(attempt)
    turn_id = store.start_turn(created.session_id, attempt.attempt_id)
    return created.session_id, turn_id


def test_project_transcript_claude_fixture_is_golden(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    blobs = BlobStore(tmp_path / "blobs")
    session_id, turn_id = _seed_session(store, tmp_path, "claude")
    output_digest = blobs.put_text("file contents\nsecond line\n")
    store.append_event(
        session_id,
        "user.message",
        role="user",
        text="fix the flaky test",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "agent.message",
        role="assistant",
        text="reading the test",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "agent.message.delta",
        role="assistant",
        text="reading",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "tool.started",
        text="Read",
        metadata={"id": "tool-1", "name": "Read"},
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "tool.completed",
        metadata={"tool_use_id": "tool-1"},
        blob_digest=output_digest,
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "file.change.completed",
        text="tests/test_flaky.py",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "turn.completed",
        status="complete",
        turn_id=turn_id,
    )
    store.finish_turn(turn_id, "complete")

    transcript = project_transcript(store, session_id, blobs=blobs)

    assert transcript.schema == TRANSCRIPT_SCHEMA
    assert transcript.session_id == session_id
    assert transcript.goal == ""
    observed = [
        (
            entry.seq,
            entry.role,
            entry.name,
            entry.provider,
            entry.text,
        )
        for entry in transcript.entries
    ]
    assert observed == [
        (1, "user", "", "claude", "fix the flaky test"),
        (2, "assistant", "", "claude", "reading the test"),
        (4, "tool_call", "Read", "claude", "Read"),
        (5, "tool_result", "", "claude", "file contents\nsecond line\n"),
        (6, "file_change", "", "claude", "tests/test_flaky.py"),
        (7, "system", "", "claude", ""),
    ]
    turn_entries = [
        entry for entry in transcript.entries if entry.turn_id == turn_id
    ]
    assert len(turn_entries) == len(transcript.entries)
    assert len({entry.digest for entry in transcript.entries}) == len(
        transcript.entries
    )
    store.close()


def test_project_transcript_codex_fixture_is_golden(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    session_id, turn_id = _seed_session(store, tmp_path, "codex")
    store.append_event(
        session_id,
        "user.message",
        role="user",
        text="add a flag",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "tool.command.started",
        text="ls",
        metadata={"id": "item-1", "status": "in_progress"},
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "tool.output.delta",
        text="agent_harness\n",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "tool.command.completed",
        text="ls\nagent_harness\n",
        metadata={"id": "item-1", "exit_code": 0},
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "agent.message",
        role="assistant",
        text="flag added",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "turn.completed",
        status="complete",
        turn_id=turn_id,
    )
    store.finish_turn(turn_id, "complete")

    transcript = project_transcript(store, session_id)

    observed = [
        (
            entry.seq,
            entry.role,
            entry.name,
            entry.provider,
            entry.text,
        )
        for entry in transcript.entries
    ]
    assert observed == [
        (1, "user", "", "codex", "add a flag"),
        (2, "tool_call", "", "codex", "ls"),
        (4, "tool_result", "", "codex", "ls\nagent_harness\n"),
        (5, "assistant", "", "codex", "flag added"),
        (6, "system", "", "codex", ""),
    ]
    store.close()


def test_project_transcript_kimi_fixture_is_golden(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    session_id, turn_id = _seed_session(store, tmp_path, "kimi")
    store.append_event(
        session_id,
        "user.message",
        role="user",
        text="summarize the diff",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "agent.message",
        role="assistant",
        text="the diff adds a flag",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "turn.completed",
        status="complete",
        turn_id=turn_id,
    )
    store.finish_turn(turn_id, "complete")
    store.append_event(
        session_id,
        "guard.tripped",
        status="warning",
        metadata={"reason": "budget"},
    )

    transcript = project_transcript(store, session_id)

    observed = [
        (
            entry.seq,
            entry.role,
            entry.provider,
            entry.turn_id,
            entry.text,
        )
        for entry in transcript.entries
    ]
    assert observed == [
        (1, "user", "kimi", turn_id, "summarize the diff"),
        (2, "assistant", "kimi", turn_id, "the diff adds a flag"),
        (3, "system", "kimi", turn_id, ""),
        (4, "system", "", "", ""),
    ]
    store.close()


def test_project_transcript_carries_goal_objective(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    created = session(tmp_path)
    store.create_session(created)
    goal = create_goal(created.session_id, "ship the transcript surface")
    store.create_goal(goal)
    store.update_session(created.session_id, goal_id=goal.goal_id)

    transcript = project_transcript(store, created.session_id)

    assert transcript.goal == "ship the transcript surface"
    assert transcript.entries == ()
    store.close()


def test_project_transcript_rebuild_is_deterministic(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    session_id, turn_id = _seed_session(store, tmp_path, "claude")
    store.append_event(
        session_id,
        "user.message",
        role="user",
        text="deterministic",
        turn_id=turn_id,
    )
    store.append_event(
        session_id,
        "agent.message",
        role="assistant",
        text="same events, same digest",
        turn_id=turn_id,
    )

    first = project_transcript(store, session_id)
    second = project_transcript(store, session_id)

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()

    store.append_event(
        session_id,
        "agent.message",
        role="assistant",
        text="a later reply changes the digest",
        turn_id=turn_id,
    )
    extended = project_transcript(store, session_id)
    assert extended.digest != first.digest
    assert extended.entries[:2] == first.entries
    store.close()


def _render_entry(
    seq: int,
    role: str,
    text: str,
    *,
    turn_id: str = "",
    provider: str = "claude",
) -> TranscriptEntry:
    return TranscriptEntry(
        seq=seq,
        turn_id=turn_id,
        provider=provider,
        role=role,
        name="",
        text=text,
        digest="digest-" + str(seq).zfill(3),
    )


def _long_transcript() -> Transcript:
    entries = [
        _render_entry(1, "user", "original instructions stay verbatim"),
    ]
    for index in range(2, 8):
        entries.append(
            _render_entry(
                index,
                "assistant",
                "turn " + str(index) + " body " + "x" * 400,
                turn_id="turn-" + str(index),
            )
        )
    return Transcript(
        session_id="session-render",
        goal="finish the work",
        entries=tuple(entries),
        digest="transcript-digest",
    )


def test_render_short_transcript_is_verbatim() -> None:
    transcript = _long_transcript()
    policy = RenderPolicy(token_budget=100_000, tail_turns=2)

    output = render(transcript, policy)

    assert "## Goal" in output
    assert "finish the work" in output
    assert "original instructions stay verbatim" in output
    for index in range(2, 8):
        assert "turn " + str(index) + " body" in output
    assert "elided" not in output
    assert "truncated" not in output
    assert render(transcript, policy) == output


def test_render_elides_middle_before_tail() -> None:
    transcript = _long_transcript()
    verbatim = render(transcript, RenderPolicy(token_budget=100_000, tail_turns=2))
    budget = (len(verbatim) // 4) - 1

    output = render(transcript, RenderPolicy(token_budget=budget, tail_turns=2))

    assert "original instructions stay verbatim" in output
    assert "turn 6 body" in output
    assert "turn 7 body" in output
    assert "turn 3 body" not in output
    assert "turn 4 body" not in output
    assert "turn 5 body" not in output
    assert "entries elided; digests retained" in output
    assert "`digest-003`" in output
    assert "truncated" not in output


def test_render_shrinks_tail_after_middle() -> None:
    transcript = _long_transcript()
    verbatim = render(transcript, RenderPolicy(token_budget=100_000, tail_turns=2))
    middle_elided = render(
        transcript,
        RenderPolicy(token_budget=(len(verbatim) // 4) - 1, tail_turns=2),
    )
    assert "turn 7 body" in middle_elided
    assert "turn 6 body" in middle_elided
    budget = (len(middle_elided) // 4) - 1

    output = render(transcript, RenderPolicy(token_budget=budget, tail_turns=2))

    assert "original instructions stay verbatim" in output
    assert "turn 7 body" in output
    assert "turn 6 body" not in output
    assert "`digest-006`" in output
    assert "truncated" not in output


def test_render_truncates_head_only_as_last_resort() -> None:
    entries = [
        _render_entry(1, "user", "instructions " + "y" * 2_000),
        _render_entry(2, "assistant", "reply", turn_id="turn-2"),
    ]
    transcript = Transcript(
        session_id="session-truncate",
        goal="",
        entries=tuple(entries),
        digest="transcript-digest",
    )
    policy = RenderPolicy(token_budget=80, tail_turns=1)

    output = render(transcript, policy)

    assert "[truncated; digest digest-001]" in output
    assert "reply" in output
    assert len(output) // 4 <= policy.token_budget + 1
    assert render(transcript, policy) == output


def test_render_policy_validation() -> None:
    with pytest.raises(ValueError, match="tail turns"):
        validate_render_policy(RenderPolicy(tail_turns=-1))
    with pytest.raises(ValueError, match="tail turns"):
        validate_render_policy(RenderPolicy(tail_turns=65))
    with pytest.raises(ValueError, match="token budget"):
        validate_render_policy(RenderPolicy(token_budget=0))
    with pytest.raises(ValueError, match="token budget"):
        render(
            Transcript(
                session_id="s",
                goal="",
                entries=(),
                digest="d",
            ),
            RenderPolicy(token_budget=0),
        )
