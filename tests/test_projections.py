import json
from pathlib import Path

import pytest

from agent_harness.context import CompiledContext
from agent_harness.ids import utc_now
from agent_harness.models import SessionEvent
from agent_harness.projections import _jsonl
from agent_harness.projections import write_session_projections


def test_projection_set_is_complete_and_private(tmp_path: Path) -> None:
    session_id = "11111111-1111-4111-8111-111111111111"
    payload = {
        "schema": "p13i/agent-harness/session-export/v1",
        "session": {
            "session_id": session_id,
            "name": "Projection test",
            "lifecycle": "paused",
            "active_provider": "codex",
        },
    }
    event = SessionEvent(
        session_id=session_id,
        sequence=1,
        event_id="22222222-2222-4222-8222-222222222222",
        event_type="agent.message",
        role="assistant",
        text="Durable output",
        status="complete",
        metadata={},
        blob_digest="",
        turn_id="",
        created_at=utc_now(),
    )
    context = CompiledContext(
        text="compiled handoff",
        estimated_tokens=4,
        included_sequences=(1,),
        omitted_events=0,
        projection={
            "schema": "p13i/agent-harness/run-context/v1",
            "session_id": session_id,
        },
    )

    paths = write_session_projections(
        tmp_path,
        payload,
        context,
        [event],
        None,
    )

    assert set(paths) == {
        "export",
        "run_context",
        "transcript_jsonl",
        "transcript_markdown",
        "goal",
    }
    for path in paths.values():
        assert path.stat().st_mode & 0o077 == 0
    context_value = json.loads(
        paths["run_context"].read_text(encoding="utf-8")
    )
    assert context_value["compiled_context"] == "compiled handoff"
    assert "Durable output" in paths[
        "transcript_markdown"
    ].read_text(encoding="utf-8")


def test_projection_rejects_invalid_session_and_empty_transcript(
    tmp_path: Path,
) -> None:
    context = CompiledContext(
        text="",
        estimated_tokens=0,
        included_sequences=(),
        omitted_events=0,
        projection={},
    )
    with pytest.raises(ValueError):
        write_session_projections(
            tmp_path,
            {"session": []},
            context,
            [],
            None,
        )
    assert _jsonl([]) == ""
