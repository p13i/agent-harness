from pathlib import Path

from agent_harness.context import compile_context
from agent_harness.context import workspace_instructions
from agent_harness.models import SessionEvent
from test_support import session


def test_context_projection_is_bounded_and_ordered(tmp_path: Path) -> None:
    current = session(tmp_path)
    events = []
    for sequence in range(1, 8):
        events.append(
            SessionEvent(
                session_id=current.session_id,
                sequence=sequence,
                event_id=str(sequence),
                event_type="user.message",
                role="user",
                text="x" * 100,
                status="complete",
                metadata={},
                blob_digest="",
                turn_id="",
                created_at="now",
            )
        )
    compiled = compile_context(
        current,
        events,
        max_input_tokens=400,
        reserve_output_tokens=100,
    )
    assert compiled.included_sequences == tuple(
        sorted(compiled.included_sequences)
    )
    assert compiled.omitted_events > 0
    assert compiled.projection["schema"].endswith("/v1")
    assert compiled.estimated_tokens <= 300


def test_workspace_instructions_include_provider_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "Agent rules",
        encoding="utf-8",
    )
    (tmp_path / "CLAUDE.md").write_text(
        "Claude rules",
        encoding="utf-8",
    )

    instructions = workspace_instructions(tmp_path)

    assert instructions == (
        "# AGENTS.md\n\nAgent rules",
        "# CLAUDE.md\n\nClaude rules",
    )
