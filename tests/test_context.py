from dataclasses import replace
from pathlib import Path

import pytest

import agent_harness.context as context_module
from agent_harness.context import compile_context
from agent_harness.context import workspace_context_artifacts
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


def _event(
    session_id: str,
    sequence: int,
    event_type: str,
    metadata: dict[str, object],
) -> SessionEvent:
    return SessionEvent(
        session_id=session_id,
        sequence=sequence,
        event_id=str(sequence),
        event_type=event_type,
        role="",
        text="",
        status="pending",
        metadata=metadata,
        blob_digest="",
        turn_id="",
        created_at="now",
    )


def test_unresolved_decisions_merge_durable_records(tmp_path: Path) -> None:
    current = session(tmp_path)
    events = [
        _event(
            current.session_id,
            1,
            "approval.requested",
            {"approval_id": "approval-resolved"},
        ),
        _event(
            current.session_id,
            2,
            "approval.resolved",
            {"approval_id": "approval-resolved"},
        ),
        _event(
            current.session_id,
            3,
            "approval.requested",
            {"approval_id": "approval-open"},
        ),
        _event(
            current.session_id,
            4,
            "reconciliation.requested",
            {"reconciliation_id": "reconciliation-resolved"},
        ),
        _event(
            current.session_id,
            5,
            "reconciliation.resolved",
            {"reconciliation_id": "reconciliation-resolved"},
        ),
        _event(
            current.session_id,
            6,
            "reconciliation.requested",
            {"reconciliation_id": "reconciliation-open"},
        ),
        _event(current.session_id, 7, "approval.requested", {}),
    ]

    compiled = compile_context(
        current,
        events,
        durable_unresolved_decisions=[
            {"kind": "approval", "id": "approval-open"},
            {"kind": "approval", "id": "approval-durable"},
        ],
    )

    decisions = compiled.text.split("# Unresolved decisions\n\n")[1]
    decisions = decisions.split("\n```")[0]
    assert "approval-open" in decisions
    assert "approval-durable" in decisions
    assert "reconciliation-open" in decisions
    assert "approval-resolved" not in decisions
    assert "reconciliation-resolved" not in decisions


def test_fixed_sections_reject_budgets_below_their_contract(
    tmp_path: Path,
) -> None:
    current = session(tmp_path)

    with pytest.raises(ValueError, match="continuity contract"):
        context_module._fixed_sections(
            current,
            None,
            [],
            [],
            "",
            [],
            "",
            {},
            10,
        )

    minimum_chars = len(
        "# Harness session\n\nSession UUID: "
        + current.session_id
        + "\n\nWorkspace: "
        + current.worktree
    )
    minimum_chars += len(
        "# Continuity contract\n\n"
        "Continue the objective from this canonical observable state. "
        "Do not claim access to hidden reasoning from a prior provider. "
        "Inspect the workspace and harness history when more detail is needed."
    )
    minimum_chars += 2
    with pytest.raises(ValueError, match="fixed component headers"):
        context_module._fixed_sections(
            current,
            None,
            [],
            [],
            "",
            [],
            "",
            {},
            minimum_chars + 1,
        )


def test_fixed_sections_bound_optional_components(tmp_path: Path) -> None:
    current = session(tmp_path)

    sections = context_module._fixed_sections(
        current,
        None,
        [],
        [],
        "",
        [{"kind": "approval", "id": "open"}],
        "inherited fork narrative",
        {"schema": "p13i/agent-harness/compacted-history/v1", "event_count": 4},
        4_000,
    )

    joined = "\n\n".join(sections)
    assert "# Fork source context" in joined
    assert "# Compacted observable history" in joined
    assert "# Unresolved decisions" in joined


def test_bounded_section_compacts_then_degrades_to_a_digest() -> None:
    content = "y" * 4_000

    compacted = context_module._bounded_section("Goal", content, 400)
    assert compacted.startswith("# Goal\n\n")
    assert "[Compacted: sha256=" in compacted
    assert len(compacted) == 400

    minimal = context_module._bounded_section("Goal", content, 90)
    assert minimal.startswith("# Goal [sha256=")
    assert len(minimal) <= 90

    with pytest.raises(ValueError, match="Goal digest"):
        context_module._bounded_section("Goal", content, 4)


def test_compile_context_rejects_oversized_fixed_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = session(tmp_path)
    monkeypatch.setattr(
        context_module,
        "_fixed_sections",
        lambda *unused_args: ["z" * 1_000_000],
    )

    with pytest.raises(ValueError, match="bounded allocation"):
        compile_context(current, [])


def test_workspace_context_artifacts_skip_unsafe_and_saturated_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    plans = workspace / "plans"
    plans.mkdir(parents=True)
    (plans / "plan.md").write_text("plan body", encoding="utf-8")
    (plans / "nested.md").mkdir()
    (plans / "link.md").symlink_to(plans / "plan.md")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escape.md").write_text("escaped", encoding="utf-8")
    (plans / "escape.md").write_text("looks local", encoding="utf-8")
    original_resolve = Path.resolve

    def racing_resolve(current: Path, strict: bool = False) -> Path:
        resolved = original_resolve(current, strict=strict)
        if resolved.name == "escape.md" and resolved.parent.name == "plans":
            return outside / "escape.md"
        return resolved

    monkeypatch.setattr(Path, "resolve", racing_resolve)

    artifacts = workspace_context_artifacts(workspace)

    assert artifacts == ("# plans/plan.md\n\nplan body",)

    monkeypatch.undo()
    (workspace / "AGENTS.md").write_text("a" * 200_000, encoding="utf-8")

    saturated = workspace_context_artifacts(workspace)

    assert len(saturated) == 1
    assert saturated[0].startswith("# AGENTS.md\n\n")


def test_render_event_joins_text_with_metadata(tmp_path: Path) -> None:
    current = session(tmp_path)
    events = [
        replace(
            _event(current.session_id, 1, "user.message", {"tool": "grep"}),
            role="user",
            text="inspect the repository",
        )
    ]

    compiled = compile_context(current, events)

    assert "## Event 1: user.message (user)" in compiled.text
    assert 'Metadata: {"tool": "grep"}' in compiled.text
