import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess

import pytest
from textual import events
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input
from textual.widgets import ListView
from textual.widgets import Static
from textual.widgets import Tabs

from agent_harness.errors import HarnessError
from agent_harness.tui import HarnessApp
from agent_harness.tui import TranscriptBlockView
from agent_harness.tui import TranscriptView
from agent_harness.tui import _connection_label
from agent_harness.tui import _cursor_part
from agent_harness.tui import _display_lifecycle
from agent_harness.tui import _expanded_blocks
from agent_harness.tui import _format_number
from agent_harness.tui import _native_command
from agent_harness.tui import _object_tuple
from agent_harness.tui import _positive_integer
from agent_harness.tui import _providers_refreshing
from agent_harness.tui import _render_event
from agent_harness.tui import _render_transcript_block
from agent_harness.tui import _render_transcript_events
from agent_harness.tui import _session_list_label
from agent_harness.tui import _session_status
from agent_harness.tui import _status_glyph
from agent_harness.tui import _system_dark_mode
from agent_harness.tui import _transcript_block_classes
from agent_harness.tui import _visible_sessions
from agent_harness.tui_presenter import ColorScheme
from agent_harness.tui_presenter import ComposerState
from agent_harness.tui_presenter import InteractionState
from agent_harness.tui_presenter import LayoutMode
from agent_harness.tui_presenter import ThemePreference
from agent_harness.tui_presenter import TranscriptBlock
from agent_harness.tui_presenter import TranscriptBlockKind
from agent_harness.tui_presenter import TranscriptBlockStatus
from agent_harness.tui_presenter import TranscriptMutationKind
from agent_harness.tui_presenter import TranscriptState
from agent_harness.tui_presenter import TuiViewState
from agent_harness.tui_presenter import contrast_ratio
from agent_harness.tui_presenter import decide_layout
from agent_harness.tui_presenter import project_event
from agent_harness.tui_presenter import project_events
from agent_harness.tui_presenter import resolve_theme
from agent_harness.tui_presenter import safe_metadata
from agent_harness.tui_widgets import ComposerAction
from agent_harness.tui_widgets import ComposerDraft
from agent_harness.tui_widgets import MultilineComposer
from agent_harness.tui_widgets import SlashCommand
from agent_harness.tui_widgets import SlashValidationState
from agent_harness.tui_widgets import apply_completion
from agent_harness.tui_widgets import complete_slash
from agent_harness.tui_widgets import composer_action_for_key
from agent_harness.tui_widgets import composer_height
from agent_harness.tui_widgets import validate_slash


def test_idle_session_lifecycle_is_presented_as_ready() -> None:
    assert _display_lifecycle("starting", "idle") == "ready"
    assert _display_lifecycle("running", "idle") == "ready"
    assert _display_lifecycle("starting", "working") == "starting"


class Client:
    def __init__(self, *, theme: str = "system") -> None:
        self.theme = theme
        self.ui_updates = []
        self.requests = []
        self.ui_state = {
            "composer": "unfinished",
            "provider": "codex",
            "theme": self.theme,
        }

    async def request(
        self,
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ):
        del idempotency_key
        self.requests.append((path, payload))
        if (
            method == "PUT"
            and payload is not None
            and path.endswith("/ui-state")
        ):
            self.ui_updates.append(payload)
            self.ui_state = dict(payload)
        if path in {"/v1/sessions", "/v1/sessions?archived=1"}:
            return {
                "sessions": [
                    {
                        "session_id": "session-1",
                        "name": "Durable chat",
                        "lifecycle": "running",
                        "active_provider": "codex",
                    },
                    {
                        "session_id": "session-2",
                        "name": "Archived migration",
                        "lifecycle": "stopped",
                        "attention": "idle",
                        "active_provider": "",
                        "archived": True,
                    },
                ]
            }
        if path.endswith("/ui-state"):
            return {"ui_state": dict(self.ui_state)}
        if path.startswith("/v1/providers"):
            return {
                "providers": {
                    "codex": {
                        "ready": True,
                        "usage": {"binding_percent": 40},
                        "usage_refreshing": False,
                    }
                }
            }
        if path == "/v1/sync":
            return {
                "state_root": "/Users/test/my/chats",
                "sync": {"state": "synced"},
            }
        if path.endswith("/events?after=0"):
            return {"events": []}
        if path == "/v1/sessions/session-1":
            return {
                "session": {
                    "session_id": "session-1",
                    "name": "Durable chat",
                    "lifecycle": "running",
                    "attention": "idle",
                    "permission_mode": "approval",
                    "active_provider": "codex",
                    "model": "default",
                    "worktree": "/workspace",
                },
                "goal": {
                    "kind": "finite",
                    "status": "active",
                    "objective": "Finish the implementation.",
                },
                "approvals": [],
                "safety": {
                    "session": {"profile": "interactive"},
                    "envelopes": [
                        {
                            "state": "running",
                            "guard_reason": "",
                            "recovery_stage": 1,
                            "limits": {
                                "max_total_tokens": 300000,
                                "max_seconds": 3600,
                                "max_tool_calls": 256,
                            },
                            "consumption": {
                                "total_tokens": 1200,
                                "input_tokens": 1120,
                                "cached_input_tokens": 100,
                                "context_tokens": 800,
                                "elapsed_seconds": 12.4,
                                "tool_calls": 3,
                                "output_tokens": 80,
                                "exact_tokens": False,
                            },
                        }
                    ],
                    "incidents": [],
                },
            }
        if path.endswith("/budget-extensions"):
            return {"safety": {"profile": "interactive"}}
        if path.endswith("/usage"):
            return {
                "safety": {
                    "session": {"profile": "interactive"},
                    "envelopes": [],
                    "incidents": [],
                }
            }
        if path.endswith("/reconciliations"):
            return {
                "reconciliations": [
                    {
                        "reconciliation_id": "reconciliation-1",
                        "command_id": "command-1",
                        "pre_dispatch_checkpoint_id": "checkpoint-1",
                        "current_workspace_summary": "one changed file",
                        "current_workspace_digest": "digest-1",
                    }
                ]
            }
        return {"ui_state": {}}


class DisconnectingClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self.ui_state["composer"] = ""
        self.failures = 1
        self.message_keys: list[str] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ):
        if path.endswith("/messages"):
            self.message_keys.append(idempotency_key)
            if self.failures > 0:
                self.failures -= 1
                raise HarnessError(
                    "E_DAEMON",
                    "daemon disconnected",
                    retryable=True,
                    status=503,
                )
            return {
                "command": {
                    "command_id": "command-1",
                    "status": "queued",
                }
            }
        return await super().request(
            method,
            path,
            payload=payload,
            idempotency_key=idempotency_key,
        )


def test_textual_workspace_restores_draft_and_inspector(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        app = HarnessApp(
            Client(),
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert (
                app.query_one("#composer", MultilineComposer).text
                == "unfinished"
            )
            inspector = app.query_one("#inspector-content", Static)
            assert "Finish the implementation." in str(inspector.render())
            assert "codex" in str(inspector.render())
            assert "SAFETY ENVELOPE" in str(inspector.render())
            assert "interactive" in str(inspector.render())
            assert "1,200 / 300,000" in str(inspector.render())
            assert "1,120 · 100 cached" in str(inspector.render())
            assert "800 context est." in str(inspector.render())
            assert "estimated" in str(inspector.render())
            assert "CHAT STORAGE" in str(inspector.render())
            assert "/Users/test/my/chats" in str(inspector.render())
            session_list = app.query_one("#session-list", ListView)
            assert session_list.children[0].has_class("active-session")

    asyncio.run(scenario())


def test_textual_send_recovery_reuses_request_and_keeps_draft(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = DisconnectingClient()
        app = HarnessApp(
            client,
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            if app._poll_timer is not None:
                app._poll_timer.stop()
            composer = app.query_one(
                "#composer",
                MultilineComposer,
            )
            composer.focus()
            composer.text = "first line\nsecond line"
            await pilot.press("enter")
            await pilot.pause()

            request_id = app._pending_request_id
            assert request_id
            assert composer.text == "first line\nsecond line"
            assert client.message_keys == [request_id]
            assert client.ui_updates[-1]["request_id"] == request_id

            await app._recover_pending_message()
            await pilot.pause()
            assert client.message_keys == [request_id, request_id]
            assert app._pending_request_id == ""
            assert composer.text == ""
            assert client.ui_updates[-1]["request_id"] == ""

    asyncio.run(scenario())


def test_textual_search_archive_and_recovery_inspector(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = Client()
        app = HarnessApp(
            client,
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            search = app.query_one("#session-search", Input)
            search.value = "archived migration"
            await pilot.pause()
            assert len(app._visible_sessions) == 1
            assert app._visible_sessions[0]["session_id"] == "session-2"

            tabs = app.query_one("#inspector-tabs", Tabs)
            tabs.active = "recovery"
            await pilot.pause()
            await app._poll()
            inspector = app.query_one("#inspector-content", Static)
            assert "Interrupted command" in str(inspector.render())
            assert "one changed file" in str(inspector.render())

            await app._slash("/archive")
            assert any(
                path.endswith("/archive")
                for path, unused in client.requests
            )

    asyncio.run(scenario())


def test_transcript_reconciles_streaming_and_hides_protocol_noise() -> None:
    events = [
        {
            "sequence": 1,
            "event_type": "tool.userMessage.started",
            "text": "",
            "turn_id": "turn-1",
        },
        {
            "sequence": 2,
            "event_type": "agent.message.delta",
            "text": "Hi",
            "turn_id": "turn-1",
        },
        {
            "sequence": 3,
            "event_type": "agent.message.delta",
            "text": "!",
            "turn_id": "turn-1",
        },
        {
            "sequence": 4,
            "event_type": "agent.message",
            "text": "Hi!",
            "turn_id": "turn-1",
        },
    ]

    rendered = _render_transcript_events(events, show_events=False)

    assert len(rendered) == 1
    assert rendered[0].count("AGENT") == 1
    assert "Hi!" in rendered[0]
    assert "USERMESSAGE" not in rendered[0]


def test_legacy_event_renderer_covers_visible_event_contract() -> None:
    assert "YOU" in _render_event(
        {"event_type": "user.steer", "text": "change course"}
    )
    assert "AGENT" in _render_event(
        {"event_type": "agent.message.delta", "text": "working"}
    )
    assert (
        _render_event(
            {
                "event_type": "tool.user_message.completed",
                "text": "hidden",
            }
        )
        == ""
    )
    assert (
        _render_event({"event_type": "tool.shell.started"})
        == ""
    )
    assert "TOOL · shell completed" in _render_event(
        {
            "event_type": "tool.shell.completed",
            "text": "done",
        }
    )
    assert "[dim]thinking[/dim]" == _render_event(
        {
            "event_type": "reasoning.summary.delta",
            "text": "thinking",
        }
    )
    assert "approval-1" in _render_event(
        {
            "event_type": "approval.requested",
            "metadata": {"approval_id": "approval-1"},
        }
    )
    assert "above 80%" in _render_event(
        {"event_type": "guard.warning"}
    )
    assert "stagnation" in _render_event(
        {
            "event_type": "guard.tripped",
            "metadata": {
                "reason": "stagnation",
                "action": "pause",
            },
        }
    )
    assert "Routed to codex · default" in _render_event(
        {
            "event_type": "routing.selected",
            "metadata": {
                "provider": "codex",
                "model": "default",
            },
        }
    )
    assert "Failing over from claude" in _render_event(
        {
            "event_type": "routing.failover",
            "metadata": {"excluded_provider": "claude"},
        }
    )
    assert _render_event({"event_type": "checkpoint.created"})
    assert _render_event({"event_type": "goal.completed"})
    assert "Turn failed" in _render_event(
        {"event_type": "turn.failed", "text": "unavailable"}
    )
    assert "EVENT · custom.state" in _render_event(
        {"event_type": "custom.state"},
        show_events=True,
    )
    assert (
        _render_event(
            {"event_type": "usage.updated"},
            show_events=True,
        )
        == ""
    )


def test_transcript_block_renderer_covers_typed_visual_states() -> None:
    kinds = (
        TranscriptBlockKind.USER,
        TranscriptBlockKind.ASSISTANT,
        TranscriptBlockKind.TOOL,
        TranscriptBlockKind.APPROVAL,
        TranscriptBlockKind.RECONCILIATION,
        TranscriptBlockKind.WARNING,
        TranscriptBlockKind.SYSTEM,
    )
    rendered: dict[TranscriptBlockKind, str] = {}
    for kind in kinds:
        block = TranscriptBlock(
            block_id=kind.value,
            kind=kind,
            status=TranscriptBlockStatus.RUNNING,
            title=kind.value,
            content="content",
            detail="detail " * 100,
        )
        rendered[kind] = _render_transcript_block(
            block,
            expanded=False,
        )
        assert kind.value in _transcript_block_classes(block)
    assert "cyan" in rendered[TranscriptBlockKind.USER]
    assert "green" in rendered[TranscriptBlockKind.ASSISTANT]
    assert "yellow" in rendered[TranscriptBlockKind.TOOL]
    assert "magenta" in rendered[TranscriptBlockKind.APPROVAL]
    assert "#fb923c" in rendered[TranscriptBlockKind.RECONCILIATION]
    assert "red" in rendered[TranscriptBlockKind.WARNING]
    assert "Click or focus" in rendered[TranscriptBlockKind.TOOL]
    assert "…" in rendered[TranscriptBlockKind.TOOL]

    expanded = _render_transcript_block(
        TranscriptBlock(
            block_id="tool",
            kind=TranscriptBlockKind.TOOL,
            status=TranscriptBlockStatus.COMPLETE,
            title="tool",
            content="",
            detail="detail " * 100,
        ),
        expanded=True,
    )
    assert "Click or focus" not in expanded
    assert "…" not in expanded


def test_tui_projection_helpers_cover_invalid_and_attention_states() -> None:
    assert _cursor_part("3:7", 0) == 3
    assert _cursor_part("3:7", 1) == 7
    assert _cursor_part("invalid", 0) == 0
    assert _cursor_part("x:7", 0) == 0
    assert _expanded_blocks("") == frozenset()
    assert _expanded_blocks("invalid") == frozenset()
    assert _expanded_blocks("{}") == frozenset()
    assert _expanded_blocks('["one", 2]') == frozenset({"one"})
    assert _positive_integer("7") == 7
    assert _positive_integer("0") is None
    assert _positive_integer(None) is None
    assert _format_number(1234) == "1,234"
    assert _format_number("unknown") == "unknown"
    assert _object_tuple("invalid") == ()
    assert _object_tuple([{"one": 1}, "ignored"]) == ({"one": 1},)

    assert _session_status("running", "working") == "working"
    assert _session_status("paused", "needs-input") == "action needed"
    assert _session_status("running", "failed") == "needs attention"
    assert _status_glyph("working").startswith("[bold yellow]")
    assert _status_glyph("needs-reconciliation").startswith(
        "[bold magenta]"
    )
    assert _status_glyph("failed").startswith("[bold red]")
    assert _status_glyph("idle").startswith("[bold green]")
    assert "agent working" in _connection_label("working")
    assert "action needed" in _connection_label("needs-input")
    assert "needs attention" in _connection_label("failed")
    assert "connected" in _connection_label("idle")
    assert _providers_refreshing(
        {"codex": {"usage_refreshing": True}}
    )
    assert not _providers_refreshing({"codex": "unknown"})

    label = _session_list_label(
        {
            "session_id": "session-1",
            "name": "Build",
            "lifecycle": "running",
            "attention": "idle",
            "active_provider": "codex",
            "external_ref": {
                "orchestrator": "test",
                "job_id": "job",
            },
        },
        "session-1",
    )
    assert "● Build" in label
    assert "external job" in label
    assert "codex" in label
    assert "archived" in _session_list_label(
        {
            "session_id": "session-2",
            "name": "Old",
            "archived": True,
        },
        "",
    )


def test_system_theme_probe_handles_platform_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent_harness.tui.sys.platform", "linux")
    assert _system_dark_mode(True)

    monkeypatch.setattr("agent_harness.tui.sys.platform", "darwin")
    monkeypatch.setattr(
        "agent_harness.tui.subprocess.run",
        lambda *unused_args, **unused_kwargs: subprocess.CompletedProcess(
            ["defaults"],
            0,
            "Dark\n",
            "",
        ),
    )
    assert _system_dark_mode(False)
    monkeypatch.setattr(
        "agent_harness.tui.subprocess.run",
        lambda *unused_args, **unused_kwargs: subprocess.CompletedProcess(
            ["defaults"],
            1,
            "",
            "",
        ),
    )
    assert not _system_dark_mode(True)

    def unavailable(
        *unused_args: object,
        **unused_kwargs: object,
    ) -> None:
        del unused_args
        del unused_kwargs
        raise OSError("unavailable")

    monkeypatch.setattr(
        "agent_harness.tui.subprocess.run",
        unavailable,
    )
    assert _system_dark_mode(True)


def test_focused_sessions_keep_attention_and_bound_idle_clutter() -> None:
    sessions = []
    for index in range(9):
        sessions.append(
            {
                "session_id": "session-" + str(index),
                "lifecycle": "running",
                "attention": "idle",
            }
        )
    sessions.append(
        {
            "session_id": "paused",
            "lifecycle": "paused",
            "attention": "needs-input",
        }
    )
    sessions.append(
        {
            "session_id": "stopped",
            "lifecycle": "stopped",
            "attention": "idle",
        }
    )

    visible, hidden = _visible_sessions(
        sessions,
        "session-8",
        show_all=False,
    )

    assert {item["session_id"] for item in visible} >= {
        "session-8",
        "paused",
    }
    assert len(visible) == 7
    assert hidden == 4
    assert _visible_sessions(
        sessions,
        "session-8",
        show_all=True,
    ) == (sessions, 0)


def test_textual_workspace_theme_and_responsive_visual_contract(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = Client(theme="dark")
        app = HarnessApp(
            client,
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            dark_svg = app.export_screenshot()
            assert dark_svg.startswith("<svg")
            assert "P13I AGENT HARNESS" in str(
                app.query_one("#brand", Static).render()
            )
            assert "SESSION CONTROL" in str(
                app.query_one("#inspector-heading", Static).render()
            )
            assert not app.screen.has_class("compact")
            assert app.query_one("#inspector", Vertical).display

            await app._slash("/theme light")
            await pilot.pause()
            light_svg = app.export_screenshot()
            assert light_svg != dark_svg
            assert "#ffffff" in light_svg
            assert "#bae6fd" in light_svg
            assert "#0369a1" in light_svg
            assert app.screen.has_class("light")
            assert client.ui_updates[-1]["theme"] == "light"

            await pilot.resize_terminal(100, 36)
            await pilot.pause()
            assert app.screen.has_class("compact")
            assert not app.query_one("#inspector", Vertical).display

            await pilot.resize_terminal(70, 32)
            await pilot.pause()
            assert app.screen.has_class("overlay")
            assert app.query_one("#sidebar", Vertical).display

            await pilot.resize_terminal(60, 20)
            await pilot.pause()
            assert app.screen.has_class("narrow")
            assert not app.query_one("#sidebar", Vertical).display

    asyncio.run(scenario())


def test_sidebar_resize_is_keyboard_pointer_and_resume_persistent(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = Client()
        app = HarnessApp(
            client,
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            sidebar = app.query_one("#sidebar", Vertical)
            initial_width = sidebar.outer_size.width

            await pilot.press("ctrl+shift+right")
            await pilot.pause()
            assert sidebar.outer_size.width == initial_width + 4

            await pilot.mouse_down(
                "#sidebar-resize-handle",
                offset=(0, 5),
            )
            await pilot.mouse_up(offset=(48, 8))
            await pilot.pause()
            assert app._sidebar_width == 48
            assert sidebar.outer_size.width == 48
            assert client.ui_updates[-1]["sidebar_width"] == "48"

        restored = HarnessApp(
            client,
            tmp_path,
            session_id="session-1",
        )
        async with restored.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            assert restored.query_one(
                "#sidebar",
                Vertical,
            ).outer_size.width == 48

            await restored._slash("/sidebar reset")
            await pilot.pause()
            assert restored.query_one(
                "#sidebar",
                Vertical,
            ).outer_size.width == 31

    asyncio.run(scenario())


def test_transcript_uses_stable_incremental_blocks(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        app = HarnessApp(
            Client(),
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            transcript = app.query_one("#transcript", TranscriptView)
            assert len(transcript.children) == 1
            assert "Durable session ready" in str(
                transcript.children[0].render()
            )
            await app._apply_transcript_events(
                [
                    {
                        "sequence": 1,
                        "event_type": "agent.message.delta",
                        "turn_id": "turn-1",
                        "text": "Work",
                    }
                ]
            )
            block = transcript.children[-1]
            assert isinstance(block, TranscriptBlockView)
            assert block.block_id == "assistant:turn-1"
            original = block
            await app._apply_transcript_events(
                [
                    {
                        "sequence": 2,
                        "event_type": "agent.message.delta",
                        "turn_id": "turn-1",
                        "text": "ing",
                    }
                ]
            )
            assert transcript.children[-1] is original
            assert "Working" in str(original.render())

    asyncio.run(scenario())


def test_system_theme_tracks_host_appearance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appearance = {"dark": False}

    def system_theme(default: bool) -> bool:
        del default
        return appearance["dark"]

    monkeypatch.setattr(
        "agent_harness.tui._system_dark_mode",
        system_theme,
    )

    async def scenario() -> None:
        app = HarnessApp(
            Client(theme="system"),
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.screen.has_class("light")
            assert app.theme == "textual-light"

            appearance["dark"] = True
            await app._sync_system_theme()
            await pilot.pause()
            assert not app.screen.has_class("light")
            assert app.theme == "textual-dark"

    asyncio.run(scenario())


def test_poll_after_textual_teardown_is_noop(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        app = HarnessApp(
            Client(),
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
        await app._poll()

    asyncio.run(scenario())


def test_native_attachment_uses_pinned_provider_packages() -> None:
    assert _native_command("codex", "full") == [
        "npx",
        "-y",
        "@openai/codex@0.146.0",
        "--yolo",
    ]
    assert _native_command("claude", "full") == [
        "npx",
        "@anthropic-ai/claude-code@2.1.220",
        "--dangerously-skip-permissions",
    ]


def test_budget_commands_require_explicit_operator_actions(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = Client()
        app = HarnessApp(
            client,
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await app._slash(
                "/budget extend 60 1000 finish bounded validation"
            )
            await app._slash("/budget xhigh inspect one hard failure")

        extensions = [
            payload
            for path, payload in client.requests
            if path.endswith("/budget-extensions")
        ]
        assert extensions == [
            {
                "reason": "finish bounded validation",
                "additional_seconds": 60,
                "additional_tokens": 1000,
            },
            {
                "reason": "inspect one hard failure",
                "allow_xhigh_once": True,
            },
        ]

    asyncio.run(scenario())


def test_presenter_applies_streaming_deltas_incrementally() -> None:
    state = TranscriptState(reader_at_bottom=False)
    update = project_events(
        state,
        [
            {
                "sequence": 1,
                "event_id": "user-event",
                "event_type": "user.message",
                "turn_id": "turn-1",
                "text": "Build it",
            },
            {
                "sequence": 2,
                "event_type": "agent.message.delta",
                "turn_id": "turn-1",
                "text": "Work",
            },
            {
                "sequence": 3,
                "event_type": "agent.message.delta",
                "turn_id": "turn-1",
                "text": "ing",
            },
        ],
    )

    assert [block.kind for block in update.state.blocks] == [
        TranscriptBlockKind.USER,
        TranscriptBlockKind.ASSISTANT,
    ]
    assistant = update.state.block("assistant:turn-1")
    assert assistant is not None
    assert assistant.content == "Working"
    assert assistant.status == TranscriptBlockStatus.STREAMING
    assert [mutation.kind for mutation in update.mutations] == [
        TranscriptMutationKind.INSERT,
        TranscriptMutationKind.INSERT,
        TranscriptMutationKind.APPEND,
    ]
    assert update.mutations[-1].delta == "ing"
    assert update.mutations[-1].block is None
    assert update.state.new_activity_count == 3

    final = project_event(
        update.state,
        {
            "sequence": 4,
            "event_type": "agent.message",
            "turn_id": "turn-1",
            "text": "Working.",
        },
    )
    assert final.mutations[0].kind == TranscriptMutationKind.REPLACE
    assistant = final.state.block("assistant:turn-1")
    assert assistant is not None
    assert assistant.content == "Working."
    assert assistant.status == TranscriptBlockStatus.COMPLETE
    assert final.state.latest_sequence == 4

    duplicate = project_event(
        final.state,
        {
            "sequence": 4,
            "event_type": "agent.message",
            "turn_id": "turn-1",
            "text": "duplicate",
        },
    )
    assert duplicate.state == final.state
    assert duplicate.mutations[0].kind == TranscriptMutationKind.IGNORE


def test_presenter_preserves_reader_state_and_filters_protocol_noise() -> None:
    state = TranscriptState(
        expanded_block_ids=frozenset({"tool:one"}),
        reader_at_bottom=False,
        new_activity_count=2,
    )
    noise = project_event(
        state,
        {
            "sequence": 1,
            "event_type": "tool.userMessage.started",
        },
    )
    assert noise.state.blocks == ()
    assert noise.state.new_activity_count == 2
    assert noise.state.latest_sequence == 1

    toggled = noise.state.toggle_expanded("tool:one")
    assert toggled.expanded_block_ids == frozenset()
    toggled = toggled.toggle_expanded("tool:two")
    assert toggled.expanded_block_ids == frozenset({"tool:two"})
    assert toggled.with_reader_at_bottom(False).new_activity_count == 2
    assert toggled.with_reader_at_bottom(True).new_activity_count == 0
    assert toggled.block("absent") is None


def test_presenter_types_tools_without_provider_arguments() -> None:
    started = project_event(
        TranscriptState(),
        {
            "sequence": 1,
            "event_type": "tool.shell.started",
            "tool_call_id": "call-1",
            "text": "GITHUB_TOKEN=secret destructive-command",
            "metadata": {
                "summary": "Run focused tests",
                "detail": "pytest tests/test_tui.py",
                "raw_arguments": "--token secret",
                "prompt": "private prompt",
            },
        },
    )
    block = started.state.block("tool:call-1")
    assert block is not None
    assert block.kind == TranscriptBlockKind.TOOL
    assert block.status == TranscriptBlockStatus.RUNNING
    assert block.content == "Run focused tests"
    assert block.detail == "pytest tests/test_tui.py"
    assert "secret" not in repr(block)

    completed = project_event(
        started.state,
        {
            "sequence": 2,
            "event_type": "tool.shell.completed",
            "metadata": {
                "tool_call_id": "call-1",
                "summary": "Focused tests passed",
            },
        },
    )
    assert completed.mutations[0].kind == TranscriptMutationKind.REPLACE
    block = completed.state.block("tool:call-1")
    assert block is not None
    assert block.status == TranscriptBlockStatus.COMPLETE

    failed = project_event(
        completed.state,
        {
            "sequence": 3,
            "event_type": "tool.read.error",
            "call_id": "call-2",
            "metadata": {"reason": "unavailable"},
        },
    )
    failed_block = failed.state.block("tool:call-2")
    assert failed_block is not None
    assert failed_block.status == TranscriptBlockStatus.FAILED

    canonical_started = project_event(
        failed.state,
        {
            "sequence": 4,
            "event_type": "tool.started",
            "text": "Shell",
            "metadata": {
                "id": "canonical-call",
                "name": "Shell",
                "input": {"command": "private provider arguments"},
            },
        },
    )
    canonical_block = canonical_started.state.block(
        "tool:canonical-call"
    )
    assert canonical_block is not None
    assert canonical_block.title == "Tool · Shell"
    assert "private provider arguments" not in repr(canonical_block)

    canonical_completed = project_event(
        canonical_started.state,
        {
            "sequence": 5,
            "event_type": "tool.completed",
            "text": "2 tests passed",
            "metadata": {
                "tool_use_id": "canonical-call",
                "is_error": True,
            },
        },
    )
    canonical_block = canonical_completed.state.block(
        "tool:canonical-call"
    )
    assert canonical_block is not None
    assert canonical_block.status == TranscriptBlockStatus.FAILED
    assert canonical_block.detail == "2 tests passed"
    assert canonical_block.title == "Tool · Shell"

    anonymous = project_event(
        canonical_completed.state,
        {
            "sequence": 6,
            "event_type": "tool.unknown.started",
        },
    )
    assert anonymous.state.blocks[-1].block_id == (
        "sequence:6:tool.unknown.started"
    )


def test_presenter_types_attention_recovery_and_lifecycle_events() -> None:
    events_to_project = [
        {
            "sequence": 1,
            "event_type": "approval.requested",
            "text": "Allow the edit?",
            "metadata": {"approval_id": "approval-1"},
        },
        {
            "sequence": 2,
            "event_type": "reconciliation.requested",
            "text": "Inspect the workspace",
            "metadata": {"reconciliation_id": "recovery-1"},
        },
        {
            "sequence": 3,
            "event_type": "reconciliation.resolved",
            "text": "Accepted current work",
            "metadata": {"reconciliation_id": "recovery-1"},
        },
        {
            "sequence": 4,
            "event_type": "guard.warning",
            "metadata": {"reason": "Budget above 80%"},
        },
        {
            "sequence": 5,
            "event_type": "guard.tripped",
            "metadata": {"reason": "Stagnation", "action": "pause"},
        },
        {
            "sequence": 6,
            "event_type": "turn.failed",
            "text": "Provider unavailable",
        },
        {
            "sequence": 7,
            "event_type": "checkpoint.created",
        },
        {
            "sequence": 8,
            "event_type": "routing.selected",
            "metadata": {"summary": "Claude · Opus"},
        },
        {
            "sequence": 9,
            "event_type": "sync.conflict",
        },
    ]

    update = project_events(TranscriptState(), events_to_project)

    assert update.state.blocks[0].kind == TranscriptBlockKind.APPROVAL
    recovery = update.state.block("reconciliation:recovery-1")
    assert recovery is not None
    assert recovery.status == TranscriptBlockStatus.COMPLETE
    warnings = [
        block
        for block in update.state.blocks
        if block.kind == TranscriptBlockKind.WARNING
    ]
    assert [block.status for block in warnings] == [
        TranscriptBlockStatus.GUARDED,
        TranscriptBlockStatus.GUARDED,
        TranscriptBlockStatus.FAILED,
    ]
    assert update.state.blocks[-1].content == (
        "Storage synchronization conflict"
    )


def test_presenter_unknown_events_are_explicit_only_when_requested() -> None:
    hidden = project_event(
        TranscriptState(),
        {
            "sequence": "not-an-integer",
            "event_id": "opaque-event",
            "event_type": "provider.private.payload",
            "metadata": {"token": "secret"},
        },
    )
    assert hidden.state.blocks == ()
    assert hidden.state.latest_sequence == 0

    duplicate = project_event(
        hidden.state,
        {
            "event_id": "opaque-event",
            "event_type": "provider.private.payload",
        },
        show_events=True,
    )
    assert duplicate.state.blocks == ()

    shown = project_event(
        TranscriptState(),
        {
            "sequence": 1,
            "event_id": "another-event",
            "event_type": "provider.private.payload",
        },
        show_events=True,
    )
    assert shown.state.blocks[0].kind == TranscriptBlockKind.SYSTEM
    assert shown.state.blocks[0].content == "provider · private · payload"

    approval_without_metadata = project_event(
        shown.state,
        {
            "sequence": 2,
            "event_type": "approval.requested",
        },
    )
    assert approval_without_metadata.state.blocks[-1].block_id == (
        "sequence:2:approval.requested"
    )


def test_layout_decisions_cover_declared_breakpoints() -> None:
    minimal = decide_layout(60, 20)
    overlay = decide_layout(80, 24, sidebar_requested=False)
    compact = decide_layout(100, 30)
    wide = decide_layout(120, 36, inspector_requested=False)
    spacious = decide_layout(160, 48)

    assert minimal.mode == LayoutMode.MINIMAL
    assert minimal.sidebar_mode == "collapsed"
    assert not minimal.sidebar_visible
    assert minimal.composer_max_lines == 3
    assert overlay.mode == LayoutMode.OVERLAY
    assert not overlay.sidebar_visible
    assert compact.mode == LayoutMode.COMPACT
    assert compact.sidebar_visible
    assert not compact.inspector_visible
    assert wide.mode == LayoutMode.WIDE
    assert not wide.inspector_visible
    assert spacious.mode == LayoutMode.SPACIOUS
    assert spacious.inspector_visible
    assert spacious.transcript_horizontal_padding == 3

    normalized = decide_layout(0, 0)
    assert normalized.width == 1
    assert normalized.height == 1
    assert normalized.composer_max_lines == 2


def test_theme_tokens_follow_system_and_keep_readable_contrast() -> None:
    system_dark = resolve_theme(
        ThemePreference.SYSTEM,
        system_dark=True,
    )
    explicit_light = resolve_theme("light", system_dark=True)
    fallback = resolve_theme("invalid", system_dark=False)

    assert system_dark.scheme == ColorScheme.DARK
    assert explicit_light.scheme == ColorScheme.LIGHT
    assert fallback.scheme == ColorScheme.LIGHT
    assert (
        contrast_ratio(
            system_dark.colors.text,
            system_dark.colors.canvas,
        )
        >= 7
    )
    assert (
        contrast_ratio(
            system_dark.colors.text_muted,
            system_dark.colors.surface,
        )
        >= 4.5
    )
    assert (
        contrast_ratio(
            explicit_light.colors.text,
            explicit_light.colors.canvas,
        )
        >= 7
    )
    assert system_dark.spacing.panel == 3
    with pytest.raises(ValueError, match="#RRGGBB"):
        contrast_ratio("white", "#000000")
    with pytest.raises(ValueError, match="#RRGGBB"):
        contrast_ratio("#GGGGGG", "#000000")


def test_safe_metadata_is_immutable_and_bounded() -> None:
    assert dict(safe_metadata(None)) == {}
    metadata = safe_metadata(
        {
            "summary": "Visible",
            "detail": "Bounded detail",
            "status": "running",
            "token": "secret",
            "raw_arguments": "--secret",
            "unrecognized": "hidden",
        }
    )
    assert dict(metadata) == {
        "summary": "Visible",
        "detail": "Bounded detail",
        "status": "running",
    }
    with pytest.raises(TypeError):
        metadata["summary"] = "changed"  # type: ignore[index]


def test_tui_view_state_preserves_interaction_during_events() -> None:
    view = TuiViewState(
        transcript=TranscriptState(),
        composer=ComposerState(
            text="draft",
            cursor_row=0,
            cursor_column=5,
            request_id="request-1",
            awaiting_acknowledgement=True,
        ),
        interaction=InteractionState(
            focus_id="composer",
            transcript_scroll_y=12,
            selection_anchor="assistant:old",
            active_inspector_tab="Recovery",
        ),
        layout=decide_layout(120, 36),
        tokens=resolve_theme("dark", system_dark=False),
        connection_state="reconnecting",
        validation_state="validating",
    )

    projected, mutations = view.with_events(
        [
            {
                "sequence": 1,
                "event_type": "session.started",
            }
        ]
    )

    assert projected.interaction is view.interaction
    assert projected.composer is view.composer
    assert projected.layout is view.layout
    assert mutations[0].kind == TranscriptMutationKind.INSERT
    with pytest.raises(FrozenInstanceError):
        projected.connection_state = "connected"  # type: ignore[misc]


def test_composer_key_and_height_semantics() -> None:
    assert composer_action_for_key("enter") == ComposerAction.SEND
    assert composer_action_for_key("SHIFT+ENTER") == ComposerAction.NEWLINE
    assert composer_action_for_key("ctrl+j") == ComposerAction.NEWLINE
    assert composer_action_for_key("x") == ComposerAction.EDIT
    assert composer_action_for_key("enter", pasted=True) == ComposerAction.EDIT

    assert composer_height("", wrap_width=10) == 1
    assert composer_height("one\ntwo", wrap_width=10) == 2
    assert composer_height("x" * 21, wrap_width=10) == 3
    assert composer_height("x" * 100, wrap_width=10, max_lines=4) == 4
    assert composer_height("", wrap_width=10, min_lines=2) == 2
    with pytest.raises(ValueError, match="wrap_width"):
        composer_height("", wrap_width=0)
    with pytest.raises(ValueError, match="min_lines"):
        composer_height("", wrap_width=1, min_lines=0)
    with pytest.raises(ValueError, match="max_lines"):
        composer_height("", wrap_width=1, min_lines=2, max_lines=1)


def test_slash_completion_is_fuzzy_navigable_and_argument_preserving() -> None:
    assert complete_slash("ordinary text").items == ()
    exact = complete_slash("/theme")
    assert exact.selected is not None
    assert exact.selected.command.name == "/theme"
    assert exact.selected.insertion == "/theme "

    fuzzy = complete_slash("/prv")
    assert fuzzy.selected is not None
    assert fuzzy.selected.command.name == "/provider"
    prefix = complete_slash("/pro")
    assert prefix.selected is not None
    assert prefix.selected.command.name == "/provider"
    moved = fuzzy.move(1)
    assert moved.selected is not None
    assert moved.move(-1).selected == fuzzy.selected
    empty = complete_slash("text")
    assert empty.selected is None
    assert empty.move(1) is empty

    completion = exact.selected
    assert completion is not None
    assert apply_completion("/th dark", completion) == "/theme dark"
    assert apply_completion("  /th", completion) == "  /theme "
    assert apply_completion("message", completion) == "message"
    with pytest.raises(ValueError, match="limit"):
        complete_slash("/", limit=0)


def test_slash_validation_keeps_invalid_commands_editable() -> None:
    not_command = validate_slash("build it")
    assert not_command.state == SlashValidationState.NOT_COMMAND
    assert not not_command.can_execute

    incomplete = validate_slash("/theme")
    assert incomplete.state == SlashValidationState.INCOMPLETE
    assert incomplete.command is not None
    assert incomplete.message == "/theme <system|light|dark>"
    assert validate_slash("/").message == "Type a harness command"

    valid = validate_slash("/theme dark")
    assert valid.state == SlashValidationState.VALID
    assert valid.can_execute
    assert valid.message == "Choose appearance"

    unknown = validate_slash("/not-a-command")
    assert unknown.state == SlashValidationState.INVALID
    assert unknown.command is None
    assert unknown.message == "Unknown harness command"
    assert validate_slash('/fork "unfinished').state == (
        SlashValidationState.INVALID
    )
    assert validate_slash("/theme ultraviolet").state == (
        SlashValidationState.INVALID
    )
    assert validate_slash("/theme dark extra").state == (
        SlashValidationState.INVALID
    )

    custom = SlashCommand(
        "/custom",
        "Custom",
        "/custom <value>",
        min_arguments=1,
        max_arguments=None,
    )
    assert validate_slash(
        "/custom one two",
        commands=(custom,),
    ).can_execute


class ComposerTestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[str] = []

    def compose(self) -> ComposeResult:
        yield MultilineComposer(
            min_lines=1,
            max_lines=4,
            wrap_width=10,
            id="multiline-composer",
        )

    def on_multiline_composer_submitted(
        self,
        event: MultilineComposer.Submitted,
    ) -> None:
        self.submissions.append(event.text)


def test_multiline_composer_sends_newlines_pastes_and_restores() -> None:
    async def scenario() -> None:
        app = ComposerTestApp()
        async with app.run_test(size=(80, 24)) as pilot:
            composer = app.query_one(
                "#multiline-composer",
                MultilineComposer,
            )
            composer.focus()
            composer.text = "first"
            composer.move_cursor((0, 5))

            await pilot.press("shift+enter")
            await pilot.press("ctrl+j")
            await pilot.pause()
            assert composer.text == "first\n\n"
            assert app.submissions == []

            await composer._on_paste(events.Paste("pasted\ncontent"))
            await pilot.pause()
            assert composer.text == "first\n\npasted\ncontent"
            assert app.submissions == []
            assert composer.refresh_auto_height() == 4

            draft = composer.capture_draft()
            assert draft == ComposerDraft(
                text="first\n\npasted\ncontent",
                cursor_row=3,
                cursor_column=7,
            )
            composer.restore_draft(
                ComposerDraft(
                    text="restored",
                    cursor_row=100,
                    cursor_column=100,
                )
            )
            assert composer.capture_draft() == ComposerDraft(
                text="restored",
                cursor_row=0,
                cursor_column=8,
            )

            await pilot.press("enter")
            await pilot.pause()
            assert app.submissions == ["restored"]
            assert composer.text == "restored"

            composer.text = "   "
            await pilot.press("enter")
            await pilot.pause()
            assert app.submissions == ["restored"]

    asyncio.run(scenario())


def test_multiline_composer_validates_bounds_and_ignores_foreign_change() -> None:
    with pytest.raises(ValueError, match="min_lines"):
        MultilineComposer(min_lines=0)
    with pytest.raises(ValueError, match="max_lines"):
        MultilineComposer(min_lines=2, max_lines=1)
    with pytest.raises(ValueError, match="wrap_width"):
        MultilineComposer(wrap_width=0)

    first = MultilineComposer()
    second = MultilineComposer()
    event = MultilineComposer.Changed(second)
    first.on_text_area_changed(event)
    submitted = MultilineComposer.Submitted(first, "message")
    assert submitted.control is first
