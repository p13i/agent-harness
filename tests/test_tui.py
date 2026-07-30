import asyncio
from pathlib import Path

import pytest
from textual.containers import Vertical
from textual.widgets import Input
from textual.widgets import ListView
from textual.widgets import Static

from agent_harness.tui import ConversationLog
from agent_harness.tui import HarnessApp
from agent_harness.tui import _display_lifecycle
from agent_harness.tui import _native_command
from agent_harness.tui import _render_transcript_events
from agent_harness.tui import _visible_sessions


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
        if path == "/v1/sessions":
            return {
                "sessions": [
                    {
                        "session_id": "session-1",
                        "name": "Durable chat",
                        "lifecycle": "running",
                        "active_provider": "codex",
                    }
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
        return {"ui_state": {}}


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
            assert app.query_one("#composer", Input).value == "unfinished"
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


def test_short_transcript_is_anchored_above_composer(
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
            transcript = app.query_one("#transcript", ConversationLog)
            assert len(transcript.lines) < (
                transcript.scrollable_content_region.height
            )
            assert transcript.render_line(0).text.strip() == ""
            padding = (
                transcript.scrollable_content_region.height
                - len(transcript.lines)
            )
            assert "Durable session ready" in (
                transcript.render_line(padding).text
            )

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
