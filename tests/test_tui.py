import asyncio
from pathlib import Path

import pytest
from textual.containers import Vertical
from textual.widgets import Input
from textual.widgets import Static

from agent_harness.tui import HarnessApp
from agent_harness.tui import _native_command


class Client:
    def __init__(self, *, theme: str = "system") -> None:
        self.theme = theme
        self.ui_updates = []
        self.requests = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ):
        del method
        del idempotency_key
        self.requests.append((path, payload))
        if payload is not None and path.endswith("/ui-state"):
            self.ui_updates.append(payload)
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
            return {
                "ui_state": {
                    "composer": "unfinished",
                    "provider": "codex",
                    "theme": self.theme,
                }
            }
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
                                "elapsed_seconds": 12.4,
                                "tool_calls": 3,
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
            assert "1200 / 300000" in str(inspector.render())
            assert "estimated" in str(inspector.render())

    asyncio.run(scenario())


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
