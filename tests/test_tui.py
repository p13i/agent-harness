import asyncio
from pathlib import Path

from textual.widgets import Input
from textual.widgets import Static

from agent_harness.tui import HarnessApp
from agent_harness.tui import _native_command


class Client:
    async def request(
        self,
        method: str,
        path: str,
        *,
        payload=None,
        idempotency_key: str = "",
    ):
        del method
        del payload
        del idempotency_key
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
                }
            }
        if path.startswith("/v1/providers"):
            return {
                "providers": {
                    "codex": {
                        "ready": True,
                        "usage": {"binding_percent": 40},
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
