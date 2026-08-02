"""Render deterministic galleries from the real Textual widget tree."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from textual.containers import Horizontal
from textual.widgets import Static

from agent_harness.tui import HarnessApp

FIXTURES = (
    "empty",
    "new-session",
    "streaming",
    "tool-heavy",
    "approval",
    "guarded",
    "disconnected",
    "reconciliation",
    "long-code",
    "archived",
    "high-session-count",
)
MODES = ("focus", "control")
BREAKPOINTS = (
    (60, 20),
    (80, 24),
    (120, 36),
    (160, 48),
)
THEMES = ("light", "dark")


class GalleryClient:
    """Network-free canonical fixture source for visual evidence."""

    def __init__(self, fixture: str, theme: str) -> None:
        self.fixture = fixture
        self.theme = theme
        self.ui_state: dict[str, Any] = {
            "composer": "",
            "theme": theme,
            "workspace_mode": "focus",
        }

    async def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        del idempotency_key
        if method == "PUT" and path.endswith("/ui-state"):
            if payload is not None:
                self.ui_state = dict(payload)
            return {"ui_state": dict(self.ui_state)}
        if method == "PATCH" and path == "/v1/sessions/gallery-session":
            return {"session": self._session()}
        if method == "POST" and path == "/v1/sync":
            return {"sync": {"state": "synced"}}
        if path in {"/v1/sessions", "/v1/sessions?archived=1"}:
            return {"sessions": self._sessions()}
        if path == "/v1/sync":
            return {
                "state_root": "/workspace/chats",
                "sync": {"state": "synced"},
            }
        if path == "/v1/providers":
            return {
                "providers": {
                    "codex": {
                        "ready": True,
                        "usage": {"binding_percent": 42},
                        "usage_refreshing": False,
                    },
                    "claude": {
                        "ready": True,
                        "usage": {"binding_percent": 31},
                        "usage_refreshing": False,
                    },
                }
            }
        if path.endswith("/ui-state"):
            return {"ui_state": dict(self.ui_state)}
        if path.endswith("/events?after=0"):
            return {"events": self._events()}
        if "/events?after=" in path:
            return {"events": []}
        if path.endswith("/turns?limit=200"):
            return {"turns": self._turns()}
        if path.endswith("/reconciliations"):
            return {"reconciliations": self._reconciliations()}
        if path.endswith("/usage"):
            return {"safety": self._safety()}
        if path.endswith("/budget-extensions"):
            return {"safety": self._safety()}
        if path == "/v1/sessions/gallery-session":
            return self._session_snapshot()
        return {}

    def _session(self) -> dict[str, Any]:
        lifecycle = "running"
        attention = "idle"
        if self.fixture == "streaming" or self.fixture == "tool-heavy":
            attention = "working"
        if self.fixture == "approval":
            attention = "needs-input"
        if self.fixture == "guarded":
            lifecycle = "paused"
            attention = "guarded"
        if self.fixture == "disconnected":
            attention = "working"
        if self.fixture == "reconciliation":
            lifecycle = "paused"
            attention = "needs-input"
        if self.fixture == "archived":
            lifecycle = "stopped"
        return {
            "session_id": "gallery-session",
            "name": self.fixture.replace("-", " ").title(),
            "workspace": "/workspace/agent-harness",
            "worktree": "/workspace/agent-harness",
            "lifecycle": lifecycle,
            "attention": attention,
            "active_provider": "codex",
            "model": "default",
            "effort": "high",
            "permission_mode": "approval",
            "archived": self.fixture == "archived",
        }

    def _session_snapshot(self) -> dict[str, Any]:
        approvals: list[dict[str, Any]] = []
        if self.fixture == "approval":
            approvals.append(
                {
                    "approval_id": "approval-gallery",
                    "kind": "workspace-write",
                    "prompt": "Apply the checkpointed workspace change?",
                    "choices": [
                        {"id": "approve", "label": "Approve"},
                        {"id": "deny", "label": "Deny"},
                    ],
                }
            )
        return {
            "session": self._session(),
            "goal": {
                "kind": "finite",
                "status": "active",
                "objective": "Ship a bounded, verified agent workspace.",
            },
            "approvals": approvals,
            "safety": self._safety(),
        }

    def _sessions(self) -> list[dict[str, Any]]:
        count = 5
        if self.fixture == "high-session-count":
            count = 32
        sessions = [self._session()]
        for index in range(1, count):
            sessions.append(
                {
                    "session_id": "gallery-" + str(index),
                    "name": "Build lane " + str(index),
                    "lifecycle": "running",
                    "attention": "idle",
                    "active_provider": "claude",
                    "archived": False,
                }
            )
        return sessions

    def _events(self) -> list[dict[str, Any]]:
        if self.fixture in {"empty", "new-session"}:
            return []
        user = {
            "sequence": 1,
            "event_id": "event-user",
            "event_type": "user.message",
            "role": "user",
            "turn_id": "turn-gallery",
            "text": "Implement the durable presentation contract.",
            "status": "complete",
            "metadata": {},
        }
        if self.fixture == "streaming":
            return [
                user,
                {
                    "sequence": 2,
                    "event_id": "event-agent",
                    "event_type": "agent.message.delta",
                    "role": "assistant",
                    "turn_id": "turn-gallery",
                    "text": "Validating the interface and contract",
                    "status": "streaming",
                    "metadata": {},
                },
            ]
        if self.fixture == "tool-heavy":
            return [
                user,
                {
                    "sequence": 2,
                    "event_id": "event-agent",
                    "event_type": "agent.message",
                    "role": "assistant",
                    "turn_id": "turn-gallery",
                    "text": "The bounded implementation is ready.",
                    "status": "complete",
                    "metadata": {},
                },
                {
                    "sequence": 3,
                    "event_id": "event-tool",
                    "event_type": "tool.result",
                    "role": "tool",
                    "turn_id": "turn-gallery",
                    "text": "142 tests passed",
                    "status": "complete",
                    "metadata": {"name": "bazel test"},
                },
            ]
        if self.fixture == "approval":
            return [
                user,
                {
                    "sequence": 2,
                    "event_id": "event-approval",
                    "event_type": "approval.requested",
                    "role": "",
                    "turn_id": "turn-gallery",
                    "text": "Apply the checkpointed workspace change?",
                    "status": "pending",
                    "metadata": {"approval_id": "approval-gallery"},
                },
            ]
        if self.fixture == "guarded":
            return [
                user,
                {
                    "sequence": 2,
                    "event_id": "event-guard",
                    "event_type": "safety.guarded",
                    "role": "",
                    "turn_id": "turn-gallery",
                    "text": "",
                    "status": "guarded",
                    "metadata": {"reason": "Budget limit reached"},
                },
            ]
        if self.fixture == "reconciliation":
            return [
                user,
                {
                    "sequence": 2,
                    "event_id": "event-recovery",
                    "event_type": "reconciliation.requested",
                    "role": "",
                    "turn_id": "turn-gallery",
                    "text": "",
                    "status": "pending",
                    "metadata": {"reconciliation_id": "reconciliation-gallery"},
                },
            ]
        if self.fixture == "long-code":
            return [
                user,
                {
                    "sequence": 2,
                    "event_id": "event-code",
                    "event_type": "agent.message",
                    "role": "assistant",
                    "turn_id": "turn-gallery",
                    "text": (
                        "```python\n"
                        "def bounded_turn(store, session_id):\n"
                        "    checkpoint = store.checkpoint(session_id)\n"
                        "    return checkpoint.as_dict()\n"
                        "```"
                    ),
                    "status": "complete",
                    "metadata": {},
                },
            ]
        if self.fixture == "archived":
            return [
                {
                    "sequence": 1,
                    "event_id": "event-archived",
                    "event_type": "session.archived",
                    "role": "",
                    "turn_id": "",
                    "text": "",
                    "status": "complete",
                    "metadata": {},
                }
            ]
        return [
            user,
            {
                "sequence": 2,
                "event_id": "event-agent",
                "event_type": "agent.message",
                "role": "assistant",
                "turn_id": "turn-gallery",
                "text": "The durable session remains available.",
                "status": "complete",
                "metadata": {},
            },
        ]

    def _turns(self) -> list[dict[str, Any]]:
        if self.fixture in {"empty", "new-session"}:
            return []
        status = "complete"
        if self.fixture in {
            "approval",
            "guarded",
            "reconciliation",
            "streaming",
        }:
            status = "running"
        recovery: dict[str, Any] = {}
        if self.fixture == "reconciliation":
            recovery = {
                "reconciliation_id": "reconciliation-gallery",
                "status": "pending",
            }
        return [
            {
                "turn_id": "turn-gallery",
                "turn_ids": ["turn-gallery"],
                "command_id": "command-gallery",
                "turn_ref": {
                    "step_id": "presentation",
                    "agent_role": "implementer",
                },
                "request": "Implement the durable presentation contract.",
                "status": status,
                "started_at": "2026-07-30T12:00:00+00:00",
                "completed_at": "2026-07-30T12:00:04+00:00",
                "first_sequence": 1,
                "last_sequence": 2,
                "attempts": [
                    {
                        "turn_id": "turn-gallery",
                        "attempt_id": "attempt-gallery",
                        "provider": "codex",
                        "model": "default",
                        "effort": "high",
                        "status": status,
                        "started_at": "2026-07-30T12:00:00+00:00",
                        "ended_at": "2026-07-30T12:00:04+00:00",
                    }
                ],
                "result": {"status": status, "provider": "codex"},
                "safety": {
                    "state": status,
                    "consumption": {
                        "total_tokens": 1842,
                        "tool_calls": 5,
                    },
                },
                "checkpoint_id": "checkpoint-gallery",
                "evidence": [],
                "reconciliation": recovery,
                "activity": self._events(),
            }
        ]

    def _reconciliations(self) -> list[dict[str, Any]]:
        if self.fixture != "reconciliation":
            return []
        return [
            {
                "reconciliation_id": "reconciliation-gallery",
                "command_id": "command-gallery",
                "pre_dispatch_checkpoint_id": "checkpoint-gallery",
                "current_workspace_summary": "one changed file",
                "current_workspace_digest": "digest-gallery",
            }
        ]

    def _safety(self) -> dict[str, Any]:
        return {
            "session": {"profile": "interactive"},
            "envelopes": [
                {
                    "state": "running",
                    "guard_reason": "",
                    "recovery_stage": 0,
                    "limits": {
                        "max_total_tokens": 300000,
                        "max_seconds": 3600,
                        "max_tool_calls": 256,
                    },
                    "consumption": {
                        "total_tokens": 1842,
                        "input_tokens": 1600,
                        "cached_input_tokens": 200,
                        "context_tokens": 1200,
                        "elapsed_seconds": 4.2,
                        "tool_calls": 5,
                        "output_tokens": 242,
                        "exact_tokens": True,
                    },
                }
            ],
            "incidents": [],
        }


def render_gallery(output: Path) -> dict[str, object]:
    """Render every normalized UI state without external processes."""

    return asyncio.run(_render_gallery(output))


async def _render_gallery(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for fixture in FIXTURES:
        for mode in MODES:
            for theme in THEMES:
                rendered = await _render_mode_theme(
                    output,
                    fixture,
                    mode,
                    theme,
                )
                files.extend(rendered)
    manifest = {
        "schema": "p13i/agent-harness/ui-gallery/v2",
        "renderer": "textual-widget-tree",
        "fixtures": list(FIXTURES),
        "modes": list(MODES),
        "themes": list(THEMES),
        "breakpoints": [
            {"width": width, "height": height} for width, height in BREAKPOINTS
        ],
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


async def _render_mode_theme(
    output: Path,
    fixture: str,
    mode: str,
    theme: str,
) -> list[str]:
    client = GalleryClient(fixture, theme)
    app = HarnessApp(
        client,  # type: ignore[arg-type]
        Path("/workspace/agent-harness"),
        session_id="gallery-session",
    )
    app.animation_level = "none"
    files: list[str] = []
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        if app._poll_timer is not None:
            app._poll_timer.stop()
        app._set_workspace_mode(mode)
        if fixture == "disconnected":
            app._set_connection_state("reconnecting")
        await pilot.pause()
        for width, height in BREAKPOINTS:
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            _validate_layout(app)
            content = _self_contained_svg(app.export_screenshot())
            if "secret" in content.casefold():
                raise ValueError("gallery contains secret-bearing text")
            name = (
                fixture
                + "-"
                + mode
                + "-"
                + theme
                + "-"
                + str(width)
                + "x"
                + str(height)
                + ".svg"
            )
            (output / name).write_text(content, encoding="utf-8")
            files.append(name)
    return files


def render_svg(
    fixture: str,
    theme: str,
    columns: int,
    rows: int,
    *,
    mode: str = "focus",
) -> str:
    """Render one actual Textual screenshot for focused tests."""

    if fixture not in FIXTURES:
        raise ValueError("unknown gallery fixture")
    if theme not in THEMES:
        raise ValueError("unknown gallery theme")
    if mode not in MODES:
        raise ValueError("unknown gallery mode")
    if (columns, rows) not in BREAKPOINTS:
        raise ValueError("unknown gallery breakpoint")
    return asyncio.run(_render_one(fixture, mode, theme, columns, rows))


async def _render_one(
    fixture: str,
    mode: str,
    theme: str,
    columns: int,
    rows: int,
) -> str:
    client = GalleryClient(fixture, theme)
    app = HarnessApp(
        client,  # type: ignore[arg-type]
        Path("/workspace/agent-harness"),
        session_id="gallery-session",
    )
    app.animation_level = "none"
    async with app.run_test(size=(columns, rows)) as pilot:
        await pilot.pause()
        if app._poll_timer is not None:
            app._poll_timer.stop()
        app._set_workspace_mode(mode)
        if fixture == "disconnected":
            app._set_connection_state("reconnecting")
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        _validate_layout(app)
        return _self_contained_svg(app.export_screenshot())


def _validate_layout(app: HarnessApp) -> None:
    composer = app.query_one("#composer-shell")
    body = app.query_one("#body")
    topbar = app.query_one("#topbar")
    notification = app.query_one("#notification-shell", Horizontal)
    if topbar.region.bottom > body.region.y:
        raise ValueError("topbar overlaps the workspace body")
    if composer.region.bottom > app.screen.region.bottom:
        raise ValueError("composer is clipped")
    if composer.region.width < 20 or composer.region.height < 3:
        raise ValueError("composer lacks usable geometry")
    if notification.display and notification.region.bottom > composer.region.y:
        raise ValueError("notification overlaps the composer")
    if app.focused is None:
        raise ValueError("gallery lacks keyboard focus")
    brand = app.query_one("#brand", Static)
    if "P13I" not in str(brand.render()):
        raise ValueError("gallery lacks product identity")


def _self_contained_svg(content: str) -> str:
    value = re.sub(
        r"terminal-\d+",
        "terminal-gallery",
        content,
    )
    value = re.sub(
        r"\s*@font-face\s*\{.*?\}\s*",
        "\n",
        value,
        flags=re.DOTALL,
    )
    value = _canonicalize_rich_classes(value)
    value = re.sub(
        r"(Generated with Rich)\s+https?://[^ <]+",
        r"\1",
        value,
    )
    resource_text = value.replace(
        'xmlns="http://www.w3.org/2000/svg"',
        "",
    )
    resources = sorted(
        set(
            re.findall(
                r"https?://[^\s\"'<>)}]+",
                resource_text,
                re.IGNORECASE,
            )
        )
    )
    if resources:
        raise ValueError(
            "gallery contains an outbound resource: " + ", ".join(resources[:3])
        )
    return value


def _canonicalize_rich_classes(content: str) -> str:
    pattern = re.compile(
        r"\s*\.(terminal-gallery-r\d+)\s*\{([^{}]*)\}",
    )
    matches = list(pattern.finditer(content))
    if not matches:
        return content
    mappings: dict[str, str] = {}
    declarations_by_name: dict[str, str] = {}
    for match in matches:
        old_name = match.group(1)
        declarations = match.group(2).strip()
        digest = hashlib.sha256(declarations.encode("utf-8")).hexdigest()[:16]
        new_name = "terminal-gallery-s-" + digest
        mappings[old_name] = new_name
        declarations_by_name[new_name] = declarations
    value = pattern.sub("", content)
    value = re.sub(
        r"terminal-gallery-r\d+",
        lambda match: mappings[match.group(0)],
        value,
    )
    used_names = set(re.findall(r"terminal-gallery-s-[0-9a-f]{16}", value))
    rules = {
        "." + name + " { " + declarations_by_name[name] + " }" for name in used_names
    }
    canonical_rules = "\n".join(sorted(rules))
    return value.replace("<style>", "<style>\n" + canonical_rules, 1)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    values = parser.parse_args(arguments)
    manifest = render_gallery(values.output)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
