"""Behavioral coverage for Textual interaction boundaries."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App
from textual.app import ComposeResult
from textual.app import ScreenStackError
from textual.widgets import Button
from textual.widgets import Input
from textual.widgets import ListView
from textual.widgets import Static
from textual.widgets import Tab
from textual.widgets import Tabs

from agent_harness import tui as tui_module
from agent_harness.errors import HarnessError
from agent_harness.notifications import Notification
from agent_harness.notifications import NotificationAction
from agent_harness.notifications import NotificationPersistence
from agent_harness.notifications import NotificationSeverity
from agent_harness.notifications import NotificationState
from agent_harness.tui import ApprovalScreen
from agent_harness.tui import CommandPaletteScreen
from agent_harness.tui import ConversationLog
from agent_harness.tui import HarnessApp
from agent_harness.tui import SidebarResizeHandle
from agent_harness.tui import TranscriptBlockView
from agent_harness.tui import TranscriptView
from agent_harness.tui_presenter import TranscriptBlock
from agent_harness.tui_presenter import TranscriptBlockKind
from agent_harness.tui_presenter import TranscriptBlockStatus
from agent_harness.tui_presenter import TranscriptMutation
from agent_harness.tui_presenter import TranscriptMutationKind
from agent_harness.tui_presenter import TranscriptState
from agent_harness.tui_widgets import MultilineComposer


class BoundaryClient:
    """Deterministic control-plane double for mounted TUI behavior."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, object, str]] = []
        self.fail_paths: set[str] = set()
        self.sessions_value: object = [self.session()]
        self.events_value: object = []
        self.turns_value: object = []
        self.reconciliations_value: object = []

    @staticmethod
    def session(session_id: str = "session-1") -> dict[str, object]:
        return {
            "session_id": session_id,
            "name": "Boundary chat",
            "lifecycle": "running",
            "attention": "idle",
            "permission_mode": "approval",
            "active_provider": "codex",
            "model": "default",
            "worktree": "/workspace",
        }

    async def request(
        self,
        method: str,
        path: str,
        *,
        payload: object = None,
        idempotency_key: str = "",
    ) -> dict[str, object]:
        self.requests.append((method, path, payload, idempotency_key))
        if path in self.fail_paths:
            raise HarnessError("E_TEST", "bounded failure")
        if path == "/v1/sessions?archived=1":
            return {"sessions": self.sessions_value}
        if path == "/v1/sessions":
            return {"session": self.session("session-new")}
        if path == "/v1/sync":
            return {
                "state_root": "/workspace/chats",
                "sync": {"state": "synced"},
            }
        if path.startswith("/v1/providers"):
            return {"providers": {}}
        if path.endswith("/ui-state"):
            return {"ui_state": {}}
        if path.endswith("/events?after=0"):
            return {"events": self.events_value}
        if "/events?after=" in path:
            return {"events": self.events_value}
        if path.endswith("/turns?limit=200"):
            return {"turns": self.turns_value}
        if path.endswith("/reconciliations"):
            return {"reconciliations": self.reconciliations_value}
        if path.endswith("/diff?limit=240"):
            return {"diff": {"changed_files": ["one.py"]}}
        if method == "PATCH":
            return {"session": self.session(path.split("/")[3])}
        if path.startswith("/v1/sessions/"):
            return {
                "session": self.session(path.split("/")[3]),
                "goal": None,
                "approvals": [],
                "safety": {
                    "session": {"profile": "interactive"},
                    "envelopes": [],
                    "incidents": [],
                },
            }
        return {}


class WidgetBoundaryApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConversationLog(id="conversation")
        yield TranscriptView(id="transcript")
        yield SidebarResizeHandle(id="handle")


def _block(block_id: str = "block-1") -> TranscriptBlock:
    return TranscriptBlock(
        block_id=block_id,
        kind=TranscriptBlockKind.ASSISTANT,
        status=TranscriptBlockStatus.COMPLETE,
        title="Assistant",
        content="Done",
    )


def test_transcript_widget_boundaries() -> None:
    async def scenario() -> None:
        app = WidgetBoundaryApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            conversation = app.query_one(
                "#conversation",
                ConversationLog,
            )
            assert conversation.render_line(0)
            conversation.write("line")
            assert conversation.render_line(23)

            transcript = app.query_one("#transcript", TranscriptView)
            block = _block()
            state = TranscriptState(blocks=(block,))
            await transcript.reset(state, "Welcome")
            await transcript.apply(
                state,
                (
                    TranscriptMutation(
                        TranscriptMutationKind.IGNORE,
                        block.block_id,
                    ),
                    TranscriptMutation(
                        TranscriptMutationKind.REPLACE,
                        "missing",
                    ),
                ),
            )
            transcript.refresh_block(_block("missing"), expanded=False)
            view = transcript.query_one(TranscriptBlockView)
            transcript.refresh_block(block, expanded=True)
            view.on_click()
            assert view.block_id == block.block_id

            handle = app.query_one("#handle", SidebarResizeHandle)
            pointer = SimpleNamespace(
                screen_x=30,
                prevent_default=lambda: None,
                stop=lambda: None,
            )
            handle.on_mouse_move(pointer)
            handle.on_mouse_up(pointer)
            handle.capture_mouse()
            handle.on_mouse_move(pointer)
            handle.release_mouse()

    asyncio.run(scenario())


def test_modal_screen_boundaries() -> None:
    async def scenario() -> None:
        app = HarnessApp(
            BoundaryClient(),  # type: ignore[arg-type]
            Path("/workspace"),
            session_id="session-1",
        )
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            if app._poll_timer is not None:
                app._poll_timer.stop()

            approval_results: list[str] = []
            app.push_screen(
                ApprovalScreen(
                    {
                        "prompt": "Proceed?",
                        "choices": [
                            "invalid",
                            {"id": "approve", "label": "Approve"},
                        ],
                    }
                ),
                approval_results.append,
            )
            await pilot.pause()
            approval = app.screen
            assert isinstance(approval, ApprovalScreen)
            approve = approval.query_one("#approval-choice-1", Button)
            approval.on_button_pressed(Button.Pressed(approve))
            await pilot.pause()
            assert approval_results == ["approve"]

            app.push_screen(
                ApprovalScreen(
                    {"prompt": "Proceed?", "choices": "invalid"}
                ),
                approval_results.append,
            )
            await pilot.pause()
            later_screen = app.screen
            assert isinstance(later_screen, ApprovalScreen)
            later = later_screen.query_one("#approval-later", Button)
            later_screen.on_button_pressed(Button.Pressed(later))
            await pilot.pause()
            assert approval_results[-1] == ""

            palette_results: list[str] = []
            app.push_screen(
                CommandPaletteScreen("help"),
                palette_results.append,
            )
            await pilot.pause()
            palette = app.screen
            assert isinstance(palette, CommandPaletteScreen)
            foreign = Input(id="foreign")
            await palette.on_input_changed(
                SimpleNamespace(input=foreign, value="help")
            )
            query = palette.query_one("#command-query", Input)
            await palette.on_input_changed(
                SimpleNamespace(input=query, value="no-match-value")
            )
            results = palette.query_one("#command-results", ListView)
            await palette.on_list_view_selected(
                SimpleNamespace(
                    list_view=SimpleNamespace(id="foreign"),
                )
            )
            results.index = None
            await palette.on_list_view_selected(
                SimpleNamespace(list_view=results)
            )
            await palette.on_list_view_selected(
                SimpleNamespace(
                    list_view=SimpleNamespace(
                        id="command-results",
                        index=999,
                    )
                )
            )
            results.index = 0
            await palette.on_list_view_selected(
                SimpleNamespace(list_view=results)
            )
            await pilot.pause()
            assert palette_results

    asyncio.run(scenario())


def test_application_event_and_action_boundaries(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = BoundaryClient()
        app = HarnessApp(
            client,  # type: ignore[arg-type]
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            if app._poll_timer is not None:
                app._poll_timer.stop()
            calls: list[tuple[str, object]] = []

            async def no_argument() -> None:
                calls.append(("async", "called"))

            async def command(
                value: str,
                payload: dict[str, Any] | None = None,
            ) -> None:
                calls.append(("command", (value, payload)))

            async def notification(value: str) -> None:
                calls.append(("notification", value))

            async def select_turn(value: int | None) -> None:
                calls.append(("turn", value))

            async def open_session(value: str) -> None:
                calls.append(("open", value))

            async def slash(value: str) -> None:
                calls.append(("slash", value))

            async def save(*, force: bool = False) -> None:
                calls.append(("save", force))

            def toggle_mode() -> None:
                calls.append(("mode", "toggle"))

            app._new_session = no_argument  # type: ignore[method-assign]
            app._command = command  # type: ignore[method-assign]
            app._handle_notification_action = (  # type: ignore[method-assign]
                notification
            )
            app._select_turn = select_turn  # type: ignore[method-assign]
            app._open_session = open_session  # type: ignore[method-assign]
            app._slash = slash  # type: ignore[method-assign]
            app._save_ui_state = save  # type: ignore[method-assign]
            app.action_toggle_mode = toggle_mode  # type: ignore[method-assign]

            composer = app.query_one("#composer", MultilineComposer)
            composer.action_submit = (  # type: ignore[method-assign]
                lambda: calls.append(("submit", "called"))
            )
            for button_id in (
                "new-session",
                "mode-toggle",
                "send-message",
                "stop-work",
                "notification-primary",
                "unknown",
            ):
                await app.on_button_pressed(
                    Button.Pressed(Button(button_id, id=button_id))
                )

            await app.on_multiline_composer_submitted(
                SimpleNamespace(text=" ", composer=composer)
            )
            await app.on_multiline_composer_submitted(
                SimpleNamespace(text="/unknown", composer=composer)
            )
            composer.text = "/help"
            await app.on_multiline_composer_submitted(
                SimpleNamespace(text="/help", composer=composer)
            )
            app._pending_request_id = "request-1"
            await app.on_multiline_composer_submitted(
                SimpleNamespace(text="blocked", composer=composer)
            )

            await app.on_input_changed(
                SimpleNamespace(
                    input=Input(id="foreign"),
                    value="value",
                )
            )
            await app.on_tabs_tab_activated(
                SimpleNamespace(
                    tabs=SimpleNamespace(id="foreign"),
                    tab=SimpleNamespace(id="summary"),
                )
            )
            await app.on_tabs_tab_activated(
                SimpleNamespace(
                    tabs=SimpleNamespace(id="control-detail-tabs"),
                    tab=SimpleNamespace(id="activity"),
                )
            )

            turn_list = app.query_one("#turn-list", ListView)
            await app.on_list_view_selected(
                SimpleNamespace(list_view=turn_list)
            )
            await app.on_list_view_selected(
                SimpleNamespace(
                    list_view=SimpleNamespace(id="foreign"),
                )
            )
            session_list = app.query_one("#session-list", ListView)
            session_list.index = None
            await app.on_list_view_selected(
                SimpleNamespace(list_view=session_list)
            )
            session_list.index = 999
            await app.on_list_view_selected(
                SimpleNamespace(list_view=session_list)
            )
            await app.on_list_view_selected(
                SimpleNamespace(
                    list_view=SimpleNamespace(
                        id="session-list",
                        index=999,
                    )
                )
            )
            app._visible_sessions = [{"session_id": "session-2"}]
            session_list.index = 0
            await app.on_list_view_selected(
                SimpleNamespace(list_view=session_list)
            )

            await app.action_interrupt()
            await app.action_pause()
            app._load_sessions = no_argument  # type: ignore[method-assign]
            app._poll = no_argument  # type: ignore[method-assign]
            await app.action_refresh()
            await app.action_new_session()
            app._checkpoint = no_argument  # type: ignore[method-assign]
            await app.action_checkpoint()
            app.action_toggle_sessions()
            app.action_toggle_inspector()
            app.action_sidebar_narrower()
            app.action_sidebar_wider()
            app.action_show_help()
            await pilot.pause()
            app.pop_screen()

            assert ("submit", "called") in calls
            assert ("command", ("interrupt", None)) in calls
            assert ("open", "session-2") in calls

    asyncio.run(scenario())


def test_application_state_and_render_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        client = BoundaryClient()
        app = HarnessApp(
            client,  # type: ignore[arg-type]
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            if app._poll_timer is not None:
                app._poll_timer.stop()

            app._workspace_mode = "control"
            app.action_toggle_mode()
            await pilot.pause()
            assert app._workspace_mode == "focus"
            app._palette_selected("")
            app._palette_selected("/help ")

            await app._new_session()
            await app._open_session("")

            await app._apply_session_snapshot(
                "session-1",
                {
                    "theme": "invalid",
                    "sidebar_width": "44",
                    "session_filter": "invalid",
                    "inspector_tab": "invalid",
                    "workspace_mode": "invalid",
                    "control_detail_tab": "invalid",
                    "last_notification_sequence": "1",
                },
                {
                    "session": BoundaryClient.session(),
                    "goal": {"objective": "Ship"},
                    "safety": {},
                    "approvals": [],
                },
                {"events": "invalid"},
                {"turns": "invalid"},
                {"reconciliations": []},
            )
            assert app._theme_preference == "system"
            assert app._sidebar_width == 44

            cached = SessionViewCacheEntry(
                session_id="session-1",
                focus_id="missing-focus",
                payload={
                    "session_state": {
                        "session": BoundaryClient.session(),
                        "safety": {},
                        "approvals": [],
                    },
                    "events_state": {"events": []},
                    "turns_state": {"turns": []},
                    "recovery_state": {"reconciliations": []},
                    "show_events": True,
                },
            )
            assert app._cached_ui_state(cached)["events"] == "on"
            await app._apply_cached_session(cached)
            app._switches.remember(cached)
            await app._open_session("session-1")
            assert app.focused is app.query_one(
                "#composer",
                MultilineComposer,
            )

            app._session = {
                "session_id": "session-1",
                "name": "Defaults",
                "attention": "idle",
                "lifecycle": "running",
            }
            app._render_session_header()
            assert "auto / default" in str(
                app.query_one("#session-meta", Static).render()
            )
            for attention in (
                "working",
                "needs-input",
                "failed",
            ):
                app._session["attention"] = attention
                app._render_connection_status()
            app._connection_status = "disconnected"
            app._render_connection_status()
            app._connection_status = "send-unacknowledged"
            app._render_connection_status()
            app.screen.add_class("narrow")
            app._switching_session_id = "session-2"
            app._render_connection_status()
            app.screen.remove_class("narrow")
            app._switching_session_id = ""

            save_calls: list[bool] = []

            async def save(*, force: bool = False) -> None:
                save_calls.append(force)

            app._save_ui_state = save  # type: ignore[method-assign]
            app._dismiss_notification("missing")
            await pilot.pause()

            activity = Notification(
                key="activity",
                title="Activity",
                detail="Detail",
                severity=NotificationSeverity.WARNING,
                persistence=NotificationPersistence.ACTIVITY,
                source_sequence=7,
            )
            app._notification_state = NotificationState(
                notifications=(activity,)
            )
            app._dismiss_notification("activity")
            await pilot.pause()
            assert app._notification_ack_sequence == 7

            calls: list[object] = []

            def present(*, force: bool = False) -> None:
                calls.append(("present", force))

            def dismiss(key: str) -> None:
                calls.append(("dismiss", key))

            def mode(value: str) -> None:
                calls.append(("mode", value))

            def render() -> None:
                calls.append("render")

            async def command(value: str) -> None:
                calls.append(("command", value))

            async def poll() -> None:
                calls.append("poll")

            app._present_approval = present  # type: ignore[method-assign]
            app._dismiss_notification = dismiss  # type: ignore[method-assign]
            app._set_workspace_mode = mode  # type: ignore[method-assign]
            app._render_selected_turn = render  # type: ignore[method-assign]
            app._render_inspector = render  # type: ignore[method-assign]
            app._apply_responsive_layout = (  # type: ignore[method-assign]
                lambda unused: calls.append("layout")
            )
            app._command = command  # type: ignore[method-assign]
            app._poll = poll  # type: ignore[method-assign]

            action = Notification(
                key="action",
                title="Action",
                detail="",
                severity=NotificationSeverity.ACTION,
                persistence=NotificationPersistence.ACTION,
                actions=(
                    NotificationAction("review-approval", "Review"),
                ),
            )
            app._notification_state = NotificationState(
                notifications=(action,)
            )
            for action_id in (
                "review-approval",
                "defer-approval",
                "review-recovery",
                "stop-session",
                "review-usage",
                "retry-connection",
                "dismiss-notification",
            ):
                app._notification_actions = {"button": action_id}
                await app._handle_notification_action("button")

            transient = Notification(
                key="transient",
                title="Transient",
                detail="",
                severity=NotificationSeverity.INFO,
                persistence=NotificationPersistence.TRANSIENT,
            )
            app._notification_state = NotificationState(
                notifications=(transient,)
            )
            app._notification_actions = {}
            await app._handle_notification_action("button")
            app._notification_state = NotificationState(
                notifications=(activity,)
            )
            await app._handle_notification_action("button")
            assert ("present", True) in calls
            assert ("command", "stop") in calls

            app._turns = [
                {"turn_id": "first"},
                {"turn_id": "second"},
            ]
            app._selected_turn_id = "second"
            app._render_selected_turn = (  # type: ignore[method-assign]
                HarnessApp._render_selected_turn.__get__(app)
            )
            app._render_selected_turn()

            await app._checkpoint()
            app.session_id = ""
            await HarnessApp._save_ui_state(app)
            app.session_id = "session-1"
            app._show_events = True
            app._saved_show_events = False
            await HarnessApp._save_ui_state(app, force=True)
            await HarnessApp._save_ui_state(app)

            client.fail_paths.add(
                "/v1/providers?workspace=" + str(tmp_path)
            )
            await app._load_providers()
            client.fail_paths.add("/v1/sync")
            await app._load_sync()
            client.fail_paths.clear()

            app._render_inspector = (  # type: ignore[method-assign]
                HarnessApp._render_inspector.__get__(app)
            )
            app._inspector_tab = "context"
            app._sync = {"error": "sync failed"}
            app._goal = {"kind": "finite", "status": "active"}
            app._safety = {"envelopes": "invalid"}
            app._providers = {}
            app._render_inspector()
            app._safety = {
                "session": {"profile": "interactive"},
                "envelopes": [
                    {
                        "state": "guarded",
                        "guard_reason": "limit",
                        "recovery_stage": 2,
                        "consumption": {
                            "exact_tokens": True,
                            "input_tokens": 10,
                            "cached_input_tokens": 4,
                        },
                        "limits": {},
                    }
                ],
            }
            app._providers = {
                "codex": {
                    "ready": False,
                    "usage": {"binding_percent": 90},
                    "usage_refreshing": True,
                }
            }
            app._render_inspector()

            app._theme_preference = "invalid"
            await app._sync_system_theme()
            app._set_workspace_mode = (  # type: ignore[method-assign]
                HarnessApp._set_workspace_mode.__get__(app)
            )
            app._set_workspace_mode("invalid")
            app._present_approval = (  # type: ignore[method-assign]
                HarnessApp._present_approval.__get__(app)
            )
            app._approvals = ()
            app._present_approval(force=True)
            app._approvals = (
                {"approval_id": "", "prompt": "No identifier"},
            )
            app._present_approval(force=True)
            app._approvals = (
                {"approval_id": "same", "prompt": "Already shown"},
            )
            app._last_approval_id = "same"
            app._present_approval(force=True)
            app._approval_decision("")

            app._last_approval_id = "approval-1"
            app._approval_decision("approve")
            await pilot.pause()
            app._poll = poll  # type: ignore[method-assign]
            await app._resolve_approval("approval-1", "approve")

            await app._native("invalid")
            app.suspend = lambda: nullcontext()  # type: ignore[method-assign]
            monkeypatch.setattr(
                tui_module.subprocess,
                "run",
                lambda *args, **kwargs: None,
            )
            await app._native("codex")

            app._transcript_notices = ["[dim]Notice[/dim]"]
            await app._rebuild_transcript()
            app._render_transcript()
            await pilot.pause()
            app._write_help()
            await pilot.pause()
            app.pop_screen()
            assert app._bounded_sidebar_width(1) >= 24
            app.toggle_transcript_block("missing")
            block = _block("toggle")
            app._transcript_state = TranscriptState(blocks=(block,))
            transcript = app.query_one("#transcript", TranscriptView)
            await transcript.reset(app._transcript_state, "Welcome")
            transcript.query_one(TranscriptBlockView).on_click()
            app.toggle_transcript_block("toggle")
            await pilot.pause()
            app._transcript_state = TranscriptState(
                new_activity_count=2,
            )
            app._render_activity_marker()
            assert "2 new" in str(
                app.query_one("#new-activity", Static).render()
            )

    asyncio.run(scenario())

    assert _notification_glyph(NotificationSeverity.DANGER)
    assert "Loading checkpoint" in _turn_detail(
        {"checkpoint_id": "checkpoint-1"},
        tab="changes",
    )
    assert _render_transcript_events(
        [
            {
                "event_type": "agent.message.delta",
                "turn_id": "turn-1",
                "text": "one",
            },
            {
                "event_type": "agent.message.delta",
                "turn_id": "turn-1",
                "text": "two",
            },
        ],
        show_events=False,
    )


def test_application_failure_and_recovery_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        client = BoundaryClient()
        app = HarnessApp(
            client,  # type: ignore[arg-type]
            tmp_path,
            session_id="session-1",
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            if app._poll_timer is not None:
                app._poll_timer.stop()
            composer = app.query_one("#composer", MultilineComposer)

            client.fail_paths.add("/v1/sessions/missing")
            await app._open_session("missing")
            client.fail_paths.clear()
            assert app._switching_session_id == ""

            with monkeypatch.context() as context:
                context.setattr(
                    type(app._switches),
                    "is_current",
                    lambda *unused: False,
                )
                await app._open_session("stale")

            async def failed_save(*, force: bool = False) -> None:
                del force
                raise HarnessError("E_TEST", "save failed")

            app._pending_request_id = ""
            app._save_ui_state = failed_save  # type: ignore[method-assign]
            await app.on_multiline_composer_submitted(
                SimpleNamespace(text="message", composer=composer)
            )
            assert app._connection_status == "reconnecting"

            app.session_id = ""
            await app._command("stop")
            app.session_id = "session-1"
            await HarnessApp._command(app, "stop")

            app._pending_request_id = ""
            await app._submit_pending_message("message")
            app._pending_request_id = "request-1"
            app._recovering_send = True
            await app._recover_pending_message()
            app._recovering_send = False
            app._pending_retry_count = 5
            await app._recover_pending_message()
            app._pending_retry_count = 0
            composer.text = ""

            async def save(*, force: bool = False) -> None:
                del force

            app._save_ui_state = save  # type: ignore[method-assign]
            await app._recover_pending_message()

            app._turns = []
            await app._select_turn(None)
            await app._select_turn(-1)
            await app._select_turn(1)

            app._turns = [
                {
                    "turn_id": "turn-1",
                    "checkpoint_id": "checkpoint-1",
                }
            ]
            await app._select_turn(0)
            diff_path = (
                "/v1/sessions/session-1/checkpoints/"
                "checkpoint-1/diff?limit=240"
            )
            client.fail_paths.add(diff_path)
            await app._select_turn(0)
            client.fail_paths.clear()

            original_request = client.request

            async def stale_request(
                method: str,
                path: str,
                *,
                payload: object = None,
                idempotency_key: str = "",
            ) -> dict[str, object]:
                result = await original_request(
                    method,
                    path,
                    payload=payload,
                    idempotency_key=idempotency_key,
                )
                if path == diff_path:
                    app._selected_turn_id = "other"
                return result

            client.request = stale_request  # type: ignore[method-assign]
            await app._select_turn(0)
            client.request = original_request  # type: ignore[method-assign]

            client.sessions_value = "invalid"
            await app._load_sessions()
            await app._render_session_list()
            await app._render_session_list()

            app.session_id = "session-1"
            client.fail_paths.add("/v1/sessions/session-1")
            await app._poll()
            client.fail_paths.clear()

            with monkeypatch.context() as context:
                context.setattr(
                    type(app._switches),
                    "is_current",
                    lambda *unused: False,
                )
                await app._poll()

            app._switches.begin("session-1")
            client.events_value = "invalid"
            await app._poll()

            current_checks = {"count": 0}

            def first_current(
                unused_self: object,
                unused_generation: int,
                unused_session: str,
            ) -> bool:
                del unused_self
                del unused_generation
                del unused_session
                current_checks["count"] += 1
                return current_checks["count"] == 1

            client.events_value = []
            app._workspace_mode = "control"
            with monkeypatch.context() as context:
                context.setattr(
                    type(app._switches),
                    "is_current",
                    first_current,
                )
                await app._poll()

            screen_checks = {"count": 0}

            def first_screen() -> bool:
                screen_checks["count"] += 1
                return screen_checks["count"] == 1

            app._screen_is_running = (  # type: ignore[method-assign]
                first_screen
            )
            await app._poll()
            app._screen_is_running = (  # type: ignore[method-assign]
                HarnessApp._screen_is_running.__get__(app)
            )

            turns_path = "/v1/sessions/session-1/turns?limit=200"
            client.fail_paths.add(turns_path)
            await app._poll()
            client.fail_paths.clear()

            app._inspector_tab = "recovery"
            recovery_path = (
                "/v1/sessions/session-1/reconciliations"
            )
            client.fail_paths.add(recovery_path)
            client.events_value = [
                "invalid",
                {
                    "sequence": 2,
                    "event_id": "event-2",
                    "event_type": "user.message",
                    "turn_id": "turn-2",
                    "text": "Continue",
                    "status": "complete",
                    "metadata": {},
                },
            ]
            client.turns_value = [{"turn_id": "turn-2"}]
            app._providers = {
                "codex": {"usage_refreshing": True}
            }
            app._provider_poll = 3
            app._sync_poll = 19
            app._theme_poll = 19
            await app._poll()
            client.fail_paths.clear()

            await app._synchronize_active_session({})
            app._sessions = [
                BoundaryClient.session("different"),
            ]
            await app._synchronize_active_session(
                BoundaryClient.session("session-new")
            )

            app.session_id = ""
            await app._poll()

    asyncio.run(scenario())


def test_sidebar_handle_non_drag_boundaries() -> None:
    handle = SidebarResizeHandle()
    event = SimpleNamespace(button=2)
    handle.on_mouse_down(event)

    class BrokenScreen:
        is_running = True

        @property
        def screen(self) -> object:
            raise ScreenStackError("no screen")

    assert not HarnessApp._screen_is_running(BrokenScreen())


def test_initial_session_and_run_tui_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        app = HarnessApp(
            BoundaryClient(),  # type: ignore[arg-type]
            tmp_path,
        )
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            if app._poll_timer is not None:
                app._poll_timer.stop()
            assert app.session_id == "session-new"

    asyncio.run(scenario())

    observed: dict[str, object] = {}

    class Application:
        def __init__(
            self,
            client: object,
            workspace: Path,
            *,
            session_id: str,
            permission_mode: str,
        ) -> None:
            observed["values"] = (
                client,
                workspace,
                session_id,
                permission_mode,
            )

        def run(self) -> None:
            observed["ran"] = True

    client = object()
    monkeypatch.setattr(tui_module, "HarnessApp", Application)
    run_tui(
        client,  # type: ignore[arg-type]
        tmp_path,
        session_id="session-1",
        permission_mode="full",
    )
    assert observed["values"] == (
        client,
        tmp_path,
        "session-1",
        "full",
    )
    assert observed["ran"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--import-mode=importlib"]))
from agent_harness.tui import _notification_glyph
from agent_harness.tui import _render_transcript_events
from agent_harness.tui import _turn_detail
from agent_harness.tui import run_tui
from agent_harness.presentation import SessionViewCacheEntry
