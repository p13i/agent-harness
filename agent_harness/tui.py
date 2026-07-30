"""Textual workspace for durable agent sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys
from typing import Any

from rich.markup import escape
from textual import events
from textual.app import App
from textual.app import ComposeResult
from textual.app import ScreenStackError
from textual.binding import Binding
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.timer import Timer
from textual.widgets import Button
from textual.widgets import Footer
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import ListItem
from textual.widgets import ListView
from textual.widgets import RichLog
from textual.widgets import Static

from agent_harness.client import HarnessClient
from agent_harness.ids import new_uuid


SIDEBAR_DEFAULT_WIDTH = 31
SIDEBAR_MIN_WIDTH = 24
SIDEBAR_MAIN_RESERVE = 48
SIDEBAR_WIDTH_STEP = 4
FOCUSED_IDLE_SESSION_LIMIT = 5


class ConversationLog(RichLog):
    """A transcript that keeps short conversations beside the composer."""

    def render_line(self, y: int) -> Strip:
        viewport_height = self.scrollable_content_region.height
        content_height = len(self.lines)
        top_padding = max(0, viewport_height - content_height)
        if y < top_padding:
            return Strip.blank(
                self.scrollable_content_region.width,
                self.rich_style,
            )
        return super().render_line(y - top_padding)


class SidebarResizeHandle(Static):
    """Mouse-draggable divider for the session sidebar."""

    can_focus = True

    def on_mount(self) -> None:
        self.tooltip = (
            "Drag to resize sessions. "
            "Ctrl+Shift+Left/Right also adjusts the width."
        )

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        self.capture_mouse()
        event.prevent_default()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self.app.mouse_captured is not self:
            return
        application = self.app
        if not isinstance(application, HarnessApp):
            return
        application._resize_sidebar_to(event.screen_x)
        event.prevent_default()
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self.app.mouse_captured is not self:
            return
        application = self.app
        self.release_mouse()
        if isinstance(application, HarnessApp):
            application._finish_sidebar_resize()
        event.prevent_default()
        event.stop()


class ApprovalScreen(ModalScreen[str]):
    CSS = """
    ApprovalScreen {
        align: center middle;
    }
    #approval-dialog {
        width: 72;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    #approval-choices {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(self, approval: dict[str, Any]) -> None:
        super().__init__()
        self.approval = approval
        self._decisions: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        prompt = str(self.approval.get("prompt", "Approval required"))
        choices = self.approval.get("choices", [])
        if not isinstance(choices, list):
            choices = []
        with Vertical(id="approval-dialog"):
            yield Label("Approval required")
            yield Static(prompt, markup=False)
            with Horizontal(id="approval-choices"):
                for index, choice in enumerate(choices):
                    if not isinstance(choice, dict):
                        continue
                    decision = str(choice.get("id", ""))
                    label = str(choice.get("label", decision))
                    button_id = "approval-choice-" + str(index)
                    self._decisions[button_id] = decision
                    yield Button(label, id=button_id)
                yield Button("Later", id="approval-later")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approval-later":
            self.dismiss("")
            return
        decision = self._decisions.get(event.button.id or "", "")
        self.dismiss(decision)


class HarnessApp(App[None]):
    TITLE = "Agent Harness"
    SUB_TITLE = "durable agent workspace"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen {
        layout: vertical;
        background: #0a0e14;
        color: #d8dee9;
    }
    #topbar {
        height: 3;
        padding: 0 1;
        background: #111823;
        border-bottom: solid #27364a;
        align-vertical: middle;
    }
    #brand {
        width: auto;
        color: #7dd3fc;
        text-style: bold;
    }
    #product-promise {
        width: 1fr;
        padding-left: 2;
        color: #7f8da3;
    }
    #connection-state {
        width: auto;
        color: #86efac;
    }
    #body {
        height: 1fr;
    }
    #sidebar {
        width: 31;
        min-width: 24;
        padding: 1;
        background: #0d131c;
    }
    #sidebar-resize-handle {
        width: 1;
        min-width: 1;
        height: 1fr;
        content-align: center middle;
        color: #526176;
        background: #111823;
    }
    #sidebar-resize-handle:hover,
    #sidebar-resize-handle:focus {
        color: #e0f2fe;
        background: #25638a;
    }
    .eyebrow {
        height: 1;
        color: #7f8da3;
        text-style: bold;
    }
    #workspace-summary {
        height: 3;
        margin-bottom: 1;
        color: #c3ccda;
    }
    #session-heading {
        height: 1;
        margin-top: 1;
        color: #7f8da3;
        text-style: bold;
    }
    #new-session {
        width: 100%;
        min-width: 0;
        height: 3;
        margin-bottom: 1;
        border: none;
        background: #16324f;
        color: #bae6fd;
        text-style: bold;
    }
    #session-list {
        height: 1fr;
        background: transparent;
        border: none;
    }
    #session-list ListItem {
        height: 3;
        padding: 0 1;
        color: #aeb9c9;
        background: #111923;
        border-left: tall #27364a;
    }
    #session-list ListItem:hover {
        color: #e5edf8;
        background: #172231;
    }
    #session-list ListItem.-highlight {
        color: #f0f9ff;
        background: #18344d;
        border-left: tall #38bdf8;
    }
    #session-list ListItem.active-session {
        color: #ffffff;
        background: #16405d;
        border-left: tall #7dd3fc;
        text-style: bold;
    }
    #sessions-help {
        height: 2;
        color: #64748b;
    }
    #main {
        width: 1fr;
        min-width: 40;
        background: #0a0e14;
    }
    #inspector {
        width: 36;
        min-width: 30;
        padding: 1;
        background: #0d131c;
        border-left: solid #27364a;
    }
    #inspector-heading {
        height: 1;
        margin-bottom: 1;
        color: #7f8da3;
        text-style: bold;
    }
    #inspector-content {
        height: 1fr;
        color: #b8c2d1;
    }
    #session-bar {
        height: 5;
        padding: 1 2 0 2;
        background: #0f1620;
        border-bottom: solid #27364a;
    }
    #session-title {
        height: 1;
        color: #f1f5f9;
        text-style: bold;
    }
    #session-meta {
        height: 1;
        color: #8fa0b7;
    }
    #resume-token {
        height: 1;
        color: #526176;
    }
    #transcript {
        height: 1fr;
        padding: 1 3;
        background: #0a0e14;
        overflow-x: hidden;
        scrollbar-size: 1 1;
        scrollbar-background: #0a0e14;
        scrollbar-color: #30445f;
        scrollbar-color-hover: #3b82a8;
        scrollbar-color-active: #38bdf8;
    }
    #composer-shell {
        height: 6;
        min-height: 6;
        margin: 0 2 1 2;
        padding: 0 1;
        background: #101822;
        border: round #33465f;
    }
    #composer-shell:focus-within {
        border: round #38bdf8;
    }
    #composer-label {
        height: 1;
        color: #7dd3fc;
        text-style: bold;
    }
    #composer {
        height: 3;
        padding: 0;
        border: none;
        background: transparent;
        color: #e8eef7;
    }
    #composer:focus {
        border: none;
    }
    #composer-help {
        height: 1;
        color: #637187;
    }
    Screen.compact #inspector {
        display: none;
    }
    Screen.compact #product-promise {
        display: none;
    }
    Screen.narrow #sidebar {
        display: none;
    }
    Screen.narrow #sidebar-resize-handle {
        display: none;
    }
    Screen.narrow #transcript {
        padding-left: 1;
        padding-right: 1;
    }
    Screen.narrow #composer-shell {
        margin-left: 1;
        margin-right: 1;
    }
    Screen.light {
        background: #f4f7fb;
        color: #1e293b;
    }
    Screen.light #topbar {
        background: #e2e8f0;
        border-bottom: solid #94a3b8;
    }
    Screen.light #brand {
        color: #075985;
    }
    Screen.light #product-promise {
        color: #475569;
    }
    Screen.light #connection-state {
        color: #166534;
    }
    Screen.light #sidebar,
    Screen.light #inspector {
        background: #f1f5f9;
    }
    Screen.light #sidebar-resize-handle {
        color: #64748b;
        background: #e2e8f0;
    }
    Screen.light #sidebar-resize-handle:hover,
    Screen.light #sidebar-resize-handle:focus {
        color: #ffffff;
        background: #0369a1;
    }
    Screen.light #inspector {
        border-left: solid #94a3b8;
    }
    Screen.light .eyebrow,
    Screen.light #session-heading,
    Screen.light #inspector-heading {
        color: #475569;
    }
    Screen.light #workspace-summary,
    Screen.light #inspector-content {
        color: #1e293b;
    }
    Screen.light #session-list ListItem {
        color: #334155;
        background: #ffffff;
        border-left: tall #94a3b8;
    }
    Screen.light #session-list ListItem:hover {
        color: #0f172a;
        background: #dbeafe;
    }
    Screen.light #session-list ListItem.-highlight {
        color: #082f49;
        background: #bfdbfe;
        border-left: tall #0284c7;
    }
    Screen.light #session-list ListItem.active-session {
        color: #082f49;
        background: #bae6fd;
        border-left: tall #0369a1;
    }
    Screen.light #sessions-help,
    Screen.light #composer-help {
        color: #475569;
    }
    Screen.light #main,
    Screen.light #transcript {
        background: #ffffff;
    }
    Screen.light #session-bar {
        background: #f8fafc;
        border-bottom: solid #94a3b8;
    }
    Screen.light #session-title {
        color: #0f172a;
    }
    Screen.light #session-meta {
        color: #475569;
    }
    Screen.light #resume-token {
        color: #64748b;
    }
    Screen.light #composer-shell {
        background: #ffffff;
        border: round #64748b;
    }
    Screen.light #composer {
        color: #0f172a;
    }
    Screen.light #composer-label {
        color: #075985;
    }
    Screen.light #transcript {
        scrollbar-background: #ffffff;
        scrollbar-color: #94a3b8;
        scrollbar-color-hover: #64748b;
        scrollbar-color-active: #0369a1;
    }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "interrupt", "Interrupt"),
        Binding("ctrl+p", "pause", "Pause"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+n", "new_session", "New"),
        Binding("ctrl+k", "checkpoint", "Checkpoint"),
        Binding("ctrl+b", "toggle_sessions", "Sessions"),
        Binding("ctrl+o", "toggle_inspector", "Inspector"),
        Binding(
            "ctrl+shift+left",
            "sidebar_narrower",
            "Narrow sessions",
            priority=True,
        ),
        Binding(
            "ctrl+shift+right",
            "sidebar_wider",
            "Widen sessions",
            priority=True,
        ),
        Binding("f1", "show_help", "Help"),
    ]

    def __init__(
        self,
        client: HarnessClient,
        workspace: Path,
        *,
        session_id: str = "",
        permission_mode: str = "approval",
    ) -> None:
        super().__init__()
        self.client = client
        self.workspace = workspace
        self.session_id = session_id
        self.permission_mode = permission_mode
        self.sequence = 0
        self._sessions: list[dict[str, Any]] = []
        self._visible_sessions: list[dict[str, Any]] = []
        self._session_list_signature = ""
        self._session: dict[str, Any] = {}
        self._goal: dict[str, Any] | None = None
        self._safety: dict[str, Any] = {}
        self._approvals: tuple[dict[str, Any], ...] = ()
        self._providers: dict[str, Any] = {}
        self._provider_override = ""
        self._model_override = ""
        self._effort_override = ""
        self._transcript_events: list[dict[str, Any]] = []
        self._transcript_notices: list[str] = []
        self._saved_composer = ""
        self._saved_sidebar_width = SIDEBAR_DEFAULT_WIDTH
        self._saved_session_filter = "focused"
        self._saved_show_events = False
        self._last_approval_id = ""
        self._provider_poll = 0
        self._theme_poll = 0
        self._poll_timer: Timer | None = None
        self._theme_preference = "system"
        self._sidebar_requested = True
        self._inspector_requested = True
        self._sidebar_width = SIDEBAR_DEFAULT_WIDTH
        self._session_filter = "focused"
        self._show_events = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("●  P13I AGENT HARNESS", id="brand")
            yield Static(
                "One durable workspace across Claude and Codex",
                id="product-promise",
            )
            yield Static("● connected", id="connection-state")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("WORKSPACE", classes="eyebrow")
                yield Static(
                    self.workspace.name
                    + "\n"
                    + escape(str(self.workspace)),
                    id="workspace-summary",
                )
                yield Button(
                    "＋  New session     Ctrl+N",
                    id="new-session",
                )
                yield Label("SESSIONS", id="session-heading")
                yield ListView(id="session-list")
                yield Static(
                    "↑↓ select · Enter open\n/sessions all for history",
                    id="sessions-help",
                )
            yield SidebarResizeHandle("│", id="sidebar-resize-handle")
            with Vertical(id="main"):
                with Vertical(id="session-bar"):
                    yield Static("Starting…", id="session-title")
                    yield Static(
                        "Preparing durable workspace",
                        id="session-meta",
                    )
                    yield Static("", id="resume-token")
                yield ConversationLog(
                    id="transcript",
                    markup=True,
                    wrap=True,
                    highlight=False,
                    min_width=1,
                    max_lines=5000,
                )
                with Vertical(id="composer-shell"):
                    yield Label("MESSAGE", id="composer-label")
                    yield Input(
                        placeholder=(
                            "Ask, build, debug, or steer the active agent…"
                        ),
                        id="composer",
                    )
                    yield Static(
                        "Enter send  ·  / commands  ·  "
                        "Ctrl+C interrupt  ·  F1 help",
                        id="composer-help",
                    )
            with Vertical(id="inspector"):
                yield Label("SESSION CONTROL", id="inspector-heading")
                yield Static("", id="inspector-content")
        yield Footer()

    async def on_mount(self) -> None:
        self._apply_responsive_layout(self.size.width)
        self._apply_sidebar_width()
        await self._sync_system_theme(force=True)
        await self._load_sessions()
        if self.session_id:
            await self._open_session(self.session_id)
        else:
            await self._new_session()
        self._poll_timer = self.set_interval(0.5, self._poll)
        self.query_one("#composer", Input).focus()

    def on_unmount(self) -> None:
        if self._poll_timer is None:
            return
        self._poll_timer.stop()
        self._poll_timer = None

    def on_resize(self, event: events.Resize) -> None:
        self._apply_responsive_layout(event.size.width)
        self._apply_sidebar_width()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "new-session":
            return
        await self._new_session()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            await self._slash(text)
            return
        self._transcript_notices = []
        await self.client.request(
            "POST",
            "/v1/sessions/" + self.session_id + "/messages",
            payload={
                "text": text,
                "provider": self._provider_override,
                "model": self._model_override,
                "effort": self._effort_override,
            },
            idempotency_key=new_uuid(),
        )
        await self._poll()

    async def on_list_view_selected(
        self,
        event: ListView.Selected,
    ) -> None:
        index = event.list_view.index
        if index is None:
            return
        if index < 0 or index >= len(self._visible_sessions):
            return
        session_id = str(
            self._visible_sessions[index].get("session_id", "")
        )
        if session_id:
            await self._open_session(session_id)

    async def action_interrupt(self) -> None:
        await self._command("interrupt")

    async def action_pause(self) -> None:
        await self._command("pause")

    async def action_refresh(self) -> None:
        await self._load_sessions()
        await self._poll()

    async def action_new_session(self) -> None:
        await self._new_session()

    async def action_checkpoint(self) -> None:
        await self._checkpoint()

    def action_toggle_sessions(self) -> None:
        sidebar = self.query_one("#sidebar", Vertical)
        handle = self.query_one(
            "#sidebar-resize-handle",
            SidebarResizeHandle,
        )
        self._sidebar_requested = not self._sidebar_requested
        sidebar.display = self._sidebar_requested
        handle.display = self._sidebar_requested

    def action_toggle_inspector(self) -> None:
        inspector = self.query_one("#inspector", Vertical)
        self._inspector_requested = not self._inspector_requested
        inspector.display = self._inspector_requested
        self._apply_sidebar_width()

    def action_sidebar_narrower(self) -> None:
        self._adjust_sidebar_width(-SIDEBAR_WIDTH_STEP)

    def action_sidebar_wider(self) -> None:
        self._adjust_sidebar_width(SIDEBAR_WIDTH_STEP)

    def action_show_help(self) -> None:
        self._write_help()

    async def _new_session(self) -> None:
        result = await self.client.request(
            "POST",
            "/v1/sessions",
            payload={
                "workspace": str(self.workspace),
                "permission_mode": self.permission_mode,
                "execution_profile": "interactive",
            },
        )
        session = _object(result.get("session"))
        session_id = str(session.get("session_id", ""))
        await self._load_sessions()
        await self._open_session(session_id)

    async def _open_session(self, session_id: str) -> None:
        self.session_id = session_id
        self.sequence = 0
        await self.client.request(
            "PATCH",
            "/v1/sessions/" + session_id,
            payload={"execution_profile": "interactive"},
        )
        state = await self.client.request(
            "GET",
            "/v1/sessions/" + session_id + "/ui-state",
        )
        ui_state = _object(state.get("ui_state"))
        self._provider_override = str(ui_state.get("provider", ""))
        self._model_override = str(ui_state.get("model", ""))
        self._effort_override = str(ui_state.get("effort", ""))
        theme = str(ui_state.get("theme", "system"))
        if theme not in {"system", "light", "dark"}:
            theme = "system"
        self._theme_preference = theme
        sidebar_width = _positive_integer(
            ui_state.get("sidebar_width"),
        )
        if sidebar_width is not None:
            self._sidebar_width = sidebar_width
        session_filter = str(
            ui_state.get("session_filter", "focused")
        )
        if session_filter not in {"focused", "all"}:
            session_filter = "focused"
        self._session_filter = session_filter
        self._show_events = (
            str(ui_state.get("events", "off")).casefold() == "on"
        )
        self._saved_sidebar_width = self._sidebar_width
        self._saved_session_filter = self._session_filter
        self._saved_show_events = self._show_events
        self._apply_sidebar_width()
        await self._sync_system_theme(force=True)
        composer = str(ui_state.get("composer", ""))
        self._saved_composer = composer
        self.query_one("#composer", Input).value = composer
        self._last_approval_id = ""
        self._transcript_events = []
        self._transcript_notices = []
        self._render_transcript()
        self._session_list_signature = ""
        await self._render_session_list()
        await self._poll()

    async def _load_sessions(self) -> None:
        result = await self.client.request("GET", "/v1/sessions")
        sessions = result.get("sessions", [])
        if not isinstance(sessions, list):
            sessions = []
        self._sessions = [
            item for item in sessions if isinstance(item, dict)
        ]
        self._session_list_signature = ""
        await self._render_session_list()

    async def _render_session_list(self) -> None:
        visible, hidden_count = _visible_sessions(
            self._sessions,
            self.session_id,
            show_all=self._session_filter == "all",
        )
        signature = repr(
            (
                visible,
                hidden_count,
                self.session_id,
                self._session_filter,
            )
        )
        if signature == self._session_list_signature:
            return
        self._session_list_signature = signature
        self._visible_sessions = visible
        view = self.query_one("#session-list", ListView)
        await view.clear()
        active_index: int | None = None
        for index, session in enumerate(visible):
            session_id = str(session.get("session_id", ""))
            classes = ""
            if session_id == self.session_id:
                classes = "active-session"
                active_index = index
            await view.append(
                ListItem(
                    Label(_session_list_label(session, self.session_id)),
                    classes=classes,
                )
            )
        if active_index is not None:
            view.index = active_index
        heading = "SESSIONS · " + self._session_filter.upper()
        self.query_one("#session-heading", Label).update(heading)
        help_text = "↑↓ select · Enter open"
        if hidden_count:
            help_text += (
                "\n"
                + str(hidden_count)
                + " hidden · /sessions all"
            )
        else:
            help_text += "\n/sessions focused"
        self.query_one("#sessions-help", Static).update(help_text)

    async def _poll(self) -> None:
        if not self._screen_is_running() or not self.session_id:
            return
        await self._save_ui_state()
        try:
            state = await self.client.request(
                "GET",
                "/v1/sessions/" + self.session_id,
            )
            result = await self.client.request(
                "GET",
                "/v1/sessions/"
                + self.session_id
                + "/events?after="
                + str(self.sequence),
            )
        except BaseException:
            return
        if not self._screen_is_running():
            return
        session = _object(state.get("session"))
        self._session = session
        await self._synchronize_active_session(session)
        self._safety = _object(state.get("safety"))
        goal = state.get("goal")
        self._goal = None
        goal_text = ""
        if isinstance(goal, dict):
            self._goal = goal
            objective = str(goal.get("objective", ""))
            if objective:
                goal_text = " · goal: " + objective
        self._approvals = _object_tuple(state.get("approvals"))
        self._provider_poll += 1
        provider_limit = 20
        if _providers_refreshing(self._providers):
            provider_limit = 4
        if not self._providers or self._provider_poll >= provider_limit:
            await self._load_providers()
            self._provider_poll = 0
        self._theme_poll += 1
        if self._theme_poll >= 20:
            await self._sync_system_theme()
            self._theme_poll = 0
        lifecycle = str(session.get("lifecycle", "starting"))
        attention = str(session.get("attention", "idle"))
        display_lifecycle = _display_lifecycle(lifecycle, attention)
        provider = str(session.get("active_provider", ""))
        if not provider:
            provider = "automatic routing"
        model = str(session.get("model", ""))
        if not model:
            model = "provider default"
        title = (
            _status_glyph(attention)
            + "  "
            + escape(str(session.get("name", "Untitled session")))
        )
        meta = (
            "[bold]"
            + escape(display_lifecycle.upper())
            + "[/bold]"
            + "  ·  "
            + escape(attention)
            + "  ·  "
            + escape(provider)
            + " / "
            + escape(model)
            + goal_text
        )
        resume = (
            "resume "
            + escape(self.session_id)
            + "  ·  "
            + escape(str(session.get("worktree", "")))
        )
        self.query_one("#session-title", Static).update(title)
        self.query_one("#session-meta", Static).update(meta)
        self.query_one("#resume-token", Static).update(resume)
        self.query_one("#connection-state", Static).update(
            _connection_label(attention)
        )
        self._render_inspector()
        events = result.get("events", [])
        if not isinstance(events, list):
            return
        transcript_changed = False
        for event in events:
            if not isinstance(event, dict):
                continue
            sequence = int(event.get("sequence", 0))
            self.sequence = max(self.sequence, sequence)
            self._transcript_events.append(event)
            transcript_changed = True
        if transcript_changed:
            self._render_transcript()
        self._present_approval()

    async def _synchronize_active_session(
        self,
        session: dict[str, Any],
    ) -> None:
        session_id = str(session.get("session_id", ""))
        if not session_id:
            return
        replaced = False
        for index, item in enumerate(self._sessions):
            if str(item.get("session_id", "")) != session_id:
                continue
            if item != session:
                self._sessions[index] = session
                replaced = True
            break
        else:
            self._sessions.insert(0, session)
            replaced = True
        if replaced:
            self._session_list_signature = ""
            await self._render_session_list()

    def _screen_is_running(self) -> bool:
        if not self.is_running:
            return False
        try:
            return self.screen.is_running
        except ScreenStackError:
            return False

    async def _command(
        self,
        command_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.session_id:
            return
        if payload is None:
            payload = {}
        await self.client.request(
            "POST",
            "/v1/sessions/"
            + self.session_id
            + "/commands/"
            + command_type,
            payload=payload,
            idempotency_key=new_uuid(),
        )

    async def _slash(self, text: str) -> None:
        parts = text.split()
        command = parts[0].casefold()
        if command == "/help":
            self._write_help()
            return
        if command == "/new":
            await self._new_session()
            return
        if command in {"/interrupt", "/pause", "/resume", "/stop"}:
            await self._command(command[1:])
            return
        if command == "/export":
            result = await self.client.request(
                "POST",
                "/v1/sessions/" + self.session_id + "/export",
                payload={},
            )
            self._write_notice(
                "[bold]Export[/bold] " + str(result.get("path", ""))
            )
            return
        if command == "/checkpoint":
            await self._checkpoint()
            return
        if command == "/fork":
            name = " ".join(parts[1:])
            result = await self.client.request(
                "POST",
                "/v1/sessions/" + self.session_id + "/fork",
                payload={"name": name},
            )
            session = _object(result.get("session"))
            await self._load_sessions()
            await self._open_session(
                str(session.get("session_id", ""))
            )
            return
        if command == "/sessions" and len(parts) == 2:
            session_filter = parts[1].casefold()
            if session_filter not in {"focused", "all"}:
                self._write_notice(
                    "[red]Sessions must be focused or all[/red]"
                )
                return
            self._session_filter = session_filter
            self._session_list_signature = ""
            await self._render_session_list()
            await self._save_ui_state(force=True)
            return
        if command == "/events" and len(parts) == 2:
            value = parts[1].casefold()
            if value not in {"on", "off"}:
                self._write_notice("[red]Events must be on or off[/red]")
                return
            self._show_events = value == "on"
            self._render_transcript()
            await self._save_ui_state(force=True)
            return
        if command == "/sidebar" and parts[1:] == ["reset"]:
            self._sidebar_width = SIDEBAR_DEFAULT_WIDTH
            self._apply_sidebar_width()
            await self._save_ui_state(force=True)
            return
        if command in {"/provider", "/model", "/effort", "/theme"}:
            if len(parts) != 2:
                self._write_notice(
                    "[red]" + command + " requires one value[/red]"
                )
                return
            value = parts[1]
            if value == "auto":
                value = ""
            if command == "/provider":
                if value not in {"", "claude", "codex"}:
                    self._write_notice("[red]Unknown provider[/red]")
                    return
                self._provider_override = value
            if command == "/model":
                self._model_override = value
            if command == "/effort":
                self._effort_override = value
            if command == "/theme":
                if value not in {"system", "light", "dark"}:
                    self._write_notice(
                        "[red]Theme must be system, light, or dark[/red]"
                    )
                    return
                self._theme_preference = value
                await self._sync_system_theme(force=True)
            await self._save_ui_state(force=True)
            self._render_inspector()
            return
        if command == "/permission" and len(parts) == 2:
            await self.client.request(
                "PATCH",
                "/v1/sessions/" + self.session_id,
                payload={"permission_mode": parts[1]},
            )
            await self._poll()
            return
        if command == "/route":
            result = await self.client.request(
                "POST",
                "/v1/sessions/" + self.session_id + "/route",
                payload={
                    "provider": self._provider_override,
                    "model": self._model_override,
                    "effort": self._effort_override,
                },
            )
            route = _object(result.get("route"))
            self._write_notice(
                "[bold]Route[/bold] "
                + escape(str(route.get("provider", "")))
                + "/"
                + escape(str(route.get("model", "")))
                + " · "
                + escape(str(route.get("reason", "")))
            )
            return
        if command == "/providers":
            await self._load_providers()
            self._render_inspector()
            return
        if command in {"/usage", "/budget"} and len(parts) == 1:
            result = await self.client.request(
                "GET",
                "/v1/sessions/" + self.session_id + "/usage",
            )
            self._safety = _object(result.get("safety"))
            self._render_inspector()
            self._write_notice("[bold]Safety usage refreshed[/bold]")
            return
        if command == "/budget" and len(parts) >= 3:
            payload: dict[str, Any] = {
                "reason": " ".join(parts[3:]) or "TUI operator extension",
            }
            if parts[1] == "xhigh":
                payload["allow_xhigh_once"] = True
                payload["reason"] = " ".join(parts[2:])
            elif parts[1] == "extend":
                try:
                    payload["additional_seconds"] = int(parts[2])
                    if len(parts) >= 4:
                        payload["additional_tokens"] = int(parts[3])
                        payload["reason"] = (
                            " ".join(parts[4:])
                            or "TUI operator extension"
                        )
                except ValueError:
                    self._write_notice(
                        "[red]Budget values must be integers[/red]"
                    )
                    return
            else:
                self._write_notice(
                    "[red]Use /budget extend or /budget xhigh[/red]"
                )
                return
            await self.client.request(
                "POST",
                "/v1/sessions/"
                + self.session_id
                + "/budget-extensions",
                payload=payload,
                idempotency_key=new_uuid(),
            )
            await self._poll()
            return
        if command == "/native" and len(parts) == 2:
            await self._native(parts[1])
            return
        if command == "/approve" and len(parts) >= 3:
            await self.client.request(
                "POST",
                "/v1/sessions/"
                + self.session_id
                + "/approvals/"
                + parts[1],
                payload={"decision": parts[2]},
            )
            return
        self._write_notice(
            "[red]Unknown harness command.[/red] Use /help."
        )

    async def _checkpoint(self) -> None:
        result = await self.client.request(
            "POST",
            "/v1/sessions/" + self.session_id + "/checkpoints",
            payload={},
        )
        checkpoint = _object(result.get("checkpoint"))
        self._write_notice(
            "[bold]Checkpoint[/bold] "
            + escape(str(checkpoint.get("checkpoint_id", "")))
        )

    async def _save_ui_state(self, *, force: bool = False) -> None:
        if not self.session_id:
            return
        composer = self.query_one("#composer", Input).value
        if not force:
            unchanged = (
                composer == self._saved_composer
                and self._sidebar_width == self._saved_sidebar_width
                and self._session_filter == self._saved_session_filter
                and self._show_events == self._saved_show_events
            )
            if unchanged:
                return
        events_value = "off"
        if self._show_events:
            events_value = "on"
        await self.client.request(
            "PUT",
            "/v1/sessions/" + self.session_id + "/ui-state",
            payload={
                "composer": composer,
                "provider": self._provider_override,
                "model": self._model_override,
                "effort": self._effort_override,
                "theme": self._theme_preference,
                "sidebar_width": str(self._sidebar_width),
                "session_filter": self._session_filter,
                "events": events_value,
            },
        )
        self._saved_composer = composer
        self._saved_sidebar_width = self._sidebar_width
        self._saved_session_filter = self._session_filter
        self._saved_show_events = self._show_events

    async def _load_providers(self) -> None:
        try:
            result = await self.client.request(
                "GET",
                "/v1/providers?workspace=" + str(self.workspace),
            )
        except BaseException:
            return
        self._providers = _object(result.get("providers"))

    def _render_inspector(self) -> None:
        lines = [
            "[bold cyan]ROUTING[/bold cyan]",
            "Provider   "
            + escape(self._provider_override or "automatic"),
            "Model      "
            + escape(self._model_override or "automatic"),
            "Effort     "
            + escape(self._effort_override or "automatic"),
            "Permission "
            + escape(str(self._session.get("permission_mode", ""))),
            "Theme      " + escape(self._theme_preference),
        ]
        if self._goal is not None:
            lines.extend(
                [
                    "",
                    "[bold cyan]GOAL[/bold cyan]",
                    escape(str(self._goal.get("kind", "")))
                    + " · "
                    + escape(str(self._goal.get("status", ""))),
                    escape(str(self._goal.get("objective", ""))),
                ]
            )
        safety_session = _object(self._safety.get("session"))
        envelopes = self._safety.get("envelopes", [])
        if not isinstance(envelopes, list):
            envelopes = []
        lines.extend(
            [
                "",
                "[bold cyan]SAFETY ENVELOPE[/bold cyan]",
                "Profile    "
                + escape(str(safety_session.get("profile", "unknown"))),
            ]
        )
        if envelopes:
            envelope = _object(envelopes[-1])
            consumption = _object(envelope.get("consumption"))
            limits = _object(envelope.get("limits"))
            accounting = "estimated"
            if bool(consumption.get("exact_tokens", False)):
                accounting = "provider-reported"
            lines.append(
                "State      "
                + escape(str(envelope.get("state", "")))
            )
            if "input_tokens" in consumption:
                lines.append(
                    "Input      "
                    + _format_number(consumption.get("input_tokens", 0))
                )
                cached_tokens = int(
                    consumption.get("cached_input_tokens", 0)
                )
                if cached_tokens:
                    lines[-1] += (
                        " · "
                        + _format_number(cached_tokens)
                        + " cached"
                    )
            lines.extend(
                [
                    "Output     "
                    + _format_number(
                        consumption.get("output_tokens", 0)
                    ),
                    "Harness    "
                    + _format_number(
                        consumption.get("context_tokens", 0)
                    )
                    + " context est.",
                    "Total      "
                    + _format_number(
                        consumption.get("total_tokens", 0)
                    )
                    + " / "
                    + _format_number(
                        limits.get("max_total_tokens", 0)
                    ),
                    "Time       "
                    + escape(
                        str(round(float(
                            consumption.get("elapsed_seconds", 0)
                        )))
                    )
                    + "s / "
                    + escape(str(limits.get("max_seconds", 0)))
                    + "s",
                    "Tools      "
                    + escape(str(consumption.get("tool_calls", 0)))
                    + " / "
                    + escape(str(limits.get("max_tool_calls", 0))),
                    "Accounting " + accounting,
                    "Recovery   stage "
                    + escape(str(envelope.get("recovery_stage", 0))),
                ]
            )
            guard_reason = str(envelope.get("guard_reason", ""))
            if guard_reason:
                lines.append(
                    "[bold red]Guard      "
                    + escape(guard_reason)
                    + "[/bold red]"
                )
        lines.extend(
            ["", "[bold cyan]PENDING APPROVALS[/bold cyan]"]
        )
        if not self._approvals:
            lines.append("[dim]None[/dim]")
        for approval in self._approvals:
            lines.append(
                escape(str(approval.get("approval_id", "")))
                + "\n"
                + escape(str(approval.get("prompt", "")))
            )
        lines.extend(
            ["", "[bold cyan]PROVIDER CAPACITY[/bold cyan]"]
        )
        for provider, value in sorted(self._providers.items()):
            provider_value = _object(value)
            usage = _object(provider_value.get("usage"))
            binding = usage.get("binding_percent")
            ready = bool(provider_value.get("ready", False))
            if ready:
                detail = "[green]●[/green] "
            else:
                detail = "[red]●[/red] "
            detail += escape(provider)
            if binding is not None:
                detail += (
                    " · "
                    + escape(str(100 - float(binding)))
                    + "% headroom"
                )
            if not ready:
                detail += " · unavailable"
            if bool(provider_value.get("usage_refreshing", False)):
                detail += " · refreshing"
            lines.append(detail)
        if not self._providers:
            lines.append("[dim]Discovering providers…[/dim]")
        lines.extend(
            [
                "",
                "[bold cyan]SHORTCUTS[/bold cyan]",
                "Ctrl+B  sessions",
                "Ctrl+O  inspector",
                "Ctrl+Shift+←/→  resize sessions",
                "Ctrl+K  checkpoint",
                "F1      command help",
            ]
        )
        self.query_one("#inspector-content", Static).update(
            "\n".join(lines)
        )

    def _write_help(self) -> None:
        self._write_notice(
            "\n[bold cyan]HARNESS COMMANDS[/bold cyan]\n"
            "[bold]Session[/bold]  /new · /checkpoint · "
            "/fork [name] · /export\n"
            "[bold]Control[/bold]  /interrupt · /pause · "
            "/resume · /stop\n"
            "[bold]Route[/bold]    /provider <auto|claude|codex> · "
            "/model <auto|id>\n"
            "         /effort <auto|level> · /route · /providers\n"
            "[bold]Display[/bold]  /theme <system|light|dark> · "
            "/sessions <focused|all>\n"
            "         /events <on|off> · /sidebar reset · "
            "Ctrl+Shift+←/→ resize\n"
            "[bold]Safety[/bold]   /usage · /budget · "
            "/budget extend <seconds> <tokens> <reason> · "
            "/budget xhigh <reason>\n"
            "[bold]Access[/bold]   /permission <mode> · "
            "/approve <uuid> <decision>\n"
            "[bold]Native[/bold]   /native <claude|codex>\n"
        )

    def _welcome_message(self) -> str:
        return (
            "[bold]Durable session ready[/bold]\n"
            "Messages, tool results, approvals, routing, "
            "and unfinished drafts persist under this resume ID.\n"
            "[dim]Type a request below. Use F1 for controls.[/dim]\n"
        )

    def _apply_responsive_layout(self, width: int) -> None:
        if width < 118:
            self.screen.add_class("compact")
        else:
            self.screen.remove_class("compact")
        if width < 78:
            self.screen.add_class("narrow")
        else:
            self.screen.remove_class("narrow")

    def _apply_sidebar_width(self) -> None:
        sidebar = self.query_one("#sidebar", Vertical)
        sidebar.styles.width = self._bounded_sidebar_width(
            self._sidebar_width
        )

    def _bounded_sidebar_width(self, width: int) -> int:
        reserve = SIDEBAR_MAIN_RESERVE
        if self.size.width >= 118 and self._inspector_requested:
            reserve += 36
        maximum = self.size.width - reserve - 1
        if maximum < SIDEBAR_MIN_WIDTH:
            maximum = SIDEBAR_MIN_WIDTH
        if width < SIDEBAR_MIN_WIDTH:
            return SIDEBAR_MIN_WIDTH
        if width > maximum:
            return maximum
        return width

    def _resize_sidebar_to(self, width: int) -> None:
        self._sidebar_width = self._bounded_sidebar_width(width)
        self._apply_sidebar_width()

    def _finish_sidebar_resize(self) -> None:
        self.run_worker(
            self._save_ui_state(force=True),
            group="sidebar-ui-state",
            exclusive=True,
        )

    def _adjust_sidebar_width(self, delta: int) -> None:
        self._resize_sidebar_to(self._sidebar_width + delta)
        self._finish_sidebar_resize()

    async def _sync_system_theme(self, *, force: bool = False) -> None:
        if self._theme_preference == "dark":
            self._apply_theme(True)
            return
        if self._theme_preference == "light":
            self._apply_theme(False)
            return
        if not force and self._theme_preference != "system":
            return
        dark = await asyncio.to_thread(
            _system_dark_mode,
            self.current_theme.dark,
        )
        self._apply_theme(dark)

    def _apply_theme(self, dark: bool) -> None:
        if dark:
            self.theme = "textual-dark"
            self.screen.remove_class("light")
            return
        self.theme = "textual-light"
        self.screen.add_class("light")

    def _present_approval(self) -> None:
        if not self._approvals:
            return
        approval = self._approvals[0]
        approval_id = str(approval.get("approval_id", ""))
        if not approval_id or approval_id == self._last_approval_id:
            return
        self._last_approval_id = approval_id
        self.push_screen(
            ApprovalScreen(approval),
            self._approval_decision,
        )

    def _approval_decision(self, decision: str) -> None:
        if not decision:
            return
        self.run_worker(
            self._resolve_approval(
                self._last_approval_id,
                decision,
            )
        )

    async def _resolve_approval(
        self,
        approval_id: str,
        decision: str,
    ) -> None:
        await self.client.request(
            "POST",
            "/v1/sessions/"
            + self.session_id
            + "/approvals/"
            + approval_id,
            payload={"decision": decision},
        )
        self._last_approval_id = ""
        await self._poll()

    async def _native(self, provider: str) -> None:
        if provider not in {"claude", "codex"}:
            self._write_notice("[red]Unknown provider[/red]")
            return
        command = _native_command(
            provider,
            str(self._session.get("permission_mode", "approval")),
        )
        workspace = str(
            self._session.get("worktree", self.workspace)
        )
        with self.suspend():
            await asyncio.to_thread(
                subprocess.run,
                command,
                cwd=workspace,
                check=False,
            )
        await self._poll()

    def _render_transcript(self) -> None:
        transcript = self.query_one("#transcript", ConversationLog)
        transcript.clear()
        transcript.write(self._welcome_message())
        for rendered in _render_transcript_events(
            self._transcript_events,
            show_events=self._show_events,
        ):
            transcript.write(rendered)
        for notice in self._transcript_notices:
            transcript.write(notice)

    def _write_notice(self, value: str) -> None:
        self._transcript_notices.append(value)
        self._render_transcript()


def _render_transcript_events(
    events: list[dict[str, Any]],
    *,
    show_events: bool,
) -> list[str]:
    final_turns = {
        str(event.get("turn_id", ""))
        for event in events
        if event.get("event_type") == "agent.message"
        and str(event.get("turn_id", ""))
    }
    live_deltas: dict[str, list[str]] = {}
    for event in events:
        if event.get("event_type") != "agent.message.delta":
            continue
        turn_id = str(event.get("turn_id", ""))
        if turn_id in final_turns:
            continue
        live_deltas.setdefault(turn_id, []).append(
            str(event.get("text", ""))
        )
    emitted_deltas: set[str] = set()
    rendered: list[str] = []
    for event in events:
        event_type = str(event.get("event_type", ""))
        if event_type == "agent.message.delta":
            turn_id = str(event.get("turn_id", ""))
            if turn_id in final_turns or turn_id in emitted_deltas:
                continue
            emitted_deltas.add(turn_id)
            projected = dict(event)
            projected["event_type"] = "agent.message"
            projected["text"] = "".join(live_deltas.get(turn_id, []))
            value = _render_event(projected, show_events=show_events)
        else:
            value = _render_event(event, show_events=show_events)
        if value:
            rendered.append(value)
    return rendered


def _render_event(
    event: dict[str, Any],
    *,
    show_events: bool = False,
) -> str:
    event_type = str(event.get("event_type", "event"))
    text = escape(str(event.get("text", "")))
    if event_type in {"user.message", "user.steer"}:
        return "\n[bold cyan]YOU[/bold cyan]\n" + text + "\n"
    if event_type in {"agent.message", "agent.message.delta"}:
        return "\n[bold green]AGENT[/bold green]\n" + text + "\n"
    if event_type.casefold() in {
        "tool.usermessage.started",
        "tool.usermessage.completed",
        "tool.user_message.started",
        "tool.user_message.completed",
    }:
        return ""
    if event_type.startswith("tool."):
        label = event_type.removeprefix("tool.").replace(".", " ")
        if not text and not show_events:
            return ""
        return (
            "\n[bold yellow]TOOL · "
            + escape(label)
            + "[/bold yellow]\n"
            + text
            + "\n"
        )
    if event_type.startswith("reasoning.summary"):
        return "[dim]" + text + "[/dim]"
    if event_type == "approval.requested":
        metadata = _object(event.get("metadata"))
        return (
            "[bold magenta]Approval required[/bold magenta] "
            + escape(str(metadata.get("approval_id", "")))
        )
    if event_type == "guard.warning":
        return (
            "[bold yellow]Safety envelope is above 80%[/bold yellow]"
        )
    if event_type == "guard.tripped":
        metadata = _object(event.get("metadata"))
        return (
            "\n[bold red]TURN INTERRUPTED BY SAFETY GUARD[/bold red]\n"
            + "Reason: "
            + escape(str(metadata.get("reason", "unknown")))
            + "\nAction: "
            + escape(str(metadata.get("action", "pause")))
            + "\n"
        )
    metadata = _object(event.get("metadata"))
    if event_type == "routing.selected":
        provider = str(metadata.get("provider", "provider"))
        model = str(metadata.get("model", "default"))
        return (
            "[dim]Routed to "
            + escape(provider)
            + " · "
            + escape(model)
            + "[/dim]"
        )
    if event_type == "routing.failover":
        provider = str(metadata.get("excluded_provider", "provider"))
        return "[dim]Failing over from " + escape(provider) + "[/dim]"
    if event_type == "checkpoint.created":
        return "[dim]Checkpoint saved[/dim]"
    if event_type == "goal.completed":
        return "[dim]Goal completed[/dim]"
    if event_type == "turn.failed":
        return "[bold red]Turn failed[/bold red]\n" + text
    if show_events and event_type not in {
        "usage.updated",
        "usage.reserved",
    }:
        return "[dim]EVENT · " + escape(event_type) + "[/dim]"
    return ""


def _visible_sessions(
    sessions: list[dict[str, Any]],
    active_session_id: str,
    *,
    show_all: bool,
) -> tuple[list[dict[str, Any]], int]:
    if show_all:
        return (list(sessions), 0)
    visible: list[dict[str, Any]] = []
    idle_sessions = 0
    for session in sessions:
        session_id = str(session.get("session_id", ""))
        attention = str(session.get("attention", "idle"))
        lifecycle = str(session.get("lifecycle", ""))
        important = (
            session_id == active_session_id
            or attention != "idle"
            or lifecycle == "paused"
        )
        if important:
            visible.append(session)
            continue
        if lifecycle not in {"starting", "running"}:
            continue
        if idle_sessions >= FOCUSED_IDLE_SESSION_LIMIT:
            continue
        visible.append(session)
        idle_sessions += 1
    return (visible, len(sessions) - len(visible))


def _session_list_label(
    session: dict[str, Any],
    active_session_id: str,
) -> str:
    session_id = str(session.get("session_id", ""))
    marker = "  "
    if session_id == active_session_id:
        marker = "● "
    name = str(session.get("name", "Untitled session"))
    provider = str(session.get("active_provider", ""))
    lifecycle = str(session.get("lifecycle", "starting"))
    attention = str(session.get("attention", "idle"))
    status = _session_status(lifecycle, attention)
    detail = marker + escape(name) + "\n  " + escape(status)
    if provider:
        detail += " · " + escape(provider)
    return detail


def _session_status(lifecycle: str, attention: str) -> str:
    if attention == "working":
        return "working"
    if attention in {"needs-input", "needs-reconciliation"}:
        return "action needed"
    if attention == "failed":
        return "needs attention"
    return _display_lifecycle(lifecycle, attention)


def _display_lifecycle(lifecycle: str, attention: str) -> str:
    if attention == "idle" and lifecycle in {"starting", "running"}:
        return "ready"
    return lifecycle.replace("-", " ")


def _positive_integer(value: object) -> int | None:
    try:
        result = int(str(value))
    except (TypeError, ValueError):
        return None
    if result <= 0:
        return None
    return result


def _format_number(value: object) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return escape(str(value))
    return f"{number:,}"


def _status_glyph(attention: str) -> str:
    if attention == "working":
        return "[bold yellow]◉[/bold yellow]"
    if attention in {"needs-input", "needs-reconciliation"}:
        return "[bold magenta]◆[/bold magenta]"
    if attention == "failed":
        return "[bold red]●[/bold red]"
    return "[bold green]●[/bold green]"


def _connection_label(attention: str) -> str:
    if attention == "working":
        return "[yellow]◉ agent working[/yellow]"
    if attention in {"needs-input", "needs-reconciliation"}:
        return "[magenta]◆ action needed[/magenta]"
    if attention == "failed":
        return "[red]● needs attention[/red]"
    return "[green]● connected[/green]"


def _providers_refreshing(providers: dict[str, Any]) -> bool:
    for value in providers.values():
        provider = _object(value)
        if bool(provider.get("usage_refreshing", False)):
            return True
    return False


def _system_dark_mode(default: bool) -> bool:
    if sys.platform != "darwin":
        return default
    try:
        completed = subprocess.run(
            (
                "/usr/bin/defaults",
                "read",
                "-g",
                "AppleInterfaceStyle",
            ),
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return default
    if completed.returncode != 0:
        return False
    return completed.stdout.strip().casefold() == "dark"


def _object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _object_tuple(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _native_command(provider: str, permission_mode: str) -> list[str]:
    if provider == "codex":
        command = ["npx", "-y", "@openai/codex@0.146.0"]
        if permission_mode == "full":
            command.append("--yolo")
        return command
    command = ["npx", "@anthropic-ai/claude-code@2.1.220"]
    if permission_mode == "full":
        command.append("--dangerously-skip-permissions")
    return command


def run_tui(
    client: HarnessClient,
    workspace: Path,
    *,
    session_id: str = "",
    permission_mode: str = "approval",
) -> None:
    app = HarnessApp(
        client,
        workspace,
        session_id=session_id,
        permission_mode=permission_mode,
    )
    app.run()
