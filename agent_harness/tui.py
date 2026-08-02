"""Textual workspace for durable agent sessions."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual import events
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.timer import Timer
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
    Tab,
    Tabs,
)

from agent_harness.client import HarnessClient
from agent_harness.errors import HarnessError
from agent_harness.ids import new_uuid
from agent_harness.notifications import (
    Notification,
    NotificationPersistence,
    NotificationSeverity,
    NotificationState,
    project_notifications,
    push_notification,
    with_connection,
)
from agent_harness.presentation import SessionSwitchCoordinator, SessionViewCacheEntry
from agent_harness.tui_presenter import (
    TranscriptBlock,
    TranscriptBlockKind,
    TranscriptMutationKind,
    TranscriptState,
    decide_layout,
    project_events,
)
from agent_harness.tui_widgets import (
    DEFAULT_SLASH_COMMANDS,
    ComposerDraft,
    MultilineComposer,
    complete_slash,
    validate_slash,
)

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


class TranscriptBlockView(Static):
    """One stable transcript block updated in place."""

    can_focus = True

    def __init__(
        self,
        block: TranscriptBlock,
        *,
        expanded: bool = False,
    ) -> None:
        super().__init__(classes=_transcript_block_classes(block))
        self.block_id = block.block_id
        self.set_block(block, expanded=expanded)

    def set_block(
        self,
        block: TranscriptBlock,
        *,
        expanded: bool,
    ) -> None:
        self.block = block
        self.set_classes(_transcript_block_classes(block))
        self.update(_render_transcript_block(block, expanded=expanded))

    def on_click(self) -> None:
        application = self.app
        if not isinstance(application, HarnessApp):
            return
        application.toggle_transcript_block(self.block_id)


class TranscriptView(VerticalScroll):
    """Incremental transcript that preserves stable block widgets."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._block_views: dict[str, TranscriptBlockView] = {}

    async def reset(
        self,
        state: TranscriptState,
        welcome: str,
    ) -> None:
        await self.remove_children()
        self._block_views = {}
        await self.mount(
            Static(
                welcome,
                classes="transcript-welcome",
                markup=True,
            )
        )
        for block in state.blocks:
            await self._mount_block(block, state)
        self.scroll_end(animate=False)

    async def apply(
        self,
        state: TranscriptState,
        mutations,
    ) -> None:
        for mutation in mutations:
            if mutation.kind == TranscriptMutationKind.IGNORE:
                continue
            block = state.block(mutation.block_id)
            if block is None:
                continue
            view = self._block_views.get(block.block_id)
            if view is None:
                await self._mount_block(block, state)
                continue
            view.set_block(
                block,
                expanded=(block.block_id in state.expanded_block_ids),
            )
        if state.reader_at_bottom:
            self.scroll_end(animate=False)

    async def _mount_block(
        self,
        block: TranscriptBlock,
        state: TranscriptState,
    ) -> None:
        view = TranscriptBlockView(
            block,
            expanded=block.block_id in state.expanded_block_ids,
        )
        self._block_views[block.block_id] = view
        await self.mount(view)

    def refresh_block(
        self,
        block: TranscriptBlock,
        *,
        expanded: bool,
    ) -> None:
        view = self._block_views.get(block.block_id)
        if view is None:
            return
        view.set_block(block, expanded=expanded)


class SidebarResizeHandle(Static):
    """Mouse-draggable divider for the session sidebar."""

    can_focus = True

    def on_mount(self) -> None:
        self.tooltip = (
            "Drag to resize sessions. Ctrl+Shift+Left/Right also adjusts the width."
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


class CommandPaletteScreen(ModalScreen[str]):
    """Searchable action palette backed by the slash-command registry."""

    CSS = """
    CommandPaletteScreen {
        align: center top;
        padding-top: 3;
        background: $background 55%;
    }
    #command-palette {
        width: 76;
        max-width: 94%;
        height: 24;
        max-height: 82%;
        padding: 1;
        border: round $accent;
        background: $surface;
    }
    #command-query {
        height: 3;
        margin-bottom: 1;
    }
    #command-results {
        height: 1fr;
    }
    #command-guidance {
        height: 2;
        color: $text-muted;
    }
    """

    def __init__(self, query: str = "") -> None:
        super().__init__()
        self._initial_query = query
        self._commands = list(DEFAULT_SLASH_COMMANDS)

    def compose(self) -> ComposeResult:
        with Vertical(id="command-palette"):
            yield Input(
                value=self._initial_query,
                placeholder="Find an action",
                id="command-query",
            )
            yield ListView(id="command-results")
            yield Static(
                "Enter choose · Esc close",
                id="command-guidance",
            )

    async def on_mount(self) -> None:
        await self._render_results(self._initial_query)
        self.query_one("#command-query", Input).focus()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "command-query":
            return
        await self._render_results(event.value)

    async def on_list_view_selected(
        self,
        event: ListView.Selected,
    ) -> None:
        if event.list_view.id != "command-results":
            return
        index = event.list_view.index
        if index is None or index < 0 or index >= len(self._commands):
            return
        self.dismiss(self._commands[index].name + " ")

    async def _render_results(self, query: str) -> None:
        value = query.strip()
        if not value:
            self._commands = list(DEFAULT_SLASH_COMMANDS)
        else:
            if not value.startswith("/"):
                value = "/" + value
            completion = complete_slash(value)
            self._commands = [item.command for item in completion.items]
        if not self._commands:
            self._commands = list(DEFAULT_SLASH_COMMANDS)
        view = self.query_one("#command-results", ListView)
        await view.clear()
        for command in self._commands:
            await view.append(
                ListItem(
                    Label(
                        "[bold]"
                        + escape(command.name)
                        + "[/bold]\n[dim]"
                        + escape(command.summary)
                        + "[/dim]",
                        markup=True,
                    )
                )
            )
        if self._commands:
            view.index = 0
        guidance = "No matching action"
        if self._commands:
            guidance = self._commands[0].usage
        self.query_one("#command-guidance", Static).update(guidance)


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
    Screen.switching #body {
        opacity: 70%;
    }
    #topbar {
        height: 4;
        padding: 0 1;
        background: #111823;
        border-bottom: solid #27364a;
        align-vertical: middle;
    }
    #brand {
        width: 20;
        color: #7dd3fc;
        text-style: bold;
        content-align: left middle;
    }
    #header-copy {
        width: 1fr;
        height: 3;
        padding-left: 1;
    }
    #session-title {
        height: 1;
        color: #f1f5f9;
        text-style: bold;
    }
    #session-meta {
        height: 1;
        color: #7f8da3;
    }
    #mode-toggle {
        width: 11;
        min-width: 11;
        height: 3;
        min-height: 3;
        border: none;
        background: #172231;
        color: #bae6fd;
    }
    #connection-state {
        width: 19;
        color: #86efac;
        content-align: right middle;
    }
    #connection-state.status-working,
    #connection-state.status-reconnecting,
    #connection-state.status-switching {
        color: #facc15;
    }
    #connection-state.status-action,
    #connection-state.status-pending {
        color: #e879f9;
    }
    #connection-state.status-failed,
    #connection-state.status-disconnected {
        color: #f87171;
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
    #session-search {
        height: 3;
        margin-bottom: 1;
        border: round #33465f;
        background: #101822;
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
        height: 1;
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
    #inspector-tabs {
        height: 3;
        margin-bottom: 1;
    }
    #inspector-content {
        height: 1fr;
        color: #b8c2d1;
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
    .transcript-welcome {
        height: auto;
        margin-bottom: 1;
        padding: 1 2;
        color: #8fa0b7;
        background: #0f1620;
        border-left: tall #33465f;
    }
    .transcript-block {
        height: auto;
        min-height: 3;
        margin-bottom: 1;
        padding: 1 1;
        background: transparent;
        border-left: tall transparent;
    }
    .transcript-block:focus {
        background: #172231;
        border-left: tall #38bdf8;
    }
    .transcript-user {
        background: #0f1620;
        border-left: tall #38bdf8;
    }
    .transcript-assistant {
        border-left: tall #294638;
    }
    .transcript-tool {
        border-left: tall #facc15;
    }
    .transcript-approval {
        border-left: tall #e879f9;
    }
    .transcript-reconciliation {
        border-left: tall #fb923c;
    }
    .transcript-warning {
        border-left: tall #f87171;
    }
    #new-activity {
        display: none;
        height: 1;
        margin: 0 3;
        color: #7dd3fc;
        content-align: right middle;
    }
    #control-view {
        display: none;
        height: 1fr;
        padding: 1 2;
    }
    #turn-list {
        width: 38;
        min-width: 28;
        height: 1fr;
        margin-right: 1;
        border: none;
        background: transparent;
    }
    #turn-list ListItem {
        height: auto;
        min-height: 3;
        padding: 0 1;
        margin-bottom: 1;
        background: #0f1620;
        border-left: tall #33465f;
    }
    #turn-list ListItem.-highlight {
        background: #18344d;
        border-left: tall #38bdf8;
    }
    #turn-detail-pane {
        width: 1fr;
        height: 1fr;
        background: #0d131c;
        border-left: solid #27364a;
    }
    #control-detail-tabs {
        height: 3;
        margin: 0 1;
    }
    #turn-detail {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
    }
    #notification-shell {
        display: none;
        height: auto;
        min-height: 3;
        margin: 0 2;
        padding: 0 1;
        background: #151d29;
        border-left: tall #64748b;
    }
    #notification-message {
        width: 1fr;
        height: auto;
        padding: 0 1;
        content-align: left middle;
    }
    #notification-primary,
    #notification-secondary {
        display: none;
        width: auto;
        min-width: 8;
        height: 3;
        min-height: 3;
        border: none;
        background: #24354a;
    }
    #notification-shell.notification-action {
        border-left: tall #e879f9;
    }
    #notification-shell.notification-warning {
        border-left: tall #facc15;
    }
    #notification-shell.notification-danger {
        border-left: tall #f87171;
    }
    #notification-shell.notification-success {
        border-left: tall #86efac;
    }
    #composer-shell {
        height: auto;
        min-height: 5;
        max-height: 14;
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
    #composer-actions {
        height: 3;
        align-vertical: middle;
    }
    #composer-help {
        width: 1fr;
        content-align: left middle;
    }
    #send-message,
    #stop-work {
        width: auto;
        min-width: 8;
        height: 3;
        min-height: 3;
        border: none;
    }
    #send-message {
        background: #16405d;
        color: #e0f2fe;
    }
    #stop-work {
        background: transparent;
        color: #fca5a5;
    }
    Screen.control #transcript,
    Screen.control #new-activity {
        display: none;
    }
    Screen.control #control-view {
        display: block;
    }
    Screen.control #inspector {
        display: none;
    }
    Screen.compact #inspector {
        display: none;
    }
    Screen.compact #session-meta {
        display: none;
    }
    Screen.narrow #sidebar {
        display: none;
    }
    Screen.narrow #sidebar-resize-handle {
        display: none;
    }
    Screen.narrow #brand {
        width: 6;
    }
    Screen.narrow #mode-toggle {
        width: 9;
        min-width: 9;
    }
    Screen.narrow #connection-state {
        width: 12;
    }
    Screen.overlay #sidebar {
        layer: overlay;
        border-right: solid #33465f;
    }
    Screen.overlay #sidebar-resize-handle {
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
    Screen.narrow #notification-shell {
        margin-left: 1;
        margin-right: 1;
    }
    Screen.narrow #turn-list {
        width: 1fr;
        margin-right: 0;
    }
    Screen.narrow #turn-detail-pane {
        display: none;
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
    Screen.light #session-meta {
        color: #475569;
    }
    Screen.light #connection-state {
        color: #166534;
    }
    Screen.light #connection-state.status-working,
    Screen.light #connection-state.status-reconnecting,
    Screen.light #connection-state.status-switching {
        color: #92400e;
    }
    Screen.light #connection-state.status-action,
    Screen.light #connection-state.status-pending {
        color: #86198f;
    }
    Screen.light #connection-state.status-failed,
    Screen.light #connection-state.status-disconnected {
        color: #b91c1c;
    }
    Screen.light #sidebar,
    Screen.light #inspector {
        background: #f1f5f9;
    }
    Screen.light #session-search {
        background: #ffffff;
        border: round #64748b;
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
    Screen.light .transcript-welcome,
    Screen.light .transcript-block {
        color: #1e293b;
        background: #f8fafc;
    }
    Screen.light .transcript-block:focus {
        background: #e2e8f0;
    }
    Screen.light #session-title {
        color: #0f172a;
    }
    Screen.light #session-meta {
        color: #475569;
    }
    Screen.light #mode-toggle {
        color: #075985;
        background: #dbeafe;
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
    Screen.light #control-view {
        background: #ffffff;
    }
    Screen.light #turn-list ListItem,
    Screen.light #turn-detail-pane,
    Screen.light #notification-shell {
        color: #1e293b;
        background: #f8fafc;
    }
    Screen.light #notification-primary,
    Screen.light #notification-secondary {
        color: #ffffff;
        background: #334155;
    }
    Screen.light #stop-work {
        color: #b91c1c;
    }
    Screen.light #transcript {
        scrollbar-background: #ffffff;
        scrollbar-color: #94a3b8;
        scrollbar-color-hover: #64748b;
        scrollbar-color-active: #0369a1;
    }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
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
        Binding("f2", "command_palette", "Actions"),
        Binding("f3", "toggle_mode", "Focus / Control"),
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
        self._reconciliations: tuple[dict[str, Any], ...] = ()
        self._providers: dict[str, Any] = {}
        self._sync: dict[str, Any] = {}
        self._state_root = ""
        self._provider_override = ""
        self._model_override = ""
        self._effort_override = ""
        self._transcript_events: list[dict[str, Any]] = []
        self._transcript_notices: list[str] = []
        self._transcript_state = TranscriptState()
        self._saved_composer = ""
        self._saved_composer_cursor = "0:0"
        self._saved_request_id = ""
        self._saved_sidebar_width = SIDEBAR_DEFAULT_WIDTH
        self._saved_session_filter = "focused"
        self._saved_session_query = ""
        self._saved_show_events = False
        self._saved_workspace_mode = "focus"
        self._saved_selected_turn_id = ""
        self._saved_notification_sequence = 0
        self._last_approval_id = ""
        self._provider_poll = 0
        self._sync_poll = 0
        self._theme_poll = 0
        self._poll_timer: Timer | None = None
        self._theme_preference = "system"
        self._inspector_tab = "context"
        self._pending_request_id = ""
        self._pending_retry_count = 0
        self._recovering_send = False
        self._connection_status = "connected"
        self._sidebar_requested = True
        self._inspector_requested = True
        self._sidebar_width = SIDEBAR_DEFAULT_WIDTH
        self._session_filter = "focused"
        self._session_query = ""
        self._show_events = False
        self._layout = decide_layout(120, 36)
        self._workspace_mode = "focus"
        self._selected_turn_id = ""
        self._control_detail_tab = "summary"
        self._turns: list[dict[str, Any]] = []
        self._notification_state = NotificationState()
        self._notification_ack_sequence = 0
        self._notification_timer: Timer | None = None
        self._notification_actions: dict[str, str] = {}
        self._switches = SessionSwitchCoordinator(maximum_entries=8)
        self._switching_session_id = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("P13I AGENT HARNESS", id="brand")
            with Vertical(id="header-copy"):
                yield Static("Starting…", id="session-title")
                yield Static(
                    "Preparing durable workspace",
                    id="session-meta",
                )
            yield Button("Focus", id="mode-toggle")
            yield Static("● connected", id="connection-state")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("WORKSPACE", classes="eyebrow")
                yield Static(
                    self.workspace.name + "\n" + escape(str(self.workspace)),
                    id="workspace-summary",
                )
                yield Button(
                    "＋  New session     Ctrl+N",
                    id="new-session",
                )
                yield Input(
                    placeholder="Search sessions",
                    id="session-search",
                )
                yield Label("SESSIONS", id="session-heading")
                yield ListView(id="session-list")
                yield Static(
                    "↑↓ select · Enter open\n/sessions all for history",
                    id="sessions-help",
                )
            yield SidebarResizeHandle("│", id="sidebar-resize-handle")
            with Vertical(id="main"):
                yield TranscriptView(
                    id="transcript",
                )
                yield Static("", id="new-activity")
                with Horizontal(id="control-view"):
                    yield ListView(id="turn-list")
                    with Vertical(id="turn-detail-pane"):
                        yield Tabs(
                            Tab("Summary", id="summary"),
                            Tab("Activity", id="activity"),
                            Tab("Changes", id="changes"),
                            Tab("Evidence", id="evidence"),
                            Tab("Recovery", id="recovery"),
                            id="control-detail-tabs",
                        )
                        yield Static(
                            "Select a turn",
                            id="turn-detail",
                            markup=True,
                        )
                with Horizontal(id="notification-shell"):
                    yield Static("", id="notification-message")
                    yield Button(
                        "Review",
                        id="notification-primary",
                    )
                    yield Button(
                        "Later",
                        id="notification-secondary",
                    )
                with Vertical(id="composer-shell"):
                    yield Label("Message", id="composer-label")
                    yield MultilineComposer(
                        "",
                        min_lines=1,
                        max_lines=8,
                        id="composer",
                    )
                    with Horizontal(id="composer-actions"):
                        yield Static(
                            "interactive · approval · connected",
                            id="composer-help",
                        )
                        yield Button("Stop", id="stop-work")
                        yield Button("Send", id="send-message")
            with Vertical(id="inspector"):
                yield Label("SESSION CONTROL", id="inspector-heading")
                yield Tabs(
                    Tab("Context", id="context"),
                    Tab("Goal", id="goal"),
                    Tab("Usage", id="usage"),
                    Tab("Approvals", id="approvals"),
                    Tab("Recovery", id="recovery"),
                    Tab("Storage", id="storage"),
                    id="inspector-tabs",
                )
                yield Static("", id="inspector-content")

    async def on_mount(self) -> None:
        self._apply_responsive_layout(self.size.width)
        self._apply_sidebar_width()
        await self._sync_system_theme(force=True)
        await self._load_sync()
        await self._load_sessions()
        if self.session_id:
            await self._open_session(self.session_id)
        else:
            await self._new_session()
        self._poll_timer = self.set_interval(0.5, self._poll)
        self.query_one("#composer", MultilineComposer).focus()

    async def on_unmount(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
        try:
            await self._save_ui_state(force=True)
            await self.client.request(
                "POST",
                "/v1/sync",
                payload={},
            )
        except BaseException:
            return

    def on_resize(self, event: events.Resize) -> None:
        self._apply_responsive_layout(event.size.width)
        self._apply_sidebar_width()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "new-session":
            await self._new_session()
            return
        if button_id == "mode-toggle":
            self.action_toggle_mode()
            return
        if button_id == "send-message":
            self.query_one("#composer", MultilineComposer).action_submit()
            return
        if button_id == "stop-work":
            await self._command("stop")
            return
        if button_id in {
            "notification-primary",
            "notification-secondary",
        }:
            await self._handle_notification_action(button_id)

    async def on_multiline_composer_submitted(
        self,
        event: MultilineComposer.Submitted,
    ) -> None:
        text = event.text.strip()
        if not text:
            return
        if text.startswith("/"):
            validation = validate_slash(text)
            if not validation.can_execute:
                self._write_notice(
                    "[bold red]" + escape(validation.message) + "[/bold red]"
                )
                return
            await self._slash(text)
            event.composer.text = ""
            await self._save_ui_state(force=True)
            return
        if self._pending_request_id:
            self._write_notice(
                "[bold yellow]Waiting for the original send "
                "acknowledgement.[/bold yellow]"
            )
            return
        self._transcript_notices = []
        self._pending_request_id = new_uuid()
        try:
            await self._save_ui_state(force=True)
        except (HarnessError, OSError):
            self._set_connection_state("reconnecting")
            return
        await self._submit_pending_message(text)

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "session-search":
            return
        self._session_query = event.value
        self._session_list_signature = ""
        await self._render_session_list()
        await self._save_ui_state(force=True)

    async def on_tabs_tab_activated(
        self,
        event: Tabs.TabActivated,
    ) -> None:
        if event.tabs.id == "control-detail-tabs":
            self._control_detail_tab = event.tab.id or "summary"
            self._render_selected_turn()
            await self._save_ui_state(force=True)
            return
        if event.tabs.id != "inspector-tabs":
            return
        self._inspector_tab = event.tab.id or "context"
        self._render_inspector()
        await self._save_ui_state(force=True)

    async def on_list_view_selected(
        self,
        event: ListView.Selected,
    ) -> None:
        if event.list_view.id == "turn-list":
            await self._select_turn(event.list_view.index)
            return
        if event.list_view.id != "session-list":
            return
        index = event.list_view.index
        if index is None:
            return
        if index < 0 or index >= len(self._visible_sessions):
            return
        session_id = str(self._visible_sessions[index].get("session_id", ""))
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
        self._sidebar_requested = not self._sidebar_requested
        self._apply_responsive_layout(self.size.width)

    def action_toggle_inspector(self) -> None:
        self._inspector_requested = not self._inspector_requested
        self._apply_responsive_layout(self.size.width)
        self._apply_sidebar_width()

    def action_sidebar_narrower(self) -> None:
        self._adjust_sidebar_width(-SIDEBAR_WIDTH_STEP)

    def action_sidebar_wider(self) -> None:
        self._adjust_sidebar_width(SIDEBAR_WIDTH_STEP)

    def action_show_help(self) -> None:
        self.action_command_palette()

    def action_command_palette(self) -> None:
        self.push_screen(
            CommandPaletteScreen(),
            self._palette_selected,
        )

    def action_toggle_mode(self) -> None:
        mode = "control"
        if self._workspace_mode == "control":
            mode = "focus"
        self._set_workspace_mode(mode)
        self.run_worker(
            self._save_ui_state(force=True),
            group="mode-ui-state",
            exclusive=True,
        )

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
        if not session_id:
            return
        self._remember_current_view()
        generation = self._switches.begin(session_id)
        self._switching_session_id = session_id
        self.screen.add_class("switching")
        self._render_connection_status()
        cached = self._switches.recall(session_id)
        if cached is not None:
            with self.batch_update():
                await self._apply_cached_session(cached)
        try:
            (
                state,
                session_state,
                events_state,
                turns_state,
                recovery_state,
            ) = await asyncio.gather(
                self.client.request(
                    "GET",
                    "/v1/sessions/" + session_id + "/ui-state",
                ),
                self.client.request(
                    "GET",
                    "/v1/sessions/" + session_id,
                ),
                self.client.request(
                    "GET",
                    "/v1/sessions/" + session_id + "/events?after=0",
                ),
                self.client.request(
                    "GET",
                    "/v1/sessions/" + session_id + "/turns?limit=200",
                ),
                self.client.request(
                    "GET",
                    "/v1/sessions/" + session_id + "/reconciliations",
                ),
            )
        except (HarnessError, OSError):
            if self._switches.is_current(generation, session_id):
                self._switching_session_id = ""
                self.screen.remove_class("switching")
                self._render_connection_status()
                self._push_local_notification(
                    "session-switch",
                    "Session unavailable",
                    severity=NotificationSeverity.WARNING,
                    persistence=NotificationPersistence.ACTIVITY,
                )
                self._render_notifications()
            return
        if not self._switches.is_current(generation, session_id):
            return
        ui_state = _object(state.get("ui_state"))
        if cached is not None:
            ui_state.update(self._cached_ui_state(cached))
        with self.batch_update():
            await self._apply_session_snapshot(
                session_id,
                ui_state,
                session_state,
                events_state,
                turns_state,
                recovery_state,
            )
            self._switching_session_id = ""
            self.screen.remove_class("switching")
            self._render_connection_status()
        await self._recover_pending_message()

    async def _apply_session_snapshot(
        self,
        session_id: str,
        ui_state: dict[str, Any],
        session_state: dict[str, Any],
        events_state: dict[str, Any],
        turns_state: dict[str, Any],
        recovery_state: dict[str, Any],
    ) -> None:
        if self._notification_timer is not None:
            self._notification_timer.stop()
            self._notification_timer = None
        self.session_id = session_id
        session = _object(session_state.get("session"))
        self._session = session
        self._goal = None
        goal = session_state.get("goal")
        if isinstance(goal, dict):
            self._goal = goal
        self._safety = _object(session_state.get("safety"))
        self._approvals = _object_tuple(session_state.get("approvals"))
        self._reconciliations = _object_tuple(recovery_state.get("reconciliations"))
        events = events_state.get("events", [])
        if not isinstance(events, list):
            events = []
        self._transcript_events = [item for item in events if isinstance(item, dict)]
        self.sequence = max(
            (int(item.get("sequence", 0)) for item in self._transcript_events),
            default=0,
        )
        turns = turns_state.get("turns", [])
        if not isinstance(turns, list):
            turns = []
        self._turns = [item for item in turns if isinstance(item, dict)]
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
        session_filter = str(ui_state.get("session_filter", "focused"))
        if session_filter not in {"focused", "all"}:
            session_filter = "focused"
        self._session_filter = session_filter
        self._session_query = str(ui_state.get("session_query", ""))
        self.query_one("#session-search", Input).value = self._session_query
        inspector_tab = str(ui_state.get("inspector_tab", "context")).casefold()
        if inspector_tab not in {
            "context",
            "goal",
            "usage",
            "approvals",
            "recovery",
            "storage",
        }:
            inspector_tab = "context"
        self._inspector_tab = inspector_tab
        self.query_one("#inspector-tabs", Tabs).active = inspector_tab
        mode = str(ui_state.get("workspace_mode", "focus")).casefold()
        if mode not in {"focus", "control"}:
            mode = "focus"
        self._workspace_mode = mode
        self._selected_turn_id = str(ui_state.get("selected_turn_id", ""))
        self._control_detail_tab = str(ui_state.get("control_detail_tab", "summary"))
        if self._control_detail_tab not in {
            "summary",
            "activity",
            "changes",
            "evidence",
            "recovery",
        }:
            self._control_detail_tab = "summary"
        self.query_one("#control-detail-tabs", Tabs).active = self._control_detail_tab
        self._saved_workspace_mode = self._workspace_mode
        self._saved_selected_turn_id = self._selected_turn_id
        self._show_events = str(ui_state.get("events", "off")).casefold() == "on"
        self._pending_request_id = str(ui_state.get("request_id", ""))
        self._pending_retry_count = 0
        self._saved_sidebar_width = self._sidebar_width
        self._saved_session_filter = self._session_filter
        self._saved_session_query = self._session_query
        self._saved_show_events = self._show_events
        self._apply_sidebar_width()
        await self._sync_system_theme(force=True)
        composer = str(ui_state.get("composer", ""))
        self._saved_composer = composer
        composer_cursor = str(ui_state.get("composer_cursor", "0:0"))
        self._saved_composer_cursor = composer_cursor
        self._saved_request_id = self._pending_request_id
        self.query_one("#composer", MultilineComposer).restore_draft(
            ComposerDraft(
                text=composer,
                cursor_row=_cursor_part(composer_cursor, 0),
                cursor_column=_cursor_part(composer_cursor, 1),
            )
        )
        self._last_approval_id = ""
        self._transcript_notices = []
        expanded = _expanded_blocks(str(ui_state.get("expanded_blocks", "")))
        initial = TranscriptState(
            expanded_block_ids=expanded,
        )
        update = project_events(
            initial,
            self._transcript_events,
            show_events=self._show_events,
        )
        self._transcript_state = update.state
        await self.query_one("#transcript", TranscriptView).reset(
            self._transcript_state,
            self._welcome_message(),
        )
        self._notification_state = project_notifications(
            NotificationState(),
            self._transcript_events,
        )
        acknowledged = _positive_integer(ui_state.get("last_notification_sequence"))
        if acknowledged is not None:
            self._notification_state = self._notification_state.acknowledge(
                acknowledged
            )
            self._notification_ack_sequence = acknowledged
            self._saved_notification_sequence = acknowledged
        else:
            self._notification_ack_sequence = 0
            self._saved_notification_sequence = 0
        self._set_workspace_mode(self._workspace_mode)
        await self._render_turn_list()
        self._render_selected_turn()
        self._render_notifications()
        self._render_session_header()
        self._render_inspector()
        self._render_composer_help()
        self._session_list_signature = ""
        await self._render_session_list()

    def _remember_current_view(self) -> None:
        if not self.session_id or not self._session or not self._screen_is_running():
            return
        composer = self.query_one("#composer", MultilineComposer)
        draft = composer.capture_draft()
        transcript = self.query_one("#transcript", TranscriptView)
        focus_id = ""
        if self.focused is not None:
            focus_id = self.focused.id or ""
        entry = SessionViewCacheEntry(
            session_id=self.session_id,
            transcript_scroll_y=float(transcript.scroll_y),
            focus_id=focus_id,
            expanded_block_ids=self._transcript_state.expanded_block_ids,
            composer=draft.text,
            composer_cursor=(str(draft.cursor_row) + ":" + str(draft.cursor_column)),
            workspace_mode=self._workspace_mode,
            selected_turn_id=self._selected_turn_id,
            detail_tab=self._control_detail_tab,
            revision=self.sequence,
            payload={
                "session_state": {
                    "session": dict(self._session),
                    "goal": self._goal,
                    "safety": dict(self._safety),
                    "approvals": list(self._approvals),
                },
                "events_state": {
                    "events": list(self._transcript_events),
                },
                "turns_state": {
                    "turns": list(self._turns),
                },
                "recovery_state": {
                    "reconciliations": list(self._reconciliations),
                },
                "theme": self._theme_preference,
                "provider": self._provider_override,
                "model": self._model_override,
                "effort": self._effort_override,
                "sidebar_width": self._sidebar_width,
                "session_filter": self._session_filter,
                "session_query": self._session_query,
                "inspector_tab": self._inspector_tab,
                "show_events": self._show_events,
                "request_id": self._pending_request_id,
            },
        )
        self._switches.remember(entry)

    async def _apply_cached_session(
        self,
        entry: SessionViewCacheEntry,
    ) -> None:
        payload = entry.payload
        await self._apply_session_snapshot(
            entry.session_id,
            self._cached_ui_state(entry),
            _object(payload.get("session_state")),
            _object(payload.get("events_state")),
            _object(payload.get("turns_state")),
            _object(payload.get("recovery_state")),
        )
        transcript = self.query_one("#transcript", TranscriptView)
        transcript.scroll_to(y=entry.transcript_scroll_y, animate=False)
        if entry.focus_id:
            try:
                self.query_one("#" + entry.focus_id).focus()
            except NoMatches:
                self.query_one("#composer", MultilineComposer).focus()

    def _cached_ui_state(
        self,
        entry: SessionViewCacheEntry,
    ) -> dict[str, Any]:
        payload = entry.payload
        ui_state: dict[str, Any] = {
            "composer": entry.composer,
            "composer_cursor": entry.composer_cursor,
            "workspace_mode": entry.workspace_mode,
            "selected_turn_id": entry.selected_turn_id,
            "control_detail_tab": entry.detail_tab,
            "expanded_blocks": json.dumps(sorted(entry.expanded_block_ids)),
            "theme": str(payload.get("theme", "system")),
            "provider": str(payload.get("provider", "")),
            "model": str(payload.get("model", "")),
            "effort": str(payload.get("effort", "")),
            "sidebar_width": str(payload.get("sidebar_width", "")),
            "session_filter": str(payload.get("session_filter", "focused")),
            "session_query": str(payload.get("session_query", "")),
            "inspector_tab": str(payload.get("inspector_tab", "context")),
            "events": "off",
            "request_id": str(payload.get("request_id", "")),
        }
        if bool(payload.get("show_events", False)):
            ui_state["events"] = "on"
        return ui_state

    def _palette_selected(self, command: str) -> None:
        if not command:
            return
        composer = self.query_one("#composer", MultilineComposer)
        composer.text = command
        composer.move_cursor((0, len(command)))
        composer.focus()

    def _set_workspace_mode(self, mode: str) -> None:
        if mode not in {"focus", "control"}:
            mode = "focus"
        self._workspace_mode = mode
        if mode == "control":
            self.screen.add_class("control")
        else:
            self.screen.remove_class("control")
        self.query_one("#mode-toggle", Button).label = mode.title()
        self._apply_responsive_layout(self.size.width)

    def _render_session_header(self) -> None:
        attention = str(self._session.get("attention", "idle"))
        lifecycle = str(self._session.get("lifecycle", "starting"))
        display_lifecycle = _display_lifecycle(lifecycle, attention)
        provider = str(self._session.get("active_provider", ""))
        if not provider:
            provider = "auto"
        model = str(self._session.get("model", ""))
        if not model:
            model = "default"
        effort = str(self._session.get("effort", ""))
        if not effort:
            effort = "auto"
        title = (
            _status_glyph(attention)
            + "  "
            + escape(str(self._session.get("name", "Untitled session")))
        )
        meta = (
            escape(display_lifecycle)
            + "  ·  "
            + escape(provider)
            + " / "
            + escape(model)
            + "  ·  "
            + escape(effort)
            + "  ·  "
            + escape(
                str(
                    self._session.get(
                        "permission_mode",
                        self.permission_mode,
                    )
                )
            )
        )
        self.query_one("#session-title", Static).update(title)
        self.query_one("#session-meta", Static).update(meta)
        self._render_connection_status()

    def _render_connection_status(self) -> None:
        attention = str(self._session.get("attention", "idle"))
        value = _connection_label(attention)
        status_class = "connected"
        if attention == "working":
            status_class = "working"
        elif attention in {"needs-input", "needs-reconciliation"}:
            status_class = "action"
        elif attention == "failed":
            status_class = "failed"
        if self._switching_session_id:
            value = "◌ switching"
            status_class = "switching"
        elif self._connection_status == "reconnecting":
            value = "◌ reconnecting"
            status_class = "reconnecting"
        elif self._connection_status == "send-unacknowledged":
            value = "◆ send pending"
            status_class = "pending"
        elif self._connection_status == "disconnected":
            value = "● disconnected"
            status_class = "disconnected"
        if self.screen.has_class("narrow"):
            value = _compact_connection_label(
                self._connection_status,
                attention,
            )
            if self._switching_session_id:
                value = "◌ switching"
        unread = self._notification_state.unread_count
        if unread:
            value += " · " + str(unread)
        connection = self.query_one("#connection-state", Static)
        connection.set_classes("status-" + status_class)
        connection.update(value)

    def _set_connection_state(self, value: str) -> None:
        previous = self._connection_status
        self._connection_status = value
        self._notification_state = with_connection(
            self._notification_state,
            previous,
            value,
        )
        self._render_notifications()
        self._render_connection_status()

    def _push_local_notification(
        self,
        key: str,
        title: str,
        *,
        detail: str = "",
        severity: NotificationSeverity = NotificationSeverity.INFO,
        persistence: NotificationPersistence = (NotificationPersistence.TRANSIENT),
    ) -> None:
        self._notification_state = push_notification(
            self._notification_state,
            Notification(
                key=key,
                title=title,
                detail=detail,
                severity=severity,
                persistence=persistence,
                source_sequence=self.sequence,
            ),
        )

    def _render_notifications(self) -> None:
        shell = self.query_one("#notification-shell", Horizontal)
        selected = self._notification_state.active_action
        if selected is None:
            selected = self._notification_state.latest_transient
        if selected is None:
            unread = [
                item for item in self._notification_state.notifications if item.unread
            ]
            if unread:
                selected = unread[-1]
        if selected is None:
            shell.display = False
            self._notification_actions = {}
            return
        shell.display = True
        shell.set_classes("notification-" + selected.severity.value)
        glyph = _notification_glyph(selected.severity)
        value = glyph + "  [bold]" + escape(selected.title) + "[/bold]"
        if selected.detail:
            value += "  [dim]" + escape(selected.detail) + "[/dim]"
        self.query_one("#notification-message", Static).update(value)
        self._notification_actions = {}
        buttons = (
            self.query_one("#notification-primary", Button),
            self.query_one("#notification-secondary", Button),
        )
        for button in buttons:
            button.display = False
        for index, action in enumerate(selected.actions[:2]):
            button = buttons[index]
            button.label = action.label
            button.display = True
            self._notification_actions[button.id or ""] = action.action_id
        if (
            not selected.actions
            and selected.persistence == NotificationPersistence.ACTIVITY
        ):
            button = buttons[0]
            button.label = "Dismiss"
            button.display = True
            self._notification_actions[button.id or ""] = "dismiss-notification"
        if selected.persistence == NotificationPersistence.TRANSIENT:
            if self._notification_timer is not None:
                self._notification_timer.stop()
            self._notification_timer = self.set_timer(
                3.0,
                lambda: self._dismiss_notification(selected.key),
            )

    def _dismiss_notification(self, key: str) -> None:
        for item in self._notification_state.notifications:
            if item.key == key:
                self._notification_ack_sequence = max(
                    self._notification_ack_sequence,
                    item.source_sequence,
                )
                break
        self._notification_state = self._notification_state.dismiss(key)
        self._render_notifications()
        self._render_connection_status()
        self.run_worker(
            self._save_ui_state(force=True),
            group="notification-ui-state",
            exclusive=True,
        )

    async def _handle_notification_action(self, button_id: str) -> None:
        action = self._notification_actions.get(button_id, "")
        active = self._notification_state.active_action
        selected = active
        if selected is None:
            selected = self._notification_state.latest_transient
        if selected is None:
            for item in reversed(self._notification_state.notifications):
                if item.unread:
                    selected = item
                    break
        if action == "review-approval":
            self._present_approval(force=True)
        elif action == "defer-approval":
            if active is not None:
                self._dismiss_notification(active.key)
        elif action == "review-recovery":
            self._set_workspace_mode("control")
            self._control_detail_tab = "recovery"
            self.query_one("#control-detail-tabs", Tabs).active = "recovery"
            self._render_selected_turn()
        elif action == "stop-session":
            await self._command("stop")
        elif action == "review-usage":
            self._inspector_tab = "usage"
            self._inspector_requested = True
            self._render_inspector()
            self._apply_responsive_layout(self.size.width)
        elif action == "retry-connection":
            await self._poll()
        elif action == "dismiss-notification":
            if selected is not None:
                self._dismiss_notification(selected.key)

    async def _render_turn_list(self) -> None:
        view = self.query_one("#turn-list", ListView)
        await view.clear()
        selected_index: int | None = None
        for index, turn in enumerate(self._turns):
            turn_id = str(turn.get("turn_id", ""))
            if turn_id == self._selected_turn_id:
                selected_index = index
            await view.append(
                ListItem(
                    Label(
                        _turn_list_label(turn),
                        markup=True,
                    )
                )
            )
        if self._turns and selected_index is None:
            selected_index = len(self._turns) - 1
            self._selected_turn_id = str(self._turns[selected_index].get("turn_id", ""))
        if selected_index is not None:
            view.index = selected_index

    async def _select_turn(self, index: int | None) -> None:
        if index is None or index < 0 or index >= len(self._turns):
            return
        turn = self._turns[index]
        self._selected_turn_id = str(turn.get("turn_id", ""))
        self._render_selected_turn()
        checkpoint_id = str(turn.get("checkpoint_id", ""))
        if checkpoint_id:
            selected_turn_id = self._selected_turn_id
            try:
                result = await self.client.request(
                    "GET",
                    "/v1/sessions/"
                    + self.session_id
                    + "/checkpoints/"
                    + checkpoint_id
                    + "/diff?limit=240",
                )
            except (HarnessError, OSError):
                return
            if selected_turn_id != self._selected_turn_id:
                return
            turn["_diff"] = _object(result.get("diff"))
            self._render_selected_turn()
        await self._save_ui_state(force=True)

    def _render_selected_turn(self) -> None:
        detail = self.query_one("#turn-detail", Static)
        for turn in self._turns:
            if str(turn.get("turn_id", "")) != self._selected_turn_id:
                continue
            detail.update(_turn_detail(turn, tab=self._control_detail_tab))
            return
        detail.update(
            "[bold]No turns yet[/bold]\n[dim]The first request will appear here.[/dim]"
        )

    async def _load_sessions(self) -> None:
        result = await self.client.request(
            "GET",
            "/v1/sessions?archived=1",
        )
        sessions = result.get("sessions", [])
        if not isinstance(sessions, list):
            sessions = []
        self._sessions = [item for item in sessions if isinstance(item, dict)]
        self._session_list_signature = ""
        await self._render_session_list()

    async def _render_session_list(self) -> None:
        visible, hidden_count = _visible_sessions(
            self._sessions,
            self.session_id,
            show_all=self._session_filter == "all",
            query=self._session_query,
        )
        signature = repr(
            (
                visible,
                hidden_count,
                self.session_id,
                self._session_filter,
                self._session_query,
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
        heading = "Sessions · " + self._session_filter
        if self._session_query:
            heading += " · search"
        self.query_one("#session-heading", Label).update(heading)
        help_text = str(len(visible)) + " shown"
        if hidden_count:
            help_text += " · " + str(hidden_count) + " hidden"
        self.query_one("#sessions-help", Static).update(help_text)

    async def _poll(self) -> None:
        if not self._screen_is_running() or not self.session_id:
            return
        poll_session_id = self.session_id
        poll_generation = self._switches.generation
        try:
            await self._save_ui_state()
            await self._recover_pending_message()
            state, result = await asyncio.gather(
                self.client.request(
                    "GET",
                    "/v1/sessions/" + poll_session_id,
                ),
                self.client.request(
                    "GET",
                    "/v1/sessions/"
                    + poll_session_id
                    + "/events?after="
                    + str(self.sequence),
                ),
            )
        except (HarnessError, OSError):
            self._set_connection_state("reconnecting")
            self.query_one("#connection-state", Static).update(
                "[yellow]◌ reconnecting[/yellow]"
            )
            self._render_composer_help()
            return
        if not self._switches.is_current(
            poll_generation,
            poll_session_id,
        ):
            return
        raw_events = result.get("events", [])
        should_refresh_turns = self._workspace_mode == "control"
        if isinstance(raw_events, list) and raw_events:
            should_refresh_turns = True
        turns_state: dict[str, Any] = {"turns": self._turns}
        if should_refresh_turns:
            try:
                turns_state = await self.client.request(
                    "GET",
                    "/v1/sessions/" + poll_session_id + "/turns?limit=200",
                )
            except (HarnessError, OSError):
                turns_state = {"turns": self._turns}
        if not self._switches.is_current(
            poll_generation,
            poll_session_id,
        ):
            return
        self._set_connection_state("connected")
        if not self._screen_is_running():
            return
        session = _object(state.get("session"))
        self._session = session
        await self._synchronize_active_session(session)
        self._safety = _object(state.get("safety"))
        goal = state.get("goal")
        self._goal = None
        if isinstance(goal, dict):
            self._goal = goal
        self._approvals = _object_tuple(state.get("approvals"))
        if (
            str(session.get("attention", "")) == "needs-reconciliation"
            or self._inspector_tab == "recovery"
        ):
            try:
                recovery = await self.client.request(
                    "GET",
                    "/v1/sessions/" + self.session_id + "/reconciliations",
                )
                self._reconciliations = _object_tuple(recovery.get("reconciliations"))
            except (HarnessError, OSError):
                self._reconciliations = ()
        else:
            self._reconciliations = ()
        self._provider_poll += 1
        provider_limit = 20
        if _providers_refreshing(self._providers):
            provider_limit = 4
        if not self._providers or self._provider_poll >= provider_limit:
            await self._load_providers()
            self._provider_poll = 0
        self._sync_poll += 1
        if self._sync_poll >= 20:
            await self._load_sync()
            self._sync_poll = 0
        self._theme_poll += 1
        if self._theme_poll >= 20:
            await self._sync_system_theme()
            self._theme_poll = 0
        self._render_session_header()
        self._render_inspector()
        self._render_composer_help()
        events = result.get("events", [])
        if not isinstance(events, list):
            return
        transcript_changed = False
        new_events: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            sequence = int(event.get("sequence", 0))
            self.sequence = max(self.sequence, sequence)
            self._transcript_events.append(event)
            new_events.append(event)
            transcript_changed = True
        if transcript_changed:
            await self._apply_transcript_events(new_events)
            self._notification_state = project_notifications(
                self._notification_state,
                new_events,
            )
            self._render_notifications()
        turns = turns_state.get("turns", [])
        if isinstance(turns, list):
            next_turns = [item for item in turns if isinstance(item, dict)]
            if next_turns != self._turns:
                self._turns = next_turns
                await self._render_turn_list()
                self._render_selected_turn()
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
            "/v1/sessions/" + self.session_id + "/commands/" + command_type,
            payload=payload,
            idempotency_key=new_uuid(),
        )

    def _message_payload(self, text: str) -> dict[str, Any]:
        return {
            "text": text,
            "provider": self._provider_override,
            "model": self._model_override,
            "effort": self._effort_override,
        }

    async def _submit_pending_message(self, text: str) -> None:
        request_id = self._pending_request_id
        if not request_id:
            return
        try:
            await self.client.request(
                "POST",
                "/v1/sessions/" + self.session_id + "/messages",
                payload=self._message_payload(text),
                idempotency_key=request_id,
            )
        except (HarnessError, OSError):
            self._pending_retry_count += 1
            self._set_connection_state("reconnecting")
            self._render_composer_help()
            return
        composer = self.query_one("#composer", MultilineComposer)
        composer.text = ""
        self._pending_request_id = ""
        self._pending_retry_count = 0
        self._set_connection_state("connected")
        await self._save_ui_state(force=True)
        await self._poll()

    async def _recover_pending_message(self) -> None:
        if not self._pending_request_id:
            return
        if self._recovering_send:
            return
        if self._pending_retry_count >= 5:
            self._set_connection_state("send-unacknowledged")
            self._render_composer_help()
            return
        composer = self.query_one("#composer", MultilineComposer)
        text = composer.text.strip()
        if not text:
            self._pending_request_id = ""
            await self._save_ui_state(force=True)
            return
        self._recovering_send = True
        try:
            await self._submit_pending_message(text)
        finally:
            self._recovering_send = False

    def _render_composer_help(self) -> None:
        permission = str(self._session.get("permission_mode", self.permission_mode))
        safety_session = _object(self._safety.get("session"))
        profile = str(safety_session.get("profile", "interactive"))
        connection = self._connection_status
        provider = self._provider_override or "auto"
        effort = self._effort_override or "auto"
        value = (
            provider
            + " · "
            + effort
            + " · "
            + permission
            + " · "
            + profile
            + " · "
            + connection
        )
        if self._pending_request_id:
            value += " · send pending"
        self.query_one("#composer-help", Static).update(value)

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
            self._write_notice("[bold]Export[/bold] " + str(result.get("path", "")))
            return
        if command == "/checkpoint":
            await self._checkpoint()
            return
        if command in {"/archive", "/unarchive"}:
            action = command[1:]
            await self.client.request(
                "POST",
                "/v1/sessions/" + self.session_id + "/" + action,
                payload={},
                idempotency_key=new_uuid(),
            )
            await self._load_sessions()
            return
        if command == "/rename":
            name = " ".join(parts[1:]).strip()
            await self.client.request(
                "PATCH",
                "/v1/sessions/" + self.session_id,
                payload={"name": name},
            )
            await self._poll()
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
            await self._open_session(str(session.get("session_id", "")))
            return
        if command == "/sessions" and len(parts) == 2:
            session_filter = parts[1].casefold()
            if session_filter not in {"focused", "all"}:
                self._write_notice("[red]Sessions must be focused or all[/red]")
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
        if command == "/mode" and len(parts) == 2:
            mode = parts[1].casefold()
            if mode not in {"focus", "control"}:
                self._write_notice("[red]Mode must be focus or control[/red]")
                return
            self._set_workspace_mode(mode)
            await self._save_ui_state(force=True)
            return
        if command in {"/provider", "/model", "/effort", "/theme"}:
            if len(parts) != 2:
                self._write_notice("[red]" + command + " requires one value[/red]")
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
                if len(parts) < 5:
                    self._write_notice(
                        "[red]Use /budget xhigh COMMAND_ID PROVIDER REASON[/red]"
                    )
                    return
                payload["allow_xhigh_once"] = True
                payload["command_id"] = parts[2]
                payload["provider"] = parts[3]
                payload["reason"] = " ".join(parts[4:])
            elif parts[1] == "extend":
                try:
                    payload["additional_seconds"] = int(parts[2])
                    if len(parts) >= 4:
                        payload["additional_tokens"] = int(parts[3])
                        payload["reason"] = (
                            " ".join(parts[4:]) or "TUI operator extension"
                        )
                except ValueError:
                    self._write_notice("[red]Budget values must be integers[/red]")
                    return
            else:
                self._write_notice("[red]Use /budget extend or /budget xhigh[/red]")
                return
            await self.client.request(
                "POST",
                "/v1/sessions/" + self.session_id + "/budget-extensions",
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
                "/v1/sessions/" + self.session_id + "/approvals/" + parts[1],
                payload={"decision": parts[2]},
            )
            return
        if command == "/reconcile" and len(parts) >= 4:
            payload: dict[str, Any] = {
                "decision": parts[1],
                "observed_workspace_digest": parts[3],
                "audit": {"surface": "tui"},
            }
            if len(parts) == 5:
                payload["approval_id"] = parts[4]
            await self.client.request(
                "POST",
                "/v1/reconciliations/" + parts[2] + "/resolution",
                payload=payload,
                idempotency_key=new_uuid(),
            )
            await self._poll()
            return
        self._write_notice("[red]Unknown action[/red]")

    async def _checkpoint(self) -> None:
        result = await self.client.request(
            "POST",
            "/v1/sessions/" + self.session_id + "/checkpoints",
            payload={},
        )
        checkpoint = _object(result.get("checkpoint"))
        del checkpoint
        self._push_local_notification(
            "checkpoint-local",
            "Checkpoint created",
            severity=NotificationSeverity.SUCCESS,
        )
        self._render_notifications()

    async def _save_ui_state(self, *, force: bool = False) -> None:
        if not self.session_id:
            return
        draft = self.query_one(
            "#composer",
            MultilineComposer,
        ).capture_draft()
        composer = draft.text
        composer_cursor = str(draft.cursor_row) + ":" + str(draft.cursor_column)
        if not force:
            unchanged = (
                composer == self._saved_composer
                and composer_cursor == self._saved_composer_cursor
                and self._pending_request_id == self._saved_request_id
                and self._sidebar_width == self._saved_sidebar_width
                and self._session_filter == self._saved_session_filter
                and self._session_query == self._saved_session_query
                and self._show_events == self._saved_show_events
                and self._workspace_mode == self._saved_workspace_mode
                and self._selected_turn_id == self._saved_selected_turn_id
                and self._notification_ack_sequence == self._saved_notification_sequence
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
                "composer_cursor": composer_cursor,
                "provider": self._provider_override,
                "model": self._model_override,
                "effort": self._effort_override,
                "theme": self._theme_preference,
                "sidebar_width": str(self._sidebar_width),
                "session_filter": self._session_filter,
                "session_query": self._session_query,
                "events": events_value,
                "inspector_tab": self._inspector_tab,
                "workspace_mode": self._workspace_mode,
                "selected_turn_id": self._selected_turn_id,
                "control_detail_tab": self._control_detail_tab,
                "last_notification_sequence": str(self._notification_ack_sequence),
                "expanded_blocks": json.dumps(
                    sorted(self._transcript_state.expanded_block_ids),
                    separators=(",", ":"),
                ),
                "request_id": self._pending_request_id,
            },
        )
        self._saved_composer = composer
        self._saved_composer_cursor = composer_cursor
        self._saved_request_id = self._pending_request_id
        self._saved_sidebar_width = self._sidebar_width
        self._saved_session_filter = self._session_filter
        self._saved_session_query = self._session_query
        self._saved_show_events = self._show_events
        self._saved_workspace_mode = self._workspace_mode
        self._saved_selected_turn_id = self._selected_turn_id
        self._saved_notification_sequence = self._notification_ack_sequence

    async def _load_providers(self) -> None:
        try:
            result = await self.client.request(
                "GET",
                "/v1/providers?workspace=" + str(self.workspace),
            )
        except BaseException:
            return
        self._providers = _object(result.get("providers"))

    async def _load_sync(self) -> None:
        try:
            result = await self.client.request("GET", "/v1/sync")
        except BaseException:
            return
        self._sync = _object(result.get("sync"))
        self._state_root = str(result.get("state_root", ""))

    def _render_inspector(self) -> None:
        if self._inspector_tab == "goal":
            self._render_inspector_lines(self._goal_lines())
            return
        if self._inspector_tab == "usage":
            self._render_inspector_lines(self._usage_lines())
            return
        if self._inspector_tab == "approvals":
            self._render_inspector_lines(self._approval_lines())
            return
        if self._inspector_tab == "recovery":
            self._render_inspector_lines(self._recovery_lines())
            return
        if self._inspector_tab == "storage":
            self._render_inspector_lines(self._storage_lines())
            return
        lines = [
            "[bold cyan]ROUTING[/bold cyan]",
            "Provider   " + escape(self._provider_override or "automatic"),
            "Model      " + escape(self._model_override or "automatic"),
            "Effort     " + escape(self._effort_override or "automatic"),
            "Permission " + escape(str(self._session.get("permission_mode", ""))),
            "Theme      " + escape(self._theme_preference),
            "",
            "[bold cyan]CHAT STORAGE[/bold cyan]",
            "State      " + escape(str(self._sync.get("state", "unknown"))),
            "Root       " + escape(self._state_root or "unknown"),
        ]
        sync_error = str(self._sync.get("error", ""))
        if sync_error:
            lines.append("[bold red]Sync       " + escape(sync_error) + "[/bold red]")
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
                "Profile    " + escape(str(safety_session.get("profile", "unknown"))),
            ]
        )
        if envelopes:
            envelope = _object(envelopes[-1])
            consumption = _object(envelope.get("consumption"))
            limits = _object(envelope.get("limits"))
            accounting = "estimated"
            if bool(consumption.get("exact_tokens", False)):
                accounting = "provider-reported"
            lines.append("State      " + escape(str(envelope.get("state", ""))))
            if "input_tokens" in consumption:
                lines.append(
                    "Input      " + _format_number(consumption.get("input_tokens", 0))
                )
                cached_tokens = int(consumption.get("cached_input_tokens", 0))
                if cached_tokens:
                    lines[-1] += " · " + _format_number(cached_tokens) + " cached"
            lines.extend(
                [
                    "Output     " + _format_number(consumption.get("output_tokens", 0)),
                    "Harness    "
                    + _format_number(consumption.get("context_tokens", 0))
                    + " context est.",
                    "Total      "
                    + _format_number(consumption.get("total_tokens", 0))
                    + " / "
                    + _format_number(limits.get("max_total_tokens", 0)),
                    "Time       "
                    + escape(str(round(float(consumption.get("elapsed_seconds", 0)))))
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
                    "[bold red]Guard      " + escape(guard_reason) + "[/bold red]"
                )
        lines.extend(["", "[bold cyan]PENDING APPROVALS[/bold cyan]"])
        if not self._approvals:
            lines.append("[dim]None[/dim]")
        for approval in self._approvals:
            lines.append(
                escape(str(approval.get("approval_id", "")))
                + "\n"
                + escape(str(approval.get("prompt", "")))
            )
        lines.extend(["", "[bold cyan]PROVIDER CAPACITY[/bold cyan]"])
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
                detail += " · " + escape(str(100 - float(binding))) + "% headroom"
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
        self.query_one("#inspector-content", Static).update("\n".join(lines))

    def _render_inspector_lines(self, lines: list[str]) -> None:
        self.query_one("#inspector-content", Static).update("\n".join(lines))

    def _goal_lines(self) -> list[str]:
        lines = ["[bold cyan]GOAL[/bold cyan]"]
        if self._goal is None:
            lines.append("[dim]No goal is attached.[/dim]")
            return lines
        lines.extend(
            [
                escape(str(self._goal.get("kind", "")))
                + " · "
                + escape(str(self._goal.get("status", ""))),
                "",
                escape(str(self._goal.get("objective", ""))),
            ]
        )
        return lines

    def _usage_lines(self) -> list[str]:
        lines = ["[bold cyan]USAGE AND SAFETY[/bold cyan]"]
        safety_session = _object(self._safety.get("session"))
        lines.append(
            "Profile  " + escape(str(safety_session.get("profile", "unknown")))
        )
        envelopes = self._safety.get("envelopes", [])
        if not isinstance(envelopes, list) or not envelopes:
            lines.append("[dim]No model turn has consumed capacity.[/dim]")
        else:
            envelope = _object(envelopes[-1])
            consumption = _object(envelope.get("consumption"))
            limits = _object(envelope.get("limits"))
            lines.extend(
                [
                    "State    " + escape(str(envelope.get("state", ""))),
                    "Tokens   "
                    + _format_number(consumption.get("total_tokens", 0))
                    + " / "
                    + _format_number(limits.get("max_total_tokens", 0)),
                    "Tools    "
                    + escape(str(consumption.get("tool_calls", 0)))
                    + " / "
                    + escape(str(limits.get("max_tool_calls", 0))),
                ]
            )
        lines.extend(["", "[bold cyan]PROVIDERS[/bold cyan]"])
        for provider, value in sorted(self._providers.items()):
            provider_value = _object(value)
            state = "unavailable"
            if bool(provider_value.get("ready", False)):
                state = "ready"
            lines.append(escape(provider + " · " + state))
        if not self._providers:
            lines.append("[dim]Capacity refresh is pending.[/dim]")
        return lines

    def _approval_lines(self) -> list[str]:
        lines = ["[bold cyan]APPROVALS[/bold cyan]"]
        if not self._approvals:
            lines.append("[dim]No pending approvals.[/dim]")
            return lines
        for approval in self._approvals:
            lines.extend(
                [
                    escape(str(approval.get("kind", "approval"))),
                    escape(str(approval.get("prompt", ""))),
                    "[dim]" + escape(str(approval.get("approval_id", ""))) + "[/dim]",
                    "",
                ]
            )
        return lines

    def _recovery_lines(self) -> list[str]:
        lines = ["[bold #fb923c]RECOVERY[/bold #fb923c]"]
        if not self._reconciliations:
            lines.append("[dim]No reconciliation barrier.[/dim]")
            return lines
        for record in self._reconciliations:
            lines.extend(
                [
                    "[bold]Interrupted command[/bold] "
                    + escape(str(record.get("command_id", ""))),
                    "Checkpoint "
                    + escape(
                        str(
                            record.get(
                                "pre_dispatch_checkpoint_id",
                                "",
                            )
                        )
                    ),
                    "Workspace  "
                    + escape(
                        str(
                            record.get(
                                "current_workspace_summary",
                                "",
                            )
                        )
                    ),
                    "Digest     "
                    + escape(
                        str(
                            record.get(
                                "current_workspace_digest",
                                "",
                            )
                        )
                    ),
                    "",
                    "[bold]Actions[/bold]",
                    "accept-current · restore-pre-turn · stop",
                    "[dim]/reconcile <decision> <id> <digest> [approval-id][/dim]",
                ]
            )
        return lines

    def _storage_lines(self) -> list[str]:
        external_ref = _object(self._session.get("external_ref"))
        lines = [
            "[bold cyan]STORAGE[/bold cyan]",
            "Session " + escape(str(self._session.get("session_id", ""))),
            "State   " + escape(str(self._sync.get("state", "unknown"))),
            "Root    " + escape(self._state_root or "unknown"),
            "Worktree " + escape(str(self._session.get("worktree", ""))),
        ]
        if external_ref:
            lines.extend(
                [
                    "",
                    "[bold]External job[/bold]",
                    escape(
                        str(external_ref.get("orchestrator", ""))
                        + " / "
                        + str(external_ref.get("job_id", ""))
                    ),
                ]
            )
        return lines

    def _write_help(self) -> None:
        self.action_command_palette()

    def _welcome_message(self) -> str:
        return (
            "[bold]Durable session ready[/bold]\n"
            "Conversation, activity, and unfinished drafts persist "
            "with this session.\n"
        )

    def _apply_responsive_layout(self, width: int) -> None:
        self._layout = decide_layout(
            width,
            self.size.height,
            sidebar_requested=self._sidebar_requested,
            inspector_requested=self._inspector_requested,
        )
        if self._layout.mode.value in {
            "minimal",
            "overlay",
            "compact",
        }:
            self.screen.add_class("compact")
        else:
            self.screen.remove_class("compact")
        if self._layout.mode.value == "minimal":
            self.screen.add_class("narrow")
            self.query_one("#brand", Static).update("P13I")
        else:
            self.screen.remove_class("narrow")
            self.query_one("#brand", Static).update("P13I AGENT HARNESS")
        if self._layout.mode.value == "overlay":
            self.screen.add_class("overlay")
        else:
            self.screen.remove_class("overlay")
        sidebar = self.query_one("#sidebar", Vertical)
        handle = self.query_one(
            "#sidebar-resize-handle",
            SidebarResizeHandle,
        )
        inspector = self.query_one("#inspector", Vertical)
        sidebar.display = self._layout.sidebar_visible
        handle.display = (
            self._layout.sidebar_visible and self._layout.sidebar_mode == "docked"
        )
        inspector.display = (
            self._layout.inspector_visible and self._workspace_mode == "focus"
        )
        composer = self.query_one("#composer", MultilineComposer)
        composer.max_lines = self._layout.composer_max_lines
        composer.refresh_auto_height()
        transcript = self.query_one("#transcript", TranscriptView)
        padding = self._layout.transcript_horizontal_padding
        transcript.styles.padding = (1, padding)

    def _apply_sidebar_width(self) -> None:
        sidebar = self.query_one("#sidebar", Vertical)
        sidebar.styles.width = self._bounded_sidebar_width(self._sidebar_width)

    def _bounded_sidebar_width(self, width: int) -> int:
        reserve = SIDEBAR_MAIN_RESERVE
        if (
            self.size.width >= 118
            and self._inspector_requested
            and self._workspace_mode == "focus"
        ):
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

    def _present_approval(self, *, force: bool = False) -> None:
        if not force:
            return
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
            "/v1/sessions/" + self.session_id + "/approvals/" + approval_id,
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
        workspace = str(self._session.get("worktree", self.workspace))
        with self.suspend():
            await asyncio.to_thread(
                subprocess.run,
                command,
                cwd=workspace,
                check=False,
            )
        await self._poll()

    async def _apply_transcript_events(
        self,
        values: list[dict[str, Any]],
    ) -> None:
        transcript = self.query_one("#transcript", TranscriptView)
        reader_at_bottom = transcript.is_vertical_scroll_end
        self._transcript_state = self._transcript_state.with_reader_at_bottom(
            reader_at_bottom
        )
        update = project_events(
            self._transcript_state,
            values,
            show_events=self._show_events,
        )
        self._transcript_state = update.state
        await transcript.apply(update.state, update.mutations)
        self._render_activity_marker()

    def _render_transcript(self) -> None:
        self.run_worker(
            self._rebuild_transcript(),
            group="transcript-rebuild",
            exclusive=True,
        )

    async def _rebuild_transcript(self) -> None:
        expanded = self._transcript_state.expanded_block_ids
        reader_at_bottom = self._transcript_state.reader_at_bottom
        state = TranscriptState(
            expanded_block_ids=expanded,
            reader_at_bottom=reader_at_bottom,
        )
        update = project_events(
            state,
            self._transcript_events,
            show_events=self._show_events,
        )
        self._transcript_state = update.state
        transcript = self.query_one("#transcript", TranscriptView)
        await transcript.reset(
            self._transcript_state,
            self._welcome_message(),
        )
        for notice in self._transcript_notices:
            await transcript.mount(
                Static(
                    notice,
                    classes="transcript-block transcript-system",
                    markup=True,
                )
            )
        self._render_activity_marker()

    def _write_notice(self, value: str) -> None:
        plain = _plain_markup(value)
        lines = [line.strip() for line in plain.splitlines() if line.strip()]
        if not lines:
            return
        severity = NotificationSeverity.INFO
        if "[red" in value or "failed" in plain.casefold():
            severity = NotificationSeverity.WARNING
        elif "[yellow" in value:
            severity = NotificationSeverity.WARNING
        title = lines[0][:96]
        detail = " · ".join(lines[1:])[:240]
        self._push_local_notification(
            "local:" + title.casefold(),
            title,
            detail=detail,
            severity=severity,
        )
        self._render_notifications()

    def toggle_transcript_block(self, block_id: str) -> None:
        self._transcript_state = self._transcript_state.toggle_expanded(block_id)
        block = self._transcript_state.block(block_id)
        if block is None:
            return
        self.query_one("#transcript", TranscriptView).refresh_block(
            block,
            expanded=(block_id in self._transcript_state.expanded_block_ids),
        )
        self.run_worker(
            self._save_ui_state(force=True),
            group="transcript-ui-state",
            exclusive=True,
        )

    def _render_activity_marker(self) -> None:
        marker = self.query_one("#new-activity", Static)
        count = self._transcript_state.new_activity_count
        marker.display = count > 0
        if count > 0:
            marker.update(str(count) + " new transcript updates")


def _render_transcript_events(
    events: list[dict[str, Any]],
    *,
    show_events: bool,
) -> list[str]:
    final_turns = {
        str(event.get("turn_id", ""))
        for event in events
        if event.get("event_type") == "agent.message" and str(event.get("turn_id", ""))
    }
    live_deltas: dict[str, list[str]] = {}
    for event in events:
        if event.get("event_type") != "agent.message.delta":
            continue
        turn_id = str(event.get("turn_id", ""))
        if turn_id in final_turns:
            continue
        live_deltas.setdefault(turn_id, []).append(str(event.get("text", "")))
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


def _transcript_block_classes(block: TranscriptBlock) -> str:
    return (
        "transcript-block transcript-"
        + block.kind.value
        + " transcript-status-"
        + block.status.value
    )


def _render_transcript_block(
    block: TranscriptBlock,
    *,
    expanded: bool,
) -> str:
    accent = "white"
    if block.kind == TranscriptBlockKind.USER:
        accent = "cyan"
    elif block.kind == TranscriptBlockKind.ASSISTANT:
        accent = "green"
    elif block.kind == TranscriptBlockKind.TOOL:
        accent = "yellow"
    elif block.kind == TranscriptBlockKind.APPROVAL:
        accent = "magenta"
    elif block.kind == TranscriptBlockKind.RECONCILIATION:
        accent = "#fb923c"
    elif block.kind == TranscriptBlockKind.WARNING:
        accent = "red"
    value = (
        "[bold " + accent + "]" + escape(block.title.upper()) + "[/bold " + accent + "]"
    )
    if block.status.value not in {"complete", "waiting"}:
        value += "  [dim]" + escape(block.status.value) + "[/dim]"
    if block.content:
        value += "\n" + escape(block.content)
    if block.detail:
        detail = block.detail
        if not expanded and len(detail) > 320:
            detail = detail[:320] + "…"
        value += "\n[dim]" + escape(detail) + "[/dim]"
        if block.kind == TranscriptBlockKind.TOOL and not expanded:
            value += "\n[dim]Details collapsed[/dim]"
    return value


def _cursor_part(value: str, index: int) -> int:
    parts = value.split(":", 1)
    if len(parts) != 2:
        return 0
    try:
        result = int(parts[index])
    except ValueError:
        return 0
    return max(0, result)


def _expanded_blocks(value: str) -> frozenset[str]:
    if not value:
        return frozenset()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return frozenset()
    if not isinstance(parsed, list):
        return frozenset()
    return frozenset(str(item) for item in parsed if isinstance(item, str))


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
            "\n[bold yellow]TOOL · " + escape(label) + "[/bold yellow]\n" + text + "\n"
        )
    if event_type.startswith("reasoning.summary"):
        return "[dim]" + text + "[/dim]"
    if event_type == "approval.requested":
        metadata = _object(event.get("metadata"))
        return "[bold magenta]Approval required[/bold magenta] " + escape(
            str(metadata.get("approval_id", ""))
        )
    if event_type == "guard.warning":
        return "[bold yellow]Safety envelope is above 80%[/bold yellow]"
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
        return "[dim]Routed to " + escape(provider) + " · " + escape(model) + "[/dim]"
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


def _notification_glyph(severity: NotificationSeverity) -> str:
    if severity == NotificationSeverity.SUCCESS:
        return "[green]●[/green]"
    if severity == NotificationSeverity.WARNING:
        return "[yellow]◆[/yellow]"
    if severity == NotificationSeverity.DANGER:
        return "[red]■[/red]"
    if severity == NotificationSeverity.ACTION:
        return "[magenta]◆[/magenta]"
    return "[cyan]●[/cyan]"


def _turn_list_label(turn: dict[str, Any]) -> str:
    status = str(turn.get("status", "unknown"))
    glyph = "○"
    color = "dim"
    if status in {"complete", "completed"}:
        glyph = "●"
        color = "green"
    elif status in {"running", "dispatching"}:
        glyph = "◌"
        color = "cyan"
    elif status in {"failed", "cancelled", "interrupted"}:
        glyph = "◆"
        color = "red"
    turn_ref = _object(turn.get("turn_ref"))
    role = str(turn_ref.get("agent_role", "turn"))
    attempts = turn.get("attempts", [])
    provider = "automatic"
    effort = ""
    if isinstance(attempts, list) and attempts:
        attempt = _object(attempts[-1])
        provider = str(attempt.get("provider", provider))
        effort = str(attempt.get("effort", ""))
    route = provider
    if effort:
        route += " · " + effort
    return (
        "["
        + color
        + "]"
        + glyph
        + "[/"
        + color
        + "]  [bold]"
        + escape(role)
        + "[/bold]\n[dim]"
        + escape(status)
        + " · "
        + escape(route)
        + "[/dim]"
    )


def _turn_detail(
    turn: dict[str, Any],
    *,
    tab: str = "summary",
) -> str:
    if tab == "activity":
        return _turn_activity_detail(turn)
    if tab == "changes":
        return _turn_changes_detail(turn)
    if tab == "evidence":
        return _turn_evidence_detail(turn)
    if tab == "recovery":
        return _turn_recovery_detail(turn)
    return _turn_summary_detail(turn)


def _turn_summary_detail(turn: dict[str, Any]) -> str:
    lines = [
        "[bold]Turn[/bold]",
        "[dim]" + escape(str(turn.get("turn_id", ""))) + "[/dim]",
        "",
    ]
    request = str(turn.get("request", "")).strip()
    if request:
        lines.extend(
            [
                "[bold cyan]Request[/bold cyan]",
                escape(request),
                "",
            ]
        )
    lines.append("[bold]Attempts[/bold]")
    attempts = turn.get("attempts", [])
    if isinstance(attempts, list):
        for attempt_value in attempts:
            attempt = _object(attempt_value)
            lines.append(
                escape(str(attempt.get("provider", "provider")))
                + " · "
                + escape(str(attempt.get("model", "default")))
                + " · "
                + escape(str(attempt.get("effort", "auto")))
                + " · "
                + escape(str(attempt.get("status", "unknown")))
            )
    safety = _object(turn.get("safety"))
    consumption = _object(safety.get("consumption"))
    if consumption:
        lines.extend(
            [
                "",
                "[bold]Usage[/bold]",
                _format_number(consumption.get("total_tokens", 0))
                + " tokens · "
                + escape(str(consumption.get("tool_calls", 0)))
                + " tools",
            ]
        )
    result = _object(turn.get("result"))
    if result:
        lines.extend(
            [
                "",
                "[bold]Result[/bold]",
                escape(str(result.get("status", turn.get("status", "")))),
            ]
        )
    return "\n".join(lines)


def _turn_activity_detail(turn: dict[str, Any]) -> str:
    lines = ["[bold]Activity[/bold]"]
    activity = turn.get("activity", [])
    if not isinstance(activity, list) or not activity:
        lines.append("[dim]No recorded activity.[/dim]")
        return "\n".join(lines)
    for value in activity:
        event = _object(value)
        event_type = str(event.get("event_type", "activity"))
        status = str(event.get("status", ""))
        label = escape(event_type)
        if status:
            label += "  [dim]" + escape(status) + "[/dim]"
        lines.extend(["", label])
        text = str(event.get("text", "")).strip()
        if text:
            lines.append(escape(text[:2000]))
    return "\n".join(lines)


def _turn_changes_detail(turn: dict[str, Any]) -> str:
    lines = ["[bold]Changes[/bold]"]
    diff = _object(turn.get("_diff"))
    if not diff:
        if turn.get("checkpoint_id"):
            lines.append("[dim]Loading checkpoint changes…[/dim]")
        else:
            lines.append("[dim]No checkpoint for this turn.[/dim]")
        return "\n".join(lines)
    files = diff.get("changed_files", [])
    if isinstance(files, list) and files:
        lines.extend(
            [
                "[dim]" + str(len(files)) + " changed file(s)[/dim]",
                "",
            ]
        )
        for path in files:
            lines.append("• " + escape(str(path)))
        lines.append("")
    content = str(diff.get("content", ""))
    if content:
        lines.append(escape(content))
    if bool(diff.get("binary", False)):
        lines.append("[dim]Binary bodies omitted[/dim]")
    if bool(diff.get("truncated", False)):
        lines.append("[dim]Diff continues[/dim]")
    redactions = int(diff.get("redactions", 0))
    if redactions:
        lines.append("[dim]" + str(redactions) + " sensitive values hidden[/dim]")
    return "\n".join(lines)


def _turn_evidence_detail(turn: dict[str, Any]) -> str:
    lines = ["[bold]Evidence[/bold]"]
    evidence = turn.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        lines.append("[dim]No evidence attached.[/dim]")
        return "\n".join(lines)
    for value in evidence:
        item = _object(value)
        event_type = str(item.get("event_type", "evidence"))
        status = str(item.get("status", ""))
        lines.extend(
            [
                "",
                escape(event_type) + "  [dim]" + escape(status) + "[/dim]",
            ]
        )
        text = str(item.get("text", "")).strip()
        if text:
            lines.append(escape(text[:2000]))
    return "\n".join(lines)


def _turn_recovery_detail(turn: dict[str, Any]) -> str:
    lines = ["[bold #fb923c]Recovery[/bold #fb923c]"]
    reconciliation = _object(turn.get("reconciliation"))
    if not reconciliation:
        lines.append("[dim]No recovery barrier.[/dim]")
        return "\n".join(lines)
    status = str(reconciliation.get("status", "pending"))
    lines.extend(["", escape(status)])
    summary = str(reconciliation.get("current_workspace_summary", "")).strip()
    if summary:
        lines.append(escape(summary))
    reconciliation_id = str(reconciliation.get("reconciliation_id", ""))
    if reconciliation_id:
        lines.extend(
            [
                "",
                "[dim]" + escape(reconciliation_id) + "[/dim]",
            ]
        )
    return "\n".join(lines)


def _plain_markup(value: str) -> str:
    return re.sub(r"\[/?[^\]]+\]", "", value)


def _visible_sessions(
    sessions: list[dict[str, Any]],
    active_session_id: str,
    *,
    show_all: bool,
    query: str = "",
) -> tuple[list[dict[str, Any]], int]:
    normalized_query = query.strip().casefold()
    candidates = list(sessions)
    if normalized_query:
        candidates = [
            session
            for session in sessions
            if normalized_query in _session_search_text(session)
        ]
    if show_all:
        return (candidates, len(sessions) - len(candidates))
    visible: list[dict[str, Any]] = []
    idle_sessions = 0
    for session in candidates:
        session_id = str(session.get("session_id", ""))
        if bool(session.get("archived", False)):
            if normalized_query:
                visible.append(session)
            continue
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
    if bool(session.get("archived", False)):
        detail = marker + escape(name) + "\n  archived"
    external_ref = _object(session.get("external_ref"))
    if external_ref:
        detail += " · external job"
    if provider:
        detail += " · " + escape(provider)
    return detail


def _session_search_text(session: dict[str, Any]) -> str:
    external_ref = _object(session.get("external_ref"))
    return " ".join(
        (
            str(session.get("session_id", "")),
            str(session.get("name", "")),
            str(session.get("active_provider", "")),
            str(external_ref.get("orchestrator", "")),
            str(external_ref.get("job_id", "")),
        )
    ).casefold()


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
        return "◉"
    if attention in {"needs-input", "needs-reconciliation"}:
        return "◆"
    if attention == "failed":
        return "●"
    return "●"


def _connection_label(attention: str) -> str:
    if attention == "working":
        return "◉ agent working"
    if attention in {"needs-input", "needs-reconciliation"}:
        return "◆ action needed"
    if attention == "failed":
        return "● needs attention"
    return "● connected"


def _compact_connection_label(
    connection_state: str,
    attention: str,
) -> str:
    if connection_state == "reconnecting":
        return "◌ retrying"
    if connection_state == "send-unacknowledged":
        return "◆ pending"
    if connection_state == "disconnected":
        return "● offline"
    if attention == "working":
        return "◉ working"
    if attention in {"needs-input", "needs-reconciliation"}:
        return "◆ action"
    if attention == "failed":
        return "● issue"
    return "● online"


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
    from agent_harness.terminal import native_provider_command

    return native_provider_command(provider, permission_mode, [])


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
