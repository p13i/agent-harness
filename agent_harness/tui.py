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
from textual.binding import Binding
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.screen import ModalScreen
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
        min-width: 26;
        padding: 1;
        background: #0d131c;
        border-right: solid #27364a;
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
        height: 4;
        margin-bottom: 1;
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
    #sessions-help {
        height: 2;
        color: #64748b;
    }
    #main {
        width: 1fr;
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
        background: #e7eef7;
        border-bottom: solid #b8c6d8;
    }
    Screen.light #brand {
        color: #0369a1;
    }
    Screen.light #product-promise {
        color: #64748b;
    }
    Screen.light #sidebar,
    Screen.light #inspector {
        background: #edf2f8;
    }
    Screen.light #sidebar {
        border-right: solid #b8c6d8;
    }
    Screen.light #inspector {
        border-left: solid #b8c6d8;
    }
    Screen.light #workspace-summary,
    Screen.light #inspector-content {
        color: #334155;
    }
    Screen.light #session-list ListItem {
        color: #475569;
        background: #f8fafc;
        border-left: tall #c4cfdd;
    }
    Screen.light #session-list ListItem:hover {
        color: #0f172a;
        background: #e2e8f0;
    }
    Screen.light #session-list ListItem.-highlight {
        color: #0c4a6e;
        background: #dbeafe;
        border-left: tall #0284c7;
    }
    Screen.light #main,
    Screen.light #transcript {
        background: #f4f7fb;
    }
    Screen.light #session-bar {
        background: #f8fafc;
        border-bottom: solid #c4cfdd;
    }
    Screen.light #session-title {
        color: #0f172a;
    }
    Screen.light #session-meta {
        color: #475569;
    }
    Screen.light #resume-token {
        color: #8290a3;
    }
    Screen.light #composer-shell {
        background: #ffffff;
        border: round #94a3b8;
    }
    Screen.light #composer {
        color: #0f172a;
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
        self._session: dict[str, Any] = {}
        self._goal: dict[str, Any] | None = None
        self._approvals: tuple[dict[str, Any], ...] = ()
        self._providers: dict[str, Any] = {}
        self._provider_override = ""
        self._model_override = ""
        self._effort_override = ""
        self._saved_composer = ""
        self._last_approval_id = ""
        self._provider_poll = 0
        self._theme_poll = 0
        self._theme_preference = "system"
        self._sidebar_requested = True
        self._inspector_requested = True

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
                    "↑↓ select  ·  Enter open\nCtrl+R synchronize",
                    id="sessions-help",
                )
            with Vertical(id="main"):
                with Vertical(id="session-bar"):
                    yield Static("Starting…", id="session-title")
                    yield Static(
                        "Preparing durable workspace",
                        id="session-meta",
                    )
                    yield Static("", id="resume-token")
                yield RichLog(
                    id="transcript",
                    markup=True,
                    wrap=True,
                    highlight=True,
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
        await self._sync_system_theme(force=True)
        await self._load_sessions()
        if self.session_id:
            await self._open_session(self.session_id)
        else:
            await self._new_session()
        self.set_interval(0.5, self._poll)
        self.query_one("#composer", Input).focus()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_responsive_layout(event.size.width)

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
        if index < 0 or index >= len(self._sessions):
            return
        session_id = str(self._sessions[index].get("session_id", ""))
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
        self._sidebar_requested = not self._sidebar_requested
        sidebar.display = self._sidebar_requested

    def action_toggle_inspector(self) -> None:
        inspector = self.query_one("#inspector", Vertical)
        self._inspector_requested = not self._inspector_requested
        inspector.display = self._inspector_requested

    def action_show_help(self) -> None:
        self._write_help()

    async def _new_session(self) -> None:
        result = await self.client.request(
            "POST",
            "/v1/sessions",
            payload={
                "workspace": str(self.workspace),
                "permission_mode": self.permission_mode,
            },
        )
        session = _object(result.get("session"))
        session_id = str(session.get("session_id", ""))
        await self._load_sessions()
        await self._open_session(session_id)

    async def _open_session(self, session_id: str) -> None:
        self.session_id = session_id
        self.sequence = 0
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
        await self._sync_system_theme(force=True)
        composer = str(ui_state.get("composer", ""))
        self._saved_composer = composer
        self.query_one("#composer", Input).value = composer
        self._last_approval_id = ""
        transcript = self.query_one("#transcript", RichLog)
        transcript.clear()
        transcript.write(self._welcome_message())
        await self._poll()

    async def _load_sessions(self) -> None:
        result = await self.client.request("GET", "/v1/sessions")
        sessions = result.get("sessions", [])
        if not isinstance(sessions, list):
            sessions = []
        self._sessions = [
            item for item in sessions if isinstance(item, dict)
        ]
        view = self.query_one("#session-list", ListView)
        await view.clear()
        for session in self._sessions:
            name = str(session.get("name", "unnamed"))
            provider = str(session.get("active_provider", ""))
            lifecycle = str(session.get("lifecycle", ""))
            detail = name + "\n" + lifecycle
            if provider:
                detail += " · " + provider
            await view.append(ListItem(Label(detail)))

    async def _poll(self) -> None:
        if not self.session_id:
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
        session = _object(state.get("session"))
        self._session = session
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
            + escape(lifecycle.upper())
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
        transcript = self.query_one("#transcript", RichLog)
        for event in events:
            if not isinstance(event, dict):
                continue
            sequence = int(event.get("sequence", 0))
            self.sequence = max(self.sequence, sequence)
            rendered = _render_event(event)
            if rendered:
                transcript.write(rendered)
        self._present_approval()

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
            self.query_one("#transcript", RichLog).write(
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
        self.query_one("#transcript", RichLog).write(
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
        if not force and composer == self._saved_composer:
            return
        await self.client.request(
            "PUT",
            "/v1/sessions/" + self.session_id + "/ui-state",
            payload={
                "composer": composer,
                "provider": self._provider_override,
                "model": self._model_override,
                "effort": self._effort_override,
                "theme": self._theme_preference,
            },
        )
        self._saved_composer = composer

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
                "Ctrl+K  checkpoint",
                "F1      command help",
            ]
        )
        self.query_one("#inspector-content", Static).update(
            "\n".join(lines)
        )

    def _write_help(self) -> None:
        self.query_one("#transcript", RichLog).write(
            "\n[bold cyan]HARNESS COMMANDS[/bold cyan]\n"
            "[bold]Session[/bold]  /new · /checkpoint · "
            "/fork [name] · /export\n"
            "[bold]Control[/bold]  /interrupt · /pause · "
            "/resume · /stop\n"
            "[bold]Route[/bold]    /provider <auto|claude|codex> · "
            "/model <auto|id>\n"
            "         /effort <auto|level> · /route · /providers\n"
            "[bold]Display[/bold]  /theme <system|light|dark>\n"
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

    def _write_notice(self, value: str) -> None:
        self.query_one("#transcript", RichLog).write(value)


def _render_event(event: dict[str, Any]) -> str:
    event_type = escape(str(event.get("event_type", "event")))
    text = escape(str(event.get("text", "")))
    if event_type in {"user.message", "user.steer"}:
        return (
            "\n[bold cyan]YOU[/bold cyan]\n" + text + "\n"
        )
    if event_type in {"agent.message", "agent.message.delta"}:
        return (
            "\n[bold green]AGENT[/bold green]\n" + text + "\n"
        )
    if event_type.startswith("tool."):
        return (
            "\n[bold yellow]"
            + event_type.upper()
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
    if event_type in {
        "routing.selected",
        "routing.failover",
        "checkpoint.created",
        "goal.completed",
        "turn.failed",
    }:
        return "[dim]" + event_type + " " + text + "[/dim]"
    return ""


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
