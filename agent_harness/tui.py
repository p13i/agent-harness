"""Textual workspace for durable agent sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
from typing import Any

from rich.markup import escape
from textual.app import App
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button
from textual.widgets import Footer
from textual.widgets import Header
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
    TITLE = "p13i agent harness"
    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #sidebar {
        width: 30;
        min-width: 24;
        border-right: solid $primary;
    }
    #session-list {
        height: 1fr;
    }
    #main {
        width: 1fr;
    }
    #inspector {
        width: 38;
        min-width: 28;
        border-left: solid $primary;
        padding: 0 1;
    }
    #status {
        height: 3;
        padding: 0 1;
        border-bottom: solid $primary;
    }
    #transcript {
        height: 1fr;
        padding: 0 1;
    }
    #composer {
        dock: bottom;
        margin: 0 1 1 1;
    }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "interrupt", "Interrupt"),
        Binding("ctrl+p", "pause", "Pause"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+n", "new_session", "New"),
        Binding("ctrl+k", "checkpoint", "Checkpoint"),
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

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("Sessions")
                yield ListView(id="session-list")
            with Vertical(id="main"):
                yield Static("Starting…", id="status")
                yield RichLog(
                    id="transcript",
                    markup=True,
                    wrap=True,
                    highlight=True,
                )
                yield Input(
                    placeholder=(
                        "Message the agent or use /help for harness commands"
                    ),
                    id="composer",
                )
            with Vertical(id="inspector"):
                yield Label("Session inspector")
                yield Static("", id="inspector-content")
        yield Footer()

    async def on_mount(self) -> None:
        await self._load_sessions()
        if self.session_id:
            await self._open_session(self.session_id)
        else:
            await self._new_session()
        self.set_interval(0.5, self._poll)
        self.query_one("#composer", Input).focus()

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
        composer = str(ui_state.get("composer", ""))
        self._saved_composer = composer
        self.query_one("#composer", Input).value = composer
        self._last_approval_id = ""
        transcript = self.query_one("#transcript", RichLog)
        transcript.clear()
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
        if not self._providers or self._provider_poll >= 20:
            await self._load_providers()
            self._provider_poll = 0
        status = (
            str(session.get("name", ""))
            + " · "
            + str(session.get("lifecycle", ""))
            + "/"
            + str(session.get("attention", ""))
            + " · "
            + str(session.get("active_provider", "unrouted"))
            + "/"
            + str(session.get("model", "default"))
            + goal_text
            + "\n"
            + self.session_id
            + " · "
            + str(session.get("worktree", ""))
        )
        self.query_one("#status", Static).update(escape(status))
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
            self.query_one("#transcript", RichLog).write(
                "[bold]Harness commands[/bold]\n"
                "/interrupt · /pause · /resume · /export · "
                "/checkpoint · /fork [name] · "
                "/provider <auto|claude|codex> · "
                "/model <auto|id> · /effort <auto|level> · "
                "/permission <mode> · /route · /providers · "
                "/native <claude|codex> · "
                "/approve <uuid> <decision> · /new"
            )
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
        if command in {"/provider", "/model", "/effort"}:
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
            "[bold]Route controls[/bold]",
            "Provider: "
            + escape(self._provider_override or "automatic"),
            "Model: " + escape(self._model_override or "automatic"),
            "Effort: " + escape(self._effort_override or "automatic"),
            "Permission: "
            + escape(str(self._session.get("permission_mode", ""))),
        ]
        if self._goal is not None:
            lines.extend(
                [
                    "",
                    "[bold]Goal[/bold]",
                    escape(str(self._goal.get("kind", "")))
                    + " · "
                    + escape(str(self._goal.get("status", ""))),
                    escape(str(self._goal.get("objective", ""))),
                ]
            )
        lines.extend(["", "[bold]Pending approvals[/bold]"])
        if not self._approvals:
            lines.append("None")
        for approval in self._approvals:
            lines.append(
                escape(str(approval.get("approval_id", "")))
                + "\n"
                + escape(str(approval.get("prompt", "")))
            )
        lines.extend(["", "[bold]Provider capacity[/bold]"])
        for provider, value in sorted(self._providers.items()):
            provider_value = _object(value)
            usage = _object(provider_value.get("usage"))
            binding = usage.get("binding_percent")
            detail = escape(provider)
            if binding is not None:
                detail += " · " + escape(str(binding)) + "%"
            if not bool(provider_value.get("ready", False)):
                detail += " · unavailable"
            lines.append(detail)
        self.query_one("#inspector-content", Static).update(
            "\n".join(lines)
        )

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
        return "[bold cyan]You[/bold cyan]\n" + text
    if event_type in {"agent.message", "agent.message.delta"}:
        return "[bold green]Agent[/bold green]\n" + text
    if event_type.startswith("tool."):
        return "[bold yellow]" + event_type + "[/bold yellow]\n" + text
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


async def run_tui(
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
    await asyncio.to_thread(app.run)
