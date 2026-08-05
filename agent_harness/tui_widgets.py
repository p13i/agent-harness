"""Reusable Textual controls and command-discovery primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import shlex
from typing import Iterable

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class ComposerAction(str, Enum):
    """Semantic result of a composer input operation."""

    SEND = "send"
    NEWLINE = "newline"
    EDIT = "edit"


@dataclass(frozen=True, slots=True)
class ComposerDraft:
    """Durable text and logical cursor location."""

    text: str
    cursor_row: int
    cursor_column: int


class MultilineComposer(TextArea):
    """Auto-growing composer with explicit send and newline keys."""

    BINDINGS = [
        Binding("enter", "submit", "Send", show=False, priority=True),
        Binding(
            "shift+enter",
            "insert_composer_newline",
            "Newline",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+j",
            "insert_composer_newline",
            "Newline",
            show=False,
            priority=True,
        ),
        *TextArea.BINDINGS,
    ]

    class Submitted(Message):
        """A send request whose text remains until the caller acknowledges it."""

        def __init__(self, composer: MultilineComposer, text: str) -> None:
            super().__init__()
            self.composer = composer
            self.text = text

        @property
        def control(self) -> MultilineComposer:
            """Return the originating composer."""

            return self.composer

    def __init__(
        self,
        text: str = "",
        *,
        min_lines: int = 1,
        max_lines: int = 8,
        wrap_width: int = 72,
        **kwargs: object,
    ) -> None:
        if min_lines < 1:
            raise ValueError("min_lines must be positive")
        if max_lines < min_lines:
            raise ValueError("max_lines must not be below min_lines")
        if wrap_width < 1:
            raise ValueError("wrap_width must be positive")
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.composer_wrap_width = wrap_width
        super().__init__(
            text,
            soft_wrap=True,
            show_line_numbers=False,
            **kwargs,
        )

    def on_mount(self) -> None:
        """Set initial auto-growing height."""

        self.refresh_auto_height()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Resize after editing this composer."""

        if event.text_area is not self:
            return
        self.refresh_auto_height()

    def action_submit(self) -> None:
        """Post a send request without clearing the durable draft."""

        text = self.text.strip()
        if not text:
            return
        self.post_message(self.Submitted(self, text))

    def action_insert_composer_newline(self) -> None:
        """Insert a newline at the current selection."""

        start, end = self.selection
        result = self.replace(
            "\n",
            start,
            end,
            maintain_selection_offset=False,
        )
        self.move_cursor(result.end_location)
        self.refresh_auto_height()

    async def _on_paste(self, event: events.Paste) -> None:
        """Insert pasted content without invoking submit behavior."""

        await super()._on_paste(event)
        self.refresh_auto_height()

    def refresh_auto_height(self) -> int:
        """Apply and return the current bounded composer height."""

        width = self.composer_wrap_width
        if self.content_region.width > 0:
            width = self.content_region.width
        lines = composer_height(
            self.text,
            wrap_width=width,
            min_lines=self.min_lines,
            max_lines=self.max_lines,
        )
        self.styles.height = lines
        return lines

    def capture_draft(self) -> ComposerDraft:
        """Capture text and cursor for durable UI state."""

        row, column = self.cursor_location
        return ComposerDraft(
            text=self.text,
            cursor_row=row,
            cursor_column=column,
        )

    def restore_draft(self, draft: ComposerDraft) -> None:
        """Restore text and a clamped logical cursor."""

        self.text = draft.text
        row = min(max(0, draft.cursor_row), self.document.line_count - 1)
        line = self.document[row]
        column = min(max(0, draft.cursor_column), len(line))
        self.move_cursor((row, column))
        self.refresh_auto_height()


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """Command metadata used by completion and inline validation."""

    name: str
    summary: str
    usage: str
    min_arguments: int = 0
    max_arguments: int | None = 0
    first_argument_choices: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SlashCompletion:
    """One ranked completion."""

    command: SlashCommand
    score: int

    @property
    def insertion(self) -> str:
        """Return the command spelling inserted into the composer."""

        return self.command.name + " "


class SlashValidationState(str, Enum):
    """Editable slash-command validation state."""

    NOT_COMMAND = "not-command"
    INCOMPLETE = "incomplete"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class SlashValidation:
    """Inline validation that never converts invalid input to a prompt."""

    state: SlashValidationState
    command: SlashCommand | None
    message: str

    @property
    def can_execute(self) -> bool:
        """Return whether the slash command is complete and valid."""

        return self.state == SlashValidationState.VALID


@dataclass(frozen=True, slots=True)
class SlashCompletionState:
    """Keyboard-navigable completion list."""

    query: str
    items: tuple[SlashCompletion, ...]
    selected_index: int = 0

    @property
    def selected(self) -> SlashCompletion | None:
        """Return the selected completion."""

        if not self.items:
            return None
        index = self.selected_index % len(self.items)
        return self.items[index]

    def move(self, offset: int) -> SlashCompletionState:
        """Move selection with deterministic wraparound."""

        if not self.items:
            return self
        index = (self.selected_index + offset) % len(self.items)
        return SlashCompletionState(
            query=self.query,
            items=self.items,
            selected_index=index,
        )


DEFAULT_SLASH_COMMANDS = (
    SlashCommand("/help", "Show command help", "/help"),
    SlashCommand("/new", "Create a session", "/new"),
    SlashCommand("/interrupt", "Interrupt current work", "/interrupt"),
    SlashCommand("/pause", "Pause the session", "/pause"),
    SlashCommand("/resume", "Resume the session", "/resume"),
    SlashCommand("/stop", "Stop the session", "/stop"),
    SlashCommand("/export", "Export the session", "/export"),
    SlashCommand("/checkpoint", "Create a checkpoint", "/checkpoint"),
    SlashCommand("/archive", "Archive the session", "/archive"),
    SlashCommand("/unarchive", "Restore the session", "/unarchive"),
    SlashCommand(
        "/rename",
        "Rename the session",
        "/rename <name>",
        min_arguments=1,
        max_arguments=None,
    ),
    SlashCommand(
        "/fork",
        "Fork the session",
        "/fork [name]",
        max_arguments=None,
    ),
    SlashCommand(
        "/sessions",
        "Choose session visibility",
        "/sessions <focused|all>",
        min_arguments=1,
        max_arguments=1,
        first_argument_choices=("focused", "all"),
    ),
    SlashCommand(
        "/events",
        "Choose event visibility",
        "/events <on|off>",
        min_arguments=1,
        max_arguments=1,
        first_argument_choices=("on", "off"),
    ),
    SlashCommand(
        "/sidebar",
        "Reset the sidebar width",
        "/sidebar reset",
        min_arguments=1,
        max_arguments=1,
        first_argument_choices=("reset",),
    ),
    SlashCommand(
        "/mode",
        "Choose Focus or Control",
        "/mode <focus|control>",
        min_arguments=1,
        max_arguments=1,
        first_argument_choices=("focus", "control"),
    ),
    SlashCommand(
        "/provider",
        "Choose a provider",
        "/provider <auto|claude|codex|kimi>",
        min_arguments=1,
        max_arguments=1,
        first_argument_choices=("auto", "claude", "codex", "kimi"),
    ),
    SlashCommand(
        "/model",
        "Choose a model",
        "/model <auto|id>",
        min_arguments=1,
        max_arguments=1,
    ),
    SlashCommand(
        "/effort",
        "Choose reasoning effort",
        "/effort <auto|level>",
        min_arguments=1,
        max_arguments=1,
    ),
    SlashCommand(
        "/theme",
        "Choose appearance",
        "/theme <system|light|dark>",
        min_arguments=1,
        max_arguments=1,
        first_argument_choices=("system", "light", "dark"),
    ),
    SlashCommand(
        "/permission",
        "Choose permission mode",
        "/permission <mode>",
        min_arguments=1,
        max_arguments=1,
    ),
    SlashCommand("/route", "Preview routing", "/route"),
    SlashCommand("/providers", "Refresh providers", "/providers"),
    SlashCommand("/usage", "Refresh usage", "/usage"),
    SlashCommand(
        "/budget",
        "Inspect or extend the safety budget",
        "/budget [extend|xhigh] ...",
        max_arguments=None,
    ),
    SlashCommand(
        "/native",
        "Open a native provider CLI",
        "/native <claude|codex>",
        min_arguments=1,
        max_arguments=1,
        first_argument_choices=("claude", "codex"),
    ),
    SlashCommand(
        "/approve",
        "Resolve an approval",
        "/approve <uuid> <decision>",
        min_arguments=2,
        max_arguments=2,
    ),
    SlashCommand(
        "/reconcile",
        "Resolve interrupted provider work",
        "/reconcile <decision> <id> <workspace-digest>",
        min_arguments=3,
        max_arguments=4,
        first_argument_choices=(
            "accept-current",
            "restore-pre-turn",
            "stop",
        ),
    ),
)


def composer_action_for_key(key: str, *, pasted: bool = False) -> ComposerAction:
    """Map terminal input to composer semantics."""

    if pasted:
        return ComposerAction.EDIT
    normalized = key.casefold()
    if normalized == "enter":
        return ComposerAction.SEND
    if normalized in {"shift+enter", "ctrl+j"}:
        return ComposerAction.NEWLINE
    return ComposerAction.EDIT


def composer_height(
    text: str,
    *,
    wrap_width: int,
    min_lines: int = 1,
    max_lines: int = 8,
) -> int:
    """Count logical and soft-wrapped lines within explicit bounds."""

    if wrap_width < 1:
        raise ValueError("wrap_width must be positive")
    if min_lines < 1:
        raise ValueError("min_lines must be positive")
    if max_lines < min_lines:
        raise ValueError("max_lines must not be below min_lines")
    lines = 0
    for logical_line in text.split("\n"):
        lines += max(1, math.ceil(len(logical_line) / wrap_width))
    return min(max_lines, max(min_lines, lines))


def complete_slash(
    text: str,
    *,
    commands: Iterable[SlashCommand] = DEFAULT_SLASH_COMMANDS,
    limit: int = 8,
) -> SlashCompletionState:
    """Fuzzy-rank commands for the token under the cursor."""

    if limit < 1:
        raise ValueError("limit must be positive")
    query = _command_token(text)
    if query is None:
        return SlashCompletionState(query="", items=())
    ranked: list[SlashCompletion] = []
    for command in commands:
        score = _fuzzy_score(query, command.name)
        if score is None:
            continue
        ranked.append(SlashCompletion(command=command, score=score))
    ranked.sort(key=lambda item: (-item.score, item.command.name))
    return SlashCompletionState(
        query=query,
        items=tuple(ranked[:limit]),
    )


def validate_slash(
    text: str,
    *,
    commands: Iterable[SlashCommand] = DEFAULT_SLASH_COMMANDS,
) -> SlashValidation:
    """Validate without ever treating unknown slash text as a model prompt."""

    stripped = text.strip()
    if not stripped.startswith("/"):
        return SlashValidation(
            state=SlashValidationState.NOT_COMMAND,
            command=None,
            message="",
        )
    if stripped == "/":
        return SlashValidation(
            state=SlashValidationState.INCOMPLETE,
            command=None,
            message="Type a harness command",
        )
    try:
        parts = shlex.split(stripped)
    except ValueError as error:
        return SlashValidation(
            state=SlashValidationState.INVALID,
            command=None,
            message=str(error),
        )
    by_name = {command.name.casefold(): command for command in commands}
    command = by_name.get(parts[0].casefold())
    if command is None:
        return SlashValidation(
            state=SlashValidationState.INVALID,
            command=None,
            message="Unknown harness command",
        )
    arguments = parts[1:]
    if len(arguments) < command.min_arguments:
        return SlashValidation(
            state=SlashValidationState.INCOMPLETE,
            command=command,
            message=command.usage,
        )
    if (
        command.max_arguments is not None
        and len(arguments) > command.max_arguments
    ):
        return SlashValidation(
            state=SlashValidationState.INVALID,
            command=command,
            message=command.usage,
        )
    if command.first_argument_choices and arguments:
        value = arguments[0].casefold()
        if value not in command.first_argument_choices:
            return SlashValidation(
                state=SlashValidationState.INVALID,
                command=command,
                message=command.usage,
            )
    return SlashValidation(
        state=SlashValidationState.VALID,
        command=command,
        message=command.summary,
    )


def apply_completion(text: str, completion: SlashCompletion) -> str:
    """Replace only the command token and retain existing arguments."""

    stripped = text.lstrip()
    leading = text[: len(text) - len(stripped)]
    if not stripped.startswith("/"):
        return text
    separator = stripped.find(" ")
    if separator < 0:
        return leading + completion.command.name + " "
    suffix = stripped[separator:]
    return leading + completion.command.name + suffix


def _command_token(text: str) -> str | None:
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None
    token = stripped.split(maxsplit=1)[0]
    return token.casefold()


def _fuzzy_score(query: str, candidate: str) -> int | None:
    normalized_query = query.casefold()
    normalized_candidate = candidate.casefold()
    if normalized_query == normalized_candidate:
        return 10_000
    if normalized_candidate.startswith(normalized_query):
        return 8_000 - len(normalized_candidate)
    position = 0
    gaps = 0
    for character in normalized_query:
        found = normalized_candidate.find(character, position)
        if found < 0:
            return None
        gaps += found - position
        position = found + 1
    return 4_000 - gaps - len(normalized_candidate)
