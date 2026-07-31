"""Deterministic presentation state for the terminal workspace."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
import math
from types import MappingProxyType
from typing import Any
from typing import Iterable
from typing import Mapping


class TranscriptBlockKind(str, Enum):
    """Provider-neutral transcript surfaces."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    APPROVAL = "approval"
    RECONCILIATION = "reconciliation"
    WARNING = "warning"
    SYSTEM = "system"


class TranscriptBlockStatus(str, Enum):
    """Presentation status for a transcript block."""

    COMPLETE = "complete"
    STREAMING = "streaming"
    RUNNING = "running"
    WAITING = "waiting"
    FAILED = "failed"
    GUARDED = "guarded"


class TranscriptMutationKind(str, Enum):
    """Smallest operation needed to update a transcript view."""

    INSERT = "insert"
    APPEND = "append"
    REPLACE = "replace"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class TranscriptBlock:
    """One stable, typed transcript block."""

    block_id: str
    kind: TranscriptBlockKind
    status: TranscriptBlockStatus
    title: str
    content: str
    detail: str = ""
    turn_id: str = ""
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class TranscriptMutation:
    """An incremental instruction for a transcript renderer."""

    kind: TranscriptMutationKind
    block_id: str
    block: TranscriptBlock | None = None
    delta: str = ""


@dataclass(frozen=True, slots=True)
class TranscriptState:
    """Immutable transcript plus reader-local interaction state."""

    blocks: tuple[TranscriptBlock, ...] = ()
    applied_sequences: frozenset[int] = frozenset()
    applied_event_ids: frozenset[str] = frozenset()
    expanded_block_ids: frozenset[str] = frozenset()
    reader_at_bottom: bool = True
    new_activity_count: int = 0
    latest_sequence: int = 0

    def block(self, block_id: str) -> TranscriptBlock | None:
        """Return a block without exposing mutable lookup state."""

        for item in self.blocks:
            if item.block_id == block_id:
                return item
        return None

    def with_reader_at_bottom(self, at_bottom: bool) -> TranscriptState:
        """Record reader position and acknowledge activity at the bottom."""

        activity = self.new_activity_count
        if at_bottom:
            activity = 0
        return replace(
            self,
            reader_at_bottom=at_bottom,
            new_activity_count=activity,
        )

    def toggle_expanded(self, block_id: str) -> TranscriptState:
        """Toggle tool or long-output expansion without changing blocks."""

        expanded = set(self.expanded_block_ids)
        if block_id in expanded:
            expanded.remove(block_id)
        else:
            expanded.add(block_id)
        return replace(self, expanded_block_ids=frozenset(expanded))


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    """State and renderer operations produced from canonical events."""

    state: TranscriptState
    mutations: tuple[TranscriptMutation, ...]


class LayoutMode(str, Enum):
    """Declared responsive layout bands."""

    MINIMAL = "minimal"
    OVERLAY = "overlay"
    COMPACT = "compact"
    WIDE = "wide"
    SPACIOUS = "spacious"


@dataclass(frozen=True, slots=True)
class LayoutDecision:
    """Deterministic visibility and sizing at a terminal breakpoint."""

    mode: LayoutMode
    width: int
    height: int
    sidebar_mode: str
    sidebar_visible: bool
    inspector_visible: bool
    header_detail_visible: bool
    composer_max_lines: int
    transcript_horizontal_padding: int


class ThemePreference(str, Enum):
    """Persisted theme choice."""

    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


class ColorScheme(str, Enum):
    """Resolved color scheme."""

    LIGHT = "light"
    DARK = "dark"


@dataclass(frozen=True, slots=True)
class SpacingTokens:
    """Shared spacing scale."""

    compact: int
    normal: int
    roomy: int
    panel: int


@dataclass(frozen=True, slots=True)
class ColorTokens:
    """Semantic colors shared by all terminal surfaces."""

    canvas: str
    surface: str
    raised: str
    border: str
    text: str
    text_muted: str
    focus: str
    active: str
    success: str
    warning: str
    danger: str
    approval: str
    reconciliation: str


@dataclass(frozen=True, slots=True)
class DesignTokens:
    """A complete resolved visual vocabulary."""

    scheme: ColorScheme
    spacing: SpacingTokens
    colors: ColorTokens
    border_style: str
    focus_border_style: str


@dataclass(frozen=True, slots=True)
class ComposerState:
    """Durable composer projection independent of Textual."""

    text: str = ""
    cursor_row: int = 0
    cursor_column: int = 0
    request_id: str = ""
    awaiting_acknowledgement: bool = False


@dataclass(frozen=True, slots=True)
class InteractionState:
    """Reader interaction that background events must preserve."""

    focus_id: str = "composer"
    transcript_scroll_y: float = 0.0
    selection_anchor: str = ""
    active_inspector_tab: str = "Context"


@dataclass(frozen=True, slots=True)
class TuiViewState:
    """Top-level immutable state consumed by the Textual composition layer."""

    transcript: TranscriptState
    composer: ComposerState
    interaction: InteractionState
    layout: LayoutDecision
    tokens: DesignTokens
    connection_state: str = "unknown"
    validation_state: str = "unknown"

    def with_events(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        show_events: bool = False,
    ) -> tuple[TuiViewState, tuple[TranscriptMutation, ...]]:
        """Project background events while preserving interaction state."""

        update = project_events(
            self.transcript,
            events,
            show_events=show_events,
        )
        return (
            replace(self, transcript=update.state),
            update.mutations,
        )


_SAFE_DETAIL_KEYS = (
    "summary",
    "detail",
    "reason",
    "action",
    "status",
    "name",
    "is_error",
)
_PROTOCOL_NOISE = {
    "tool.usermessage.started",
    "tool.usermessage.completed",
    "tool.user_message.started",
    "tool.user_message.completed",
}


def project_events(
    state: TranscriptState,
    events: Iterable[Mapping[str, Any]],
    *,
    show_events: bool = False,
) -> TranscriptUpdate:
    """Project ordered canonical events into stable transcript blocks."""

    current = state
    mutations: list[TranscriptMutation] = []
    for event in events:
        update = project_event(current, event, show_events=show_events)
        current = update.state
        mutations.extend(update.mutations)
    return TranscriptUpdate(state=current, mutations=tuple(mutations))


def project_event(
    state: TranscriptState,
    event: Mapping[str, Any],
    *,
    show_events: bool = False,
) -> TranscriptUpdate:
    """Project one event and return only the required renderer mutation."""

    sequence = _integer(event.get("sequence"))
    event_id = _text(event.get("event_id"))
    if sequence > 0 and sequence in state.applied_sequences:
        return _ignored_update(state)
    if event_id and event_id in state.applied_event_ids:
        return _ignored_update(state)

    event_type = _text(event.get("event_type")).casefold()
    if event_type in _PROTOCOL_NOISE:
        return _advance_event_identity(state, sequence, event_id)

    existing_id = _block_id(event, event_type, sequence)
    existing = state.block(existing_id)
    block = _block_from_event(event, event_type, existing_id, sequence)
    if block is None:
        if not show_events:
            return _advance_event_identity(state, sequence, event_id)
        block = TranscriptBlock(
            block_id=existing_id,
            kind=TranscriptBlockKind.SYSTEM,
            status=TranscriptBlockStatus.COMPLETE,
            title="Event",
            content=_display_event_type(event_type),
            sequence=sequence,
        )
    if (
        existing is not None
        and existing.kind == TranscriptBlockKind.TOOL
        and block.kind == TranscriptBlockKind.TOOL
        and block.title == "Tool"
    ):
        block = replace(block, title=existing.title)

    mutation = TranscriptMutationKind.INSERT
    delta = ""
    blocks = list(state.blocks)
    if existing is not None:
        index = blocks.index(existing)
        if (
            event_type == "agent.message.delta"
            and existing.status == TranscriptBlockStatus.STREAMING
        ):
            delta = block.content
            block = replace(
                existing,
                content=existing.content + delta,
                sequence=sequence,
            )
            mutation = TranscriptMutationKind.APPEND
        else:
            mutation = TranscriptMutationKind.REPLACE
        blocks[index] = block
    else:
        blocks.append(block)

    activity = state.new_activity_count
    if not state.reader_at_bottom:
        activity += 1
    next_state = replace(
        state,
        blocks=tuple(blocks),
        applied_sequences=_with_sequence(state, sequence),
        applied_event_ids=_with_event_id(state, event_id),
        new_activity_count=activity,
        latest_sequence=max(state.latest_sequence, sequence),
    )
    view_block: TranscriptBlock | None = block
    if mutation == TranscriptMutationKind.APPEND:
        view_block = None
    return TranscriptUpdate(
        state=next_state,
        mutations=(
            TranscriptMutation(
                kind=mutation,
                block_id=block.block_id,
                block=view_block,
                delta=delta,
            ),
        ),
    )


def decide_layout(
    width: int,
    height: int,
    *,
    sidebar_requested: bool = True,
    inspector_requested: bool = True,
) -> LayoutDecision:
    """Resolve layouts from the supported 60x20 through 160x48 range."""

    normalized_width = max(1, width)
    normalized_height = max(1, height)
    if normalized_width < 70:
        return LayoutDecision(
            mode=LayoutMode.MINIMAL,
            width=normalized_width,
            height=normalized_height,
            sidebar_mode="collapsed",
            sidebar_visible=False,
            inspector_visible=False,
            header_detail_visible=False,
            composer_max_lines=_composer_lines(normalized_height, 3),
            transcript_horizontal_padding=0,
        )
    if normalized_width < 96:
        return LayoutDecision(
            mode=LayoutMode.OVERLAY,
            width=normalized_width,
            height=normalized_height,
            sidebar_mode="overlay",
            sidebar_visible=sidebar_requested,
            inspector_visible=False,
            header_detail_visible=False,
            composer_max_lines=_composer_lines(normalized_height, 4),
            transcript_horizontal_padding=1,
        )
    if normalized_width < 120:
        return LayoutDecision(
            mode=LayoutMode.COMPACT,
            width=normalized_width,
            height=normalized_height,
            sidebar_mode="docked",
            sidebar_visible=sidebar_requested,
            inspector_visible=False,
            header_detail_visible=True,
            composer_max_lines=_composer_lines(normalized_height, 5),
            transcript_horizontal_padding=1,
        )
    if normalized_width < 150:
        return LayoutDecision(
            mode=LayoutMode.WIDE,
            width=normalized_width,
            height=normalized_height,
            sidebar_mode="docked",
            sidebar_visible=sidebar_requested,
            inspector_visible=inspector_requested,
            header_detail_visible=True,
            composer_max_lines=_composer_lines(normalized_height, 7),
            transcript_horizontal_padding=2,
        )
    return LayoutDecision(
        mode=LayoutMode.SPACIOUS,
        width=normalized_width,
        height=normalized_height,
        sidebar_mode="docked",
        sidebar_visible=sidebar_requested,
        inspector_visible=inspector_requested,
        header_detail_visible=True,
        composer_max_lines=_composer_lines(normalized_height, 8),
        transcript_horizontal_padding=3,
    )


def resolve_theme(
    preference: ThemePreference | str,
    *,
    system_dark: bool,
) -> DesignTokens:
    """Resolve a durable preference against current system appearance."""

    selected = _theme_preference(preference)
    scheme = ColorScheme.LIGHT
    if selected == ThemePreference.DARK:
        scheme = ColorScheme.DARK
    elif selected == ThemePreference.SYSTEM and system_dark:
        scheme = ColorScheme.DARK
    spacing = SpacingTokens(compact=0, normal=1, roomy=2, panel=3)
    if scheme == ColorScheme.DARK:
        colors = ColorTokens(
            canvas="#0a0e14",
            surface="#0f1620",
            raised="#172231",
            border="#33465f",
            text="#e8eef7",
            text_muted="#8fa0b7",
            focus="#38bdf8",
            active="#7dd3fc",
            success="#86efac",
            warning="#facc15",
            danger="#f87171",
            approval="#e879f9",
            reconciliation="#fb923c",
        )
    else:
        colors = ColorTokens(
            canvas="#ffffff",
            surface="#f8fafc",
            raised="#e2e8f0",
            border="#64748b",
            text="#0f172a",
            text_muted="#475569",
            focus="#0369a1",
            active="#075985",
            success="#166534",
            warning="#854d0e",
            danger="#b91c1c",
            approval="#86198f",
            reconciliation="#9a3412",
        )
    return DesignTokens(
        scheme=scheme,
        spacing=spacing,
        colors=colors,
        border_style="solid",
        focus_border_style="round",
    )


def contrast_ratio(foreground: str, background: str) -> float:
    """Return WCAG contrast ratio for two six-digit hexadecimal colors."""

    foreground_luminance = _relative_luminance(foreground)
    background_luminance = _relative_luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def safe_metadata(metadata: Any) -> Mapping[str, str]:
    """Expose only presentation metadata and reject secret-shaped keys."""

    if not isinstance(metadata, Mapping):
        return MappingProxyType({})
    result: dict[str, str] = {}
    for key in _SAFE_DETAIL_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        result[key] = _text(value)
    return MappingProxyType(result)


def _block_from_event(
    event: Mapping[str, Any],
    event_type: str,
    block_id: str,
    sequence: int,
) -> TranscriptBlock | None:
    text = _text(event.get("text"))
    turn_id = _text(event.get("turn_id"))
    metadata = safe_metadata(event.get("metadata"))
    if event_type in {"user.message", "user.steer"}:
        return TranscriptBlock(
            block_id=block_id,
            kind=TranscriptBlockKind.USER,
            status=TranscriptBlockStatus.COMPLETE,
            title="You",
            content=text,
            turn_id=turn_id,
            sequence=sequence,
        )
    if event_type in {"agent.message", "agent.message.delta"}:
        status = TranscriptBlockStatus.COMPLETE
        if event_type == "agent.message.delta":
            status = TranscriptBlockStatus.STREAMING
        return TranscriptBlock(
            block_id=block_id,
            kind=TranscriptBlockKind.ASSISTANT,
            status=status,
            title="Agent",
            content=text,
            turn_id=turn_id,
            sequence=sequence,
        )
    if event_type.startswith("tool."):
        status = _tool_status(event_type, metadata)
        detail = metadata.get("detail", "")
        if status != TranscriptBlockStatus.RUNNING and text:
            detail = text
        return TranscriptBlock(
            block_id=block_id,
            kind=TranscriptBlockKind.TOOL,
            status=status,
            title=_tool_title(event_type, metadata.get("name", "")),
            content=metadata.get("summary", ""),
            detail=detail,
            turn_id=turn_id,
            sequence=sequence,
        )
    if event_type == "approval.requested":
        return TranscriptBlock(
            block_id=block_id,
            kind=TranscriptBlockKind.APPROVAL,
            status=TranscriptBlockStatus.WAITING,
            title="Approval required",
            content=text,
            detail=metadata.get("summary", ""),
            turn_id=turn_id,
            sequence=sequence,
        )
    if event_type.startswith("reconciliation."):
        status = TranscriptBlockStatus.WAITING
        if event_type.endswith(".resolved"):
            status = TranscriptBlockStatus.COMPLETE
        return TranscriptBlock(
            block_id=block_id,
            kind=TranscriptBlockKind.RECONCILIATION,
            status=status,
            title="Recovery required",
            content=text,
            detail=metadata.get("summary", ""),
            turn_id=turn_id,
            sequence=sequence,
        )
    if event_type in {"guard.warning", "guard.tripped"}:
        status = TranscriptBlockStatus.GUARDED
        title = "Safety warning"
        if event_type == "guard.tripped":
            title = "Turn interrupted by safety guard"
        content = text
        if not content:
            content = metadata.get("reason", "")
        return TranscriptBlock(
            block_id=block_id,
            kind=TranscriptBlockKind.WARNING,
            status=status,
            title=title,
            content=content,
            detail=metadata.get("action", ""),
            turn_id=turn_id,
            sequence=sequence,
        )
    if event_type == "turn.failed":
        return TranscriptBlock(
            block_id=block_id,
            kind=TranscriptBlockKind.WARNING,
            status=TranscriptBlockStatus.FAILED,
            title="Turn failed",
            content=text,
            turn_id=turn_id,
            sequence=sequence,
        )
    if _is_lifecycle_event(event_type):
        return TranscriptBlock(
            block_id=block_id,
            kind=TranscriptBlockKind.SYSTEM,
            status=TranscriptBlockStatus.COMPLETE,
            title="Session",
            content=_lifecycle_text(event_type, metadata),
            turn_id=turn_id,
            sequence=sequence,
        )
    return None


def _block_id(
    event: Mapping[str, Any],
    event_type: str,
    sequence: int,
) -> str:
    if event_type in {"agent.message", "agent.message.delta"}:
        turn_id = _text(event.get("turn_id"))
        if turn_id:
            return "assistant:" + turn_id
    if event_type.startswith("tool."):
        tool_id = _first_identifier(
            event,
            (
                "tool_call_id",
                "tool_id",
                "call_id",
                "command_id",
                "tool_use_id",
                "id",
            ),
        )
        if tool_id:
            return "tool:" + tool_id
    if event_type == "approval.requested":
        approval_id = _metadata_identifier(event, "approval_id")
        if approval_id:
            return "approval:" + approval_id
    if event_type.startswith("reconciliation."):
        reconciliation_id = _metadata_identifier(
            event,
            "reconciliation_id",
        )
        if reconciliation_id:
            return "reconciliation:" + reconciliation_id
    event_id = _text(event.get("event_id"))
    if event_id:
        return "event:" + event_id
    return "sequence:" + str(sequence) + ":" + event_type


def _tool_status(
    event_type: str,
    metadata: Mapping[str, str],
) -> TranscriptBlockStatus:
    if metadata.get("is_error", "").casefold() == "true":
        return TranscriptBlockStatus.FAILED
    if event_type.endswith((".failed", ".error")):
        return TranscriptBlockStatus.FAILED
    if event_type.endswith((".completed", ".result", ".finished")):
        return TranscriptBlockStatus.COMPLETE
    return TranscriptBlockStatus.RUNNING


def _tool_title(event_type: str, name: str) -> str:
    if name:
        return "Tool · " + name
    label = event_type.removeprefix("tool.")
    if label in {
        "started",
        "running",
        "completed",
        "result",
        "finished",
        "failed",
        "error",
    }:
        return "Tool"
    for suffix in (
        ".started",
        ".running",
        ".completed",
        ".result",
        ".finished",
        ".failed",
        ".error",
    ):
        if label.endswith(suffix):
            label = label.removesuffix(suffix)
            break
    return "Tool · " + label.replace("_", " ").replace(".", " ")


def _is_lifecycle_event(event_type: str) -> bool:
    return event_type in {
        "checkpoint.created",
        "goal.completed",
        "routing.failover",
        "routing.selected",
        "session.started",
        "session.stopped",
        "sync.conflict",
    }


def _lifecycle_text(
    event_type: str,
    metadata: Mapping[str, str],
) -> str:
    labels = {
        "checkpoint.created": "Checkpoint saved",
        "goal.completed": "Goal completed",
        "routing.failover": "Provider failover started",
        "routing.selected": "Provider route selected",
        "session.started": "Session started",
        "session.stopped": "Session stopped",
        "sync.conflict": "Storage synchronization conflict",
    }
    label = labels[event_type]
    summary = metadata.get("summary", "")
    if summary:
        return label + " · " + summary
    return label


def _advance_event_identity(
    state: TranscriptState,
    sequence: int,
    event_id: str,
) -> TranscriptUpdate:
    next_state = replace(
        state,
        applied_sequences=_with_sequence(state, sequence),
        applied_event_ids=_with_event_id(state, event_id),
        latest_sequence=max(state.latest_sequence, sequence),
    )
    return _ignored_update(next_state)


def _ignored_update(state: TranscriptState) -> TranscriptUpdate:
    return TranscriptUpdate(
        state=state,
        mutations=(
            TranscriptMutation(
                kind=TranscriptMutationKind.IGNORE,
                block_id="",
            ),
        ),
    )


def _with_sequence(
    state: TranscriptState,
    sequence: int,
) -> frozenset[int]:
    if sequence <= 0:
        return state.applied_sequences
    values = set(state.applied_sequences)
    values.add(sequence)
    return frozenset(values)


def _with_event_id(
    state: TranscriptState,
    event_id: str,
) -> frozenset[str]:
    if not event_id:
        return state.applied_event_ids
    values = set(state.applied_event_ids)
    values.add(event_id)
    return frozenset(values)


def _first_identifier(
    event: Mapping[str, Any],
    keys: tuple[str, ...],
) -> str:
    for key in keys:
        value = _text(event.get(key))
        if value:
            return value
    metadata = event.get("metadata")
    if isinstance(metadata, Mapping):
        for key in keys:
            value = _text(metadata.get(key))
            if value:
                return value
    return ""


def _metadata_identifier(
    event: Mapping[str, Any],
    key: str,
) -> str:
    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    return _text(metadata.get(key))


def _display_event_type(event_type: str) -> str:
    return event_type.replace("_", " ").replace(".", " · ")


def _composer_lines(height: int, maximum: int) -> int:
    available = max(2, height // 4)
    return min(maximum, available)


def _theme_preference(
    value: ThemePreference | str,
) -> ThemePreference:
    if isinstance(value, ThemePreference):
        return value
    try:
        return ThemePreference(value)
    except ValueError:
        return ThemePreference.SYSTEM


def _relative_luminance(value: str) -> float:
    if len(value) != 7 or not value.startswith("#"):
        raise ValueError("color must use #RRGGBB")
    try:
        channels = [
            int(value[index : index + 2], 16) / 255
            for index in (1, 3, 5)
        ]
    except ValueError as error:
        raise ValueError("color must use #RRGGBB") from error
    converted = [_linear_channel(channel) for channel in channels]
    return (
        0.2126 * converted[0]
        + 0.7152 * converted[1]
        + 0.0722 * converted[2]
    )


def _linear_channel(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return math.pow((value + 0.055) / 1.055, 2.4)


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
