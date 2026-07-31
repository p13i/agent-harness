"""Render deterministic agent-harness interface galleries."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from agent_harness.tui_presenter import decide_layout
from agent_harness.tui_presenter import resolve_theme


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
BREAKPOINTS = (
    (60, 20),
    (80, 24),
    (120, 36),
    (160, 48),
)
THEMES = ("light", "dark")


def render_gallery(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for fixture in FIXTURES:
        for theme in THEMES:
            for width, height in BREAKPOINTS:
                name = (
                    fixture
                    + "-"
                    + theme
                    + "-"
                    + str(width)
                    + "x"
                    + str(height)
                    + ".svg"
                )
                content = render_svg(fixture, theme, width, height)
                if "secret" in content.casefold():
                    raise ValueError("gallery contains secret-bearing text")
                path = output / name
                path.write_text(content, encoding="utf-8")
                files.append(name)
    manifest = {
        "schema": "p13i/agent-harness/ui-gallery/v1",
        "fixtures": list(FIXTURES),
        "themes": list(THEMES),
        "breakpoints": [
            {"width": width, "height": height}
            for width, height in BREAKPOINTS
        ],
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def render_svg(
    fixture: str,
    theme: str,
    columns: int,
    rows: int,
) -> str:
    if fixture not in FIXTURES:
        raise ValueError("unknown gallery fixture")
    if theme not in THEMES:
        raise ValueError("unknown gallery theme")
    layout = decide_layout(columns, rows)
    tokens = resolve_theme(theme, system_dark=False)
    scale = 8
    width = columns * scale
    height = rows * 16
    sidebar_width = 0
    if layout.sidebar_visible:
        sidebar_width = min(248, max(176, width // 5))
    inspector_width = 0
    if layout.inspector_visible:
        inspector_width = min(288, max(232, width // 5))
    main_x = sidebar_width
    main_width = width - sidebar_width - inspector_width
    if main_width < 320:
        main_width = width
        main_x = 0
    colors = tokens.colors
    status_color = colors.success
    if fixture == "approval":
        status_color = colors.approval
    elif fixture == "guarded":
        status_color = colors.danger
    elif fixture == "disconnected":
        status_color = colors.warning
    elif fixture == "reconciliation":
        status_color = colors.reconciliation
    transcript = _fixture_lines(fixture)
    composer_y = height - 104
    blocks = _transcript_blocks(
        transcript,
        main_x + 20,
        96,
        main_width - 40,
        colors,
        status_color,
        composer_y - 12,
    )
    sidebar = ""
    if sidebar_width:
        sidebar = _sidebar(sidebar_width, height, colors, fixture)
    inspector = ""
    if inspector_width:
        inspector = _inspector(
            width - inspector_width,
            inspector_width,
            height,
            colors,
            fixture,
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="'
        + str(width)
        + '" height="'
        + str(height)
        + '" viewBox="0 0 '
        + str(width)
        + " "
        + str(height)
        + '">'
        + '<rect width="100%" height="100%" fill="'
        + colors.canvas
        + '"/>'
        + '<rect width="100%" height="48" fill="'
        + colors.surface
        + '"/>'
        + _text(16, 30, "P13I AGENT HARNESS", colors.active, 14, True)
        + _text(
            width - 116,
            30,
            "● " + _fixture_status(fixture),
            status_color,
            12,
            False,
        )
        + sidebar
        + inspector
        + _text(
            main_x + 20,
            76,
            _fixture_title(fixture),
            colors.text,
            16,
            True,
        )
        + blocks
        + _composer(
            main_x + 20,
            composer_y,
            main_width - 40,
            colors,
            fixture,
        )
        + "</svg>\n"
    )


def _sidebar(width: int, height: int, colors, fixture: str) -> str:
    labels = (
        "NEEDS ATTENTION",
        "◆ Durable recovery",
        "ACTIVE",
        "● Agent harness",
        "RECENT",
        "○ API contract",
    )
    content = (
        '<rect y="48" width="'
        + str(width)
        + '" height="'
        + str(height - 48)
        + '" fill="'
        + colors.surface
        + '"/>'
        + _text(16, 76, "WORKSPACE", colors.text_muted, 10, True)
        + _text(16, 98, "agent-harness", colors.text, 13, True)
        + _text(16, 126, "Search sessions", colors.text_muted, 11, False)
    )
    y = 166
    for label in labels:
        color = colors.text
        size = 12
        bold = False
        if label.isupper():
            color = colors.text_muted
            size = 10
            bold = True
        content += _text(16, y, label, color, size, bold)
        y += 24
    if fixture == "archived":
        content += _text(16, y, "ARCHIVED", colors.text_muted, 10, True)
        content += _text(
            16,
            y + 24,
            "○ Migration notes",
            colors.text,
            12,
            False,
        )
    return content


def _inspector(
    x: int,
    width: int,
    height: int,
    colors,
    fixture: str,
) -> str:
    content = (
        '<rect x="'
        + str(x)
        + '" y="48" width="'
        + str(width)
        + '" height="'
        + str(height - 48)
        + '" fill="'
        + colors.surface
        + '"/>'
        + _text(
            x + 16,
            76,
            "CONTEXT  GOAL  USAGE",
            colors.text_muted,
            10,
            True,
        )
    )
    if fixture == "reconciliation":
        lines = (
            "RECOVERY",
            "Interrupted command",
            "checkpoint-01",
            "one changed file",
            "accept · restore · stop",
        )
    else:
        lines = (
            "ROUTING",
            "provider  automatic",
            "effort    bounded",
            "",
            "STORAGE",
            "state     synchronized",
            "session   resume UUID",
        )
    y = 112
    for line in lines:
        color = colors.text
        if line.isupper():
            color = colors.active
        content += _text(x + 16, y, line, color, 11, line.isupper())
        y += 24
    return content


def _transcript_blocks(
    lines: tuple[tuple[str, str], ...],
    x: int,
    y: int,
    width: int,
    colors,
    status_color: str,
    maximum_y: int,
) -> str:
    content = ""
    current_y = y
    previous_end = y
    for index, (title, body) in enumerate(lines):
        block_height = 68
        if "\n" in body:
            block_height = 88
        if current_y + block_height > maximum_y:
            marker_y = current_y
            if marker_y + 28 > maximum_y:
                marker_y = maximum_y - 20
            marker_y = max(previous_end + 4, marker_y)
            marker_height = maximum_y - marker_y
            if marker_height >= 20:
                remaining = len(lines) - index
                content += (
                    '<rect x="'
                    + str(x)
                    + '" y="'
                    + str(marker_y)
                    + '" width="'
                    + str(max(120, width))
                    + '" height="'
                    + str(marker_height)
                    + '" rx="5" fill="'
                    + colors.surface
                    + '" stroke="'
                    + colors.border
                    + '"/>'
                    + _text(
                        x + 14,
                        marker_y + min(19, marker_height - 5),
                        str(remaining)
                        + " more blocks · scroll transcript",
                        colors.text_muted,
                        10,
                        True,
                    )
                )
            break
        border = colors.border
        title_color = colors.text_muted
        if title == "AGENT":
            border = colors.success
            title_color = colors.success
        if title in {"RECOVERY", "APPROVAL", "WARNING"}:
            border = status_color
            title_color = status_color
        content += (
            '<rect x="'
            + str(x)
            + '" y="'
            + str(current_y)
            + '" width="'
            + str(max(120, width))
            + '" height="'
            + str(block_height)
            + '" rx="5" fill="'
            + colors.surface
            + '" stroke="'
            + border
            + '"/>'
        )
        content += _text(
            x + 14,
            current_y + 22,
            title,
            title_color,
            10,
            True,
        )
        body_y = current_y + 44
        for body_line in body.splitlines():
            content += _text(
                x + 14,
                body_y,
                body_line,
                colors.text,
                12,
                False,
            )
            body_y += 17
        previous_end = current_y + block_height
        current_y += block_height + 12
    return content


def _composer(
    x: int,
    y: int,
    width: int,
    colors,
    fixture: str,
) -> str:
    draft = "Ask, build, debug, or steer the active agent…"
    if fixture == "new-session":
        draft = "Describe the first bounded outcome…"
    return (
        '<rect x="'
        + str(x)
        + '" y="'
        + str(y)
        + '" width="'
        + str(max(120, width))
        + '" height="76" rx="8" fill="'
        + colors.surface
        + '" stroke="'
        + colors.focus
        + '"/>'
        + _text(x + 14, y + 22, "MESSAGE", colors.active, 10, True)
        + _text(x + 14, y + 46, draft, colors.text_muted, 12, False)
        + _text(
            x + 14,
            y + 66,
            "Enter send · Shift+Enter newline · interactive · approval",
            colors.text_muted,
            9,
            False,
        )
    )


def _fixture_lines(fixture: str) -> tuple[tuple[str, str], ...]:
    if fixture == "empty":
        return (("SYSTEM", "Durable session ready"),)
    if fixture == "new-session":
        return (("SYSTEM", "No model starts until you send a message."),)
    if fixture == "streaming":
        return (
            ("YOU", "Implement the public session contract."),
            ("AGENT", "I am validating the migration and API…"),
        )
    if fixture == "tool-heavy":
        return (
            ("AGENT", "I am running bounded local checks."),
            ("TOOL · TEST", "120 passed · details collapsed"),
            ("TOOL · BUILD", "bundle verified · details collapsed"),
        )
    if fixture == "approval":
        return (("APPROVAL", "Restore the pre-turn checkpoint?"),)
    if fixture == "guarded":
        return (("WARNING", "Turn paused at the safety envelope."),)
    if fixture == "disconnected":
        return (("SYSTEM", "Reconnecting · original send retained"),)
    if fixture == "reconciliation":
        return (
            (
                "RECOVERY",
                "Provider dispatch was interrupted.\n"
                "Inspect before accepting, restoring, or stopping.",
            ),
        )
    if fixture == "long-code":
        return (
            (
                "AGENT",
                "def bounded_turn():\n"
                "    return checkpoint_before_dispatch()",
            ),
        )
    if fixture == "archived":
        return (("SYSTEM", "Archived sessions remain searchable."),)
    return (
        ("SYSTEM", "32 sessions · attention grouped first"),
        ("AGENT", "The active stream remains stable."),
    )


def _fixture_title(fixture: str) -> str:
    return fixture.replace("-", " ").title()


def _fixture_status(fixture: str) -> str:
    if fixture == "disconnected":
        return "reconnecting"
    if fixture == "reconciliation":
        return "action needed"
    if fixture == "approval":
        return "approval"
    if fixture == "guarded":
        return "guarded"
    return "connected"


def _text(
    x: int,
    y: int,
    value: str,
    color: str,
    size: int,
    bold: bool,
) -> str:
    weight = "400"
    if bold:
        weight = "700"
    return (
        '<text x="'
        + str(x)
        + '" y="'
        + str(y)
        + '" fill="'
        + color
        + '" font-family="ui-monospace, monospace" font-size="'
        + str(size)
        + '" font-weight="'
        + weight
        + '">'
        + html.escape(value)
        + "</text>"
    )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    values = parser.parse_args(arguments)
    manifest = render_gallery(values.output)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
