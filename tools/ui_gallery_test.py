from pathlib import Path

import pytest

from tools.ui_gallery import BREAKPOINTS
from tools.ui_gallery import FIXTURES
from tools.ui_gallery import MODES
from tools.ui_gallery import _self_contained_svg
from tools.ui_gallery import render_gallery
from tools.ui_gallery import render_svg
from tools.ui_gallery import THEMES


def test_gallery_renders_every_normalized_surface(tmp_path: Path) -> None:
    gallery = render_gallery(tmp_path / "gallery")

    expected = (
        len(FIXTURES)
        * len(MODES)
        * len(THEMES)
        * len(BREAKPOINTS)
    )
    assert len(gallery["files"]) == expected
    assert gallery["renderer"] == "textual-widget-tree"
    for name in gallery["files"]:
        value = (tmp_path / "gallery" / name).read_text(
            encoding="utf-8"
        )
        assert value.startswith("<svg")
        assert "P13I AGENT HARNESS" in value
        assert "secret" not in value.casefold()
        assert 'xmlns="http://www.w3.org/2000/svg"' in value
        resources = value.replace(
            'xmlns="http://www.w3.org/2000/svg"',
            "",
        )
        assert "http://" not in resources.casefold()
        assert "https://" not in resources.casefold()


def test_gallery_is_deterministic_for_one_actual_widget_tree() -> None:
    first = render_svg("approval", "dark", 80, 24, mode="control")
    second = render_svg("approval", "dark", 80, 24, mode="control")

    assert first == second
    assert "Approval needed" in first
    assert "Control" in first


def test_gallery_rejects_unknown_fixture_and_theme() -> None:
    with pytest.raises(ValueError, match="fixture"):
        render_svg("unknown", "dark", 80, 24)
    with pytest.raises(ValueError, match="theme"):
        render_svg("empty", "sepia", 80, 24)
    with pytest.raises(ValueError, match="mode"):
        render_svg("empty", "dark", 80, 24, mode="observe")
    with pytest.raises(ValueError, match="breakpoint"):
        render_svg("empty", "dark", 81, 24)


def test_gallery_removes_only_known_export_resources() -> None:
    source = (
        "<svg><!-- Generated with Rich https://www.textualize.io -->"
        "<style>@font-face {src: "
        'url("https://cdn.example/font.woff");}'
        ".local {fill: red;}</style></svg>"
    )
    rendered = _self_contained_svg(source)

    assert "<!-- Generated with Rich -->" in rendered
    assert ".local {fill: red;}" in rendered
    assert "http" not in rendered

    with pytest.raises(ValueError, match="outbound"):
        _self_contained_svg(
            '<svg><image href="http://example.test/image.png"/></svg>'
        )
