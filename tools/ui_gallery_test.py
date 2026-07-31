from pathlib import Path
import xml.etree.ElementTree as ElementTree

import pytest

from tools.ui_gallery import BREAKPOINTS
from tools.ui_gallery import FIXTURES
from tools.ui_gallery import render_gallery
from tools.ui_gallery import render_svg
from tools.ui_gallery import THEMES


def test_gallery_renders_every_normalized_surface(tmp_path: Path) -> None:
    first = render_gallery(tmp_path / "first")
    second = render_gallery(tmp_path / "second")

    expected = len(FIXTURES) * len(THEMES) * len(BREAKPOINTS)
    assert len(first["files"]) == expected
    assert first == second
    for name in first["files"]:
        first_value = (tmp_path / "first" / name).read_text(
            encoding="utf-8"
        )
        second_value = (tmp_path / "second" / name).read_text(
            encoding="utf-8"
        )
        assert first_value == second_value
        assert first_value.startswith("<svg")
        assert "P13I AGENT HARNESS" in first_value
        assert "secret" not in first_value.casefold()
        root = ElementTree.fromstring(first_value)
        composer_y = int(root.attrib["height"]) - 104
        for rectangle in root.findall(
            "{http://www.w3.org/2000/svg}rect"
        ):
            if rectangle.attrib.get("rx") != "5":
                continue
            rectangle_end = int(rectangle.attrib["y"]) + int(
                rectangle.attrib["height"]
            )
            assert rectangle_end <= composer_y - 12


def test_gallery_rejects_unknown_fixture_and_theme() -> None:
    with pytest.raises(ValueError, match="fixture"):
        render_svg("unknown", "dark", 80, 24)
    with pytest.raises(ValueError, match="theme"):
        render_svg("empty", "sepia", 80, 24)
