import asyncio
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_harness import ui_gallery
from agent_harness.ui_gallery import (
    BREAKPOINTS,
    FIXTURES,
    MODES,
    THEMES,
    GalleryClient,
    _render_mode_theme,
    _self_contained_svg,
    _validate_layout,
    render_gallery,
    render_svg,
)


def test_gallery_renders_every_normalized_surface(tmp_path: Path) -> None:
    gallery = render_gallery(tmp_path / "gallery")

    expected = len(FIXTURES) * len(MODES) * len(THEMES) * len(BREAKPOINTS)
    assert len(gallery["files"]) == expected
    assert gallery["renderer"] == "textual-widget-tree"
    for name in gallery["files"]:
        value = (tmp_path / "gallery" / name).read_text(encoding="utf-8")
        assert value.startswith("<svg")
        assert "P13I" in value
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


def test_gallery_client_static_request_boundaries() -> None:
    client = GalleryClient("empty", "dark")

    async def requests() -> None:
        assert "codex" in (await client.request("GET", "/v1/providers"))["providers"]
        assert "safety" in await client.request(
            "GET",
            "/v1/sessions/gallery-session/usage",
        )
        assert "safety" in await client.request(
            "GET",
            "/v1/sessions/gallery-session/budget-extensions",
        )
        assert "session" in await client.request(
            "PATCH",
            "/v1/sessions/gallery-session",
        )
        assert "sync" in await client.request("POST", "/v1/sync")
        assert await client.request(
            "GET",
            "/v1/sessions/gallery-session/events?after=42",
        ) == {"events": []}
        assert await client.request("GET", "/unknown") == {}

    asyncio.run(requests())


def test_gallery_disconnected_svg_and_secret_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "P13I" in render_svg(
        "disconnected",
        "dark",
        80,
        24,
    )
    monkeypatch.setattr(
        ui_gallery.HarnessApp,
        "export_screenshot",
        lambda unused_self: "<svg>secret</svg>",
    )
    with pytest.raises(ValueError, match="secret-bearing"):
        asyncio.run(
            _render_mode_theme(
                tmp_path,
                "empty",
                "focus",
                "dark",
            )
        )


@pytest.mark.parametrize(
    "failure",
    [
        "topbar",
        "clipped",
        "geometry",
        "notification",
        "focus",
        "brand",
    ],
)
def test_gallery_layout_validation_boundaries(failure: str) -> None:
    composer_region = SimpleNamespace(
        bottom=20,
        y=17,
        width=80,
        height=3,
    )
    body_region = SimpleNamespace(y=3)
    topbar_region = SimpleNamespace(bottom=3)
    notification_region = SimpleNamespace(bottom=16)
    screen_region = SimpleNamespace(bottom=24)
    notification = SimpleNamespace(
        display=False,
        region=notification_region,
    )

    class Brand:
        def render(self) -> str:
            return "P13I"

    brand: object = Brand()
    focused: object | None = object()
    if failure == "topbar":
        topbar_region.bottom = 4
    elif failure == "clipped":
        composer_region.bottom = 25
    elif failure == "geometry":
        composer_region.width = 19
    elif failure == "notification":
        notification.display = True
        notification_region.bottom = 18
    elif failure == "focus":
        focused = None
    else:
        brand = SimpleNamespace(render=lambda: "other")

    widgets = {
        "#composer-shell": SimpleNamespace(region=composer_region),
        "#body": SimpleNamespace(region=body_region),
        "#topbar": SimpleNamespace(region=topbar_region),
        "#notification-shell": notification,
        "#brand": brand,
    }

    class App:
        screen = SimpleNamespace(region=screen_region)

        def __init__(self) -> None:
            self.focused = focused

        def query_one(
            self,
            selector: str,
            unused_type: object | None = None,
        ) -> object:
            del unused_type
            return widgets[selector]

    with pytest.raises(ValueError):
        _validate_layout(App())  # type: ignore[arg-type]


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
        _self_contained_svg('<svg><image href="http://example.test/image.png"/></svg>')


def test_gallery_main_and_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        ui_gallery,
        "render_gallery",
        lambda output: {"output": str(output)},
    )
    assert ui_gallery.main(["--output", str(tmp_path)]) == 0
    assert str(tmp_path) in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["ui-gallery"])
    with pytest.raises(SystemExit, match="2"):
        runpy.run_module(
            "agent_harness.ui_gallery",
            run_name="__main__",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
