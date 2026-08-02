import asyncio
import json
import math
import subprocess
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_harness import usage
from agent_harness.usage import UsageSnapshot, normalize_usage


def test_codex_binding_is_maximum_window() -> None:
    snapshot = normalize_usage(
        "codex",
        {
            "rate_limit": {
                "primary_window": {"used_percent": 25},
                "secondary_window": {"used_percent": 70},
            }
        },
    )
    assert snapshot.binding_percent == 70


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
def test_usage_normalization_rejects_malformed_binding(value: float) -> None:
    snapshot = normalize_usage(
        "codex",
        {"rate_limit": {"primary_window": {"used_percent": value}}},
    )
    assert snapshot.binding_percent is None


def test_claude_extra_usage_is_metered() -> None:
    snapshot = normalize_usage(
        "claude",
        {
            "five_hour": {"utilization": 20},
            "seven_day": {"utilization": 100},
            "extra_usage": {"is_enabled": True},
        },
    )
    assert snapshot.binding_percent == 100
    assert snapshot.credits_engaged


def test_usage_probes_are_bounded_and_provider_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_timed_probe = usage._timed_probe

    async def timed(provider: str, probe: object) -> UsageSnapshot:
        del probe
        return UsageSnapshot(provider, 25, False, {})

    monkeypatch.setattr(usage, "_timed_probe", timed)
    results = asyncio.run(usage.probe_all())
    assert set(results) == {"codex", "claude"}
    monkeypatch.setattr(usage, "_timed_probe", original_timed_probe)

    assert (
        asyncio.run(
            usage._timed_probe(
                "codex",
                lambda: UsageSnapshot("codex", 10, False, {}),
            )
        ).binding_percent
        == 10
    )

    async def timeout(coroutine: object, *, timeout: float) -> object:
        del timeout
        close = getattr(coroutine, "close")
        close()
        raise TimeoutError

    monkeypatch.setattr(usage.asyncio, "wait_for", timeout)
    timed_out = asyncio.run(usage._timed_probe("claude", lambda: None))
    assert timed_out.error == "probe timed out"


def test_provider_probes_handle_credentials_success_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        usage,
        "_codex_credentials",
        lambda: (None, None),
    )
    assert usage._probe_codex().error == "credentials unavailable"
    monkeypatch.setattr(
        usage,
        "_codex_credentials",
        lambda: ("token", "account"),
    )
    monkeypatch.setattr(
        usage,
        "_json_get",
        lambda unused_url, unused_headers: {
            "rate_limit": {
                "primary_window": {"used_percent": 30},
            }
        },
    )
    assert usage._probe_codex().binding_percent == 30

    def unavailable(
        unused_url: str,
        unused_headers: dict[str, str],
    ) -> dict[str, object]:
        del unused_url
        del unused_headers
        raise OSError("unavailable")

    monkeypatch.setattr(usage, "_json_get", unavailable)
    assert usage._probe_codex().error == "OSError"

    monkeypatch.setattr(usage, "_claude_token", lambda: None)
    assert usage._probe_claude().error == "credentials unavailable"
    monkeypatch.setattr(usage, "_claude_token", lambda: "token")
    monkeypatch.setattr(
        usage,
        "_json_get",
        lambda unused_url, unused_headers: {
            "five_hour": {"utilization": 40},
        },
    )
    assert usage._probe_claude().binding_percent == 40
    monkeypatch.setattr(usage, "_json_get", unavailable)
    assert usage._probe_claude().error == "OSError"


def test_usage_http_and_credential_parsers_are_defensive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(usage.sys, "platform", "darwin")

    class Response:
        def __init__(self, value: object) -> None:
            self.value = value

        def __enter__(self):
            return self

        def __exit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> None:
            del exception_type
            del exception
            del traceback

        def read(self) -> bytes:
            return json.dumps(self.value).encode("utf-8")

    monkeypatch.setattr(
        usage.urllib.request,
        "urlopen",
        lambda unused_request, timeout: Response({"value": 1}),
    )
    assert usage._json_get("https://example.invalid", {}) == {"value": 1}
    monkeypatch.setattr(
        usage.urllib.request,
        "urlopen",
        lambda unused_request, timeout: Response([]),
    )
    with pytest.raises(json.JSONDecodeError):
        usage._json_get("https://example.invalid", {})

    monkeypatch.setattr(usage.Path, "home", lambda: tmp_path)
    assert usage._codex_credentials() == (None, None)
    auth = tmp_path / ".codex" / "auth.json"
    auth.parent.mkdir()
    auth.write_text("invalid", encoding="utf-8")
    assert usage._codex_credentials() == (None, None)
    auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": 7,
                    "account_id": [],
                }
            }
        ),
        encoding="utf-8",
    )
    assert usage._codex_credentials() == (None, None)
    auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "token",
                    "account_id": "account",
                }
            }
        ),
        encoding="utf-8",
    )
    assert usage._codex_credentials() == ("token", "account")

    monkeypatch.setattr(
        usage.subprocess,
        "run",
        lambda *unused_args, **unused_kwargs: subprocess.CompletedProcess(
            ["security"],
            1,
            "",
            "",
        ),
    )
    assert usage._claude_token() is None
    monkeypatch.setattr(
        usage.subprocess,
        "run",
        lambda *unused_args, **unused_kwargs: subprocess.CompletedProcess(
            ["security"],
            0,
            "invalid",
            "",
        ),
    )
    assert usage._claude_token() is None
    monkeypatch.setattr(
        usage.subprocess,
        "run",
        lambda *unused_args, **unused_kwargs: subprocess.CompletedProcess(
            ["security"],
            0,
            '{"claudeAiOauth":{"accessToken":7}}',
            "",
        ),
    )
    assert usage._claude_token() is None
    monkeypatch.setattr(
        usage.subprocess,
        "run",
        lambda *unused_args, **unused_kwargs: subprocess.CompletedProcess(
            ["security"],
            0,
            '{"claudeAiOauth":{"accessToken":"token"}}',
            "",
        ),
    )
    assert usage._claude_token() == "token"

    def missing_security(
        *unused_args: object,
        **unused_kwargs: object,
    ) -> SimpleNamespace:
        del unused_args
        del unused_kwargs
        raise OSError("missing")

    monkeypatch.setattr(usage.subprocess, "run", missing_security)
    assert usage._claude_token() is None

    error = urllib.error.HTTPError(
        "https://example.invalid",
        429,
        "limited",
        {},
        None,
    )
    assert usage._safe_error(error) == "HTTP 429"


def test_claude_linux_credentials_are_read_without_projecting_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(usage.sys, "platform", "linux")
    monkeypatch.setattr(usage.Path, "home", lambda: tmp_path)
    credentials = tmp_path / ".claude" / ".credentials.json"
    credentials.parent.mkdir()
    credentials.write_text(
        '{"claudeAiOauth":{"accessToken":"linux-token"}}',
        encoding="utf-8",
    )
    credentials.chmod(0o600)
    assert usage._claude_token() == "linux-token"
    credentials.chmod(0o644)
    assert usage._claude_token() is None
    credentials.chmod(0o600)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "environment-token")
    assert usage._claude_token() == "environment-token"


def test_claude_credential_files_fail_closed_on_unsafe_material(
    tmp_path: Path,
) -> None:
    linked = tmp_path / "linked.json"
    linked.symlink_to(tmp_path / "absent.json")
    assert usage._claude_file_token(linked) is None

    oversized = tmp_path / "oversized.json"
    oversized.write_text("{}" + " " * (1024 * 1024), encoding="utf-8")
    oversized.chmod(0o600)
    assert usage._claude_file_token(oversized) is None

    assert usage._claude_file_token(tmp_path / "absent.json") is None

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    malformed.chmod(0o600)
    assert usage._claude_file_token(malformed) is None

    blank = tmp_path / "blank.json"
    blank.write_text('{"claudeAiOauth":{"accessToken":"  "}}', encoding="utf-8")
    blank.chmod(0o600)
    assert usage._claude_file_token(blank) is None


def test_provider_auth_readiness_reports_each_launch_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(usage, "_claude_token", lambda: "token")
    assert usage.provider_auth_ready("claude") is True
    monkeypatch.setattr(usage, "_claude_token", lambda: None)
    assert usage.provider_auth_ready("claude") is False

    monkeypatch.setattr(
        usage,
        "_codex_credentials",
        lambda: ("token", "account"),
    )
    assert usage.provider_auth_ready("codex") is True
    monkeypatch.setattr(usage, "_codex_credentials", lambda: ("token", None))
    assert usage.provider_auth_ready("codex") is False

    with pytest.raises(ValueError, match="unknown provider: kimi"):
        usage.provider_auth_ready("kimi")


def test_usage_normalization_omits_private_payloads() -> None:
    codex = normalize_usage(
        "codex",
        {
            "rate_limit": {
                "allowed": True,
                "limit_reached": True,
                "primary_window": {
                    "used_percent": 100,
                    "reset_at": "later",
                    "secret": "hidden",
                },
            },
            "credits": {
                "has_credits": True,
                "unlimited": False,
                "overage_limit_reached": False,
                "balance": 100,
            },
            "token": "secret",
        },
    )
    assert codex.credits_engaged
    assert "token" not in codex.payload
    assert "secret" not in str(codex.payload)
    assert usage._number(True) is None
    assert usage._number("unknown") is None
    assert usage._object("unknown") == {}
    assert usage._usage_window(None, "utilization") is None
    assert UsageSnapshot("test", None, False, {}).as_dict()["provider"] == ("test")
