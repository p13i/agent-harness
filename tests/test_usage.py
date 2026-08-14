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
    assert set(results) == {"codex", "claude", "kimi", "grok"}
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
    monkeypatch.setattr(
        usage,
        "_probe_claude_cli",
        lambda: UsageSnapshot(
            "claude",
            None,
            False,
            {},
            "claude-code usage unavailable",
        ),
    )
    assert usage._probe_claude().error == (
        "credentials unavailable; claude-code usage unavailable"
    )
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
    assert usage._probe_claude().error == (
        "OSError; claude-code usage unavailable"
    )


def test_claude_probe_falls_back_to_headless_local_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(usage, "_claude_token", lambda: "expired-token")

    def unavailable(
        unused_url: str,
        unused_headers: dict[str, str],
    ) -> dict[str, object]:
        del unused_url
        del unused_headers
        raise urllib.error.HTTPError("https://example.invalid", 401, "", {}, None)

    monkeypatch.setattr(usage, "_json_get", unavailable)
    monkeypatch.setattr(
        usage,
        "_probe_claude_cli",
        lambda: UsageSnapshot(
            "claude",
            63.0,
            False,
            {
                "source": "claude-code-local-usage",
                "windows": [{"label": "week (all models)", "percent": 63.0}],
            },
        ),
    )
    snapshot = usage._probe_claude()
    assert snapshot.binding_percent == 63.0
    assert snapshot.error == ""
    assert snapshot.payload["source"] == "claude-code-local-usage"
    assert len(snapshot.payload["live_probe_error_sha256"]) == 64


def test_claude_headless_usage_parser_is_closed_and_redacted() -> None:
    result = """You are currently using your subscription to power your Claude Code usage

Current session: 0% used · resets later
Current week (all models): 63% used · resets later
Current week (Fable): 6% used · resets later

private diagnostic material that must not be retained
"""
    snapshot = usage._parse_claude_cli_usage(result)
    assert snapshot.binding_percent == 63.0
    assert snapshot.credits_engaged is False
    assert snapshot.payload == {
        "source": "claude-code-local-usage",
        "windows": [
            {"label": "session", "percent": 0.0},
            {"label": "week (all models)", "percent": 63.0},
            {"label": "week (Fable)", "percent": 6.0},
        ],
    }
    assert "private" not in str(snapshot.payload)

    missing_subscription = usage._parse_claude_cli_usage(
        "Current session: 0% used"
    )
    assert missing_subscription.binding_percent is None
    assert "subscription" in missing_subscription.error

    malformed_window = usage._parse_claude_cli_usage(
        usage.CLAUDE_SUBSCRIPTION_USAGE + "\nCurrent session: 101% used"
    )
    assert malformed_window.binding_percent is None
    assert "windows" in malformed_window.error


def test_claude_headless_usage_invocation_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess:
        captured["command"] = command
        captured.update(kwargs)
        result = (
            usage.CLAUDE_SUBSCRIPTION_USAGE
            + "\nCurrent session: 0% used"
            + "\nCurrent week (all models): 63% used"
        )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"is_error": False, "result": result}),
            "",
        )

    monkeypatch.setattr(usage, "_trusted_npx", lambda: "/usr/bin/npx")
    monkeypatch.setattr(
        usage,
        "provider_environment",
        lambda unused_provider: {
            "HOME": "/home/operator",
            "PATH": "/usr/bin:/bin",
            "npm_config_cache": "/home/operator/.cache/npm",
        },
    )
    monkeypatch.setattr(usage.subprocess, "run", run)
    snapshot = usage._probe_claude_cli()
    assert snapshot.binding_percent == 63.0
    command = captured["command"]
    assert isinstance(command, tuple)
    assert command[0] == "/usr/bin/npx"
    assert usage.CLAUDE_CODE_PACKAGE in command
    assert "--no-session-persistence" in command
    assert "--safe-mode" in command
    assert "--no-chrome" in command
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--tools") + 1] == ""
    assert command[-1] == "/usage"
    assert "shell" not in captured
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["CLAUDE_CODE_SAFE_MODE"] == "1"


def test_claude_headless_usage_invocation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(usage, "_trusted_npx", lambda: None)
    assert usage._probe_claude_cli().error == "npx unavailable"

    monkeypatch.setattr(usage, "_trusted_npx", lambda: "/usr/bin/npx")
    monkeypatch.setattr(
        usage,
        "provider_environment",
        lambda unused_provider: {
            "HOME": "/home/operator",
            "PATH": "/usr/bin:/bin",
        },
    )

    def launch_failure(
        *unused_args: object,
        **unused_kwargs: object,
    ) -> subprocess.CompletedProcess:
        del unused_args
        del unused_kwargs
        raise OSError("private launch detail")

    monkeypatch.setattr(usage.subprocess, "run", launch_failure)
    assert usage._probe_claude_cli().error == "claude-code usage launch failed"

    def environment_failure(unused_provider: str) -> dict[str, str]:
        del unused_provider
        raise RuntimeError("missing node")

    monkeypatch.setattr(usage, "provider_environment", environment_failure)
    assert usage._probe_claude_cli().error == "claude-code usage launch failed"

    monkeypatch.setattr(
        usage,
        "provider_environment",
        lambda unused_provider: {"PATH": "/usr/bin:/bin"},
    )

    outputs = (
        (1, "", "claude-code usage exited nonzero"),
        (0, "x" * (usage.CLAUDE_CLI_OUTPUT_LIMIT + 1), "exceeded the limit"),
        (0, "not JSON", "was not JSON"),
        (0, "[]", "returned an error"),
        (0, '{"is_error":true}', "returned an error"),
        (0, '{"is_error":false}', "result was unavailable"),
    )
    for returncode, stdout, expected in outputs:
        monkeypatch.setattr(
            usage.subprocess,
            "run",
            lambda *unused_args, _code=returncode, _stdout=stdout, **unused_kwargs: (
                subprocess.CompletedProcess(
                    ["npx"],
                    _code,
                    _stdout,
                    "private stderr",
                )
            ),
        )
        snapshot = usage._probe_claude_cli()
        assert expected in snapshot.error
        assert "private" not in snapshot.error


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

    monkeypatch.setattr(usage, "_grok_auth_present", lambda: True)
    assert usage.provider_auth_ready("grok") is True
    monkeypatch.setattr(usage, "_grok_auth_present", lambda: False)
    assert usage.provider_auth_ready("grok") is False


def test_grok_auth_ready_reports_full_headroom() -> None:
    snapshot = normalize_usage(
        "grok",
        {"auth_ready": True, "auth_source": "oauth", "detail": "oauth present"},
    )
    assert snapshot.provider == "grok"
    assert snapshot.binding_percent == 0.0
    assert snapshot.credits_engaged is False
    assert snapshot.payload["auth_ready"] is True


def test_grok_probe_missing_auth(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    snapshot = usage._probe_grok()
    assert snapshot.provider == "grok"
    assert snapshot.binding_percent is None
    assert "credentials" in snapshot.error


def test_grok_probe_present_auth(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    auth_dir = tmp_path / ".grok"
    auth_dir.mkdir()
    (auth_dir / "auth.json").write_text(
        '{"https://auth.x.ai::x": {"expires_at": 9999999999, "refresh_token": "r"}}',
        encoding="utf-8",
    )
    snapshot = usage._probe_grok()
    assert snapshot.binding_percent == 0.0
    assert snapshot.error == ""


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


def test_kimi_binding_is_maximum_derived_window() -> None:
    snapshot = normalize_usage(
        "kimi",
        {
            "limits": [
                {
                    "window": {
                        "timeUnit": "TIME_UNIT_MINUTE",
                        "duration": 300,
                    },
                    "detail": {
                        "limit": "100",
                        "remaining": "60",
                        "resetTime": "later",
                    },
                },
                {
                    "window": {
                        "timeUnit": "TIME_UNIT_DAY",
                        "duration": 1,
                    },
                    "detail": {"limit": "100", "remaining": "0"},
                },
                "not-a-dict",
                {
                    "window": {
                        "timeUnit": "TIME_UNIT_MINUTE",
                        "duration": 90,
                    },
                    "detail": {"limit": "0", "remaining": "0"},
                },
            ],
            "usage": {
                "limit": "200",
                "remaining": "100",
                "resetTime": "later",
            },
            "user": {"membership": {"level": "LEVEL_PRO"}},
        },
    )
    # The 5-hour window derives 40 percent and the weekly window 50.
    assert snapshot.binding_percent == 50
    assert not snapshot.credits_engaged
    assert snapshot.payload["windows"] == [
        {"label": "5-hour", "percent": 40.0, "resets_at": "later"},
        {"label": "weekly", "percent": 50.0, "resets_at": "later"},
    ]
    assert snapshot.payload["extra_usage"] == {"engaged": False}
    assert snapshot.payload["membership_level"] == "LEVEL_PRO"


def test_kimi_spent_weekly_quota_engages_extra_usage() -> None:
    snapshot = normalize_usage(
        "kimi",
        {"usage": {"limit": "200", "remaining": "0"}},
    )
    assert snapshot.binding_percent == 100
    assert snapshot.credits_engaged
    assert snapshot.payload["extra_usage"] == {"engaged": True}


def test_kimi_empty_payload_has_no_binding() -> None:
    snapshot = normalize_usage("kimi", {})
    assert snapshot.binding_percent is None
    assert not snapshot.credits_engaged


def test_kimi_quota_derivation_is_defensive() -> None:
    assert usage._kimi_quota(None, "weekly") is None
    assert usage._kimi_quota({"limit": "0"}, "weekly") is None
    assert usage._kimi_quota({"limit": "abc"}, "weekly") is None
    assert usage._kimi_quota(
        {"limit": "100", "remaining": "25", "resetTime": "later"},
        "weekly",
    ) == {"label": "weekly", "percent": 75.0, "resets_at": "later"}
    assert usage._kimi_window_label(300) == "5-hour"
    assert usage._kimi_window_label(90) == "90-minute"
    assert usage._kimi_window_label(0) == "0-minute"
    assert usage._kimi_quantity(True) == 0.0
    assert usage._kimi_quantity(None) == 0.0
    assert usage._kimi_quantity("nan") == 0.0
    assert usage._kimi_quantity("5") == 5.0


def test_kimi_credentials_resolve_config_key_then_oauth_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(usage.Path, "home", lambda: tmp_path)
    assert usage._kimi_credential() is None
    assert usage._probe_kimi().error == "credentials unavailable"

    config = tmp_path / ".kimi-code" / "config.toml"
    config.parent.mkdir()
    config.write_text("not = [toml", encoding="utf-8")
    assert usage._kimi_config_api_key() is None

    config.write_text(
        '[providers."managed:kimi-code"]\napi_key = 7\n',
        encoding="utf-8",
    )
    assert usage._kimi_config_api_key() is None

    config.write_text(
        '[providers."managed:kimi-code"]\napi_key = "  "\n',
        encoding="utf-8",
    )
    assert usage._kimi_config_api_key() is None

    credentials = tmp_path / ".kimi-code" / "credentials" / "kimi-code.json"
    credentials.parent.mkdir()
    credentials.write_text("{not json", encoding="utf-8")
    assert usage._kimi_oauth_token() is None
    credentials.write_text("[1]", encoding="utf-8")
    assert usage._kimi_oauth_token() is None
    credentials.write_text(
        json.dumps({"access_token": "  "}),
        encoding="utf-8",
    )
    assert usage._kimi_oauth_token() is None
    credentials.write_text(
        json.dumps({"access_token": "oauth-token"}),
        encoding="utf-8",
    )
    assert usage._kimi_credential() == "oauth-token"

    config.write_text(
        '[providers."managed:kimi-code"]\napi_key = "console-key"\n',
        encoding="utf-8",
    )
    assert usage._kimi_config_api_key() == "console-key"
    assert usage._kimi_credential() == "console-key"


def test_kimi_probe_reports_http_errors_without_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(usage, "_kimi_credential", lambda: "credential")

    def expired(
        unused_url: str,
        unused_headers: dict[str, str],
    ) -> dict[str, object]:
        del unused_url
        del unused_headers
        raise urllib.error.HTTPError(
            "https://example.invalid",
            401,
            "expired",
            {},
            None,
        )

    monkeypatch.setattr(usage, "_json_get", expired)
    assert usage._probe_kimi().error == "HTTP 401"

    monkeypatch.setattr(
        usage,
        "_json_get",
        lambda unused_url, unused_headers: {
            "usage": {"limit": "100", "remaining": "80"},
        },
    )
    snapshot = usage._probe_kimi()
    assert snapshot.binding_percent == 20
    assert not snapshot.credits_engaged
