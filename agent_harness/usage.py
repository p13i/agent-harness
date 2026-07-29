"""Read-only subscription capacity probes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any
import urllib.error
import urllib.request


CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"


@dataclass(frozen=True)
class UsageSnapshot:
    provider: str
    binding_percent: float | None
    credits_engaged: bool
    payload: dict[str, Any]
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "binding_percent": self.binding_percent,
            "credits_engaged": self.credits_engaged,
            "payload": self.payload,
            "error": self.error,
        }


async def probe_all() -> dict[str, UsageSnapshot]:
    results = await asyncio.gather(
        asyncio.to_thread(_probe_codex),
        asyncio.to_thread(_probe_claude),
    )
    return {item.provider: item for item in results}


def normalize_usage(
    provider: str,
    payload: dict[str, Any],
) -> UsageSnapshot:
    windows: list[float] = []
    credits_engaged = False
    if provider == "codex":
        rate_limit = _object(payload.get("rate_limit"))
        for name in ("primary_window", "secondary_window"):
            window = _object(rate_limit.get(name))
            percent = _number(window.get("used_percent"))
            if percent is not None:
                windows.append(percent)
        credits = _object(payload.get("credits"))
        has_credits = bool(credits.get("has_credits", False))
        limit_reached = bool(rate_limit.get("limit_reached", False))
        credits_engaged = has_credits and limit_reached
    if provider == "claude":
        for name in (
            "five_hour",
            "seven_day",
            "seven_day_opus",
            "seven_day_sonnet",
        ):
            window = _object(payload.get(name))
            percent = _number(window.get("utilization"))
            if percent is not None:
                windows.append(percent)
        extra = _object(payload.get("extra_usage"))
        extra_active = bool(extra.get("is_enabled", False))
        if windows:
            credits_engaged = extra_active and max(windows) >= 100.0
    binding_percent: float | None = None
    if windows:
        binding_percent = max(windows)
    return UsageSnapshot(
        provider=provider,
        binding_percent=binding_percent,
        credits_engaged=credits_engaged,
        payload=_public_payload(provider, payload),
    )


def _probe_codex() -> UsageSnapshot:
    token, account_id = _codex_credentials()
    if not token or not account_id:
        return UsageSnapshot(
            provider="codex",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error="credentials unavailable",
        )
    try:
        payload = _json_get(
            CODEX_USAGE_URL,
            {
                "Authorization": "Bearer " + token,
                "ChatGPT-Account-Id": account_id,
                "Accept": "application/json",
                "User-Agent": "p13i-agent-harness",
            },
        )
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        return UsageSnapshot(
            provider="codex",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error=_safe_error(error),
        )
    return normalize_usage("codex", payload)


def _probe_claude() -> UsageSnapshot:
    token = _claude_token()
    if not token:
        return UsageSnapshot(
            provider="claude",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error="credentials unavailable",
        )
    try:
        payload = _json_get(
            CLAUDE_USAGE_URL,
            {
                "Authorization": "Bearer " + token,
                "anthropic-beta": "oauth-2025-04-20",
                "Accept": "application/json",
                "User-Agent": "p13i-agent-harness",
            },
        )
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        return UsageSnapshot(
            provider="claude",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error=_safe_error(error),
        )
    return normalize_usage("claude", payload)


def _json_get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("object required", "", 0)
    return payload


def _codex_credentials() -> tuple[str | None, str | None]:
    try:
        payload = json.loads(
            (Path.home() / ".codex" / "auth.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return None, None
    tokens = _object(payload.get("tokens"))
    token = tokens.get("access_token")
    account_id = tokens.get("account_id")
    if not isinstance(token, str):
        token = None
    if not isinstance(account_id, str):
        account_id = None
    return token, account_id


def _claude_token() -> str | None:
    try:
        completed = subprocess.run(
            (
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                "Claude Code-credentials",
                "-w",
            ),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    oauth = _object(payload.get("claudeAiOauth"))
    token = oauth.get("accessToken")
    if not isinstance(token, str):
        return None
    return token


def _object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_error(error: BaseException) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return "HTTP " + str(error.code)
    return type(error).__name__


def _public_payload(
    provider: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if provider == "codex":
        rate_limit = _object(payload.get("rate_limit"))
        result["rate_limit"] = {
            "allowed": rate_limit.get("allowed"),
            "limit_reached": rate_limit.get("limit_reached"),
            "primary_window": _usage_window(
                rate_limit.get("primary_window"),
                "used_percent",
            ),
            "secondary_window": _usage_window(
                rate_limit.get("secondary_window"),
                "used_percent",
            ),
        }
        credits = _object(payload.get("credits"))
        result["credits"] = {
            "has_credits": credits.get("has_credits"),
            "unlimited": credits.get("unlimited"),
            "overage_limit_reached": credits.get(
                "overage_limit_reached"
            ),
        }
    if provider == "claude":
        for name in (
            "five_hour",
            "seven_day",
            "seven_day_opus",
            "seven_day_sonnet",
        ):
            result[name] = _usage_window(
                payload.get(name),
                "utilization",
            )
        extra = _object(payload.get("extra_usage"))
        result["extra_usage"] = {
            "is_enabled": extra.get("is_enabled"),
            "spend_limit_reached": extra.get("spend_limit_reached"),
            "utilization": extra.get("utilization"),
        }
    return result


def _usage_window(value: object, percent_field: str) -> dict[str, Any] | None:
    window = _object(value)
    if not window:
        return None
    return {
        percent_field: window.get(percent_field),
        "resets_at": window.get("resets_at", window.get("reset_at")),
        "limit_window_seconds": window.get("limit_window_seconds"),
    }
