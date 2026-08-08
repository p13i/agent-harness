"""Read-only subscription capacity probes."""

from __future__ import annotations

import asyncio
import json
import math
import os
import stat
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
# Undocumented endpoint shared with my/tools/usage_monitor.gpt.py; an
# outage here is endpoint drift, not spent quota.
KIMI_USAGE_URL = "https://api.kimi.com/coding/v1/usages"


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
        _timed_probe("codex", _probe_codex),
        _timed_probe("claude", _probe_claude),
        _timed_probe("kimi", _probe_kimi),
        _timed_probe("grok", _probe_grok),
    )
    return {item.provider: item for item in results}


def provider_auth_ready(provider: str) -> bool:
    """Return whether a provider has its required local launch credential."""
    if provider == "claude":
        token = _claude_token()
        return bool(token)
    if provider == "codex":
        token, account_id = _codex_credentials()
        return bool(token and account_id)
    if provider == "grok":
        return _grok_auth_present()
    raise ValueError("unknown provider: " + provider)


async def _timed_probe(
    provider: str,
    probe: Any,
) -> UsageSnapshot:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(probe),
            timeout=5.0,
        )
    except TimeoutError:
        return UsageSnapshot(
            provider=provider,
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error="probe timed out",
        )


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
    if provider == "kimi":
        for window in _kimi_windows(payload):
            windows.append(float(window["percent"]))
        credits_engaged = _kimi_extra_usage_engaged(_object(payload.get("usage")))
    if provider == "grok":
        # Auth-readiness only: signed-in OAuth is treated as full headroom
        # until a real rate-limit probe exists. Without a finite binding,
        # unattended routing with a binding ceiling rejects the provider.
        if payload.get("auth_ready"):
            windows.append(0.0)
        credits_engaged = False
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


def _probe_kimi() -> UsageSnapshot:
    # The OAuth token is short-lived and only the kimi CLI refreshes it,
    # so a 401 is reported as the probe error; no refresh is attempted.
    credential = _kimi_credential()
    if not credential:
        return UsageSnapshot(
            provider="kimi",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error="credentials unavailable",
        )
    try:
        payload = _json_get(
            KIMI_USAGE_URL,
            {
                "Authorization": "Bearer " + credential,
                "Accept": "application/json",
                "User-Agent": "kimi-code",
            },
        )
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        return UsageSnapshot(
            provider="kimi",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error=_safe_error(error),
        )
    return normalize_usage("kimi", payload)


def _probe_grok() -> UsageSnapshot:
    """Auth-readiness probe for Grok Build OAuth credentials.

    SpaceXAI does not yet expose a harness-facing rate-limit window API.
    Unattended admission requires a finite binding percent when a
    binding ceiling is set, so a present non-expired ``~/.grok/auth.json``
    entry is reported as binding 0.0 (full headroom). This is not real
    quota metering.
    """
    ready, detail = _grok_auth_detail()
    if not ready:
        return UsageSnapshot(
            provider="grok",
            binding_percent=None,
            credits_engaged=False,
            payload={},
            error=detail or "credentials unavailable",
        )
    return normalize_usage(
        "grok",
        {
            "auth_ready": True,
            "auth_source": "oauth",
            "detail": detail,
        },
    )


def _grok_auth_present() -> bool:
    ready, _detail = _grok_auth_detail()
    return ready


def _grok_auth_detail() -> tuple[bool, str]:
    path = Path.home() / ".grok" / "auth.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return False, "credentials unavailable"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False, "credentials unreadable"
    if not isinstance(payload, dict) or not payload:
        return False, "credentials unavailable"
    now = _unix_now()
    for _key, entry in payload.items():
        block = _object(entry)
        if not block:
            continue
        expires_at = block.get("expires_at")
        if expires_at is None:
            # Present entry without expiry still counts as signed-in.
            return True, "oauth present"
        expires = _number(expires_at)
        if expires is None:
            # Non-numeric expiry: treat as present if a refresh token exists.
            if block.get("refresh_token") or block.get("key"):
                return True, "oauth present"
            continue
        # expires_at may be seconds or milliseconds.
        if expires > 1_000_000_000_000:
            expires = expires / 1000.0
        if expires > now:
            return True, "oauth present"
        if block.get("refresh_token"):
            # Refreshable expired access token: still signed in.
            return True, "oauth refreshable"
    return False, "credentials expired"


def _unix_now() -> float:
    return time.time()


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
            (Path.home() / ".codex" / "auth.json").read_text(encoding="utf-8")
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
    environment_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if environment_token:
        return environment_token
    if sys.platform != "darwin":
        return _claude_file_token(Path.home() / ".claude" / ".credentials.json")
    return _claude_keychain_token()


def _claude_keychain_token() -> str | None:
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


def _claude_file_token(path: Path) -> str | None:
    try:
        if path.is_symlink():
            return None
        details = path.stat()
        if details.st_size > 1024 * 1024:
            return None
        if stat.S_IMODE(details.st_mode) & 0o077:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    oauth = _object(payload.get("claudeAiOauth"))
    token = oauth.get("accessToken")
    if not isinstance(token, str) or not token.strip():
        return None
    return token


def _kimi_credential() -> str | None:
    api_key = _kimi_config_api_key()
    if api_key:
        return api_key
    return _kimi_oauth_token()


def _kimi_config_api_key() -> str | None:
    """Static Console API key from the kimi CLI config, when present."""
    try:
        config = tomllib.loads(
            (Path.home() / ".kimi-code" / "config.toml").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, tomllib.TOMLDecodeError):
        return None
    providers = _object(config.get("providers"))
    managed = _object(providers.get("managed:kimi-code"))
    api_key = managed.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        return None
    return api_key


def _kimi_oauth_token() -> str | None:
    try:
        payload = json.loads(
            (Path.home() / ".kimi-code" / "credentials" / "kimi-code.json")
            .read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    token = _object(payload).get("access_token")
    if not isinstance(token, str) or not token.strip():
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
        normalized = float(value)
        if math.isfinite(normalized) and normalized >= 0:
            return normalized
    return None


def _kimi_quantity(value: object) -> float:
    """Kimi reports ``limit`` and ``remaining`` as strings."""
    if isinstance(value, bool):
        return 0.0
    try:
        quantity = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(quantity):
        return 0.0
    return quantity


def _kimi_window_label(minutes: float) -> str:
    if minutes and minutes % 60 == 0:
        return str(int(minutes // 60)) + "-hour"
    return str(int(minutes)) + "-minute"


def _kimi_quota(value: object, label: str) -> dict[str, Any] | None:
    """Normalize one Kimi quota block, or None when unusable.

    Kimi reports ``limit`` and ``remaining`` as strings and does NOT
    report a ``used`` figure, so the percentage is derived.
    """
    block = _object(value)
    if not block:
        return None
    limit = _kimi_quantity(block.get("limit"))
    if limit <= 0:
        return None
    remaining = _kimi_quantity(block.get("remaining"))
    return {
        "label": label,
        "percent": (limit - remaining) / limit * 100.0,
        "resets_at": block.get("resetTime"),
    }


def _kimi_windows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    limits = payload.get("limits")
    # The rolling window arrives in limits[]; match it by duration
    # rather than position, since the array is not order-guaranteed.
    if isinstance(limits, list):
        for entry in limits:
            block = _object(entry)
            window = _object(block.get("window"))
            if window.get("timeUnit") != "TIME_UNIT_MINUTE":
                continue
            label = _kimi_window_label(_kimi_quantity(window.get("duration")))
            quota = _kimi_quota(block.get("detail"), label)
            if quota is not None:
                derived.append(quota)
    weekly = _kimi_quota(payload.get("usage"), "weekly")
    if weekly is not None:
        derived.append(weekly)
    return derived


def _kimi_extra_usage_engaged(usage: dict[str, Any]) -> bool:
    # Kimi has no credits object; Extra Usage bills at roughly API
    # rates once the weekly quota is spent, so an exhausted pool is
    # reported as engaged credits.
    limit = _kimi_quantity(usage.get("limit"))
    remaining = _kimi_quantity(usage.get("remaining"))
    return limit > 0 and remaining <= 0


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
            "overage_limit_reached": credits.get("overage_limit_reached"),
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
    if provider == "kimi":
        # Derived windows only; the raw string quantities stay out of
        # status, audit, and diagnostic surfaces.
        result["windows"] = _kimi_windows(payload)
        result["extra_usage"] = {
            "engaged": _kimi_extra_usage_engaged(_object(payload.get("usage"))),
        }
        membership = _object(_object(payload.get("user")).get("membership"))
        result["membership_level"] = membership.get("level")
    if provider == "grok":
        result["auth_ready"] = bool(payload.get("auth_ready"))
        result["auth_source"] = payload.get("auth_source")
        result["detail"] = payload.get("detail")
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
