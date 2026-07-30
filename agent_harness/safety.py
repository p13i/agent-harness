"""Mandatory execution limits and provider-neutral turn guards."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
import time
from typing import Any

from agent_harness.context import estimate_tokens
from agent_harness.providers.base import ProviderEvent


INTERACTIVE = "interactive"
UNATTENDED = "unattended"
LIVE_SMOKE = "live-smoke"
PROFILES = frozenset({INTERACTIVE, UNATTENDED, LIVE_SMOKE})


@dataclass(frozen=True)
class SafetyLimits:
    profile: str
    workload: str
    max_seconds: int
    max_context_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    max_tool_calls: int
    stagnation_seconds: int
    binding_ceiling: float
    default_effort: str
    max_attempts: int
    repeated_tool_limit: int = 3
    repeated_cycle_limit: int = 2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyConsumption:
    context_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    attempts: int = 0
    elapsed_seconds: float = 0.0
    exact_tokens: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_profile(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized not in PROFILES:
        raise ValueError("unsupported execution profile")
    return normalized


def limits_for(profile: str, workload: str) -> SafetyLimits:
    profile = validate_profile(profile)
    normalized_workload = workload.strip().casefold()
    if not normalized_workload:
        normalized_workload = "implementation"
    if profile == LIVE_SMOKE:
        return SafetyLimits(
            profile=profile,
            workload=normalized_workload,
            max_seconds=300,
            max_context_tokens=16_000,
            max_output_tokens=4_000,
            max_total_tokens=20_000,
            max_tool_calls=10,
            stagnation_seconds=120,
            binding_ceiling=50.0,
            default_effort="low",
            max_attempts=1,
        )
    if profile == INTERACTIVE:
        return SafetyLimits(
            profile=profile,
            workload=normalized_workload,
            max_seconds=3_600,
            max_context_tokens=256_000,
            max_output_tokens=32_000,
            max_total_tokens=300_000,
            max_tool_calls=256,
            stagnation_seconds=900,
            binding_ceiling=90.0,
            default_effort="high",
            max_attempts=3,
        )
    if normalized_workload in {"operations", "operation", "sre"}:
        return SafetyLimits(
            profile=profile,
            workload=normalized_workload,
            max_seconds=900,
            max_context_tokens=64_000,
            max_output_tokens=16_000,
            max_total_tokens=100_000,
            max_tool_calls=64,
            stagnation_seconds=600,
            binding_ceiling=70.0,
            default_effort="medium",
            max_attempts=3,
        )
    return SafetyLimits(
        profile=profile,
        workload=normalized_workload,
        max_seconds=2_700,
        max_context_tokens=128_000,
        max_output_tokens=32_000,
        max_total_tokens=200_000,
        max_tool_calls=128,
        stagnation_seconds=900,
        binding_ceiling=70.0,
        default_effort="high",
        max_attempts=3,
    )


def effective_effort(
    requested: str,
    limits: SafetyLimits,
    *,
    xhigh_authorized: bool,
) -> str:
    effort = requested.strip().casefold()
    if not effort:
        return limits.default_effort
    if effort == "xhigh" and limits.profile != INTERACTIVE:
        if not xhigh_authorized:
            raise ValueError(
                "xhigh effort requires an explicit unattended authorization"
            )
    return effort


def apply_extension(
    limits: SafetyLimits,
    extension: dict[str, Any],
) -> SafetyLimits:
    seconds = limits.max_seconds
    tokens = limits.max_total_tokens
    additional_seconds = extension.get("additional_seconds", 0)
    if isinstance(additional_seconds, int):
        seconds += max(0, additional_seconds)
    additional_tokens = extension.get("additional_tokens", 0)
    if isinstance(additional_tokens, int):
        tokens += max(0, additional_tokens)
    return replace(
        limits,
        max_seconds=seconds,
        max_total_tokens=tokens,
    )


def lower_effort(value: str) -> str:
    order = ("low", "medium", "high", "xhigh")
    normalized = value.strip().casefold()
    if normalized not in order:
        return "medium"
    index = order.index(normalized)
    if index == 0:
        return "low"
    return order[index - 1]


def normalize_usage(value: object) -> dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    _collect_usage(value, totals)
    if totals["total_tokens"] == 0:
        totals["total_tokens"] = (
            totals["input_tokens"] + totals["output_tokens"]
        )
    totals["exact"] = totals["total_tokens"] > 0
    return totals


def _collect_usage(value: object, totals: dict[str, int]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_usage(item, totals)
        return
    if not isinstance(value, dict):
        return
    for raw_name, item in value.items():
        name = str(raw_name).replace("-", "_").casefold()
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            amount = max(0, int(item))
            if name in {
                "input_tokens",
                "inputtokens",
                "input",
            }:
                totals["input_tokens"] = max(
                    totals["input_tokens"],
                    amount,
                )
            elif name in {
                "cache_read_input_tokens",
                "cached_input_tokens",
                "cachedinputtokens",
            }:
                totals["cached_input_tokens"] = max(
                    totals["cached_input_tokens"],
                    amount,
                )
            elif name in {
                "output_tokens",
                "outputtokens",
                "output",
            }:
                totals["output_tokens"] = max(
                    totals["output_tokens"],
                    amount,
                )
            elif name in {
                "total_tokens",
                "totaltokens",
                "total",
            }:
                totals["total_tokens"] = max(
                    totals["total_tokens"],
                    amount,
                )
        else:
            _collect_usage(item, totals)


class TurnGuard:
    """Tracks one command across every provider attempt."""

    def __init__(
        self,
        limits: SafetyLimits,
        consumption: SafetyConsumption | None = None,
        *,
        monotonic=None,
    ) -> None:
        self.limits = limits
        if consumption is None:
            consumption = SafetyConsumption()
        self.consumption = consumption
        if monotonic is None:
            monotonic = time.monotonic
        self._monotonic = monotonic
        self._started = monotonic()
        self._last_progress = self._started
        self._tool_pairs: list[str] = []
        self._pending_tool = ""
        self._violation = ""
        self._warning_sent = False
        self._attempt_estimated_total = 0
        self._attempt_estimated_output = 0

    def begin_attempt(self, context_tokens: int) -> str:
        self.consumption.attempts += 1
        submitted = max(0, context_tokens)
        self.consumption.context_tokens += submitted
        self._attempt_estimated_total = submitted
        self._attempt_estimated_output = 0
        self._refresh_total()
        return self.violation()

    def observe(self, event: ProviderEvent) -> str:
        if event.event_type == "usage.updated":
            self._observe_usage(event.metadata)
        if event.event_type == "turn.completed":
            self._observe_usage(event.metadata)
        if event.event_type == "turn.failed":
            self._observe_usage(event.metadata)
        if event.event_type in {
            "agent.message",
            "agent.message.delta",
            "reasoning.summary.delta",
        }:
            estimated = estimate_tokens(event.text)
            self.consumption.output_tokens += estimated
            self._attempt_estimated_output += estimated
            self._attempt_estimated_total += estimated
            self._refresh_total()
        if _tool_started(event.event_type):
            self.consumption.tool_calls += 1
            self._pending_tool = _event_fingerprint(event)
        if _tool_completed(event.event_type):
            completed = _event_fingerprint(event)
            pair = self._pending_tool + ":" + completed
            self._pending_tool = ""
            self._observe_tool_pair(pair)
        return self.violation()

    def note_material_progress(self) -> None:
        self._last_progress = self._monotonic()
        self._tool_pairs.clear()

    def violation(self) -> str:
        if self._violation:
            return self._violation
        now = self._monotonic()
        self.consumption.elapsed_seconds = now - self._started
        if self.consumption.attempts > self.limits.max_attempts:
            self._violation = "attempts"
        elif self.consumption.elapsed_seconds >= self.limits.max_seconds:
            self._violation = "seconds"
        elif self.consumption.context_tokens > self.limits.max_context_tokens:
            self._violation = "context-tokens"
        elif self.consumption.output_tokens > self.limits.max_output_tokens:
            self._violation = "output-tokens"
        elif self.consumption.total_tokens > self.limits.max_total_tokens:
            self._violation = "total-tokens"
        elif self.consumption.tool_calls > self.limits.max_tool_calls:
            self._violation = "tool-calls"
        elif now - self._last_progress >= self.limits.stagnation_seconds:
            self._violation = "stagnation"
        return self._violation

    def warning_due(self) -> bool:
        if self._warning_sent:
            return False
        ratios = (
            self.consumption.elapsed_seconds / self.limits.max_seconds,
            self.consumption.context_tokens
            / self.limits.max_context_tokens,
            self.consumption.output_tokens
            / self.limits.max_output_tokens,
            self.consumption.total_tokens
            / self.limits.max_total_tokens,
            self.consumption.tool_calls / self.limits.max_tool_calls,
        )
        if max(ratios) < 0.8:
            return False
        self._warning_sent = True
        return True

    def recover(self) -> None:
        if self._violation not in {
            "repeated-tool",
            "repeated-cycle",
            "stagnation",
        }:
            raise ValueError("hard safety violations cannot recover")
        self._violation = ""
        self._last_progress = self._monotonic()
        self._tool_pairs.clear()
        self._pending_tool = ""

    def snapshot(self) -> dict[str, Any]:
        self.violation()
        return {
            "limits": self.limits.as_dict(),
            "consumption": self.consumption.as_dict(),
            "warning": self._warning_sent,
            "violation": self._violation,
        }

    def _observe_usage(self, value: object) -> None:
        normalized = normalize_usage(value)
        if not normalized["exact"]:
            return
        self.consumption.output_tokens = max(
            0,
            self.consumption.output_tokens
            + int(normalized["output_tokens"])
            - self._attempt_estimated_output,
        )
        provider_total = int(normalized["total_tokens"])
        self.consumption.total_tokens = max(
            0,
            self.consumption.total_tokens
            + provider_total
            - self._attempt_estimated_total,
        )
        self._attempt_estimated_output = int(
            normalized["output_tokens"]
        )
        self._attempt_estimated_total = provider_total
        self.consumption.exact_tokens = True

    def _refresh_total(self) -> None:
        estimated = (
            self.consumption.context_tokens
            + self.consumption.output_tokens
        )
        self.consumption.total_tokens = max(
            self.consumption.total_tokens,
            estimated,
        )

    def _observe_tool_pair(self, pair: str) -> None:
        self._tool_pairs.append(pair)
        repeated = 0
        for item in reversed(self._tool_pairs):
            if item != pair:
                break
            repeated += 1
        if repeated >= self.limits.repeated_tool_limit:
            self._violation = "repeated-tool"
            return
        maximum = min(16, len(self._tool_pairs) // 2)
        for length in range(2, maximum + 1):
            previous = self._tool_pairs[-2 * length : -length]
            current = self._tool_pairs[-length:]
            if previous != current:
                continue
            self._violation = "repeated-cycle"
            return
        self._last_progress = self._monotonic()


def _tool_started(event_type: str) -> bool:
    return event_type in {
        "tool.started",
        "tool.command.started",
        "tool.command_execution.started",
    }


def _tool_completed(event_type: str) -> bool:
    return event_type in {
        "tool.completed",
        "tool.command.completed",
        "tool.command_execution.completed",
    }


def _event_fingerprint(event: ProviderEvent) -> str:
    payload = {
        "event_type": event.event_type,
        "text": event.text,
        "metadata": event.metadata,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
