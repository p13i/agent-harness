"""Mandatory execution limits and provider-neutral turn guards."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from agent_harness.context import estimate_tokens
from agent_harness.errors import SafetyGuardError
from agent_harness.providers.base import ProviderEvent

INTERACTIVE = "interactive"
UNATTENDED = "unattended"
LIVE_SMOKE = "live-smoke"
PROFILES = frozenset({INTERACTIVE, UNATTENDED, LIVE_SMOKE})
MINIMUM_STATE_FREE_BYTES = 2 * 1024 * 1024 * 1024
EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class SafetyLimits:
    profile: str
    workload: str
    max_seconds: int
    max_context_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    max_tool_calls: int
    max_child_agents: int
    max_dollars: float | None
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
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    child_agents: int = 0
    dollars: float = 0.0
    attempts: int = 0
    elapsed_seconds: float = 0.0
    exact_tokens: bool = False
    exact_dollars: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_profile(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized not in PROFILES:
        raise ValueError("unsupported execution profile")
    return normalized


def require_state_headroom(
    path: Path,
    provider: str,
    *,
    free_bytes: int | None = None,
) -> int:
    if free_bytes is None:
        free_bytes = shutil.disk_usage(path).free
    if free_bytes < MINIMUM_STATE_FREE_BYTES:
        raise SafetyGuardError(
            "state-volume-headroom",
            provider,
        )
    return free_bytes


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
            max_child_agents=0,
            max_dollars=None,
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
            max_child_agents=16,
            max_dollars=None,
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
            max_child_agents=2,
            max_dollars=None,
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
        max_child_agents=2,
        max_dollars=None,
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
    if effort_requires_xhigh_authorization(effort) and limits.profile != INTERACTIVE:
        if not xhigh_authorized:
            raise ValueError(
                "xhigh effort requires an explicit unattended authorization"
            )
    return effort


def effort_requires_xhigh_authorization(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized not in EFFORT_ORDER:
        return False
    return EFFORT_ORDER.index(normalized) >= EFFORT_ORDER.index("xhigh")


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


def tighten_limits(
    limits: SafetyLimits,
    requested: object,
) -> SafetyLimits:
    if requested is None:
        return limits
    if not isinstance(requested, dict):
        raise ValueError("safety_limits must be an object")
    integer_minimums = {
        "max_seconds": 1,
        "max_context_tokens": 1,
        "max_output_tokens": 1,
        "max_total_tokens": 1,
        "max_tool_calls": 0,
        "max_child_agents": 0,
        "stagnation_seconds": 1,
        "max_attempts": 1,
        "repeated_tool_limit": 1,
        "repeated_cycle_limit": 1,
    }
    numeric_fields = {"max_dollars", "binding_ceiling"}
    unknown = set(requested) - set(integer_minimums) - numeric_fields
    if unknown:
        raise ValueError("safety_limits contains an unsupported field")
    replacements: dict[str, Any] = {}
    for name, minimum in integer_minimums.items():
        if name not in requested:
            continue
        value = requested[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("safety_limits " + name + " must be an integer")
        if value < minimum:
            raise ValueError("safety_limits " + name + " is below its minimum")
        current = int(getattr(limits, name))
        if value > current:
            raise ValueError("safety_limits cannot widen " + name)
        replacements[name] = value
    for name in numeric_fields:
        if name not in requested:
            continue
        value = requested[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("safety_limits " + name + " must be numeric")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("safety_limits " + name + " must be finite")
        if normalized < 0:
            raise ValueError("safety_limits " + name + " must not be negative")
        current_value = getattr(limits, name)
        if name == "max_dollars" and current_value is None and normalized > 0:
            raise ValueError("safety_limits cannot authorize metered spend")
        if current_value is not None and normalized > float(current_value):
            raise ValueError("safety_limits cannot widen " + name)
        replacements[name] = normalized
    return replace(limits, **replacements)


def lower_effort(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized not in EFFORT_ORDER:
        return "medium"
    index = EFFORT_ORDER.index(normalized)
    if index == 0:
        return "low"
    return EFFORT_ORDER[index - 1]


def normalize_usage(value: object) -> dict[str, Any]:
    value = _incremental_usage(value)
    totals: dict[str, Any] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "invalid": False,
    }
    _collect_usage(value, totals)
    minimum_total = totals["input_tokens"] + totals["output_tokens"]
    totals["total_tokens"] = max(totals["total_tokens"], minimum_total)
    totals["exact"] = totals["total_tokens"] > 0 and not totals.pop("invalid")
    return totals


def normalize_cost(value: object) -> float:
    cost, unused_exact, unused_invalid = _collect_cost(value)
    del unused_exact, unused_invalid
    return cost


def has_exact_cost(value: object) -> bool:
    unused_cost, exact, invalid = _collect_cost(value)
    del unused_cost
    return exact and not invalid


def _collect_cost(value: object) -> tuple[float, bool, bool]:
    if isinstance(value, list):
        cost = 0.0
        exact = False
        invalid = False
        for item in value:
            item_cost, item_exact, item_invalid = _collect_cost(item)
            cost = max(cost, item_cost)
            exact = exact or item_exact
            invalid = invalid or item_invalid
        return cost, exact, invalid
    if not isinstance(value, dict):
        return 0.0, False, False
    values: list[float] = []
    exact = False
    invalid = False
    for raw_name, item in value.items():
        name = str(raw_name).replace("-", "_").casefold()
        if name in {"cost_usd", "total_cost_usd", "totalcostusd"}:
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                normalized = float(item)
                if math.isfinite(normalized) and normalized >= 0:
                    values.append(normalized)
                    exact = True
                else:
                    invalid = True
            continue
        nested_cost, nested_exact, nested_invalid = _collect_cost(item)
        values.append(nested_cost)
        exact = exact or nested_exact
        invalid = invalid or nested_invalid
    return max(values, default=0.0), exact, invalid


def _incremental_usage(value: object) -> object:
    if not isinstance(value, dict):
        return value
    for raw_name, item in value.items():
        name = str(raw_name).replace("-", "_").casefold()
        if name in {"tokenusage", "token_usage"} and isinstance(item, dict):
            last = item.get("last")
            if isinstance(last, dict):
                return last
    return value


def _collect_usage(value: object, totals: dict[str, Any]) -> None:
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
            recognized = name in {
                "input_tokens",
                "inputtokens",
                "input",
                "cache_read_input_tokens",
                "cached_input_tokens",
                "cachedinputtokens",
                "output_tokens",
                "outputtokens",
                "output",
                "total_tokens",
                "totaltokens",
                "total",
            }
            normalized = float(item)
            if recognized and (not math.isfinite(normalized) or normalized < 0):
                totals["invalid"] = True
                continue
            if not math.isfinite(normalized):
                continue
            amount = int(item)
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
        self._elapsed_offset = max(0.0, float(consumption.elapsed_seconds))
        self._last_progress = self._started
        self._tool_pairs: list[str] = []
        self._pending_tools: dict[str, list[str]] = {}
        self._completed_tool_pair = ""
        self._seen_child_ids: set[str] = set()
        self._violation = ""
        self._warning_sent = False
        self._attempt_estimated_total = 0
        self._attempt_estimated_output = 0
        self._attempt_exact_input = 0
        self._attempt_exact_cached_input = 0
        self._attempt_exact_dollars = 0.0
        self._material_state_digest = ""

    def begin_attempt(self, context_tokens: int) -> str:
        self.consumption.attempts += 1
        submitted = max(0, context_tokens)
        self.consumption.context_tokens += submitted
        self._attempt_estimated_total = submitted
        self._attempt_estimated_output = 0
        self._attempt_exact_input = 0
        self._attempt_exact_cached_input = 0
        self._attempt_exact_dollars = 0.0
        self.consumption.total_tokens += submitted
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
            self.consumption.total_tokens += estimated
        if _tool_started(event.event_type):
            self.consumption.tool_calls += 1
            identity = _tool_identity(event)
            self._pending_tools.setdefault(identity, []).append(
                _event_fingerprint(event)
            )
        if event.event_type == "agent.child.started":
            self.consumption.child_agents += self._new_child_start_count(event)
        if _tool_completed(event.event_type):
            completed = _event_fingerprint(event)
            identity = _tool_identity(event)
            pending = self._pending_tools.get(identity, [])
            started = ""
            if pending:
                started = pending.pop(0)
            if not pending:
                self._pending_tools.pop(identity, None)
            pair = started + ":" + completed
            self._completed_tool_pair = pair
            self._observe_tool_pair(pair)
        return self.violation()

    def take_completed_tool_pair(self) -> str:
        pair = self._completed_tool_pair
        self._completed_tool_pair = ""
        return pair

    def establish_material_state(self, digest: str) -> None:
        if not digest:
            raise ValueError("material state digest is required")
        self._material_state_digest = digest

    def note_material_progress(self, digest: str) -> bool:
        if not digest:
            raise ValueError("material state digest is required")
        if digest == self._material_state_digest:
            return False
        self._material_state_digest = digest
        self._last_progress = self._monotonic()
        self._tool_pairs.clear()
        return True

    def note_child_admissions(self, consumed: int) -> None:
        if consumed < 0:
            raise ValueError("child admission count must not be negative")
        self.consumption.child_agents = max(
            self.consumption.child_agents,
            consumed,
        )

    def violation(self) -> str:
        if self._violation:
            return self._violation
        now = self._monotonic()
        self.consumption.elapsed_seconds = self._elapsed_offset + now - self._started
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
        elif self.consumption.child_agents > self.limits.max_child_agents:
            self._violation = "child-agents"
        elif (
            self.limits.max_dollars is not None
            and self.consumption.dollars > self.limits.max_dollars
        ):
            self._violation = "dollars"
        elif now - self._last_progress >= self.limits.stagnation_seconds:
            self._violation = "stagnation"
        return self._violation

    def warning_due(self) -> bool:
        if self._warning_sent:
            return False
        ratios = (
            _ratio(
                self.consumption.elapsed_seconds,
                self.limits.max_seconds,
            ),
            _ratio(
                self.consumption.context_tokens,
                self.limits.max_context_tokens,
            ),
            _ratio(
                self.consumption.output_tokens,
                self.limits.max_output_tokens,
            ),
            _ratio(
                self.consumption.total_tokens,
                self.limits.max_total_tokens,
            ),
            _ratio(
                self.consumption.tool_calls,
                self.limits.max_tool_calls,
            ),
            _ratio(
                self.consumption.child_agents,
                self.limits.max_child_agents,
            ),
            _optional_ratio(
                self.consumption.dollars,
                self.limits.max_dollars,
            ),
        )
        if max(ratios) < 0.8:
            return False
        self._warning_sent = True
        return True

    def recover(self) -> None:
        if self._violation != "stagnation":
            raise ValueError("hard safety violations cannot recover")
        self._violation = ""
        self._last_progress = self._monotonic()
        self._tool_pairs.clear()
        self._pending_tools.clear()
        self._completed_tool_pair = ""

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
        cost = normalize_cost(value)
        if has_exact_cost(value):
            exact_dollars = max(self._attempt_exact_dollars, cost)
            self.consumption.dollars += exact_dollars - self._attempt_exact_dollars
            self._attempt_exact_dollars = exact_dollars
            self.consumption.exact_dollars = True
        if not normalized["exact"]:
            return
        exact_output = max(
            self._attempt_estimated_output,
            int(normalized["output_tokens"]),
        )
        self.consumption.output_tokens += exact_output - self._attempt_estimated_output
        exact_input = max(
            self._attempt_exact_input,
            int(normalized["input_tokens"]),
        )
        self.consumption.input_tokens += exact_input - self._attempt_exact_input
        exact_cached_input = max(
            self._attempt_exact_cached_input,
            int(normalized["cached_input_tokens"]),
        )
        self.consumption.cached_input_tokens += (
            exact_cached_input - self._attempt_exact_cached_input
        )
        provider_total = max(
            self._attempt_estimated_total,
            int(normalized["total_tokens"]),
        )
        self.consumption.total_tokens += provider_total - self._attempt_estimated_total
        self._attempt_estimated_output = exact_output
        self._attempt_estimated_total = provider_total
        self._attempt_exact_input = exact_input
        self._attempt_exact_cached_input = exact_cached_input
        self.consumption.exact_tokens = True

    def _new_child_start_count(self, event: ProviderEvent) -> int:
        identities = _child_start_identities(event)
        if not identities:
            return 1
        count = 0
        for identity in identities:
            if identity in self._seen_child_ids:
                continue
            self._seen_child_ids.add(identity)
            count += 1
        return count

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
        "metadata": _stable_fingerprint_value(event.metadata),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tool_identity(event: ProviderEvent) -> str:
    metadata = event.metadata
    if isinstance(metadata, dict):
        for name in ("tool_use_id", "toolUseId", "item_id", "itemId", "id"):
            value = metadata.get(name)
            if isinstance(value, str) and value:
                return value
    return "anonymous"


def _stable_fingerprint_value(value: object) -> object:
    if isinstance(value, dict):
        stable: dict[str, object] = {}
        for raw_name, item in value.items():
            name = str(raw_name)
            normalized = name.replace("-", "_").casefold()
            if normalized in {
                "id",
                "item_id",
                "itemid",
                "request_id",
                "requestid",
                "tool_use_id",
                "tooluseid",
            }:
                continue
            stable[name] = _stable_fingerprint_value(item)
        return stable
    if isinstance(value, list):
        return [_stable_fingerprint_value(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_fingerprint_value(item) for item in value]
    return value


def _child_start_identities(event: ProviderEvent) -> tuple[str, ...]:
    metadata = event.metadata
    if metadata is None:
        return ()
    for name in ("receiver_thread_ids", "receiverThreadIds"):
        values = metadata.get(name)
        if isinstance(values, list) and values:
            return tuple(str(item) for item in values if str(item))
    for name in ("tool_use_id", "toolUseId", "child_id", "id"):
        value = metadata.get(name)
        if isinstance(value, str) and value:
            return (value,)
    return ()


def _ratio(consumed: float, maximum: float) -> float:
    if maximum <= 0:
        if consumed > 0:
            return 1.0
        return 0.0
    return consumed / maximum


def _optional_ratio(
    consumed: float,
    maximum: float | None,
) -> float:
    if maximum is None:
        return 0.0
    return _ratio(consumed, maximum)
