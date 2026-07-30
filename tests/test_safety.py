"""Execution-envelope and runaway-turn regression coverage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_harness.errors import SafetyGuardError
from agent_harness.providers.base import ProviderEvent
from agent_harness.safety import INTERACTIVE
from agent_harness.safety import LIVE_SMOKE
from agent_harness.safety import MINIMUM_STATE_FREE_BYTES
from agent_harness.safety import SafetyConsumption
from agent_harness.safety import UNATTENDED
from agent_harness.safety import TurnGuard
from agent_harness.safety import apply_extension
from agent_harness.safety import effective_effort
from agent_harness.safety import limits_for
from agent_harness.safety import lower_effort
from agent_harness.safety import normalize_usage
from agent_harness.safety import require_state_headroom
from agent_harness.safety import validate_profile


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_profiles_preserve_unattended_headroom() -> None:
    interactive = limits_for(INTERACTIVE, "implementation")
    operations = limits_for(UNATTENDED, "operations")
    engineering = limits_for(UNATTENDED, "implementation")
    smoke = limits_for(LIVE_SMOKE, "implementation")

    assert interactive.binding_ceiling == 90.0
    assert operations.binding_ceiling == 70.0
    assert engineering.binding_ceiling == 70.0
    assert smoke.binding_ceiling == 50.0
    assert operations.max_seconds == 900
    assert engineering.max_seconds == 2_700
    assert smoke.max_attempts == 1
    assert limits_for(UNATTENDED, "").workload == "implementation"
    extended = apply_extension(
        engineering,
        {
            "additional_seconds": 30,
            "additional_tokens": 1_000,
        },
    )
    assert extended.max_seconds == engineering.max_seconds + 30
    assert (
        extended.max_total_tokens
        == engineering.max_total_tokens + 1_000
    )
    with pytest.raises(ValueError, match="profile"):
        validate_profile("unbounded")


def test_state_headroom_fails_closed_before_provider_use(
    tmp_path: Path,
) -> None:
    available = MINIMUM_STATE_FREE_BYTES

    assert (
        require_state_headroom(
            tmp_path,
            "codex",
            free_bytes=available,
        )
        == available
    )
    with pytest.raises(
        SafetyGuardError,
        match="state-volume-headroom",
    ) as raised:
        require_state_headroom(
            tmp_path,
            "claude",
            free_bytes=available - 1,
        )
    assert raised.value.provider == "claude"
    assert raised.value.recoverable is False


def test_xhigh_requires_an_unattended_authorization() -> None:
    limits = limits_for(UNATTENDED, "implementation")

    assert effective_effort("", limits, xhigh_authorized=False) == "high"
    with pytest.raises(ValueError, match="authorization"):
        effective_effort("xhigh", limits, xhigh_authorized=False)
    assert (
        effective_effort("xhigh", limits, xhigh_authorized=True)
        == "xhigh"
    )
    assert lower_effort("xhigh") == "high"
    assert lower_effort("high") == "medium"
    assert lower_effort("low") == "low"
    assert lower_effort("provider-default") == "medium"


def test_usage_normalization_accepts_both_provider_shapes() -> None:
    normalized = normalize_usage(
        {
            "usage": {
                "input_tokens": 120,
                "cache_read_input_tokens": 40,
                "output_tokens": 30,
            },
            "tokenUsage": {
                "totalTokens": 175,
            },
        }
    )

    assert normalized == {
        "input_tokens": 120,
        "cached_input_tokens": 40,
        "output_tokens": 30,
        "total_tokens": 175,
        "exact": True,
    }
    fallback = normalize_usage(
        [
            None,
            {"input": 2, "output": 3, "total": False},
            {"nested": {"cached-input-tokens": 1}},
        ]
    )
    assert fallback == {
        "input_tokens": 2,
        "cached_input_tokens": 1,
        "output_tokens": 3,
        "total_tokens": 5,
        "exact": True,
    }
    incremental = normalize_usage(
        {
            "tokenUsage": {
                "last": {
                    "inputTokens": 18_527,
                    "outputTokens": 11,
                    "totalTokens": 18_538,
                },
                "total": {
                    "inputTokens": 37_019,
                    "outputTokens": 21,
                    "totalTokens": 37_040,
                },
            }
        }
    )
    assert incremental == {
        "input_tokens": 18_527,
        "cached_input_tokens": 0,
        "output_tokens": 11,
        "total_tokens": 18_538,
        "exact": True,
    }


def test_guard_shares_context_and_usage_across_attempts() -> None:
    limits = limits_for(LIVE_SMOKE, "implementation")
    guard = TurnGuard(limits)

    assert guard.begin_attempt(12_000) == ""
    guard.observe(
        ProviderEvent(
            "usage.updated",
            metadata={
                "tokenUsage": {
                    "inputTokens": 15_000,
                    "outputTokens": 6_000,
                    "totalTokens": 21_000,
                }
            },
        )
    )

    assert guard.violation() == "output-tokens"
    consumption = guard.snapshot()["consumption"]
    assert consumption["exact_tokens"] is True
    assert consumption["input_tokens"] == 15_000
    assert consumption["cached_input_tokens"] == 0
    assert consumption["output_tokens"] == 6_000


def test_guard_trips_repeated_tool_pair() -> None:
    guard = TurnGuard(limits_for(UNATTENDED, "implementation"))
    guard.begin_attempt(100)
    started = ProviderEvent(
        "tool.started",
        text="Read",
        metadata={"input": {"file_path": "SKILL.md"}},
    )
    completed = ProviderEvent(
        "tool.completed",
        text="same content",
    )

    for unused in range(3):
        del unused
        guard.observe(started)
        guard.observe(completed)

    assert guard.violation() == "repeated-tool"


def test_guard_trips_repeated_multi_tool_cycle() -> None:
    guard = TurnGuard(limits_for(UNATTENDED, "implementation"))
    guard.begin_attempt(100)

    for name in ("Read A", "Read B", "Read A", "Read B"):
        guard.observe(ProviderEvent("tool.started", text=name))
        guard.observe(ProviderEvent("tool.completed", text=name))

    assert guard.violation() == "repeated-cycle"


def test_guard_trips_stagnation_without_provider_usage() -> None:
    clock = Clock()
    limits = limits_for(LIVE_SMOKE, "implementation")
    guard = TurnGuard(limits, monotonic=clock)
    guard.begin_attempt(100)

    clock.advance(limits.stagnation_seconds)

    assert guard.violation() == "stagnation"


def test_guard_covers_every_hard_limit_and_recovery_boundary() -> None:
    base = limits_for(LIVE_SMOKE, "implementation")

    attempts = TurnGuard(replace(base, max_attempts=0))
    assert attempts.begin_attempt(0) == "attempts"

    clock = Clock()
    seconds = TurnGuard(
        replace(base, max_seconds=1),
        monotonic=clock,
    )
    seconds.begin_attempt(0)
    clock.advance(1)
    assert seconds.violation() == "seconds"

    context = TurnGuard(replace(base, max_context_tokens=1))
    assert context.begin_attempt(2) == "context-tokens"

    total = TurnGuard(
        base,
        SafetyConsumption(total_tokens=base.max_total_tokens + 1),
    )
    assert total.violation() == "total-tokens"

    tools = TurnGuard(
        base,
        SafetyConsumption(tool_calls=base.max_tool_calls + 1),
    )
    assert tools.violation() == "tool-calls"

    with pytest.raises(ValueError, match="hard safety"):
        tools.recover()

    progress_clock = Clock()
    progress = TurnGuard(base, monotonic=progress_clock)
    progress.begin_attempt(1)
    progress.observe(
        ProviderEvent(
            "turn.completed",
            metadata={"usage": {"total_tokens": 3}},
        )
    )
    progress.observe(
        ProviderEvent(
            "turn.failed",
            metadata={"usage": {"total_tokens": 4}},
        )
    )
    progress.observe(
        ProviderEvent("usage.updated", metadata={"usage": {}})
    )
    progress.note_material_progress()
    assert progress.violation() == ""
    assert progress.warning_due() is False


def test_guard_warning_recovery_and_nonrepeating_cycles() -> None:
    base = limits_for(UNATTENDED, "implementation")
    guard = TurnGuard(base)
    guard.begin_attempt(base.max_context_tokens * 4 // 5)

    assert guard.warning_due() is True
    assert guard.warning_due() is False

    for name in ("A", "B", "C", "D", "E", "F"):
        guard.observe(ProviderEvent("tool.started", text=name))
        guard.observe(ProviderEvent("tool.completed", text=name))
    assert guard.violation() == ""

    repeating = TurnGuard(base)
    repeating.begin_attempt(1)
    started = ProviderEvent("tool.started", text="same")
    completed = ProviderEvent("tool.completed", text="same")
    for unused in range(3):
        del unused
        repeating.observe(started)
        repeating.observe(completed)
    assert repeating.violation() == "repeated-tool"
    repeating.recover()
    assert repeating.violation() == ""
