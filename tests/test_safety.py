"""Execution-envelope and runaway-turn regression coverage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from agent_harness.errors import SafetyGuardError
from agent_harness.providers.base import ProviderEvent
from agent_harness.providers.normalize import grok_payload
from agent_harness.safety import (
    INTERACTIVE,
    LIVE_SMOKE,
    MINIMUM_STATE_FREE_BYTES,
    UNATTENDED,
    SafetyConsumption,
    TurnGuard,
    apply_extension,
    effective_effort,
    has_exact_cost,
    limits_for,
    lower_effort,
    normalize_cost,
    normalize_usage,
    require_state_headroom,
    tighten_limits,
    validate_profile,
)


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
    assert operations.binding_ceiling == 90.0
    assert engineering.binding_ceiling == 90.0
    assert smoke.binding_ceiling == 50.0
    assert operations.max_seconds == 900
    assert engineering.max_seconds == 2_700
    assert smoke.max_attempts == 1
    assert smoke.max_child_agents == 0
    assert operations.max_child_agents == 2
    assert limits_for(UNATTENDED, "").workload == "implementation"
    extended = apply_extension(
        engineering,
        {
            "additional_seconds": 30,
            "additional_tokens": 1_000,
        },
    )
    assert extended.max_seconds == engineering.max_seconds + 30
    assert extended.max_total_tokens == engineering.max_total_tokens + 1_000
    with pytest.raises(ValueError, match="profile"):
        validate_profile("unbounded")


def test_per_command_limits_only_tighten_the_effective_envelope() -> None:
    base = limits_for(UNATTENDED, "implementation")
    tightened = tighten_limits(
        base,
        {
            "max_seconds": 300,
            "max_attempts": 2,
            "max_child_agents": 1,
            "max_dollars": 0,
        },
    )

    assert tightened.max_seconds == 300
    assert tightened.max_attempts == 2
    assert tightened.max_child_agents == 1
    assert tightened.max_dollars == 0
    with pytest.raises(ValueError, match="cannot widen max_attempts"):
        tighten_limits(base, {"max_attempts": base.max_attempts + 1})
    with pytest.raises(ValueError, match="cannot authorize metered spend"):
        tighten_limits(base, {"max_dollars": 1})
    with pytest.raises(ValueError, match="unsupported field"):
        tighten_limits(base, {"unbounded": 1})
    with pytest.raises(ValueError, match="must be an object"):
        tighten_limits(base, "300")
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="must be finite"):
            tighten_limits(base, {"binding_ceiling": value})
    with pytest.raises(ValueError, match="max_seconds must be an integer"):
        tighten_limits(base, {"max_seconds": 300.5})
    with pytest.raises(ValueError, match="max_attempts must be an integer"):
        tighten_limits(base, {"max_attempts": True})
    with pytest.raises(ValueError, match="max_seconds is below its minimum"):
        tighten_limits(base, {"max_seconds": 0})
    with pytest.raises(ValueError, match="binding_ceiling must be numeric"):
        tighten_limits(base, {"binding_ceiling": "90"})
    with pytest.raises(ValueError, match="max_dollars must be numeric"):
        tighten_limits(base, {"max_dollars": True})
    with pytest.raises(ValueError, match="must not be negative"):
        tighten_limits(base, {"binding_ceiling": -1})


def test_invalid_provider_cost_is_not_exact_or_chargeable() -> None:
    for value in (math.nan, math.inf, -math.inf, -1):
        usage = {"total_cost_usd": value}
        assert normalize_cost(usage) == 0
        assert has_exact_cost(usage) is False
    for invalid in (-1, math.nan, math.inf):
        mixed = {
            "total_cost_usd": invalid,
            "nested": {"cost_usd": 0},
        }
        assert normalize_cost(mixed) == 0
        assert has_exact_cost(mixed) is False


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
    assert raised.value.detail.retryable is False


def test_xhigh_requires_an_unattended_authorization() -> None:
    limits = limits_for(UNATTENDED, "implementation")

    assert effective_effort("", limits, xhigh_authorized=False) == "high"
    with pytest.raises(ValueError, match="authorization"):
        effective_effort("xhigh", limits, xhigh_authorized=False)
    assert effective_effort("xhigh", limits, xhigh_authorized=True) == "xhigh"
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
    cached_heavy = normalize_usage(
        {
            "input_tokens": 161_197,
            "cache_read_input_tokens": 3_466_112,
            "output_tokens": 31_155,
            "total_tokens": 3_658_464,
        }
    )
    assert cached_heavy == {
        "input_tokens": 161_197,
        "cached_input_tokens": 3_466_112,
        "output_tokens": 31_155,
        "total_tokens": 192_352,
        "exact": True,
    }
    inclusive_input = normalize_usage(
        {
            "input_tokens": 120,
            "cached_input_tokens": 40,
            "output_tokens": 30,
            "total_tokens": 150,
        }
    )
    assert inclusive_input["total_tokens"] == 150
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
    inconsistent = normalize_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 1,
        }
    )
    assert inconsistent == {
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "output_tokens": 20,
        "total_tokens": 120,
        "exact": True,
    }
    malformed = normalize_usage(
        {
            "input_tokens": -1,
            "output_tokens": 20,
            "total_tokens": math.inf,
        }
    )
    assert malformed["exact"] is False
    assert normalize_cost(
        {
            "usage": [
                {"total_cost_usd": 0.25},
                {"nested": {"cost-usd": 0.5}},
                {"cost_usd": True},
            ]
        }
    ) == pytest.approx(0.5)
    assert normalize_cost("unknown") == 0.0
    assert has_exact_cost(
        [
            {},
            {
                "cost_usd": True,
                "total_cost_usd": "unknown",
                "nested": {"cost-usd": 0.5},
            },
        ]
    )


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


def test_provider_usage_cannot_reduce_accounted_consumption() -> None:
    base = limits_for(UNATTENDED, "implementation")
    guard = TurnGuard(replace(base, max_dollars=1.0))
    guard.begin_attempt(10)
    guard.observe(
        ProviderEvent(
            "usage.updated",
            metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 1,
                "total_cost_usd": 0.75,
            },
        )
    )
    first = guard.snapshot()["consumption"]
    assert first["input_tokens"] == 100
    assert first["output_tokens"] == 20
    assert first["total_tokens"] == 120
    assert first["dollars"] == pytest.approx(0.75)

    guard.observe(
        ProviderEvent(
            "usage.updated",
            metadata={
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
                "total_cost_usd": 0.25,
            },
        )
    )
    cumulative = guard.snapshot()["consumption"]
    assert cumulative["input_tokens"] == 100
    assert cumulative["output_tokens"] == 20
    assert cumulative["total_tokens"] == 120
    assert cumulative["dollars"] == pytest.approx(0.75)

    guard.observe(
        ProviderEvent(
            "usage.updated",
            metadata={
                "tokenUsage": {
                    "last": {
                        "inputTokens": 3,
                        "outputTokens": 1,
                        "totalTokens": 4,
                    }
                }
            },
        )
    )
    incremental = guard.snapshot()["consumption"]
    assert incremental["input_tokens"] == 100
    assert incremental["output_tokens"] == 20
    assert incremental["total_tokens"] == 120


def test_cached_input_does_not_trip_the_total_token_guard() -> None:
    base = limits_for(UNATTENDED, "implementation")
    guard = TurnGuard(replace(base, max_total_tokens=200_000))
    guard.begin_attempt(9_128)
    guard.observe(
        ProviderEvent(
            "turn.completed",
            metadata={
                "input_tokens": 161_197,
                "cache_read_input_tokens": 3_466_112,
                "output_tokens": 31_155,
                "total_tokens": 3_658_464,
            },
        )
    )

    assert guard.violation() == ""
    consumption = guard.snapshot()["consumption"]
    assert consumption["total_tokens"] == 192_352
    assert consumption["cached_input_tokens"] == 3_466_112


def test_cached_input_does_not_hide_unclassified_tokens() -> None:
    base = limits_for(UNATTENDED, "implementation")
    guard = TurnGuard(replace(base, max_total_tokens=160))
    guard.begin_attempt(10)
    guard.observe(
        ProviderEvent(
            "turn.completed",
            metadata={
                "input_tokens": 120,
                "cache_read_input_tokens": 40,
                "output_tokens": 30,
                "total_tokens": 175,
            },
        )
    )

    assert guard.violation() == "total-tokens"
    assert guard.snapshot()["consumption"]["total_tokens"] == 175


def test_subscription_attempt_ignores_provider_cost_equivalent() -> None:
    base = limits_for(UNATTENDED, "implementation")
    guard = TurnGuard(replace(base, max_dollars=0.0))
    guard.begin_attempt(1, charge_reported_cost=False)
    guard.observe(
        ProviderEvent(
            "turn.completed",
            metadata={"total_cost_usd": 0.25},
        )
    )

    assert guard.violation() == ""
    assert guard.consumption.dollars == 0.0
    assert guard.consumption.exact_dollars is False


def test_invalid_provider_usage_retains_conservative_estimates() -> None:
    guard = TurnGuard(limits_for(UNATTENDED, "implementation"))
    guard.begin_attempt(500)
    guard.observe(
        ProviderEvent(
            "agent.message.delta",
            text="estimated output remains accounted",
        )
    )
    estimated = guard.snapshot()["consumption"]
    guard.observe(
        ProviderEvent(
            "usage.updated",
            metadata={
                "input_tokens": -1,
                "output_tokens": 1,
                "total_tokens": math.nan,
                "unrelated_metric": math.inf,
            },
        )
    )
    malformed = guard.snapshot()["consumption"]
    assert malformed["input_tokens"] == estimated["input_tokens"]
    assert malformed["output_tokens"] == estimated["output_tokens"]
    assert malformed["total_tokens"] == estimated["total_tokens"]
    assert malformed["exact_tokens"] is False


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


def test_guard_ignores_volatile_provider_ids_in_tool_fingerprints() -> None:
    guard = TurnGuard(limits_for(UNATTENDED, "implementation"))
    guard.begin_attempt(100)

    for index in range(3):
        guard.observe(
            ProviderEvent(
                "tool.started",
                text="Read",
                metadata={
                    "id": "request-" + str(index),
                    "input": {
                        "file_path": "PLAN.gpt.md",
                        "request-id": "nested-" + str(index),
                        "parts": ("same",),
                        "segments": ["same"],
                    },
                },
            )
        )
        guard.observe(
            ProviderEvent(
                "tool.completed",
                text="unchanged",
                metadata={"tool_use_id": "request-" + str(index)},
            )
        )

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

    children = TurnGuard(replace(base, max_child_agents=1))
    children.observe(ProviderEvent("agent.child.started"))
    children.observe(ProviderEvent("agent.child.started"))
    assert children.violation() == "child-agents"
    admitted_children = TurnGuard(replace(base, max_child_agents=2))
    admitted_children.note_child_admissions(1)
    admitted_children.note_child_admissions(0)
    assert admitted_children.consumption.child_agents == 1
    with pytest.raises(ValueError, match="must not be negative"):
        admitted_children.note_child_admissions(-1)

    dollars = TurnGuard(replace(base, max_dollars=0.5))
    dollars.begin_attempt(1)
    dollars.observe(
        ProviderEvent(
            "usage.updated",
            metadata={"total_cost_usd": 0.6},
        )
    )
    assert dollars.violation() == "dollars"
    assert dollars.warning_due() is True
    assert dollars.snapshot()["consumption"]["exact_dollars"] is True

    cumulative = TurnGuard(replace(base, max_dollars=1.0))
    cumulative.begin_attempt(1)
    for unused in range(2):
        del unused
        cumulative.observe(
            ProviderEvent(
                "usage.updated",
                metadata={"total_cost_usd": 0.25},
            )
        )
    assert cumulative.snapshot()["consumption"]["dollars"] == 0.25

    multiple_children = TurnGuard(replace(base, max_child_agents=1))
    multiple_children.observe(
        ProviderEvent(
            "agent.child.started",
            metadata={"receiver_thread_ids": ["one", "two"]},
        )
    )
    assert multiple_children.violation() == "child-agents"

    deduplicated_child = TurnGuard(replace(base, max_child_agents=1))
    deduplicated_child.observe(
        ProviderEvent(
            "agent.child.started",
            metadata={"id": "tool-child"},
        )
    )
    deduplicated_child.observe(
        ProviderEvent(
            "agent.child.started",
            metadata={
                "child_id": "task-child",
                "tool_use_id": "tool-child",
            },
        )
    )
    assert deduplicated_child.consumption.child_agents == 1

    zero_clock = Clock()
    zero = TurnGuard(
        replace(base, max_seconds=0),
        monotonic=zero_clock,
    )
    zero_clock.advance(1)
    assert zero.violation() == "seconds"
    assert zero.warning_due() is True

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
    progress.observe(ProviderEvent("usage.updated", metadata={"usage": {}}))
    progress.establish_material_state("state-a")
    with pytest.raises(ValueError, match="digest"):
        progress.establish_material_state("")
    with pytest.raises(ValueError, match="digest"):
        progress.note_material_progress("")
    assert progress.note_material_progress("state-a") is False
    assert progress.note_material_progress("state-b") is True
    assert progress.violation() == ""
    assert progress.warning_due() is False

    camel_children = TurnGuard(replace(base, max_child_agents=1))
    camel_children.observe(
        ProviderEvent(
            "agent.child.started",
            metadata={"receiverThreadIds": []},
        )
    )
    camel_children.observe(ProviderEvent("agent.child.started", metadata={}))
    assert camel_children.violation() == "child-agents"


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
    with pytest.raises(ValueError, match="hard safety"):
        repeating.recover()
    interleaved = TurnGuard(base)
    interleaved.observe(
        ProviderEvent(
            "tool.started",
            text="first input",
            metadata={"id": "tool-a"},
        )
    )
    interleaved.observe(
        ProviderEvent(
            "tool.started",
            text="second input",
            metadata={"id": "tool-b"},
        )
    )
    interleaved.observe(
        ProviderEvent(
            "tool.completed",
            text="first output",
            metadata={"tool_use_id": "tool-a"},
        )
    )
    first_pair = interleaved.take_completed_tool_pair()
    interleaved.observe(
        ProviderEvent(
            "tool.completed",
            text="second output",
            metadata={"tool_use_id": "tool-b"},
        )
    )
    second_pair = interleaved.take_completed_tool_pair()
    assert first_pair
    assert second_pair
    assert first_pair != second_pair
    assert interleaved.violation() == ""

    stagnating_clock = Clock()
    stagnating = TurnGuard(
        replace(base, stagnation_seconds=1),
        monotonic=stagnating_clock,
    )
    stagnating_clock.advance(1)
    assert stagnating.violation() == "stagnation"
    stagnating.recover()
    assert stagnating.violation() == ""


def test_noop_file_change_cannot_clear_repetition_history() -> None:
    guard = TurnGuard(limits_for(UNATTENDED, "implementation"))
    guard.establish_material_state("workspace-a")
    started = ProviderEvent("tool.started", text="same")
    completed = ProviderEvent("tool.completed", text="same")

    for unused in range(2):
        del unused
        guard.observe(started)
        guard.observe(completed)
        assert guard.note_material_progress("workspace-a") is False
    guard.observe(started)
    guard.observe(completed)

    assert guard.violation() == "repeated-tool"


def test_changing_grok_tool_progress_refreshes_stagnation_clock() -> None:
    clock = Clock()
    limits = replace(
        limits_for(LIVE_SMOKE, "implementation"),
        stagnation_seconds=10,
    )
    guard = TurnGuard(limits, monotonic=clock)
    guard.begin_attempt(100)

    for event in grok_payload(
        {
            "type": "tool_call",
            "toolCallId": "call_1",
            "toolName": "read_file",
            "status": "in_progress",
            "rawInput": {"path": "src/main.rs"},
        }
    ):
        guard.observe(event)
    clock.advance(9)
    assert guard.violation() == ""

    for event in grok_payload(
        {
            "type": "tool_call_update",
            "toolCallId": "call_1",
            "status": "in_progress",
            "rawOutput": {"bytes": 128},
        }
    ):
        guard.observe(event)
    clock.advance(9)
    assert guard.violation() == ""

    for event in grok_payload(
        {
            "type": "tool_call_update",
            "toolCallId": "call_1",
            "status": "in_progress",
            "rawOutput": {"bytes": 256},
        }
    ):
        guard.observe(event)
    clock.advance(9)
    assert guard.violation() == ""

    for event in grok_payload(
        {
            "type": "tool_call_update",
            "toolCallId": "call_1",
            "status": "completed",
            "rawOutput": {"lines": 42},
        }
    ):
        guard.observe(event)
    clock.advance(9)
    assert guard.violation() == ""


def test_repeated_identical_tool_progress_does_not_defeat_stagnation() -> None:
    clock = Clock()
    limits = replace(
        limits_for(LIVE_SMOKE, "implementation"),
        stagnation_seconds=10,
    )
    guard = TurnGuard(limits, monotonic=clock)
    guard.begin_attempt(100)

    stuck = grok_payload(
        {
            "type": "tool_call_update",
            "toolCallId": "call_1",
            "status": "in_progress",
            "rawOutput": {"bytes": 128},
        }
    )
    for event in stuck:
        guard.observe(event)
    clock.advance(5)
    for event in stuck:
        guard.observe(event)
    clock.advance(5)
    for event in stuck:
        guard.observe(event)

    assert guard.violation() == "stagnation"


def test_alternating_grok_tool_progress_does_not_defeat_stagnation() -> None:
    clock = Clock()
    limits = replace(
        limits_for(LIVE_SMOKE, "implementation"),
        stagnation_seconds=10,
    )
    guard = TurnGuard(limits, monotonic=clock)
    guard.begin_attempt(100)

    first = grok_payload(
        {
            "type": "tool_call_update",
            "toolCallId": "call_1",
            "status": "in_progress",
            "rawOutput": {"line": "heartbeat-a"},
        }
    )
    second = grok_payload(
        {
            "type": "tool_call_update",
            "toolCallId": "call_1",
            "status": "in_progress",
            "rawOutput": {"line": "heartbeat-b"},
        }
    )
    for event in first:
        guard.observe(event)
    clock.advance(1)
    for event in second:
        guard.observe(event)
    for unused in range(20):
        del unused
        clock.advance(9)
        for event in first:
            guard.observe(event)
        if guard.violation():
            break
        clock.advance(9)
        for event in second:
            guard.observe(event)
        if guard.violation():
            break

    assert guard.violation() == "stagnation"


def test_grok_tool_starts_do_not_widen_stagnation_but_keep_absolute_limits() -> None:
    clock = Clock()
    limits = replace(
        limits_for(LIVE_SMOKE, "implementation"),
        stagnation_seconds=10,
        max_tool_calls=2,
    )
    guard = TurnGuard(limits, monotonic=clock)
    guard.begin_attempt(100)

    for event in grok_payload(
        {
            "type": "tool_call",
            "toolCallId": "call_1",
            "toolName": "read_file",
            "status": "in_progress",
            "rawInput": {"path": "a.py"},
        }
    ):
        guard.observe(event)
    clock.advance(10)
    assert guard.violation() == "stagnation"

    clock = Clock()
    limited = TurnGuard(limits, monotonic=clock)
    limited.begin_attempt(100)
    for event in grok_payload(
        {
            "type": "tool_call",
            "toolCallId": "call_1",
            "toolName": "read_file",
            "status": "in_progress",
            "rawInput": {"path": "a.py"},
        }
    ):
        limited.observe(event)
    for event in grok_payload(
        {
            "type": "tool_call",
            "toolCallId": "call_2",
            "toolName": "grep",
            "status": "in_progress",
            "rawInput": {"pattern": "TODO"},
        }
    ):
        limited.observe(event)
    for event in grok_payload(
        {
            "type": "tool_call",
            "toolCallId": "call_3",
            "toolName": "list_dir",
            "status": "in_progress",
            "rawInput": {"path": "."},
        }
    ):
        limited.observe(event)
    assert limited.violation() == "tool-calls"


def test_claude_tool_progress_does_not_refresh_stagnation() -> None:
    clock = Clock()
    limits = replace(
        limits_for(LIVE_SMOKE, "implementation"),
        stagnation_seconds=10,
    )
    guard = TurnGuard(limits, monotonic=clock)
    guard.begin_attempt(100)

    for index in range(3):
        guard.observe(
            ProviderEvent(
                "tool.progress",
                text="child heartbeat " + str(index),
                metadata={
                    "child_id": "bash-process",
                    "tool_use_id": "bash-tool",
                    "status": "running",
                    "output": {"tick": index},
                },
            )
        )
        clock.advance(9)

    assert guard.violation() == "stagnation"
    guard.observe(ProviderEvent("tool.progress"))
    guard.observe(
        ProviderEvent(
            "tool.progress",
            metadata={"tool_call_id": 7},
        )
    )
    guard.observe(
        ProviderEvent(
            "tool.progress",
            metadata={"tool_call_id": ""},
        )
    )
    clock.advance(1)
    assert guard.violation() == "stagnation"


def test_grok_tool_progress_ring_bounds_repeated_history() -> None:
    clock = Clock()
    limits = replace(
        limits_for(LIVE_SMOKE, "implementation"),
        stagnation_seconds=10,
    )
    guard = TurnGuard(limits, monotonic=clock)
    guard.begin_attempt(100)

    for index in range(10):
        for event in grok_payload(
            {
                "type": "tool_call_update",
                "toolCallId": "call_1",
                "status": "in_progress",
                "rawOutput": {"step": index},
            }
        ):
            guard.observe(event)
        clock.advance(1)

    assert guard.violation() == ""
    guard.establish_material_state("workspace-a")
    assert guard.note_material_progress("workspace-b") is True
    clock.advance(10)
    assert guard.violation() == "stagnation"


def test_normalized_tool_completion_still_detects_repeated_pairs() -> None:
    guard = TurnGuard(limits_for(UNATTENDED, "implementation"))
    guard.begin_attempt(100)
    for index in range(3):
        call_id = "call-" + str(index)
        for event in grok_payload(
            {
                "type": "tool_call",
                "toolCallId": call_id,
                "toolName": "read_file",
                "status": "in_progress",
                "rawInput": {"path": "same.py"},
            }
        ):
            guard.observe(event)
        for event in grok_payload(
            {
                "type": "tool_call_update",
                "toolCallId": call_id,
                "status": "completed",
                "rawOutput": {"lines": 1, "content": "same content"},
            }
        ):
            guard.observe(event)
    assert guard.violation() == "repeated-tool"


def test_grok_normalized_tool_stream_refreshes_and_still_bounds_stagnation() -> None:
    clock = Clock()
    limits = replace(
        limits_for(LIVE_SMOKE, "implementation"),
        stagnation_seconds=10,
    )
    guard = TurnGuard(limits, monotonic=clock)
    guard.begin_attempt(100)

    for event in grok_payload(
        {
            "type": "tool_call",
            "toolCallId": "call_1",
            "toolName": "run_terminal_cmd",
            "title": "Shell",
            "status": "in_progress",
            "rawInput": {"command": "make test"},
        }
    ):
        guard.observe(event)
    clock.advance(9)
    assert guard.violation() == ""

    for index in range(3):
        for event in grok_payload(
            {
                "type": "tool_call_update",
                "toolCallId": "call_1",
                "status": "in_progress",
                "rawOutput": {"line": "progress-" + str(index)},
            }
        ):
            guard.observe(event)
        clock.advance(9)
        assert guard.violation() == ""

    stuck = grok_payload(
        {
            "type": "tool_call_update",
            "toolCallId": "call_1",
            "status": "in_progress",
            "rawOutput": {"line": "progress-stuck"},
        }
    )
    for event in stuck:
        guard.observe(event)
    clock.advance(5)
    for event in stuck:
        guard.observe(event)
    clock.advance(5)
    for event in stuck:
        guard.observe(event)
    assert guard.violation() == "stagnation"
