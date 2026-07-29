from agent_harness.usage import normalize_usage


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
