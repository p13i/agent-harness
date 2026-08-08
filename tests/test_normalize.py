from agent_harness.providers.normalize import (
    claude_payload,
    codex_notification,
    kimi_payload,
    payload_text,
    redact_observable,
    sanitize,
)


def test_codex_agent_message_normalizes() -> None:
    events = codex_notification(
        "item/completed",
        {"item": {"type": "agentMessage", "text": "done"}},
    )
    assert events[0].event_type == "agent.message"
    assert events[0].text == "done"


def test_claude_result_normalizes() -> None:
    events = claude_payload(
        {
            "type": "result",
            "is_error": False,
            "result": "done",
            "session_id": "native",
        }
    )
    assert events[0].event_type == "turn.completed"
    assert events[0].native_session_id == "native"


def test_terminal_control_sequences_are_removed() -> None:
    assert sanitize("\x1b[31mred\x1b[0m\x00") == "red"


def test_codex_protocol_surface_normalizes() -> None:
    cases = [
        (
            "thread/started",
            {"thread": {"id": "thread-1"}},
            "session.started",
        ),
        (
            "turn/started",
            {"turn": {"id": "turn-1"}},
            "turn.started",
        ),
        (
            "turn/completed",
            {
                "turn": {
                    "id": "turn-1",
                    "status": "failed",
                    "durationMs": 4,
                    "error": {"message": "capacity"},
                }
            },
            "turn.completed",
        ),
        (
            "item/started",
            {
                "item": {
                    "type": "command_execution",
                    "id": "item-1",
                    "command": "pwd",
                    "aggregated_output": "workspace",
                    "status": "running",
                }
            },
            "tool.command.started",
        ),
        (
            "item/completed",
            {
                "item": {
                    "type": "commandExecution",
                    "command": "pwd",
                    "aggregatedOutput": "workspace",
                    "exitCode": 0,
                }
            },
            "tool.command.completed",
        ),
        (
            "item/completed",
            {
                "item": {
                    "type": "file_change",
                    "changes": [{"path": "file.txt"}],
                }
            },
            "file.change.completed",
        ),
        (
            "item/started",
            {
                "item": {
                    "type": "collabAgentToolCall",
                    "id": "child-1",
                    "receiverThreadIds": ["thread-2"],
                    "status": "running",
                }
            },
            "agent.child.started",
        ),
        (
            "item/completed",
            {"item": {"type": "subagent", "id": "child-1"}},
            "agent.child.completed",
        ),
        (
            "item/completed",
            {
                "item": {
                    "type": "collab_agent_tool_call",
                    "id": "child-2",
                    "status": "failed",
                }
            },
            "agent.child.failed",
        ),
        (
            "item/completed",
            {
                "item": {
                    "type": "collabAgentToolCall",
                    "id": "child-3",
                    "status": "cancelled",
                }
            },
            "agent.child.cancelled",
        ),
        (
            "item/completed",
            {"item": {"type": "reasoning", "summary": "summary"}},
            "reasoning.summary.completed",
        ),
        (
            "item/started",
            {"item": {"type": "mcp_call", "text": "lookup"}},
            "tool.mcp_call.started",
        ),
        (
            "item/started",
            {"item": {"type": "userMessage", "text": "prompt"}},
            "provider.event",
        ),
        (
            "item/agentMessage/delta",
            {"delta": "stream"},
            "agent.message.delta",
        ),
        (
            "item/commandExecution/outputDelta",
            {"delta": "output", "itemId": "item-1"},
            "tool.output.delta",
        ),
        (
            "item/reasoning/summaryTextDelta",
            {"delta": "reason", "itemId": "item-1"},
            "reasoning.summary.delta",
        ),
        (
            "thread/tokenUsage/updated",
            {"tokenUsage": {"totalTokens": 12}},
            "usage.updated",
        ),
        (
            "error",
            {"error": {"message": "failed"}},
            "provider.error",
        ),
        ("future/event", {}, "provider.event"),
    ]

    for method, params, expected in cases:
        events = codex_notification(method, params)
        assert events[0].event_type == expected

    assert (
        codex_notification(
            "item/reasoning/textDelta",
            {"delta": "hidden", "itemId": "item-1"},
        )
        == []
    )
    assert (
        codex_notification(
            "item/completed",
            {"item": {"type": "reasoning", "content": "hidden"}},
        )
        == []
    )

    assert (
        codex_notification("thread/started", {"thread": None})[0].native_session_id
        == ""
    )
    assert codex_notification("turn/started", {"turn": None})[0].native_turn_id == ""
    assert (
        codex_notification(
            "item/completed",
            {"item": {"type": "agent_message", "text": ""}},
        )
        == []
    )
    child = codex_notification(
        "item/started",
        {
            "item": {
                "type": "collab_agent_tool_call",
                "receiver_thread_ids": ["thread-2"],
            }
        },
    )[0]
    assert child.metadata == {"receiver_thread_ids": ["thread-2"]}


def test_claude_protocol_surface_normalizes() -> None:
    initialized = claude_payload(
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session-1",
            "model": "opus",
            "permissionMode": "approval",
            "tools": ["Read"],
        }
    )
    assert initialized[0].event_type == "session.started"

    message = {
        "content": [
            {"type": "text", "text": "answer"},
            {
                "type": "tool_use",
                "id": "tool-1",
                "name": "Read",
                "input": {"path": "file"},
            },
            {
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "content": "contents",
                "is_error": False,
            },
            None,
        ]
    }
    assistant = claude_payload({"type": "assistant", "message": message})
    assert [event.event_type for event in assistant] == [
        "agent.message",
        "tool.started",
        "tool.completed",
    ]
    for tool_name in ("Agent", "Task", "spawn_agent"):
        child = claude_payload(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "child-tool",
                            "name": tool_name,
                            "input": {"description": "Inspect"},
                        }
                    ]
                },
            }
        )
        assert child[0].event_type == "agent.child.started"
    user = claude_payload({"type": "user", "message": message})
    assert user[0].event_type == "user.message"
    assert claude_payload({"type": "assistant", "message": None}) == []
    assert claude_payload({"type": "assistant", "message": {"content": "text"}}) == []

    for delta, expected in (
        ({"type": "text_delta", "text": "text"}, "agent.message.delta"),
        (
            {"type": "input_json_delta", "partial_json": "{}"},
            "agent.message.delta",
        ),
    ):
        events = claude_payload(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": delta,
                },
            }
        )
        assert events[0].event_type == expected

    assert (
        claude_payload(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": "hidden",
                    },
                },
            }
        )
        == []
    )

    assert (
        claude_payload(
            {
                "type": "stream_event",
                "event": {"type": "message_stop"},
            }
        )[0].event_type
        == "provider.event"
    )
    assert (
        claude_payload({"type": "stream_event", "event": None})[0].event_type
        == "provider.event"
    )
    failed = claude_payload(
        {
            "type": "result",
            "is_error": True,
            "result": "failed",
            "duration_ms": 3,
            "usage": {"input_tokens": 2},
        }
    )
    assert failed[0].event_type == "turn.failed"
    assert failed[0].status == "failed"
    assert claude_payload({"type": "future"})[0].event_type == "provider.event"


def test_claude_task_lifecycle_normalizes_with_identity_and_usage() -> None:
    cases = (
        (
            {
                "type": "system",
                "subtype": "task_started",
                "task_id": "task-1",
                "tool_use_id": "tool-1",
                "session_id": "session-1",
            },
            "agent.child.started",
            "running",
        ),
        (
            {
                "type": "system",
                "subtype": "task_progress",
                "task_id": "task-1",
                "status": "completed",
                "usage": {"total_tokens": 10},
            },
            "agent.child.completed",
            "complete",
        ),
        (
            {
                "type": "system",
                "subtype": "task_notification",
                "task_id": "task-2",
                "status": "failed",
            },
            "agent.child.failed",
            "failed",
        ),
        (
            {
                "type": "system",
                "subtype": "task_updated",
                "task_id": "task-3",
                "patch": {"status": "cancelled"},
            },
            "agent.child.cancelled",
            "cancelled",
        ),
    )

    for payload, event_type, status in cases:
        event = claude_payload(payload)[0]
        assert event.event_type == event_type
        assert event.status == status
        assert event.metadata["child_id"] == payload["task_id"]
    assert cases[0][0]["session_id"] == "session-1"


def test_payload_text_handles_every_supported_shape() -> None:
    assert payload_text(["one", None, {"text": "two"}]) == "one\ntwo"
    assert payload_text({"message": "nested"}) == "nested"
    assert payload_text({"unknown": 1}) == '{"unknown": 1}'
    assert payload_text(None) == ""
    assert payload_text(42) == "42"


def test_redact_observable_walks_every_container_shape() -> None:
    assert redact_observable(["safe", {"api_key": "leaked"}]) == [
        "safe",
        {"api_key": "[REDACTED]"},
    ]
    assert redact_observable(7) == 7


def test_kimi_payload_falls_back_to_a_neutral_provider_event() -> None:
    empty = kimi_payload({"role": "assistant", "content": ""})
    assert [(item.event_type, item.text) for item in empty] == [
        ("provider.event", "")
    ]

    unknown_role = kimi_payload({"role": "system", "content": "note"})
    assert [(item.event_type, item.text) for item in unknown_role] == [
        ("provider.event", "note")
    ]


def test_grok_payload_maps_end_and_error() -> None:
    from agent_harness.providers.normalize import grok_payload

    end = grok_payload(
        {"type": "end", "sessionId": "s1", "stopReason": "end_turn"}
    )
    assert end[0].event_type == "turn.completed"
    assert end[0].native_session_id == "s1"

    err = grok_payload({"type": "error", "message": "boom"})
    assert err[0].event_type == "turn.failed"
    assert err[0].text == "boom"
