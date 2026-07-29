from agent_harness.providers.normalize import claude_payload
from agent_harness.providers.normalize import codex_notification
from agent_harness.providers.normalize import sanitize


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
