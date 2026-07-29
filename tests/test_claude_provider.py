from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import PermissionResultAllow
from claude_agent_sdk import PermissionResultDeny
from claude_agent_sdk import TextBlock
from claude_agent_sdk import ToolPermissionContext

from agent_harness.providers.claude import CLAUDE_CODE_PACKAGE
from agent_harness.providers.claude import NpxClaudeTransport
from agent_harness.providers.claude import _approval_request
from agent_harness.providers.claude import _message_events
from agent_harness.providers.claude import _permission_result


async def _empty_prompt():
    if False:
        yield {}


def test_npx_transport_pins_claude_code_package() -> None:
    options = ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code"},
        session_id="30d731ec-95c1-421c-9848-549419951165",
    )
    transport = NpxClaudeTransport(_empty_prompt(), options)
    transport._cli_path = "/usr/bin/npx"

    command = transport._build_command()

    assert command[:2] == ["/usr/bin/npx", CLAUDE_CODE_PACKAGE]
    assert "--input-format" in command
    assert "stream-json" in command


def test_npx_transport_maps_explicit_full_access() -> None:
    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        extra_args={"dangerously-skip-permissions": None},
    )
    transport = NpxClaudeTransport(_empty_prompt(), options)
    transport._cli_path = "/usr/bin/npx"

    command = transport._build_command()

    assert "--permission-mode" in command
    assert "bypassPermissions" in command
    assert "--dangerously-skip-permissions" in command


def test_permission_bridge_preserves_tool_context() -> None:
    context = ToolPermissionContext(
        tool_use_id="tool-1",
        title="Run tests?",
        description="Executes the scoped test target",
    )

    request = _approval_request("Bash", {"command": "make test"}, context)

    assert request["id"] == "tool-1"
    assert request["prompt"] == "Run tests?"
    assert request["input"] == {"command": "make test"}
    assert [choice["id"] for choice in request["choices"]] == [
        "accept",
        "decline",
    ]


def test_permission_decision_maps_to_sdk_types() -> None:
    allowed = _permission_result(
        {"decision": "accept"},
        {"command": "make test"},
    )
    denied = _permission_result(
        {"decision": "decline", "message": "Not now"},
        {"command": "make test"},
    )

    assert isinstance(allowed, PermissionResultAllow)
    assert allowed.updated_input == {"command": "make test"}
    assert isinstance(denied, PermissionResultDeny)
    assert denied.message == "Not now"


def test_sdk_assistant_message_normalizes() -> None:
    message = AssistantMessage(
        content=[TextBlock(text="done")],
        model="opus",
        session_id="native-session",
    )

    events = _message_events(message)

    assert events[0].event_type == "agent.message"
    assert events[0].text == "done"
