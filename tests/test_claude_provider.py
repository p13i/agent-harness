import sys

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    TextBlock,
    ToolPermissionContext,
)

from agent_harness.providers.claude import (
    CLAUDE_CODE_PACKAGE,
    NpxClaudeTransport,
    _approval_request,
    _message_events,
    _message_exhausted,
    _permission_result,
)


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

    assert command[:6] == [
        sys.executable,
        "-I",
        "-B",
        "-S",
        "-c",
        "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])",
    ]
    assert command.index("-B") < command.index("-c")
    assert command[6:8] == ["/usr/bin/npx", CLAUDE_CODE_PACKAGE]
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


def test_monthly_spend_limit_requests_provider_failover() -> None:
    message = AssistantMessage(
        content=[
            TextBlock(
                text=(
                    "You've hit your monthly spend limit. "
                    "Run /usage-credits to manage your limit."
                )
            )
        ],
        model="fable",
        session_id="native-session",
    )
    ordinary = AssistantMessage(
        content=[TextBlock(text="The quota plan is healthy.")],
        model="opus",
        session_id="native-session",
    )

    assert _message_exhausted(message)
    assert not _message_exhausted(ordinary)
