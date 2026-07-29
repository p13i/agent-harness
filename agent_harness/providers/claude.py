"""Claude Agent SDK adapter with a pinned npx transport."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
import shutil
from typing import Any
import uuid

from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ClaudeSDKClient
from claude_agent_sdk import ClaudeSDKError
from claude_agent_sdk import CLINotFoundError
from claude_agent_sdk import PermissionResultAllow
from claude_agent_sdk import PermissionResultDeny
from claude_agent_sdk import RateLimitEvent
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import ServerToolResultBlock
from claude_agent_sdk import ServerToolUseBlock
from claude_agent_sdk import StreamEvent
from claude_agent_sdk import SystemMessage
from claude_agent_sdk import TextBlock
from claude_agent_sdk import ThinkingBlock
from claude_agent_sdk import ToolPermissionContext
from claude_agent_sdk import ToolResultBlock
from claude_agent_sdk import ToolUseBlock
from claude_agent_sdk import UserMessage
from claude_agent_sdk._internal.transport.subprocess_cli import (
    SubprocessCLITransport,
)

from agent_harness.errors import ProviderExhaustedError
from agent_harness.errors import ProviderUnavailableError
from agent_harness.providers.base import ApprovalHandler
from agent_harness.providers.base import EventHandler
from agent_harness.providers.base import ProviderAdapter
from agent_harness.providers.base import ProviderEvent
from agent_harness.providers.base import ProviderModel
from agent_harness.providers.base import ProviderResult
from agent_harness.providers.base import ProviderStatus
from agent_harness.providers.base import provider_environment
from agent_harness.providers.normalize import claude_payload


CLAUDE_CODE_PACKAGE = "@anthropic-ai/claude-code@2.1.220"


class NpxClaudeTransport(SubprocessCLITransport):
    """Agent SDK subprocess transport pinned to the npm Claude Code package."""

    def _find_cli(self) -> str:
        path = shutil.which("npx")
        if path is None:
            raise CLINotFoundError("npx was not found")
        return path

    def _build_command(self) -> list[str]:
        command = super()._build_command()
        return [command[0], CLAUDE_CODE_PACKAGE, *command[1:]]

    async def _check_claude_version(self) -> None:
        # The package version is pinned above and is newer than the SDK minimum.
        return


class ClaudeAdapter(ProviderAdapter):
    provider_id = "claude"

    def __init__(self) -> None:
        self._client: ClaudeSDKClient | None = None

    def status(self) -> ProviderStatus:
        ready = shutil.which("npx") is not None
        detail = "npx is available"
        if not ready:
            detail = "npx was not found"
        return ProviderStatus(
            provider="claude",
            ready=ready,
            detail=detail,
            capabilities=frozenset(
                {
                    "approval",
                    "checkpoint",
                    "fork",
                    "hooks",
                    "mcp",
                    "plugins",
                    "resume",
                    "skills",
                    "streaming",
                    "subagents",
                    "tools",
                    "worktree",
                }
            ),
        )

    async def models(self, workspace: Path) -> tuple[ProviderModel, ...]:
        del workspace
        efforts = ("low", "medium", "high", "xhigh", "max")
        return (
            ProviderModel("opus", "Opus", efforts, None),
            ProviderModel("sonnet", "Sonnet", efforts, None, default=True),
            ProviderModel("haiku", "Haiku", efforts, None),
        )

    async def run_turn(
        self,
        *,
        workspace: Path,
        prompt: str,
        native_session_id: str,
        permission_mode: str,
        model: str,
        effort: str,
        event_handler: EventHandler,
        approval_handler: ApprovalHandler,
    ) -> ProviderResult:
        if not native_session_id:
            native_session_id = str(uuid.uuid4())

        async def can_use_tool(
            tool_name: str,
            tool_input: dict[str, Any],
            context: ToolPermissionContext,
        ) -> PermissionResultAllow | PermissionResultDeny:
            request = _approval_request(tool_name, tool_input, context)
            decision = await approval_handler("claude/tool_permission", request)
            return _permission_result(decision, tool_input)

        permission = _permission_mode(permission_mode)
        callback = None
        permission_prompt_tool_name = None
        if permission_mode == "approval":
            callback = can_use_tool
            permission_prompt_tool_name = "stdio"

        stderr_tail: list[str] = []

        def capture_stderr(line: str) -> None:
            stderr_tail.append(line)
            if len(stderr_tail) > 100:
                del stderr_tail[0]

        extra_args = {"replay-user-messages": None}
        if permission_mode == "full":
            extra_args["dangerously-skip-permissions"] = None
        options = ClaudeAgentOptions(
            tools={"type": "preset", "preset": "claude_code"},
            system_prompt={"type": "preset", "preset": "claude_code"},
            permission_mode=permission,
            resume=native_session_id,
            model=model,
            effort=effort,
            cwd=workspace,
            env=provider_environment("claude"),
            can_use_tool=callback,
            permission_prompt_tool_name=permission_prompt_tool_name,
            include_partial_messages=True,
            enable_file_checkpointing=True,
            extra_args=extra_args,
            max_buffer_size=16 * 1024 * 1024,
            stderr=capture_stderr,
        )
        if not _has_native_session(workspace, native_session_id):
            options.resume = None
            options.session_id = native_session_id

        transport = NpxClaudeTransport(_empty_stream(), options)
        client = ClaudeSDKClient(options=options, transport=transport)
        self._client = client
        usage: dict[str, Any] = {}
        result_status = "failed"
        try:
            await client.connect(_prompt_stream(prompt, native_session_id))
            async for message in client.receive_response():
                message_session_id = _message_session_id(message)
                if message_session_id:
                    native_session_id = message_session_id
                for event in _message_events(message):
                    if event.event_type == "turn.completed":
                        result_status = "complete"
                    if event.event_type == "turn.failed":
                        result_status = "failed"
                    if event.metadata is not None:
                        event_usage = event.metadata.get("usage")
                        if isinstance(event_usage, dict):
                            usage.update(event_usage)
                    await event_handler(event)
                if _message_exhausted(message):
                    raise ProviderExhaustedError("claude")
            return ProviderResult(
                provider="claude",
                native_session_id=native_session_id,
                native_turn_id="",
                status=result_status,
                usage=usage,
            )
        except ProviderExhaustedError:
            raise
        except ClaudeSDKError as error:
            detail = "\n".join(stderr_tail)
            if not detail:
                detail = str(error)
            if _looks_exhausted(detail):
                raise ProviderExhaustedError("claude") from error
            if detail:
                await event_handler(
                    ProviderEvent(
                        "provider.error",
                        text=_bounded(detail),
                        status="failed",
                    )
                )
            raise ProviderUnavailableError("claude") from error
        finally:
            try:
                await client.disconnect()
            finally:
                self._client = None

    async def interrupt(self) -> None:
        client = self._client
        if client is None:
            return
        await client.interrupt()

    async def steer(self, text: str) -> None:
        client = self._client
        if client is None:
            return
        await client.query(text)


async def _empty_stream() -> AsyncIterator[dict[str, Any]]:
    if False:
        yield {}


async def _prompt_stream(
    prompt: str,
    session_id: str,
) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "user",
        "session_id": session_id,
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
        "parent_tool_use_id": None,
    }


def _permission_mode(permission_mode: str) -> str:
    if permission_mode == "full":
        return "bypassPermissions"
    if permission_mode in {"plan", "read-only"}:
        return "plan"
    if permission_mode == "approval":
        return "default"
    raise ValueError("unsupported permission mode")


def _approval_request(
    tool_name: str,
    tool_input: dict[str, Any],
    context: ToolPermissionContext,
) -> dict[str, Any]:
    prompt = context.title
    if not prompt:
        prompt = "Claude wants to use " + tool_name
    return {
        "id": context.tool_use_id or str(uuid.uuid4()),
        "prompt": prompt,
        "reason": context.description or context.decision_reason or prompt,
        "tool": tool_name,
        "input": tool_input,
        "agent_id": context.agent_id,
        "blocked_path": context.blocked_path,
        "choices": [
            {"id": "accept", "label": "Accept"},
            {"id": "decline", "label": "Decline"},
        ],
    }


def _permission_result(
    decision: dict[str, Any],
    original_input: dict[str, Any],
) -> PermissionResultAllow | PermissionResultDeny:
    selected = str(
        decision.get(
            "decision",
            decision.get("choice", decision.get("choice_id", "")),
        )
    ).casefold()
    if selected in {"accept", "allow", "approve", "approved", "yes"}:
        updated_input = decision.get("updated_input")
        if not isinstance(updated_input, dict):
            updated_input = original_input
        return PermissionResultAllow(updated_input=updated_input)
    message = str(decision.get("message", "Declined by the operator"))
    return PermissionResultDeny(message=message)


def _message_events(message: object) -> list[ProviderEvent]:
    if isinstance(message, SystemMessage):
        return claude_payload(dict(message.data))
    if isinstance(message, AssistantMessage):
        payload = {
            "type": "assistant",
            "session_id": message.session_id,
            "message": {
                "content": [_content_payload(block) for block in message.content],
                "model": message.model,
                "usage": message.usage,
                "id": message.message_id,
                "stop_reason": message.stop_reason,
            },
        }
        return claude_payload(payload)
    if isinstance(message, UserMessage):
        content: object = message.content
        if isinstance(message.content, list):
            content = [_content_payload(block) for block in message.content]
        return claude_payload(
            {
                "type": "user",
                "message": {"role": "user", "content": content},
                "uuid": message.uuid,
            }
        )
    if isinstance(message, ResultMessage):
        payload = asdict(message)
        payload["type"] = "result"
        payload["modelUsage"] = payload.pop("model_usage", None)
        return claude_payload(payload)
    if isinstance(message, StreamEvent):
        return claude_payload(
            {
                "type": "stream_event",
                "uuid": message.uuid,
                "session_id": message.session_id,
                "event": message.event,
                "parent_tool_use_id": message.parent_tool_use_id,
            }
        )
    if isinstance(message, RateLimitEvent):
        metadata = asdict(message.rate_limit_info)
        return [
            ProviderEvent(
                "rate_limit.updated",
                status=str(metadata.get("status", "")),
                metadata=metadata,
                native_session_id=message.session_id,
            )
        ]
    return [ProviderEvent("provider.event")]


def _content_payload(block: object) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    if isinstance(block, ServerToolUseBlock):
        return {
            "type": "server_tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ServerToolResultBlock):
        return {
            "type": "advisor_tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
        }
    return {"type": "unknown"}


def _message_session_id(message: object) -> str:
    session_id = getattr(message, "session_id", None)
    if isinstance(session_id, str):
        return session_id
    if isinstance(message, SystemMessage):
        value = message.data.get("session_id")
        if isinstance(value, str):
            return value
    return ""


def _message_exhausted(message: object) -> bool:
    if isinstance(message, RateLimitEvent):
        return message.rate_limit_info.status == "rejected"
    if isinstance(message, AssistantMessage):
        return message.error == "rate_limit"
    if isinstance(message, ResultMessage):
        if message.api_error_status in {429, 529}:
            return True
        if message.is_error and _looks_exhausted(message.subtype):
            return True
    return False


def _has_native_session(workspace: Path, session_id: str) -> bool:
    slug = str(workspace.resolve()).replace("/", "-")
    session_file = Path.home() / ".claude" / "projects" / slug / (session_id + ".jsonl")
    return session_file.is_file()


def _looks_exhausted(message: str) -> bool:
    lowered = message.casefold()
    terms = ("rate limit", "usage limit", "quota", "capacity", "overage")
    return any(term in lowered for term in terms)


def _bounded(value: str) -> str:
    return value[:4096]
