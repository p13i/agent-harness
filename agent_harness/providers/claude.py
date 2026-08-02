"""Claude Agent SDK adapter with a pinned npx transport."""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    CLINotFoundError,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    RateLimitEvent,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk._internal.transport.subprocess_cli import (
    SubprocessCLITransport,
)

from agent_harness.child_gate import (
    _admission_key as child_admission_key,
)
from agent_harness.child_gate import (
    admit as admit_child_launch,
)
from agent_harness.errors import ProviderExhaustedError, ProviderUnavailableError
from agent_harness.process_control import (
    ProcessGroupIdentity,
    process_group_identity,
    terminate_process_group,
)
from agent_harness.providers.base import (
    ApprovalHandler,
    ChildLaunchGate,
    EventHandler,
    PrePromptGate,
    ProviderAdapter,
    ProviderEvent,
    ProviderModel,
    ProviderResult,
    ProviderStatus,
    provider_environment,
    trusted_executable,
)
from agent_harness.providers.normalize import claude_payload

CLAUDE_CODE_PACKAGE = "@anthropic-ai/claude-code@2.1.220"


def _child_gate(child_launch_gate: ChildLaunchGate | None) -> Any:
    async def gate_child_launch(
        hook_input: dict[str, Any],
        tool_use_id: str | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        del context
        tool_name = str(hook_input.get("tool_name", ""))
        if tool_name not in {"Agent", "spawn_agent"}:
            return {}
        if child_launch_gate is None:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "The harness child-agent launch gate is unavailable."
                    ),
                }
            }
        admission_key = child_admission_key(
            "claude",
            tool_name,
            hook_input,
            tool_use_id,
        )
        admitted = await asyncio.to_thread(
            admit_child_launch,
            child_launch_gate.database,
            child_launch_gate.command_id,
            child_launch_gate.limit,
            admission_key,
        )
        if admitted:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "The harness child-agent launch limit is exhausted."
                ),
            }
        }

    return gate_child_launch


class NpxClaudeTransport(SubprocessCLITransport):
    """Agent SDK subprocess transport pinned to the npm Claude Code package."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._process_group: ProcessGroupIdentity | None = None

    def _find_cli(self) -> str:
        try:
            return trusted_executable("npx")
        except (OSError, RuntimeError):
            raise CLINotFoundError("npx was not found")

    def _build_command(self) -> list[str]:
        command = super()._build_command()
        npx_command = [command[0], CLAUDE_CODE_PACKAGE, *command[1:]]
        launcher = "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])"
        return [sys.executable, "-I", "-S", "-c", launcher, *npx_command]

    async def connect(self) -> None:
        await super().connect()
        process = getattr(self, "_process", None)
        if process is None:
            raise RuntimeError("Claude transport process is missing")
        for unused_attempt in range(40):
            del unused_attempt
            try:
                self._process_group = process_group_identity(process.pid)
                return
            except RuntimeError:
                await asyncio.sleep(0.025)
        raise RuntimeError("Claude transport did not isolate its process group")

    async def close(self) -> None:
        process = getattr(self, "_process", None)
        identity = self._process_group
        try:
            await super().close()
        finally:
            if process is not None and identity is not None:
                cleanup = asyncio.create_task(
                    terminate_process_group(
                        process,
                        identity,
                        terminate_timeout=2.0,
                        kill_timeout=2.0,
                    )
                )
                await asyncio.shield(cleanup)
            self._process_group = None

    async def _check_claude_version(self) -> None:
        # The package version is pinned above and is newer than the SDK minimum.
        return


class ClaudeAdapter(ProviderAdapter):
    provider_id = "claude"

    def __init__(self) -> None:
        self._client: ClaudeSDKClient | None = None
        self._transport: NpxClaudeTransport | None = None

    def status(self) -> ProviderStatus:
        ready = True
        detail = "trusted node and npx are available"
        try:
            trusted_executable("node")
            trusted_executable("npx")
        except (OSError, RuntimeError) as error:
            ready = False
            detail = str(error)
        return ProviderStatus(
            provider="claude",
            ready=ready,
            detail=detail,
            capabilities=frozenset(
                {
                    "approval",
                    "checkpoint",
                    "cost-reporting",
                    "fork",
                    "hooks",
                    "mcp",
                    "plugins",
                    "proof-fault-barrier",
                    "proof-service-fault-barrier",
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
        child_launch_gate: ChildLaunchGate | None = None,
        pre_prompt_gate: PrePromptGate | None = None,
    ) -> ProviderResult:
        requested_native_session_id = native_session_id
        if requested_native_session_id:
            if not _has_native_session(workspace, requested_native_session_id):
                await event_handler(
                    ProviderEvent(
                        "recovery.native_resume_missing",
                        status="failed",
                        metadata={
                            "provider": "claude",
                            "native_session_id": requested_native_session_id,
                        },
                    )
                )
                raise ProviderUnavailableError("claude resume")
        else:
            native_session_id = str(uuid.uuid4())

        async def can_use_tool(
            tool_name: str,
            tool_input: dict[str, Any],
            context: ToolPermissionContext,
        ) -> PermissionResultAllow | PermissionResultDeny:
            request = _approval_request(tool_name, tool_input, context)
            decision = await approval_handler("claude/tool_permission", request)
            return _permission_result(decision, tool_input)

        gate_child_launch = _child_gate(child_launch_gate)

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
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="^(Agent|spawn_agent)$",
                        hooks=[gate_child_launch],
                    )
                ]
            },
            permission_prompt_tool_name=permission_prompt_tool_name,
            include_partial_messages=True,
            enable_file_checkpointing=True,
            extra_args=extra_args,
            max_buffer_size=16 * 1024 * 1024,
            stderr=capture_stderr,
        )
        if not requested_native_session_id:
            options.resume = None
            options.session_id = native_session_id

        transport = NpxClaudeTransport(_empty_stream(), options)
        client = ClaudeSDKClient(options=options, transport=transport)
        self._client = client
        self._transport = transport
        usage: dict[str, Any] = {}
        result_status = "failed"
        child_tool_ids: set[str] = set()
        try:
            await client.connect(
                _prompt_stream(
                    prompt,
                    native_session_id,
                    pre_prompt_gate=pre_prompt_gate,
                )
            )
            await event_handler(
                ProviderEvent(
                    "provider.prompt.accepted",
                    status="accepted",
                    native_session_id=native_session_id,
                )
            )
            async for message in client.receive_response():
                message_session_id = _message_session_id(message)
                if message_session_id:
                    native_session_id = message_session_id
                for event in _message_events(message):
                    event = _normalized_child_event(
                        event,
                        child_tool_ids,
                    )
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
                self._transport = None

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

    def process_identity(self) -> tuple[int, str]:
        transport = self._transport
        if transport is None:
            return (0, "")
        process = getattr(transport, "_process", None)
        if process is None:
            return (0, "")
        if process.returncode is not None:
            return (0, "")
        pid = int(getattr(process, "pid", 0) or 0)
        if pid <= 0:
            return (0, "")
        identity = transport._process_group
        if identity is None or identity.pid != pid:
            return (0, "")
        return (identity.pid, identity.pid_start)

    def native_session_available(
        self,
        workspace: Path,
        native_session_id: str,
    ) -> bool:
        return _has_native_session(workspace, native_session_id)


async def _empty_stream() -> AsyncIterator[dict[str, Any]]:
    if False:
        yield {}


async def _prompt_stream(
    prompt: str,
    session_id: str,
    *,
    pre_prompt_gate: PrePromptGate | None = None,
) -> AsyncIterator[dict[str, Any]]:
    if pre_prompt_gate is not None:
        await pre_prompt_gate()
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
        payload = dict(message.data)
        values = asdict(message)
        values.pop("data", None)
        for name, value in values.items():
            payload.setdefault(name, value)
        payload["type"] = "system"
        payload.setdefault("subtype", message.subtype)
        return claude_payload(payload)
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
        return {"type": "thinking"}
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


def _normalized_child_event(
    event: ProviderEvent,
    child_tool_ids: set[str],
) -> ProviderEvent:
    metadata = event.metadata
    if metadata is None:
        metadata = {}
    if event.event_type == "agent.child.started":
        tool_id = str(metadata.get("id", ""))
        if tool_id:
            child_tool_ids.add(tool_id)
        return event
    if event.event_type != "tool.completed":
        return event
    tool_id = str(metadata.get("tool_use_id", ""))
    if tool_id not in child_tool_ids:
        return event
    child_tool_ids.remove(tool_id)
    if bool(metadata.get("is_error", False)):
        return replace(
            event,
            event_type="agent.child.failed",
            status="failed",
        )
    return replace(event, event_type="agent.child.completed")


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
        if message.error == "rate_limit":
            return True
        for block in message.content:
            if not isinstance(block, TextBlock):
                continue
            if _looks_like_spend_limit(block.text):
                return True
        return False
    if isinstance(message, ResultMessage):
        if message.api_error_status in {429, 529}:
            return True
        if message.is_error and _looks_exhausted(message.subtype):
            return True
    return False


def _has_native_session(workspace: Path, session_id: str) -> bool:
    session_file = _native_session_path(workspace, session_id)
    return session_file.is_file()


def _native_session_path(workspace: Path, session_id: str) -> Path:
    slug = str(workspace.resolve()).replace("/", "-")
    return Path.home() / ".claude" / "projects" / slug / (session_id + ".jsonl")


def _looks_exhausted(message: str) -> bool:
    lowered = message.casefold()
    terms = (
        "rate limit",
        "usage limit",
        "spend limit",
        "quota",
        "capacity",
        "overage",
    )
    return any(term in lowered for term in terms)


def _looks_like_spend_limit(message: str) -> bool:
    lowered = message.casefold()
    return "monthly spend limit" in lowered and "/usage-credits" in lowered


def _bounded(value: str) -> str:
    return value[:4096]
