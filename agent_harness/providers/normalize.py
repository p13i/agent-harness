"""Normalize Claude and Codex payloads into harness events."""

from __future__ import annotations

import json
import re
from typing import Any

from agent_harness.providers.base import ProviderEvent

ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SENSITIVE_NAME = re.compile(
    r"(?:TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY|API_KEY)",
    re.IGNORECASE,
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|password|secret|token)[\"']?)"
    r"(\s*[:=]\s*)"
    r"(?:bearer\s+[A-Za-z0-9._~+/=-]+|\"(?:\\.|[^\"])*\"|"
    r"'(?:\\.|[^'])*'|[^\s,;]+)"
)
BEARER_VALUE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
SECRET_TOKEN = re.compile(r"\b(?:gh[opsu]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b")


def sanitize(text: str) -> str:
    value = CONTROL.sub("", ANSI.sub("", text))
    value = SECRET_ASSIGNMENT.sub(
        lambda match: match.group(1) + match.group(2) + "[REDACTED]",
        value,
    )
    value = BEARER_VALUE.sub(
        lambda match: match.group(1) + "[REDACTED]",
        value,
    )
    return SECRET_TOKEN.sub("[REDACTED]", value)


def redact_observable(value: object) -> object:
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, list):
        return [redact_observable(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for name, item in value.items():
            if SENSITIVE_NAME.search(str(name)):
                result[str(name)] = "[REDACTED]"
                continue
            result[str(name)] = redact_observable(item)
        return result
    return value


def codex_notification(
    method: str,
    params: dict[str, Any],
) -> list[ProviderEvent]:
    raw = {"method": method, "params": params}
    if method == "thread/started":
        thread = params.get("thread")
        native_id = ""
        if isinstance(thread, dict):
            native_id = _optional_text(thread.get("id"))
        return [
            ProviderEvent(
                "session.started",
                raw=raw,
                native_session_id=native_id,
            )
        ]
    if method == "turn/started":
        turn = params.get("turn")
        turn_id = ""
        if isinstance(turn, dict):
            turn_id = _optional_text(turn.get("id"))
        return [
            ProviderEvent(
                "turn.started",
                status="running",
                raw=raw,
                native_turn_id=turn_id,
            )
        ]
    if method == "turn/completed":
        turn = params.get("turn")
        status = "complete"
        text = ""
        metadata: dict[str, Any] = {}
        turn_id = ""
        if isinstance(turn, dict):
            status = str(turn.get("status", "complete"))
            turn_id = _optional_text(turn.get("id"))
            error = turn.get("error")
            if error is not None:
                text = payload_text(error)
            for field in ("status", "durationMs"):
                if field in turn:
                    metadata[field] = turn[field]
        return [
            ProviderEvent(
                "turn.completed",
                text=text,
                status=status,
                metadata=metadata,
                raw=raw,
                native_turn_id=turn_id,
            )
        ]
    if method in {"item/started", "item/completed"}:
        item = params.get("item")
        if isinstance(item, dict):
            return _codex_item(method, item, raw)
    if method == "item/agentMessage/delta":
        return [
            ProviderEvent(
                "agent.message.delta",
                text=payload_text(params.get("delta")),
                raw=raw,
            )
        ]
    if method == "item/commandExecution/outputDelta":
        return [
            ProviderEvent(
                "tool.output.delta",
                text=payload_text(params.get("delta")),
                metadata=_selected(params, {"itemId"}),
                raw=raw,
            )
        ]
    if method == "item/reasoning/summaryTextDelta":
        return [
            ProviderEvent(
                "reasoning.summary.delta",
                text=payload_text(params.get("delta")),
                metadata=_selected(params, {"itemId"}),
                raw=raw,
            )
        ]
    if method == "item/reasoning/textDelta":
        return []
    if method == "thread/tokenUsage/updated":
        return [
            ProviderEvent(
                "usage.updated",
                metadata=_selected(params, {"tokenUsage"}),
                raw=raw,
            )
        ]
    if method == "error":
        return [
            ProviderEvent(
                "provider.error",
                text=payload_text(params.get("error")),
                status="failed",
                raw=raw,
            )
        ]
    return [ProviderEvent("provider.event", raw=raw)]


def claude_payload(payload: dict[str, Any]) -> list[ProviderEvent]:
    event_type = str(payload.get("type", "provider.event"))
    if event_type == "system":
        task_events = _claude_task(payload)
        if task_events:
            return task_events
    if event_type == "system" and payload.get("subtype") == "init":
        return [
            ProviderEvent(
                "session.started",
                metadata=_selected(
                    payload,
                    {"model", "permissionMode", "tools"},
                ),
                raw=payload,
                native_session_id=_optional_text(payload.get("session_id")),
            )
        ]
    if event_type == "assistant":
        return _claude_message(payload.get("message"), "agent", payload)
    if event_type == "user":
        return _claude_message(payload.get("message"), "user", payload)
    if event_type == "stream_event":
        event = payload.get("event")
        if isinstance(event, dict):
            return _claude_stream_event(event, payload)
    if event_type == "result":
        failed = bool(payload.get("is_error", False))
        normalized = "turn.completed"
        status = "complete"
        if failed:
            normalized = "turn.failed"
            status = "failed"
        return [
            ProviderEvent(
                normalized,
                text=payload_text(payload.get("result")),
                status=status,
                metadata=_selected(
                    payload,
                    {
                        "duration_ms",
                        "duration_api_ms",
                        "total_cost_usd",
                        "usage",
                    },
                ),
                raw=payload,
                native_session_id=_optional_text(payload.get("session_id")),
            )
        ]
    return [ProviderEvent("provider.event", raw=payload)]


def _claude_task(payload: dict[str, Any]) -> list[ProviderEvent]:
    subtype = str(payload.get("subtype", ""))
    if subtype not in {
        "task_started",
        "task_progress",
        "task_notification",
        "task_updated",
    }:
        return []
    child_id = str(payload.get("task_id", ""))
    metadata: dict[str, Any] = {"child_id": child_id}
    tool_use_id = payload.get("tool_use_id")
    if isinstance(tool_use_id, str):
        metadata["tool_use_id"] = tool_use_id
    usage = payload.get("usage")
    if isinstance(usage, dict):
        metadata["usage"] = usage
    status = str(payload.get("status", "")).casefold()
    if subtype == "task_updated":
        patch = payload.get("patch")
        if isinstance(patch, dict):
            status = str(patch.get("status", status)).casefold()
    suffix = "progress"
    event_status = status
    if subtype == "task_started":
        suffix = "started"
        event_status = "running"
    if status == "completed":
        suffix = "completed"
        event_status = "complete"
    if status == "failed":
        suffix = "failed"
    if status in {"stopped", "killed", "cancelled", "canceled"}:
        suffix = "cancelled"
        event_status = "cancelled"
    return [
        ProviderEvent(
            "agent.child." + suffix,
            status=event_status,
            metadata=metadata,
            raw=payload,
            native_session_id=_optional_text(payload.get("session_id")),
        )
    ]


def _codex_item(
    method: str,
    item: dict[str, Any],
    raw: dict[str, Any],
) -> list[ProviderEvent]:
    item_type = str(item.get("type", "item"))
    suffix = "started"
    if method == "item/completed":
        suffix = "completed"
    if item_type in {"agent_message", "agentMessage"}:
        text = payload_text(item.get("text"))
        if not text:
            return []
        return [ProviderEvent("agent.message", text=text, raw=raw)]
    if item_type in {"user_message", "userMessage"}:
        return [ProviderEvent("provider.event", raw=raw)]
    if item_type in {"command_execution", "commandExecution"}:
        command = payload_text(item.get("command"))
        output = payload_text(item.get("aggregated_output"))
        if not output:
            output = payload_text(item.get("aggregatedOutput"))
        text = command
        if output:
            text += "\n" + output
        return [
            ProviderEvent(
                "tool.command." + suffix,
                text=text,
                metadata=_selected(
                    item,
                    {"id", "exit_code", "exitCode", "status"},
                ),
                raw=raw,
            )
        ]
    if item_type in {"file_change", "fileChange"}:
        return [
            ProviderEvent(
                "file.change." + suffix,
                text=payload_text(item.get("changes")),
                raw=raw,
            )
        ]
    if item_type in {
        "collab_agent_tool_call",
        "collabAgentToolCall",
        "subagent",
    }:
        child_suffix = suffix
        status = str(item.get("status", "")).casefold()
        if status in {"failed", "error"}:
            child_suffix = "failed"
        if status in {"cancelled", "canceled"}:
            child_suffix = "cancelled"
        return [
            ProviderEvent(
                "agent.child." + child_suffix,
                metadata=_selected(
                    item,
                    {
                        "id",
                        "receiver_thread_ids",
                        "receiverThreadIds",
                        "status",
                    },
                ),
                raw=raw,
            )
        ]
    if item_type == "reasoning":
        summary = payload_text(item.get("summary"))
        if not summary:
            return []
        return [
            ProviderEvent(
                "reasoning.summary." + suffix,
                text=summary,
                raw=raw,
            )
        ]
    return [
        ProviderEvent(
            "tool." + item_type + "." + suffix,
            text=payload_text(item.get("text")),
            metadata=_selected_redacted(
                item,
                {"arguments", "id", "input", "name", "status"},
            ),
            raw=raw,
        )
    ]


def _claude_message(
    value: object,
    role: str,
    raw: dict[str, Any],
) -> list[ProviderEvent]:
    if not isinstance(value, dict):
        return []
    content = value.get("content")
    if not isinstance(content, list):
        return []
    events: list[ProviderEvent] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", ""))
        if block_type == "text":
            text = payload_text(block.get("text"))
            if text:
                events.append(
                    ProviderEvent(
                        role + ".message",
                        text=text,
                        raw=raw,
                    )
                )
        elif block_type == "tool_use":
            tool_name = payload_text(block.get("name"))
            normalized = "tool.started"
            if tool_name.casefold() in {"agent", "task", "spawn_agent"}:
                normalized = "agent.child.started"
            metadata = _selected(block, {"id", "name"})
            if "input" in block:
                metadata["input"] = redact_observable(block["input"])
            events.append(
                ProviderEvent(
                    normalized,
                    text=tool_name,
                    metadata=metadata,
                    raw=raw,
                )
            )
        elif block_type == "tool_result":
            events.append(
                ProviderEvent(
                    "tool.completed",
                    text=payload_text(block.get("content")),
                    metadata=_selected(
                        block,
                        {"tool_use_id", "is_error"},
                    ),
                    raw=raw,
                )
            )
    return events


def _claude_stream_event(
    event: dict[str, Any],
    raw: dict[str, Any],
) -> list[ProviderEvent]:
    event_type = str(event.get("type", ""))
    if event_type == "content_block_delta":
        delta = event.get("delta")
        if isinstance(delta, dict):
            delta_type = str(delta.get("type", ""))
            if delta_type == "thinking_delta":
                return []
            text = payload_text(delta.get("text") or delta.get("partial_json"))
            normalized = "agent.message.delta"
            return [ProviderEvent(normalized, text=text, raw=raw)]
    return [ProviderEvent("provider.event", raw=raw)]


def grok_payload(payload: dict[str, Any]) -> list[ProviderEvent]:
    """Normalize one Grok Build ``--output-format streaming-json`` line.

    Grok emits type-tagged NDJSON (``text``, ``thought``, ``tool_call``,
    ``usage``, ``end``, ``error``, …). ``end`` always carries
    ``sessionId`` for resume; ``error`` is terminal for the turn.
    """
    event_type = str(payload.get("type", ""))
    session_id = _optional_text(
        payload.get("sessionId") or payload.get("session_id")
    )
    if event_type == "text":
        text = payload_text(payload.get("data"))
        return [
            ProviderEvent(
                "agent.message",
                text=text,
                raw=payload,
                native_session_id=session_id,
            )
        ]
    if event_type == "thought":
        text = payload_text(payload.get("data"))
        return [
            ProviderEvent(
                "provider.event",
                text=text,
                raw=payload,
                native_session_id=session_id,
            )
        ]
    if event_type in {"tool_call", "tool_call_update"}:
        return [
            ProviderEvent(
                "provider.event",
                text=str(payload.get("toolName") or payload.get("title") or ""),
                raw=payload,
                native_session_id=session_id,
                metadata={
                    "grok_type": event_type,
                    "tool_call_id": payload.get("toolCallId"),
                    "status": payload.get("status"),
                },
            )
        ]
    if event_type == "usage":
        usage = payload.get("usage")
        metadata: dict[str, Any] = {}
        if isinstance(usage, dict):
            metadata["usage"] = usage
        return [
            ProviderEvent(
                "provider.event",
                raw=payload,
                native_session_id=session_id,
                metadata=metadata or None,
            )
        ]
    if event_type == "end":
        metadata = {}
        usage = payload.get("usage")
        if isinstance(usage, dict):
            metadata["usage"] = usage
        return [
            ProviderEvent(
                "turn.completed",
                status="complete",
                raw=payload,
                native_session_id=session_id,
                metadata=metadata or None,
            )
        ]
    if event_type == "error":
        message = payload_text(payload.get("message") or payload.get("data"))
        return [
            ProviderEvent(
                "turn.failed",
                text=message,
                status="failed",
                raw=payload,
                native_session_id=session_id,
            )
        ]
    return [
        ProviderEvent(
            "provider.event",
            raw=payload,
            native_session_id=session_id,
        )
    ]


def kimi_payload(payload: dict[str, Any]) -> list[ProviderEvent]:
    """Normalize one Kimi Code ``--output-format stream-json`` line.

    Kimi's envelope is flat ``{"role": ..., "content": ...}`` with a
    string content, NOT Claude's ``{"type": "assistant", "message":
    {"content": [blocks]}}``. Reusing the Claude normalizer here silently
    yields nothing, which is why this is its own function.

    The stream ends with a ``role: "meta"`` line of type
    ``session.resume_hint`` carrying the session id, so the turn's
    completion and its native session id arrive together.
    """
    role = str(payload.get("role", ""))
    if role == "meta":
        if str(payload.get("type", "")) != "session.resume_hint":
            return [ProviderEvent("provider.event", raw=payload)]
        return [
            ProviderEvent(
                "turn.completed",
                status="complete",
                raw=payload,
                native_session_id=_optional_text(payload.get("session_id")),
            )
        ]
    text = payload_text(payload.get("content"))
    if not text:
        return [ProviderEvent("provider.event", raw=payload)]
    if role == "assistant":
        return [ProviderEvent("agent.message", text=text, raw=payload)]
    if role == "user":
        return [ProviderEvent("user.message", text=text, raw=payload)]
    return [ProviderEvent("provider.event", text=text, raw=payload)]


def payload_text(value: object) -> str:
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, list):
        parts = [payload_text(item) for item in value]
        return "\n".join(item for item in parts if item)
    if isinstance(value, dict):
        for field in ("text", "message", "content"):
            if field in value:
                return payload_text(value[field])
        return sanitize(json.dumps(value, sort_keys=True))
    if value is None:
        return ""
    return sanitize(str(value))


def _optional_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _selected(
    value: dict[str, Any],
    fields: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in sorted(fields):
        if field in value:
            result[field] = value[field]
    return result


def _selected_redacted(
    value: dict[str, Any],
    fields: set[str],
) -> dict[str, Any]:
    selected = _selected(value, fields)
    redacted = redact_observable(selected)
    if isinstance(redacted, dict):
        return redacted
    return {}
