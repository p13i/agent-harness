"""Normalize Claude and Codex payloads into harness events."""

from __future__ import annotations

import json
import re
from typing import Any

from agent_harness.providers.base import ProviderEvent


ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize(text: str) -> str:
    return CONTROL.sub("", ANSI.sub("", text))


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
    if method in {
        "item/reasoning/summaryTextDelta",
        "item/reasoning/textDelta",
    }:
        return [
            ProviderEvent(
                "reasoning.summary.delta",
                text=payload_text(params.get("delta")),
                metadata=_selected(params, {"itemId"}),
                raw=raw,
            )
        ]
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
    if item_type == "reasoning":
        return [
            ProviderEvent(
                "reasoning.summary." + suffix,
                text=payload_text(
                    item.get("summary")
                    or item.get("content")
                    or item.get("text")
                ),
                raw=raw,
            )
        ]
    return [
        ProviderEvent(
            "tool." + item_type + "." + suffix,
            text=payload_text(item.get("text")),
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
            events.append(
                ProviderEvent(
                    "tool.started",
                    text=payload_text(block.get("name")),
                    metadata=_selected(block, {"id", "input", "name"}),
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
            text = payload_text(
                delta.get("text")
                or delta.get("thinking")
                or delta.get("partial_json")
            )
            normalized = "agent.message.delta"
            if delta_type == "thinking_delta":
                normalized = "reasoning.summary.delta"
            return [ProviderEvent(normalized, text=text, raw=raw)]
    return [ProviderEvent("provider.event", raw=raw)]


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

