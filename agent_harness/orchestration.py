"""Provider-neutral inputs for retry-safe external orchestration."""

from __future__ import annotations

import hashlib
import json
from typing import Any


MAX_ORCHESTRATOR_LENGTH = 128
MAX_JOB_ID_LENGTH = 256
MAX_STEP_ID_LENGTH = 256
MAX_AGENT_ROLE_LENGTH = 128


def normalize_external_ref(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("external_ref must be an object")
    orchestrator = _identifier(
        value.get("orchestrator"),
        "external_ref.orchestrator",
        MAX_ORCHESTRATOR_LENGTH,
    )
    job_id = _identifier(
        value.get("job_id"),
        "external_ref.job_id",
        MAX_JOB_ID_LENGTH,
    )
    if not orchestrator and not job_id:
        return {}
    if not orchestrator or not job_id:
        raise ValueError(
            "external_ref requires both orchestrator and job_id"
        )
    return {
        "orchestrator": orchestrator,
        "job_id": job_id,
    }


def normalize_turn_ref(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("turn_ref must be an object")
    step_id = _identifier(
        value.get("step_id"),
        "turn_ref.step_id",
        MAX_STEP_ID_LENGTH,
    )
    agent_role = _identifier(
        value.get("agent_role"),
        "turn_ref.agent_role",
        MAX_AGENT_ROLE_LENGTH,
    )
    if not step_id and not agent_role:
        return {}
    if not step_id or not agent_role:
        raise ValueError("turn_ref requires both step_id and agent_role")
    return {
        "step_id": step_id,
        "agent_role": agent_role,
    }


def normalize_creation_input(value: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in (
        "name",
        "workspace",
        "permission_mode",
        "execution_profile",
        "direct",
    ):
        if field not in value:
            continue
        normalized[field] = value[field]
    external_ref = normalize_external_ref(value.get("external_ref"))
    if external_ref:
        normalized["external_ref"] = external_ref
    routing = value.get("routing")
    if routing is not None:
        if not isinstance(routing, dict):
            raise ValueError("routing must be an object")
        normalized["routing"] = _normalized_object(routing)
    goal = value.get("goal")
    if goal is not None:
        if not isinstance(goal, dict):
            raise ValueError("goal must be an object")
        normalized["goal"] = _normalized_object(goal)
    return _normalized_object(normalized)


def normalized_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        _normalized_object(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def creation_digest(value: dict[str, Any]) -> str:
    return normalized_digest(normalize_creation_input(value))


def _identifier(value: object, field: str, maximum: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(field + " must be a string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(field + " is too long")
    if any(character.isspace() for character in normalized):
        raise ValueError(field + " must not contain whitespace")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(field + " contains a control character")
    return normalized


def _normalized_object(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _normalized_value(item)
        for key, item in sorted(value.items())
    }


def _normalized_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _normalized_object(value)
    if isinstance(value, list):
        return [_normalized_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalized_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("normalized input contains an unsupported value")
