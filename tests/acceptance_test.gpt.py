"""Versioned executable-evidence gate for product journeys."""

from __future__ import annotations

import json
import re
from pathlib import Path

import agent_harness

LOCAL_TARGETS = frozenset(
    {
        "//tests:acceptance_test",
        "//tests:chat_pty_test",
        "//tests:e2e_tests",
        "//tests:integration_scale_tests",
        "//tests:integration_tests",
        "//tests:parity_test",
        "//tests:provider_boundaries_test",
        "//tests:unit_tests",
        "//tools:install_test",
        "//tools:live_smoke_test",
        "//tools:ui_gallery_test",
        "//tools:wsl_e2e_test",
    }
)
EXTERNAL_TARGET = re.compile(r"^external://[A-Za-z0-9_./:-]+$")
EXTERNAL_STATES = frozenset({"pending", "passed", "failed"})


def main() -> int:
    root = Path(agent_harness.__file__).resolve().parent.parent
    contract = json.loads(
        (root / "contracts" / "acceptance.gpt.json").read_text(encoding="utf-8")
    )
    openapi = (root / "contracts" / "openapi.gpt.yaml").read_text(encoding="utf-8")
    presentation_schema = json.loads(
        (root / "schemas" / "presentation.schema.json").read_text(encoding="utf-8")
    )
    journeys = contract.get("journeys")
    if not isinstance(journeys, list):
        raise AssertionError("acceptance journeys must be an array")
    expected_ids = ["AH-AC-" + str(index).zfill(3) for index in range(1, 73)]
    actual_ids = [str(item.get("id", "")) for item in journeys]
    if actual_ids != expected_ids:
        raise AssertionError("acceptance journey IDs are incomplete")
    version = str(contract.get("contract_version", ""))
    if "version: " + version not in openapi:
        raise AssertionError("acceptance evidence is stale")
    if presentation_schema.get("$id") != ("p13i/agent-harness/presentation/v1"):
        raise AssertionError("presentation schema is stale")
    for path in (
        "/v1/sessions/{session_id}/turns:",
        "/v1/sessions/{session_id}/turns/{turn_id}:",
        "/v1/sessions/{session_id}/proof:",
        ("/v1/sessions/{session_id}/checkpoints/{checkpoint_id}/diff:"),
    ):
        if path not in openapi:
            raise AssertionError("presentation endpoint is undocumented")
    for control_contract in (
        '$ref: "#/components/schemas/ControlCommandRequest"',
        "ControlCommandRequest:",
        "target_command_id:",
        "Required when a later managed",
    ):
        if control_contract not in openapi:
            raise AssertionError("control command request is undocumented")
    if (
        re.search(
            r"pause, resume, and stop require an\s+empty object\.",
            openapi,
        )
        is None
    ):
        raise AssertionError("control command request is undocumented")
    for journey in journeys:
        _validate_journey(journey)
    return 0


def _validate_journey(journey: object) -> None:
    if not isinstance(journey, dict):
        raise AssertionError("acceptance journey must be an object")
    title = journey.get("title")
    if not isinstance(title, str) or not title.strip():
        raise AssertionError("acceptance journey title is required")
    evidence = journey.get("local_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AssertionError(str(journey.get("id")) + " lacks local evidence")
    for target in evidence:
        if not isinstance(target, str):
            raise AssertionError("evidence target must be text")
        if target in LOCAL_TARGETS:
            continue
        raise AssertionError("unknown executable evidence " + target)
    external = journey.get("external_evidence", [])
    if not isinstance(external, list):
        raise AssertionError("external evidence must be an array")
    for declaration in external:
        _validate_external_evidence(declaration)


def _validate_external_evidence(declaration: object) -> None:
    if not isinstance(declaration, dict):
        raise AssertionError("external evidence must be an object")
    target = declaration.get("target")
    if not isinstance(target, str):
        raise AssertionError("external evidence target must be text")
    if not EXTERNAL_TARGET.fullmatch(target):
        raise AssertionError("invalid external evidence target " + target)
    status = declaration.get("status")
    if status not in EXTERNAL_STATES:
        raise AssertionError("invalid external evidence status")
    if status == "passed":
        run_id = declaration.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise AssertionError("passed external evidence requires a run identifier")


if __name__ == "__main__":
    raise SystemExit(main())
