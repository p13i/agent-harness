"""Versioned executable-evidence gate for product journeys."""

from __future__ import annotations

import json
from pathlib import Path
import re

import agent_harness


LOCAL_TARGETS = frozenset(
    {
        "//tests:acceptance_test",
        "//tests:chat_pty_test",
        "//tests:e2e_tests",
        "//tests:integration_tests",
        "//tests:parity_test",
        "//tests:unit_tests",
    }
)
EXTERNAL_TARGET = re.compile(r"^external://[A-Za-z0-9_./:-]+$")


def main() -> int:
    root = Path(agent_harness.__file__).resolve().parent.parent
    contract = json.loads(
        (root / "contracts" / "acceptance.gpt.json").read_text(
            encoding="utf-8"
        )
    )
    openapi = (root / "contracts" / "openapi.gpt.yaml").read_text(
        encoding="utf-8"
    )
    journeys = contract.get("journeys")
    if not isinstance(journeys, list):
        raise AssertionError("acceptance journeys must be an array")
    expected_ids = [
        "AH-AC-" + str(index).zfill(3) for index in range(1, 31)
    ]
    actual_ids = [str(item.get("id", "")) for item in journeys]
    if actual_ids != expected_ids:
        raise AssertionError("acceptance journey IDs are incomplete")
    version = str(contract.get("contract_version", ""))
    if "version: " + version not in openapi:
        raise AssertionError("acceptance evidence is stale")
    for journey in journeys:
        _validate_journey(journey)
    return 0


def _validate_journey(journey: object) -> None:
    if not isinstance(journey, dict):
        raise AssertionError("acceptance journey must be an object")
    title = journey.get("title")
    if not isinstance(title, str) or not title.strip():
        raise AssertionError("acceptance journey title is required")
    evidence = journey.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AssertionError(str(journey.get("id")) + " lacks evidence")
    for target in evidence:
        if not isinstance(target, str):
            raise AssertionError("evidence target must be text")
        if target in LOCAL_TARGETS:
            continue
        if EXTERNAL_TARGET.fullmatch(target):
            continue
        raise AssertionError("unknown executable evidence " + target)


if __name__ == "__main__":
    raise SystemExit(main())
