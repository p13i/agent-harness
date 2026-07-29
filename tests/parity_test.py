"""Contract gate for common and passthrough capabilities."""

import json
from pathlib import Path

import agent_harness

from agent_harness.providers.claude import ClaudeAdapter
from agent_harness.providers.codex import CodexAdapter


def main() -> int:
    root = Path(agent_harness.__file__).resolve().parent.parent
    contract = json.loads(
        (root / "contracts" / "parity.gpt.json").read_text(
            encoding="utf-8"
        )
    )
    statuses = {
        "claude": ClaudeAdapter().status(),
        "codex": CodexAdapter().status(),
    }
    for capability in contract["common"]:
        for status in statuses.values():
            if capability not in status.capabilities:
                raise AssertionError(
                    status.provider + " lacks " + capability
                )
    for capability in contract["codex"]:
        if capability not in statuses["codex"].capabilities:
            raise AssertionError("codex lacks " + capability)
    accepted = set(contract["statuses"])
    features = contract["features"]
    for feature in features:
        feature_id = feature["id"]
        for provider, status in statuses.items():
            coverage = feature[provider]
            if coverage not in accepted:
                raise AssertionError(
                    feature_id
                    + " has unsupported "
                    + provider
                    + " coverage"
                )
            capability = feature.get("capability")
            if not capability or coverage != "normalized":
                continue
            if capability not in status.capabilities:
                raise AssertionError(
                    provider
                    + " lacks normalized "
                    + feature_id
                    + " capability"
                )
    providers = set(contract["escape_hatch"]["providers"])
    if providers != set(statuses):
        raise AssertionError("passthrough provider coverage is incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
