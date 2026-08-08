"""Contract gate for common and passthrough capabilities."""

import json
from pathlib import Path

import agent_harness

from agent_harness.providers.claude import ClaudeAdapter
from agent_harness.providers.codex import CodexAdapter
from agent_harness.providers.grok import GrokAdapter
from agent_harness.providers.kimi import KimiAdapter


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
        "kimi": KimiAdapter().status(),
        "grok": GrokAdapter().status(),
    }
    # "common" is the full-parity baseline kimi does not claim yet.
    for capability in contract["common"]:
        for provider in ("claude", "codex"):
            if capability not in statuses[provider].capabilities:
                raise AssertionError(
                    provider + " lacks " + capability
                )
    for capability in contract["codex"]:
        if capability not in statuses["codex"].capabilities:
            raise AssertionError("codex lacks " + capability)
    for capability in contract["kimi"]:
        if capability not in statuses["kimi"].capabilities:
            raise AssertionError("kimi lacks " + capability)
    for capability in contract["grok"]:
        if capability not in statuses["grok"].capabilities:
            raise AssertionError("grok lacks " + capability)
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
    if not providers.issubset(set(statuses)):
        raise AssertionError("passthrough provider coverage is incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
