import json
from pathlib import Path

import pytest

from agent_harness.errors import HarnessError
from agent_harness.transfer import MachineKeys
from agent_harness.transfer import load_machine_keys
from agent_harness.transfer import open_transfer
from agent_harness.transfer import seal_transfer


def test_signed_encrypted_transfer_round_trip() -> None:
    source = MachineKeys.generate()
    destination = MachineKeys.generate()
    envelope = seal_transfer(
        {"session_id": "abc", "owner_epoch": 2},
        destination_encryption_public=(
            destination.public_bundle()["encryption"]
        ),
        source_signing_private=source.signing_private,
    )
    opened = open_transfer(
        envelope,
        destination_encryption_private=destination.encryption_private,
        source_signing_public=source.public_bundle()["signing"],
    )
    assert opened["owner_epoch"] == 2


def test_transfer_rejects_tampering() -> None:
    source = MachineKeys.generate()
    destination = MachineKeys.generate()
    envelope = seal_transfer(
        {"session_id": "abc"},
        destination_encryption_public=(
            destination.public_bundle()["encryption"]
        ),
        source_signing_private=source.signing_private,
    )
    payload = json.loads(envelope)
    payload["signature"] = payload["signature"][:-2] + "AA"
    tampered = json.dumps(payload).encode("utf-8")
    with pytest.raises(HarnessError):
        open_transfer(
            tampered,
            destination_encryption_private=destination.encryption_private,
            source_signing_public=source.public_bundle()["signing"],
        )


def test_machine_keys_persist(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    first = load_machine_keys(path)
    second = load_machine_keys(path)
    assert first.public_bundle() == second.public_bundle()
    assert path.stat().st_mode & 0o077 == 0
