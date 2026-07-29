"""Signed and encrypted checkpoint-transfer envelopes."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import NoEncryption
from cryptography.hazmat.primitives.serialization import PrivateFormat
from cryptography.hazmat.primitives.serialization import PublicFormat

from agent_harness.errors import HarnessError


TRANSFER_SCHEMA = "p13i/agent-harness/transfer/v1"


@dataclass(frozen=True)
class MachineKeys:
    encryption_private: X25519PrivateKey
    signing_private: Ed25519PrivateKey

    @classmethod
    def generate(cls) -> "MachineKeys":
        return cls(
            encryption_private=X25519PrivateKey.generate(),
            signing_private=Ed25519PrivateKey.generate(),
        )

    def public_bundle(self) -> dict[str, str]:
        encryption_public = self.encryption_private.public_key().public_bytes(
            Encoding.Raw,
            PublicFormat.Raw,
        )
        signing_public = self.signing_private.public_key().public_bytes(
            Encoding.Raw,
            PublicFormat.Raw,
        )
        return {
            "encryption": _encode(encryption_public),
            "signing": _encode(signing_public),
        }

    def private_bundle(self) -> dict[str, str]:
        encryption = self.encryption_private.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        )
        signing = self.signing_private.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        )
        return {
            "encryption": _encode(encryption),
            "signing": _encode(signing),
        }

    @classmethod
    def from_private_bundle(cls, value: dict[str, str]) -> "MachineKeys":
        return cls(
            encryption_private=X25519PrivateKey.from_private_bytes(
                _decode(value["encryption"])
            ),
            signing_private=Ed25519PrivateKey.from_private_bytes(
                _decode(value["signing"])
            ),
        )


def load_machine_keys(path: Path) -> MachineKeys:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HarnessError(
                "E_KEYS",
                "machine key bundle cannot be read",
            ) from error
        if not isinstance(payload, dict):
            raise HarnessError("E_KEYS", "machine key bundle is invalid")
        try:
            return MachineKeys.from_private_bundle(payload)
        except (KeyError, ValueError, binascii.Error) as error:
            raise HarnessError(
                "E_KEYS",
                "machine key bundle is invalid",
            ) from error
    keys = MachineKeys.generate()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(keys.private_bundle(), stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return keys


def seal_transfer(
    payload: dict[str, Any],
    *,
    destination_encryption_public: str,
    source_signing_private: Ed25519PrivateKey,
) -> bytes:
    plaintext = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    ephemeral = X25519PrivateKey.generate()
    destination = X25519PublicKey.from_public_bytes(
        _decode(destination_encryption_public)
    )
    shared = ephemeral.exchange(destination)
    key = _derive_key(shared)
    nonce = os.urandom(12)
    ephemeral_public = ephemeral.public_key().public_bytes(
        Encoding.Raw,
        PublicFormat.Raw,
    )
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, ephemeral_public)
    signed = ephemeral_public + nonce + ciphertext
    signature = source_signing_private.sign(signed)
    envelope = {
        "schema": TRANSFER_SCHEMA,
        "ephemeral_public": _encode(ephemeral_public),
        "nonce": _encode(nonce),
        "ciphertext": _encode(ciphertext),
        "signature": _encode(signature),
    }
    return json.dumps(envelope, sort_keys=True).encode("utf-8")


def open_transfer(
    envelope_bytes: bytes,
    *,
    destination_encryption_private: X25519PrivateKey,
    source_signing_public: str,
) -> dict[str, Any]:
    try:
        envelope = json.loads(envelope_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HarnessError(
            "E_TRANSFER_INVALID",
            "transfer envelope is not valid JSON",
        ) from error
    if not isinstance(envelope, dict):
        raise HarnessError("E_TRANSFER_INVALID", "transfer envelope is invalid")
    if envelope.get("schema") != TRANSFER_SCHEMA:
        raise HarnessError(
            "E_TRANSFER_SCHEMA",
            "transfer schema is unsupported",
        )
    try:
        ephemeral_public = _decode(str(envelope["ephemeral_public"]))
        nonce = _decode(str(envelope["nonce"]))
        ciphertext = _decode(str(envelope["ciphertext"]))
        signature = _decode(str(envelope["signature"]))
    except (KeyError, ValueError) as error:
        raise HarnessError(
            "E_TRANSFER_INVALID",
            "transfer envelope is incomplete",
        ) from error
    signed = ephemeral_public + nonce + ciphertext
    verifier = Ed25519PublicKey.from_public_bytes(_decode(source_signing_public))
    try:
        verifier.verify(signature, signed)
    except (InvalidSignature, ValueError) as error:
        raise HarnessError(
            "E_TRANSFER_SIGNATURE",
            "transfer signature is invalid",
        ) from error
    source_ephemeral = X25519PublicKey.from_public_bytes(ephemeral_public)
    shared = destination_encryption_private.exchange(source_ephemeral)
    key = _derive_key(shared)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, ephemeral_public)
    except ValueError as error:
        raise HarnessError(
            "E_TRANSFER_DECRYPT",
            "transfer cannot be decrypted",
        ) from error
    decoded = json.loads(plaintext)
    if not isinstance(decoded, dict):
        raise HarnessError(
            "E_TRANSFER_INVALID",
            "transfer payload is not an object",
        )
    return decoded


def _derive_key(shared: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"p13i-agent-harness-transfer-v1",
    ).derive(shared)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode(value: str) -> bytes:
    return base64.b64decode(
        value.encode("ascii"),
        altchars=b"-_",
        validate=True,
    )
