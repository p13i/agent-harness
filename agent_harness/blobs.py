"""Private content-addressed blob storage."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from agent_harness.errors import NotFoundError


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def put(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        target = self.path(digest)
        if target.exists():
            return digest
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="blob.",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            os.chmod(target, 0o600)
        finally:
            if temporary.exists():
                temporary.unlink()
        return digest

    def put_text(self, content: str) -> str:
        return self.put(content.encode("utf-8"))

    def get(self, digest: str) -> bytes:
        target = self.path(digest)
        try:
            return target.read_bytes()
        except OSError as error:
            raise NotFoundError("blob") from error

    def get_text(self, digest: str) -> str:
        return self.get(digest).decode("utf-8", errors="replace")

    def path(self, digest: str) -> Path:
        if len(digest) != 64:
            raise ValueError("blob digest must be a SHA-256 hex digest")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError("blob digest must be hexadecimal") from error
        return self.root / digest[:2] / digest[2:]

