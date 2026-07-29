"""Identifier and timestamp helpers."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
import re
import uuid


IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,255}$")


def new_uuid() -> str:
    return str(uuid.uuid4())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_uuid(value: str, field: str = "identifier") -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError(field + " must be a UUID") from error
    return str(parsed)


def require_identifier(value: str, field: str = "identifier") -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(field + " is invalid")
    return value

