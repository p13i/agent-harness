"""Structured session handoff envelope for cross-provider continuity.

When a dispatch routes to a provider that holds no native session for
the harness session, the worker prepends a ``session-handoff/v1``
envelope — a budgeted ``transcript.render`` of the canonical
transcript — ahead of the compiled context. The envelope is
harness-generated context, never operator text, so the
repeated-dispatch guard and attestation paths can distinguish it from
the operator's instruction stream.
"""

from __future__ import annotations

from agent_harness.providers.base import ProviderModel
from agent_harness.transcript import DEFAULT_TOKEN_BUDGET

HANDOFF_SCHEMA = "session-handoff/v1"

ORIGIN_PROVIDER_SWITCH = "provider-switch"
ORIGIN_FORK_SEED = "fork-seed"


def model_context_window(
    models: tuple[ProviderModel, ...],
    model_id: str,
) -> int | None:
    """Resolve the context window for one model from a models listing."""
    if model_id:
        for item in models:
            if item.model_id == model_id:
                return item.context_window
    for item in models:
        if item.default:
            return item.context_window
    if models:
        return models[0].context_window
    return None


def handoff_token_budget(
    context_window: int | None,
    reserve_output_tokens: int,
    committed_tokens: int,
) -> int:
    """Token budget for one envelope against the target model window."""
    if context_window is None:
        return DEFAULT_TOKEN_BUDGET
    budget = context_window - reserve_output_tokens - committed_tokens
    if budget < 1:
        return 1
    return budget


def handoff_envelope(
    *,
    session_id: str,
    source_provider: str,
    target_provider: str,
    target_model: str,
    transcript_digest: str,
    rendered: str,
) -> str:
    """Structured envelope marking harness-generated handoff context."""
    lines = [
        "# Session handoff",
        "",
        "Harness-generated context carried across a provider switch. It",
        "is not an operator instruction and does not change the",
        "operator's standing instructions.",
        "",
        "- Schema: `" + HANDOFF_SCHEMA + "`",
        "- Session: `" + session_id + "`",
        "- Source provider: `" + (source_provider or "none") + "`",
        "- Target provider: `" + target_provider + "`",
    ]
    if target_model:
        lines.append("- Target model: `" + target_model + "`")
    lines.extend(
        [
            "- Transcript digest: `" + transcript_digest + "`",
            "",
            rendered.rstrip(),
            "",
        ]
    )
    return "\n".join(lines)
