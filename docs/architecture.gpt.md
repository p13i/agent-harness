# Agent harness architecture

The harness owns the durable conversation. Claude Code and
Codex are execution providers behind adapters; neither
provider transcript is the canonical database.

## State layers

SQLite in WAL mode stores the session UUID, ordered event
log, commands, goals, evidence, approvals, provider
attempts, checkpoints, routing decisions, usage snapshots,
and fleet ownership. A content-addressed blob store holds
large raw payloads without placing them in status or audit
responses.

The export command derives five private, atomic projections:

- session metadata JSON;
- `run-context.gpt.json`;
- transcript JSONL;
- readable transcript Markdown;
- formal goal JSON.

These files are rebuildable views. They do not determine
what is currently inside a model context window.

## Context windows

For a same-provider continuation, the harness resumes the
provider-native session ID when its native transcript is
available. The provider decides how to compact that native
history into its model context window.

For a provider change, the harness compiles a bounded
context package from the formal goal, constraints,
unresolved decisions, current workspace checkpoint,
evidence, compacted history, and recent events. The target
adapter submits that package as a new provider turn. Full
observable history stays in the harness even when only a
bounded subset fits into the target context window.

The `run-context.gpt.json` projection records the compiled
context and durable state used for inspection and recovery.
It is neither Claude's context window nor Codex's context
window.

## Execution

The local service listens on a mode-`0600` Unix socket and
requires a bearer token. Each session has one worker and one
ordered command queue. Message and control writes require
idempotency keys. Workers route only at turn boundaries,
preserve provider affinity when it remains eligible, and
fail over after capacity or readiness failures.

Claude uses the Python Agent SDK with a custom transport
that launches the pinned Claude Code npm package. Codex uses
the pinned app-server protocol. The stable event model
normalizes both providers while retaining raw payloads in
private blobs.

## Goals

A session may carry a finite or invariant formal goal with
constraints, completion predicates, evidence, and token,
turn, command, time, or cost budgets. Budget exhaustion
stops further provider turns and moves the session to
operator attention. Completion remains evidence-backed
rather than inferred from a provider's final sentence.

## Fleet transfer

Transfers encrypt a canonical session bundle to a
destination X25519 public key, derive the payload key with
HKDF, encrypt with AES-GCM, and sign the envelope with
Ed25519. The destination imports into a new owner epoch.
Finalization prevents the source from continuing as an
active owner.
