# Agent harness architecture

The harness owns the durable conversation. Claude Code and
Codex are execution providers behind adapters; neither
provider transcript is the canonical database.

## State layers

SQLite schema version 3 in WAL mode stores the session UUID,
optional external orchestration reference, ordered event
log, commands and turn references, goals, evidence,
approvals, provider attempts, pre-dispatch checkpoints,
reconciliation records, routing decisions, usage snapshots,
safety profiles and envelopes, process leases, idempotent
mutation receipts, and fleet ownership. A content-addressed
blob store holds large raw payloads without placing them in
status or audit responses.

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

## External orchestration

An external orchestrator can bind its stable job identifier
to one harness session through `external_ref`. The pair of
orchestrator and job ID is unique and survives restart,
portable synchronization, provider switching, and fleet
transfer. `ensure_session` and managed turns combine this
binding with caller-owned idempotency keys so client retries
cannot duplicate sessions or provider turns.

Managed turns may carry a provider-neutral `turn_ref` for
step and agent-role correlation. The harness persists this
reference with the command and events but does not add it to
the provider prompt.

## Execution

The local service listens on a mode-`0600` Unix socket and
requires a bearer token. Each session has one worker and one
ordered command queue. Message and control writes require
idempotency keys. Workers checkpoint the workspace
immediately before dispatch and record whether the command
crossed the provider boundary. They route only at turn
boundaries, preserve provider affinity when it remains
eligible, and fail over after capacity or readiness
failures.

A command that did not cross the boundary is safe to
requeue after restart. A command that crossed the boundary
enters `needs-reconciliation`; subsequent turns remain
queued until an operator accepts the current worktree,
restores the pre-turn checkpoint, or stops the session.
Restore follows the session permission policy and consumes
an explicit approval when approval mode is active.

Claude uses the Python Agent SDK with a custom transport
that launches the pinned Claude Code npm package. Codex uses
the pinned app-server protocol. The stable event model
normalizes both providers while retaining raw payloads in
private blobs.

## Terminal presentation

The Textual application consumes canonical events through a
deterministic presenter. Stable typed transcript blocks
receive streaming mutations in place, preserving scroll,
selection, focus, and expanded state. A retained request ID
connects the multiline composer to command idempotency
during disconnect and reconnect.

Durable interface state is limited to presentation concerns:
draft and cursor, selected inspector tab, sidebar width and
visibility, session query, theme override, expanded block
IDs, and any unacknowledged request ID. The canonical
conversation, goal, approvals, and recovery state remain in
their existing durable records.

The layout presenter declares deterministic modes from
`60x20` through `160x48`. The system appearance drives the
default theme while an explicit light or dark override
remains durable.

## Usage safety

Every command has one provider-neutral safety envelope
before a provider process starts. Its profile fixes wall
time, submitted context, output, total tokens, tool calls,
stagnation, provider attempts, and the maximum allowed
binding-window usage. Unattended work reserves 30 percent of
each provider's binding window and fails closed when usage
is unknown. Live smoke tests reserve 50 percent.

Runtime events update the same envelope across retries.
Exact provider accounting replaces conservative estimates
without discarding earlier-attempt consumption. The generic
guard fingerprints tool operations to detect repeated
identical calls and short cycles, so rereading the same
instructions cannot reset by changing provider.

Eligible soft failures may lower effort once on the same
provider and then use one alternate provider. Neither step
creates a new envelope. Hard limits interrupt the active
process, checkpoint observable work, and pause. Unattended
xhigh requires one explicit authorization, which one command
consumes.

All harness-managed background Claude and Codex processes
hold a durable lease containing provider, profile, PID, PID
start identity, heartbeat expiry, and session. Machines
launchers use the same lease contract, and a host watchdog
terminates unleased background processes after a grace
period. Foreground terminal sessions are outside that
watchdog policy.

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

## Installed runtime

The Bazel bundle copies and validates the executable,
runfiles, Python runtime, and native dependencies into a
versioned self-contained directory. It dereferences build
cache links and rejects any path that escapes the bundle.
An atomic launcher selects the active build while retaining
the previous build for rollback.

The optional systemd user service runs that installed
bundle over the authenticated Unix socket. Service
installation and removal do not alter chat state, system
services, or lingering policy.
