# Agent harness architecture

The harness owns the durable conversation. Claude Code and
Codex are execution providers behind adapters; neither
provider transcript is the canonical database.

## State layers

SQLite schema version 4 in WAL mode stores the session UUID,
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
history into its model context window. A requested Claude
resume whose transcript is absent from the canonical
workspace-scoped path fails before the provider starts and
emits recovery evidence; it is never recreated under the
requested UUID.

For a provider change, the harness compiles a bounded
context package from the formal goal, constraints,
unresolved decisions, current workspace checkpoint,
evidence, compacted history, and recent events. The target
adapter submits that package as a new provider turn. Full
observable history stays in the harness even when only a
bounded subset fits into the target context window.

Each context package has a durable `prepared` or `delivered`
lifecycle keyed by its checkpoint generation and payload
digest. The worker prepares it before launch and marks it
delivered only after the adapter emits
`provider.prompt.accepted`. Claude emits that acknowledgment
after its prompt stream is accepted; Codex emits it only
after `turn/start` returns a nonempty turn ID. Recovery and
provider-error events cannot acknowledge context. A crossed
provider boundary without that acknowledgment is ambiguous
and cannot silently resend the package. Incremental
Claude-to-Codex-to-Claude transfers receive distinct payload
digests and remain independently inspectable. The proof
projection binds each delivery to its command, turn,
attempt, provider, checkpoint, context, and payload. It
exposes a stable `idempotency_digest` for that binding and a
`context_delivery_digest` that also covers the explicit
lifecycle and acceptance timestamps.

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

Checkpoint archives retain regular untracked files and
relative symlinks whose lexical and resolved targets remain
inside the workspace. Unsafe paths, absolute or escaping
links, hard links, special objects, duplicate members, and
colliding archive topologies fail before extraction or a
provider boundary.

Live workspace continuity hashes the canonical executable
bit for every untracked regular file, symlink target text,
and a type marker for unsupported untracked objects. A
chmod-only or special-object provider effect therefore
changes the recovery generation and cannot be retried as an
unchanged workspace; a later checkpoint still rejects the
unsupported object before dispatch.

A command that did not cross the boundary is safe to requeue
after restart. A command that crossed the boundary enters
`needs-reconciliation`; subsequent turns remain queued until
an operator accepts the current worktree, restores the
pre-turn checkpoint, or stops the session. Restore follows
the session permission policy and consumes an explicit
approval when approval mode is active.

Accept-current and restore-pre-turn resolutions capture the
exact post-resolution live workspace material digest after
the resolution checkpoint is stable. The protected audit and
bounded proof projection retain that digest. A resolved
reconciliation cannot be replayed or used as a later
dispatch anchor when the digest is missing, malformed, or
different from current live material.

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

Focus and Control are two projections over that same
session. Focus prioritizes conversation and steering.
Control groups provider attempts under their logical command
and exposes Summary, Activity, Changes, Evidence, and
Recovery views. The projection is rebuilt from commands,
turns, attempts, dispatches, events, checkpoints, safety,
and reconciliation records. It is not a second source of
truth.

The authenticated presentation API returns allowlisted
logical turns and line-paged checkpoint diffs. It excludes
private blob identifiers and raw provider payloads. Diff
reads omit sensitive paths and binary bodies and redact
credential-shaped values before returning content.

Notifications are immutable projections of canonical events
and connection transitions. Stable keys coalesce repeat
polls and replace progressing state in place. Transient
outcomes, retained activity, and decision-required cards use
one presentation model. Notification copy does not enter the
transcript.

Session selection uses a monotonic generation. The outgoing
tree remains mounted while the incoming UI state, session,
events, turns, and reconciliation state load concurrently.
Only the latest generation may apply. A bounded
least-recently-used view cache retains the draft, cursor,
scroll, focus, mode, expanded blocks, and timeline selection
for immediate revisits.

Durable interface state is limited to presentation concerns:
draft and cursor, selected inspector tab, sidebar width and
visibility, session query, theme override, expanded block
IDs, Focus or Control mode, selected turn, Control detail
tab, active drawer, last acknowledged notification sequence,
and any unacknowledged request ID. The canonical
conversation, goal, approvals, and recovery state remain in
their existing durable records.

The layout presenter declares deterministic modes from
`60x20` through `160x48`. The system appearance drives the
default theme while an explicit light or dark override
remains durable. Visual acceptance evidence is exported from
the real Textual widget tree rather than a parallel
illustration renderer.

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
guard removes volatile provider request identifiers before
fingerprinting tool operations, so repeated identical calls
and short cycles cannot reset by changing request ID or
provider. Child-agent starts and provider-reported dollar
costs are counted by the same envelope.

Before dispatch, a formal goal's remaining time, total and
context/output token, tool-call, attempt, child-agent, and
dollar budgets clamp the one-time session envelope. Durable
envelope consumption from failed as well as completed
commands counts toward those goal limits. A command cannot
authorize metered credits with a zero or negative budget.
Explicit effort pins remain unchanged during failover, and
every recovery attempt repeats concurrency admission while
excluding only its own durable command.

Child-agent admission occurs before provider execution.
Claude uses a pre-tool hook for Agent and `spawn_agent`;
Codex uses the app-server pre-tool hook and its native
concurrent-thread setting. Both hooks atomically consume a
SQLite permit keyed by the command and provider tool-call
identity. The key survives process loss and provider
failover, and a replayed hook returns its retained decision
without consuming another permit. The provider-neutral event
stream records stable child call identities, native thread
identities, parent command/turn/attempt joins, terminal
state, and numeric usage.

Dispatch repetition records instruction, compiled context,
workspace-instruction, plan, and skill digests. Managed
turns bind those digests to their durable `turn_ref`, so an
exact replay of one logical step pauses while a distinct
declared step may reuse governing artifacts. Unmanaged
interactive turns pause on an exact repeated instruction. A
provider file-change notification clears within-turn tool
repetition history only after an independently recomputed
workspace digest changed.

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
watchdog policy. Restart recovery verifies the recorded
leader and its entire process group before releasing a
lease. A dead leader with surviving group members or an
unverifiable identity moves the lease to the durable
`recovery-blocked` state. Worker supervision excludes that
session until the exact lease is explicitly released, and
reports the block through health state without repeatedly
spawning a worker.

## Goals

A session may carry a finite or invariant formal goal with
constraints, completion predicates, evidence, and token,
turn, command, time, or cost budgets. Budget exhaustion
stops further provider turns and moves the session to
operator attention. Completion remains evidence-backed
rather than inferred from a provider's final sentence.

Multi-tier proofs promote a completed finite goal on the
same session. Promotion requires an idempotency key,
explicit retained authorization, an immutable ordered
predicate prefix, a later milestone, all prior constraints,
and budgets that never decrease. The authorization binds the
session, source goal, stage, exact additive budget map, next
contract digest, and a retained source receipt. The prior
goal and evidence remain immutable. One atomic promotion
receipt links both contract digests, copies prior evidence
with digest-checked lineage, reactivates the same session
UUID, and preserves cumulative safety consumption and
provider-native resume state.

Existing exact-UUID sessions can adopt a complete
`p13i/machines` contract only at a quiescent boundary. A
typed retained authorization binds the previous goal digest,
normalized external reference, and complete next goal
envelope. Adoption preserves event, attempt, checkpoint,
native-resume, and session history; clears one-time safety
extensions and xhigh authorization; and retains its receipt
through export and proof capture.

Formal goals carry milestones, provider and effort
allowlists, maximum concurrency, completion policy, incident
policy, and finite budgets for time, turns, total and
context/output tokens, tools, attempts, child agents, and
dollars. Non-finite budget values are rejected. Finite goals
use evidence-all completion; invariant goals use never; both
use recover-then-pause incident handling. Machines evidence
mutations require stable idempotency keys and atomically
retain one evidence row, event, and mutation receipt. Keyed
synchronous API mutations run their state change and receipt
inside one immediate SQLite transaction. Nested store writes
use savepoints, so a failed mutation rolls back both its
state and receipt while a concurrent retry observes one
retained response.

## Fleet transfer

Transfers encrypt a canonical session bundle to a
destination X25519 public key, derive the payload key with
HKDF, encrypt with AES-GCM, and sign the envelope with
Ed25519. The destination imports into a new owner epoch.
Finalization prevents the source from continuing as an
active owner.

Independent proof verifiers use the bounded
`/v1/sessions/{session_id}/proof` projection. It carries
canonical command, attempt, child, event-range, routing,
usage, approval, reconciliation, lease, checkpoint, and
context-delivery identities and digests. Command, turn,
attempt, provider, child, checkpoint, and usage-sample joins
remain explicit. Each route binds the selected usage sample
to its observation time, attempt admission time, and route
recording time. Admission freshness and admissibility are
therefore evaluated at the historical boundary rather than
at final proof capture. Prompt text, raw provider arguments,
credentials, and host paths are never projected. The proof
snapshot refuses a noncontiguous event sequence; each
retained page is derived from the exact sequence
`1..through_sequence`.

Reconciliation proof exposes the pre-dispatch, discovery,
and resolution checkpoint identities and whether each binds
to a projected checkpoint row. An `accept-current`
resolution is recovery-material certified only when the
discovery and resolution rows have identical base commit,
patch, untracked, and context digests. A missing or tampered
checkpoint identity leaves certification false.

The first request materializes one sanitized immutable
snapshot. Later pages provide its `snapshot_id` and original
`through_sequence`, so concurrent events cannot change any
join or digest. A page holds at most 1,000 events and a
snapshot at most 50,000; excess fails with HTTP 413 before
materialization. Snapshots remain addressable for at least
336 hours, with a 128-snapshot session quota that fails
closed instead of evicting a still-bound proof. Any bounded
record-set truncation reports `complete: false` and names
the record class.

Managed transition policies are content-addressed and
retained once per session, goal, and epoch. The first
authorization carries and fully validates at most 1,000
ordered stages. Follow-up authorizations carry only the
exact retained policy reference. An indexed transition
ledger provides constant-index sequence and predecessor
checks and records authorization, reservation, release, and
consumption without scanning prior receipts or events. The
proof projection validates the ledger against its canonical
policy and compact authorization receipts and keeps all
1,000 invariant stages outside generic record truncation.

## Installed runtime

The Bazel bundle copies and validates the executable,
runfiles, Python runtime, and native dependencies into a
versioned self-contained directory. It dereferences build
cache links and rejects any path that escapes the bundle. An
atomic launcher selects the active build while retaining the
previous build for rollback.

The optional systemd user service runs that installed bundle
over the authenticated Unix socket. Service installation and
removal do not alter chat state, system services, or
lingering policy.
