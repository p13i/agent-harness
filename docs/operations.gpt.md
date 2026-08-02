# Agent harness operations

## Entry points

Run the Textual workspace from this repository:

```sh
make chat
make chat ARGS="resume <session-uuid>"
```

When another repository includes `make/chat.mk`, its
`make chat` target runs this Bazel workspace while keeping
the caller as the agent work directory.

The durable service starts on demand. Closing the TUI leaves
workers running. The following commands provide non-TUI
control:

```sh
make install
agent-harness service install
agent-harness service start
agent-harness service status
agent-harness service restart
agent-harness service stop
agent-harness service uninstall
```

The installer creates a versioned, self-contained bundle
under `~/.local/lib/p13i-agent-harness/` and atomically
updates the launcher under `~/.local/bin/`. Service removal
preserves all chat state. `service install` reports when a
systemd user manager is unavailable or lingering is
inactive; it does not modify either system setting.

Before promoting or restarting an installed release, query
the authenticated Unix-socket health endpoint:

```sh
curl --unix-socket ~/my/chats/.runtime/control.sock \
  -H "Authorization: Bearer $(cat ~/my/chats/.runtime/token)" \
  http://localhost/healthz
```

`runtime_build_id` is the selected verified-bundle build
identifier. Release automation passes the source commit as
the bundle build identifier and requires an exact match.
`quiescence.restart_safe` is true only while both
`quiescence.active_commands` and `quiescence.active_proofs`
are zero. Restart automation refuses the restart when it is
false. `control_build_id` remains the digest of the running
Python control sources. The prompt-free
`active_command_details`, `active_unattended_commands`, and
`active_proof_sessions` fields identify each durable restart
blocker. Comparing the desired release commit to
`runtime_build_id` distinguishes same-release recovery from
a cross-release mutation. `agent-harness quiescence` exposes
the same contract and falls back to durable command state
when the daemon is inactive. If systemd reports an active
service but health is unreachable, `proof_state_known` and
`restart_safe` are both false.

The session sidebar starts in a focused view that retains
the current session, attention-required sessions, and five
recent idle sessions. `/sessions all` exposes complete
unarchived history, and `/sessions focused` restores the
bounded view. Drag the divider beside the session list or
press `Ctrl+Shift+Left` and `Ctrl+Shift+Right` to resize it.
`/sidebar reset` restores the default width. The width is
restored with the session and inherited by new chats in the
same workspace.

Use the sidebar search field to match session UUID, name,
provider, or exact external job reference. `/rename`,
`/archive`, and `/unarchive` update lifecycle state without
deleting history. Archived sessions appear in their own
group when the complete view is active.

The transcript reconciles streamed deltas with the provider
final message. Provider protocol lifecycle events remain
hidden unless `/events on` activates the diagnostic view.
`/theme system` tracks the host appearance; `/theme light`
and `/theme dark` select an explicit appearance.

Focus mode is the default conversation workspace. Press `F3`
or run `/mode control` to open the turn timeline; repeat the
action or run `/mode focus` to return. Control mode groups
retries and failover attempts under one turn. Its detail
tabs separate Summary, Activity, Changes, Evidence, and
Recovery. Selecting a checkpointed turn loads its bounded,
redacted diff without restoring historical workspace state.

`F2` opens the searchable action palette. `F1` opens the
same palette, and slash completion uses the same command
registry. Ordinary status messages appear in one compact
notification surface. Decision-required states provide
buttons for their existing actions and remain visible until
resolved or deferred; notification text is not inserted into
the conversation.

Thread changes are atomic. The current transcript, header,
and composer remain stable until the selected session is
ready. Rapid selection ignores late responses from obsolete
requests. Returning to a recently viewed session restores
its draft, selection, focus, scroll position, and mode from
the bounded local view cache while canonical state
refreshes.

The composer accepts multiple lines. `Enter` sends,
`Shift+Enter` and `Ctrl+J` insert a newline, and pasted text
never submits by itself. During a connection loss the
harness retains the draft, cursor, and request identifier;
reconnection queries or repeats that same idempotent request
rather than creating a second turn.

The inspector separates Context, Goal, Usage, Approvals,
Recovery, and Storage. The Recovery tab shows ambiguous
commands and their checkpoint, worktree, usage, and
resolution state. The Storage tab shows identifiers, paths,
and synchronization state without credentials.

Health responses carry a control protocol version and a
build fingerprint derived from the harness Python package.
Before opening chat, a new CLI replaces an older managed
daemon when either value differs. This prevents a rebuilt
client from silently using stale in-memory service code.

`make doctor` checks the selected bundle, systemd user
service, daemon compatibility, SQLite integrity, Git and
`npx`, disk headroom, synchronization lag, and the pinned
provider launch commands. It fails closed when the state
directory, runtime directory, or live Unix socket has
non-private permissions, and when worker heartbeats or
active process leases are stale. The diagnostic reports the
affected records without repairing or deleting them.

## Chat data

The default data root is `~/my/chats`, normally a symlink to
the private `github.com/p13i/chats` submodule. Override it
for an isolated invocation with the global
`--state-dir <path>` option, or set `CHAT_STATE_DIR` for the
Make targets in this repository.

Each `sessions/<uuid>/` directory contains a deterministic
machine record, JSONL event history, and readable Markdown
transcript. Referenced blobs and retained exports are
tracked beside those records. The `.runtime/` directory
contains the SQLite database, worktrees, API token, socket,
logs, locks, and sync status; it is private local state and
is never committed.

The daemon materializes and pushes portable records after a
completed turn, every 30 seconds, and during clean shutdown.
Synchronization is serialized, retries three times, never
force-pushes, and leaves a visible pending or conflict state
instead of discarding local work. The TUI inspector and
these commands expose the same state:

```sh
npx --yes @bazel/bazelisk run //cmd:agent-harness -- \
  sync-status
npx --yes @bazel/bazelisk run //cmd:agent-harness -- sync
```

Move a legacy installation only while using the new binary:

```sh
npx --yes @bazel/bazelisk run //cmd:agent-harness -- \
  migrate-state \
  --from ~/.local/state/p13i-agent-harness \
  --to ~/my/chats \
  --trash-source
```

Migration stops only harness-managed processes using the
source or destination root. Existing destination sessions
are retained through a collision-checked portable merge and
rollback snapshot. The command verifies session, event,
blob, portable-record, and Git-worktree fidelity, pushes the
combined destination, and only then moves the old root to
Trash.

```sh
make doctor
npx --yes @bazel/bazelisk run //cmd:agent-harness -- status
npx --yes @bazel/bazelisk run //cmd:agent-harness -- \
  events <session-uuid>
npx --yes @bazel/bazelisk run //cmd:agent-harness -- \
  checkpoint <session-uuid>
```

## Session lifecycle

Create a session with a formal goal:

```sh
npx --yes @bazel/bazelisk run //cmd:agent-harness -- \
  --cwd /absolute/workspace new \
  --goal "Complete the scoped implementation." \
  --goal-kind finite \
  --predicate '{"type":"command","subject":"make test"}' \
  --execution-profile unattended
```

Unattended xhigh requires one explicit authorization before
the command:

```sh
npx --yes @bazel/bazelisk run //cmd:agent-harness -- \
  extend-budget <session-uuid> \
  --allow-xhigh-once \
  --reason "One bounded architecture pass"
```

Then submit the explicit route:

```sh
npx --yes @bazel/bazelisk run //cmd:agent-harness -- \
  send <session-uuid> "Continue the implementation." \
  --provider codex --effort xhigh --wait
```

An explicit provider, model, or effort never silently falls
back. Automatic routing uses readiness, capability
requirements, subscription headroom, metered-credit policy,
role bias, provider affinity, queue share, quality, and
context-transfer cost.

Inside chat, `/usage` displays the current profile,
envelope, provider-reported input and output, estimated
harness context, provider headroom, recovery stage, and
guard reason. `/budget` shows the same state. The extension
command extends one future envelope. The xhigh command
authorizes one exact queued command and provider:

```text
/budget extend 300 10000 Finish bounded validation
/budget xhigh 550e8400-e29b-41d4-a716-446655440000 codex One bounded architecture pass
```

On Linux and WSL, Claude subscription probing first reads
`CLAUDE_CODE_OAUTH_TOKEN` when present, then the current
user's mode-restricted `~/.claude/.credentials.json`.
Symlinks, files larger than 1 MiB, and group- or
world-readable credential files are rejected. The provider
child receives only the exact Claude OAuth variable from
credential-shaped environment values. macOS subscription
probing continues to use the Claude Code Keychain entry.
Tokens never appear in usage, proof, health, or diagnostic
projections.

The dashboard exposes the same allowlisted usage state and
bounded extension control. Every extension requires a
reason. Retrying an extension or lease mutation with the
same idempotency key replays the original response; changing
the request under that key conflicts. The mutation and
receipt commit together, so response loss and concurrent
same-key calls cannot apply additive quota or create a
second lease or checkpoint.

## API

The versioned contract is `contracts/openapi.gpt.yaml`.
Local clients read the bearer token and Unix socket from the
local `.runtime` directory. HTTP writes return durable
command receipts. SSE accepts `Last-Event-ID` for lossless
reconnection. Every response has an `X-Correlation-ID`;
errors expose the same ID without including prompts,
arguments, credentials, or host paths.

`GET /v1/sessions/{session_id}/proof` is the independent
verifier surface. The first response returns a durable
`snapshot_id` and fixed `through_sequence`. Pass both on
every later request while advancing `after_sequence`. The
page limit is 1,000 events and the immutable snapshot limit
is 50,000 events; a larger session returns HTTP 413 without
persisting a partial proof. Retained snapshots remain
readable for at least 336 hours and the bounded session
quota fails closed rather than evicting a proof inside that
window.

Accept a final capture only when its top-level `complete`
and `event_range.complete` fields are true and `truncated`
is empty. Verify `event_range.digest` for each page,
`event_range.snapshot_digest` for the assembled ordered
event array, and `snapshot_digest` for the reconstructed
retained payload. Digests use SHA-256 over UTF-8 Python
`json.dumps` output with `sort_keys=True` and separators
`(",", ":")`. To reconstruct the retained payload, combine
all event pages; remove response-only `snapshot_id`,
`snapshot_digest`, and `event_range`; remove `events` from
`truncated`; and derive `complete` from the remaining empty
or nonempty `truncated` list. The payload declares
`sha256-python-canonical-json-v1` as this algorithm. The
server rejects missing, duplicated, zero, or reordered event
sequence positions rather than certifying a partial history.

For restart recovery, require every projected reconciliation
checkpoint identity to report its matching checkpoint row.
An `accept-current` record is eligible only when
`recovery_material_certified` is true; this derives from
equal discovery and resolution base, patch, untracked, and
context digests. The pre-dispatch checkpoint remains a
separate identity because ambiguous provider work may have
changed material before discovery.

For both `accept-current` and `restore-pre-turn`, require
`resolution_workspace_digest_valid` and
`resolution_material_certified` to be true. Bind the next
transition receipt's `prior_material_digest` to the exact
`resolution_workspace_digest`. The service captures this
digest only after the resolution checkpoint is stable and
rejects an idempotent resolution replay when the protected
field is absent or malformed.

The projection returns digests instead of prompt,
event-text, provider-argument, decision, or usage-source
bodies. Child lifecycle, routing decisions, historical
routing usage samples, and context deliveries retain typed
identity joins. Context rows expose command, turn, attempt,
provider, checkpoint, state, a stable idempotency digest,
and a versioned lifecycle-bound context-delivery digest.
Command rows with `command_envelope_version` 2 bind the
current request and immutable execution profile. An absent
version denotes the legacy version 1 shape, whose retained
normalization is exposed as `legacy_command_envelope_digest`
for historical verification. Delivery version 2 adds
`transport` outside the stable idempotency binding, so
existing idempotency digests retain their definition while
new lifecycle digests explicitly bind `context-package`,
`native-resume`, or a legacy migration state. Routing rows
include `usage_sample_bound`, `route_recorded_at`,
`attempt_admitted_at`, usage age and freshness at both
boundaries, and `admissible_at_route`. Long-tier verifiers
use these immutable historical values;
`usage[].fresh_at_capture` describes capture age only and
does not invalidate a route that was fresh when admitted.
`GET /v1/providers` exposes current sample identity,
observation time, freshness, admission state, binding
percentage, credits state, error state, and provider
capabilities. A caller may perform a provider-free admission
check with `POST /v1/sessions/{session_id}/route` before
submitting a managed turn.

TCP listening is off by default. It must be requested with
an explicit resolvable host and port. Wildcard listeners are
rejected. The fleet integration normally keeps the harness
on its Unix socket and exposes only a narrow authenticated
machines bridge.

External orchestrators can bind one stable job to one
session and submit retry-safe turns through the Python SDK:

```python
from pathlib import Path

session = await client.ensure_session(
    Path("/absolute/workspace"),
    orchestrator="p13i/machines/cs-builder",
    job_id="build-42",
    name="Bounded build 42",
    goal="Produce and certify build 42.",
    goal_kind="finite",
    constraints=("Preserve the original command envelope.",),
    predicates=(build_predicate,),
    milestones=(build_milestone,),
    budgets=bounded_goal_budgets,
    permitted_providers=("claude", "codex"),
    permitted_efforts=("low", "medium"),
    max_concurrency=1,
    completion_policy="evidence-all",
    idempotency_key="create-build-42",
)
receipt = await client.submit_managed_turn(
    session["session_id"],
    "Execute the bounded build step.",
    step_id="compile",
    agent_role="implementer",
    permission_mode="read-only",
    idempotency_key="build-42-compile",
)
result = await client.wait_command(
    receipt.command_id,
    timeout=300,
)
```

Managed methods require caller-supplied idempotency keys.
Interactive conveniences may generate keys automatically.

After a finite tier is independently certified, promote its
completed goal without replacing the session or native
provider resume chain:

```python
promotion = await client.promote_goal(
    session["session_id"],
    from_goal_id=completed_goal_id,
    stage="tier-24h",
    objective="Certify the 24-hour tier.",
    constraints=prior_constraints,
    predicates=prior_predicates + [tier_24h_predicate],
    milestones=prior_milestones + [tier_24h_milestone],
    budgets=tier_24h_budgets,
    permitted_providers=prior_permitted_providers,
    permitted_efforts=prior_permitted_efforts,
    max_concurrency=prior_max_concurrency,
    authorization=promotion_authorization,
    idempotency_key="promote:" + run_id + ":24h",
)
```

Promotion is accepted only from the current completed goal
at a command-quiescent boundary without pending approvals or
reconciliation. The next contract must retain every
constraint, add or increase budget, and name a different
later predicate. `goal_history` and `goal_promotions` in the
proof snapshot retain the prior and next contract digests,
authorization digest, request digest, evidence lineage, and
transition event. `promotion_authorization` uses schema
`p13i/agent-harness/goal-promotion-authorization/v1` and
binds the session ID, source goal ID, stage, exact additive
budget map, next goal contract digest, retained receipt, and
receipt SHA-256.

Adopt a preexisting exact-UUID session with
`adopt_session_contract`. Its complete flat creation
envelope must match the existing workspace and direct or
worktree mode. The external reference must start with
`p13i/machines`. Typed authorization uses schema
`p13i/agent-harness/session-contract-adoption-authorization/v1`
and binds the session ID, external reference, previous goal
digest, next goal-envelope digest, retained receipt, and
receipt SHA-256. Adoption requires command, process-lease,
and attention quiescence.

Machines evidence calls always pass a stable idempotency
key:

```python
evidence = await client.add_evidence(
    session["session_id"],
    evidence_type="machines-proof",
    subject="build-42",
    outcome="passed",
    value={"report_digest": report_digest},
    idempotency_key="evidence:build-42:report",
)
```

Each message may carry `safety_limits` with only tighter
numeric limits than the effective profile, goal, and session
envelope. This generic monotone request can cap an
unattended stage at, for example, `max_attempts: 2` and
`max_seconds: 300`; any widening or unknown, non-finite, or
spend-granting value fails before provider admission.
Non-finite provider usage and reported cost are not treated
as headroom or exact accounting. The normalized command
digest binds the request, while the immutable command
envelope and proof safety projection retain the requested
limits, request digest, effective limits, and consumption.

If an independently authorized operator action permits an
otherwise identical interactive dispatch after no material
checkpoint change, call `invalidate_dispatch_generation` at
a quiescent boundary. The authorization schema is
`p13i/agent-harness/dispatch-invalidation-authorization/v1`
and binds the session ID, exact reason, retained receipt,
and receipt SHA-256. Generic invalidation is rejected for a
managed execution profile. The prior fingerprints remain in
proof.

Managed builder stages and recurring `cs-sre` ticks use an
ordered transition instead. The goal must predeclare a
`p13i/agent-harness/dispatch-generation-transition-policy/v1`
object through the constraint
`dispatch-generation-transition-policy-sha256:<digest>` and
the separate constraint
`dispatch-generation-transition-epoch:<epoch-id>`. The
policy binds the same `epoch_id`. Its finite `transitions`
list binds each per-epoch monotonic sequence to one exact
next `turn_ref` and one exact normalized command envelope
digest. The envelope normalization is canonical JSON over
`command_type`, the stored command payload, and the
immutable command-envelope execution profile. The payload
includes the instruction, workload, required capabilities,
route pins, effort, permission mode, metered budget, turn
reference, and any proof probes when supplied; only HTTP
idempotency transport headers are outside it.

Database v5 does not rewrite historical command payloads.
Effort spelling is normalized for new request idempotency,
while retained transition receipts continue to bind the
stored payload bytes used when they were authorized.

The transition authorization schema is
`p13i/agent-harness/dispatch-generation-transition-authorization/v1`.
It binds the session and external reference, current goal,
latest command identity and type, typed transition anchor,
latest certified checkpoint, prior generation and live
workspace-material digests, exact next turn reference and
command digest, monotonic per-goal epoch sequence, epoch
identifier, policy digest, retained receipt, and receipt
digest. Sequence one carries the complete policy. The
harness validates every predeclared stage, retains that
policy once by session, goal, epoch, and digest, and
replaces it in durable authorization storage with an exact
`policy_ref`. Sequences two through 1,000 must carry only
that reference; an unknown, changed, cross-session,
cross-goal, or stale-epoch reference fails closed. The
response `authorization_digest` remains the digest of the
wire authorization, while `request_digest` binds the full
invalidation payload. A `provider-result` anchor requires a
completed message whose result checkpoint and material are
current. A `control-command` anchor permits only a completed
`interrupt`, `pause`, `resume`, `stop`, or `steer` command.
After the first transition, an interrupt anchor must bind
the exact predecessor-consumed message. That message must
end cancelled or failed at one terminal provider boundary;
the interrupt event and control result must both bind its
command identifier and the current certified checkpoint and
material. Only completed control commands may occur between
that target and the anchor. A `resolved-reconciliation`
anchor requires a failed `E_NEEDS_RECONCILIATION` or
`E_SAFETY_GUARD` message,
exactly one resolved reconciliation, an `accept-current` or
`restore-pre-turn` decision, and its latest resolution
checkpoint. Its protected `resolution_workspace_digest` must
equal the current live material before the anchor is
eligible. The proof snapshot exposes the current prompt-free
`transition_anchor`, including its eligibility, goal, epoch,
checkpoint, material, generation, and reconciliation
bindings. Storage independently revalidates those bindings
and quiescence atomically. A workspace change after
authorization is rejected before reservation. Storage
re-inspects live material inside atomic route admission, so
a write after reservation is also rejected before the
provider boundary. The matching next command reserves the
transition before repetition checks and consumes it only in
the same transaction that crosses provider admission. A
terminal pre-boundary routing failure releases the
reservation for the same exact envelope; a post-boundary
failure requires reconciliation.

The proof snapshot partitions transition authorizations and
invalidations into `dispatch_transition_ledger`. Its
indexed, bounded 1,200-row projection retains the canonical
policy once and all exact request, authorization,
prior-anchor, stage, reservation, release, and consumption
bindings for a 1,000-turn invariant goal. The ledger
validates policy identity, every policy stage, contiguous
sequence, compact authorization and receipt digests, and the
exact command that consumed each reservation. This keeps a
stable `cs-sre` session UUID and invariant goal certifiable
through the 72-hour tier with linear storage. A later
explicit contract adoption retains the original
session-creation digest and idempotency receipt while
recording separate goal-lineage and adoption digests.

Every transition-ledger receipt includes its sanitized
authorization proof: the exact safe binding, binding digest,
source-receipt digest and retained receipt digest, operation
identity, and both internal digest-validity results. It also
exposes a `safe_request_binding` and its digest. That
binding replaces the free-text reason with its digest and
replaces the nested authorization with its authorization
digest while retaining every prior-anchor, checkpoint,
material, stage, sequence, and next-command field.
Independent verification recomputes both safe binding
digests, reconstructs the exact source receipt from the
binding, and matches it to the retained policy and ledger
row; it must not trust `valid: true` alone. The wire
authorization and full request digests remain visible but
their free-text reason material is not projected into the
prompt-free proof.

Read logical turns and a safe checkpoint diff through the
same typed client:

```python
page = await client.turns(
    session_id,
    after_sequence=0,
    limit=50,
)
turn = await client.turn(
    session_id,
    page["turns"][0]["turn_id"],
)
diff = await client.checkpoint_diff(
    session_id,
    turn["turn"]["checkpoint_id"],
    start_line=0,
    limit=400,
)
```

The corresponding authenticated HTTP endpoints are:

```text
GET /v1/sessions/{session_id}/turns
GET /v1/sessions/{session_id}/turns/{turn_id}
GET /v1/sessions/{session_id}/checkpoints/{checkpoint_id}/diff
```

## Recovery

The stable resume identifier is the harness session UUID,
not a Claude or Codex native ID. On service restart, running
sessions are rediscovered and their workers resume ordered
commands from SQLite. The export projections can be
regenerated at any time.

If a provider mutation has an ambiguous result, the session
enters reconciliation instead of replaying the mutation on
another provider. Read-only or clearly unstarted work can
fail over automatically.

Inspect and resolve an ambiguous turn:

```sh
agent-harness reconcile list <session-uuid>
agent-harness reconcile inspect <reconciliation-id>
agent-harness reconcile resolve <reconciliation-id> \
  accept-current \
  --workspace-digest <observed-digest> \
  --idempotency-key <stable-key>
```

The other decisions are `restore-pre-turn` and `stop`.
Approval-mode restores first return an approval ID. Resolve
that approval, then retry with the same observed digest and
a new reconciliation idempotency key carrying the approval
ID.

Hard safety limits and repeated digest guards do not fail
over. Before `provider.prompt.accepted`, stagnation may lower
an unpinned effort once and then change provider once while
retaining the original command envelope and cumulative
consumption. Other guard stops return `E_SAFETY_GUARD`
without another provider attempt. After acknowledgment, a
guard stop retains `E_SAFETY_GUARD`,
adds a `reconciliation_id` while the provider outcome is
ambiguous, records the specific guard reason in the incident
and reconciliation-requested event, and blocks another
message until the ambiguous turn is resolved. A stop
detected after a known terminal provider result records a
post-turn checkpoint and returns `E_SAFETY_GUARD` without a
retry. A session requiring more capacity stays paused until
an operator explicitly extends time, tokens, or one xhigh
authorization.

The current goal's remaining time, total and context/output
token, tool-call, attempt, child-agent, and dollar budgets
clamp that envelope after a one-time extension is consumed.
Consumption from failed commands remains charged to the
goal. A zero or negative metered budget is rejected before
provider work. Unattended implementation and operations
turns allow at most two child-agent starts; live-smoke turns
allow none. Child permits are retained by command and stable
provider tool-call identity, so provider failover cannot
reset the count and a duplicate pre-tool hook cannot consume
the permit twice. Provider request IDs are excluded from
repetition fingerprints so a repeated tool call cannot evade
the guard with a new transport ID.

Machines-managed background processes must reserve a lease
before launch. Lease admission requires a fresh provider
usage sample below the profile ceiling with metered credits
off. The process then attaches its PID start identity,
heartbeats while active, and releases at exit. The host
watchdog allows a short registration grace period and then
terminates unleased, expired, or PID-reused background
Claude and Codex processes. It does not target foreground
terminal sessions.

If restart recovery cannot verify the recorded process
identity, or the leader exited while children remain in its
process group, the lease becomes `recovery-blocked`. Health
reports the exact session and lease and worker supervision
does not retry it on a timer. Inspect the retained process
and lease evidence, then use the typed lease update with
`action=release` for that exact lease. Release records an
unblock event and permits one supervised worker restart; it
does not resume a paused session or invoke a provider.

## Validation

```sh
make lint
make test
make build
make doctor
make ui-gallery
```

These targets use scripted adapters and do not invoke live
providers. The gallery renders every declared interface
fixture in Focus and Control modes, both themes, and all
four breakpoints: 176 screenshots exported from the actual
Textual widget tree. It rejects secret-bearing content,
outbound resources, clipping, overlapping notification and
composer regions, and missing keyboard focus without network
or provider access.

On WSL, the opt-in service journey installs and exercises an
isolated user unit:

```sh
make wsl-e2e ARGS=--confirm
```

On other platforms or without `--confirm`, it reports the
journey as pending rather than passing.

A separate smoke validates one Claude and one Codex turn,
with low effort, read-only permission, one attempt per turn,
and a 50 percent binding ceiling:

```sh
make live-smoke ARGS="--confirm-spend"
```

Without `--confirm-spend`, the smoke exits before creating a
session or invoking a provider.
