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

Focus mode is the default conversation workspace. Press
`F3` or run `/mode control` to open the turn timeline;
repeat the action or run `/mode focus` to return. Control
mode groups retries and failover attempts under one turn.
Its detail tabs separate Summary, Activity, Changes,
Evidence, and Recovery. Selecting a checkpointed turn loads
its bounded, redacted diff without restoring historical
workspace state.

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
the bounded local view cache while canonical state refreshes.

The composer accepts multiple lines. `Enter` sends,
`Shift+Enter` and `Ctrl+J` insert a newline, and pasted text
never submits by itself. During a connection loss the
harness retains the draft, cursor, and request identifier;
reconnection queries or repeats that same idempotent
request rather than creating a second turn.

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
guard reason. `/budget` shows the same state. These commands
extend one future envelope:

```text
/budget extend 300 10000 Finish bounded validation
/budget xhigh One bounded architecture pass
```

The dashboard exposes the same allowlisted usage state and
bounded extension control. Every extension requires a
reason. Retrying an extension or lease mutation with the
same idempotency key replays the original response; changing
the request under that key conflicts.

## API

The versioned contract is
`contracts/openapi.gpt.yaml`. Local clients read the bearer
token and Unix socket from the local `.runtime` directory.
HTTP writes return durable command receipts. SSE accepts
`Last-Event-ID` for lossless reconnection. Every response
has an `X-Correlation-ID`; errors expose the same ID without
including prompts, arguments, credentials, or host paths.

TCP listening is off by default. It must be requested with
an explicit resolvable host and port. Wildcard listeners are
rejected. The fleet integration normally keeps the harness
on its Unix socket and exposes only a narrow authenticated
machines bridge.

External orchestrators can bind one stable job to one
session and submit retry-safe turns through the Python SDK:

```python
session = client.ensure_session(
    external_ref={
        "orchestrator": "p13i/machines",
        "job_id": "build-42",
    },
    workspace="/absolute/workspace",
    idempotency_key="create-build-42",
)
receipt = client.submit_managed_turn(
    session["id"],
    "Execute the bounded build step.",
    turn_ref={
        "step_id": "compile",
        "agent_role": "implementer",
    },
    idempotency_key="build-42-compile",
)
result = client.wait_command(
    receipt["command_id"],
    timeout_seconds=300,
)
```

Managed methods require caller-supplied idempotency keys.
Interactive conveniences may generate keys automatically.

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

Hard safety limits do not fail over. Recoverable soft
violations can lower effort once and switch provider once,
while retaining cumulative use in the original command
envelope. A session requiring more capacity stays paused
until an operator explicitly extends time, tokens, or one
xhigh authorization.

Machines-managed background processes must reserve a lease
before launch. Lease admission requires a fresh provider
usage sample below the profile ceiling with metered credits
off. The process then attaches its PID start identity,
heartbeats while active, and releases at exit. The host
watchdog allows a short registration grace period and then
terminates unleased, expired, or PID-reused background
Claude and Codex processes. It does not target foreground
terminal sessions.

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
composer regions, and missing keyboard focus without
network or provider access.

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
