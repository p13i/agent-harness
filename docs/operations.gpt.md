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

Health responses carry a control protocol version and a
build fingerprint derived from the harness Python package.
Before opening chat, a new CLI replaces an older managed
daemon when either value differs. This prevents a rebuilt
client from silently using stale in-memory service code.

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
envelope, exact or estimated accounting, provider headroom,
recovery stage, and guard reason. `/budget` shows the same
state. These commands extend one future envelope:

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
token and Unix socket from the private state directory.
HTTP writes return durable command receipts. SSE accepts
`Last-Event-ID` for lossless reconnection. Every response
has an `X-Correlation-ID`; errors expose the same ID without
including prompts, arguments, credentials, or host paths.

TCP listening is off by default. It must be requested with
an explicit resolvable host and port. Wildcard listeners are
rejected. The fleet integration normally keeps the harness
on its Unix socket and exposes only a narrow authenticated
machines bridge.

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
```

These targets use scripted adapters and do not invoke live
providers. A separate smoke validates one initial and one
native-resumed turn, with low effort, read-only permission,
one attempt per turn, and a 50 percent binding ceiling:

```sh
make live-smoke ARGS="codex --confirm-spend"
make live-smoke ARGS="claude --confirm-spend"
```

Without `--confirm-spend`, the smoke exits before creating a
session or invoking a provider.
