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
  --predicate '{"type":"command","subject":"make test"}'
```

Submit a non-interactive turn with an explicit route when
needed:

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

## Validation

```sh
make lint
make test
make build
make doctor
```
