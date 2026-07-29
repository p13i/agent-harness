# p13i/agent-harness

`agent-harness` is a provider-neutral, durable workspace for
Claude Code and Codex sessions. It owns a stable chat UUID,
observable transcript, approvals, goals, routing history,
and workspace checkpoints while provider-native sessions
remain replaceable execution backends.

## Entry points

```sh
make chat
make chat ARGS="resume <uuid>"
make doctor
make test
```

`make chat` starts or connects to the per-user harness
service and opens the Textual workspace. Closing the TUI
does not stop active sessions.

Repositories can include `make/chat.mk` after installing the
repository. The include invokes this Bazel workspace and
passes the calling repository as the agent work directory.

## State contract

Canonical state uses SQLite WAL plus a content-addressed
blob directory. `run-context.gpt.json`, JSONL transcripts,
and Markdown exports are generated projections; they are
not databases and are not equivalent to a provider context
window.

Same-provider resume uses the provider-native session when
available. Cross-provider resume compiles the goal,
constraints, unresolved decisions, workspace checkpoint,
evidence, compacted history, and recent events into the
target model's verified context budget. Complete observable
history remains queryable from the harness.

The harness also provides formal finite or invariant goals,
evidence, budget gates, explicit routing previews, provider
failover, encrypted fleet transfer, an authenticated Unix
socket API, SSE event recovery, and a PTY WebSocket escape
hatch.

See
[`docs/architecture.gpt.md`](docs/architecture.gpt.md) for
the state and context model,
[`docs/operations.gpt.md`](docs/operations.gpt.md) for
operator workflows, and
[`contracts/openapi.gpt.yaml`](contracts/openapi.gpt.yaml)
for the versioned control plane.

## Development

Python and dependencies are resolved through Bazel.

```sh
make build
make lint
make test
make integration
make parity
```

The repository is private and has no publication license.
