# Agent instructions

## Scope

This repository owns a provider-neutral agent chat harness.
Provider-specific behavior belongs behind adapters. The
canonical session, goal, event, approval, checkpoint, and
routing models must not depend on Claude or Codex wire
types.

## Python file names

Importable Python packages, modules, and tests use normal
Python file names. This repository does not apply the
`.gpt` file-labeling convention. Generated fixtures and
exports may use a `.gpt` label when the label communicates
their provenance.

## Code style

- Do not use ternary expressions.
- Do not write single-line conditional returns.
- Pass provider commands as argument vectors without a
  shell.
- Keep credentials, prompts, and raw provider arguments out
  of status, audit, and diagnostic surfaces.
- Use inclusive, neutral terminology.

## Provider execution

- Invoke Codex through
  `npx -y @openai/codex@0.146.0`.
- Invoke Claude Code through
  `npx @anthropic-ai/claude-code@2.1.220`.
- Provider app servers and structured transports are
  exempt from global permission-bypass flags. The harness
  maps its explicit permission mode onto each provider.
- Add a bypass flag only for an explicit `full` permission
  session.

## Changes

- Keep the versioned HTTP, SSE, and Python client contracts
  backward compatible within a major version.
- Add tests for every behavior change.
- Run `make lint` and `make test` before committing.
- Work directly on `main`. Do not add agent attribution to
  commits.

