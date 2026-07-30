SHELL := /bin/bash
ARGS ?=
BAZEL := npx --yes @bazel/bazelisk
INSTALL_BIN ?= $(HOME)/.local/bin
WORKSPACE ?= $(CURDIR)
CHAT_STATE_DIR ?=
CHAT_STATE_ARG = $(if $(CHAT_STATE_DIR),--state-dir "$(CHAT_STATE_DIR)",)

.PHONY: acceptance build chat coverage daemon doctor e2e install integration lint live-smoke package parity sync test

build:
	@$(BAZEL) build //...

package:
	@$(BAZEL) build //cmd:agent-harness

install:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //tools:install -- --repo "$(CURDIR)" --destination "$(INSTALL_BIN)/agent-harness"

chat:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) --cwd "$(WORKSPACE)" chat $(ARGS)

daemon:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) daemon $(ARGS)

doctor:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) doctor $(ARGS)

sync:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) sync

lint:
	@$(BAZEL) test //tests:style_test

test:
	@$(BAZEL) test //tests:acceptance_test //tests:unit_tests //tests:integration_tests //tests:e2e_tests //tests:chat_pty_test //tests:parity_test //tests:style_test //tools:coverage_gate_test //tools:install_test //tools:live_smoke_test

coverage:
	@$(BAZEL) coverage //tests:unit_tests //tests:integration_tests //tests:e2e_tests --combined_report=lcov --instrumentation_filter='//agent_harness[/:]'
	@$(BAZEL) run //tools:coverage_gate -- --lcov "$(CURDIR)/bazel-out/_coverage/_coverage_report.dat" --minimum 75 --group "safety=100:agent_harness/safety.py" --group "deterministic=98:agent_harness/blobs.py,agent_harness/config.py,agent_harness/context.py,agent_harness/errors.py,agent_harness/goals.py,agent_harness/ids.py,agent_harness/models.py,agent_harness/projections.py,agent_harness/providers/normalize.py,agent_harness/routing.py,agent_harness/transfer.py,agent_harness/workspace.py" --group "execution=90:agent_harness/scheduler.py,agent_harness/storage.py,agent_harness/worker.py" --group "portable-state=90:agent_harness/records.py,agent_harness/sync.py" --group "migration=75:agent_harness/migration.py"

integration:
	@$(BAZEL) test //tests:integration_tests

live-smoke:
	@$(BAZEL) run //tools:live_smoke -- $(ARGS)

e2e:
	@$(BAZEL) test //tests:acceptance_test //tests:e2e_tests //tests:integration_tests //tests:chat_pty_test //tests:parity_test

parity:
	@$(BAZEL) test //tests:parity_test

acceptance:
	@$(BAZEL) test //tests:local_acceptance
