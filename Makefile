SHELL := /bin/bash
ARGS ?=
BAZEL := npx --yes @bazel/bazelisk
INSTALL_BIN ?= $(HOME)/.local/bin
WORKSPACE ?= $(CURDIR)

.PHONY: build chat coverage daemon doctor e2e install integration lint package parity test

build:
	@$(BAZEL) build //...

package:
	@$(BAZEL) build //cmd:agent-harness

install:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //tools:install -- --repo "$(CURDIR)" --destination "$(INSTALL_BIN)/agent-harness"

chat:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- --cwd "$(WORKSPACE)" chat $(ARGS)

daemon:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- daemon $(ARGS)

doctor:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- doctor $(ARGS)

lint:
	@$(BAZEL) test //tests:style_test

test:
	@$(BAZEL) test //tests:unit_tests //tests:integration_tests //tests:e2e_tests //tests:chat_pty_test //tests:parity_test //tests:style_test //tools:coverage_gate_test //tools:install_test

coverage:
	@$(BAZEL) coverage //tests:unit_tests //tests:integration_tests //tests:e2e_tests --combined_report=lcov --instrumentation_filter='//agent_harness[/:]'
	@$(BAZEL) run //tools:coverage_gate -- --lcov "$(CURDIR)/bazel-out/_coverage/_coverage_report.dat" --minimum 60 --group "deterministic=95:agent_harness/blobs.py,agent_harness/config.py,agent_harness/context.py,agent_harness/errors.py,agent_harness/goals.py,agent_harness/ids.py,agent_harness/models.py,agent_harness/projections.py,agent_harness/routing.py,agent_harness/transfer.py,agent_harness/workspace.py" --group "execution=75:agent_harness/scheduler.py,agent_harness/storage.py,agent_harness/worker.py"

integration:
	@$(BAZEL) test //tests:integration_tests

e2e:
	@$(BAZEL) test //tests:e2e_tests //tests:integration_tests //tests:chat_pty_test //tests:parity_test

parity:
	@$(BAZEL) test //tests:parity_test
