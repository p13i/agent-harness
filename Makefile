SHELL := /bin/bash
ARGS ?=
BAZEL := npx --yes @bazel/bazelisk
INSTALL_BIN ?= $(HOME)/.local/bin
WORKSPACE ?= $(CURDIR)

.PHONY: build chat daemon doctor install integration lint package parity test

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
	@$(BAZEL) test //tests:unit_tests //tests:integration_tests //tests:parity_test //tests:style_test //tools:install_test

integration:
	@$(BAZEL) test //tests:integration_tests

parity:
	@$(BAZEL) test //tests:parity_test
