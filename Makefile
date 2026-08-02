SHELL := /bin/bash
ARGS ?=
BAZEL := npx --yes @bazel/bazelisk
INSTALL_BIN ?= $(HOME)/.local/bin
WORKSPACE ?= $(CURDIR)
CHAT_STATE_DIR ?=
CHAT_STATE_ARG = $(if $(CHAT_STATE_DIR),--state-dir "$(CHAT_STATE_DIR)",)

.PHONY: acceptance build chat coverage daemon doctor e2e install integration lint live-smoke package parity service sync test ui-gallery wsl-e2e

build:
	@$(BAZEL) build //...

package:
	@$(BAZEL) build //cmd:agent-harness

install: package
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //tools:install -- --repo "$(CURDIR)" --destination "$(INSTALL_BIN)/agent-harness"

chat:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) --cwd "$(WORKSPACE)" chat $(ARGS)

daemon:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) daemon $(ARGS)

doctor:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) doctor $(ARGS)

service:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) service $(ARGS)

ui-gallery:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //agent_harness:ui_gallery -- --output "$(CURDIR)/bazel-bin/ui-gallery"

wsl-e2e:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //agent_harness:wsl_e2e -- $(ARGS)

sync:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) sync

lint:
	@$(BAZEL) test //tests:style_test

test:
	@$(BAZEL) test //tests:acceptance_test //tests:unit_tests //tests:integration_tests //tests:integration_scale_tests //tests:e2e_tests //tests:chat_pty_test //tests:core_boundaries_test //tests:parity_test //tests:provider_boundaries_test //tests:storage_boundaries_test //tests:style_test //tests:tui_boundaries_test //tools:coverage_gate_test //tools:install_test //tools:live_smoke_test //tools:ui_gallery_test //tools:wsl_e2e_test

coverage:
	@$(BAZEL) coverage //tests:unit_tests //tests:integration_tests //tests:integration_scale_tests //tests:e2e_tests //tests:acceptance_test //tests:core_boundaries_test //tests:parity_test //tests:provider_boundaries_test //tests:storage_boundaries_test //tests:style_test //tests:tui_boundaries_test //tools:coverage_gate_test //tools:install_test //tools:live_smoke_test //tools:ui_gallery_test //tools:wsl_e2e_test --combined_report=lcov --instrumentation_filter='//agent_harness[/:],//cmd[/:],//tools[/:]'
	@$(BAZEL) run //tools:coverage_gate -- \
		--lcov "$(CURDIR)/bazel-out/_coverage/_coverage_report.dat" \
		--minimum 100 \
		--per-file-minimum 100 \
		--source-root "$(CURDIR)"

integration:
	@$(BAZEL) test //tests:integration_tests //tests:integration_scale_tests

live-smoke:
	@$(BAZEL) run //tools:live_smoke -- $(ARGS)

e2e:
	@$(BAZEL) test //tests:acceptance_test //tests:e2e_tests //tests:integration_tests //tests:integration_scale_tests //tests:chat_pty_test //tests:parity_test

parity:
	@$(BAZEL) test //tests:parity_test

acceptance:
	@$(BAZEL) test //tests:local_acceptance
