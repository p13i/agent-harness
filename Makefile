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
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //tools:ui_gallery -- --output "$(CURDIR)/bazel-bin/ui-gallery"

wsl-e2e:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //tools:wsl_e2e -- $(ARGS)

sync:
	@$(BAZEL) run --ui_event_filters=-info --noshow_progress //cmd:agent-harness -- $(CHAT_STATE_ARG) sync

lint:
	@$(BAZEL) test //tests:style_test

test:
	@$(BAZEL) test //tests:acceptance_test //tests:unit_tests //tests:integration_tests //tests:e2e_tests //tests:chat_pty_test //tests:parity_test //tests:style_test //tools:coverage_gate_test //tools:install_test //tools:live_smoke_test //tools:ui_gallery_test //tools:wsl_e2e_test

coverage:
	@$(BAZEL) coverage //tests:unit_tests //tests:integration_tests //tests:e2e_tests //tests:acceptance_test //tests:chat_pty_test //tests:parity_test //tests:style_test //tools:install_test //tools:live_smoke_test //tools:ui_gallery_test //tools:wsl_e2e_test --combined_report=lcov --instrumentation_filter='//agent_harness[/:],//tools[/:]'
	@$(BAZEL) run //tools:coverage_gate -- \
		--lcov "$(CURDIR)/bazel-out/_coverage/_coverage_report.dat" \
		--minimum 90 \
		--exclude agent_harness/providers/claude.py \
		--exclude agent_harness/providers/codex.py \
		--exclude agent_harness/terminal.py \
		--group "presenter=100:agent_harness/tui_presenter.py,agent_harness/tui_widgets.py" \
		--group "interaction-state=100:agent_harness/tui_presenter.py,agent_harness/tui_widgets.py" \
		--group "presentation-state=100:agent_harness/presentation.py,agent_harness/notifications.py" \
		--group "orchestration=100:agent_harness/orchestration.py" \
		--group "reconciliation=100:agent_harness/reconciliation.py" \
		--group "safety=100:agent_harness/safety.py" \
		--group "service-unit=100:agent_harness/service_manager.py" \
		--group "api=95:agent_harness/api.py" \
		--group "sdk=95:agent_harness/sdk.py" \
		--group "storage=95:agent_harness/storage.py" \
		--group "worker=95:agent_harness/worker.py" \
		--group "client=95:agent_harness/client.py" \
		--group "bundle=95:tools/bundle.py" \
		--group "installer=95:tools/install.py"

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
