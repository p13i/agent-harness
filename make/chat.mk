AGENT_HARNESS_REPO ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
ARGS ?=

.PHONY: chat

chat:
	@$(MAKE) --no-print-directory -C "$(AGENT_HARNESS_REPO)" chat WORKSPACE="$(CURDIR)" ARGS="$(ARGS)"
