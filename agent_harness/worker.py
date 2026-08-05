"""One durable worker per active harness session."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_harness.blobs import BlobStore
from agent_harness.config import HarnessPaths
from agent_harness.context import (
    CompiledContext,
    compile_context,
    workspace_context_artifacts,
    workspace_instructions,
)
from agent_harness.errors import (
    HarnessError,
    NotFoundError,
    PolicyBlockedError,
    PolicyDeferredError,
    ProviderExhaustedError,
    ProviderUnavailableError,
    ReconciliationRequiredError,
    SafetyGuardError,
    WorkerOwnershipLostError,
)
from agent_harness.goals import (
    GoalConsumption,
    evaluate_goal,
    evaluate_milestones,
    exhausted_budget,
    goal_consumption,
    make_evidence,
)
from agent_harness.handoff import (
    HANDOFF_SCHEMA,
    ORIGIN_FORK_SEED,
    ORIGIN_PROVIDER_SWITCH,
    handoff_envelope,
    handoff_token_budget,
    model_context_window,
)
from agent_harness.ids import new_uuid, utc_now
from agent_harness.models import (
    Attention,
    CommandReceipt,
    CommandStatus,
    Evidence,
    Goal,
    GoalKind,
    GoalStatus,
    Lifecycle,
    ProviderAttempt,
    RoutingDecision,
    Session,
)
from agent_harness.policy import (
    ALLOW,
    BLOCK,
    DEFER,
    LEVEL_COMMAND,
    LEVEL_GOAL,
    RULE_APPROVAL_REQUIRED,
    RULE_SERVER_POLICY,
    PolicyDecision,
    PolicyEvaluator,
    implementation_providers,
    is_review_workload,
    limit_decisions,
    load_server_policy,
)
from agent_harness.process_control import terminate_recorded_process_group
from agent_harness.providers.base import (
    ChildLaunchGate,
    ProviderAdapter,
    ProviderEvent,
)
from agent_harness.providers.normalize import redact_observable, sanitize
from agent_harness.reconciliation import ReconciliationManager
from agent_harness.safety import (
    INTERACTIVE,
    SafetyConsumption,
    SafetyLimits,
    TurnGuard,
    apply_extension,
    effective_effort,
    effort_requires_xhigh_authorization,
    limits_for,
    lower_effort,
    require_state_headroom,
    tighten_limits,
)
from agent_harness.scheduler import Scheduler
from agent_harness.storage import StateStore
from agent_harness.sync import publish_session
from agent_harness.transcript import RenderPolicy, project_transcript, render
from agent_harness.workspace import checkpoint_workspace, workspace_summary
from agent_harness.workspace_state import inspect_workspace

CONTROL_COMMANDS = frozenset({"interrupt", "pause", "resume", "stop", "steer"})
APPROVAL_POLL_LIMIT = 3600
WORKER_HEARTBEAT_INTERVAL_SECONDS = 5.0


def _observe_bash_command_result(
    event: ProviderEvent,
    bash_commands: dict[str, str],
    failures: list[dict[str, Any]],
    *,
    provider: str,
    attempt_id: str,
    turn_id: str,
) -> None:
    metadata = event.metadata
    if metadata is None:
        return
    if event.event_type == "tool.started":
        if str(metadata.get("name", "")).casefold() != "bash":
            return
        tool_id = str(metadata.get("id", ""))
        tool_input = metadata.get("input")
        if not tool_id or not isinstance(tool_input, dict):
            return
        command = tool_input.get("command")
        if isinstance(command, str) and command:
            bash_commands[tool_id] = command
        return
    if event.event_type != "tool.completed":
        return
    tool_id = str(metadata.get("tool_use_id", ""))
    command = bash_commands.pop(tool_id, "")
    if not command:
        return
    exit_code = metadata.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return
    if exit_code == 0:
        return
    failures.append(
        {
            "attempt_id": attempt_id,
            "command": command,
            "exit_code": exit_code,
            "provider": provider,
            "tool_use_id": tool_id,
            "turn_id": turn_id,
        }
    )


def _failed_command_evidence(
    goal: Goal | None,
    failures: object,
) -> tuple[Evidence, ...]:
    if goal is None or goal.kind != GoalKind.FINITE:
        return ()
    if not isinstance(failures, list):
        return ()
    predicates = [
        predicate
        for predicate in goal.predicates
        if str(predicate.get("type", "")) == "command"
        and str(predicate.get("outcome", "passed")) == "passed"
    ]
    result: list[Evidence] = []
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        command = str(failure.get("command", ""))
        exact = [
            predicate
            for predicate in predicates
            if str(predicate.get("subject", "")) == command
        ]
        predicate = None
        if len(exact) == 1:
            predicate = exact[0]
        elif len(predicates) == 1:
            predicate = predicates[0]
        if predicate is None:
            continue
        value = {
            "attempt_id": str(failure.get("attempt_id", "")),
            "command_digest": hashlib.sha256(command.encode("utf-8")).hexdigest(),
            "exit_code": int(failure["exit_code"]),
            "predicate_digest": _structured_digest(predicate),
            "provider": str(failure.get("provider", "")),
            "schema": "p13i/agent-harness/failed-command-evidence/v1",
            "tool_use_id": str(failure.get("tool_use_id", "")),
            "turn_id": str(failure.get("turn_id", "")),
        }
        result.append(
            make_evidence(
                goal.goal_id,
                "command",
                str(predicate.get("subject", "")),
                "failed",
                value,
            )
        )
    return tuple(result)


class SessionWorker:
    def __init__(
        self,
        store: StateStore,
        blobs: BlobStore,
        scheduler: Scheduler,
        adapters: dict[str, ProviderAdapter],
        session_id: str,
        paths: HarnessPaths | None = None,
        policy_approval_handler: Any = None,
    ) -> None:
        self.store = store
        self.blobs = blobs
        self.scheduler = scheduler
        self.adapters = adapters
        self.session_id = session_id
        self.paths = paths
        self._policy_approval_handler = policy_approval_handler
        self.incarnation = new_uuid()
        self._stopping = False
        self._active_adapter: ProviderAdapter | None = None
        self._active_command_id = ""
        self._active_turn_id = ""
        self._worker_heartbeat_at = 0.0

    async def run(self) -> None:
        prior_leases = [
            lease
            for lease in self.store.active_process_leases()
            if lease["session_id"] == self.session_id
        ]
        self.store.register_worker(
            self.session_id,
            os.getpid(),
            self.incarnation,
        )
        self._worker_heartbeat_at = time.monotonic()
        try:
            for lease in prior_leases:
                try:
                    termination = await terminate_recorded_process_group(
                        int(lease["pid"]),
                        str(lease["pid_start"]),
                    )
                except Exception as error:
                    self._pause_for_unresolved_process_lease(
                        lease,
                        "termination-error",
                        type(error).__name__,
                    )
                    return
                if termination == "identity-invalid":
                    self._pause_for_unresolved_process_lease(
                        lease,
                        termination,
                    )
                    return
                self.store.update_process_lease(
                    str(lease["lease_id"]),
                    state="released",
                )
                self.store.append_event(
                    self.session_id,
                    "lease.recovered",
                    status="released",
                    metadata={
                        "lease_id": str(lease["lease_id"]),
                        "command_id": str(lease["command_id"]),
                        "attempt_id": str(lease["attempt_id"]),
                        "provider": str(lease["provider"]),
                        "pid": int(lease["pid"]),
                        "pid_start": str(lease["pid_start"]),
                        "termination": termination,
                    },
                )
            recovery = await ReconciliationManager(
                self.store,
                self.blobs,
            ).recover_after_restart(self.session_id)
            if recovery.reconciliations:
                self.store.update_session(
                    self.session_id,
                    attention=Attention.NEEDS_RECONCILIATION,
                )
                self.store.append_event(
                    self.session_id,
                    "worker.recovered",
                    status="needs-reconciliation",
                    metadata=recovery.as_dict(),
                )
            elif recovery.requeued_command_ids:
                self.store.append_event(
                    self.session_id,
                    "worker.recovered",
                    status="requeued",
                    metadata=recovery.as_dict(),
                )
            await self._loop()
        finally:
            self.store.remove_worker(self.session_id, self.incarnation)

    def _pause_for_unresolved_process_lease(
        self,
        lease: dict[str, Any],
        termination: str,
        error_type: str = "",
    ) -> None:
        self.store.update_process_lease(
            str(lease["lease_id"]),
            state="recovery-blocked",
        )
        self.store.update_session(
            self.session_id,
            lifecycle=Lifecycle.PAUSED,
            attention=Attention.NEEDS_RECONCILIATION,
        )
        self.store.append_event(
            self.session_id,
            "lease.recovery.blocked",
            status="needs-reconciliation",
            metadata={
                "lease_id": str(lease["lease_id"]),
                "command_id": str(lease["command_id"]),
                "attempt_id": str(lease["attempt_id"]),
                "provider": str(lease["provider"]),
                "pid": int(lease["pid"]),
                "pid_start": str(lease["pid_start"]),
                "termination": termination,
                "error_type": error_type,
            },
        )

    async def _loop(self) -> None:
        while not self._stopping:
            if not self._maintain_worker_ownership():
                return
            session = self.store.get_session(self.session_id)
            if session.lifecycle in {
                Lifecycle.STOPPED,
                Lifecycle.COMPLETED,
                Lifecycle.FAILED,
            }:
                return
            control = self.store.claim_command(
                self.session_id,
                CONTROL_COMMANDS,
            )
            if control is not None:
                await self._control(control)
                continue
            if session.lifecycle == Lifecycle.PAUSED:
                await asyncio.sleep(0.2)
                continue
            if self.store.pending_reconciliations(self.session_id):
                if session.attention != Attention.NEEDS_RECONCILIATION:
                    self.store.update_session(
                        self.session_id,
                        attention=Attention.NEEDS_RECONCILIATION,
                    )
                await asyncio.sleep(0.2)
                continue
            command = self.store.claim_command(self.session_id)
            if command is None:
                if (
                    session.lifecycle != Lifecycle.RUNNING
                    or session.attention != Attention.IDLE
                ):
                    self.store.update_session(
                        self.session_id,
                        lifecycle=Lifecycle.RUNNING,
                        attention=Attention.IDLE,
                    )
                await asyncio.sleep(0.2)
                continue
            if command.command_type == "message":
                await self._message(command)
                continue
            self.store.resolve_command(
                command.command_id,
                CommandStatus.FAILED,
                {
                    "code": "E_COMMAND",
                    "message": "unsupported command type",
                },
            )

    def _maintain_worker_ownership(self, *, force: bool = False) -> bool:
        observed_at = time.monotonic()
        heartbeat_due = (
            observed_at - self._worker_heartbeat_at
            >= WORKER_HEARTBEAT_INTERVAL_SECONDS
        )
        if force:
            heartbeat_due = True
        if not heartbeat_due:
            return self.store.worker_owned(
                self.session_id,
                self.incarnation,
            )
        owned = self.store.heartbeat_worker(
            self.session_id,
            self.incarnation,
        )
        if owned:
            self._worker_heartbeat_at = observed_at
        return owned

    async def _message(self, command: CommandReceipt) -> None:
        try:
            await self._execute_message(command)
        finally:
            if self.paths is not None:
                await asyncio.to_thread(
                    publish_session,
                    self.paths,
                    self.store,
                    self.session_id,
                )

    async def _execute_message(self, command: CommandReceipt) -> None:
        payload = self.store.command_payload(command.command_id)
        text = str(payload.get("text", "")).strip()
        if not text:
            self.store.resolve_command(
                command.command_id,
                CommandStatus.FAILED,
                {"code": "E_INPUT", "message": "message text is required"},
            )
            return
        budget = self._exhausted_budget()
        if budget:
            result = {
                "code": "E_GOAL_BUDGET",
                "message": budget + " goal budget is exhausted",
            }
            self.store.update_session(
                self.session_id,
                lifecycle=Lifecycle.PAUSED,
                attention=Attention.NEEDS_INPUT,
            )
            self.store.append_event(
                self.session_id,
                "goal.budget_exhausted",
                status="paused",
                metadata={"budget": budget},
            )
            self.store.resolve_command(
                command.command_id,
                CommandStatus.FAILED,
                result,
            )
            return
        safety = self.store.session_safety(self.session_id)
        profile = str(safety.get("profile", ""))
        if not profile:
            result = {
                "code": "E_SAFETY_PROFILE",
                "message": "session execution profile must be claimed",
            }
            self.store.update_session(
                self.session_id,
                lifecycle=Lifecycle.PAUSED,
                attention=Attention.NEEDS_INPUT,
            )
            self.store.resolve_command(
                command.command_id,
                CommandStatus.FAILED,
                result,
            )
            return
        evaluator: PolicyEvaluator | None = None
        try:
            state_dir = None
            if self.paths is not None:
                state_dir = self.paths.state_dir
            evaluator = PolicyEvaluator(load_server_policy(state_dir))
        except ValueError as error:
            self._emit_policy_decisions(
                [
                    PolicyDecision(
                        outcome=BLOCK,
                        rule=RULE_SERVER_POLICY,
                        reason=str(error),
                        command_id=command.command_id,
                    )
                ]
            )
            self.store.resolve_command(
                command.command_id,
                CommandStatus.FAILED,
                {
                    "code": "E_POLICY_INVALID",
                    "message": str(error),
                },
            )
            return
        existing_envelope: dict[str, Any] | None = None
        try:
            existing_envelope = self.store.command_envelope(command.command_id)
        except NotFoundError:
            existing_envelope = None
        workload = str(payload.get("workload", "implementation"))
        if existing_envelope is None:
            limits = limits_for(profile, workload)
        else:
            profile = str(existing_envelope["profile"])
            limits = SafetyLimits(**existing_envelope["limits"])
        requested_effort = str(payload.get("effort", "")).strip().casefold()
        xhigh_authorization = None
        requires_xhigh = effort_requires_xhigh_authorization(requested_effort)
        if requires_xhigh and profile != "interactive":
            xhigh_authorization = self.store.xhigh_authorization_or_park(
                command.command_id
            )
            if xhigh_authorization is None:
                self.store.update_session(
                    self.session_id,
                    lifecycle=Lifecycle.PAUSED,
                    attention=Attention.NEEDS_INPUT,
                )
                self.store.append_event(
                    self.session_id,
                    "xhigh.authorization.required",
                    status="waiting",
                    metadata={"command_id": command.command_id},
                )
                return
        else:
            xhigh_authorization = self.store.xhigh_authorization(command.command_id)
        if existing_envelope is None:
            extension = self.store.consume_session_extensions(self.session_id)
            limits = apply_extension(limits, extension)
            try:
                limits, server_decisions = evaluator.apply_server_limits(
                    limits,
                    command_id=command.command_id,
                )
                metered_budget = _optional_number(payload.get("metered_budget"))
                goal_base = limits
                limits = self._goal_limited_limits(
                    limits,
                    metered_budget=metered_budget,
                )
                goal_decisions = limit_decisions(
                    LEVEL_GOAL,
                    goal_base,
                    limits,
                    command_id=command.command_id,
                )
                command_base = limits
                limits = tighten_limits(limits, payload.get("safety_limits"))
                command_decisions = limit_decisions(
                    LEVEL_COMMAND,
                    command_base,
                    limits,
                    command_id=command.command_id,
                )
                self._emit_policy_decisions(
                    [
                        *server_decisions,
                        *goal_decisions,
                        *command_decisions,
                    ]
                )
            except ValueError as error:
                self.store.resolve_command(
                    command.command_id,
                    CommandStatus.FAILED,
                    {
                        "code": "E_SAFETY_BUDGET",
                        "message": str(error),
                    },
                )
                return
        xhigh_authorized = xhigh_authorization is not None
        if requires_xhigh and xhigh_authorization is not None:
            authorized_provider = str(xhigh_authorization.get("provider", ""))
            requested_provider = str(payload.get("provider", ""))
            if requested_provider and requested_provider != authorized_provider:
                xhigh_authorized = False
            else:
                payload = dict(payload)
                payload["provider"] = authorized_provider
        try:
            effort = effective_effort(
                requested_effort,
                limits,
                xhigh_authorized=xhigh_authorized,
            )
        except ValueError as error:
            self.store.resolve_command(
                command.command_id,
                CommandStatus.FAILED,
                {
                    "code": "E_SAFETY_EFFORT",
                    "message": str(error),
                },
            )
            return
        if effort_requires_xhigh_authorization(effort) and profile != "interactive":
            limits = replace(limits, max_attempts=1)
        payload = dict(payload)
        payload["_effort_pinned"] = bool(requested_effort.strip())
        payload["effort"] = effort
        envelope = self.store.create_command_envelope(
            command.command_id,
            self.session_id,
            profile,
            limits.as_dict(),
        )
        self.store.create_child_launch_gate(
            command.command_id,
            self.session_id,
            limits.max_child_agents,
        )
        if existing_envelope is None:
            self.store.append_event(
                self.session_id,
                "usage.reserved",
                status="complete",
                metadata={
                    "command_id": command.command_id,
                    "profile": profile,
                    "limits": envelope["limits"],
                },
            )
        consumption = SafetyConsumption(**envelope["consumption"])
        guard = TurnGuard(limits, consumption)
        self.store.update_session(
            self.session_id,
            lifecycle=Lifecycle.RUNNING,
            attention=Attention.WORKING,
        )
        try:
            provider = str(payload.get("provider", "")).strip()
            if not provider:
                provider = "automatic-route"
            require_state_headroom(
                self.store.path.parent,
                provider,
            )
            result = await self._execute_with_failover(
                command.command_id,
                payload,
                text,
                guard,
                evaluator,
            )
        except PolicyDeferredError as error:
            self.store.append_event(
                self.session_id,
                "policy.paused",
                status="paused",
                metadata={
                    "command_id": command.command_id,
                    "rule": error.rule,
                    "reason": error.reason,
                    "provider": error.provider,
                },
            )
            self.store.update_session(
                self.session_id,
                lifecycle=Lifecycle.PAUSED,
                attention=Attention.NEEDS_INPUT,
            )
            self.store.update_command_envelope(
                command.command_id,
                state="paused",
                consumption=guard.consumption.as_dict(),
                guard_reason="policy-deferred:" + error.rule,
            )
            self.store.requeue_command(command.command_id)
            return
        except ReconciliationRequiredError as error:
            self.store.update_session(
                self.session_id,
                lifecycle=Lifecycle.PAUSED,
                attention=Attention.NEEDS_RECONCILIATION,
            )
            self.store.update_command_envelope(
                command.command_id,
                state="paused",
                consumption=guard.consumption.as_dict(),
                guard_reason=error.reason,
            )
            return
        except HarnessError as error:
            self.store.append_event(
                self.session_id,
                "turn.failed",
                status="failed",
                text=error.detail.message,
                metadata={
                    "code": error.detail.code,
                    "retryable": error.detail.retryable,
                },
            )
            self.store.update_session(
                self.session_id,
                lifecycle=Lifecycle.PAUSED,
                attention=Attention.NEEDS_INPUT,
            )
            self.store.update_command_envelope(
                command.command_id,
                state="paused",
                consumption=guard.consumption.as_dict(),
                guard_reason=getattr(error, "reason", ""),
            )
            self.store.resolve_command(
                command.command_id,
                CommandStatus.FAILED,
                {
                    "code": error.detail.code,
                    "message": error.detail.message,
                    "retryable": error.detail.retryable,
                },
            )
            return
        self.store.complete_command_execution(
            command.command_id,
            str(result["turn_id"]),
            str(result["native_session_id"]),
            guard.consumption.as_dict(),
            result,
            goal_evidence=_failed_command_evidence(
                self.store.goal_for_session(self.session_id),
                result.pop("failed_command_results", []),
            ),
        )
        await self._evaluate_goal()

    async def _execute_with_failover(
        self,
        command_id: str,
        payload: dict[str, Any],
        text: str,
        guard: TurnGuard,
        evaluator: PolicyEvaluator | None = None,
    ) -> dict[str, Any]:
        excluded: set[str] = set()
        first_error: HarnessError | None = None
        attempt_payload = dict(payload)
        recovery_stage = 0
        for attempt_index in range(guard.limits.max_attempts):
            try:
                return await self._execute_attempt(
                    command_id,
                    attempt_payload,
                    text,
                    frozenset(excluded),
                    guard,
                    recovery_stage,
                    evaluator,
                    enforce_concurrency=True,
                )
            except SafetyGuardError as error:
                if first_error is None:
                    first_error = error
                if not error.recoverable:
                    raise
                if recovery_stage >= 2:
                    raise
                effort_pinned = bool(payload.get("_effort_pinned", False))
                if recovery_stage == 0 and not effort_pinned:
                    lowered = lower_effort(str(attempt_payload.get("effort", "")))
                    if lowered != str(attempt_payload.get("effort", "")):
                        guard.recover()
                        attempt_payload["provider"] = error.provider
                        attempt_payload["effort"] = lowered
                        recovery_stage = 1
                        self._record_recovery(
                            command_id,
                            recovery_stage,
                            "downgrade",
                            error.provider,
                            lowered,
                        )
                        continue
                if str(payload.get("provider", "")):
                    raise
                guard.recover()
                excluded.add(error.provider)
                attempt_payload.pop("provider", None)
                if not effort_pinned:
                    attempt_payload["effort"] = lower_effort(
                        str(attempt_payload.get("effort", ""))
                    )
                recovery_stage = 2
                self._record_recovery(
                    command_id,
                    recovery_stage,
                    "failover",
                    error.provider,
                    str(attempt_payload.get("effort", "")),
                )
                continue
            except (ProviderExhaustedError, ProviderUnavailableError) as error:
                if first_error is None:
                    first_error = error
                provider = error.provider
                if str(payload.get("provider", "")):
                    raise
                excluded.add(provider)
                attempt_payload.pop("provider", None)
                recovery_stage = 2
                self.store.update_command_envelope(
                    command_id,
                    state="recovering",
                    recovery_stage=recovery_stage,
                    consumption=guard.consumption.as_dict(),
                )
                self.store.append_event(
                    self.session_id,
                    "routing.failover",
                    status="retrying",
                    metadata={
                        "excluded_provider": provider,
                        "reason": error.detail.code,
                    },
                )
                continue
            break
        if first_error is not None:
            raise first_error
        raise ProviderUnavailableError("all providers")

    async def _execute_attempt(
        self,
        command_id: str,
        payload: dict[str, Any],
        text: str,
        excluded: frozenset[str],
        guard: TurnGuard,
        recovery_stage: int,
        evaluator: PolicyEvaluator | None = None,
        *,
        enforce_concurrency: bool,
    ) -> dict[str, Any]:
        session = self.store.get_session(self.session_id)
        goal = self.store.goal_for_session(self.session_id)
        turn_permission_mode = str(
            payload.get("permission_mode", session.permission_mode)
        )
        context = self._compile_context(session, guard.limits, command_id)
        turn_ref = payload.get("turn_ref")
        if not isinstance(turn_ref, dict):
            turn_ref = {}
        self._guard_repeated_dispatch(
            command_id,
            text,
            context.text,
            Path(session.worktree),
            turn_ref,
            guard.limits.profile,
        )
        metered_budget = _optional_number(payload.get("metered_budget"))
        routing_metered_budget = metered_budget
        if guard.limits.max_dollars is not None:
            if routing_metered_budget is None:
                routing_metered_budget = guard.limits.max_dollars
            else:
                routing_metered_budget = min(
                    routing_metered_budget,
                    guard.limits.max_dollars,
                )
        goal_id = ""
        permitted_providers: frozenset[str] = frozenset()
        permitted_efforts: frozenset[str] = frozenset()
        max_concurrency = 1
        if goal is not None:
            goal_id = goal.goal_id
            permitted_providers = frozenset(goal.permitted_providers)
            permitted_efforts = frozenset(goal.permitted_efforts)
            max_concurrency = goal.max_concurrency
        routing_provider = str(payload.get("provider", ""))
        fault_probe = payload.get("proof_fault_probe")
        if not isinstance(fault_probe, dict):
            fault_probe = {}
        service_fault_probe = payload.get("proof_service_fault_probe")
        if not isinstance(service_fault_probe, dict):
            service_fault_probe = {}
        if recovery_stage == 0 and fault_probe:
            routing_provider = str(fault_probe.get("provider", ""))
        elif recovery_stage == 0 and service_fault_probe:
            routing_provider = str(service_fault_probe.get("provider", ""))
        workload_value = str(payload.get("workload", "implementation"))
        implementing: frozenset[str] = frozenset()
        if (
            evaluator is not None
            and evaluator.review_provider_must_differ
            and is_review_workload(workload_value)
        ):
            implementing = implementation_providers(
                self.store.attempts(self.session_id),
                self.store.routing_decisions(self.session_id),
            )
        decision = await self.scheduler.choose(
            session,
            workload=workload_value,
            required_capabilities=frozenset(
                str(item) for item in payload.get("required_capabilities", [])
            ),
            provider=routing_provider,
            model=str(payload.get("model", "")),
            effort=str(payload.get("effort", "")),
            metered_budget=routing_metered_budget,
            excluded=excluded,
            context_transfer_tokens=context.estimated_tokens,
            binding_ceiling=guard.limits.binding_ceiling,
            execution_profile=guard.limits.profile,
            enforce_concurrency=enforce_concurrency,
            command_id=command_id,
            goal_id=goal_id,
            permitted_providers=permitted_providers,
            permitted_efforts=permitted_efforts,
            max_concurrency=max_concurrency,
            policy=evaluator,
            implementation_providers=implementing,
        )
        await self._enforce_dispatch_approval(
            decision,
            evaluator,
            command_id,
            guard.limits.profile,
        )
        adapter = self.adapters[decision.provider]
        native_session_id = self._native_session(decision.provider)
        unavailable_native_session_id = ""
        if native_session_id and not adapter.native_session_available(
            Path(session.worktree),
            native_session_id,
        ):
            unavailable_native_session_id = native_session_id
            native_session_id = ""
        handoff_block = ""
        handoff_metadata: dict[str, Any] = {}
        if not native_session_id:
            handoff_block, handoff_metadata = await self._handoff_context(
                session,
                decision,
                guard.limits,
                context,
            )
        prompt = text
        context_digest = ""
        context_payload_digest = ""
        context_transport = "context-package"
        generation_digest = self.store.repetition_generation(self.session_id)[
            "generation_digest"
        ]
        if not native_session_id or session.active_provider != decision.provider:
            prompt = (
                handoff_block + context.text + "\n\n# Next instruction\n\n" + text
            )
            context_payload_digest = hashlib.sha256(
                context.text.encode("utf-8")
            ).hexdigest()
            context_digest = _structured_digest(
                {
                    "checkpoint_generation": generation_digest,
                    "payload_digest": context_payload_digest,
                }
            )
        else:
            context_transport = "native-resume"
            context_payload_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            context_digest = _structured_digest(
                {
                    "checkpoint_generation": generation_digest,
                    "command_id": command_id,
                    "instruction_digest": context_payload_digest,
                    "native_session_id": native_session_id,
                    "transport": "native-resume",
                }
            )
        submitted_tokens = (len(prompt) + 3) // 4
        violation = guard.begin_attempt(
            submitted_tokens,
            charge_reported_cost=decision.credits_engaged,
        )
        if violation:
            raise SafetyGuardError(
                violation,
                decision.provider,
            )
        lease_id = ""
        attempt_id = new_uuid()
        attempt = ProviderAttempt(
            attempt_id=attempt_id,
            session_id=self.session_id,
            provider=decision.provider,
            native_session_id=native_session_id,
            model=decision.model,
            effort=decision.effort,
            auth_mode="subscription",
            status="running",
            started_at=utc_now(),
            ended_at="",
        )
        self.store.create_attempt(attempt)
        self.store.update_command_envelope(
            command_id,
            recovery_stage=recovery_stage,
            consumption=guard.consumption.as_dict(),
        )
        turn_id = self.store.start_turn(
            self.session_id,
            attempt_id,
            turn_ref=turn_ref,
        )
        self.store.record_routing(
            self.session_id,
            turn_id,
            decision.provider,
            decision.model,
            decision.effort,
            decision.as_dict(),
        )
        self.store.update_session(
            self.session_id,
            active_provider=decision.provider,
            model=decision.model,
            effort=decision.effort,
        )
        child_gate_state = self.store.child_launch_gate(command_id)
        self.store.append_event(
            self.session_id,
            "routing.selected",
            status="complete",
            metadata={
                **decision.as_dict(),
                "turn_ref": turn_ref,
                "execution_profile": guard.limits.profile,
                "parent_permission_mode": turn_permission_mode,
                "child_permission_mode": turn_permission_mode,
                "child_gate_remaining": max(
                    0,
                    guard.limits.max_child_agents - int(child_gate_state["consumed"]),
                ),
            },
            turn_id=turn_id,
        )
        if handoff_metadata:
            self.store.append_event(
                self.session_id,
                "session.handoff",
                status="complete",
                metadata=handoff_metadata,
                turn_id=turn_id,
            )
        if unavailable_native_session_id:
            self.store.append_event(
                self.session_id,
                "provider.native_resume.fallback",
                status="canonical-context",
                metadata={
                    "provider": decision.provider,
                    "unavailable_native_session_id": unavailable_native_session_id,
                    "context_digest": context_digest,
                    "context_payload_digest": context_payload_digest,
                },
                turn_id=turn_id,
            )
        resolved_native_session_id = native_session_id
        context_delivery_accepted = False
        activity_event_count = 0
        bash_commands: dict[str, str] = {}
        failed_command_results: list[dict[str, Any]] = []
        service_fault_active = (
            recovery_stage == 0
            and str(service_fault_probe.get("provider", "")) == decision.provider
            and str(service_fault_probe.get("stage", ""))
            == "after-acceptance-before-terminal"
        )
        service_fault_ready = False

        async def event_handler(event: ProviderEvent) -> None:
            nonlocal resolved_native_session_id
            nonlocal context_delivery_accepted
            nonlocal activity_event_count
            nonlocal service_fault_ready
            if (
                event.event_type.startswith("tool.")
                or event.event_type.startswith("file.change.")
                or (
                    event.event_type.startswith("agent.message")
                    and event.text.strip()
                )
            ):
                activity_event_count += 1
            if (
                not context_delivery_accepted
                and event.event_type == "provider.prompt.accepted"
            ):
                self.store.accept_context_delivery(
                    self.session_id,
                    decision.provider,
                    context_digest,
                    attempt_id,
                )
                context_delivery_accepted = True
            if (
                event.native_session_id
                and event.native_session_id != resolved_native_session_id
            ):
                resolved_native_session_id = event.native_session_id
                self.store.update_attempt(
                    attempt_id,
                    status="running",
                    native_session_id=resolved_native_session_id,
                )
            _observe_bash_command_result(
                event,
                bash_commands,
                failed_command_results,
                provider=decision.provider,
                attempt_id=attempt_id,
                turn_id=turn_id,
            )
            guard.observe(event)
            tool_pair = guard.take_completed_tool_pair()
            if tool_pair:
                self._record_tool_result_fingerprint(
                    command_id,
                    turn_id,
                    tool_pair,
                )
            if event.event_type.startswith("file.change."):
                material_digest, unused_summary = await asyncio.to_thread(
                    inspect_workspace,
                    Path(session.worktree),
                )
                del unused_summary
                guard.note_material_progress(material_digest)
            self.store.update_command_envelope(
                command_id,
                consumption=guard.consumption.as_dict(),
            )
            if guard.warning_due():
                self.store.append_event(
                    self.session_id,
                    "guard.warning",
                    status="warning",
                    metadata={
                        "command_id": command_id,
                        "snapshot": guard.snapshot(),
                    },
                    turn_id=turn_id,
                )
            await self._provider_event(turn_id, event)
            if (
                service_fault_active
                and not service_fault_ready
                and event.event_type == "provider.prompt.accepted"
            ):
                service_fault_ready = True
                readiness = {
                    "schema": "p13i/agent-harness/service-fault-readiness/v1",
                    "command_id": command_id,
                    "attempt_id": attempt_id,
                    "provider": decision.provider,
                    "native_session_id": resolved_native_session_id,
                    "context_digest": context_digest,
                    "authorization_digest": str(
                        service_fault_probe.get("authorization_digest", "")
                    ),
                    "expires_at": _fault_probe_expiry(),
                }
                self.store.append_event(
                    self.session_id,
                    "proof.service-fault.ready",
                    status="waiting",
                    metadata=readiness,
                    turn_id=turn_id,
                )
                await asyncio.sleep(30)
                self.store.append_event(
                    self.session_id,
                    "proof.service-fault.timeout",
                    status="failed",
                    metadata=readiness,
                    turn_id=turn_id,
                )
                raise SafetyGuardError(
                    "proof-service-fault-timeout",
                    decision.provider,
                )

        async def approval_handler(
            method: str,
            request: dict[str, Any],
        ) -> dict[str, Any]:
            return await self._approval(turn_id, method, request)

        pre_dispatch_checkpoint = await asyncio.to_thread(
            checkpoint_workspace,
            session,
            self.blobs,
            sequence=self.store.last_sequence(self.session_id),
            provider=decision.provider,
            native_session_id=native_session_id,
            context_text=context.text,
        )
        self.store.add_checkpoint(pre_dispatch_checkpoint)
        self.store.record_dispatch_checkpoint(
            command_id,
            attempt_id,
            turn_id,
            pre_dispatch_checkpoint.checkpoint_id,
        )
        material_digest, unused_summary = await asyncio.to_thread(
            inspect_workspace,
            Path(session.worktree),
        )
        del unused_summary
        guard.establish_material_state(material_digest)
        try:
            self.store.prepare_context_delivery(
                self.session_id,
                decision.provider,
                context_digest,
                pre_dispatch_checkpoint.checkpoint_id,
                command_id,
                attempt_id,
                context_payload_digest,
                transport=context_transport,
            )
        except BaseException:
            self.store.update_attempt(attempt_id, status="failed")
            self.store.finish_turn(turn_id, "failed")
            self.store.complete_dispatch(attempt_id, "failed")
            self.store.update_command_envelope(
                command_id,
                state="reserved",
                consumption=guard.consumption.as_dict(),
            )
            raise
        self.store.append_event(
            self.session_id,
            "checkpoint.created",
            status="pre-dispatch",
            metadata={
                **pre_dispatch_checkpoint.as_dict(),
                "command_id": command_id,
                "attempt_id": attempt_id,
            },
            turn_id=turn_id,
        )
        lease_attached = False
        lease_pid = 0
        lease_pid_start = ""
        last_lease_heartbeat = time.monotonic()
        fault_probe = payload.get("proof_fault_probe")
        if not isinstance(fault_probe, dict):
            fault_probe = {}
        fault_probe_active = (
            recovery_stage == 0
            and str(fault_probe.get("provider", "")) == decision.provider
            and str(fault_probe.get("stage", "")) == "after-lease-before-acceptance"
        )
        fault_probe_expires_at = _fault_probe_expiry()

        def attach_process_lease() -> bool:
            nonlocal lease_attached
            nonlocal lease_pid
            nonlocal lease_pid_start
            nonlocal last_lease_heartbeat
            if not lease_id:
                return False
            if lease_attached:
                return True
            pid, pid_start = adapter.process_identity()
            if pid <= 0 or not pid_start:
                return False
            self.store.update_process_lease(
                lease_id,
                pid=pid,
                pid_start=pid_start,
                state="active",
                expires_at=_lease_expiry(),
            )
            lease_attached = True
            lease_pid = pid
            lease_pid_start = pid_start
            last_lease_heartbeat = time.monotonic()
            self.store.append_event(
                self.session_id,
                "lease.attached",
                status="complete",
                metadata={
                    "lease_id": lease_id,
                    "command_id": command_id,
                    "attempt_id": attempt_id,
                    "provider": decision.provider,
                    "pid": pid,
                    "pid_start": pid_start,
                },
                turn_id=turn_id,
            )
            return True

        async def pre_prompt_gate() -> None:
            if not fault_probe_active:
                return
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 30.0
            while not attach_process_lease():
                if loop.time() >= deadline:
                    raise SafetyGuardError(
                        "proof-fault-readiness-timeout",
                        decision.provider,
                    )
                await asyncio.sleep(0.025)
            readiness = {
                "schema": "p13i/agent-harness/provider-fault-readiness/v1",
                "command_id": command_id,
                "attempt_id": attempt_id,
                "lease_id": lease_id,
                "provider": decision.provider,
                "pid": lease_pid,
                "pid_start": lease_pid_start,
                "expires_at": fault_probe_expires_at,
                "authorization_digest": str(
                    fault_probe.get("authorization_digest", "")
                ),
            }
            self.store.update_command_envelope(
                command_id,
                state="fault-ready",
                consumption=guard.consumption.as_dict(),
            )
            self.store.append_event(
                self.session_id,
                "proof.fault.ready",
                status="waiting",
                metadata=readiness,
                turn_id=turn_id,
            )
            while loop.time() < deadline:
                pid, pid_start = adapter.process_identity()
                if pid <= 0:
                    observed = dict(readiness)
                    observed["observed_at"] = utc_now()
                    observed["termination"] = "external-process-termination"
                    self.store.append_event(
                        self.session_id,
                        "proof.fault.observed",
                        status="complete",
                        metadata=observed,
                        turn_id=turn_id,
                    )
                    raise ProviderUnavailableError(
                        decision.provider,
                        detail="authorized external proof fault observed",
                    )
                if pid != lease_pid or pid_start != lease_pid_start:
                    raise SafetyGuardError(
                        "proof-fault-process-identity-changed",
                        decision.provider,
                    )
                await asyncio.sleep(0.05)
            self.store.append_event(
                self.session_id,
                "proof.fault.timeout",
                status="failed",
                metadata=readiness,
                turn_id=turn_id,
            )
            raise SafetyGuardError(
                "proof-fault-termination-timeout",
                decision.provider,
            )

        try:
            admission = self.store.reserve_route_admission(
                command_id,
                decision.provider,
                guard.limits.profile,
                effort=decision.effort,
                attempt_id=attempt_id,
                worker_incarnation=self.incarnation,
                goal_id=goal_id,
                max_concurrency=max_concurrency,
                lease_expires_at=_lease_expiry(),
            )
            if not bool(admission["admitted"]):
                raise ProviderUnavailableError(
                    decision.provider,
                    detail=(
                        "atomic route admission denied: " + str(admission["reason"])
                    ),
                )
            lease_id = str(admission["lease_id"])
            if lease_id:
                self.store.append_event(
                    self.session_id,
                    "lease.reserved",
                    status="complete",
                    metadata={
                        "lease_id": lease_id,
                        "command_id": command_id,
                        "attempt_id": attempt_id,
                        "provider": decision.provider,
                    },
                    turn_id=turn_id,
                )
            run_task = asyncio.create_task(
                adapter.run_turn(
                    workspace=Path(session.worktree),
                    prompt=prompt,
                    native_session_id=native_session_id,
                    permission_mode=turn_permission_mode,
                    model=decision.model,
                    effort=decision.effort,
                    event_handler=event_handler,
                    approval_handler=approval_handler,
                    child_launch_gate=ChildLaunchGate(
                        database=self.store.path,
                        command_id=command_id,
                        limit=guard.limits.max_child_agents,
                    ),
                    pre_prompt_gate=pre_prompt_gate,
                )
            )
            self._active_adapter = adapter
            self._active_command_id = command_id
            self._active_turn_id = turn_id
        except BaseException:
            if lease_id:
                self.store.update_process_lease(
                    lease_id,
                    state="released",
                )
            self.store.update_attempt(attempt_id, status="failed")
            self.store.finish_turn(turn_id, "failed")
            self.store.complete_dispatch(attempt_id, "failed")
            self.store.update_command_envelope(
                command_id,
                state="recovering",
                consumption=guard.consumption.as_dict(),
            )
            raise
        try:
            while not run_task.done():
                if not self._maintain_worker_ownership():
                    await self._interrupt_guarded_turn(adapter, run_task)
                    raise WorkerOwnershipLostError(
                        "worker incarnation lost active dispatch ownership"
                    )
                if lease_id:
                    attach_process_lease()
                    now = time.monotonic()
                    if now - last_lease_heartbeat >= 15:
                        self.store.update_process_lease(
                            lease_id,
                            expires_at=_lease_expiry(),
                        )
                        last_lease_heartbeat = now
                control = self.store.claim_command(
                    self.session_id,
                    CONTROL_COMMANDS,
                )
                if control is not None:
                    await self._control(control)
                child_gate_state = self.store.child_launch_gate(command_id)
                guard.note_child_admissions(int(child_gate_state["consumed"]))
                violation = guard.violation()
                if violation:
                    await self._interrupt_guarded_turn(
                        adapter,
                        run_task,
                    )
                    recoverable = (
                        violation == "stagnation"
                        and not context_delivery_accepted
                    )
                    action = "pause"
                    if recoverable and recovery_stage == 0:
                        action = "downgrade"
                    elif recoverable and recovery_stage == 1:
                        action = "failover"
                    snapshot = guard.snapshot()
                    self.store.add_guard_incident(
                        self.session_id,
                        command_id,
                        attempt_id,
                        violation,
                        action,
                        snapshot,
                    )
                    self.store.append_event(
                        self.session_id,
                        "guard.tripped",
                        status="interrupted",
                        metadata={
                            "command_id": command_id,
                            "attempt_id": attempt_id,
                            "reason": violation,
                            "action": action,
                            "snapshot": snapshot,
                        },
                        turn_id=turn_id,
                    )
                    await self._guard_checkpoint(
                        session,
                        decision.provider,
                        resolved_native_session_id,
                        context.text,
                        turn_id,
                    )
                    raise SafetyGuardError(
                        violation,
                        decision.provider,
                        recoverable=recoverable,
                    )
                await asyncio.sleep(0.1)
            result = await run_task
            if not self._maintain_worker_ownership(force=True):
                raise WorkerOwnershipLostError(
                    "worker incarnation lost result ownership"
                )
            guard.observe(
                ProviderEvent(
                    "usage.updated",
                    metadata=result.usage,
                )
            )
        except SafetyGuardError as error:
            matching_incident = any(
                item["command_id"] == command_id
                and item["attempt_id"] == attempt_id
                and item["reason"] == error.reason
                for item in self.store.guard_incidents(self.session_id)
            )
            if not matching_incident:
                self.store.add_guard_incident(
                    self.session_id,
                    command_id,
                    attempt_id,
                    error.reason,
                    "pause",
                    guard.snapshot(),
                )
            self.store.update_attempt(attempt_id, status="interrupted")
            self.store.finish_turn(turn_id, "interrupted")
            self.store.complete_dispatch(attempt_id, "interrupted")
            if context_delivery_accepted:
                reconciliation_id = await self._pause_for_ambiguous_dispatch(
                    command_id,
                    turn_id,
                    decision.provider,
                    error.reason,
                    command_error=error,
                )
                raise ReconciliationRequiredError(
                    reconciliation_id,
                    reason=error.reason,
                ) from error
            raise
        except (ProviderExhaustedError, ProviderUnavailableError) as error:
            if not context_delivery_accepted:
                attempt_status = "failed"
                if isinstance(error, ProviderExhaustedError):
                    attempt_status = "exhausted"
                self.store.update_attempt(attempt_id, status=attempt_status)
                self.store.finish_turn(turn_id, "failed")
                self.store.complete_dispatch(attempt_id, "failed")
                raise
            reconciliation_id = await self._pause_for_ambiguous_dispatch(
                command_id,
                turn_id,
                decision.provider,
                error.detail.code,
            )
            raise ReconciliationRequiredError(reconciliation_id) from error
        except ReconciliationRequiredError:
            raise
        except WorkerOwnershipLostError:
            await self._interrupt_guarded_turn(adapter, run_task)
            raise
        except BaseException:
            await self._interrupt_guarded_turn(adapter, run_task)
            self.store.update_attempt(attempt_id, status="failed")
            self.store.finish_turn(turn_id, "failed")
            self.store.complete_dispatch(attempt_id, "failed")
            raise
        finally:
            worker_owned = self.store.worker_owned(
                self.session_id,
                self.incarnation,
            )
            if worker_owned:
                child_gate_state = self.store.child_launch_gate(command_id)
                guard.note_child_admissions(int(child_gate_state["consumed"]))
                self.store.update_command_envelope(
                    command_id,
                    consumption=guard.consumption.as_dict(),
                )
            self._active_adapter = None
            if self._active_command_id == command_id:
                self._active_command_id = ""
                self._active_turn_id = ""
            if lease_id and worker_owned:
                self.store.update_process_lease(
                    lease_id,
                    state="released",
                )
                self.store.append_event(
                    self.session_id,
                    "lease.released",
                    status="complete",
                    metadata={
                        "lease_id": lease_id,
                        "command_id": command_id,
                        "attempt_id": attempt_id,
                        "provider": decision.provider,
                        "pid": lease_pid,
                        "pid_start": lease_pid_start,
                    },
                    turn_id=turn_id,
                )
        terminal_guard_reason = ""
        if decision.credits_engaged and not guard.consumption.exact_dollars:
            terminal_guard_reason = "dollar-accounting"
        if not terminal_guard_reason:
            terminal_guard_reason = guard.violation()
        checkpoint = pre_dispatch_checkpoint
        completed_material_digest = ""
        if terminal_guard_reason or (
            result.status == "complete" and not result.ambiguous_mutation
        ):
            current = self.store.get_session(self.session_id)
            checkpoint = await asyncio.to_thread(
                checkpoint_workspace,
                current,
                self.blobs,
                sequence=self.store.last_sequence(self.session_id),
                provider=decision.provider,
                native_session_id=result.native_session_id,
                context_text=context.text,
            )
            self.store.add_checkpoint(checkpoint)
            completed_material_digest, unused_summary = await asyncio.to_thread(
                inspect_workspace,
                Path(current.worktree),
            )
            del unused_summary
            self.store.append_event(
                self.session_id,
                "checkpoint.created",
                status=result.status,
                metadata=checkpoint.as_dict(),
                turn_id=turn_id,
            )
        if terminal_guard_reason:
            terminal_action = "pause"
            if result.ambiguous_mutation:
                terminal_action = "reconcile"
            snapshot = guard.snapshot()
            self.store.add_guard_incident(
                self.session_id,
                command_id,
                attempt_id,
                terminal_guard_reason,
                terminal_action,
                snapshot,
            )
            self.store.append_event(
                self.session_id,
                "guard.tripped",
                status="failed",
                metadata={
                    "command_id": command_id,
                    "attempt_id": attempt_id,
                    "reason": terminal_guard_reason,
                    "action": terminal_action,
                    "snapshot": snapshot,
                    "provider_terminal": True,
                    "checkpoint_id": checkpoint.checkpoint_id,
                },
                turn_id=turn_id,
            )
            terminal_error = SafetyGuardError(
                terminal_guard_reason,
                decision.provider,
            )
            if result.ambiguous_mutation:
                reconciliation_id = await self._pause_for_ambiguous_dispatch(
                    command_id,
                    turn_id,
                    decision.provider,
                    terminal_guard_reason,
                    command_error=terminal_error,
                )
                raise ReconciliationRequiredError(
                    reconciliation_id,
                    reason=terminal_guard_reason,
                )
            self.store.update_attempt(
                attempt_id,
                status="failed",
                native_session_id=result.native_session_id,
            )
            self.store.finish_turn(turn_id, "failed")
            self.store.complete_dispatch(attempt_id, "failed")
            raise terminal_error
        if result.ambiguous_mutation:
            reconciliation_id = await self._pause_for_ambiguous_dispatch(
                command_id,
                turn_id,
                decision.provider,
                "E_PROVIDER_AMBIGUOUS_MUTATION",
            )
            raise ReconciliationRequiredError(reconciliation_id)
        if (
            result.status == "complete"
            and activity_event_count == 0
            and completed_material_digest == material_digest
        ):
            self.store.update_attempt(
                attempt_id,
                status="failed",
                native_session_id=result.native_session_id,
            )
            self.store.finish_turn(turn_id, "no-progress")
            self.store.complete_dispatch(attempt_id, "failed")
            raise HarnessError(
                "E_PROVIDER_NO_PROGRESS",
                decision.provider
                + " completed the turn with no output and no workspace change",
                status=502,
            )
        if result.status != "complete":
            terminal_status = "failed"
            if result.status == "cancelled":
                terminal_status = "cancelled"
            self.store.update_attempt(
                attempt_id,
                status=terminal_status,
                native_session_id=result.native_session_id,
            )
            self.store.finish_turn(turn_id, terminal_status)
            self.store.complete_dispatch(attempt_id, "failed")
            raise HarnessError(
                "E_PROVIDER_RESULT",
                decision.provider + " returned a non-complete result",
                status=502,
            )
        self.store.update_session(
            self.session_id,
            attention=Attention.IDLE,
        )
        return {
            "provider": decision.provider,
            "model": decision.model,
            "effort": decision.effort,
            "turn_id": turn_id,
            "native_session_id": result.native_session_id,
            "status": result.status,
            "checkpoint_id": checkpoint.checkpoint_id,
            "workspace_material_digest": completed_material_digest,
            "usage": result.usage,
            "safety": guard.snapshot(),
            "failed_command_results": failed_command_results,
        }

    async def _pause_for_ambiguous_dispatch(
        self,
        command_id: str,
        turn_id: str,
        provider: str,
        reason: str,
        *,
        command_error: HarnessError | None = None,
    ) -> str:
        manager = ReconciliationManager(self.store, self.blobs)
        recovery = await manager.recover_after_restart(self.session_id)
        record = next(
            (
                item
                for item in recovery.reconciliations
                if item.command_id == command_id
            ),
            None,
        )
        if record is None:
            raise RuntimeError("ambiguous dispatch did not create reconciliation")
        if command_error is not None:
            self.store.set_reconciliation_command_error(
                record.reconciliation_id,
                command_error.detail.code,
                command_error.detail.message,
            )
        self.store.finish_turn(turn_id, "ambiguous")
        self.store.update_session(
            self.session_id,
            lifecycle=Lifecycle.PAUSED,
            attention=Attention.NEEDS_RECONCILIATION,
        )
        self.store.append_event(
            self.session_id,
            "reconciliation.requested",
            status="pending",
            metadata={
                "reconciliation_id": record.reconciliation_id,
                "command_id": command_id,
                "provider": provider,
                "reason": reason,
                "discovery_checkpoint_id": str(
                    record.audit.get("discovery_checkpoint_id", "")
                ),
            },
            turn_id=turn_id,
        )
        return record.reconciliation_id

    async def _provider_event(
        self,
        turn_id: str,
        event: ProviderEvent,
    ) -> None:
        digest = ""
        role = ""
        if event.event_type.startswith("agent."):
            role = "assistant"
        if event.event_type.startswith("user."):
            role = "user"
        metadata = {}
        if event.metadata is not None:
            redacted_metadata = redact_observable(event.metadata)
            if isinstance(redacted_metadata, dict):
                metadata = redacted_metadata
        turn_ref = self.store.turn_ref(turn_id)
        if turn_ref:
            metadata["turn_ref"] = turn_ref
        self.store.append_event(
            self.session_id,
            event.event_type,
            role=role,
            text=sanitize(event.text),
            status=event.status,
            metadata=metadata,
            blob_digest=digest,
            turn_id=turn_id,
        )

    def _emit_policy_decisions(
        self,
        decisions: list[PolicyDecision],
        *,
        turn_id: str = "",
    ) -> None:
        for decision in decisions:
            self.store.append_event(
                self.session_id,
                "policy.decision",
                status=decision.outcome,
                metadata=decision.as_dict(),
                turn_id=turn_id,
            )

    async def _enforce_dispatch_approval(
        self,
        decision: RoutingDecision,
        evaluator: PolicyEvaluator | None,
        command_id: str,
        profile: str,
    ) -> None:
        if evaluator is None:
            return
        if not evaluator.require_approval:
            return
        capabilities = self.adapters[decision.provider].status().capabilities
        if "approval" not in capabilities:
            reason = (
                "provider "
                + decision.provider
                + " cannot prompt for the policy-required dispatch approval"
            )
            self._emit_policy_decisions(
                [
                    PolicyDecision(
                        outcome=BLOCK,
                        rule=RULE_APPROVAL_REQUIRED,
                        reason=reason,
                        provider=decision.provider,
                        command_id=command_id,
                    )
                ]
            )
            raise PolicyBlockedError(
                RULE_APPROVAL_REQUIRED,
                reason,
                provider=decision.provider,
            )
        if profile != "interactive":
            reason = (
                "the policy-required dispatch approval needs an interactive "
                "operator; deferring instead of blocking on a human"
            )
            self._emit_policy_decisions(
                [
                    PolicyDecision(
                        outcome=DEFER,
                        rule=RULE_APPROVAL_REQUIRED,
                        reason=reason,
                        provider=decision.provider,
                        command_id=command_id,
                    )
                ]
            )
            raise PolicyDeferredError(
                RULE_APPROVAL_REQUIRED,
                reason,
                provider=decision.provider,
            )
        approved = await self._request_policy_approval(decision, command_id)
        if approved:
            self._emit_policy_decisions(
                [
                    PolicyDecision(
                        outcome=ALLOW,
                        rule=RULE_APPROVAL_REQUIRED,
                        reason="the operator approved the dispatch",
                        provider=decision.provider,
                        command_id=command_id,
                    )
                ]
            )
            return
        reason = "the operator declined the policy-required dispatch approval"
        self._emit_policy_decisions(
            [
                PolicyDecision(
                    outcome=BLOCK,
                    rule=RULE_APPROVAL_REQUIRED,
                    reason=reason,
                    provider=decision.provider,
                    command_id=command_id,
                )
            ]
        )
        raise PolicyBlockedError(
            RULE_APPROVAL_REQUIRED,
            reason,
            provider=decision.provider,
        )

    async def _request_policy_approval(
        self,
        decision: RoutingDecision,
        command_id: str,
    ) -> bool:
        if self._policy_approval_handler is not None:
            return bool(await self._policy_approval_handler(decision))
        approval_id = self.store.create_approval(
            self.session_id,
            "",
            new_uuid(),
            "policy/dispatch",
            "Policy requires approval to dispatch "
            + decision.provider
            + "/"
            + decision.model
            + " at "
            + decision.effort
            + " effort",
            [
                {"id": "accept", "label": "Dispatch"},
                {"id": "decline", "label": "Hold"},
            ],
        )
        self.store.update_session(
            self.session_id,
            attention=Attention.NEEDS_INPUT,
        )
        self.store.append_event(
            self.session_id,
            "approval.requested",
            status="pending",
            metadata={
                "approval_id": approval_id,
                "method": "policy/dispatch",
                "command_id": command_id,
            },
        )
        for unused in range(APPROVAL_POLL_LIMIT):
            del unused
            resolved = self.store.approval_decision(approval_id)
            if resolved is not None:
                self.store.update_session(
                    self.session_id,
                    attention=Attention.WORKING,
                )
                selected = str(
                    resolved.get(
                        "decision",
                        resolved.get("choice", resolved.get("choice_id", "")),
                    )
                ).casefold()
                return selected in {"accept", "allow", "approve", "approved", "yes"}
            await asyncio.sleep(1.0)
        return False

    async def _approval(
        self,
        turn_id: str,
        method: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        choices = request.get("choices")
        if not isinstance(choices, list):
            choices = [
                {"id": "accept", "label": "Accept"},
                {"id": "decline", "label": "Decline"},
            ]
        request_id = str(request.get("id", new_uuid()))
        approval_id = self.store.create_approval(
            self.session_id,
            turn_id,
            request_id,
            method,
            str(request.get("reason", request.get("prompt", method))),
            choices,
        )
        self.store.update_session(
            self.session_id,
            attention=Attention.NEEDS_INPUT,
        )
        self.store.append_event(
            self.session_id,
            "approval.requested",
            status="pending",
            metadata={"approval_id": approval_id, "method": method},
            turn_id=turn_id,
        )
        for unused in range(APPROVAL_POLL_LIMIT):
            del unused
            decision = self.store.approval_decision(approval_id)
            if decision is not None:
                self.store.update_session(
                    self.session_id,
                    attention=Attention.WORKING,
                )
                return decision
            await asyncio.sleep(1.0)
        return {"decision": "decline"}

    async def _control(self, command: CommandReceipt) -> None:
        result: dict[str, Any] = {}
        if command.command_type == "interrupt":
            payload = self.store.command_payload(command.command_id)
            target_command_id = str(payload.get("target_command_id", ""))
            if target_command_id:
                target = None
                try:
                    target = self.store.get_command(target_command_id)
                except NotFoundError:
                    pass
                if (
                    target is None
                    or target.session_id != self.session_id
                    or target.status != CommandStatus.DISPATCHING
                    or self._active_command_id != target_command_id
                ):
                    result = {
                        "code": "E_CONTROL_TARGET",
                        "message": "interrupt target is not the active command",
                    }
                    self.store.resolve_command(
                        command.command_id,
                        CommandStatus.FAILED,
                        result,
                    )
                    return
            if self._active_adapter is not None:
                await self._active_adapter.interrupt()
            generation = self.store.repetition_generation(self.session_id)
            session = self.store.get_session(self.session_id)
            material_digest, unused_summary = inspect_workspace(Path(session.worktree))
            del unused_summary
            result = {
                "target_command_id": target_command_id,
                "checkpoint_id": generation["checkpoint_id"],
                "workspace_material_digest": material_digest,
            }
            self.store.append_event(
                self.session_id,
                "turn.interrupted",
                status="interrupted",
                metadata={
                    "control_command_id": command.command_id,
                    **result,
                },
            )
        elif command.command_type == "steer":
            payload = self.store.command_payload(command.command_id)
            text = str(payload.get("text", ""))
            if self._active_adapter is None:
                result = {
                    "code": "E_NO_ACTIVE_TURN",
                    "message": "there is no active turn to steer",
                }
                self.store.resolve_command(
                    command.command_id,
                    CommandStatus.FAILED,
                    result,
                )
                return
            await self._active_adapter.steer(text)
            self.store.append_event(
                self.session_id,
                "user.steer",
                role="user",
                text=text,
                status="complete",
                metadata={"target_command_id": self._active_command_id},
                turn_id=self._active_turn_id,
            )
        elif command.command_type == "pause":
            self.store.update_session(
                self.session_id,
                lifecycle=Lifecycle.PAUSED,
                attention=Attention.IDLE,
            )
        elif command.command_type == "resume":
            self.store.update_session(
                self.session_id,
                lifecycle=Lifecycle.RUNNING,
                attention=Attention.IDLE,
            )
        elif command.command_type == "stop":
            if self._active_adapter is not None:
                await self._active_adapter.interrupt()
            self.store.update_session(
                self.session_id,
                lifecycle=Lifecycle.STOPPED,
                attention=Attention.IDLE,
            )
            self._stopping = True
        self.store.resolve_command(
            command.command_id,
            CommandStatus.COMPLETE,
            result,
        )

    def _record_recovery(
        self,
        command_id: str,
        stage: int,
        action: str,
        provider: str,
        effort: str,
    ) -> None:
        self.store.update_command_envelope(
            command_id,
            state="recovering",
            recovery_stage=stage,
        )
        self.store.append_event(
            self.session_id,
            "recovery.started",
            status="retrying",
            metadata={
                "command_id": command_id,
                "stage": stage,
                "action": action,
                "provider": provider,
                "effort": effort,
            },
        )

    async def _interrupt_guarded_turn(
        self,
        adapter: ProviderAdapter,
        run_task: asyncio.Task,
    ) -> None:
        try:
            await adapter.interrupt()
        except BaseException:
            pass
        try:
            await asyncio.wait_for(
                asyncio.shield(run_task),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
        except BaseException:
            pass

    async def _guard_checkpoint(
        self,
        session: Session,
        provider: str,
        native_session_id: str,
        context_text: str,
        turn_id: str,
    ) -> None:
        checkpoint = await asyncio.to_thread(
            checkpoint_workspace,
            session,
            self.blobs,
            sequence=self.store.last_sequence(self.session_id),
            provider=provider,
            native_session_id=native_session_id,
            context_text=context_text,
        )
        self.store.add_checkpoint(checkpoint)
        self.store.append_event(
            self.session_id,
            "checkpoint.created",
            status="guard",
            metadata=checkpoint.as_dict(),
            turn_id=turn_id,
        )

    def _compile_context(
        self,
        session: Session,
        limits: SafetyLimits,
        command_id: str,
    ):
        goal = self.store.goal_for_session(self.session_id)
        evidence = []
        if goal is not None:
            evidence = self.store.evidence(goal.goal_id)
        summary = workspace_summary(Path(session.worktree))
        lineage = self.store.fork_lineage(self.session_id)
        inherited_context = ""
        inherited_context_digest = ""
        if lineage:
            inherited_context_digest = str(lineage["source_context_digest"])
            inherited_context = self.blobs.get_text(inherited_context_digest)
        try:
            instruction_sequence = self.store.command_instruction_sequence(
                self.session_id,
                command_id,
            )
        except NotFoundError:
            instruction_sequence = self.store.repair_command_instruction_event(
                self.session_id,
                command_id,
            )
        context_events = self.store.context_events_for_command(
            self.session_id,
            command_id,
            instruction_sequence,
            limit=5000,
        )
        before_sequence = 0
        if context_events:
            before_sequence = context_events[0].sequence - 1
        compacted_history = self.store.context_history_summary(
            self.session_id,
            before_sequence,
        )
        compile_input_tokens = limits.max_context_tokens + limits.max_output_tokens
        minimum_compile_tokens = limits.max_output_tokens + 1_024
        if compile_input_tokens < minimum_compile_tokens:
            compile_input_tokens = minimum_compile_tokens
        return compile_context(
            session,
            context_events,
            goal=goal,
            evidence=evidence,
            instructions=workspace_context_artifacts(Path(session.worktree)),
            workspace_summary=summary,
            max_input_tokens=compile_input_tokens,
            reserve_output_tokens=limits.max_output_tokens,
            total_event_count=self.store.event_count(self.session_id),
            inherited_context=inherited_context,
            inherited_context_digest=inherited_context_digest,
            compacted_history=compacted_history,
            durable_unresolved_decisions=self.store.context_unresolved_decisions(
                self.session_id
            ),
        )

    def _native_session(self, provider: str) -> str:
        for attempt in reversed(self.store.attempts(self.session_id)):
            if attempt.provider != provider:
                continue
            if attempt.native_session_id:
                return attempt.native_session_id
        return ""

    async def _handoff_context(
        self,
        session: Session,
        decision: RoutingDecision,
        limits: SafetyLimits,
        context: CompiledContext,
    ) -> tuple[str, dict[str, Any]]:
        """Harness-generated handoff envelope for a provider switch.

        The envelope is prepended to the compiled context only; it is
        never folded into the operator instruction or the digests the
        repeated-dispatch guard and attestation paths compute.
        """
        attempts = self.store.attempts(self.session_id)
        if attempts:
            source_provider = session.active_provider
            if not source_provider:
                source_provider = attempts[-1].provider
            models = await self.scheduler.models(Path(session.worktree))
            token_budget = handoff_token_budget(
                model_context_window(
                    models.get(decision.provider, ()),
                    decision.model,
                ),
                limits.max_output_tokens,
                context.estimated_tokens,
            )
            transcript = project_transcript(
                self.store,
                self.session_id,
                blobs=self.blobs,
            )
            block = handoff_envelope(
                session_id=self.session_id,
                source_provider=source_provider,
                target_provider=decision.provider,
                target_model=decision.model,
                transcript_digest=transcript.digest,
                rendered=render(transcript, RenderPolicy(token_budget=token_budget)),
            )
            metadata = {
                "schema": HANDOFF_SCHEMA,
                "origin": ORIGIN_PROVIDER_SWITCH,
                "source_provider": source_provider,
                "target_provider": decision.provider,
                "target_model": decision.model,
                "transcript_digest": transcript.digest,
                "handoff_digest": hashlib.sha256(
                    block.encode("utf-8")
                ).hexdigest(),
                "blob_digest": self.blobs.put_text(block),
                "token_budget": token_budget,
                "rendered_tokens": (len(block) + 3) // 4,
            }
            return block + "\n\n", metadata
        seeded = self.store.seeded_handoff(self.session_id)
        if seeded:
            block = self.blobs.get_text(str(seeded["blob_digest"]))
            metadata = {
                "schema": HANDOFF_SCHEMA,
                "origin": ORIGIN_FORK_SEED,
                "source_session_id": str(seeded.get("source_session_id", "")),
                "source_provider": str(seeded.get("source_provider", "")),
                "target_provider": str(seeded.get("target_provider", "")),
                "transcript_digest": str(seeded.get("transcript_digest", "")),
                "handoff_digest": hashlib.sha256(
                    block.encode("utf-8")
                ).hexdigest(),
                "blob_digest": str(seeded["blob_digest"]),
            }
            return block + "\n\n", metadata
        return "", {}

    def _guard_repeated_dispatch(
        self,
        command_id: str,
        instruction: str,
        context_text: str,
        workspace: Path,
        turn_ref: dict[str, Any],
        execution_profile: str,
    ) -> None:
        live_material_digest, unused_summary = inspect_workspace(workspace)
        del unused_summary
        transition = self.store.reserve_dispatch_generation_transition(
            self.session_id,
            command_id,
            turn_ref,
            live_material_digest,
        )
        if transition == "stage-mismatch":
            raise SafetyGuardError(
                "dispatch-transition-stage-mismatch",
                "automatic routing",
            )
        if transition == "epoch-mismatch":
            raise SafetyGuardError(
                "dispatch-transition-epoch-mismatch",
                "automatic routing",
            )
        if transition == "already-consumed":
            raise SafetyGuardError(
                "dispatch-transition-already-consumed",
                "automatic routing",
            )
        if transition == "command-mismatch":
            raise SafetyGuardError(
                "dispatch-transition-command-mismatch",
                "automatic routing",
            )
        if transition == "material-mismatch":
            raise SafetyGuardError(
                "dispatch-transition-material-mismatch",
                "automatic routing",
            )
        generation = self._dispatch_generation(
            workspace,
            material_digest=live_material_digest,
        )
        generation_digest = generation["generation_digest"]
        fixed_context = context_text.split("\n\n## Event ", 1)[0]
        workspace_instruction_text = "\n\n".join(workspace_instructions(workspace))
        components = {
            "instruction_digest": _text_digest(instruction),
            "context_digest": _text_digest(fixed_context),
            "workspace_instruction_digest": _text_digest(workspace_instruction_text),
            "plan_digest": _workspace_artifact_digest(
                workspace,
                ("plans/**/*.md", ".agents/plans/**/*.md"),
            ),
            "skill_digest": _workspace_artifact_digest(
                workspace,
                (
                    "skills/**/SKILL.md",
                    ".agents/skills/**/SKILL.md",
                    ".codex/skills/**/SKILL.md",
                ),
            ),
        }
        step_digest = ""
        if str(turn_ref.get("step_id", "")):
            step_digest = _structured_digest(turn_ref)
        fingerprint = _structured_digest(components)
        for event in reversed(self.store.all_events(self.session_id)):
            if event.event_type != "dispatch.fingerprint":
                continue
            previous_command_id = str(event.metadata.get("command_id", ""))
            if previous_command_id == command_id:
                return
            if self.store.command_failed_before_provider_boundary(previous_command_id):
                continue
            if str(event.metadata.get("generation_digest", "")) != generation_digest:
                continue
            if step_digest:
                if str(event.metadata.get("step_digest", "")) != step_digest:
                    continue
                if str(event.metadata.get("fingerprint", "")) != fingerprint:
                    continue
            repeated_component = ""
            for name, digest in components.items():
                if execution_profile == INTERACTIVE and name != "instruction_digest":
                    continue
                if not digest:
                    continue
                if str(event.metadata.get(name, "")) == digest:
                    repeated_component = name.removesuffix("_digest")
                    break
            if not repeated_component:
                continue
            self.store.append_event(
                self.session_id,
                "guard.tripped",
                status="paused",
                metadata={
                    "command_id": command_id,
                    "reason": "repeated-" + repeated_component,
                    "fingerprint": fingerprint,
                },
            )
            raise SafetyGuardError(
                "repeated-" + repeated_component,
                "automatic routing",
            )
        self.store.append_event(
            self.session_id,
            "dispatch.fingerprint",
            status="reserved",
            metadata={
                "command_id": command_id,
                "fingerprint": fingerprint,
                "step_digest": step_digest,
                **generation,
                **components,
            },
        )

    def _record_tool_result_fingerprint(
        self,
        command_id: str,
        turn_id: str,
        semantic_pair: str,
    ) -> None:
        pair_digest = _text_digest(semantic_pair)
        generation = self._dispatch_generation()
        generation_digest = generation["generation_digest"]
        for event in reversed(self.store.all_events(self.session_id)):
            if event.event_type != "tool.result.fingerprint":
                continue
            if str(event.metadata.get("generation_digest", "")) != generation_digest:
                continue
            if str(event.metadata.get("pair_digest", "")) != pair_digest:
                continue
            self.store.append_event(
                self.session_id,
                "guard.tripped",
                status="paused",
                metadata={
                    "command_id": command_id,
                    "reason": "repeated-tool-result",
                    "pair_digest": pair_digest,
                    **generation,
                },
                turn_id=turn_id,
            )
            raise SafetyGuardError(
                "repeated-tool-result",
                "automatic routing",
            )
        self.store.append_event(
            self.session_id,
            "tool.result.fingerprint",
            status="complete",
            metadata={
                "command_id": command_id,
                "pair_digest": pair_digest,
                **generation,
            },
            turn_id=turn_id,
        )

    def _dispatch_generation(
        self,
        workspace: Path | None = None,
        *,
        material_digest: str = "",
    ) -> dict[str, str]:
        selected_workspace = workspace
        if selected_workspace is None:
            session = self.store.get_session(self.session_id)
            selected_workspace = Path(session.worktree)
        if not material_digest:
            material_digest, unused_summary = inspect_workspace(selected_workspace)
            del unused_summary
        generation = self.store.repetition_generation(self.session_id)
        generation["generation_digest"] = _structured_digest(
            {
                "material_digest": material_digest,
                "invalidation_id": generation["invalidation_id"],
            }
        )
        return generation

    async def _evaluate_goal(self) -> None:
        goal = self.store.goal_for_session(self.session_id)
        if goal is None:
            return
        evidence = self.store.evidence(goal.goal_id)
        milestones = evaluate_milestones(goal, evidence)
        if milestones != goal.milestones:
            goal = self.store.update_milestone_statuses(
                goal.goal_id,
                milestones,
            )
        evaluation = evaluate_goal(
            goal,
            evidence,
        )
        if not evaluation.satisfied:
            budget = self._exhausted_budget()
            if budget:
                self.store.update_goal_status(
                    goal.goal_id,
                    GoalStatus.WAITING,
                )
                self.store.update_session(
                    self.session_id,
                    lifecycle=Lifecycle.PAUSED,
                    attention=Attention.NEEDS_INPUT,
                )
                self.store.append_event(
                    self.session_id,
                    "goal.budget_exhausted",
                    status="paused",
                    metadata={"budget": budget},
                )
            return
        self.store.update_goal_status(goal.goal_id, GoalStatus.COMPLETE)
        self.store.update_session(
            self.session_id,
            lifecycle=Lifecycle.COMPLETED,
            attention=Attention.READY,
        )
        self.store.append_event(
            self.session_id,
            "goal.completed",
            status="complete",
            metadata={"matched": list(evaluation.matched)},
        )

    def _exhausted_budget(self) -> str:
        goal = self.store.goal_for_session(self.session_id)
        if goal is None:
            return ""
        consumption = self._goal_consumption(goal)
        return exhausted_budget(goal, consumption)

    def _goal_consumption(self, goal: Goal) -> GoalConsumption:
        consumptions = [
            envelope["consumption"]
            for envelope in self.store.session_envelopes(self.session_id)
        ]
        return goal_consumption(
            goal,
            [],
            self.store.countable_turn_count(self.session_id),
            safety_consumptions=consumptions,
            active_seconds=self.store.active_turn_seconds(
                self.session_id,
                datetime.datetime.now(datetime.UTC),
            ),
        )

    def _goal_limited_limits(
        self,
        limits: SafetyLimits,
        *,
        metered_budget: float | None,
    ) -> SafetyLimits:
        if metered_budget is not None and metered_budget <= 0:
            raise ValueError("metered budget must be positive")
        goal = self.store.goal_for_session(self.session_id)
        selected = limits
        if goal is not None:
            consumption = self._goal_consumption(goal)
            tokens = goal.budgets.get("tokens")
            if isinstance(tokens, (int, float)) and not isinstance(
                tokens,
                bool,
            ):
                remaining_tokens = max(
                    0,
                    int(float(tokens) - consumption.tokens),
                )
                selected = replace(
                    selected,
                    max_context_tokens=min(
                        selected.max_context_tokens,
                        remaining_tokens,
                    ),
                    max_output_tokens=min(
                        selected.max_output_tokens,
                        remaining_tokens,
                    ),
                    max_total_tokens=min(
                        selected.max_total_tokens,
                        remaining_tokens,
                    ),
                )
            context_tokens = goal.budgets.get("context_tokens")
            if isinstance(context_tokens, (int, float)) and not isinstance(
                context_tokens,
                bool,
            ):
                remaining_context_tokens = max(
                    0,
                    int(float(context_tokens) - consumption.context_tokens),
                )
                selected = replace(
                    selected,
                    max_context_tokens=min(
                        selected.max_context_tokens,
                        remaining_context_tokens,
                    ),
                )
            output_tokens = goal.budgets.get("output_tokens")
            if isinstance(output_tokens, (int, float)) and not isinstance(
                output_tokens,
                bool,
            ):
                remaining_output_tokens = max(
                    0,
                    int(float(output_tokens) - consumption.output_tokens),
                )
                selected = replace(
                    selected,
                    max_output_tokens=min(
                        selected.max_output_tokens,
                        remaining_output_tokens,
                    ),
                )
            for budget_name, limit_name, consumed in (
                ("tool_calls", "max_tool_calls", consumption.tool_calls),
                ("attempts", "max_attempts", consumption.attempts),
                (
                    "child_agents",
                    "max_child_agents",
                    consumption.child_agents,
                ),
            ):
                budget = goal.budgets.get(budget_name)
                if not isinstance(budget, (int, float)):
                    continue
                if isinstance(budget, bool):
                    continue
                remaining = max(0, int(float(budget) - consumed))
                selected = replace(
                    selected,
                    **{
                        limit_name: min(
                            int(getattr(selected, limit_name)),
                            remaining,
                        )
                    },
                )
            seconds = goal.budgets.get("seconds")
            if isinstance(seconds, (int, float)) and not isinstance(
                seconds,
                bool,
            ):
                remaining_seconds = max(
                    0,
                    int(float(seconds) - consumption.elapsed_seconds),
                )
                selected = replace(
                    selected,
                    max_seconds=min(
                        selected.max_seconds,
                        remaining_seconds,
                    ),
                )
            dollars = goal.budgets.get("dollars")
            if isinstance(dollars, (int, float)) and not isinstance(
                dollars,
                bool,
            ):
                remaining_dollars = max(
                    0.0,
                    float(dollars) - consumption.dollars,
                )
                selected = replace(
                    selected,
                    max_dollars=remaining_dollars,
                )
        if metered_budget is not None:
            allowed_dollars = metered_budget
            if selected.max_dollars is not None:
                allowed_dollars = min(
                    allowed_dollars,
                    selected.max_dollars,
                )
            selected = replace(
                selected,
                max_dollars=allowed_dollars,
            )
        return selected


def _text_digest(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _structured_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _workspace_artifact_digest(
    workspace: Path,
    patterns: tuple[str, ...],
) -> str:
    root = workspace.resolve()
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.glob(pattern))
    digest = hashlib.sha256()
    retained = 0
    for path in sorted(paths):
        if retained >= 200_000:
            break
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        content = resolved.read_bytes()[: 200_000 - retained]
        relative = resolved.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        retained += len(content)
    if retained == 0:
        return ""
    return digest.hexdigest()


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("numeric value must be finite")
        return normalized
    return None


def _lease_expiry() -> str:
    value = datetime.datetime.now(datetime.UTC)
    value += datetime.timedelta(seconds=90)
    return value.isoformat()


def _fault_probe_expiry() -> str:
    value = datetime.datetime.now(datetime.UTC)
    value += datetime.timedelta(seconds=30)
    return value.isoformat()
