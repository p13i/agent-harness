"""One durable worker per active harness session."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from agent_harness.blobs import BlobStore
from agent_harness.context import compile_context
from agent_harness.context import workspace_instructions
from agent_harness.errors import HarnessError
from agent_harness.errors import ProviderExhaustedError
from agent_harness.errors import ProviderUnavailableError
from agent_harness.errors import SafetyGuardError
from agent_harness.goals import evaluate_goal
from agent_harness.goals import exhausted_budget
from agent_harness.goals import goal_consumption
from agent_harness.ids import new_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Attention
from agent_harness.models import CommandReceipt
from agent_harness.models import CommandStatus
from agent_harness.models import GoalStatus
from agent_harness.models import Lifecycle
from agent_harness.models import ProviderAttempt
from agent_harness.models import Session
from agent_harness.providers.base import ProviderAdapter
from agent_harness.providers.base import ProviderEvent
from agent_harness.scheduler import Scheduler
from agent_harness.safety import SafetyLimits
from agent_harness.safety import TurnGuard
from agent_harness.safety import INTERACTIVE
from agent_harness.safety import apply_extension
from agent_harness.safety import effective_effort
from agent_harness.safety import limits_for
from agent_harness.safety import lower_effort
from agent_harness.safety import require_state_headroom
from agent_harness.storage import StateStore
from agent_harness.workspace import checkpoint_workspace
from agent_harness.workspace import workspace_summary


CONTROL_COMMANDS = frozenset(
    {"interrupt", "pause", "resume", "stop", "steer"}
)


class SessionWorker:
    def __init__(
        self,
        store: StateStore,
        blobs: BlobStore,
        scheduler: Scheduler,
        adapters: dict[str, ProviderAdapter],
        session_id: str,
    ) -> None:
        self.store = store
        self.blobs = blobs
        self.scheduler = scheduler
        self.adapters = adapters
        self.session_id = session_id
        self.incarnation = new_uuid()
        self._stopping = False
        self._active_adapter: ProviderAdapter | None = None

    async def run(self) -> None:
        self.store.register_worker(
            self.session_id,
            os.getpid(),
            self.incarnation,
        )
        ambiguous = self.store.recover_dispatching(self.session_id)
        if ambiguous:
            self.store.update_session(
                self.session_id,
                attention=Attention.NEEDS_RECONCILIATION,
            )
            self.store.append_event(
                self.session_id,
                "worker.recovered",
                status="needs-reconciliation",
                metadata={"ambiguous_commands": ambiguous},
            )
        try:
            await self._loop()
        finally:
            self.store.remove_worker(self.session_id, self.incarnation)

    async def _loop(self) -> None:
        while not self._stopping:
            if not self.store.heartbeat_worker(
                self.session_id,
                self.incarnation,
            ):
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
            command = self.store.claim_command(self.session_id)
            if command is None:
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

    async def _message(self, command: CommandReceipt) -> None:
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
        workload = str(payload.get("workload", "implementation"))
        limits = limits_for(profile, workload)
        extension = self.store.consume_session_extensions(
            self.session_id
        )
        limits = apply_extension(limits, extension)
        requested_effort = str(payload.get("effort", ""))
        xhigh_authorized = int(
            safety.get("xhigh_authorizations", 0)
        ) > 0
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
        if effort == "xhigh" and profile != "interactive":
            self.store.consume_xhigh_authorization(self.session_id)
        payload = dict(payload)
        payload["effort"] = effort
        envelope = self.store.create_command_envelope(
            command.command_id,
            self.session_id,
            profile,
            limits.as_dict(),
        )
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
        guard = TurnGuard(limits)
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
            )
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
        self.store.update_command_envelope(
            command.command_id,
            state="complete",
            consumption=guard.consumption.as_dict(),
        )
        self.store.resolve_command(
            command.command_id,
            CommandStatus.COMPLETE,
            result,
        )
        await self._evaluate_goal()

    async def _execute_with_failover(
        self,
        command_id: str,
        payload: dict[str, Any],
        text: str,
        guard: TurnGuard,
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
                    enforce_concurrency=attempt_index == 0,
                )
            except SafetyGuardError as error:
                if first_error is None:
                    first_error = error
                if not error.recoverable:
                    raise
                if recovery_stage == 0:
                    lowered = lower_effort(
                        str(attempt_payload.get("effort", ""))
                    )
                    if lowered != str(
                        attempt_payload.get("effort", "")
                    ):
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
                provider = error.detail.message.split(" ", 1)[0]
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
        *,
        enforce_concurrency: bool,
    ) -> dict[str, Any]:
        session = self.store.get_session(self.session_id)
        context = self._compile_context(session, guard.limits)
        metered_budget = _optional_number(payload.get("metered_budget"))
        decision = await self.scheduler.choose(
            session,
            workload=str(payload.get("workload", "implementation")),
            required_capabilities=frozenset(
                str(item)
                for item in payload.get("required_capabilities", [])
            ),
            provider=str(payload.get("provider", "")),
            model=str(payload.get("model", "")),
            effort=str(payload.get("effort", "")),
            metered_budget=metered_budget,
            excluded=excluded,
            context_transfer_tokens=context.estimated_tokens,
            binding_ceiling=guard.limits.binding_ceiling,
            execution_profile=guard.limits.profile,
            enforce_concurrency=enforce_concurrency,
        )
        native_session_id = self._native_session(decision.provider)
        prompt = text
        if not native_session_id or session.active_provider != decision.provider:
            prompt = context.text + "\n\n# Next instruction\n\n" + text
            digest = hashlib.sha256(
                context.text.encode("utf-8")
            ).hexdigest()
            self.store.record_context_delivery(
                self.session_id,
                decision.provider,
                digest,
                "",
            )
        submitted_tokens = (len(prompt) + 3) // 4
        violation = guard.begin_attempt(submitted_tokens)
        if violation:
            raise SafetyGuardError(
                violation,
                decision.provider,
                recoverable=False,
            )
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
            provider=decision.provider,
            state="running",
            recovery_stage=recovery_stage,
            consumption=guard.consumption.as_dict(),
        )
        turn_id = self.store.start_turn(self.session_id, attempt_id)
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
        self.store.append_event(
            self.session_id,
            "routing.selected",
            status="complete",
            metadata=decision.as_dict(),
            turn_id=turn_id,
        )
        adapter = self.adapters[decision.provider]
        self._active_adapter = adapter
        lease_id = ""
        if guard.limits.profile != INTERACTIVE:
            lease = self.store.create_process_lease(
                self.session_id,
                decision.provider,
                guard.limits.profile,
                _lease_expiry(),
            )
            lease_id = str(lease["lease_id"])
            self.store.append_event(
                self.session_id,
                "lease.reserved",
                status="complete",
                metadata={
                    "lease_id": lease_id,
                    "provider": decision.provider,
                },
                turn_id=turn_id,
            )

        resolved_native_session_id = native_session_id

        async def event_handler(event: ProviderEvent) -> None:
            nonlocal resolved_native_session_id
            if (
                event.native_session_id
                and event.native_session_id
                != resolved_native_session_id
            ):
                resolved_native_session_id = event.native_session_id
                self.store.update_attempt(
                    attempt_id,
                    status="running",
                    native_session_id=resolved_native_session_id,
                )
            guard.observe(event)
            if event.event_type.startswith("file.change."):
                guard.note_material_progress()
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

        async def approval_handler(
            method: str,
            request: dict[str, Any],
        ) -> dict[str, Any]:
            return await self._approval(turn_id, method, request)

        run_task = asyncio.create_task(
            adapter.run_turn(
                workspace=Path(session.worktree),
                prompt=prompt,
                native_session_id=native_session_id,
                permission_mode=session.permission_mode,
                model=decision.model,
                effort=decision.effort,
                event_handler=event_handler,
                approval_handler=approval_handler,
            )
        )
        lease_attached = False
        last_lease_heartbeat = time.monotonic()
        try:
            while not run_task.done():
                if lease_id:
                    pid, pid_start = adapter.process_identity()
                    if not lease_attached and pid > 0 and pid_start:
                        self.store.update_process_lease(
                            lease_id,
                            pid=pid,
                            pid_start=pid_start,
                            state="active",
                            expires_at=_lease_expiry(),
                        )
                        lease_attached = True
                        last_lease_heartbeat = time.monotonic()
                        self.store.append_event(
                            self.session_id,
                            "lease.attached",
                            status="complete",
                            metadata={
                                "lease_id": lease_id,
                                "pid": pid,
                            },
                            turn_id=turn_id,
                        )
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
                violation = guard.violation()
                if violation:
                    await self._interrupt_guarded_turn(
                        adapter,
                        run_task,
                    )
                    recoverable = violation in {
                        "repeated-tool",
                        "repeated-cycle",
                        "stagnation",
                    }
                    action = "pause"
                    if recoverable and recovery_stage == 0:
                        action = "downgrade"
                    elif recoverable:
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
            guard.observe(
                ProviderEvent(
                    "usage.updated",
                    metadata=result.usage,
                )
            )
            violation = guard.violation()
            if violation:
                raise SafetyGuardError(
                    violation,
                    decision.provider,
                    recoverable=False,
                )
        except SafetyGuardError:
            self.store.update_attempt(attempt_id, status="interrupted")
            self.store.finish_turn(turn_id, "interrupted")
            raise
        except (ProviderExhaustedError, ProviderUnavailableError):
            self.store.update_attempt(attempt_id, status="exhausted")
            self.store.finish_turn(turn_id, "exhausted")
            raise
        except BaseException:
            self.store.update_attempt(attempt_id, status="failed")
            self.store.finish_turn(turn_id, "failed")
            raise
        finally:
            self._active_adapter = None
            if lease_id:
                self.store.update_process_lease(
                    lease_id,
                    state="released",
                )
                self.store.append_event(
                    self.session_id,
                    "lease.released",
                    status="complete",
                    metadata={"lease_id": lease_id},
                    turn_id=turn_id,
                )
        self.store.update_attempt(
            attempt_id,
            status=result.status,
            native_session_id=result.native_session_id,
        )
        self.store.finish_turn(turn_id, result.status)
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
        self.store.append_event(
            self.session_id,
            "checkpoint.created",
            status="complete",
            metadata=checkpoint.as_dict(),
            turn_id=turn_id,
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
            "usage": result.usage,
            "safety": guard.snapshot(),
        }

    async def _provider_event(
        self,
        turn_id: str,
        event: ProviderEvent,
    ) -> None:
        digest = ""
        if event.raw is not None:
            encoded = json.dumps(
                event.raw,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = self.blobs.put(encoded)
        role = ""
        if event.event_type.startswith("agent."):
            role = "assistant"
        if event.event_type.startswith("user."):
            role = "user"
        metadata = {}
        if event.metadata is not None:
            metadata = event.metadata
        self.store.append_event(
            self.session_id,
            event.event_type,
            role=role,
            text=event.text,
            status=event.status,
            metadata=metadata,
            blob_digest=digest,
            turn_id=turn_id,
        )

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
        for unused in range(3600):
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
            if self._active_adapter is not None:
                await self._active_adapter.interrupt()
            self.store.append_event(
                self.session_id,
                "turn.interrupted",
                status="interrupted",
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
    ):
        goal = self.store.goal_for_session(self.session_id)
        evidence = []
        if goal is not None:
            evidence = self.store.evidence(goal.goal_id)
        summary = workspace_summary(Path(session.worktree))
        return compile_context(
            session,
            self.store.events(self.session_id, limit=5000),
            goal=goal,
            evidence=evidence,
            instructions=workspace_instructions(
                Path(session.worktree)
            ),
            workspace_summary=summary,
            max_input_tokens=(
                limits.max_context_tokens
                + limits.max_output_tokens
            ),
            reserve_output_tokens=limits.max_output_tokens,
        )

    def _native_session(self, provider: str) -> str:
        for attempt in reversed(self.store.attempts(self.session_id)):
            if attempt.provider != provider:
                continue
            if attempt.native_session_id:
                return attempt.native_session_id
        return ""

    async def _evaluate_goal(self) -> None:
        goal = self.store.goal_for_session(self.session_id)
        if goal is None:
            return
        evaluation = evaluate_goal(
            goal,
            self.store.evidence(goal.goal_id),
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
        consumption = goal_consumption(
            goal,
            self.store.completed_command_results(
                self.session_id
            ),
            self.store.turn_count(self.session_id),
        )
        return exhausted_budget(goal, consumption)


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _lease_expiry() -> str:
    value = datetime.datetime.now(datetime.UTC)
    value += datetime.timedelta(seconds=90)
    return value.isoformat()
