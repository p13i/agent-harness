"""One durable worker per active harness session."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from agent_harness.blobs import BlobStore
from agent_harness.context import compile_context
from agent_harness.context import workspace_instructions
from agent_harness.errors import HarnessError
from agent_harness.errors import ProviderExhaustedError
from agent_harness.errors import ProviderUnavailableError
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
from agent_harness.storage import StateStore
from agent_harness.workspace import checkpoint_workspace
from agent_harness.workspace import workspace_summary


CONTROL_COMMANDS = frozenset({"interrupt", "pause", "stop", "steer"})


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
            if command.command_type == "resume":
                self.store.update_session(
                    self.session_id,
                    lifecycle=Lifecycle.RUNNING,
                    attention=Attention.IDLE,
                )
                self.store.resolve_command(
                    command.command_id,
                    CommandStatus.COMPLETE,
                    {},
                )
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
        self.store.update_session(
            self.session_id,
            lifecycle=Lifecycle.RUNNING,
            attention=Attention.WORKING,
        )
        try:
            result = await self._execute_with_failover(payload, text)
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
                attention=Attention.FAILED,
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
        self.store.resolve_command(
            command.command_id,
            CommandStatus.COMPLETE,
            result,
        )
        await self._evaluate_goal()

    async def _execute_with_failover(
        self,
        payload: dict[str, Any],
        text: str,
    ) -> dict[str, Any]:
        excluded: set[str] = set()
        first_error: HarnessError | None = None
        for unused in range(2):
            del unused
            try:
                return await self._execute_attempt(
                    payload,
                    text,
                    frozenset(excluded),
                )
            except (ProviderExhaustedError, ProviderUnavailableError) as error:
                if first_error is None:
                    first_error = error
                provider = error.detail.message.split(" ", 1)[0]
                excluded.add(provider)
                self.store.append_event(
                    self.session_id,
                    "routing.failover",
                    status="retrying",
                    metadata={
                        "excluded_provider": provider,
                        "reason": error.detail.code,
                    },
                )
        if first_error is not None:
            raise first_error
        raise ProviderUnavailableError("all providers")

    async def _execute_attempt(
        self,
        payload: dict[str, Any],
        text: str,
        excluded: frozenset[str],
    ) -> dict[str, Any]:
        session = self.store.get_session(self.session_id)
        context = self._compile_context(session)
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
        )
        native_session_id = self._native_session(decision.provider)
        prompt = text
        if not native_session_id or session.active_provider != decision.provider:
            prompt = context.text + "\n\n# Next instruction\n\n" + text
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

        async def event_handler(event: ProviderEvent) -> None:
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
        try:
            while not run_task.done():
                control = self.store.claim_command(
                    self.session_id,
                    CONTROL_COMMANDS,
                )
                if control is not None:
                    await self._control(control)
                await asyncio.sleep(0.1)
            result = await run_task
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

    def _compile_context(self, session: Session):
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
