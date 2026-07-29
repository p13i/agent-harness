"""Application service shared by HTTP, CLI, and tests."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
import signal
import subprocess
from typing import Any

from agent_harness.blobs import BlobStore
from agent_harness.config import HarnessPaths
from agent_harness.config import host_id
from agent_harness.context import compile_context
from agent_harness.context import workspace_instructions
from agent_harness.goals import create_goal
from agent_harness.goals import make_evidence
from agent_harness.ids import new_uuid
from agent_harness.ids import require_uuid
from agent_harness.ids import utc_now
from agent_harness.models import Attention
from agent_harness.models import CommandStatus
from agent_harness.models import Lifecycle
from agent_harness.models import PermissionMode
from agent_harness.models import Session
from agent_harness.providers.claude import ClaudeAdapter
from agent_harness.providers.codex import CodexAdapter
from agent_harness.projections import write_session_projections
from agent_harness.scheduler import Scheduler
from agent_harness.storage import StateStore
from agent_harness.transfer import load_machine_keys
from agent_harness.transfer import open_transfer
from agent_harness.transfer import seal_transfer
from agent_harness.workspace import create_worktree
from agent_harness.workspace import checkpoint_workspace
from agent_harness.workspace import restore_checkpoint
from agent_harness.workspace import workspace_summary
from agent_harness.runtime import launcher_command


class WorkerManager:
    def __init__(self, paths: HarnessPaths) -> None:
        self.paths = paths
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def ensure(self, session_id: str) -> None:
        process = self._processes.get(session_id)
        if process is not None and process.poll() is None:
            return
        command = [
            *launcher_command(),
            "--state-dir",
            str(self.paths.state_dir),
            "worker",
            session_id,
        ]
        log_path = self.paths.logs / ("worker-" + session_id + ".log")
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
            )
        self._processes[session_id] = process

    def stop_all(self) -> None:
        for process in self._processes.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)


class HarnessService:
    def __init__(
        self,
        paths: HarnessPaths,
        *,
        worker_manager: WorkerManager | None = None,
    ) -> None:
        self.paths = paths
        self.store = StateStore(paths.database)
        self.blobs = BlobStore(paths.blobs)
        self.adapters = {
            "claude": ClaudeAdapter(),
            "codex": CodexAdapter(),
        }
        self.scheduler = Scheduler(self.store, self.adapters)
        self.machine_keys = load_machine_keys(paths.machine_keys)
        self.workers = worker_manager
        if self.workers is None:
            self.workers = WorkerManager(paths)

    def close(self) -> None:
        self.store.close()

    def recover_workers(self) -> None:
        for session in self.store.list_sessions():
            if session.lifecycle not in {
                Lifecycle.STARTING,
                Lifecycle.RUNNING,
            }:
                continue
            self.workers.ensure(session.session_id)

    def create_session(self, payload: dict[str, Any]) -> Session:
        workspace_text = str(payload.get("workspace", "")).strip()
        if not workspace_text:
            raise ValueError("workspace is required")
        workspace = Path(workspace_text).expanduser().resolve()
        direct = bool(payload.get("direct", False))
        session_id = new_uuid()
        worktree = create_worktree(
            workspace,
            self.paths.worktrees,
            session_id,
            direct=direct,
        )
        name = str(payload.get("name", "")).strip()
        if not name:
            name = workspace.name + " " + session_id[:8]
        permission_mode = str(
            payload.get("permission_mode", PermissionMode.APPROVAL)
        )
        if permission_mode not in set(PermissionMode):
            raise ValueError("unsupported permission mode")
        now = utc_now()
        session = Session(
            session_id=session_id,
            name=name,
            workspace=str(workspace),
            worktree=str(worktree),
            lifecycle=Lifecycle.STARTING,
            attention=Attention.IDLE,
            permission_mode=permission_mode,
            active_provider="",
            model=str(payload.get("model", "")),
            effort=str(payload.get("effort", "")),
            goal_id="",
            owner_host=host_id(),
            owner_epoch=1,
            created_at=now,
            updated_at=now,
        )
        self.store.create_session(session)
        objective = str(payload.get("goal", "")).strip()
        if objective:
            predicates = payload.get("predicates", [])
            if not isinstance(predicates, list):
                predicates = []
            budgets = payload.get("budgets", {})
            if not isinstance(budgets, dict):
                budgets = {}
            goal = create_goal(
                session_id,
                objective,
                kind=str(payload.get("goal_kind", "finite")),
                constraints=tuple(
                    str(item) for item in payload.get("constraints", [])
                ),
                predicates=tuple(
                    item for item in predicates if isinstance(item, dict)
                ),
                budgets=budgets,
            )
            self.store.create_goal(goal)
            session = self.store.get_session(session_id)
        self.store.append_event(
            session_id,
            "session.created",
            status="complete",
            metadata={
                "workspace": str(workspace),
                "worktree": str(worktree),
                "direct": direct,
            },
        )
        return session

    def submit_message(
        self,
        session_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("message text is required")
        receipt = self.store.enqueue_command(
            session_id,
            "message",
            payload,
            idempotency_key,
        )
        if receipt.status == CommandStatus.QUEUED:
            self.store.append_event(
                session_id,
                "user.message",
                role="user",
                text=text,
                status="accepted",
                metadata={"command_id": receipt.command_id},
            )
        self.workers.ensure(session_id)
        return receipt.as_dict()

    def configure_session(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> Session:
        require_uuid(session_id, "session_id")
        changes: dict[str, Any] = {}
        if "name" in payload:
            name = str(payload.get("name", "")).strip()
            if not name or len(name) > 200:
                raise ValueError("session name must contain 1 to 200 characters")
            changes["name"] = name
        if "permission_mode" in payload:
            permission_mode = str(payload.get("permission_mode", ""))
            if permission_mode not in set(PermissionMode):
                raise ValueError("unsupported permission mode")
            changes["permission_mode"] = permission_mode
        if not changes:
            raise ValueError("no supported session settings were provided")
        session = self.store.update_session(session_id, **changes)
        self.store.append_event(
            session_id,
            "session.configured",
            status="complete",
            metadata={"fields": sorted(changes)},
        )
        return session

    def ui_state(self, session_id: str) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        self.store.get_session(session_id)
        return self.store.get_ui_state("session:" + session_id)

    def set_ui_state(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        self.store.get_session(session_id)
        allowed = {
            "active_pane",
            "composer",
            "effort",
            "model",
            "provider",
            "theme",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "unsupported UI state fields: "
                + ", ".join(sorted(unknown))
            )
        state: dict[str, Any] = {}
        for name, value in payload.items():
            if not isinstance(value, str):
                raise ValueError("UI state values must be strings")
            limit = 128
            if name == "composer":
                limit = 131_072
            if len(value) > limit:
                raise ValueError("UI state value is too long")
            state[name] = value
        self.store.set_ui_state("session:" + session_id, state)
        return state

    def command(
        self,
        session_id: str,
        command_type: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        receipt = self.store.enqueue_command(
            session_id,
            command_type,
            payload,
            idempotency_key,
        )
        self.workers.ensure(session_id)
        return receipt.as_dict()

    def resolve_approval(
        self,
        session_id: str,
        approval_id: str,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        if not self.store.resolve_approval(approval_id, decision):
            return {"resolved": False}
        self.store.append_event(
            session_id,
            "approval.resolved",
            status="complete",
            metadata={
                "approval_id": approval_id,
                "decision": decision,
            },
        )
        return {"resolved": True}

    def add_evidence(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        goal = self.store.goal_for_session(session_id)
        if goal is None:
            raise ValueError("session has no goal")
        value = payload.get("value", {})
        if not isinstance(value, dict):
            value = {}
        evidence = make_evidence(
            goal.goal_id,
            str(payload.get("type", "")),
            str(payload.get("subject", "")),
            str(payload.get("outcome", "")),
            value,
        )
        self.store.add_evidence(evidence)
        self.store.append_event(
            session_id,
            "goal.evidence",
            status="complete",
            metadata=evidence.as_dict(),
        )
        return evidence.as_dict()

    def export(self, session_id: str) -> Path:
        payload = self.store.export_session(session_id)
        session = self.store.get_session(session_id)
        goal = self.store.goal_for_session(session_id)
        evidence = []
        if goal is not None:
            evidence = self.store.evidence(goal.goal_id)
        events = self.store.all_events(session_id)
        context = compile_context(
            session,
            events,
            goal=goal,
            evidence=evidence,
            instructions=workspace_instructions(
                Path(session.worktree)
            ),
            workspace_summary=workspace_summary(Path(session.worktree)),
        )
        projections = write_session_projections(
            self.paths.exports,
            payload,
            context,
            events,
            goal,
        )
        return projections["export"]

    def checkpoint(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        goal = self.store.goal_for_session(session_id)
        evidence = []
        if goal is not None:
            evidence = self.store.evidence(goal.goal_id)
        events = self.store.all_events(session_id)
        context = compile_context(
            session,
            events,
            goal=goal,
            evidence=evidence,
            instructions=workspace_instructions(
                Path(session.worktree)
            ),
            workspace_summary=workspace_summary(Path(session.worktree)),
        )
        native_session_id = ""
        for attempt in reversed(self.store.attempts(session_id)):
            if attempt.provider != session.active_provider:
                continue
            if attempt.native_session_id:
                native_session_id = attempt.native_session_id
                break
        checkpoint = checkpoint_workspace(
            session,
            self.blobs,
            sequence=self.store.last_sequence(session_id),
            provider=session.active_provider,
            native_session_id=native_session_id,
            context_text=context.text,
        )
        self.store.add_checkpoint(checkpoint)
        self.store.append_event(
            session_id,
            "checkpoint.created",
            status="complete",
            metadata=checkpoint.as_dict(),
        )
        return checkpoint.as_dict()

    def fork_session(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> Session:
        source = self.store.get_session(session_id)
        checkpoint = self.checkpoint(session_id)
        goal = self.store.goal_for_session(session_id)
        create_payload: dict[str, Any] = {
            "workspace": source.worktree,
            "name": str(payload.get("name", "")).strip(),
            "permission_mode": source.permission_mode,
        }
        if not create_payload["name"]:
            create_payload["name"] = source.name + " fork"
        if goal is not None:
            create_payload.update(
                {
                    "goal": goal.objective,
                    "goal_kind": goal.kind,
                    "constraints": list(goal.constraints),
                    "predicates": list(goal.predicates),
                    "budgets": goal.budgets,
                }
            )
        forked = self.create_session(create_payload)
        checkpoints = self.store.checkpoints(session_id)
        if checkpoints:
            restore_checkpoint(
                Path(forked.worktree),
                checkpoints[-1],
                self.blobs,
            )
        self.store.append_event(
            forked.session_id,
            "session.forked",
            status="complete",
            metadata={
                "source_session_id": session_id,
                "source_sequence": self.store.last_sequence(session_id),
                "source_checkpoint_id": str(
                    checkpoint.get("checkpoint_id", "")
                ),
            },
        )
        return self.store.get_session(forked.session_id)

    async def preview_route(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        session = self.store.get_session(session_id)
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
            metered_budget=_optional_number(
                payload.get("metered_budget")
            ),
        )
        return decision.as_dict()

    def public_keys(self) -> dict[str, str]:
        return self.machine_keys.public_bundle()

    def create_transfer(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if session.attention == Attention.WORKING:
            raise ValueError("pause or interrupt the active turn before transfer")
        destination_host = str(payload.get("destination_host", "")).strip()
        destination_public = str(
            payload.get("destination_encryption_public", "")
        ).strip()
        if not destination_host or not destination_public:
            raise ValueError("destination host and encryption key are required")
        exported = self.store.export_session(session_id)
        blob_values: dict[str, str] = {}
        digests = _export_digests(exported)
        for digest in digests:
            blob_values[digest] = base64.b64encode(
                self.blobs.get(digest)
            ).decode("ascii")
        transfer_payload = {
            "schema": "p13i/agent-harness/session-transfer/v1",
            "source_host": host_id(),
            "destination_host": destination_host,
            "owner_epoch": session.owner_epoch + 1,
            "export": exported,
            "blobs": blob_values,
        }
        envelope = seal_transfer(
            transfer_payload,
            destination_encryption_public=destination_public,
            source_signing_private=self.machine_keys.signing_private,
        )
        self.store.update_session(
            session_id,
            lifecycle=Lifecycle.TRANSFERRING,
        )
        return {
            "session_id": session_id,
            "owner_epoch": session.owner_epoch + 1,
            "source_host": host_id(),
            "source_signing_public": self.public_keys()["signing"],
            "envelope": base64.b64encode(envelope).decode("ascii"),
        }

    def import_transfer(self, payload: dict[str, Any]) -> dict[str, Any]:
        envelope_text = str(payload.get("envelope", ""))
        source_signing_public = str(
            payload.get("source_signing_public", "")
        )
        try:
            envelope = base64.b64decode(envelope_text, validate=True)
        except ValueError as error:
            raise ValueError("transfer envelope must be base64") from error
        opened = open_transfer(
            envelope,
            destination_encryption_private=(
                self.machine_keys.encryption_private
            ),
            source_signing_public=source_signing_public,
        )
        if opened.get("schema") != (
            "p13i/agent-harness/session-transfer/v1"
        ):
            raise ValueError("session transfer schema is unsupported")
        if opened.get("destination_host") != host_id():
            raise ValueError("transfer is addressed to another host")
        exported = opened.get("export")
        if not isinstance(exported, dict):
            raise ValueError("transfer export is missing")
        blobs = opened.get("blobs")
        if not isinstance(blobs, dict):
            raise ValueError("transfer blobs are missing")
        for digest, encoded in blobs.items():
            if not isinstance(digest, str) or not isinstance(encoded, str):
                raise ValueError("transfer blob entry is invalid")
            try:
                content = base64.b64decode(encoded, validate=True)
            except ValueError as error:
                raise ValueError("transfer blob is not base64") from error
            if self.blobs.put(content) != digest:
                raise ValueError("transfer blob digest does not match")
        session_value = exported.get("session")
        if not isinstance(session_value, dict):
            raise ValueError("session transfer is missing session state")
        session_id = str(session_value.get("session_id", ""))
        workspace = Path(str(session_value.get("workspace", ""))).resolve()
        worktree = create_worktree(
            workspace,
            self.paths.worktrees,
            session_id,
        )
        owner_epoch = int(opened.get("owner_epoch", 0))
        session = self.store.import_session(
            exported,
            worktree=str(worktree),
            owner_host=host_id(),
            owner_epoch=owner_epoch,
        )
        checkpoints = self.store.checkpoints(session_id)
        if checkpoints:
            restore_checkpoint(worktree, checkpoints[-1], self.blobs)
        self.store.append_event(
            session_id,
            "transfer.imported",
            status="complete",
            metadata={
                "source_host": str(opened.get("source_host", "")),
                "owner_epoch": owner_epoch,
            },
        )
        return {
            "session": session.as_dict(),
            "owner_epoch": owner_epoch,
            "destination_host": host_id(),
        }

    def finalize_transfer(
        self,
        session_id: str,
        destination_host: str,
        owner_epoch: int,
    ) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if owner_epoch <= session.owner_epoch:
            raise ValueError("transfer acknowledgment owner epoch is stale")
        session = self.store.update_session(
            session_id,
            lifecycle=Lifecycle.STOPPED,
            attention=Attention.IDLE,
            owner_host=destination_host,
            owner_epoch=owner_epoch,
        )
        self.store.append_event(
            session_id,
            "transfer.finalized",
            status="complete",
            metadata={
                "destination_host": destination_host,
                "owner_epoch": owner_epoch,
            },
        )
        return session.as_dict()

    async def wait_for_command(
        self,
        command_id: str,
        *,
        timeout: float = 3600.0,
    ) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            receipt = self.store.get_command(command_id)
            if receipt.status in {
                CommandStatus.COMPLETE,
                CommandStatus.FAILED,
                CommandStatus.CANCELLED,
            }:
                return receipt.as_dict()
            if asyncio.get_running_loop().time() >= deadline:
                return receipt.as_dict()
            await asyncio.sleep(0.1)


def _export_digests(exported: dict[str, Any]) -> set[str]:
    digests: set[str] = set()
    events = exported.get("events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            digest = str(event.get("blob_digest", ""))
            if digest:
                digests.add(digest)
    checkpoints = exported.get("checkpoints", [])
    if isinstance(checkpoints, list):
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            for field in (
                "patch_digest",
                "untracked_digest",
                "context_digest",
            ):
                digest = str(checkpoint.get(field, ""))
                if digest:
                    digests.add(digest)
    return digests


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
