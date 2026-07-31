"""Application service shared by HTTP, CLI, and tests."""

from __future__ import annotations

import asyncio
import base64
import datetime
from pathlib import Path
import signal
import subprocess
import threading
from typing import Any

from agent_harness.blobs import BlobStore
from agent_harness.config import HarnessPaths
from agent_harness.config import host_id
from agent_harness.context import compile_context
from agent_harness.context import workspace_instructions
from agent_harness.errors import ConflictError
from agent_harness.errors import HarnessError
from agent_harness.errors import SafetyGuardError
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
from agent_harness.orchestration import normalize_external_ref
from agent_harness.presentation import checkpoint_diff
from agent_harness.presentation import session_turn
from agent_harness.presentation import session_turns
from agent_harness.providers.claude import ClaudeAdapter
from agent_harness.providers.codex import CodexAdapter
from agent_harness.projections import write_session_projections
from agent_harness.reconciliation import ReconciliationManager
from agent_harness.scheduler import Scheduler
from agent_harness.safety import UNATTENDED
from agent_harness.safety import effective_effort
from agent_harness.safety import limits_for
from agent_harness.safety import require_state_headroom
from agent_harness.safety import validate_profile
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
        self.reconciliations = ReconciliationManager(
            self.store,
            self.blobs,
        )
        self.machine_keys = load_machine_keys(paths.machine_keys)
        self._session_creation_lock = threading.Lock()
        self._reconciliation_resolution_lock = asyncio.Lock()
        self.workers = worker_manager
        if self.workers is None:
            self.workers = WorkerManager(paths)

    def close(self) -> None:
        self.store.close()

    def recover_workers(self) -> None:
        for session in self.store.list_sessions():
            session = self._name_session_from_history(session)
            if session.lifecycle not in {
                Lifecycle.PAUSED,
                Lifecycle.STARTING,
                Lifecycle.RUNNING,
            }:
                continue
            self.workers.ensure(session.session_id)

    def _name_session_from_history(self, session: Session) -> Session:
        if not _automatic_session_name(session):
            return session
        for event in self.store.events(session.session_id, limit=100):
            if event.event_type != "user.message" or not event.text.strip():
                continue
            return self.store.update_session(
                session.session_id,
                name=_message_session_name(event.text),
            )
        return session

    def create_session(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str = "",
    ) -> Session:
        with self._session_creation_lock:
            return self._create_session(
                payload,
                idempotency_key=idempotency_key,
            )

    def _create_session(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> Session:
        workspace_text = str(payload.get("workspace", "")).strip()
        if not workspace_text:
            raise ValueError("workspace is required")
        workspace = Path(workspace_text).expanduser().resolve()
        direct = bool(payload.get("direct", False))
        requested_name = str(payload.get("name", "")).strip()
        permission_mode = str(
            payload.get("permission_mode", PermissionMode.APPROVAL)
        )
        if permission_mode not in set(PermissionMode):
            raise ValueError("unsupported permission mode")
        execution_profile = validate_profile(
            str(payload.get("execution_profile", UNATTENDED))
        )
        external_ref = normalize_external_ref(
            payload.get("external_ref")
        )
        objective = str(payload.get("goal", "")).strip()
        predicates = payload.get("predicates", [])
        if not isinstance(predicates, list):
            raise ValueError("predicates must be an array")
        constraints = payload.get("constraints", [])
        if not isinstance(constraints, list):
            raise ValueError("constraints must be an array")
        budgets = payload.get("budgets", {})
        if not isinstance(budgets, dict):
            raise ValueError("budgets must be an object")
        creation_input: dict[str, Any] = {
            "workspace": str(workspace),
            "name": requested_name,
            "permission_mode": permission_mode,
            "execution_profile": execution_profile,
            "direct": direct,
            "external_ref": external_ref,
            "routing": {
                "model": str(payload.get("model", "")),
                "effort": str(payload.get("effort", "")),
            },
        }
        if objective:
            creation_input["goal"] = {
                "objective": objective,
                "kind": str(payload.get("goal_kind", "finite")),
                "constraints": [str(item) for item in constraints],
                "predicates": [
                    item for item in predicates if isinstance(item, dict)
                ],
                "budgets": budgets,
            }
        existing = self.store.existing_ensured_session(
            creation_input,
            idempotency_key=idempotency_key,
            external_ref=external_ref,
        )
        if existing is not None:
            return existing
        inherited_ui_state = self._workspace_ui_state(str(workspace))
        session_id = new_uuid()
        worktree = create_worktree(
            workspace,
            self.paths.worktrees,
            session_id,
            direct=direct,
        )
        name = requested_name
        if not name:
            name = workspace.name + " " + session_id[:8]
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
            external_ref=external_ref,
        )
        session, created = self.store.ensure_session(
            session,
            creation_input,
            idempotency_key=idempotency_key,
        )
        if not created:
            return session
        if inherited_ui_state:
            self.store.set_ui_state(
                "session:" + session_id,
                inherited_ui_state,
            )
        self.store.set_session_safety(session_id, execution_profile)
        if objective:
            goal = create_goal(
                session_id,
                objective,
                kind=str(payload.get("goal_kind", "finite")),
                constraints=tuple(
                    str(item) for item in constraints
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
                "execution_profile": execution_profile,
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
        provider = str(payload.get("provider", "")).strip()
        if not provider:
            provider = "automatic-route"
        require_state_headroom(self.paths.state_dir, provider)
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("message text is required")
        receipt, created = self.store.ensure_command(
            session_id,
            "message",
            payload,
            idempotency_key,
        )
        if created:
            session = self.store.get_session(session_id)
            if _automatic_session_name(session):
                self.store.update_session(
                    session_id,
                    name=_message_session_name(text),
                )
            self.store.append_event(
                session_id,
                "user.message",
                role="user",
                text=text,
                status="accepted",
                metadata={
                    "command_id": receipt.command_id,
                    "turn_ref": receipt.turn_ref,
                },
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
        if "execution_profile" in payload:
            profile = validate_profile(
                str(payload.get("execution_profile", ""))
            )
            self.store.set_session_safety(session_id, profile)
        if not changes:
            if "execution_profile" not in payload:
                raise ValueError(
                    "no supported session settings were provided"
                )
        session = self.store.update_session(session_id, **changes)
        configured_fields = sorted(changes)
        if "execution_profile" in payload:
            configured_fields.append("execution_profile")
        self.store.append_event(
            session_id,
            "session.configured",
            status="complete",
            metadata={"fields": configured_fields},
        )
        return session

    def set_session_archived(
        self,
        session_id: str,
        archived: bool,
    ) -> Session:
        require_uuid(session_id, "session_id")
        session = self.store.get_session(session_id)
        if session.archived == archived:
            return session
        session = self.store.update_session(
            session_id,
            archived=archived,
        )
        event_type = "session.archived"
        if not archived:
            event_type = "session.unarchived"
        self.store.append_event(
            session_id,
            event_type,
            status="complete",
            metadata={"archived": archived},
        )
        return session

    def pending_reconciliations(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        require_uuid(session_id, "session_id")
        self.store.get_session(session_id)
        return [
            item.as_dict()
            for item in self.store.pending_reconciliations(session_id)
        ]

    def turns(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        return session_turns(
            self.store,
            session_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    def turn(
        self,
        session_id: str,
        turn_id: str,
    ) -> dict[str, Any]:
        return session_turn(self.store, session_id, turn_id)

    def checkpoint_diff(
        self,
        session_id: str,
        checkpoint_id: str,
        *,
        start_line: int = 0,
        limit: int = 400,
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        require_uuid(checkpoint_id, "checkpoint_id")
        self.store.get_session(session_id)
        checkpoint = self.store.checkpoint(checkpoint_id)
        if checkpoint.session_id != session_id:
            raise ValueError("checkpoint does not belong to the session")
        return checkpoint_diff(
            checkpoint,
            self.blobs,
            start_line=start_line,
            limit=limit,
        )

    def inspect_reconciliation(
        self,
        reconciliation_id: str,
    ) -> dict[str, Any]:
        require_uuid(reconciliation_id, "reconciliation_id")
        return self.reconciliations.inspect(
            reconciliation_id
        ).as_dict()

    async def resolve_reconciliation(
        self,
        reconciliation_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._reconciliation_resolution_lock:
            return await self._resolve_reconciliation(
                reconciliation_id,
                payload,
            )

    async def _resolve_reconciliation(
        self,
        reconciliation_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        require_uuid(reconciliation_id, "reconciliation_id")
        decision = str(payload.get("decision", "")).strip()
        observed_digest = str(
            payload.get("observed_workspace_digest", "")
        ).strip()
        if not observed_digest:
            raise ValueError("observed workspace digest is required")
        audit = payload.get("audit")
        if audit is not None and not isinstance(audit, dict):
            raise ValueError("reconciliation audit must be an object")
        record = self.reconciliations.inspect(reconciliation_id)
        approved = self._reconciliation_restore_approved(
            record.session_id,
            reconciliation_id,
            decision,
            str(payload.get("approval_id", "")).strip(),
        )
        record = await self.reconciliations.resolve(
            reconciliation_id,
            decision,
            observed_digest,
            audit=audit,
            approved=approved,
        )
        self.store.append_event(
            record.session_id,
            "reconciliation.resolved",
            status=record.status,
            metadata={
                "reconciliation_id": record.reconciliation_id,
                "command_id": record.command_id,
                "decision": record.resolution,
                "workspace_digest": (
                    record.current_workspace_digest
                ),
            },
        )
        if record.resolution != "stop":
            self.workers.ensure(record.session_id)
        return record.as_dict()

    def _reconciliation_restore_approved(
        self,
        session_id: str,
        reconciliation_id: str,
        decision: str,
        approval_id: str,
    ) -> bool:
        if decision != "restore-pre-turn":
            return False
        session = self.store.get_session(session_id)
        if session.permission_mode != PermissionMode.APPROVAL:
            return False
        if not approval_id:
            approval_id = self._pending_reconciliation_approval(
                session_id,
                reconciliation_id,
            )
            if not approval_id:
                approval_id = self.store.create_approval(
                    session_id,
                    "",
                    reconciliation_id,
                    "reconciliation.restore",
                    "Restore the exact pre-turn workspace checkpoint?",
                    [
                        {"id": "approve", "label": "Restore"},
                        {"id": "decline", "label": "Keep current"},
                    ],
                )
                self.store.append_event(
                    session_id,
                    "approval.requested",
                    status="pending",
                    metadata={
                        "approval_id": approval_id,
                        "method": "reconciliation.restore",
                        "reconciliation_id": reconciliation_id,
                    },
                )
            raise HarnessError(
                "E_APPROVAL_REQUIRED",
                "reconciliation restore requires approval "
                + approval_id,
                status=409,
            )
        approval = self.store.approval(approval_id)
        if (
            approval["session_id"] != session_id
            or approval["provider_request_id"] != reconciliation_id
            or approval["kind"] != "reconciliation.restore"
        ):
            raise ConflictError(
                "approval does not authorize this reconciliation"
            )
        if approval["status"] != "resolved":
            raise HarnessError(
                "E_APPROVAL_REQUIRED",
                "reconciliation restore approval is pending",
                status=409,
            )
        approval_decision = approval.get("decision")
        if not isinstance(approval_decision, dict):
            raise ConflictError("approval decision is invalid")
        if approval_decision.get("decision") != "approve":
            raise HarnessError(
                "E_APPROVAL_DECLINED",
                "reconciliation restore was not approved",
                status=409,
            )
        return True

    def _pending_reconciliation_approval(
        self,
        session_id: str,
        reconciliation_id: str,
    ) -> str:
        for approval in self.store.pending_approvals(session_id):
            if (
                approval["provider_request_id"] == reconciliation_id
                and approval["kind"] == "reconciliation.restore"
            ):
                return str(approval["approval_id"])
        return ""

    def safety_state(self, session_id: str) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        self.store.get_session(session_id)
        return {
            "session": self.store.session_safety(session_id),
            "envelopes": self.store.session_envelopes(session_id),
            "incidents": self.store.guard_incidents(session_id),
        }

    def extend_budget(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise ValueError("budget extension reason is required")
        extension: dict[str, Any] = {"reason": reason}
        for name in ("additional_seconds", "additional_tokens"):
            value = payload.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(name + " must be an integer")
            if value < 1:
                raise ValueError(name + " must be positive")
            maximum = 3_600
            if name == "additional_tokens":
                maximum = 300_000
            if value > maximum:
                raise ValueError(name + " exceeds one profile envelope")
            extension[name] = value
        allow_xhigh = payload.get("allow_xhigh_once", False)
        if not isinstance(allow_xhigh, bool):
            raise ValueError("allow_xhigh_once must be boolean")
        extension["allow_xhigh_once"] = allow_xhigh
        if len(extension) == 2 and not allow_xhigh:
            raise ValueError("budget extension has no additive capacity")
        result = self.store.extend_session_safety(
            session_id,
            extension,
        )
        self.store.append_event(
            session_id,
            "budget.extended",
            status="complete",
            metadata={
                "additional_seconds": extension.get(
                    "additional_seconds",
                    0,
                ),
                "additional_tokens": extension.get(
                    "additional_tokens",
                    0,
                ),
                "allow_xhigh_once": allow_xhigh,
                "reason": reason,
            },
        )
        return result

    def create_process_lease(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        provider = str(payload.get("provider", "")).strip()
        if provider not in {"claude", "codex"}:
            raise ValueError("unsupported lease provider")
        profile = validate_profile(
            str(payload.get("execution_profile", UNATTENDED))
        )
        if profile == "interactive":
            raise ValueError("background leases cannot be interactive")
        session_id = str(payload.get("session_id", "")).strip()
        if session_id:
            require_uuid(session_id, "session_id")
            self.store.get_session(session_id)
        self._require_process_lease_capacity(provider, profile)
        return self.store.create_process_lease(
            session_id,
            provider,
            profile,
            _lease_expiry(),
        )

    def _require_process_lease_capacity(
        self,
        provider: str,
        profile: str,
    ) -> None:
        usage = self.store.latest_usage().get(provider)
        if usage is None:
            raise SafetyGuardError(
                "fresh provider usage is required",
                provider,
                recoverable=False,
            )
        observed_at = str(usage.get("observed_at", ""))
        try:
            observed = datetime.datetime.fromisoformat(observed_at)
        except ValueError as error:
            raise SafetyGuardError(
                "provider usage timestamp is invalid",
                provider,
                recoverable=False,
            ) from error
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=datetime.UTC)
        age = datetime.datetime.now(datetime.UTC) - observed
        if age > datetime.timedelta(seconds=90):
            raise SafetyGuardError(
                "provider usage is stale",
                provider,
                recoverable=False,
            )
        binding = _optional_number(usage.get("binding_percent"))
        if binding is None:
            raise SafetyGuardError(
                "provider binding usage is unavailable",
                provider,
                recoverable=False,
            )
        if bool(usage.get("credits_engaged", False)):
            raise SafetyGuardError(
                "metered provider credits would engage",
                provider,
                recoverable=False,
            )
        ceiling = limits_for(profile, "operations").binding_ceiling
        if binding >= ceiling:
            raise SafetyGuardError(
                "provider binding usage reached the safety ceiling",
                provider,
                recoverable=False,
            )

    def update_process_lease(
        self,
        lease_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        action = str(payload.get("action", "heartbeat"))
        if action == "release":
            return self.store.update_process_lease(
                lease_id,
                state="released",
            )
        if action not in {"attach", "heartbeat"}:
            raise ValueError("unsupported lease action")
        pid: int | None = None
        pid_start: str | None = None
        state: str | None = None
        if action == "attach":
            raw_pid = payload.get("pid")
            if isinstance(raw_pid, bool) or not isinstance(raw_pid, int):
                raise ValueError("lease pid must be an integer")
            if raw_pid < 1:
                raise ValueError("lease pid must be positive")
            pid = raw_pid
            pid_start = str(payload.get("pid_start", "")).strip()
            if not pid_start:
                raise ValueError("lease pid_start is required")
            state = "active"
        return self.store.update_process_lease(
            lease_id,
            pid=pid,
            pid_start=pid_start,
            state=state,
            expires_at=_lease_expiry(),
        )

    def process_leases(self) -> list[dict[str, Any]]:
        return self.store.active_process_leases()

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
            "active_drawer",
            "active_pane",
            "composer_cursor",
            "composer",
            "control_detail_tab",
            "effort",
            "events",
            "expanded_blocks",
            "inspector_tab",
            "last_notification_sequence",
            "model",
            "provider",
            "request_id",
            "selected_turn_id",
            "session_filter",
            "session_query",
            "sidebar_width",
            "theme",
            "workspace_mode",
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
            if name in {"composer", "expanded_blocks"}:
                limit = 131_072
            if len(value) > limit:
                raise ValueError("UI state value is too long")
            state[name] = value
        self.store.set_ui_state("session:" + session_id, state)
        return state

    def _workspace_ui_state(self, workspace: str) -> dict[str, Any]:
        inherited_fields = {
            "events",
            "session_filter",
            "session_query",
            "sidebar_width",
            "theme",
            "workspace_mode",
        }
        for session in self.store.list_sessions():
            if session.workspace != workspace:
                continue
            state = self.store.get_ui_state(
                "session:" + session.session_id
            )
            return {
                name: value
                for name, value in state.items()
                if name in inherited_fields
            }
        return {}

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
        source_profile = str(
            self.store.session_safety(session_id)["profile"]
        )
        if not source_profile:
            source_profile = UNATTENDED
        create_payload: dict[str, Any] = {
            "workspace": source.worktree,
            "name": str(payload.get("name", "")).strip(),
            "permission_mode": source.permission_mode,
            "execution_profile": source_profile,
        }
        if "external_ref" in payload:
            create_payload["external_ref"] = payload["external_ref"]
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
        workload = str(payload.get("workload", "implementation"))
        safety = self.store.session_safety(session_id)
        profile = str(safety["profile"])
        if not profile:
            profile = UNATTENDED
        limits = limits_for(profile, workload)
        effort = effective_effort(
            str(payload.get("effort", "")),
            limits,
            xhigh_authorized=int(
                safety["xhigh_authorizations"]
            )
            > 0,
        )
        decision = await self.scheduler.choose(
            session,
            workload=workload,
            required_capabilities=frozenset(
                str(item)
                for item in payload.get("required_capabilities", [])
            ),
            provider=str(payload.get("provider", "")),
            model=str(payload.get("model", "")),
            effort=effort,
            metered_budget=_optional_number(
                payload.get("metered_budget")
            ),
            binding_ceiling=limits.binding_ceiling,
            execution_profile=profile,
            enforce_concurrency=True,
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
        imported_safety = exported.get("safety")
        if not isinstance(imported_safety, dict):
            imported_safety = {}
        profile = str(imported_safety.get("profile", UNATTENDED))
        self.store.set_session_safety(
            session_id,
            validate_profile(profile),
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


def _automatic_session_name(session: Session) -> bool:
    expected = Path(session.workspace).name + " " + session.session_id[:8]
    return session.name == expected


def _message_session_name(text: str) -> str:
    value = " ".join(text.split())
    maximum = 72
    if len(value) <= maximum:
        return value
    return value[: maximum - 1].rstrip() + "…"


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
