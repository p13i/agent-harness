"""Application service shared by HTTP, CLI, and tests."""

from __future__ import annotations

import asyncio
import base64
import datetime
import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_harness.blobs import BlobStore
from agent_harness.config import HarnessPaths, host_id
from agent_harness.context import compile_context, workspace_instructions
from agent_harness.errors import ConflictError, HarnessError, SafetyGuardError
from agent_harness.goals import (
    SUPPORTED_EFFORTS,
    create_goal,
    evaluate_goal,
    evaluate_milestones,
    goal_contract_digest,
    make_evidence,
    promoted_milestones,
    promoted_predicates,
)
from agent_harness.ids import new_uuid, require_uuid, utc_now
from agent_harness.models import (
    Attention,
    CommandStatus,
    Goal,
    GoalStatus,
    Lifecycle,
    PermissionMode,
    ReconciliationStatus,
    Session,
)
from agent_harness.orchestration import (
    creation_digest,
    normalize_external_ref,
    normalize_turn_ref,
    normalized_digest,
)
from agent_harness.presentation import checkpoint_diff, session_turn, session_turns
from agent_harness.projections import write_session_projections
from agent_harness.proof import proof_snapshot
from agent_harness.providers.claude import ClaudeAdapter
from agent_harness.providers.codex import CodexAdapter
from agent_harness.providers.grok import GrokAdapter
from agent_harness.providers.kimi import KimiAdapter
from agent_harness.reconciliation import (
    ReconciliationManager,
    validate_reconciliation_audit,
)
from agent_harness.handoff import (
    HANDOFF_SCHEMA,
    ORIGIN_FORK_SEED,
    handoff_envelope,
    handoff_token_budget,
    model_context_window,
)
from agent_harness.timeline import project_timeline, render_timeline
from agent_harness.transcript import (
    RenderPolicy,
    project_transcript,
    render,
    validate_render_policy,
)
from agent_harness.runtime import launcher_command
from agent_harness.safety import (
    UNATTENDED,
    effective_effort,
    limits_for,
    require_state_headroom,
    validate_profile,
)
from agent_harness.scheduler import Scheduler
from agent_harness.storage import (
    DISPATCH_TRANSITION_ANCHOR_KINDS,
    STOPPED_SESSION_COMMANDS,
    StateStore,
)
from agent_harness.transfer import load_machine_keys, open_transfer, seal_transfer
from agent_harness.workspace import (
    checkpoint_workspace,
    create_worktree,
    remove_worktree,
    restore_checkpoint,
    workspace_summary,
)

CONTROL_COMMAND_TYPES = frozenset({"interrupt", "pause", "resume", "steer", "stop"})
MAX_STEER_TEXT_LENGTH = 65_536
# Consecutive supervision ticks reporting an unreachable workspace
# before the service parks the session (attention=needs-input plus one
# session.unreachable event) and stops scheduling a worker for it.
UNREACHABLE_PARK_TICKS = 3


class WorkerManager:
    _RESTART_LIMIT = 3
    _RESTART_WINDOW_SECONDS = 60.0

    def __init__(self, paths: HarnessPaths) -> None:
        self.paths = paths
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._restart_history: dict[str, list[float]] = {}
        self._unreachable_ticks: dict[str, int] = {}
        self._last_supervision: dict[str, Any] = {
            "status": "initializing",
            "expected_sessions": [],
            "running_sessions": [],
            "restarted_sessions": [],
            "unrecovered": [],
        }

    def ensure(self, session_id: str, *, force: bool = False) -> None:
        process = self._processes.get(session_id)
        live = process is not None and process.poll() is None
        if live and not force:
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

    def supervise(self, sessions: list[tuple[str, str]]) -> dict[str, Any]:
        now = time.monotonic()
        expected_pairs = sorted(set(sessions))
        expected = [pair[0] for pair in expected_pairs]
        for stale in set(self._unreachable_ticks) - set(expected):
            self._unreachable_ticks.pop(stale, None)
        running: list[str] = []
        restarted: list[str] = []
        unrecovered: list[dict[str, Any]] = []
        for session_id, workspace in expected_pairs:
            process = self._processes.get(session_id)
            if process is not None and process.poll() is None:
                running.append(session_id)
                self._unreachable_ticks.pop(session_id, None)
                continue
            restarting = process is not None
            if process is not None:
                self._processes.pop(session_id, None)
            if not _workspace_reachable(workspace):
                ticks = self._unreachable_ticks.get(session_id, 0) + 1
                self._unreachable_ticks[session_id] = ticks
                unrecovered.append(
                    {
                        "session_id": session_id,
                        "reason": "workspace-unreachable",
                        "workspace": workspace,
                        "consecutive_unreachable": ticks,
                    }
                )
                continue
            self._unreachable_ticks.pop(session_id, None)
            history = self._restart_history.get(session_id, [])
            history = [
                observed_at
                for observed_at in history
                if now - observed_at <= self._RESTART_WINDOW_SECONDS
            ]
            self._restart_history[session_id] = history
            if len(history) >= self._RESTART_LIMIT:
                unrecovered.append(
                    {
                        "session_id": session_id,
                        "reason": "worker-restart-limit",
                        "recent_exits": len(history),
                    }
                )
                continue
            history.append(now)
            try:
                self.ensure(session_id)
            except (OSError, RuntimeError) as error:
                unrecovered.append(
                    {
                        "session_id": session_id,
                        "reason": "worker-spawn-failed",
                        "detail": str(error),
                    }
                )
                continue
            running.append(session_id)
            if restarting:
                restarted.append(session_id)
        status = "ok"
        if unrecovered:
            status = "failed"
        self._last_supervision = {
            "status": status,
            "expected_sessions": expected,
            "running_sessions": running,
            "restarted_sessions": restarted,
            "unrecovered": unrecovered,
        }
        return dict(self._last_supervision)

    def status(self) -> dict[str, Any]:
        return dict(self._last_supervision)

    def stop_all(self) -> None:
        for process in self._processes.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)


def _reserve_creation_intent(
    harness_paths: HarnessPaths,
    idempotency_key: str,
    creation_input: dict[str, Any],
    preferred_session_id: str,
) -> tuple[str, Path | None]:
    if not idempotency_key:
        return preferred_session_id, None
    intent_path = _creation_intent_path(harness_paths, idempotency_key)
    request_digest = creation_digest(creation_input)
    payload = {
        "schema": "p13i/agent-harness/session-creation-intent/v1",
        "idempotency_key_digest": hashlib.sha256(
            idempotency_key.encode("utf-8")
        ).hexdigest(),
        "request_digest": request_digest,
        "session_id": preferred_session_id,
        "created_at": utc_now(),
    }
    intent_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            intent_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError:
        existing = _read_creation_intent(intent_path)
        if str(existing.get("request_digest", "")) != request_digest:
            raise ConflictError("session creation idempotency key was reused")
        session_id = str(existing.get("session_id", ""))
        require_uuid(session_id, "creation intent session_id")
        return session_id, intent_path
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(intent_path.parent)
    except BaseException:
        intent_path.unlink(missing_ok=True)
        raise
    return preferred_session_id, intent_path


def _complete_creation_intent(
    harness_paths: HarnessPaths,
    idempotency_key: str,
    creation_input: dict[str, Any],
    session_id: str,
) -> None:
    if not idempotency_key:
        return
    intent_path = _creation_intent_path(harness_paths, idempotency_key)
    if not intent_path.is_file():
        return
    payload = _read_creation_intent(intent_path)
    if str(payload.get("request_digest", "")) != creation_digest(creation_input):
        raise ConflictError("session creation idempotency key was reused")
    if str(payload.get("session_id", "")) != session_id:
        raise ConflictError("session creation intent identity changed")
    _remove_creation_intent(intent_path)


def _creation_intent_path(
    harness_paths: HarnessPaths,
    idempotency_key: str,
) -> Path:
    key_digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return harness_paths.runtime / "creation-intents" / (key_digest + ".json")


def _read_creation_intent(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or path.stat().st_size > 64 * 1024:
            raise ValueError("session creation intent is invalid")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConflictError("session creation intent is unreadable") from error
    if not isinstance(payload, dict):
        raise ConflictError("session creation intent is invalid")
    if payload.get("schema") != "p13i/agent-harness/session-creation-intent/v1":
        raise ConflictError("session creation intent is incompatible")
    return payload


def _remove_creation_intent(path: Path | None) -> None:
    if path is None:
        return
    path.unlink(missing_ok=True)
    _sync_directory(path.parent)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_proof_fault_probe(
    session: Session,
    goal: Goal | None,
    payload: dict[str, Any],
    idempotency_key: str,
    supported_providers: frozenset[str],
) -> None:
    probe = payload.get("proof_fault_probe")
    if probe is None:
        return
    if not isinstance(probe, dict):
        raise ValueError("proof_fault_probe must be an object")
    if not idempotency_key:
        raise ValueError("proof fault probe requires an idempotency key")
    if str(payload.get("provider", "")):
        raise ValueError("proof fault probe requires automatic routing")
    permitted_orchestrators = {
        "p13i/machines/agent-harness-proof",
        "p13i/machines/agent-harness-proof-builder",
        "p13i/machines/agent-harness-multi-day-proof-builder",
    }
    if session.external_ref.get("orchestrator", "") not in permitted_orchestrators:
        raise ValueError("proof fault probe requires p13i/machines ownership")
    turn_ref = payload.get("turn_ref")
    if not isinstance(turn_ref, dict):
        raise ValueError("proof fault probe requires a managed turn reference")
    if str(turn_ref.get("agent_role", "")) != "proof-fault-probe":
        raise ValueError("proof fault probe role is unauthorized")
    required_capabilities = payload.get("required_capabilities")
    if not isinstance(required_capabilities, list):
        raise ValueError("proof fault probe requires capabilities")
    if "proof-fault-barrier" not in required_capabilities:
        raise ValueError("proof fault probe requires proof-fault-barrier")
    if str(probe.get("stage", "")) != "after-lease-before-acceptance":
        raise ValueError("proof fault probe stage is unsupported")
    if str(probe.get("provider", "")) not in supported_providers:
        raise ValueError("proof fault probe provider is unsupported")
    authorization_digest = str(probe.get("authorization_digest", ""))
    if len(authorization_digest) != 64:
        raise ValueError("proof fault probe authorization digest is invalid")
    try:
        int(authorization_digest, 16)
    except ValueError as error:
        raise ValueError("proof fault probe authorization digest is invalid") from error
    authorization = probe.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("proof fault probe authorization is required")
    expected_authorization = {
        "schema": "p13i/machines/provider-fault-authorization/v1",
        "external_ref": session.external_ref,
        "idempotency_key": idempotency_key,
        "stage": str(probe.get("stage", "")),
        "provider": str(probe.get("provider", "")),
        "agent_role": str(turn_ref.get("agent_role", "")),
    }
    if authorization != expected_authorization:
        raise ValueError("proof fault probe authorization does not match")
    if normalized_digest(authorization) != authorization_digest:
        raise ValueError("proof fault probe authorization digest does not match")
    authorization_constraint = (
        "proof-fault-authorization-sha256:" + authorization_digest
    )
    if goal is None or authorization_constraint not in goal.constraints:
        raise ValueError("proof fault probe authorization is not precommitted")


def _validate_service_fault_probe(
    session: Session,
    goal: Goal | None,
    payload: dict[str, Any],
    idempotency_key: str,
    supported_providers: frozenset[str],
) -> None:
    probe = payload.get("proof_service_fault_probe")
    if probe is None:
        return
    if not isinstance(probe, dict):
        raise ValueError("proof_service_fault_probe must be an object")
    if not idempotency_key:
        raise ValueError("proof service fault probe requires an idempotency key")
    permitted_orchestrators = {
        "p13i/machines/agent-harness-proof",
        "p13i/machines/agent-harness-proof-builder",
        "p13i/machines/agent-harness-multi-day-proof-builder",
    }
    if session.external_ref.get("orchestrator", "") not in permitted_orchestrators:
        raise ValueError("proof service fault probe requires p13i/machines ownership")
    turn_ref = payload.get("turn_ref")
    if not isinstance(turn_ref, dict):
        raise ValueError("proof service fault probe requires a managed turn reference")
    if str(turn_ref.get("agent_role", "")) != "proof-service-fault-probe":
        raise ValueError("proof service fault probe role is unauthorized")
    required_capabilities = payload.get("required_capabilities")
    if not isinstance(required_capabilities, list):
        raise ValueError("proof service fault probe requires capabilities")
    if "proof-service-fault-barrier" not in required_capabilities:
        raise ValueError(
            "proof service fault probe requires proof-service-fault-barrier"
        )
    if str(probe.get("stage", "")) != "after-acceptance-before-terminal":
        raise ValueError("proof service fault probe stage is unsupported")
    provider = str(probe.get("provider", ""))
    if provider not in supported_providers:
        raise ValueError("proof service fault probe provider is unsupported")
    requested_provider = str(payload.get("provider", ""))
    if requested_provider and requested_provider != provider:
        raise ValueError("proof service fault probe provider does not match")
    authorization_digest = str(probe.get("authorization_digest", ""))
    if len(authorization_digest) != 64:
        raise ValueError("proof service fault probe authorization digest is invalid")
    try:
        int(authorization_digest, 16)
    except ValueError as error:
        raise ValueError(
            "proof service fault probe authorization digest is invalid"
        ) from error
    authorization = probe.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("proof service fault probe authorization is required")
    expected_authorization = {
        "schema": "p13i/machines/service-fault-authorization/v1",
        "external_ref": session.external_ref,
        "idempotency_key": idempotency_key,
        "stage": str(probe.get("stage", "")),
        "provider": provider,
        "agent_role": str(turn_ref.get("agent_role", "")),
    }
    if authorization != expected_authorization:
        raise ValueError("proof service fault probe authorization does not match")
    if normalized_digest(authorization) != authorization_digest:
        raise ValueError(
            "proof service fault probe authorization digest does not match"
        )
    authorization_constraint = (
        "proof-service-fault-authorization-sha256:" + authorization_digest
    )
    if goal is None or authorization_constraint not in goal.constraints:
        raise ValueError("proof service fault probe authorization is not precommitted")


def _validate_workspace_reachable(workspace: Path) -> None:
    if workspace.is_relative_to(Path("/tmp")) or workspace.is_relative_to(
        Path("/private/tmp")
    ):
        raise HarnessError(
            "E_WORKSPACE_UNREACHABLE",
            "workspace "
            + str(workspace)
            + " is under /tmp, but the daemon runs with systemd"
            " PrivateTmp=true, so its /tmp is a private namespace and the"
            " workspace can never be reached daemon-side",
            status=400,
        )
    if not workspace.is_dir():
        raise HarnessError(
            "E_WORKSPACE_UNREACHABLE",
            "workspace " + str(workspace) + " does not exist on this host",
            status=400,
        )
    completed = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != "true":
        raise HarnessError(
            "E_WORKSPACE_UNREACHABLE",
            "workspace " + str(workspace) + " is not inside a git repository",
            status=400,
        )


def _workspace_reachable(workspace: str) -> bool:
    """Cheap local probe: the path exists and is inside a git repository."""
    path = Path(workspace)
    if not path.is_dir():
        return False
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0


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
            "kimi": KimiAdapter(),
            "grok": GrokAdapter(),
        }
        self.scheduler = Scheduler(self.store, self.adapters)
        self.reconciliations = ReconciliationManager(
            self.store,
            self.blobs,
        )
        self.machine_keys = load_machine_keys(paths.machine_keys)
        self._session_creation_lock = threading.Lock()
        self._reconciliation_resolution_lock = asyncio.Lock()
        self._proof_lock = threading.Lock()
        self._active_proofs: dict[str, int] = {}
        self._worker_supervision: dict[str, Any] = {
            "status": "initializing",
            "expected_sessions": [],
            "running_sessions": [],
            "restarted_sessions": [],
            "unrecovered": [],
        }
        self.workers = worker_manager
        if self.workers is None:
            self.workers = WorkerManager(paths)

    def close(self) -> None:
        self.store.close()

    def recover_workers(self) -> None:
        session_ids = self._recoverable_worker_sessions()
        blocked_recoveries = self._blocked_worker_recoveries()
        supervise = getattr(self.workers, "supervise", None)
        if callable(supervise):
            self._worker_supervision = supervise(
                self._worker_session_pairs(session_ids)
            )
            self._record_blocked_worker_recoveries(blocked_recoveries)
            return
        for session_id in session_ids:
            self.workers.ensure(session_id)
        self._worker_supervision = {
            "status": "ok",
            "expected_sessions": session_ids,
            "running_sessions": session_ids,
            "restarted_sessions": [],
            "unrecovered": [],
        }
        self._record_blocked_worker_recoveries(blocked_recoveries)

    def _record_blocked_worker_recoveries(
        self,
        blocked_recoveries: list[dict[str, Any]],
    ) -> None:
        if not blocked_recoveries:
            return
        report = dict(self._worker_supervision)
        unrecovered = list(report.get("unrecovered", []))
        unrecovered.extend(blocked_recoveries)
        report["status"] = "failed"
        report["unrecovered"] = unrecovered
        self._worker_supervision = report

    def supervise_workers(self) -> dict[str, Any]:
        self.recover_workers()
        for item in self._worker_supervision["restarted_sessions"]:
            self.store.append_event(
                item,
                "worker.supervised",
                status="restarted",
                metadata={"session_id": item},
            )
        self._park_unreachable_workers()
        return dict(self._worker_supervision)

    def _park_unreachable_workers(self) -> None:
        unrecovered = self._worker_supervision.get("unrecovered", [])
        if not isinstance(unrecovered, list):
            return
        for item in unrecovered:
            if not isinstance(item, dict):
                continue
            if item.get("reason") != "workspace-unreachable":
                continue
            ticks = item.get("consecutive_unreachable", 0)
            if not isinstance(ticks, int) or ticks < UNREACHABLE_PARK_TICKS:
                continue
            session_id = str(item.get("session_id", ""))
            if not session_id:
                continue
            session = self.store.get_session(session_id)
            if self._parked_unreachable(session):
                continue
            self.store.update_session(
                session_id,
                attention=Attention.NEEDS_INPUT,
            )
            self.store.append_event(
                session_id,
                "session.unreachable",
                status="needs-input",
                metadata={
                    "workspace": str(item.get("workspace", "")),
                    "host": host_id(),
                    "consecutive_unreachable": ticks,
                },
            )

    def _parked_unreachable(self, session: Session) -> bool:
        if session.attention != Attention.NEEDS_INPUT:
            return False
        return self._has_unreachable_event(session.session_id)

    def _has_unreachable_event(self, session_id: str) -> bool:
        for event in self.store.all_events(session_id):
            if event.event_type == "session.unreachable":
                return True
        return False

    def worker_supervision(self) -> dict[str, Any]:
        if self._worker_supervision["status"] == "initializing":
            self.supervise_workers()
        return dict(self._worker_supervision)

    def record_worker_supervision_failure(self, error: BaseException) -> None:
        expected_sessions = self._worker_supervision.get(
            "expected_sessions",
            [],
        )
        if not isinstance(expected_sessions, list):
            expected_sessions = []
        self._worker_supervision = {
            "status": "failed",
            "expected_sessions": expected_sessions,
            "running_sessions": [],
            "restarted_sessions": [],
            "unrecovered": [
                {
                    "reason": "worker-supervision-failed",
                    "detail": str(error)[:500],
                }
            ],
        }

    def _recoverable_worker_sessions(self) -> list[str]:
        blocked_sessions = {
            str(lease["session_id"])
            for lease in self.store.all_process_leases()
            if lease["state"] == "recovery-blocked"
        }
        session_ids: list[str] = []
        for session in self.store.list_sessions():
            session = self._name_session_from_history(session)
            if not self._needs_worker(session):
                continue
            if session.session_id in blocked_sessions:
                continue
            if self._parked_unreachable(session):
                continue
            session_ids.append(session.session_id)
        return sorted(session_ids)

    def _needs_worker(self, session: Session) -> bool:
        """Report whether supervision must keep a worker for a session.

        A stopped session keeps no worker of its own, but a resume
        queued for one has to reach a worker even when the service
        restarted between the enqueue and the claim.
        """

        if session.lifecycle in {
            Lifecycle.PAUSED,
            Lifecycle.STARTING,
            Lifecycle.RUNNING,
        }:
            return True
        if session.lifecycle != Lifecycle.STOPPED:
            return False
        return self.store.queued_command_exists(
            session.session_id,
            STOPPED_SESSION_COMMANDS,
        )

    def _worker_session_pairs(
        self,
        session_ids: list[str],
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for session_id in session_ids:
            session = self.store.get_session(session_id)
            pairs.append((session.session_id, session.workspace))
        return pairs

    def _blocked_worker_recoveries(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": str(lease["session_id"]),
                "lease_id": str(lease["lease_id"]),
                "command_id": str(lease["command_id"]),
                "attempt_id": str(lease["attempt_id"]),
                "provider": str(lease["provider"]),
                "reason": "process-lease-recovery-blocked",
            }
            for lease in self.store.all_process_leases()
            if lease["state"] == "recovery-blocked"
        ]

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
        permission_mode = str(payload.get("permission_mode", PermissionMode.APPROVAL))
        if permission_mode not in set(PermissionMode):
            raise ValueError("unsupported permission mode")
        execution_profile = validate_profile(
            str(payload.get("execution_profile", UNATTENDED))
        )
        external_ref = normalize_external_ref(payload.get("external_ref"))
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
        milestones = payload.get("milestones", [])
        if not isinstance(milestones, list):
            raise ValueError("milestones must be an array")
        permitted_providers = payload.get("permitted_providers", [])
        if not isinstance(permitted_providers, list):
            raise ValueError("permitted_providers must be an array")
        permitted_efforts = payload.get("permitted_efforts", [])
        if not isinstance(permitted_efforts, list):
            raise ValueError("permitted_efforts must be an array")
        max_concurrency = payload.get("max_concurrency", 1)
        completion_policy = str(payload.get("completion_policy", ""))
        incident_policy = str(payload.get("incident_policy", "recover-then-pause"))
        if external_ref.get("orchestrator", "").startswith("p13i/machines"):
            _require_machines_goal_envelope(
                payload,
                objective=objective,
                constraints=constraints,
                predicates=predicates,
                milestones=milestones,
                permitted_providers=permitted_providers,
                permitted_efforts=permitted_efforts,
                budgets=budgets,
                supported_providers=frozenset(self.adapters),
            )
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
                "predicates": [item for item in predicates if isinstance(item, dict)],
                "milestones": [item for item in milestones if isinstance(item, dict)],
                "budgets": budgets,
                "permitted_providers": [str(item) for item in permitted_providers],
                "permitted_efforts": [str(item) for item in permitted_efforts],
                "max_concurrency": max_concurrency,
                "completion_policy": completion_policy,
                "incident_policy": incident_policy,
            }
        session_id = new_uuid()
        goal = None
        if objective:
            goal = create_goal(
                session_id,
                objective,
                kind=str(payload.get("goal_kind", "finite")),
                constraints=tuple(str(item) for item in constraints),
                predicates=tuple(item for item in predicates if isinstance(item, dict)),
                milestones=tuple(item for item in milestones if isinstance(item, dict)),
                budgets=budgets,
                permitted_providers=tuple(str(item) for item in permitted_providers),
                permitted_efforts=tuple(str(item) for item in permitted_efforts),
                max_concurrency=max_concurrency,
                completion_policy=completion_policy,
                incident_policy=incident_policy,
            )
        existing = self.store.existing_ensured_session(
            creation_input,
            idempotency_key=idempotency_key,
            external_ref=external_ref,
        )
        if existing is not None:
            _complete_creation_intent(
                self.paths,
                idempotency_key,
                creation_input,
                existing.session_id,
            )
            return existing
        session_id, creation_intent = _reserve_creation_intent(
            self.paths,
            idempotency_key,
            creation_input,
            session_id,
        )
        if goal is not None:
            goal = replace(goal, session_id=session_id)
        inherited_ui_state = self._workspace_ui_state(str(workspace))
        _validate_workspace_reachable(workspace)
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
        created = False
        try:
            with self.store.transaction():
                session, created = self.store.ensure_session(
                    session,
                    creation_input,
                    idempotency_key=idempotency_key,
                )
                if created:
                    if inherited_ui_state:
                        self.store.set_ui_state(
                            "session:" + session_id,
                            inherited_ui_state,
                        )
                    self.store.set_session_safety(session_id, execution_profile)
                    if goal is not None:
                        self.store.create_goal(goal)
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
        except BaseException:
            if not direct:
                remove_worktree(workspace, worktree, session_id)
            raise
        if not created:
            if not direct:
                remove_worktree(workspace, worktree, session_id)
            _remove_creation_intent(creation_intent)
            return session
        completed = self.store.get_session(session_id)
        _remove_creation_intent(creation_intent)
        return completed

    def submit_message(
        self,
        session_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        requested_effort = str(payload.get("effort", "")).strip().casefold()
        if requested_effort and requested_effort not in SUPPORTED_EFFORTS:
            raise ValueError("unsupported message effort")
        if "effort" in payload:
            payload = dict(payload)
            payload["effort"] = requested_effort
        session = self.store.get_session(session_id)
        _validate_proof_fault_probe(
            session,
            self.store.goal_for_session(session_id),
            payload,
            idempotency_key,
            frozenset(self.adapters),
        )
        _validate_service_fault_probe(
            session,
            self.store.goal_for_session(session_id),
            payload,
            idempotency_key,
            frozenset(self.adapters),
        )
        if "proof_fault_probe" in payload and "proof_service_fault_probe" in payload:
            raise ValueError("only one proof fault probe may be requested")
        provider = str(payload.get("provider", "")).strip()
        if not provider:
            provider = "automatic-route"
        require_state_headroom(self.paths.state_dir, provider)
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("message text is required")
        if "permission_mode" in payload:
            requested_permission = str(payload.get("permission_mode", ""))
            if requested_permission not in set(PermissionMode):
                raise ValueError("unsupported per-turn permission mode")
            _require_permission_restriction(
                session.permission_mode,
                requested_permission,
            )
        receipt, created = self.store.ensure_message_command(
            session_id,
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
        self.workers.ensure(session_id)
        return receipt.as_dict()

    def configure_session(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> Session:
        require_uuid(session_id, "session_id")
        session = self.store.get_session(session_id)
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
            profile = validate_profile(str(payload.get("execution_profile", "")))
            current_profile = str(self.store.session_safety(session_id)["profile"])
            if profile != current_profile:
                raise ConflictError(
                    "session execution profile is immutable; create a new session"
                )
        if not changes:
            if "execution_profile" not in payload:
                raise ValueError("no supported session settings were provided")
            return session
        session = self.store.update_session(session_id, **changes)
        configured_fields = sorted(changes)
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
            item.as_dict() for item in self.store.pending_reconciliations(session_id)
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

    def transcript(
        self,
        session_id: str,
        *,
        tail_turns: int,
        token_budget: int,
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        self.store.get_session(session_id)
        policy = validate_render_policy(
            RenderPolicy(
                token_budget=token_budget,
                tail_turns=tail_turns,
            )
        )
        transcript = project_transcript(
            self.store,
            session_id,
            blobs=self.blobs,
        )
        return {
            "transcript": transcript.as_dict(),
            "rendered": render(transcript, policy),
        }

    def timeline(self, session_id: str) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        self.store.get_session(session_id)
        timeline = project_timeline(self.store, session_id)
        return {
            "timeline": timeline,
            "rendered": render_timeline(timeline),
        }

    def proof(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        event_limit: int = 1_000,
        through_sequence: int | None = None,
        snapshot_id: str = "",
    ) -> dict[str, Any]:
        with self._proof_lock:
            self._active_proofs[session_id] = self._active_proofs.get(session_id, 0) + 1
        try:
            return proof_snapshot(
                self.store,
                session_id,
                after_sequence=after_sequence,
                event_limit=event_limit,
                through_sequence=through_sequence,
                snapshot_id=snapshot_id,
            )
        finally:
            with self._proof_lock:
                remaining = self._active_proofs[session_id] - 1
                if remaining == 0:
                    self._active_proofs.pop(session_id)
                else:
                    self._active_proofs[session_id] = remaining

    def quiescence(self) -> dict[str, Any]:
        command_details = self.store.active_command_summaries()
        unattended = [
            item for item in command_details if item.get("profile") == "unattended"
        ]
        with self._proof_lock:
            active_proof_sessions = sorted(self._active_proofs)
            active_proofs = sum(self._active_proofs.values())
        restart_safe = not command_details and active_proofs == 0
        return {
            "restart_safe": restart_safe,
            "active_commands": len(command_details),
            "active_command_details": command_details,
            "active_unattended_commands": unattended,
            "active_proofs": active_proofs,
            "active_proof_sessions": active_proof_sessions,
        }

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
        return self.reconciliations.inspect(reconciliation_id).as_dict()

    async def resolve_reconciliation(
        self,
        reconciliation_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str = "",
        request_digest: str = "",
    ) -> dict[str, Any]:
        async with self._reconciliation_resolution_lock:
            return await self._resolve_reconciliation(
                reconciliation_id,
                payload,
                idempotency_key=idempotency_key,
                request_digest=request_digest,
            )

    async def _resolve_reconciliation(
        self,
        reconciliation_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str = "",
        request_digest: str = "",
    ) -> dict[str, Any]:
        require_uuid(reconciliation_id, "reconciliation_id")
        if bool(idempotency_key) != bool(request_digest):
            raise ValueError("reconciliation mutation identity requires key and digest")
        decision = str(payload.get("decision", "")).strip()
        observed_digest = str(payload.get("observed_workspace_digest", "")).strip()
        if not observed_digest:
            raise ValueError("observed workspace digest is required")
        audit = payload.get("audit")
        if audit is not None and not isinstance(audit, dict):
            raise ValueError("reconciliation audit must be an object")
        validate_reconciliation_audit(audit)
        record = self.reconciliations.inspect(reconciliation_id)
        previously_resolved = record.status == ReconciliationStatus.RESOLVED
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
            idempotency_key=idempotency_key,
            request_digest=request_digest,
            operation="reconciliation-resolve:" + reconciliation_id,
        )
        if not idempotency_key and not previously_resolved:
            self.store.append_event(
                record.session_id,
                "reconciliation.resolved",
                status=record.status,
                metadata={
                    "reconciliation_id": record.reconciliation_id,
                    "command_id": record.command_id,
                    "decision": record.resolution,
                    "workspace_digest": (record.current_workspace_digest),
                    "discovery_checkpoint_id": str(
                        record.audit.get("discovery_checkpoint_id", "")
                    ),
                    "resolution_checkpoint_id": str(
                        record.audit.get("resolution_checkpoint_id", "")
                    ),
                },
            )
            if record.resolution != "stop":
                self.workers.ensure(record.session_id)
        if idempotency_key and record.resolution != "stop":
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
                "reconciliation restore requires approval " + approval_id,
                status=409,
            )
        approval = self.store.approval(approval_id)
        if (
            approval["session_id"] != session_id
            or approval["provider_request_id"] != reconciliation_id
            or approval["kind"] != "reconciliation.restore"
        ):
            raise ConflictError("approval does not authorize this reconciliation")
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
        *,
        idempotency_key: str = "",
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
        if len(extension) == 1 and not allow_xhigh:
            raise ValueError("budget extension has no additive capacity")
        authorization: dict[str, Any] | None = None
        if allow_xhigh:
            command_id = str(payload.get("command_id", ""))
            require_uuid(command_id, "xhigh command_id")
            provider = str(payload.get("provider", ""))
            if provider not in self.adapters:
                raise ValueError("xhigh authorization provider is unsupported")
            if not idempotency_key:
                raise ValueError("xhigh authorization idempotency key is required")
            expires_at = (
                datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=15)
            ).isoformat()
            authorization = self.store.create_xhigh_authorization(
                session_id,
                command_id,
                provider,
                authorization_request_digest=normalized_digest(payload),
                idempotency_key=idempotency_key,
                expires_at=expires_at,
            )
        result = self.store.session_safety(session_id)
        if len(extension) > 1:
            result = self.store.extend_session_safety(
                session_id,
                extension,
            )
        authorization_id = ""
        authorized_command_id = ""
        authorized_provider = ""
        if authorization is not None:
            authorization_id = str(authorization["authorization_id"])
            authorized_command_id = str(authorization["command_id"])
            authorized_provider = str(authorization["provider"])
            self.workers.ensure(session_id)
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
                "xhigh_authorization_id": authorization_id,
                "command_id": authorized_command_id,
                "provider": authorized_provider,
                "reason": reason,
            },
        )
        return result

    def create_process_lease(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        provider = str(payload.get("provider", "")).strip()
        if provider not in self.adapters:
            raise ValueError("unsupported lease provider")
        profile = validate_profile(str(payload.get("execution_profile", UNATTENDED)))
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
            )
        observed_at = str(usage.get("observed_at", ""))
        try:
            observed = datetime.datetime.fromisoformat(observed_at)
        except ValueError as error:
            raise SafetyGuardError(
                "provider usage timestamp is invalid",
                provider,
            ) from error
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=datetime.UTC)
        age = datetime.datetime.now(datetime.UTC) - observed
        if age > datetime.timedelta(seconds=90):
            raise SafetyGuardError(
                "provider usage is stale",
                provider,
            )
        binding = _optional_number(usage.get("binding_percent"))
        if binding is None:
            raise SafetyGuardError(
                "provider binding usage is unavailable",
                provider,
            )
        if binding < 0:
            raise SafetyGuardError(
                "provider binding usage is invalid",
                provider,
            )
        if bool(usage.get("credits_engaged", False)):
            raise SafetyGuardError(
                "metered provider credits would engage",
                provider,
            )
        ceiling = limits_for(profile, "operations").binding_ceiling
        if binding >= ceiling:
            raise SafetyGuardError(
                "provider binding usage reached the safety ceiling",
                provider,
            )

    def update_process_lease(
        self,
        lease_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        action = str(payload.get("action", "heartbeat"))
        if action == "release":
            prior = self.store.process_lease(lease_id)
            updated = self.store.update_process_lease(
                lease_id,
                state="released",
            )
            if prior["state"] == "recovery-blocked":
                self.store.append_event(
                    str(prior["session_id"]),
                    "lease.recovery.unblocked",
                    status="released",
                    metadata={
                        "lease_id": lease_id,
                        "command_id": str(prior["command_id"]),
                        "attempt_id": str(prior["attempt_id"]),
                        "provider": str(prior["provider"]),
                    },
                )
            return updated
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
        prior = self.store.process_lease(lease_id)
        prior_state = str(prior["state"])
        if action == "attach":
            if prior_state == "recovery-blocked":
                raise ConflictError(
                    "recovery-blocked lease requires an explicit release"
                )
            if prior_state == "active":
                if pid != prior["pid"] or pid_start != prior["pid_start"]:
                    raise ConflictError("active lease process identity is immutable")
            elif prior_state != "reserved":
                raise ConflictError("terminal lease cannot be attached")
        elif prior_state not in {"reserved", "active", "recovery-blocked"}:
            raise ConflictError("terminal lease cannot be heartbeated")
        return self.store.update_process_lease(
            lease_id,
            pid=pid,
            pid_start=pid_start,
            state=state,
            expires_at=_lease_expiry(),
        )

    def process_leases(self) -> list[dict[str, Any]]:
        return self.store.all_process_leases()

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
                "unsupported UI state fields: " + ", ".join(sorted(unknown))
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
            state = self.store.get_ui_state("session:" + session.session_id)
            return {
                name: value for name, value in state.items() if name in inherited_fields
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
        if command_type not in CONTROL_COMMAND_TYPES:
            raise ValueError("control command type is unsupported")
        normalized_payload: dict[str, Any] = {}
        if command_type == "interrupt":
            unknown = set(payload) - {"target_command_id"}
            if unknown:
                raise ValueError("interrupt command has unsupported fields")
            if "target_command_id" in payload:
                target_command_id = payload.get("target_command_id")
                if not isinstance(target_command_id, str):
                    raise ValueError("target_command_id must be a UUID")
                normalized_payload["target_command_id"] = require_uuid(
                    target_command_id,
                    "target_command_id",
                )
        elif command_type == "steer":
            if set(payload) != {"text"}:
                raise ValueError("steer command requires only text")
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError("steer text must be a string")
            if not text.strip() or len(text) > MAX_STEER_TEXT_LENGTH:
                raise ValueError("steer text must contain 1 to 65536 characters")
            normalized_payload["text"] = text
        elif payload:
            raise ValueError(command_type + " command requires an empty object")
        receipt = self.store.enqueue_command(
            session_id,
            command_type,
            normalized_payload,
            idempotency_key,
        )
        self._ensure_control_worker(session_id)
        return receipt.as_dict()

    def _ensure_control_worker(self, session_id: str) -> None:
        """Start the worker that must claim a freshly queued control.

        A stopped session normally keeps no worker, and a worker that
        drained its queue retires its registration a moment before its
        process exits. A resume enqueued in that window must not be
        handed to the exiting process, so an unregistered stopped
        session forces a replacement worker.
        """

        force = False
        session = self.store.get_session(session_id)
        if session.lifecycle == Lifecycle.STOPPED:
            force = not self.store.worker_registered(session_id)
        self.workers.ensure(session_id, force=force)

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
        *,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        orchestrator = session.external_ref.get("orchestrator", "")
        if orchestrator.startswith("p13i/machines") and not idempotency_key:
            raise ValueError("p13i/machines evidence requires an idempotency key")
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
        if idempotency_key:
            evidence = self.store.add_evidence_once(
                session_id,
                evidence,
                idempotency_key=idempotency_key,
                request_digest=normalized_digest(payload),
            )
        else:
            self.store.add_evidence(evidence)
            self.store.append_event(
                session_id,
                "goal.evidence",
                status="complete",
                metadata=evidence.as_dict(),
            )
        evaluation = evaluate_goal(
            goal,
            self.store.evidence(goal.goal_id),
        )
        milestones = evaluate_milestones(
            goal,
            self.store.evidence(goal.goal_id),
        )
        if milestones != goal.milestones:
            goal = self.store.update_milestone_statuses(
                goal.goal_id,
                milestones,
            )
            evaluation = evaluate_goal(
                goal,
                self.store.evidence(goal.goal_id),
            )
        if evaluation.satisfied and goal.status != GoalStatus.COMPLETE:
            self.store.update_goal_status(goal.goal_id, GoalStatus.COMPLETE)
            self.store.update_session(
                session_id,
                lifecycle=Lifecycle.COMPLETED,
                attention=Attention.READY,
            )
            self.store.append_event(
                session_id,
                "goal.completed",
                status="complete",
                metadata={"matched": list(evaluation.matched)},
            )
        return evidence.as_dict()

    def promote_goal(
        self,
        session_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        if not idempotency_key:
            raise ValueError("goal promotion requires an idempotency key")
        request_digest = normalized_digest(payload)
        existing = self.store.goal_promotion_by_key(idempotency_key)
        if existing is not None:
            if (
                str(existing["session_id"]) != session_id
                or str(existing["request_digest"]) != request_digest
            ):
                raise ConflictError("goal promotion idempotency key was reused")
            current_goal = self.store.get_goal(str(existing["next_goal_id"]))
            current_session = self.store.get_session(session_id)
            return {
                "promotion": _goal_promotion_receipt(existing),
                "goal": current_goal.as_dict(),
                "session": current_session.as_dict(),
            }
        previous_goal_id = str(payload.get("from_goal_id", ""))
        require_uuid(previous_goal_id, "from_goal_id")
        previous = self.store.get_goal(previous_goal_id)
        if previous.session_id != session_id:
            raise ConflictError("goal promotion source belongs to another session")
        stage = str(payload.get("stage", "")).strip()
        if not stage or len(stage) > 128:
            raise ValueError("promotion stage must contain 1 to 128 characters")
        authorization = payload.get("authorization")
        if not isinstance(authorization, dict) or not authorization:
            raise ValueError("goal promotion requires explicit authorization")
        objective = str(payload.get("objective", "")).strip()
        constraints_value = payload.get("constraints", [])
        predicates_value = payload.get("predicates", [])
        milestones_value = payload.get("milestones", [])
        budgets = payload.get("budgets", {})
        permitted_providers_value = payload.get("permitted_providers", [])
        permitted_efforts_value = payload.get("permitted_efforts", [])
        if not isinstance(constraints_value, list):
            raise ValueError("goal constraints must be a list")
        if not isinstance(predicates_value, list):
            raise ValueError("goal predicates must be a list")
        if not isinstance(milestones_value, list):
            raise ValueError("goal milestones must be a list")
        if not isinstance(budgets, dict):
            raise ValueError("goal budgets must be an object")
        if not isinstance(permitted_providers_value, list):
            raise ValueError("permitted providers must be a list")
        if not isinstance(permitted_efforts_value, list):
            raise ValueError("permitted efforts must be a list")
        constraints = tuple(str(item) for item in constraints_value)
        if not set(previous.constraints).issubset(set(constraints)):
            raise ValueError("goal promotion cannot remove constraints")
        predicates = tuple(item for item in predicates_value if isinstance(item, dict))
        predicates = promoted_predicates(previous.predicates, predicates)
        next_goal = create_goal(
            session_id,
            objective,
            kind=str(payload.get("goal_kind", previous.kind)),
            constraints=constraints,
            predicates=predicates,
            milestones=tuple(
                item for item in milestones_value if isinstance(item, dict)
            ),
            budgets=budgets,
            permitted_providers=tuple(str(item) for item in permitted_providers_value),
            permitted_efforts=tuple(str(item) for item in permitted_efforts_value),
            max_concurrency=payload.get(
                "max_concurrency",
                previous.max_concurrency,
            ),
            completion_policy=str(
                payload.get("completion_policy", previous.completion_policy)
            ),
            incident_policy=str(
                payload.get("incident_policy", previous.incident_policy)
            ),
        )
        if previous.kind != "finite" or next_goal.kind != "finite":
            raise ValueError("goal promotion requires finite goals")
        if not set(next_goal.permitted_providers).issubset(
            set(previous.permitted_providers)
        ):
            raise ValueError("goal promotion cannot widen permitted providers")
        if not set(next_goal.permitted_efforts).issubset(
            set(previous.permitted_efforts)
        ):
            raise ValueError("goal promotion cannot widen permitted efforts")
        if next_goal.max_concurrency > previous.max_concurrency:
            raise ValueError("goal promotion cannot increase concurrency")
        if next_goal.completion_policy != previous.completion_policy:
            raise ValueError("goal promotion cannot change completion policy")
        if next_goal.incident_policy != previous.incident_policy:
            raise ValueError("goal promotion cannot change incident policy")
        budget_increased = False
        budget_increases: dict[str, float] = {}
        for name, previous_limit in previous.budgets.items():
            next_limit = next_goal.budgets.get(name)
            if not isinstance(next_limit, (int, float)) or isinstance(
                next_limit,
                bool,
            ):
                raise ValueError("goal promotion cannot remove a budget")
            if float(next_limit) < float(previous_limit):
                raise ValueError("goal promotion cannot reduce a budget")
            if float(next_limit) > float(previous_limit):
                budget_increased = True
                budget_increases[name] = float(next_limit) - float(previous_limit)
        for name, next_limit in next_goal.budgets.items():
            if name in previous.budgets:
                continue
            if float(next_limit) > 0:
                budget_increased = True
                budget_increases[name] = float(next_limit)
        if not budget_increased:
            raise ValueError("goal promotion requires an additive budget")
        _require_promotion_authorization(
            authorization,
            session_id=session_id,
            previous_goal_id=previous.goal_id,
            stage=stage,
            budget_increases=budget_increases,
            next_goal_contract_digest=goal_contract_digest(next_goal),
        )
        next_goal = replace(
            next_goal,
            created_at=previous.created_at,
            milestones=promoted_milestones(previous, next_goal),
        )
        receipt = self.store.promote_goal(
            previous,
            next_goal,
            stage=stage,
            authorization_digest=normalized_digest(authorization),
            authorization=authorization,
            request_digest=request_digest,
            idempotency_key=idempotency_key,
        )
        current_goal = self.store.get_goal(str(receipt["next_goal_id"]))
        session = self.store.get_session(session_id)
        self.workers.ensure(session_id)
        return {
            "promotion": _goal_promotion_receipt(receipt),
            "goal": current_goal.as_dict(),
            "session": session.as_dict(),
        }

    def adopt_session_contract(
        self,
        session_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        if not idempotency_key:
            raise ValueError("contract adoption requires an idempotency key")
        request_digest = normalized_digest(payload)
        existing = self.store.goal_contract_adoption_by_key(idempotency_key)
        if existing is not None:
            if (
                str(existing["session_id"]) != session_id
                or str(existing["request_digest"]) != request_digest
            ):
                raise ConflictError("contract adoption idempotency key was reused")
            current_goal = self.store.get_goal(str(existing["next_goal_id"]))
            current_session = self.store.get_session(session_id)
            return {
                "adoption": _goal_contract_adoption_receipt(existing),
                "goal": current_goal.as_dict(),
                "session": current_session.as_dict(),
            }
        session = self.store.get_session(session_id)
        workspace = Path(str(payload.get("workspace", ""))).expanduser().resolve()
        if workspace != Path(session.workspace).resolve():
            raise ValueError("contract adoption workspace does not match")
        direct = payload.get("direct")
        if not isinstance(direct, bool):
            raise ValueError("contract adoption direct must be boolean")
        actual_direct = Path(session.worktree).resolve() == workspace
        if direct != actual_direct:
            raise ValueError("contract adoption cannot change worktree mode")
        external_ref = normalize_external_ref(payload.get("external_ref"))
        if not external_ref.get("orchestrator", "").startswith("p13i/machines"):
            raise ValueError("contract adoption is reserved for p13i/machines")
        objective = str(payload.get("goal", "")).strip()
        constraints_value = payload.get("constraints", [])
        predicates_value = payload.get("predicates", [])
        milestones_value = payload.get("milestones", [])
        budgets = payload.get("budgets", {})
        permitted_providers_value = payload.get("permitted_providers", [])
        permitted_efforts_value = payload.get("permitted_efforts", [])
        if not isinstance(constraints_value, list):
            raise ValueError("constraints must be an array")
        if not isinstance(predicates_value, list):
            raise ValueError("predicates must be an array")
        if not isinstance(milestones_value, list):
            raise ValueError("milestones must be an array")
        if not isinstance(budgets, dict):
            raise ValueError("budgets must be an object")
        if not isinstance(permitted_providers_value, list):
            raise ValueError("permitted_providers must be an array")
        if not isinstance(permitted_efforts_value, list):
            raise ValueError("permitted_efforts must be an array")
        _require_machines_goal_envelope(
            payload,
            objective=objective,
            constraints=constraints_value,
            predicates=predicates_value,
            milestones=milestones_value,
            permitted_providers=permitted_providers_value,
            permitted_efforts=permitted_efforts_value,
            budgets=budgets,
            supported_providers=frozenset(self.adapters),
        )
        execution_profile = validate_profile(str(payload.get("execution_profile", "")))
        permission_mode = str(payload.get("permission_mode", ""))
        if permission_mode not in set(PermissionMode):
            raise ValueError("unsupported permission mode")
        next_goal = create_goal(
            session_id,
            objective,
            kind=str(payload.get("goal_kind", "finite")),
            constraints=tuple(str(item) for item in constraints_value),
            predicates=tuple(
                item for item in predicates_value if isinstance(item, dict)
            ),
            milestones=tuple(
                item for item in milestones_value if isinstance(item, dict)
            ),
            budgets=budgets,
            permitted_providers=tuple(str(item) for item in permitted_providers_value),
            permitted_efforts=tuple(str(item) for item in permitted_efforts_value),
            max_concurrency=payload.get("max_concurrency", 1),
            completion_policy=str(payload.get("completion_policy", "")),
            incident_policy=str(payload.get("incident_policy", "")),
        )
        creation_input = {
            "workspace": str(workspace),
            "name": str(payload.get("name", "")),
            "permission_mode": permission_mode,
            "execution_profile": execution_profile,
            "direct": direct,
            "external_ref": external_ref,
            "routing": {
                "model": str(payload.get("model", "")),
                "effort": str(payload.get("effort", "")),
            },
            "goal": {
                "objective": objective,
                "kind": next_goal.kind,
                "constraints": list(next_goal.constraints),
                "predicates": list(next_goal.predicates),
                "milestones": [
                    {
                        "milestone_id": item.milestone_id,
                        "title": item.title,
                        "dependencies": list(item.dependencies),
                        "predicates": list(item.predicates),
                    }
                    for item in next_goal.milestones
                ],
                "budgets": next_goal.budgets,
                "permitted_providers": list(next_goal.permitted_providers),
                "permitted_efforts": list(next_goal.permitted_efforts),
                "max_concurrency": next_goal.max_concurrency,
                "completion_policy": next_goal.completion_policy,
                "incident_policy": next_goal.incident_policy,
            },
        }
        previous = self.store.goal_for_session(session_id)
        previous_digest = ""
        if previous is not None:
            previous_digest = goal_contract_digest(previous)
        authorization = payload.get("authorization")
        if not isinstance(authorization, dict):
            raise ValueError("contract adoption requires typed authorization")
        _require_contract_adoption_authorization(
            authorization,
            session_id=session_id,
            external_ref=external_ref,
            previous_goal_digest=previous_digest,
            goal_envelope_digest=normalized_digest(creation_input["goal"]),
        )
        receipt = self.store.adopt_session_contract(
            session_id,
            next_goal,
            external_ref=external_ref,
            creation_input=creation_input,
            authorization_digest=normalized_digest(authorization),
            authorization=authorization,
            request_digest=request_digest,
            idempotency_key=idempotency_key,
        )
        current_goal = self.store.get_goal(str(receipt["next_goal_id"]))
        current_session = self.store.get_session(session_id)
        self.workers.ensure(session_id)
        return {
            "adoption": _goal_contract_adoption_receipt(receipt),
            "goal": current_goal.as_dict(),
            "session": current_session.as_dict(),
        }

    def invalidate_dispatch_generation(
        self,
        session_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        if not idempotency_key:
            raise ValueError("dispatch invalidation requires an idempotency key")
        reason = str(payload.get("reason", "")).strip()
        if not reason or len(reason) > 500:
            raise ValueError("invalidation reason must contain 1 to 500 characters")
        authorization = payload.get("authorization")
        if not isinstance(authorization, dict):
            raise ValueError("dispatch invalidation requires typed authorization")
        operator_schema = "p13i/agent-harness/dispatch-invalidation-authorization/v1"
        transition_schema = (
            "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
        )
        schema = str(authorization.get("schema", ""))
        if schema not in {operator_schema, transition_schema}:
            raise ValueError("dispatch invalidation authorization is unsupported")
        authorization_digest = normalized_digest(authorization)
        stored_authorization = authorization
        if schema == operator_schema:
            safety = self.store.session_safety(session_id)
            if str(safety.get("profile", "")) != "interactive":
                raise ValueError("managed dispatch invalidation requires a transition")
        if authorization.get("session_id") != session_id:
            raise ValueError("dispatch invalidation session does not match")
        if authorization.get("reason") != reason:
            raise ValueError("dispatch invalidation reason does not match")
        receipt_sha256 = str(authorization.get("receipt_sha256", ""))
        if len(receipt_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in receipt_sha256
        ):
            raise ValueError("dispatch invalidation receipt is invalid")
        _require_retained_authorization_receipt(
            authorization,
            receipt_sha256,
        )
        request_digest = normalized_digest(payload)
        replay = self.store.dispatch_invalidation_replay(
            session_id,
            idempotency_key,
            request_digest,
        )
        if replay is not None:
            return _dispatch_invalidation_receipt(replay)
        prior_command_id = ""
        next_turn_ref: dict[str, str] = {}
        if schema == transition_schema:
            prior_command_id = str(payload.get("prior_command_id", ""))
            require_uuid(prior_command_id, "prior_command_id")
            next_turn_ref = normalize_turn_ref(payload.get("next_turn_ref"))
            if not next_turn_ref.get("step_id"):
                raise ValueError("dispatch transition requires a next step identifier")
            if authorization.get("prior_command_id") != prior_command_id:
                raise ValueError("dispatch transition prior command does not match")
            prior_command_type = str(payload.get("prior_command_type", ""))
            prior_anchor_kind = str(payload.get("prior_anchor_kind", ""))
            prior_reconciliation_id = str(payload.get("prior_reconciliation_id", ""))
            prior_reconciliation_resolution = str(
                payload.get("prior_reconciliation_resolution", "")
            )
            if not prior_command_type or len(prior_command_type) > 64:
                raise ValueError("dispatch transition prior command type is invalid")
            if prior_anchor_kind not in DISPATCH_TRANSITION_ANCHOR_KINDS:
                raise ValueError("dispatch transition anchor kind is invalid")
            if prior_anchor_kind == "resolved-reconciliation":
                require_uuid(
                    prior_reconciliation_id,
                    "prior_reconciliation_id",
                )
                if prior_reconciliation_resolution not in {
                    "accept-current",
                    "restore-pre-turn",
                }:
                    raise ValueError(
                        "dispatch transition reconciliation resolution is invalid"
                    )
            elif prior_reconciliation_id or prior_reconciliation_resolution:
                raise ValueError(
                    "dispatch transition reconciliation binding is unexpected"
                )
            if normalize_turn_ref(authorization.get("next_turn_ref")) != next_turn_ref:
                raise ValueError("dispatch transition next stage does not match")
            session = self.store.get_session(session_id)
            external_ref = session.external_ref
            if (
                authorization.get("external_orchestrator")
                != external_ref.get("orchestrator")
                or authorization.get("external_job_id") != external_ref.get("job_id")
                or not external_ref.get("orchestrator")
                or not external_ref.get("job_id")
            ):
                raise ValueError("dispatch transition orchestrator does not match")
            anchor = self.store.dispatch_transition_anchor(session_id)
            if not bool(anchor.get("eligible", False)):
                raise ConflictError(
                    "dispatch transition anchor is unavailable: "
                    + str(anchor.get("reason", "unknown"))
                )
            for name, expected in (
                ("prior_command_id", prior_command_id),
                ("prior_command_type", prior_command_type),
                ("prior_anchor_kind", prior_anchor_kind),
                ("prior_reconciliation_id", prior_reconciliation_id),
                (
                    "prior_reconciliation_resolution",
                    prior_reconciliation_resolution,
                ),
            ):
                if anchor.get(name) != expected:
                    raise ConflictError("dispatch transition " + name + " is stale")
            transition_sequence = payload.get("transition_sequence")
            if (
                isinstance(transition_sequence, bool)
                or not isinstance(transition_sequence, int)
                or transition_sequence < 1
            ):
                raise ValueError("dispatch transition sequence is invalid")
            prior_checkpoint_id = str(payload.get("prior_checkpoint_id", ""))
            require_uuid(prior_checkpoint_id, "prior_checkpoint_id")
            prior_generation_digest = str(payload.get("prior_generation_digest", ""))
            if len(prior_generation_digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in prior_generation_digest
            ):
                raise ValueError("dispatch transition generation digest is invalid")
            prior_material_digest = str(payload.get("prior_material_digest", ""))
            if len(prior_material_digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in prior_material_digest
            ):
                raise ValueError("dispatch transition material digest is invalid")
            goal_id = str(session.goal_id)
            if not goal_id or authorization.get("goal_id") != goal_id:
                raise ValueError("dispatch transition goal does not match")
            epoch_id = str(authorization.get("epoch_id", ""))
            if (
                not epoch_id
                or len(epoch_id) > 128
                or any(character.isspace() for character in epoch_id)
            ):
                raise ValueError("dispatch transition epoch is invalid")
            policy_value = authorization.get("policy")
            policy_ref_value = authorization.get("policy_ref")
            retained_policy = False
            if transition_sequence == 1:
                if not isinstance(policy_value, dict) or not policy_value:
                    raise ValueError(
                        "first dispatch transition requires the full policy"
                    )
                if policy_ref_value is not None:
                    raise ValueError(
                        "first dispatch transition must not carry a policy reference"
                    )
                policy = policy_value
            else:
                if policy_value is not None:
                    raise ValueError(
                        "follow-up dispatch transition requires a policy reference"
                    )
                if not isinstance(policy_ref_value, dict):
                    raise ValueError(
                        "follow-up dispatch transition policy reference is required"
                    )
                policy_sha256_value = str(authorization.get("policy_sha256", ""))
                expected_policy_ref = {
                    "policy_sha256": policy_sha256_value,
                    "session_id": session_id,
                    "goal_id": goal_id,
                    "epoch_id": epoch_id,
                }
                if policy_ref_value != expected_policy_ref:
                    raise ValueError(
                        "dispatch transition policy reference does not match"
                    )
                policy = self.store.dispatch_transition_policy(
                    session_id,
                    goal_id,
                    epoch_id,
                    policy_sha256_value,
                )
                retained_policy = True
                stored_authorization = dict(authorization)
                stored_authorization["policy"] = policy
            policy_sha256 = normalized_digest(policy)
            for name, expected in (
                ("goal_id", goal_id),
                ("epoch_id", epoch_id),
                ("prior_checkpoint_id", prior_checkpoint_id),
                ("prior_generation_digest", prior_generation_digest),
                ("prior_material_digest", prior_material_digest),
            ):
                if anchor.get(name) != expected:
                    raise ConflictError("dispatch transition " + name + " is stale")
            next_command_digest = str(payload.get("next_command_digest", ""))
            if len(next_command_digest) != 64 or any(
                character not in "0123456789abcdef" for character in next_command_digest
            ):
                raise ValueError("dispatch transition command digest is invalid")
            _require_dispatch_transition_policy(
                self.store,
                session_id=session_id,
                session=session,
                policy=policy,
                policy_sha256=policy_sha256,
                next_turn_ref=next_turn_ref,
                transition_sequence=transition_sequence,
                next_command_digest=next_command_digest,
                retained_policy=retained_policy,
            )
            for name, expected in (
                ("transition_sequence", transition_sequence),
                ("prior_command_type", prior_command_type),
                ("prior_anchor_kind", prior_anchor_kind),
                ("prior_reconciliation_id", prior_reconciliation_id),
                (
                    "prior_reconciliation_resolution",
                    prior_reconciliation_resolution,
                ),
                ("prior_checkpoint_id", prior_checkpoint_id),
                ("prior_generation_digest", prior_generation_digest),
                ("prior_material_digest", prior_material_digest),
                ("epoch_id", epoch_id),
                ("policy_sha256", policy_sha256),
            ):
                if authorization.get(name) != expected:
                    raise ValueError("dispatch transition " + name + " does not match")
            if authorization.get("next_command_digest") != next_command_digest:
                raise ValueError("dispatch transition command digest does not match")
            receipt = authorization.get("receipt")
            if not isinstance(receipt, dict):
                raise ValueError("dispatch transition receipt is invalid")
            exact_receipt = {
                "session_id": session_id,
                "external_ref": session.external_ref,
                "goal_id": goal_id,
                "prior_command_id": prior_command_id,
                "prior_command_type": prior_command_type,
                "prior_anchor_kind": prior_anchor_kind,
                "prior_reconciliation_id": prior_reconciliation_id,
                "prior_reconciliation_resolution": (prior_reconciliation_resolution),
                "prior_checkpoint_id": prior_checkpoint_id,
                "prior_generation_digest": prior_generation_digest,
                "prior_material_digest": prior_material_digest,
                "next_turn_ref": next_turn_ref,
                "transition_sequence": transition_sequence,
                "epoch_id": epoch_id,
                "policy_sha256": policy_sha256,
                "next_command_digest": next_command_digest,
            }
            if receipt != exact_receipt:
                raise ValueError("dispatch transition source receipt does not match")
        receipt = self.store.create_dispatch_invalidation(
            session_id,
            reason=reason,
            authorization=stored_authorization,
            request_digest=request_digest,
            idempotency_key=idempotency_key,
            prior_command_id=prior_command_id,
            next_turn_ref=next_turn_ref,
            authorization_digest=authorization_digest,
        )
        return _dispatch_invalidation_receipt(receipt)

    def dispatch_transition_anchor(self, session_id: str) -> dict[str, Any]:
        require_uuid(session_id, "session_id")
        return self.store.dispatch_transition_anchor(session_id)

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
            instructions=workspace_instructions(Path(session.worktree)),
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
            instructions=workspace_instructions(Path(session.worktree)),
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
        target_provider = str(payload.get("target_provider", "")).strip()
        if target_provider and target_provider not in self.adapters:
            raise ValueError("fork target provider is unsupported")
        source = self.store.get_session(session_id)
        checkpoint = self.checkpoint(session_id)
        goal = self.store.goal_for_session(session_id)
        source_profile = str(self.store.session_safety(session_id)["profile"])
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
                    "milestones": [item.as_dict() for item in goal.milestones],
                    "budgets": goal.budgets,
                    "permitted_providers": list(goal.permitted_providers),
                    "permitted_efforts": list(goal.permitted_efforts),
                    "max_concurrency": goal.max_concurrency,
                    "completion_policy": goal.completion_policy,
                    "incident_policy": goal.incident_policy,
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
        fork_metadata: dict[str, Any] = {
            "source_session_id": session_id,
            "source_sequence": self.store.last_sequence(session_id),
            "source_checkpoint_id": str(checkpoint.get("checkpoint_id", "")),
        }
        if target_provider:
            fork_metadata["target_provider"] = target_provider
        self.store.append_event(
            forked.session_id,
            "session.forked",
            status="complete",
            metadata=fork_metadata,
        )
        if target_provider:
            self._seed_fork_handoff(source, forked, target_provider)
        return self.store.get_session(forked.session_id)

    def _seed_fork_handoff(
        self,
        source: Session,
        forked: Session,
        target_provider: str,
    ) -> None:
        """Seed a provider-targeted fork with a handoff envelope.

        The forked session's first dispatch delivers the seeded
        envelope ahead of the compiled context, in addition to the
        inherited lineage blob.
        """
        transcript = project_transcript(
            self.store,
            source.session_id,
            blobs=self.blobs,
        )
        token_budget = handoff_token_budget(
            model_context_window(
                self.scheduler.cached_models(target_provider),
                "",
            ),
            0,
            0,
        )
        block = handoff_envelope(
            session_id=forked.session_id,
            source_provider=source.active_provider,
            target_provider=target_provider,
            target_model="",
            transcript_digest=transcript.digest,
            rendered=render(transcript, RenderPolicy(token_budget=token_budget)),
        )
        self.store.append_event(
            forked.session_id,
            "session.handoff",
            status="complete",
            metadata={
                "schema": HANDOFF_SCHEMA,
                "origin": ORIGIN_FORK_SEED,
                "source_session_id": source.session_id,
                "source_provider": source.active_provider,
                "target_provider": target_provider,
                "transcript_digest": transcript.digest,
                "handoff_digest": hashlib.sha256(
                    block.encode("utf-8")
                ).hexdigest(),
                "blob_digest": self.blobs.put_text(block),
                "token_budget": token_budget,
            },
        )

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
            xhigh_authorized=int(safety["xhigh_authorizations"]) > 0,
        )
        goal = self.store.goal_for_session(session_id)
        goal_id = ""
        permitted_providers: frozenset[str] = frozenset()
        permitted_efforts: frozenset[str] = frozenset()
        max_concurrency = 1
        if goal is not None:
            goal_id = goal.goal_id
            permitted_providers = frozenset(goal.permitted_providers)
            permitted_efforts = frozenset(goal.permitted_efforts)
            max_concurrency = goal.max_concurrency
        metered_budget = _optional_number(payload.get("metered_budget"))
        if metered_budget is not None and metered_budget <= 0:
            raise ValueError("metered budget must be positive")
        decision = await self.scheduler.choose(
            session,
            workload=workload,
            required_capabilities=frozenset(
                str(item) for item in payload.get("required_capabilities", [])
            ),
            permission_mode=str(
                payload.get("permission_mode", session.permission_mode)
            ),
            provider=str(payload.get("provider", "")),
            model=str(payload.get("model", "")),
            effort=effort,
            metered_budget=metered_budget,
            binding_ceiling=limits.binding_ceiling,
            execution_profile=profile,
            enforce_concurrency=True,
            goal_id=goal_id,
            permitted_providers=permitted_providers,
            permitted_efforts=permitted_efforts,
            max_concurrency=max_concurrency,
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
            blob_values[digest] = base64.b64encode(self.blobs.get(digest)).decode(
                "ascii"
            )
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
        source_signing_public = str(payload.get("source_signing_public", ""))
        try:
            envelope = base64.b64decode(envelope_text, validate=True)
        except ValueError as error:
            raise ValueError("transfer envelope must be base64") from error
        opened = open_transfer(
            envelope,
            destination_encryption_private=(self.machine_keys.encryption_private),
            source_signing_public=source_signing_public,
        )
        if opened.get("schema") != ("p13i/agent-harness/session-transfer/v1"):
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


def _require_machines_goal_envelope(
    payload: dict[str, Any],
    *,
    objective: str,
    constraints: list[Any],
    predicates: list[Any],
    milestones: list[Any],
    permitted_providers: list[Any],
    permitted_efforts: list[Any],
    budgets: dict[str, Any],
    supported_providers: frozenset[str],
) -> None:
    required_fields = {
        "goal_kind",
        "predicates",
        "constraints",
        "milestones",
        "budgets",
        "permitted_providers",
        "permitted_efforts",
        "max_concurrency",
        "completion_policy",
        "incident_policy",
    }
    if not objective or not required_fields.issubset(payload):
        raise ValueError("p13i/machines requires a complete typed goal envelope")
    if not constraints or any(
        not isinstance(item, str) or not item.strip() for item in constraints
    ):
        raise ValueError("p13i/machines requires typed nonempty constraints")
    if not predicates or any(
        not isinstance(item, dict) or not item for item in predicates
    ):
        raise ValueError("p13i/machines requires typed evidence predicates")
    if not milestones:
        raise ValueError("p13i/machines requires at least one goal milestone")
    if any(not isinstance(item, dict) or not item for item in milestones):
        raise ValueError("p13i/machines requires typed goal milestones")
    if not permitted_providers or not permitted_efforts:
        raise ValueError("p13i/machines requires explicit routing bounds")
    if set(str(item) for item in permitted_providers) - supported_providers:
        raise ValueError("p13i/machines contains an unsupported provider")
    required_budgets = {
        "seconds",
        "turns",
        "tokens",
        "tool_calls",
        "output_tokens",
        "context_tokens",
        "dollars",
        "attempts",
        "child_agents",
    }
    if not required_budgets.issubset(budgets):
        raise ValueError("p13i/machines requires all bounded budget dimensions")


def _require_permission_restriction(current: str, requested: str) -> None:
    ranks = {
        PermissionMode.PLAN: 0,
        PermissionMode.READ_ONLY: 1,
        PermissionMode.APPROVAL: 2,
        PermissionMode.FULL: 3,
    }
    if ranks[requested] > ranks[current]:
        raise ValueError("per-turn permission mode cannot widen the session")


def _require_promotion_authorization(
    authorization: dict[str, Any],
    *,
    session_id: str,
    previous_goal_id: str,
    stage: str,
    budget_increases: dict[str, float],
    next_goal_contract_digest: str,
) -> None:
    schema = "p13i/agent-harness/goal-promotion-authorization/v1"
    if authorization.get("schema") != schema:
        raise ValueError("goal promotion authorization schema is unsupported")
    if authorization.get("session_id") != session_id:
        raise ValueError("goal promotion authorization session does not match")
    if authorization.get("from_goal_id") != previous_goal_id:
        raise ValueError("goal promotion authorization source does not match")
    if authorization.get("stage") != stage:
        raise ValueError("goal promotion authorization stage does not match")
    receipt_sha256 = str(authorization.get("receipt_sha256", ""))
    if len(receipt_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in receipt_sha256
    ):
        raise ValueError("goal promotion authorization receipt is invalid")
    _require_retained_authorization_receipt(authorization, receipt_sha256)
    if authorization.get("budget_increases") != budget_increases:
        raise ValueError("goal promotion authorization budget does not match")
    if authorization.get("next_goal_contract_digest") != next_goal_contract_digest:
        raise ValueError("goal promotion authorization contract does not match")


def _require_contract_adoption_authorization(
    authorization: dict[str, Any],
    *,
    session_id: str,
    external_ref: dict[str, str],
    previous_goal_digest: str,
    goal_envelope_digest: str,
) -> None:
    schema = "p13i/agent-harness/session-contract-adoption-authorization/v1"
    if authorization.get("schema") != schema:
        raise ValueError("contract adoption authorization schema is unsupported")
    if authorization.get("session_id") != session_id:
        raise ValueError("contract adoption authorization session does not match")
    if authorization.get("external_ref") != external_ref:
        raise ValueError("contract adoption external reference does not match")
    if authorization.get("previous_goal_digest") != previous_goal_digest:
        raise ValueError("contract adoption previous goal does not match")
    if authorization.get("goal_envelope_digest") != goal_envelope_digest:
        raise ValueError("contract adoption goal envelope does not match")
    receipt_sha256 = str(authorization.get("receipt_sha256", ""))
    if len(receipt_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in receipt_sha256
    ):
        raise ValueError("contract adoption authorization receipt is invalid")
    _require_retained_authorization_receipt(authorization, receipt_sha256)


def _require_retained_authorization_receipt(
    authorization: dict[str, Any],
    receipt_sha256: str,
) -> None:
    receipt = authorization.get("receipt")
    if not isinstance(receipt, dict) or not receipt:
        raise ValueError("authorization must retain its source receipt")
    if normalized_digest(receipt) != receipt_sha256:
        raise ValueError("authorization source receipt digest does not match")


def _require_dispatch_transition_policy(
    store: StateStore,
    *,
    session_id: str,
    session: Session,
    policy: dict[str, Any],
    policy_sha256: str,
    next_turn_ref: dict[str, str],
    transition_sequence: int,
    next_command_digest: str,
    retained_policy: bool,
) -> None:
    schema = "p13i/agent-harness/dispatch-generation-transition-policy/v1"
    if policy.get("schema") != schema:
        raise ValueError("dispatch transition policy schema is unsupported")
    if policy.get("session_id") != session_id:
        raise ValueError("dispatch transition policy session does not match")
    if policy.get("external_ref") != session.external_ref:
        raise ValueError("dispatch transition policy orchestrator does not match")
    epoch_id = str(policy.get("epoch_id", ""))
    if (
        not epoch_id
        or len(epoch_id) > 128
        or any(character.isspace() for character in epoch_id)
    ):
        raise ValueError("dispatch transition policy epoch is invalid")
    allowed_roles = policy.get("allowed_agent_roles")
    allowed_prefixes = policy.get("allowed_step_prefixes")
    max_transitions = policy.get("max_transitions")
    if not isinstance(allowed_roles, list) or not allowed_roles:
        raise ValueError("dispatch transition policy roles are invalid")
    if not isinstance(allowed_prefixes, list) or not allowed_prefixes:
        raise ValueError("dispatch transition policy step prefixes are invalid")
    if (
        isinstance(max_transitions, bool)
        or not isinstance(max_transitions, int)
        or max_transitions < 1
        or max_transitions > 1_000
    ):
        raise ValueError("dispatch transition policy limit is invalid")
    if next_turn_ref["agent_role"] not in allowed_roles:
        raise ValueError("dispatch transition role is outside policy")
    step_id = next_turn_ref["step_id"]
    if not any(
        isinstance(prefix, str) and prefix and step_id.startswith(prefix)
        for prefix in allowed_prefixes
    ):
        raise ValueError("dispatch transition step is outside policy")
    if transition_sequence > max_transitions:
        raise ValueError("dispatch transition exceeds policy limit")
    transitions = policy.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("dispatch transition policy stages are invalid")
    if max_transitions != len(transitions):
        raise ValueError("dispatch transition policy limit does not match stages")
    if transition_sequence > len(transitions):
        raise ValueError("dispatch transition sequence is outside policy")
    matching_transition = transitions[transition_sequence - 1]
    if not isinstance(matching_transition, dict):
        raise ValueError("dispatch transition policy stage is invalid")
    if matching_transition.get("sequence") != transition_sequence:
        raise ValueError("dispatch transition policy stages are not ordered")
    if not retained_policy:
        for index, transition in enumerate(transitions, start=1):
            if not isinstance(transition, dict):
                raise ValueError("dispatch transition policy stage is invalid")
            if transition.get("sequence") != index:
                raise ValueError("dispatch transition policy stages are not ordered")
            transition_ref = normalize_turn_ref(transition.get("next_turn_ref"))
            if transition_ref["agent_role"] not in allowed_roles:
                raise ValueError("dispatch transition policy stage role is invalid")
            transition_step = transition_ref["step_id"]
            if not any(
                isinstance(prefix, str)
                and prefix
                and transition_step.startswith(prefix)
                for prefix in allowed_prefixes
            ):
                raise ValueError("dispatch transition policy stage step is invalid")
            transition_digest = str(transition.get("next_command_digest", ""))
            if len(transition_digest) != 64 or any(
                character not in "0123456789abcdef" for character in transition_digest
            ):
                raise ValueError("dispatch transition policy stage digest is invalid")
    if normalize_turn_ref(matching_transition.get("next_turn_ref")) != next_turn_ref:
        raise ValueError("dispatch transition stage is outside policy")
    command_digest = str(matching_transition.get("next_command_digest", ""))
    if len(command_digest) != 64 or command_digest != next_command_digest:
        raise ValueError("dispatch transition policy command digest is invalid")
    goal = store.goal_for_session(session_id)
    if goal is None:
        raise ValueError("dispatch transition requires an active goal")
    constraint = "dispatch-generation-transition-policy-sha256:" + policy_sha256
    if constraint not in goal.constraints:
        raise ValueError("dispatch transition policy is not goal-authorized")
    epoch_constraint = "dispatch-generation-transition-epoch:" + epoch_id
    if epoch_constraint not in goal.constraints:
        raise ValueError("dispatch transition epoch is not goal-authorized")


def _goal_promotion_receipt(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "promotion_id": str(value.get("promotion_id", "")),
        "session_id": str(value.get("session_id", "")),
        "previous_goal_id": str(value.get("previous_goal_id", "")),
        "next_goal_id": str(value.get("next_goal_id", "")),
        "stage": str(value.get("stage", "")),
        "authorization_digest": str(value.get("authorization_digest", "")),
        "request_digest": str(value.get("request_digest", "")),
        "previous_goal_digest": str(value.get("previous_goal_digest", "")),
        "next_goal_digest": str(value.get("next_goal_digest", "")),
        "created_at": str(value.get("created_at", "")),
    }


def _goal_contract_adoption_receipt(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "adoption_id": str(value.get("adoption_id", "")),
        "session_id": str(value.get("session_id", "")),
        "previous_goal_id": str(value.get("previous_goal_id", "")),
        "next_goal_id": str(value.get("next_goal_id", "")),
        "external_ref": {
            "orchestrator": str(value.get("external_orchestrator", "")),
            "job_id": str(value.get("external_job_id", "")),
        },
        "authorization_digest": str(value.get("authorization_digest", "")),
        "request_digest": str(value.get("request_digest", "")),
        "creation_digest": str(value.get("creation_digest", "")),
        "previous_goal_digest": str(value.get("previous_goal_digest", "")),
        "next_goal_digest": str(value.get("next_goal_digest", "")),
        "created_at": str(value.get("created_at", "")),
    }


def _dispatch_invalidation_receipt(value: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "invalidation_id": str(value.get("invalidation_id", "")),
        "session_id": str(value.get("session_id", "")),
        "reason": str(value.get("reason", "")),
        "authorization_digest": str(value.get("authorization_digest", "")),
        "request_digest": str(value.get("request_digest", "")),
        "created_at": str(value.get("created_at", "")),
    }
    prior_command_id = str(value.get("prior_command_id", ""))
    if prior_command_id:
        receipt["prior_command_id"] = prior_command_id
        receipt["prior_command_type"] = str(value.get("prior_command_type", ""))
        receipt["prior_anchor_kind"] = str(value.get("prior_anchor_kind", ""))
        receipt["prior_reconciliation_id"] = str(
            value.get("prior_reconciliation_id", "")
        )
        receipt["prior_reconciliation_resolution"] = str(
            value.get("prior_reconciliation_resolution", "")
        )
        receipt["next_turn_ref"] = normalize_turn_ref(value.get("next_turn_ref"))
        receipt["prior_checkpoint_id"] = str(value.get("prior_checkpoint_id", ""))
        receipt["prior_generation_digest"] = str(
            value.get("prior_generation_digest", "")
        )
        receipt["prior_material_digest"] = str(value.get("prior_material_digest", ""))
        receipt["goal_id"] = str(value.get("goal_id", ""))
        receipt["transition_sequence"] = int(value.get("transition_sequence", 0))
        receipt["epoch_id"] = str(value.get("epoch_id", ""))
        receipt["policy_sha256"] = str(value.get("policy_sha256", ""))
        receipt["next_command_digest"] = str(value.get("next_command_digest", ""))
    return receipt


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
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("numeric value must be finite")
        return normalized
    return None


def _lease_expiry() -> str:
    value = datetime.datetime.now(datetime.UTC)
    value += datetime.timedelta(seconds=90)
    return value.isoformat()
