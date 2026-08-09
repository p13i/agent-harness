"""Provider-neutral reconciliation for ambiguous provider dispatches."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_harness.blobs import BlobStore
from agent_harness.errors import ConflictError, HarnessError
from agent_harness.models import (
    Attention,
    Lifecycle,
    ReconciliationDecision,
    ReconciliationRecord,
    ReconciliationStatus,
    RestartRecovery,
)
from agent_harness.storage import StateStore
from agent_harness.workspace import (
    checkpoint_workspace,
    restore_checkpoint,
)
from agent_harness.workspace_state import inspect_workspace

SYSTEM_RECONCILIATION_AUDIT_KEYS = frozenset(
    {
        "checkpoint_id",
        "current_workspace_digest",
        "discovery_checkpoint_id",
        "discovery_workspace_digest",
        "dispatch_identity",
        "observed_workspace_digest",
        "pre_dispatch_checkpoint_id",
        "resolution_checkpoint_id",
        "resolution_workspace_digest",
        "restored_checkpoint_id",
        "topology_receipt",
    }
)


def validate_reconciliation_audit(audit: dict[str, Any] | None) -> None:
    if audit is None:
        return
    protected = sorted(SYSTEM_RECONCILIATION_AUDIT_KEYS.intersection(audit))
    if protected:
        raise ValueError(
            "reconciliation audit contains system-owned fields: " + ", ".join(protected)
        )


class ReconciliationManager:
    def __init__(self, store: StateStore, blobs: BlobStore) -> None:
        self.store = store
        self.blobs = blobs

    def inspect(self, reconciliation_id: str) -> ReconciliationRecord:
        return self.store.reconciliation(reconciliation_id)

    async def recover_after_restart(
        self,
        session_id: str,
    ) -> RestartRecovery:
        session = self.store.get_session(session_id)
        workspace = Path(session.worktree)
        digest, summary = await asyncio.to_thread(
            inspect_workspace,
            workspace,
        )
        recovery = self.store.recover_interrupted_commands(
            session_id,
            digest,
            summary,
        )
        reconciliations: list[ReconciliationRecord] = []
        for record in recovery.reconciliations:
            reconciliations.append(await self._ensure_discovery_checkpoint(record))
        recovery = replace(
            recovery,
            reconciliations=tuple(reconciliations),
        )
        if recovery.reconciliations:
            self.store.update_session(
                session_id,
                attention=Attention.NEEDS_RECONCILIATION,
            )
        return recovery

    async def _ensure_discovery_checkpoint(
        self,
        record: ReconciliationRecord,
    ) -> ReconciliationRecord:
        checkpoint_id = str(record.audit.get("discovery_checkpoint_id", ""))
        if checkpoint_id:
            self.store.checkpoint(checkpoint_id)
            return record
        session = self.store.get_session(record.session_id)
        workspace = Path(session.worktree)
        before_digest, unused_summary = await asyncio.to_thread(
            inspect_workspace,
            workspace,
        )
        del unused_summary
        if before_digest != record.current_workspace_digest:
            raise ConflictError(
                "workspace changed before reconciliation discovery checkpoint"
            )
        pre_dispatch = self.store.checkpoint(record.pre_dispatch_checkpoint_id)
        checkpoint = await asyncio.to_thread(
            checkpoint_workspace,
            session,
            self.blobs,
            sequence=self.store.last_sequence(session.session_id),
            provider=pre_dispatch.provider,
            native_session_id=pre_dispatch.native_session_id,
            context_text="",
        )
        after_digest, unused_summary = await asyncio.to_thread(
            inspect_workspace,
            workspace,
        )
        del unused_summary
        if after_digest != record.current_workspace_digest:
            raise ConflictError(
                "workspace changed during reconciliation discovery checkpoint"
            )
        return self.store.record_reconciliation_discovery(
            record.reconciliation_id,
            checkpoint,
            record.current_workspace_digest,
        )

    async def resolve(
        self,
        reconciliation_id: str,
        decision: str,
        observed_workspace_digest: str,
        *,
        audit: dict[str, Any] | None = None,
        approved: bool = False,
        idempotency_key: str = "",
        request_digest: str = "",
        operation: str = "",
    ) -> ReconciliationRecord:
        validate_reconciliation_audit(audit)
        record = self.store.reconciliation(reconciliation_id)
        if record.status != ReconciliationStatus.RESOLVED and not str(
            record.audit.get("discovery_checkpoint_id", "")
        ):
            record = await self._ensure_discovery_checkpoint(record)
        selected = _decision(decision)
        if record.status == ReconciliationStatus.RESOLVED:
            if (
                record.resolution == selected
                and record.current_workspace_digest == observed_workspace_digest
            ):
                _require_resolution_workspace_digest(record, selected)
                replayed = None
                if idempotency_key:
                    # Settle the key before projecting: a reused or
                    # conflicting key must leave a stranded command
                    # exactly as it found it.
                    replayed = self.store.mutation_receipt(
                        idempotency_key,
                        operation,
                        request_digest,
                    )
                if replayed is None:
                    self.store.project_resolved_reconciliation(reconciliation_id)
                if idempotency_key:
                    resolved, unused_created = self.store.resolve_reconciliation_once(
                        reconciliation_id,
                        selected,
                        observed_workspace_digest,
                        record.audit,
                        None,
                        idempotency_key=idempotency_key,
                        operation=operation,
                        request_digest=request_digest,
                    )
                    del unused_created
                    return resolved
                self._apply_session_resolution(record, selected)
                # The projection rewrote the topology receipt, so the
                # record read before it is already stale.
                return self.store.reconciliation(reconciliation_id)
            raise ConflictError("reconciliation was already resolved differently")
        if record.current_workspace_digest != observed_workspace_digest:
            raise ConflictError("observed workspace digest is stale")
        session = self.store.get_session(record.session_id)
        resuming_resolution = record.status == ReconciliationStatus.RESOLVING
        if resuming_resolution and record.resolution != selected:
            raise ConflictError("reconciliation resolution is already in progress")
        if not resuming_resolution:
            if selected != ReconciliationDecision.STOP:
                current_digest, unused_summary = await asyncio.to_thread(
                    inspect_workspace,
                    Path(session.worktree),
                )
                del unused_summary
                if current_digest != observed_workspace_digest:
                    raise ConflictError("workspace changed after inspection")
            if selected == ReconciliationDecision.RESTORE_PRE_TURN:
                _require_restore_permission(
                    session.permission_mode,
                    approved=approved,
                )
            record = self.store.begin_reconciliation_resolution(
                reconciliation_id,
                selected,
                observed_workspace_digest,
            )
        elif selected == ReconciliationDecision.ACCEPT_CURRENT:
            current_digest, unused_summary = await asyncio.to_thread(
                inspect_workspace,
                Path(session.worktree),
            )
            del unused_summary
            if current_digest != observed_workspace_digest:
                raise ConflictError("workspace changed during reconciliation")
        resolution_audit: dict[str, Any] = dict(record.audit)
        if audit is not None:
            resolution_audit.update(audit)
        resolution_checkpoint = None
        resolution_workspace_digest = ""
        if selected == ReconciliationDecision.ACCEPT_CURRENT:
            pre_dispatch = self.store.checkpoint(record.pre_dispatch_checkpoint_id)
            checkpoint = await asyncio.to_thread(
                checkpoint_workspace,
                session,
                self.blobs,
                sequence=self.store.last_sequence(session.session_id),
                provider=pre_dispatch.provider,
                native_session_id=pre_dispatch.native_session_id,
                context_text="",
            )
            resolution_workspace_digest, unused_summary = await asyncio.to_thread(
                inspect_workspace,
                Path(session.worktree),
            )
            del unused_summary
            if resolution_workspace_digest != observed_workspace_digest:
                raise ConflictError(
                    "workspace changed during reconciliation resolution checkpoint"
                )
            resolution_audit["checkpoint_id"] = checkpoint.checkpoint_id
            resolution_audit["resolution_checkpoint_id"] = checkpoint.checkpoint_id
            resolution_checkpoint = checkpoint
        elif selected == ReconciliationDecision.RESTORE_PRE_TURN:
            checkpoint = self.store.checkpoint(record.pre_dispatch_checkpoint_id)
            await asyncio.to_thread(
                _restore_exact,
                Path(session.worktree),
                checkpoint,
                self.blobs,
            )
            restored_workspace_digest, unused_summary = await asyncio.to_thread(
                inspect_workspace,
                Path(session.worktree),
            )
            del unused_summary
            restored = await asyncio.to_thread(
                checkpoint_workspace,
                session,
                self.blobs,
                sequence=self.store.last_sequence(session.session_id),
                provider=checkpoint.provider,
                native_session_id=checkpoint.native_session_id,
                context_text="",
            )
            resolution_workspace_digest, unused_summary = await asyncio.to_thread(
                inspect_workspace,
                Path(session.worktree),
            )
            del unused_summary
            if resolution_workspace_digest != restored_workspace_digest:
                raise ConflictError(
                    "workspace changed during restored resolution checkpoint"
                )
            resolution_audit["checkpoint_id"] = restored.checkpoint_id
            resolution_audit["resolution_checkpoint_id"] = restored.checkpoint_id
            resolution_audit["restored_checkpoint_id"] = checkpoint.checkpoint_id
            resolution_checkpoint = restored
        resolution_audit["observed_workspace_digest"] = observed_workspace_digest
        if resolution_workspace_digest:
            resolution_audit["resolution_workspace_digest"] = (
                resolution_workspace_digest
            )
        if idempotency_key:
            resolved, unused_created = self.store.resolve_reconciliation_once(
                reconciliation_id,
                selected,
                observed_workspace_digest,
                resolution_audit,
                resolution_checkpoint,
                idempotency_key=idempotency_key,
                operation=operation,
                request_digest=request_digest,
            )
            del unused_created
            return resolved
        if resolution_checkpoint is not None:
            self.store.add_checkpoint(resolution_checkpoint)
        resolved = self.store.resolve_reconciliation_record(
            reconciliation_id,
            selected,
            observed_workspace_digest,
            resolution_audit,
        )
        self._apply_session_resolution(resolved, selected)
        return resolved

    def _apply_session_resolution(
        self,
        record: ReconciliationRecord,
        decision: str,
    ) -> None:
        if decision == ReconciliationDecision.STOP:
            self.store.update_session(
                record.session_id,
                lifecycle=Lifecycle.STOPPED,
                attention=Attention.IDLE,
            )
            return
        self.store.update_session(
            record.session_id,
            lifecycle=Lifecycle.RUNNING,
            attention=Attention.IDLE,
        )


def _restore_exact(
    workspace: Path,
    checkpoint,
    blobs: BlobStore,
) -> None:
    _git(workspace, "reset", "--hard", checkpoint.base_commit)
    _git(workspace, "clean", "-fd")
    restore_checkpoint(workspace, checkpoint, blobs)


def _git(workspace: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise HarnessError(
            "E_GIT",
            "Git operation failed during reconciliation",
            status=409,
        )
    return completed.stdout


def _decision(value: str) -> str:
    try:
        return str(ReconciliationDecision(value))
    except ValueError as error:
        raise ValueError("reconciliation decision is unsupported") from error


def _require_resolution_workspace_digest(
    record: ReconciliationRecord,
    decision: str,
) -> str:
    if decision not in {
        ReconciliationDecision.ACCEPT_CURRENT,
        ReconciliationDecision.RESTORE_PRE_TURN,
    }:
        return ""
    value = str(record.audit.get("resolution_workspace_digest", ""))
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ConflictError(
            "resolved reconciliation lacks certified workspace material"
        )
    return value


def _require_restore_permission(
    permission_mode: str,
    *,
    approved: bool,
) -> None:
    if permission_mode == "full":
        return
    if permission_mode == "approval" and approved:
        return
    if permission_mode == "approval":
        raise HarnessError(
            "E_APPROVAL_REQUIRED",
            "restoring the pre-turn checkpoint requires approval",
            status=409,
        )
    raise HarnessError(
        "E_PERMISSION",
        "session permission mode does not permit workspace restore",
        status=409,
    )
