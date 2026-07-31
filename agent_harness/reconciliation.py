"""Provider-neutral reconciliation for ambiguous provider dispatches."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import subprocess
from typing import Any

from agent_harness.blobs import BlobStore
from agent_harness.errors import ConflictError
from agent_harness.errors import HarnessError
from agent_harness.models import Attention
from agent_harness.models import Lifecycle
from agent_harness.models import ReconciliationDecision
from agent_harness.models import ReconciliationRecord
from agent_harness.models import ReconciliationStatus
from agent_harness.models import RestartRecovery
from agent_harness.storage import StateStore
from agent_harness.workspace import checkpoint_workspace
from agent_harness.workspace import restore_checkpoint
from agent_harness.workspace import workspace_summary


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
        if recovery.reconciliations:
            self.store.update_session(
                session_id,
                attention=Attention.NEEDS_RECONCILIATION,
            )
        return recovery

    async def resolve(
        self,
        reconciliation_id: str,
        decision: str,
        observed_workspace_digest: str,
        *,
        audit: dict[str, Any] | None = None,
        approved: bool = False,
    ) -> ReconciliationRecord:
        record = self.store.reconciliation(reconciliation_id)
        selected = _decision(decision)
        if record.status == ReconciliationStatus.RESOLVED:
            if (
                record.resolution == selected
                and record.current_workspace_digest
                == observed_workspace_digest
            ):
                self._apply_session_resolution(record, selected)
                return record
            raise ConflictError(
                "reconciliation was already resolved differently"
            )
        if record.current_workspace_digest != observed_workspace_digest:
            raise ConflictError("observed workspace digest is stale")
        session = self.store.get_session(record.session_id)
        resuming_resolution = (
            record.status == ReconciliationStatus.RESOLVING
        )
        if resuming_resolution and record.resolution != selected:
            raise ConflictError(
                "reconciliation resolution is already in progress"
            )
        if not resuming_resolution:
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
                raise ConflictError(
                    "workspace changed during reconciliation"
                )
        resolution_audit: dict[str, Any] = {}
        if audit is not None:
            resolution_audit.update(audit)
        if selected == ReconciliationDecision.ACCEPT_CURRENT:
            pre_dispatch = self.store.checkpoint(
                record.pre_dispatch_checkpoint_id
            )
            checkpoint = await asyncio.to_thread(
                checkpoint_workspace,
                session,
                self.blobs,
                sequence=self.store.last_sequence(session.session_id),
                provider=pre_dispatch.provider,
                native_session_id=pre_dispatch.native_session_id,
                context_text="",
            )
            self.store.add_checkpoint(checkpoint)
            resolution_audit["checkpoint_id"] = checkpoint.checkpoint_id
        elif selected == ReconciliationDecision.RESTORE_PRE_TURN:
            checkpoint = self.store.checkpoint(
                record.pre_dispatch_checkpoint_id
            )
            await asyncio.to_thread(
                _restore_exact,
                Path(session.worktree),
                checkpoint,
                self.blobs,
            )
            restored = await asyncio.to_thread(
                checkpoint_workspace,
                session,
                self.blobs,
                sequence=self.store.last_sequence(session.session_id),
                provider=checkpoint.provider,
                native_session_id=checkpoint.native_session_id,
                context_text="",
            )
            self.store.add_checkpoint(restored)
            resolution_audit["checkpoint_id"] = restored.checkpoint_id
            resolution_audit["restored_checkpoint_id"] = (
                checkpoint.checkpoint_id
            )
        resolution_audit["observed_workspace_digest"] = (
            observed_workspace_digest
        )
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


def inspect_workspace(workspace: Path) -> tuple[str, str]:
    root = workspace.resolve()
    digest = hashlib.sha256()
    digest.update(_git(root, "rev-parse", "HEAD"))
    digest.update(b"\0tracked\0")
    digest.update(_git(root, "diff", "--binary", "--no-ext-diff", "HEAD"))
    digest.update(b"\0untracked\0")
    untracked = _git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    for raw_path in sorted(
        item for item in untracked.split(b"\0") if item
    ):
        relative = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        candidate = root / relative
        if candidate.is_symlink():
            continue
        target = candidate.resolve()
        if not target.is_relative_to(root):
            continue
        if not target.is_file():
            continue
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(target.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), workspace_summary(root)


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
