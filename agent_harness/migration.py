"""Offline migration from the legacy hidden harness state root."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time
from typing import Any

from agent_harness.config import HarnessPaths
from agent_harness.config import paths
from agent_harness.config import prepare_paths
from agent_harness.records import load_portable_records
from agent_harness.records import materialize_all
from agent_harness.storage import StateStore
from agent_harness.sync import sync_repository


@dataclass(frozen=True)
class LegacyPaths:
    root: Path
    database: Path
    blobs: Path
    worktrees: Path
    exports: Path
    logs: Path
    secrets: Path
    socket: Path
    daemon_pid: Path


@dataclass(frozen=True)
class DestinationBackup:
    root: Path
    had_database: bool


def legacy_paths(root: Path) -> LegacyPaths:
    resolved = root.expanduser().resolve()
    return LegacyPaths(
        root=resolved,
        database=resolved / "state.sqlite3",
        blobs=resolved / "blobs",
        worktrees=resolved / "worktrees",
        exports=resolved / "exports",
        logs=resolved / "logs",
        secrets=resolved / "secrets",
        socket=resolved / "control.sock",
        daemon_pid=resolved / "daemon.pid",
    )


def stop_legacy_processes(source: LegacyPaths) -> None:
    _stop_managed_processes(source.root, source.daemon_pid)


def _stop_managed_processes(root: Path, pid_path: Path) -> None:
    candidates = _legacy_processes(root)
    daemon_pid = _read_pid(pid_path)
    if daemon_pid is not None:
        candidates.add(daemon_pid)
    for pid in sorted(candidates):
        if not _is_managed_legacy_process(pid, root):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        remaining = {
            pid
            for pid in candidates
            if _is_managed_legacy_process(pid, root)
        }
        if not remaining:
            return
        time.sleep(0.1)
    raise RuntimeError("legacy harness processes did not stop")


def migrate_state(
    source_root: Path,
    destination_root: Path,
    *,
    trash_source: bool,
) -> dict[str, Any]:
    source = legacy_paths(source_root)
    destination = paths(destination_root)
    _validate_roots(source, destination)
    stop_legacy_processes(source)
    _require_quiescent(source)
    _stop_managed_processes(
        destination.state_dir,
        destination.daemon_pid,
    )
    _require_root_quiescent(destination.state_dir)
    prepare_paths(destination)
    _require_headroom(source.root, destination.state_dir)
    source_store = StateStore(source.database)
    moved: list[tuple[Path, Path, Path]] = []
    backup = _create_destination_backup(destination)
    try:
        source_inventory = _inventory(source_store, source)
        destination_store = StateStore(destination.database)
        try:
            destination_inventory = _inventory(
                destination_store,
                destination,
            )
            preserved = {
                session.session_id: destination_store.portable_session(
                    session.session_id
                )
                for session in destination_store.list_sessions(
                    include_archived=True
                )
            }
            records = [
                source_store.portable_session(session.session_id)
                for session in source_store.list_sessions(
                    include_archived=True
                )
            ]
            destination_store.merge_portable(
                records,
                source_store.portable_global(),
            )
            destination_store.clear_runtime_state()
            _copy_tree_verified(source.blobs, destination.blobs)
            _copy_tree_verified(source.exports, destination.exports)
            _copy_tree_verified(
                source.logs,
                destination.logs / "legacy",
            )
            _copy_file_if_missing(
                source.secrets / "api-token",
                destination.token,
            )
            _copy_file_if_missing(
                source.secrets / "machine-keys.json",
                destination.machine_keys,
            )
            _move_worktrees(
                source_store,
                source,
                destination,
                moved,
            )
            _rewrite_worktrees(
                destination_store,
                source.worktrees,
                destination.worktrees,
            )
            merged_inventory = _inventory(
                destination_store,
                destination,
            )
            _verify_merged_inventory(
                source_inventory,
                destination_inventory,
                merged_inventory,
                source.worktrees,
                destination.worktrees,
            )
            _verify_source_sessions(
                destination_store,
                records,
                source.worktrees,
                destination.worktrees,
            )
            _verify_preserved_sessions(
                destination_store,
                preserved,
            )
            materialize_all(destination, destination_store)
            _verify_portable_round_trip(
                destination,
                destination_store,
            )
            synchronized = sync_repository(destination)
            if synchronized.get("state") != "synced":
                raise RuntimeError(
                    "chat repository did not synchronize successfully"
                )
        finally:
            destination_store.close()
    except BaseException:
        _rollback_worktrees(moved)
        _restore_destination_backup(destination, backup)
        raise
    finally:
        source_store.close()
    _remove_destination_backup(backup)
    trashed = ""
    if trash_source:
        trashed = str(_trash_source(source.root))
    return {
        "source": str(source.root),
        "destination": str(destination.state_dir),
        "sessions": source_inventory["sessions"],
        "events": source_inventory["events"],
        "worktrees": len(moved),
        "source_trashed": bool(trashed),
        "trash_path": trashed,
    }


def _validate_roots(
    source: LegacyPaths,
    destination: HarnessPaths,
) -> None:
    if source.root == destination.state_dir:
        raise ValueError("source and destination must differ")
    if not source.database.is_file():
        raise ValueError("legacy state database does not exist")
    if not destination.state_dir.is_dir():
        raise ValueError("destination chat repository does not exist")
    git = _run(
        ["git", "-C", str(destination.state_dir), "rev-parse", "--git-dir"],
        check=False,
    )
    if git.returncode != 0:
        raise ValueError("destination is not a Git repository")


def _create_destination_backup(
    destination: HarnessPaths,
) -> DestinationBackup:
    root = destination.runtime / "migration-rollback"
    if root.exists():
        raise RuntimeError(
            "an incomplete destination migration backup already exists"
        )
    root.mkdir(parents=True, mode=0o700)
    had_database = destination.database.is_file()
    if had_database:
        source = sqlite3.connect(destination.database)
        target = sqlite3.connect(root / "state.sqlite3")
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
    directories = {
        "sessions": destination.sessions,
        "blobs": destination.blobs,
        "exports": destination.exports,
        "logs": destination.logs,
        "secrets": destination.token.parent,
    }
    for name, source_path in directories.items():
        if source_path.is_dir():
            shutil.copytree(source_path, root / name)
    files = {
        "global.gpt.json": destination.state_dir / "global.gpt.json",
        "sync-status.json": destination.sync_status,
    }
    for name, source_path in files.items():
        if source_path.is_file():
            shutil.copy2(source_path, root / name)
    return DestinationBackup(root=root, had_database=had_database)


def _restore_destination_backup(
    destination: HarnessPaths,
    backup: DestinationBackup,
) -> None:
    directories = {
        "sessions": destination.sessions,
        "blobs": destination.blobs,
        "exports": destination.exports,
        "logs": destination.logs,
        "secrets": destination.token.parent,
    }
    for name, destination_path in directories.items():
        if destination_path.is_dir():
            shutil.rmtree(destination_path)
        source_path = backup.root / name
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
    database_files = (
        destination.database,
        Path(str(destination.database) + "-shm"),
        Path(str(destination.database) + "-wal"),
    )
    for path in database_files:
        if path.exists():
            path.unlink()
    if backup.had_database:
        shutil.copy2(
            backup.root / "state.sqlite3",
            destination.database,
        )
    files = {
        "global.gpt.json": destination.state_dir / "global.gpt.json",
        "sync-status.json": destination.sync_status,
    }
    for name, destination_path in files.items():
        if destination_path.exists():
            destination_path.unlink()
        source_path = backup.root / name
        if source_path.is_file():
            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
                mode=0o700,
            )
            shutil.copy2(source_path, destination_path)
    _remove_destination_backup(backup)


def _remove_destination_backup(backup: DestinationBackup) -> None:
    if backup.root.is_dir():
        shutil.rmtree(backup.root)


def _copy_tree_verified(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_symlink():
            raise RuntimeError(
                "migration source contains an unsupported symlink"
            )
        if source_path.is_dir():
            destination_path.mkdir(
                parents=True,
                exist_ok=True,
                mode=0o700,
            )
            continue
        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )
        if destination_path.exists():
            if _file_digest(source_path) != _file_digest(destination_path):
                raise RuntimeError(
                    "migration file conflicts with destination: "
                    + str(relative)
                )
            continue
        shutil.copy2(source_path, destination_path)


def _copy_file_if_missing(source: Path, destination: Path) -> None:
    if not source.is_file() or destination.exists():
        return
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o700,
    )
    shutil.copy2(source, destination)


def _move_worktrees(
    store: StateStore,
    source: LegacyPaths,
    destination: HarnessPaths,
    moved: list[tuple[Path, Path, Path]],
) -> None:
    for session in store.list_sessions(include_archived=True):
        current = Path(session.worktree)
        if current.parent.resolve() != source.worktrees.resolve():
            continue
        target = destination.worktrees / current.name
        if not current.is_dir():
            raise RuntimeError(
                "session worktree is missing: " + session.session_id
            )
        if target.exists():
            raise RuntimeError(
                "destination worktree already exists: "
                + session.session_id
            )
        workspace = Path(session.workspace)
        shutil.move(str(current), str(target))
        repaired = _run(
            [
                "git",
                "-C",
                str(workspace),
                "worktree",
                "repair",
                str(target),
            ],
            check=False,
        )
        if repaired.returncode != 0:
            shutil.move(str(target), str(current))
            _run(
                [
                    "git",
                    "-C",
                    str(workspace),
                    "worktree",
                    "repair",
                    str(current),
                ],
                check=False,
            )
            raise RuntimeError(
                "Git worktree repair failed: " + session.session_id
            )
        moved.append((current, target, workspace))
        status = _run(
            ["git", "-C", str(target), "status", "--porcelain=v1"],
            check=False,
        )
        if status.returncode != 0:
            raise RuntimeError(
                "migrated worktree is invalid: " + session.session_id
            )


def _rollback_worktrees(
    moved: list[tuple[Path, Path, Path]],
) -> None:
    for source, destination, workspace in reversed(moved):
        if not destination.exists() or source.exists():
            continue
        shutil.move(str(destination), str(source))
        _run(
            [
                "git",
                "-C",
                str(workspace),
                "worktree",
                "repair",
                str(source),
            ],
            check=False,
        )


def _rewrite_worktrees(
    store: StateStore,
    source: Path,
    destination: Path,
) -> None:
    store.rewrite_worktree_prefix(
        str(source.resolve()),
        str(destination.resolve()),
    )


def _inventory(
    store: StateStore,
    storage: LegacyPaths | HarnessPaths,
) -> dict[str, Any]:
    sessions = store.list_sessions(include_archived=True)
    session_values = []
    events = 0
    for session in sorted(sessions, key=lambda item: item.session_id):
        sequence = store.last_sequence(session.session_id)
        events += len(store.all_events(session.session_id))
        session_values.append(
            {
                "session_id": session.session_id,
                "worktree": session.worktree,
                "last_sequence": sequence,
                "portable_digest": _digest(
                    store.portable_session(session.session_id)
                ),
            }
        )
    blobs = storage.blobs
    return {
        "sessions": len(sessions),
        "events": events,
        "session_values": session_values,
        "blob_hashes": _file_hashes(blobs),
    }


def _verify_merged_inventory(
    source: dict[str, Any],
    destination_before: dict[str, Any],
    destination_after: dict[str, Any],
    source_worktrees: Path,
    destination_worktrees: Path,
) -> None:
    expected_sessions = (
        int(source["sessions"])
        + int(destination_before["sessions"])
    )
    if expected_sessions != destination_after["sessions"]:
        raise RuntimeError("session count changed during migration")
    expected_events = (
        int(source["events"])
        + int(destination_before["events"])
    )
    if expected_events != destination_after["events"]:
        raise RuntimeError("event count changed during migration")
    source_sessions = source["session_values"]
    destination_sessions = destination_before["session_values"]
    merged_sessions = destination_after["session_values"]
    if not isinstance(source_sessions, list):
        raise RuntimeError("source inventory is invalid")
    if not isinstance(destination_sessions, list):
        raise RuntimeError("destination inventory is invalid")
    if not isinstance(merged_sessions, list):
        raise RuntimeError("merged inventory is invalid")
    normalized_source = _normalize_inventory_worktrees(
        source_sessions,
        source_worktrees,
        destination_worktrees,
    )
    normalized_destination = _normalize_inventory_worktrees(
        destination_sessions,
        destination_worktrees,
        destination_worktrees,
    )
    normalized_merged = _normalize_inventory_worktrees(
        merged_sessions,
        destination_worktrees,
        destination_worktrees,
    )
    expected_values = sorted(
        normalized_source + normalized_destination,
        key=lambda item: str(item.get("session_id", "")),
    )
    if expected_values != normalized_merged:
        raise RuntimeError("session content changed during migration")
    merged_blobs = destination_after["blob_hashes"]
    if not isinstance(merged_blobs, dict):
        raise RuntimeError("merged blob inventory is invalid")
    for inventory in (
        source["blob_hashes"],
        destination_before["blob_hashes"],
    ):
        if not isinstance(inventory, dict):
            raise RuntimeError("source blob inventory is invalid")
        for name, digest in inventory.items():
            if merged_blobs.get(name) != digest:
                raise RuntimeError("blob content changed during migration")


def _verify_source_sessions(
    store: StateStore,
    records: list[dict[str, Any]],
    source_worktrees: Path,
    destination_worktrees: Path,
) -> None:
    source_text = str(source_worktrees.resolve())
    destination_text = str(destination_worktrees.resolve())
    for record in records:
        expected = json.loads(json.dumps(record))
        tables = expected.get("tables", {})
        if not isinstance(tables, dict):
            raise RuntimeError("portable source tables are invalid")
        sessions = tables.get("sessions", [])
        if not isinstance(sessions, list) or len(sessions) != 1:
            raise RuntimeError("portable source session is invalid")
        session = sessions[0]
        if not isinstance(session, dict):
            raise RuntimeError("portable source session is invalid")
        worktree = str(session.get("worktree", ""))
        if worktree.startswith(source_text + os.sep):
            session["worktree"] = (
                destination_text + worktree[len(source_text) :]
            )
        session_id = str(record.get("session_id", ""))
        if store.portable_session(session_id) != expected:
            raise RuntimeError(
                "portable source session changed during migration"
            )


def _verify_preserved_sessions(
    store: StateStore,
    preserved: dict[str, dict[str, Any]],
) -> None:
    for session_id, expected in preserved.items():
        if store.portable_session(session_id) != expected:
            raise RuntimeError(
                "destination session changed during migration"
            )


def _normalize_inventory_worktrees(
    sessions: list[dict[str, Any]],
    source: Path,
    destination: Path,
) -> list[dict[str, Any]]:
    normalized = []
    source_text = str(source.resolve())
    destination_text = str(destination.resolve())
    for session in sessions:
        value = dict(session)
        worktree = str(value.get("worktree", ""))
        if worktree.startswith(source_text + os.sep):
            value["worktree"] = (
                destination_text + worktree[len(source_text) :]
            )
        value.pop("portable_digest", None)
        normalized.append(value)
    return normalized


def _verify_portable_round_trip(
    destination: HarnessPaths,
    store: StateStore,
) -> None:
    records, global_record = load_portable_records(destination)
    verification = destination.runtime / "portable-verify.sqlite3"
    for suffix in ("", "-shm", "-wal"):
        candidate = Path(str(verification) + suffix)
        if candidate.exists():
            candidate.unlink()
    restored = StateStore(verification)
    try:
        restored.import_portable(records, global_record)
        for session in store.list_sessions(include_archived=True):
            expected = store.portable_session(session.session_id)
            actual = restored.portable_session(session.session_id)
            if expected != actual:
                raise RuntimeError(
                    "portable round trip changed session "
                    + session.session_id
                )
        if store.portable_global() != restored.portable_global():
            raise RuntimeError("portable round trip changed global state")
    finally:
        restored.close()
        for suffix in ("", "-shm", "-wal"):
            candidate = Path(str(verification) + suffix)
            if candidate.exists():
                candidate.unlink()


def _require_quiescent(source: LegacyPaths) -> None:
    _require_root_quiescent(source.root)


def _require_root_quiescent(root: Path) -> None:
    active = {
        pid
        for pid in _legacy_processes(root)
        if _is_managed_legacy_process(pid, root)
    }
    if active:
        raise RuntimeError("legacy harness state is still in use")


def _legacy_processes(root: Path) -> set[int]:
    completed = _run(["ps", "-axo", "pid=,command="], check=False)
    result: set[int] = set()
    root_text = str(root)
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        if root_text not in fields[1]:
            continue
        if "agent-harness" not in fields[1]:
            continue
        try:
            result.add(int(fields[0]))
        except ValueError:
            continue
    return result


def _is_managed_legacy_process(pid: int, root: Path) -> bool:
    completed = _run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
    )
    command = completed.stdout
    return (
        completed.returncode == 0
        and "agent-harness" in command
        and str(root) in command
        and (" daemon" in command or " worker " in command)
    )


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if value <= 1:
        return None
    return value


def _require_headroom(source: Path, destination: Path) -> None:
    required = _tree_size(source) * 2 + 256 * 1024**2
    free = shutil.disk_usage(destination).free
    if free < required:
        raise RuntimeError("insufficient disk space for verified migration")


def _tree_size(root: Path) -> int:
    total = 0
    for item in root.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def _file_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    result = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item.is_symlink():
            continue
        relative = str(item.relative_to(root))
        result[relative] = _file_digest(item)
    return result


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trash_source(source: Path) -> Path:
    trash = Path.home() / ".Trash"
    trash.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    destination = trash / ("p13i-agent-harness-" + timestamp)
    if destination.exists():
        raise RuntimeError("migration Trash destination already exists")
    shutil.move(str(source), str(destination))
    return destination


def _run(
    command: list[str],
    *,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=check,
        timeout=30,
    )
