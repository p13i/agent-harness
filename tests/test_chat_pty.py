"""Process-level smoke for the real terminal chat entry point."""

from __future__ import annotations

import os
from pathlib import Path
import pty
import select
import subprocess
import time


def test_chat_starts_and_quits_in_a_real_pty(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _repository(workspace)
    state = tmp_path / "state"
    launcher = _launcher()
    daemon = subprocess.Popen(
        [str(launcher), "--state-dir", str(state), "daemon"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    chat: subprocess.Popen[bytes] | None = None
    primary = -1
    replica = -1
    try:
        _wait_for_socket(state / "control.sock", daemon)
        primary, replica = pty.openpty()
        environment = dict(os.environ)
        environment["TERM"] = "xterm-256color"
        chat = subprocess.Popen(
            [
                str(launcher),
                "--state-dir",
                str(state),
                "--cwd",
                str(workspace),
                "chat",
            ],
            stdin=replica,
            stdout=replica,
            stderr=replica,
            env=environment,
            start_new_session=True,
        )
        os.close(replica)
        replica = -1
        output = _read_until(primary, b"P13I AGENT HARNESS", timeout=8)
        os.write(primary, b"\x11")
        assert chat.wait(timeout=5) == 0
        output += _read_available(primary)
        assert b"P13I AGENT HARNESS" in output
        assert b"Traceback" not in output
        assert b"signal only works in main thread" not in output
    finally:
        if replica >= 0:
            os.close(replica)
        if primary >= 0:
            os.close(primary)
        if chat is not None and chat.poll() is None:
            chat.terminate()
            chat.wait(timeout=5)
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def _launcher() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "_main")
    candidate = runfiles / workspace / "cmd" / "agent-harness"
    if candidate.is_file():
        return candidate
    fallback = runfiles / "_main" / "cmd" / "agent-harness"
    if fallback.is_file():
        return fallback
    raise AssertionError("agent-harness launcher is absent from runfiles")


def _wait_for_socket(
    socket: Path,
    daemon: subprocess.Popen[bytes],
) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if socket.exists():
            return
        if daemon.poll() is not None:
            output = daemon.stdout
            detail = b""
            if output is not None:
                detail = output.read()
            raise AssertionError("daemon exited before ready: " + repr(detail))
        time.sleep(0.05)
    raise AssertionError("daemon socket did not become ready")


def _read_until(
    descriptor: int,
    marker: bytes,
    *,
    timeout: float,
) -> bytes:
    deadline = time.monotonic() + timeout
    content = b""
    while time.monotonic() < deadline:
        content += _read_available(descriptor)
        if marker in content:
            return content
        time.sleep(0.02)
    raise AssertionError(
        "terminal marker was not rendered: " + repr(content[-2000:])
    )


def _read_available(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        ready, unused, errors = select.select(
            [descriptor],
            [],
            [descriptor],
            0,
        )
        del unused
        if errors or not ready:
            break
        try:
            chunk = os.read(descriptor, 65_536)
        except OSError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
    )
    (path / "file.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "file.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "initial"],
        check=True,
    )
