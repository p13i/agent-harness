"""Process-level smoke for the real terminal chat entry point."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import pty
import select
import sqlite3
import struct
import subprocess
import tempfile
import termios
import time


STARTUP_TIMEOUT_SECONDS = 20


def test_chat_starts_and_quits_in_a_real_pty(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _repository(workspace)
    temporary_state = tempfile.TemporaryDirectory(
        prefix="agent-harness-pty-",
        dir="/tmp",
    )
    state = Path(temporary_state.name)
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
        _wait_for_socket(
            state / ".runtime" / "control.sock",
            daemon,
            timeout=STARTUP_TIMEOUT_SECONDS,
        )
        primary, replica = pty.openpty()
        terminal_attributes = termios.tcgetattr(replica)
        terminal_attributes[0] &= ~termios.IXON
        termios.tcsetattr(replica, termios.TCSANOW, terminal_attributes)
        environment = dict(os.environ)
        environment["TERM"] = "xterm-256color"

        def configure_child_terminal() -> None:
            os.setsid()
            fcntl.ioctl(replica, termios.TIOCSCTTY, 0)

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
            preexec_fn=configure_child_terminal,
        )
        output = _read_until(
            primary,
            b"P13I AGENT HARNESS",
            timeout=STARTUP_TIMEOUT_SECONDS,
        )
        terminal_attributes = termios.tcgetattr(replica)
        terminal_attributes[0] &= ~termios.IXON
        termios.tcsetattr(replica, termios.TCSANOW, terminal_attributes)
        os.write(primary, b"\x1bOR")
        output += _read_until(primary, b"Control", timeout=5)
        fcntl.ioctl(
            replica,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", 20, 60, 0, 0),
        )
        time.sleep(0.5)
        os.write(primary, b"\x11")
        return_code, trailing_output = _wait_for_exit(
            chat,
            primary,
            timeout=5,
        )
        output += trailing_output
        assert return_code == 0
        os.close(replica)
        replica = -1
        output += _read_available(primary)
        assert b"P13I AGENT HARNESS" in output
        assert b"Traceback" not in output
        assert b"signal only works in main thread" not in output

        with sqlite3.connect(state / ".runtime" / "state.sqlite3") as db:
            row = db.execute(
                "SELECT session_id FROM sessions "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()
        assert row is not None
        session_id = str(row[0])

        os.close(primary)
        primary = -1
        primary, replica = pty.openpty()
        terminal_attributes = termios.tcgetattr(replica)
        terminal_attributes[0] &= ~termios.IXON
        termios.tcsetattr(replica, termios.TCSANOW, terminal_attributes)
        chat = subprocess.Popen(
            [
                str(launcher),
                "--state-dir",
                str(state),
                "--cwd",
                str(workspace),
                "chat",
                "resume",
                session_id,
            ],
            stdin=replica,
            stdout=replica,
            stderr=replica,
            env=environment,
            preexec_fn=configure_child_terminal,
        )
        resumed_output = _read_until(
            primary,
            b"Control",
            timeout=STARTUP_TIMEOUT_SECONDS,
        )
        os.write(primary, b"\x11")
        return_code, trailing_output = _wait_for_exit(
            chat,
            primary,
            timeout=5,
        )
        resumed_output += trailing_output
        assert return_code == 0
        assert b"Traceback" not in resumed_output
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
        temporary_state.cleanup()


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
    *,
    timeout: float = 8,
) -> None:
    deadline = time.monotonic() + timeout
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


def _wait_for_exit(
    process: subprocess.Popen[bytes],
    descriptor: int,
    *,
    timeout: float,
) -> tuple[int, bytes]:
    deadline = time.monotonic() + timeout
    output = b""
    while time.monotonic() < deadline:
        output += _read_available(descriptor)
        return_code = process.poll()
        if return_code is not None:
            return return_code, output
        time.sleep(0.02)
    raise AssertionError(
        "chat did not exit after Ctrl+Q: " + repr(output[-4000:])
    )


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
