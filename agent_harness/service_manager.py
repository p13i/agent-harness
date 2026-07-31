"""Hermetic systemd user-service generation and lifecycle helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import pwd
import re
import subprocess
import uuid


SERVICE_NAME = "p13i-agent-harness.service"
UNIT_VERSION = 1
Runner = Callable[..., subprocess.CompletedProcess[str]]


class ServiceManagerError(RuntimeError):
    """Raised when a user-service mutation cannot complete."""


@dataclass(frozen=True)
class UnitConfiguration:
    executable: Path
    state_dir: Path
    build_id: str


@dataclass(frozen=True)
class ServiceStatus:
    active: bool
    installed: bool
    unit_version: int | None
    build_id: str
    detail: str


@dataclass(frozen=True)
class DiagnosticProbe:
    name: str
    status: str
    detail: str
    remediation: str = ""


def default_unit_path(home: Path | None = None) -> Path:
    """Return the agent-harness-owned systemd user unit path."""
    selected_home = home
    if selected_home is None:
        selected_home = Path.home()
    return selected_home / ".config" / "systemd" / "user" / SERVICE_NAME


def render_unit(configuration: UnitConfiguration) -> str:
    """Render a deterministic user unit with no network listener."""
    executable = configuration.executable.expanduser()
    state_dir = configuration.state_dir.expanduser()
    if not executable.is_absolute() or not state_dir.is_absolute():
        raise ValueError("service executable and state directory must be absolute")
    build_id = _safe_metadata(configuration.build_id, "build identifier")
    executable_argument = _systemd_quote(str(executable))
    state_argument = _systemd_quote(str(state_dir))
    return (
        "# Managed by p13i/agent-harness.\n"
        + "# X-P13I-Agent-Harness-Unit-Version: "
        + str(UNIT_VERSION)
        + "\n"
        + "# X-P13I-Agent-Harness-Build-ID: "
        + build_id
        + "\n"
        + "[Unit]\n"
        + "Description=p13i agent harness\n"
        + "After=default.target\n"
        + "\n"
        + "[Service]\n"
        + "Type=simple\n"
        + "ExecStart="
        + executable_argument
        + " --state-dir "
        + state_argument
        + " service run\n"
        + "Restart=on-failure\n"
        + "RestartSec=2\n"
        + "KillMode=control-group\n"
        + "TimeoutStopSec=30\n"
        + "UMask=0077\n"
        + "NoNewPrivileges=true\n"
        + "PrivateTmp=true\n"
        + "\n"
        + "[Install]\n"
        + "WantedBy=default.target\n"
    )


def unit_metadata(content: str) -> tuple[int | None, str]:
    """Read the unit version and build identifier from managed comments."""
    version_prefix = "# X-P13I-Agent-Harness-Unit-Version: "
    build_prefix = "# X-P13I-Agent-Harness-Build-ID: "
    version: int | None = None
    build_id = ""
    for line in content.splitlines():
        if line.startswith(version_prefix):
            raw_version = line[len(version_prefix) :]
            try:
                version = int(raw_version)
            except ValueError:
                version = None
        if line.startswith(build_prefix):
            build_id = line[len(build_prefix) :]
    return version, build_id


class SystemdUserService:
    """Manage only the harness-owned systemd user unit."""

    def __init__(
        self,
        unit_path: Path | None = None,
        *,
        runner: Runner = subprocess.run,
        user_id: int | None = None,
        service_name: str = SERVICE_NAME,
    ) -> None:
        selected_path = unit_path
        if selected_path is None:
            selected_path = default_unit_path()
        self.unit_path = selected_path.expanduser()
        self.runner = runner
        self.service_name = _service_name(service_name)
        selected_user_id = user_id
        if selected_user_id is None:
            selected_user_id = os.getuid()
        self.user_id = selected_user_id

    def install(self, configuration: UnitConfiguration) -> None:
        """Atomically install, reload, and activate the user unit definition."""
        content = render_unit(configuration)
        previous: bytes | None = None
        if self.unit_path.exists():
            previous = self.unit_path.read_bytes()
        _atomic_write(self.unit_path, content, 0o600)
        try:
            self._systemctl("daemon-reload")
            self._systemctl("enable", self.service_name)
        except BaseException:
            if previous is None:
                self.unit_path.unlink(missing_ok=True)
            else:
                _atomic_write_bytes(self.unit_path, previous, 0o600)
            self._systemctl("daemon-reload", required=False)
            raise

    def start(self) -> None:
        self._systemctl("start", self.service_name)

    def restart(self) -> None:
        self._systemctl("restart", self.service_name)

    def stop(self) -> None:
        self._systemctl("stop", self.service_name)

    def status(self) -> ServiceStatus:
        """Return service state without mutating the unit or daemon."""
        installed = self.unit_path.is_file()
        version: int | None = None
        build_id = ""
        if installed:
            try:
                content = self.unit_path.read_text(encoding="utf-8")
                version, build_id = unit_metadata(content)
            except OSError:
                installed = False
        result = self._systemctl(
            "is-active",
            self.service_name,
            required=False,
        )
        detail = result.stdout.strip()
        if not detail:
            detail = result.stderr.strip()
        active = result.returncode == 0 and detail == "active"
        return ServiceStatus(
            active=active,
            installed=installed,
            unit_version=version,
            build_id=build_id,
            detail=detail,
        )

    def uninstall(self) -> None:
        """Remove only the user unit and preserve all chat state."""
        self._systemctl(
            "disable",
            "--now",
            self.service_name,
            required=False,
        )
        if not self.unit_path.exists():
            self._systemctl("daemon-reload")
            return
        removed = self.unit_path.with_name(
            "." + self.unit_path.name + ".removed-" + uuid.uuid4().hex
        )
        self.unit_path.replace(removed)
        try:
            self._systemctl("daemon-reload")
            removed.unlink()
            _sync_directory(self.unit_path.parent)
        except BaseException:
            removed.replace(self.unit_path)
            self._systemctl("daemon-reload", required=False)
            raise

    def diagnostics(self) -> tuple[DiagnosticProbe, ...]:
        """Run read-only user-systemd, lingering, and unit probes."""
        return (
            self._probe_user_systemd(),
            self._probe_lingering(),
            self._probe_unit(),
        )

    def _probe_user_systemd(self) -> DiagnosticProbe:
        result = self._run(
            ["systemctl", "--user", "show-environment"],
            required=False,
        )
        if result.returncode == 0:
            return DiagnosticProbe(
                "user-systemd",
                "pass",
                "systemd user manager is available",
            )
        return DiagnosticProbe(
            "user-systemd",
            "fail",
            "systemd user manager is unavailable",
            "Start a systemd user session, then run "
            "systemctl --user daemon-reload.",
        )

    def _probe_lingering(self) -> DiagnosticProbe:
        result = self._run(
            [
                "loginctl",
                "show-user",
                str(self.user_id),
                "--property=Linger",
                "--value",
            ],
            required=False,
        )
        if result.returncode != 0:
            return DiagnosticProbe(
                "user-lingering",
                "warning",
                "user lingering status is unavailable",
                "Inspect it with: loginctl show-user "
                + str(self.user_id)
                + " --property=Linger.",
            )
        if result.stdout.strip().casefold() == "yes":
            return DiagnosticProbe(
                "user-lingering",
                "pass",
                "user lingering is active",
            )
        user_name = pwd.getpwuid(self.user_id).pw_name
        return DiagnosticProbe(
            "user-lingering",
            "warning",
            "user lingering is inactive",
            "An administrator can activate it with: "
            "loginctl enable-linger "
            + user_name
            + ".",
        )

    def _probe_unit(self) -> DiagnosticProbe:
        if not self.unit_path.is_file():
            return DiagnosticProbe(
                "service-unit",
                "warning",
                "agent-harness user unit is not installed",
                "Run: agent-harness service install.",
            )
        try:
            content = self.unit_path.read_text(encoding="utf-8")
        except OSError:
            return DiagnosticProbe(
                "service-unit",
                "fail",
                "agent-harness user unit cannot be read",
                "Reinstall it with: agent-harness service install.",
            )
        version, build_id = unit_metadata(content)
        if version != UNIT_VERSION or not build_id:
            return DiagnosticProbe(
                "service-unit",
                "warning",
                "agent-harness user unit version is incompatible",
                "Update it with: agent-harness service install.",
            )
        return DiagnosticProbe(
            "service-unit",
            "pass",
            "agent-harness user unit version "
            + str(version)
            + " selects build "
            + build_id,
        )

    def _systemctl(
        self,
        *arguments: str,
        required: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["systemctl", "--user", *arguments],
            required=required,
        )

    def _run(
        self,
        command: list[str],
        *,
        required: bool,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self.runner(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            if not required:
                return subprocess.CompletedProcess(
                    command,
                    127,
                    "",
                    str(error),
                )
            raise ServiceManagerError(
                "service command could not be executed: " + command[0]
            ) from error
        if required and result.returncode != 0:
            raise ServiceManagerError(
                "service command failed: " + " ".join(command[:3])
            )
        return result


def _systemd_quote(value: str) -> str:
    if not value or "\0" in value or "\n" in value or "\r" in value:
        raise ValueError("systemd argument is empty or contains control data")
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace('"', '\\"')
    escaped = escaped.replace("%", "%%")
    return '"' + escaped + '"'


def _safe_metadata(value: str, name: str) -> str:
    if not value or "\0" in value or "\n" in value or "\r" in value:
        raise ValueError(name + " is empty or contains control data")
    return value


def _service_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", value):
        raise ValueError("systemd service name is invalid")
    return value


def _atomic_write(path: Path, content: str, mode: int) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"), mode)


def _atomic_write_bytes(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(
        "." + path.name + ".new-" + uuid.uuid4().hex
    )
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        temporary.replace(path)
        _sync_directory(path.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
