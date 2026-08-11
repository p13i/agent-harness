"""Grok Build CLI adapter over headless ``streaming-json`` runs.

Grok has no npm package. The official binary is installed to
``~/.grok/bin/grok`` (or ``~/.local/bin/grok``) via
``https://x.ai/cli/install.sh``. This adapter launches that binary
one-shot in headless mode — the same shape as the Kimi adapter, not
Claude's SDK or Codex's long-lived app-server.

Deliberate v1 limits:

- ``steer`` stays the base no-op; interrupt terminates the process group.
- Only harness permission mode ``full`` is accepted (mapped to
  ``--always-approve``).
- A positive child-agent limit cannot be metered; zero is accepted and
  enforced with ``--no-subagents``.
- Usage capacity is auth-readiness only (see ``usage._probe_grok``), not
  a real rate-limit window.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import tempfile

from agent_harness.providers.base import ApprovalHandler
from agent_harness.providers.base import ChildLaunchGate
from agent_harness.providers.base import EventHandler
from agent_harness.providers.base import PrePromptGate
from agent_harness.providers.base import ProviderAdapter
from agent_harness.providers.base import ProviderEvent
from agent_harness.providers.base import ProviderModel
from agent_harness.providers.base import ProviderResult
from agent_harness.providers.base import ProviderStatus
from agent_harness.providers.base import provider_environment
from agent_harness.providers.normalize import grok_payload
from agent_harness.process_control import ProcessGroupIdentity
from agent_harness.process_control import process_group_identity
from agent_harness.process_control import terminate_process_group

# Argv payloads above this size use --prompt-file instead of -p.
_PROMPT_ARGV_LIMIT = 8192
_SUBPROCESS_STREAM_LIMIT = 16 * 1024 * 1024

_DEFAULT_EFFORTS = ("low", "medium", "high", "xhigh")

# Wire types that exist only once Grok is running the turn.
_TURN_STARTED_TYPES = frozenset(
    {
        "end",
        "text",
        "thought",
        "tool_call",
        "tool_call_update",
        "usage",
    }
)


def _turn_started(payload: dict[str, object]) -> bool:
    """Report whether one Grok line proves the turn itself began.

    The harness treats acceptance as proof the provider holds the
    instruction, and stops resending it elsewhere on that basis, so it
    has to come from the turn's own output. ``error`` is terminal and
    reports a failure rather than a run (see ``grok_payload``), and a
    type the normalizer does not recognize is as likely to be startup
    metadata as turn output. Neither is evidence, so neither accepts.
    """

    return str(payload.get("type", "")) in _TURN_STARTED_TYPES


def resolve_grok_binary() -> str | None:
    """Return the first trusted grok binary path, or None."""
    found = shutil.which("grok")
    if found:
        return found
    home = Path.home()
    for candidate in (
        home / ".grok" / "bin" / "grok",
        home / ".local" / "bin" / "grok",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _launch_argv(
    binary: str,
    *,
    prompt: str,
    model: str,
    effort: str,
    session_id: str,
    prompt_file: Path | None = None,
) -> list[str]:
    argv = [
        binary,
        "--always-approve",
        "--no-subagents",
        "--output-format",
        "streaming-json",
    ]
    if model:
        argv.extend(["-m", model])
    if effort:
        argv.extend(["--reasoning-effort", effort])
    if session_id:
        argv.extend(["-r", session_id])
    if prompt_file is not None:
        argv.extend(["--prompt-file", str(prompt_file)])
    else:
        argv.extend(["-p", prompt])
    return argv


class GrokAdapter(ProviderAdapter):
    provider_id = "grok"

    def __init__(self) -> None:
        self._active_process: asyncio.subprocess.Process | None = None
        self._process_group: ProcessGroupIdentity | None = None

    async def run_turn(
        self,
        *,
        workspace: Path,
        prompt: str,
        native_session_id: str,
        permission_mode: str,
        model: str,
        effort: str,
        event_handler: EventHandler,
        approval_handler: ApprovalHandler,
        child_launch_gate: ChildLaunchGate | None = None,
        pre_prompt_gate: PrePromptGate | None = None,
    ) -> ProviderResult:
        del approval_handler
        if permission_mode != "full":
            raise RuntimeError("Grok cannot map the requested permission mode")
        if child_launch_gate is not None and child_launch_gate.limit != 0:
            raise RuntimeError(
                "Grok cannot meter a positive child-agent limit"
            )

        binary = resolve_grok_binary()
        if not binary:
            raise RuntimeError("Grok binary was not found on PATH")

        if pre_prompt_gate is not None:
            await pre_prompt_gate()

        prompt_file_path: Path | None = None
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if len(prompt) > _PROMPT_ARGV_LIMIT:
            temp_dir = tempfile.TemporaryDirectory(prefix="grok-prompt-")
            prompt_file_path = Path(temp_dir.name) / "prompt.txt"
            prompt_file_path.write_text(prompt, encoding="utf-8")

        argv = _launch_argv(
            binary,
            prompt=prompt,
            model=model,
            effort=effort,
            session_id=native_session_id,
            prompt_file=prompt_file_path,
        )
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(workspace),
            env=provider_environment(self.provider_id),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=_SUBPROCESS_STREAM_LIMIT,
        )
        pid = int(getattr(process, "pid", 0) or 0)
        if pid > 0:
            try:
                self._process_group = process_group_identity(pid)
            except BaseException:
                process.terminate()
                await process.wait()
                if temp_dir is not None:
                    temp_dir.cleanup()
                raise
        self._active_process = process

        session_id = native_session_id
        status = "complete"
        usage: dict[str, object] = {}
        prompt_accepted = False
        try:
            assert process.stdout is not None
            async for line in process.stdout:
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if not prompt_accepted and _turn_started(payload):
                    # The first line of turn output is the earliest
                    # proof Grok is running the prompt. Without it the
                    # harness treats every Grok guard stop as
                    # pre-acceptance and may resend the instruction to
                    # another provider after Grok already ran tools.
                    prompt_accepted = True
                    await event_handler(
                        ProviderEvent(
                            "provider.prompt.accepted",
                            status="accepted",
                            native_session_id=session_id,
                        )
                    )
                for event in grok_payload(payload):
                    if event.native_session_id:
                        session_id = event.native_session_id
                    if event.event_type == "turn.failed":
                        status = "failed"
                    if (
                        event.event_type == "turn.completed"
                        and event.metadata
                        and isinstance(event.metadata.get("usage"), dict)
                    ):
                        usage = dict(event.metadata["usage"])
                    await event_handler(event)

            stderr = b""
            if process.stderr is not None:
                stderr = await process.stderr.read()
            returncode = await process.wait()
            if returncode != 0:
                status = "failed"
                await event_handler(
                    ProviderEvent(
                        "turn.failed",
                        text=stderr.decode("utf-8", "replace").strip(),
                        status="failed",
                        native_session_id=session_id,
                    )
                )
        finally:
            self._active_process = None
            self._process_group = None
            if temp_dir is not None:
                temp_dir.cleanup()

        return ProviderResult(
            provider=self.provider_id,
            native_session_id=session_id,
            native_turn_id="",
            status=status,
            usage=usage,
        )

    async def interrupt(self) -> None:
        process = self._active_process
        identity = self._process_group
        if process is None or identity is None:
            return
        await terminate_process_group(process, identity)

    def process_identity(self) -> tuple[int, str]:
        process = self._active_process
        identity = self._process_group
        if process is None or identity is None:
            return (0, "")
        if process.returncode is not None:
            return (0, "")
        if process.pid != identity.pid:
            return (0, "")
        return (identity.pid, identity.pid_start)

    async def models(self, workspace: Path) -> tuple[ProviderModel, ...]:
        del workspace
        return (
            ProviderModel(
                model_id="grok-4.5",
                display_name="Grok 4.5",
                efforts=_DEFAULT_EFFORTS,
                context_window=None,
                default=True,
            ),
        )

    def status(self) -> ProviderStatus:
        binary = resolve_grok_binary()
        ready = binary is not None
        detail = "grok binary was not found"
        if ready:
            detail = (
                "one-shot headless streaming-json; full permission mode "
                "only; child agents disabled via --no-subagents; "
                "capacity is auth-readiness only"
            )
        return ProviderStatus(
            provider=self.provider_id,
            ready=ready,
            detail=detail,
            capabilities=frozenset(
                {
                    "resume",
                    "streaming",
                    "worktree",
                }
            ),
        )
