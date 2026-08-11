"""Kimi Code CLI adapter over a one-shot ``--output-format stream-json`` run.

Unlike the Claude and Codex adapters there is no SDK and no long-lived
server here. Kimi exposes an Agent Client Protocol mode (``kimi acp``)
and a local REST server (``kimi web``), but both are heavier surfaces
than this harness needs, and the subprocess path shares one wire format
with ``tools/ingest_agent.gpt.py`` in the consuming repo.

Three consequences of that choice, all deliberate:

- ``steer`` stays the base class's no-op. The active one-shot process can
  still be interrupted and is contained in an isolated process group.
- ``--yolo`` is NOT passed. Kimi rejects it together with ``--prompt``.
- Restrictive harness permission modes stay unmappable (only ``full``
  is accepted) and a positive child-agent limit cannot be metered, so
  ``run_turn`` raises for those. A zero child-agent limit is accepted:
  containment comes from deny rules for the Agent/AgentSwarm tools in
  the host's ``~/.kimi-code/config.toml`` rather than from the harness
  gate. Tool calls are accounted: the stream-json ``tool_calls`` and
  ``role: "tool"`` lines normalize to the canonical tool events.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil

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
from agent_harness.providers.normalize import kimi_payload
from agent_harness.providers.normalize import payload_text
from agent_harness.process_control import ProcessGroupIdentity
from agent_harness.process_control import process_group_identity
from agent_harness.process_control import terminate_process_group

KIMI_CODE_PACKAGE = "@moonshot-ai/kimi-code"

# Roles whose lines carry turn content rather than run metadata.
_TURN_STARTED_ROLES = frozenset({"assistant", "tool", "user"})
_RESUME_HINT = "session.resume_hint"


def _carries_tool_calls(payload: dict[str, object]) -> bool:
    """Report whether a line carries an OpenAI-shaped ``tool_calls`` array.

    Mirrors the gate the normalizer applies before it mints tool starts
    (see ``_kimi_tool_starts``), so a line this accepts is a line that
    reaches the worker as tool activity.
    """

    tool_calls = payload.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    for call in tool_calls:
        if isinstance(call, dict):
            return True
    return False


def _turn_started(payload: dict[str, object]) -> bool:
    """Report whether one Kimi line proves the turn itself began.

    The harness treats acceptance as proof the provider holds the
    instruction, and stops resending it elsewhere on that basis, so it
    has to come from the turn's own output. A launched process is not
    that: an unconfigured model alias fails the run before a turn ever
    starts (see ``models``). Neither is a ``meta`` line, which reports
    on the run rather than on the turn -- except for the closing resume
    hint, which only a session that ran emits.

    Kimi announces a tool call on an assistant line whose ``content``
    is empty, so text alone is not the test. That line is the one that
    matters most: a turn stopped by a guard right after running a tool
    must not be resent to another provider. An assistant line with
    neither text nor tool calls remains an empty frame and proves
    nothing.
    """

    role = str(payload.get("role", ""))
    if role == "meta":
        return str(payload.get("type", "")) == _RESUME_HINT
    if role not in _TURN_STARTED_ROLES:
        return False
    if payload_text(payload.get("content")):
        return True
    return _carries_tool_calls(payload)


def _launch_argv(prompt: str, model: str, session_id: str) -> list[str]:
    argv = [
        "npx",
        "--yes",
        "--package",
        KIMI_CODE_PACKAGE,
        "kimi",
        "--prompt",
        prompt,
        "--output-format",
        "stream-json",
    ]
    if model:
        argv.extend(["--model", model])
    if session_id:
        argv.extend(["--session", session_id])
    return argv


class KimiAdapter(ProviderAdapter):
    provider_id = "kimi"

    def __init__(self) -> None:
        self._active_process: asyncio.subprocess.Process | None = None
        self._process_group: ProcessGroupIdentity | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

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
        # Kimi Code exposes no reasoning-effort control. Its one-shot
        # prompt surface also cannot map restrictive harness permission
        # modes or meter a positive child-agent limit; a zero limit is
        # accepted because containment then comes from the host config's
        # Agent/AgentSwarm deny rules.
        del effort, approval_handler
        if permission_mode != "full":
            raise RuntimeError("Kimi cannot map the requested permission mode")
        if child_launch_gate is not None and child_launch_gate.limit != 0:
            raise RuntimeError("Kimi cannot meter a positive child-agent limit")

        if pre_prompt_gate is not None:
            await pre_prompt_gate()

        argv = _launch_argv(prompt, model, native_session_id)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(workspace),
            env=provider_environment(self.provider_id),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pid = int(getattr(process, "pid", 0) or 0)
        if pid > 0:
            try:
                self._process_group = process_group_identity(pid)
            except BaseException:
                process.terminate()
                await process.wait()
                raise
        self._active_process = process
        self._cleanup_task = None

        session_id = native_session_id
        status = "complete"
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
                    # proof Kimi is running the prompt. Without it the
                    # harness treats every Kimi guard stop as
                    # pre-acceptance and may resend the instruction to
                    # another provider after Kimi already ran tools.
                    prompt_accepted = True
                    await event_handler(
                        ProviderEvent(
                            "provider.prompt.accepted",
                            status="accepted",
                            native_session_id=session_id,
                        )
                    )
                for event in kimi_payload(payload):
                    if event.native_session_id:
                        session_id = event.native_session_id
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
            identity = self._process_group
            if identity is not None:
                await self._terminate_active_process(process, identity)
            self._active_process = None
            self._process_group = None
            self._cleanup_task = None

        return ProviderResult(
            provider=self.provider_id,
            native_session_id=session_id,
            native_turn_id="",
            status=status,
            usage={},
        )

    async def interrupt(self) -> None:
        process = self._active_process
        identity = self._process_group
        if process is None or identity is None:
            return
        await self._terminate_active_process(process, identity)

    async def _terminate_active_process(
        self,
        process: asyncio.subprocess.Process,
        identity: ProcessGroupIdentity,
    ) -> None:
        cleanup = self._cleanup_task
        if cleanup is None:
            cleanup = asyncio.create_task(
                terminate_process_group(process, identity)
            )
            self._cleanup_task = cleanup
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

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
        # These ids go straight into `--model`, which resolves an alias
        # key in ~/.kimi-code/config.toml -- the [models."<alias>"]
        # table name -- not the upstream API model id. Kimi's managed
        # OAuth provider namespaces its aliases, so a bare "k3" matched
        # nothing and every turn failed with:
        #   config.invalid: Model "k3" is not configured in config.toml
        # Ids, display names, and context windows below are the values
        # `kimi provider list --json` reports for those aliases.
        del workspace
        return (
            ProviderModel(
                model_id="kimi-code/k3",
                display_name="K3",
                efforts=(),
                context_window=262_144,
                default=True,
            ),
            ProviderModel(
                model_id="kimi-code/kimi-for-coding",
                display_name="K2.7 Coding",
                efforts=(),
                context_window=262_144,
            ),
        )

    def status(self) -> ProviderStatus:
        npx_available = shutil.which("npx") is not None
        ready = npx_available
        detail = "npx was not found"
        if npx_available:
            detail = (
                "one-shot run; full permission mode only, "
                "child agents contained by host config denies"
            )
        return ProviderStatus(
            provider=self.provider_id,
            ready=ready,
            detail=detail,
            # The one-shot adapter advertises only behavior its stream
            # can observe. No steering exists on a one-shot run.
            capabilities=frozenset(
                {
                    "resume",
                    "streaming",
                    "tools",
                    "worktree",
                }
            ),
        )
