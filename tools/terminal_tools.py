"""
Kalki Nexus - Terminal Tools

Real (but unsandboxed) shell command execution, registered under the
"terminal" category. Harden this before exposing it to untrusted input:
allow-list commands, drop privileges, or run inside a disposable container.
"""
from __future__ import annotations

import asyncio

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

# TODO: allow-list permitted commands and/or run inside a disposable container
# before exposing this tool to untrusted agents or users.
DEFAULT_TIMEOUT_SECONDS = 30


@ToolRegistry.register()
class RunShellCommandTool(BaseTool):
    name = "run_shell_command"
    description = "Run a shell command and return its combined stdout/stderr output."
    category = "terminal"
    required_permissions = [Permission.TERMINAL]

    async def run(self, command: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            return f"command timed out after {timeout}s: {command}"
        return stdout.decode("utf-8", errors="replace")
