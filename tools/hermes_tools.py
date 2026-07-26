"""
Kalki Nexus - Hermes Agent Profile Integration Tools

Invokes Hermes Agent profiles running inside the `hermes-dashboard` Docker
container using the correct CLI syntax:
    hermes -p <profile_name> -z "<prompt>" --cli

This gives Kalki Nexus full access to each Hermes profile's 69+ skills,
tools, memory, and learned knowledge.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import List

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

HERMES_CONTAINER_NAME = os.getenv("HERMES_CONTAINER_NAME", "hermes-dashboard")
HERMES_BIN = "/opt/hermes/.venv/bin/hermes"
PROFILES_DIR = Path(os.getenv("HERMES_PROFILES_DIR", "/home/azureuser/.hermes/profiles"))


@ToolRegistry.register()
class ListHermesProfilesTool(BaseTool):
    name = "list_hermes_profiles"
    description = "List all custom Hermes Agent profiles available in the Hermes Agent environment."
    category = "hermes"
    required_permissions = [Permission.TERMINAL]

    async def run(self) -> List[str]:
        # Check profiles directory on host if accessible
        if PROFILES_DIR.exists() and PROFILES_DIR.is_dir():
            profiles = [p.name for p in PROFILES_DIR.iterdir() if p.is_dir()]
            if profiles:
                return sorted(list(set(profiles)))

        # Fallback: exec inside docker container
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", HERMES_CONTAINER_NAME,
            HERMES_BIN, "profile", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        # Parse profile names from `hermes profile list` output
        profiles = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("Gateway"):
                continue
            # Lines look like: "✗ ai_architect  — not running"
            match = re.match(r'^[✓✗⚡●○]\s+(\S+)', line)
            if match:
                profiles.append(match.group(1))
        return sorted(profiles)


@ToolRegistry.register()
class RunHermesProfileTool(BaseTool):
    name = "run_hermes_profile"
    description = "Execute a one-shot query against a specific Hermes Agent profile (with its custom skills, tools, and memory)."
    category = "hermes"
    required_permissions = [Permission.TERMINAL]

    async def run(self, profile: str, prompt: str) -> str:
        """Execute a prompt using: hermes -p <profile> -z "<prompt>" --cli"""
        cmd = [
            "docker", "exec", HERMES_CONTAINER_NAME,
            HERMES_BIN,
            "-p", profile,
            "-z", prompt,
            "--cli",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            output = stdout.decode("utf-8", errors="replace").strip()
            if not output and stderr:
                output = stderr.decode("utf-8", errors="replace").strip()
            return output if output else f"Hermes profile '{profile}' returned no output."
        except asyncio.TimeoutError:
            proc.kill()
            return f"Hermes profile '{profile}' execution timed out after 180s."
