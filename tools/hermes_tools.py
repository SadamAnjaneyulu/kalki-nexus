"""
Kalki Nexus - Hermes Agent Profile Integration Tools

BaseTool wrappers for invoking Hermes Agent profiles running inside the
`hermes-dashboard` / `hermes` Docker container. Allows Kalki Nexus agents
to delegate tasks to custom, skill-rich Hermes profiles.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

HERMES_CONTAINER_NAME = os.getenv("HERMES_CONTAINER_NAME", "hermes-dashboard")
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
            profiles = [p.stem for p in PROFILES_DIR.glob("*.yaml")] + [p.name for p in PROFILES_DIR.iterdir() if p.is_dir()]
            if profiles:
                return sorted(list(set(profiles)))

        # Fallback: exec inside docker container
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", HERMES_CONTAINER_NAME,
            "/opt/hermes/.venv/bin/hermes", "profile", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("-")]


@ToolRegistry.register()
class RunHermesProfileTool(BaseTool):
    name = "run_hermes_profile"
    description = "Execute a query against a specific Hermes Agent profile (with its custom skills, tools, and memory)."
    category = "hermes"
    required_permissions = [Permission.TERMINAL]

    async def run(self, profile: str, prompt: str) -> str:
        cmd = [
            "docker", "exec", HERMES_CONTAINER_NAME,
            "/opt/hermes/.venv/bin/hermes", "run",
            "--profile", profile,
            prompt,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
            return stdout.decode("utf-8", errors="replace").strip()
        except asyncio.TimeoutError:
            proc.kill()
            return f"Hermes profile '{profile}' execution timed out after 180s."
