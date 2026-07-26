"""
Kalki Nexus - Docker Tools (Real Implementation)

BaseTool wrappers around asyncio subprocess calls to the Docker CLI,
registered under the "docker" category. Requires Docker to be installed
and the current user to have Docker socket access.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

DEFAULT_TIMEOUT = 60


async def _run_docker(*args: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a docker CLI command and return combined stdout/stderr."""
    process = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        return f"docker command timed out after {timeout}s"
    return stdout.decode("utf-8", errors="replace").strip()


@ToolRegistry.register()
class DockerPsTool(BaseTool):
    name = "docker_ps"
    description = "List running Docker containers (name, image, status, ports)."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self) -> List[Dict[str, Any]]:
        output = await _run_docker("ps", "--format", "{{json .}}")
        containers = []
        for line in output.splitlines():
            line = line.strip()
            if line:
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    containers.append({"raw": line})
        return containers


@ToolRegistry.register()
class DockerRunTool(BaseTool):
    name = "docker_run"
    description = "Run a Docker container from an image, optionally executing a command. Returns container output."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self, image: str, command: str = "", remove: bool = True) -> str:
        args = ["run"]
        if remove:
            args.append("--rm")
        args.append(image)
        if command:
            args.extend(command.split())
        return await _run_docker(*args)


@ToolRegistry.register()
class DockerBuildTool(BaseTool):
    name = "docker_build"
    description = "Build a Docker image from a Dockerfile and tag it."
    category = "docker"
    required_permissions = [Permission.DOCKER, Permission.FS_READ]

    async def run(self, dockerfile_path: str, tag: str, context: str = ".") -> str:
        return await _run_docker("build", "-f", dockerfile_path, "-t", tag, context, timeout=300)


@ToolRegistry.register()
class DockerLogsTool(BaseTool):
    name = "docker_logs"
    description = "Fetch the last N lines of logs for a running container."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self, container_id: str, tail: int = 100) -> str:
        return await _run_docker("logs", "--tail", str(tail), container_id)


@ToolRegistry.register()
class DockerStopTool(BaseTool):
    name = "docker_stop"
    description = "Stop a running Docker container gracefully."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self, container_id: str) -> str:
        return await _run_docker("stop", container_id)
