"""
Kalki Nexus - Docker Tools

BaseTool wrappers around common Docker operations, registered under the
"docker" category. Wire the TODOs up to the `docker` CLI (via subprocess)
or the Docker SDK for Python (`docker-py`) before using these in production.
"""
from __future__ import annotations

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry


@ToolRegistry.register()
class DockerPsTool(BaseTool):
    name = "docker_ps"
    description = "List running Docker containers."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self) -> str:
        # TODO: shell out to `docker ps --format json` or use docker-py's client.containers.list()
        raise NotImplementedError("docker_ps: wire up the Docker CLI or SDK call.")


@ToolRegistry.register()
class DockerRunTool(BaseTool):
    name = "docker_run"
    description = "Run a Docker container from an image, optionally executing a command."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self, image: str, command: str = "") -> str:
        # TODO: shell out to `docker run {image} {command}` or use docker-py's client.containers.run()
        raise NotImplementedError("docker_run: wire up the Docker CLI or SDK call.")


@ToolRegistry.register()
class DockerBuildTool(BaseTool):
    name = "docker_build"
    description = "Build a Docker image from a Dockerfile and tag it."
    category = "docker"
    required_permissions = [Permission.DOCKER, Permission.FS_READ]

    async def run(self, dockerfile_path: str, tag: str) -> str:
        # TODO: shell out to `docker build -f {dockerfile_path} -t {tag} .` or use docker-py
        raise NotImplementedError("docker_build: wire up the Docker CLI or SDK call.")


@ToolRegistry.register()
class DockerLogsTool(BaseTool):
    name = "docker_logs"
    description = "Fetch the last N lines of logs for a container."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self, container_id: str, tail: int = 100) -> str:
        # TODO: shell out to `docker logs --tail {tail} {container_id}` or use docker-py
        raise NotImplementedError("docker_logs: wire up the Docker CLI or SDK call.")
