"""
Kalki Nexus - Permissions & Security Context

Every tool declares the permissions it needs. Every agent is granted a set
of permissions (see config.py-adjacent DEFAULT_AGENT_PERMISSIONS below).
`SecurityContext.check(...)` is called by `BaseTool.__call__` before a tool
actually runs, so an agent can only reach what it has been explicitly
granted - not everything that happens to be importable.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Set

from core.exceptions import PermissionDeniedError


class Permission(str, Enum):
    """A single grantable capability a tool may require."""

    FS_READ = "fs:read"
    FS_WRITE = "fs:write"
    FS_DELETE = "fs:delete"
    TERMINAL = "terminal:exec"
    DOCKER = "docker:manage"
    GITHUB = "github:manage"
    BROWSER = "browser:automate"
    NETWORK = "network:http"
    DISCORD = "discord:send"


class SecurityContext:
    """The set of permissions a particular agent (or tool call) has been granted."""

    def __init__(self, granted: Iterable[Permission]) -> None:
        self.granted: Set[Permission] = set(granted)

    def check(self, required: List[Permission], tool_name: str) -> None:
        """Raise PermissionDeniedError if any required permission is not granted."""
        missing = [permission for permission in required if permission not in self.granted]
        if missing:
            raise PermissionDeniedError(tool_name, missing)

    def has(self, permission: Permission) -> bool:
        return permission in self.granted

    @classmethod
    def full_access(cls) -> "SecurityContext":
        """A SecurityContext granting every permission - useful for tests/admin tooling."""
        return cls(list(Permission))


# Default permission grants per agent name. Agents call
# `SecurityContext(DEFAULT_AGENT_PERMISSIONS.get(self.name, []))` inside
# `load_tools()` (see core/base_agent.py) rather than hardcoding this per module.
DEFAULT_AGENT_PERMISSIONS: Dict[str, List[Permission]] = {
    "python_agent": [Permission.FS_READ, Permission.FS_WRITE],
    "docker_agent": [Permission.DOCKER, Permission.FS_READ],
    "github_agent": [Permission.GITHUB, Permission.NETWORK],
    "research_agent": [Permission.NETWORK, Permission.BROWSER],
    "quant_agent": [Permission.FS_READ, Permission.FS_WRITE],
    "automation_agent": [Permission.FS_READ, Permission.FS_WRITE, Permission.TERMINAL],
    "mcp_agent": [Permission.NETWORK],
    "fallback_agent": [],
}
