"""
Kalki Nexus - Docker Agent

Handles containerization requests: writing Dockerfiles, docker-compose
configs, and diagnosing container build/runtime issues.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent


class DockerAgent(BaseAgent):
    name = "docker_agent"
    description = "Writes and debugs Docker/Compose configuration."
    channel_hints: ClassVar[List[str]] = ["docker"]
    default_tool_categories: ClassVar[List[str]] = ["docker"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        return await self._default_llm_run(state)
