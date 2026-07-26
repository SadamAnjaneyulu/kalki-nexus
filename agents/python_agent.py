"""
Kalki Nexus - Python Agent

Handles Python code generation, debugging, and refactoring requests.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent


class PythonAgent(BaseAgent):
    name = "python_agent"
    description = "Writes, debugs, and refactors Python code."
    channel_hints: ClassVar[List[str]] = ["python", "vajra-python"]
    default_tool_categories: ClassVar[List[str]] = ["filesystem", "terminal"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        return await self._default_llm_run(state)
