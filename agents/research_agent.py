"""
Kalki Nexus - Research Agent

Gathers, summarizes, and cites information from the web to support other
agents (e.g. feeding the Quant Agent with instrument or market research).
Every finding is cached into this agent's namespaced Agent Memory.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent


class ResearchAgent(BaseAgent):
    name = "research_agent"
    description = "Gathers and summarizes information from the web, with citations."
    channel_hints: ClassVar[List[str]] = ["research"]
    default_tool_categories: ClassVar[List[str]] = ["web", "browser"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        return await self._default_llm_run(state)

    async def post_process(self, state: Dict[str, Any], result: AgentResult) -> AgentResult:
        memory = self.load_memory()
        topic = state.get("user_input", "")[:80]
        await memory.set(f"source:{topic}", {"summary": result.answer[:500]})
        return result
