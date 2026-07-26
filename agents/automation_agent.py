"""
Kalki Nexus - Automation Agent

Handles scheduling, workflow orchestration, and repetitive task automation
across the local filesystem and terminal. Flags destructive shell commands
for human approval rather than running them unattended.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent

_DESTRUCTIVE_HINTS = ("rm -rf", "format ", "drop table", "del /f", ":(){ :|:& };:")


class AutomationAgent(BaseAgent):
    name = "automation_agent"
    description = "Designs and executes repeatable filesystem/terminal workflows."
    channel_hints: ClassVar[List[str]] = ["automation"]
    default_tool_categories: ClassVar[List[str]] = ["filesystem", "terminal"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        result = await self._default_llm_run(state)
        text = state.get("user_input", "").lower()
        if any(hint in text for hint in _DESTRUCTIVE_HINTS):
            result.metadata["requires_approval"] = True
            result.metadata["approval_reason"] = "Request appears to involve a destructive shell command."
        return result
