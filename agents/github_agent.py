"""
Kalki Nexus - GitHub Agent

Drafts and manages issues, pull requests, and repository content via the
GitHub tools. Flags destructive actions for human approval instead of
performing them unattended.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent

_DESTRUCTIVE_HINTS = ("force-push", "force push", "delete branch", "delete repo", "merge pull request", "merge pr")


class GithubAgent(BaseAgent):
    name = "github_agent"
    description = "Drafts and manages GitHub issues, pull requests, and repository content."
    channel_hints: ClassVar[List[str]] = ["github"]
    default_tool_categories: ClassVar[List[str]] = ["github"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        result = await self._default_llm_run(state)
        text = state.get("user_input", "").lower()
        if any(hint in text for hint in _DESTRUCTIVE_HINTS):
            result.metadata["requires_approval"] = True
            result.metadata["approval_reason"] = "Request appears to involve a destructive GitHub operation."
        return result
