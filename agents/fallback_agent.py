"""
Kalki Nexus - Fallback Agent

The last resort once the graph's error_node has exhausted its bounded
retries (see graph.py::route_after_error). Never raises: it always returns
a graceful, honest degraded response rather than letting the run crash, and
surfaces whatever partial agent_results exist so the user isn't left with
nothing.
"""
from __future__ import annotations

from typing import Any, Dict

from core.base_agent import AgentResult, BaseAgent


class FallbackAgent(BaseAgent):
    """Graph-level last resort: never raises, always returns a usable AgentResult."""

    name = "fallback_agent"
    description = "Produces a graceful degraded response when every retry has been exhausted."
    retry_policy = BaseAgent.retry_policy  # no further retries: this IS the end of the retry chain

    def load_tools(self):
        return []

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        error = state.get("error") or {}
        partial = {
            name: result.answer
            for name, result in (state.get("agent_results") or {}).items()
            if result.answer
        }

        if partial:
            body = "\n\n".join(f"[{name}] {answer}" for name, answer in partial.items())
            answer = (
                "One or more agents hit an error and retries were exhausted, but here is "
                f"the partial progress that was made before that happened:\n\n{body}"
            )
        else:
            answer = (
                "Sorry - this request could not be completed. "
                f"Last error: {error.get('message', 'unknown error')}. "
                "Please try rephrasing the request or check the service configuration."
            )

        return AgentResult(
            agent=self.name,
            answer=answer,
            confidence=0.2,
            metadata={"fallback": True, "original_error": error},
        )

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = await self.run(state)
        return {
            "agent_results": {self.name: result},
            "final_answer": result.answer,
            "error": None,
        }
