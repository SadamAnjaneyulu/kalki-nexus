"""
Kalki Nexus - Fallback Agent

The last resort once the graph's error_node has exhausted its bounded
retries (see graph.py::route_after_error). For casual conversations or
off-topic messages, uses an LLM to give a helpful general response.
When there are actual errors, gracefully surfaces partial results.
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from core.base_agent import AgentResult, BaseAgent

_GENERAL_SYSTEM_PROMPT = """\
You are Kalki Nexus, an advanced multi-agent AI assistant specialized in:
- Python programming, debugging, and code generation
- Quantitative finance and algorithmic trading strategies
- Software research, documentation, and analysis
- GitHub repository management
- Docker containerization
- Web research and information retrieval

For casual messages, greet warmly and explain briefly what you can help with.
For off-topic questions, answer helpfully from your general knowledge.
Always be concise, friendly, and professional.
"""


class FallbackAgent(BaseAgent):
    """Graph-level last resort: handles casual messages and graceful error degradation."""

    name = "fallback_agent"
    description = "Handles general conversations and produces graceful responses when retries are exhausted."
    retry_policy = BaseAgent.retry_policy  # no further retries

    def load_tools(self):
        return []

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        error = state.get("error") or {}
        user_input = state.get("user_input", "")
        partial = {
            name: result.answer
            for name, result in (state.get("agent_results") or {}).items()
            if result.answer and result.answer.strip()
        }

        # If there are actual errors and no partial results, show error + try LLM
        if error and not partial:
            # Try a general LLM response first
            try:
                settings = get_settings()
                llm = settings.build_chat_model(temperature=0.5)
                messages = [
                    SystemMessage(content=_GENERAL_SYSTEM_PROMPT),
                    HumanMessage(content=user_input),
                ]
                response = await llm.ainvoke(messages)
                answer = str(response.content).strip()
                if answer:
                    return AgentResult(
                        agent=self.name,
                        answer=answer,
                        confidence=0.6,
                        metadata={"fallback": True, "mode": "general_llm"},
                    )
            except Exception:  # noqa: BLE001
                pass

            # If LLM also fails, return the error message
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

        # No error - this is a general/casual/off-topic message: use LLM directly
        if not error:
            try:
                settings = get_settings()
                llm = settings.build_chat_model(temperature=0.5)
                messages = [
                    SystemMessage(content=_GENERAL_SYSTEM_PROMPT),
                    HumanMessage(content=user_input),
                ]
                response = await llm.ainvoke(messages)
                answer = str(response.content).strip()
                if answer:
                    return AgentResult(
                        agent=self.name,
                        answer=answer,
                        confidence=0.7,
                        metadata={"fallback": True, "mode": "general_llm"},
                    )
            except Exception:  # noqa: BLE001
                pass

        # Final fallback: surface partial results
        if partial:
            body = "\n\n".join(f"[{name}] {ans}" for name, ans in partial.items())
            answer = (
                "Here is the partial progress made before an error occurred:\n\n"
                f"{body}"
            )
            return AgentResult(
                agent=self.name,
                answer=answer,
                confidence=0.4,
                metadata={"fallback": True, "mode": "partial_results"},
            )

        return AgentResult(
            agent=self.name,
            answer="I'm Kalki Nexus, your AI assistant. I can help with Python, quantitative finance, research, GitHub, Docker, and more. What would you like to work on?",
            confidence=0.5,
            metadata={"fallback": True, "mode": "static_greeting"},
        )

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = await self.run(state)
        return {
            "agent_results": {self.name: result},
            "final_answer": result.answer,
            "error": None,
        }
