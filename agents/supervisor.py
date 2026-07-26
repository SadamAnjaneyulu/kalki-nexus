"""
Kalki Nexus - Supervisor

An LLM-powered router with a Pydantic structured-output schema
(`RouteDecision`). Inspects the user message, the Discord channel, attached
files, requested tools, and prior state, then decides which specialist
agent(s) should run.

The Discord channel contributes a *hint*, not a rule: CHANNEL_HINTS maps a
channel name to the agent it should bias the Supervisor toward, and that
hint is folded into the routing prompt rather than short-circuiting the LLM
call - the Supervisor can and does override it (e.g. someone asking a
research question in #docker still routes to the Research Agent).

If the structured LLM call is unavailable (no API key, offline dev/tests),
`heuristic_routes()` is used as a deterministic fallback so the graph always
degrades gracefully instead of failing closed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.base_agent import BaseAgent
from core.observability import get_logger
from core.registry import discover_agents

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "supervisor.md"

logger = get_logger("kalki.agents.supervisor")

# Discord channel name -> agent it should bias routing toward. "Bias", not
# "force": see module docstring.
CHANNEL_HINTS: Dict[str, str] = {
    "vajra-python": "python_agent",
    "python": "python_agent",
    "docker": "docker_agent",
    "github": "github_agent",
    "research": "research_agent",
    "quant": "quant_agent",
    "automation": "automation_agent",
    "mcp": "mcp_agent",
}

# Keyword hints used for the deterministic fallback router. This is
# intentionally kept in sync with, but independent of, the LLM path so the
# graph still works with zero configured API keys.
_KEYWORDS: Dict[str, List[str]] = {
    "python_agent": ["python", "script", "bug", "traceback", "refactor"],
    "docker_agent": ["docker", "container", "compose", "image", "dockerfile"],
    "github_agent": ["github", "pull request", "pr ", "repo", "commit", "issue"],
    "research_agent": ["research", "paper", "summarize", "compare", "sources"],
    "quant_agent": ["quant", "backtest", "strategy", "pnl", "sharpe", "risk"],
    "automation_agent": ["automate", "schedule", "workflow", "cron"],
    "mcp_agent": ["mcp", "connector", "tool call"],
}


class RouteDecision(BaseModel):
    """Structured output schema the Supervisor's LLM call is constrained to."""

    agents: List[str] = Field(..., description="Ordered list of specialist agent names that should run.")
    reasoning: str = Field(..., description="A short explanation of why these agent(s) were chosen.")
    confidence: float = Field(0.7, ge=0.0, le=1.0, description="Confidence in this routing decision, 0-1.")


def normalize_channel(discord_channel: Optional[str]) -> Optional[str]:
    """Strip a leading '#' and lowercase a Discord channel name for lookup in CHANNEL_HINTS."""
    if not discord_channel:
        return None
    return discord_channel.lstrip("#").strip().lower()


def heuristic_routes(
    user_input: str,
    attached_files: List[str],
    requested_tools: List[str],
    channel_hint: Optional[str] = None,
) -> List[str]:
    """Deterministic keyword-based routing, used when the LLM path is unavailable.

    The channel hint is appended only if nothing else matched, matching the
    "hint, not a rule" contract described in the module docstring.
    """
    text = user_input.lower()
    selected = [
        agent for agent, keywords in _KEYWORDS.items() if any(keyword in text for keyword in keywords)
    ]

    for tool in requested_tools:
        tool_lower = tool.lower()
        for agent, keywords in _KEYWORDS.items():
            if agent in selected:
                continue
            if any(keyword in tool_lower for keyword in keywords):
                selected.append(agent)

    if attached_files and "python_agent" not in selected:
        selected.append("python_agent")

    if not selected and channel_hint:
        selected.append(channel_hint)

    return selected or ["python_agent"]


class SupervisorAgent(BaseAgent):
    """LangGraph node: populates state["route"] via an LLM structured-output call."""

    name = "supervisor"
    description = "Routes each request to one or more specialist agents."
    prompt_file = "supervisor.md"

    def load_prompt(self) -> str:
        return PROMPT_PATH.read_text(encoding="utf-8")

    def _build_routing_prompt(self, state: Dict[str, Any], channel_hint: Optional[str], valid_agents: List[str]) -> str:
        return (
            f"{self.load_prompt()}\n\n"
            f"Available agents: {', '.join(valid_agents)}\n"
            f"Discord channel: {state.get('discord_channel') or 'n/a'}"
            f"{f' (hint: prefer {channel_hint}, but only if the message actually fits)' if channel_hint else ''}\n"
            f"Attached files: {state.get('attached_files') or 'none'}\n"
            f"Requested tools: {state.get('requested_tools') or 'none'}\n"
            f"Prior route (if any): {state.get('route') or 'none'}\n\n"
            f"User message:\n{state.get('user_input', '')}"
        )

    async def decide(self, state: Dict[str, Any]) -> RouteDecision:
        valid_agents = sorted(name for name in discover_agents() if name != "fallback_agent")
        channel_hint = CHANNEL_HINTS.get(normalize_channel(state.get("discord_channel")))

        try:
            llm = self.settings.build_chat_model(temperature=0.0).with_structured_output(RouteDecision)
            prompt = self._build_routing_prompt(state, channel_hint, valid_agents)
            decision: RouteDecision = await llm.ainvoke(prompt)
            decision.agents = [agent for agent in decision.agents if agent in valid_agents]
            if not decision.agents:
                decision.agents = heuristic_routes(
                    state.get("user_input", ""), state.get("attached_files", []),
                    state.get("requested_tools", []), channel_hint,
                )
            return decision
        except Exception as exc:  # noqa: BLE001 - any LLM/config failure falls back to the heuristic router
            logger.warning("LLM routing unavailable (%s); falling back to heuristic_routes().", exc)
            routes = heuristic_routes(
                state.get("user_input", ""), state.get("attached_files", []),
                state.get("requested_tools", []), channel_hint,
            )
            return RouteDecision(agents=routes, reasoning="heuristic fallback (LLM routing unavailable)", confidence=0.4)

    async def run(self, state: Dict[str, Any]):
        # SupervisorAgent does not produce an AgentResult; it overrides __call__ instead.
        raise NotImplementedError

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision = await self.decide(state)
        self.logger.info("route=%s confidence=%.2f reasoning=%s", decision.agents, decision.confidence, decision.reasoning)
        return {
            "route": decision.agents,
            "route_reasoning": decision.reasoning,
            "route_confidence": decision.confidence,
        }
