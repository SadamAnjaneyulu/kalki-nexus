"""
Kalki Nexus - Supervisor

Routes ALL Discord channel messages to hermes_agent, which then delegates
to the correct Hermes Agent profile based on the channel name. The channel-
to-profile mapping lives in agents/hermes_agent.py (CHANNEL_PROFILE_MAP).

Kalki Nexus's purpose is routing — Hermes does the actual thinking with
its 69+ skills, tools, memory, and learned knowledge.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.base_agent import BaseAgent
from core.observability import get_logger
from core.registry import discover_agents

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "supervisor.md"

logger = get_logger("kalki.agents.supervisor")


# ALL channels route to hermes_agent — Hermes profiles handle everything.
CHANNEL_HINTS: Dict[str, str] = {
    # Quant Sanctum
    "market-watch": "hermes_agent",
    "alpha-research": "hermes_agent",
    "backtesting": "hermes_agent",
    "analytics": "hermes_agent",
    "macro-news": "hermes_agent",
    "quant": "hermes_agent",
    "kurukshetra": "hermes_agent",
    # Engineering & Coding
    "vajra-python": "hermes_agent",
    "python": "hermes_agent",
    "brahmastra-terminal": "hermes_agent",
    "debug-zone": "hermes_agent",
    "sandbox": "hermes_agent",
    "bug-hunt": "hermes_agent",
    "agnipariksha": "hermes_agent",
    # Research & Akashic Library
    "rajya-grantham": "hermes_agent",
    "research": "hermes_agent",
    "amarendra-ai": "hermes_agent",
    "rag-vault": "hermes_agent",
    "akashic-library": "hermes_agent",
    "prompt-engineering": "hermes_agent",
    # Machine Learning & AI Lab
    "model-training": "hermes_agent",
    "evaluations": "hermes_agent",
    "llm-lab": "hermes_agent",
    # Docker & Infrastructure
    "docker": "hermes_agent",
    # GitHub & Projects
    "github": "hermes_agent",
    "projects": "hermes_agent",
    # Automation
    "garuda-automation": "hermes_agent",
    "automation": "hermes_agent",
    # Architecture & Core
    "simhasanam": "hermes_agent",
    "katappa-core": "hermes_agent",
    # MCP & Protocols
    "mcp": "hermes_agent",
    "mcp-mantra": "hermes_agent",
    "api-gateway": "hermes_agent",
    # Hermes direct
    "hermes": "hermes_agent",
    # Catch-all misc
    "aadesham": "hermes_agent",
}

# Keyword hints for the deterministic fallback router
_KEYWORDS: Dict[str, List[str]] = {
    "hermes_agent": [
        "hermes", "profile", "python", "script", "bug", "traceback", "refactor",
        "docker", "container", "compose", "dockerfile",
        "github", "pull request", "pr ", "repo", "commit", "issue",
        "research", "paper", "summarize", "compare", "sources", "document",
        "quant", "backtest", "strategy", "pnl", "sharpe", "risk", "vwap",
        "automate", "schedule", "workflow", "cron", "n8n",
        "mcp", "connector", "tool call", "protocol",
        "katappa", "simhasanam", "grantham",
    ],
}


class RouteDecision(BaseModel):
    """Structured output schema the Supervisor's LLM call is constrained to."""

    agents: List[str] = Field(..., description="Ordered list of specialist agent names that should run.")
    reasoning: str = Field(..., description="A short explanation of why these agent(s) were chosen.")
    confidence: float = Field(0.7, ge=0.0, le=1.0, description="Confidence in this routing decision, 0-1.")


def normalize_channel(discord_channel: Optional[str]) -> Optional[str]:
    """Clean emojis, leading symbols, and lowercase a Discord channel name for lookup in CHANNEL_HINTS."""
    if not discord_channel:
        return None
    # Remove emojis, hashtags, and leading punctuation
    cleaned = re.sub(r'^[^\w]+', '', discord_channel).strip().lower()
    return cleaned


def heuristic_routes(
    user_input: str,
    attached_files: List[str],
    requested_tools: List[str],
    channel_hint: Optional[str] = None,
) -> List[str]:
    """Deterministic keyword-based routing, used when the LLM path is unavailable.

    Always routes to hermes_agent since Hermes handles everything.
    """
    # Always route to hermes_agent — Hermes profiles handle all domains
    return ["hermes_agent"]


class SupervisorAgent(BaseAgent):
    """LangGraph node: always routes to hermes_agent.

    Kalki Nexus's role is routing Discord messages to the correct Hermes
    profile. The hermes_agent maps channel → profile and invokes
    `hermes -p <profile> -z "<prompt>" --cli` inside the Docker container.
    """

    name = "supervisor"
    description = "Routes each request to the Hermes Agent for execution via the appropriate profile."
    prompt_file = "supervisor.md"

    def load_prompt(self) -> str:
        return PROMPT_PATH.read_text(encoding="utf-8")

    async def decide(self, state: Dict[str, Any]) -> RouteDecision:
        """Always route to hermes_agent — no LLM call needed for routing."""
        channel = normalize_channel(state.get("discord_channel"))
        channel_hint = CHANNEL_HINTS.get(channel, "hermes_agent")

        return RouteDecision(
            agents=[channel_hint],
            reasoning=f"Routing to hermes_agent for channel '{channel or 'direct'}' — Hermes profile handles execution.",
            confidence=0.95,
        )

    async def run(self, state: Dict[str, Any]):
        raise NotImplementedError

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision = await self.decide(state)
        self.logger.info("route=%s confidence=%.2f reasoning=%s", decision.agents, decision.confidence, decision.reasoning)
        return {
            "route": decision.agents,
            "route_reasoning": decision.reasoning,
            "route_confidence": decision.confidence,
        }
