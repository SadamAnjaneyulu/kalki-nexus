"""
Kalki Nexus - Hermes Agent Specialist

The central bridge between Kalki Nexus and Hermes Agent profiles running
inside Docker. Every Discord message flows through here:

    1. Normalize channel name (strip emojis, symbols)
    2. Map channel → Hermes profile via CHANNEL_PROFILE_MAP
    3. Execute: hermes -p <profile> -z "<prompt>" --cli
    4. Return Hermes's full response to Discord
"""
from __future__ import annotations

import re
from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent
from tools.hermes_tools import RunHermesProfileTool


# Discord channel name → Hermes profile mapping
# After normalization: "📈-market-watch" → "market-watch" → "quant_research"
CHANNEL_PROFILE_MAP: Dict[str, str] = {
    # ── Quant Sanctum → quant_research ──
    "market-watch": "quant_research",
    "alpha-research": "quant_research",
    "backtesting": "quant_research",
    "analytics": "quant_research",
    "macro-news": "quant_research",
    "quant": "quant_research",
    "kurukshetra": "quant_research",

    # ── Engineering & Coding → software_engineer ──
    "vajra-python": "software_engineer",
    "python": "software_engineer",
    "brahmastra-terminal": "software_engineer",
    "debug-zone": "software_engineer",
    "sandbox": "software_engineer",
    "bug-hunt": "software_engineer",
    "agnipariksha": "software_engineer",

    # ── Research & Akashic Library → research_analyst ──
    "rajya-grantham": "research_analyst",
    "research": "research_analyst",
    "amarendra-ai": "research_analyst",
    "rag-vault": "research_analyst",
    "akashic-library": "research_analyst",
    "prompt-engineering": "technical_writer",

    # ── Machine Learning & AI Lab → ml_engineer ──
    "model-training": "ml_engineer",
    "evaluations": "ml_engineer",
    "llm-lab": "ml_engineer",

    # ── Docker & Infrastructure → system_architect ──
    "docker": "system_architect",

    # ── GitHub & Projects → software_engineer ──
    "github": "software_engineer",
    "projects": "software_engineer",

    # ── Automation → automation ──
    "garuda-automation": "automation",
    "automation": "automation",

    # ── Architecture & Core → ai_architect ──
    "simhasanam": "ai_architect",
    "katappa-core": "system_architect",
    "aadesham": "ai_architect",

    # ── MCP & Protocols → system_architect ──
    "mcp": "system_architect",
    "mcp-mantra": "system_architect",
    "api-gateway": "system_architect",

    # ── Hermes direct → ai_architect ──
    "hermes": "ai_architect",
}

# Default profile when no channel mapping matches
DEFAULT_PROFILE = "ai_architect"


def _normalize_channel(raw: str) -> str:
    """Strip emojis, symbols, hashtags and lowercase a Discord channel name.

    Examples:
        '📈-market-watch' → 'market-watch'
        '#🐍-vajra-python' → 'vajra-python'
        '👑 SIMHASANAM'   → 'simhasanam'
    """
    # Remove leading # if present
    raw = raw.lstrip("#").strip()
    # Remove leading non-word characters (emojis, symbols, dashes after emoji)
    cleaned = re.sub(r'^[^\w]+', '', raw).strip().lower()
    return cleaned


class HermesAgent(BaseAgent):
    name = "hermes_agent"
    description = "Routes Discord messages to the correct Hermes Agent profile and returns the response."
    channel_hints: ClassVar[List[str]] = list(CHANNEL_PROFILE_MAP.keys())
    default_tool_categories: ClassVar[List[str]] = ["hermes", "terminal"]
    temperature = 0.2

    # Human-readable profile descriptions for context
    PROFILE_ROLES: ClassVar[Dict[str, str]] = {
        "ai_architect": "Kalki Nexus AI Architect — central command, planning, task execution",
        "quant_research": "Quantitative Research Analyst — market analysis, trading strategies, financial data",
        "software_engineer": "Software Engineer — Python coding, debugging, code generation, GitHub",
        "research_analyst": "Research Analyst — documentation, papers, knowledge synthesis",
        "ml_engineer": "ML Engineer — model training, evaluations, LLM experiments",
        "system_architect": "System Architect — infrastructure, Docker, MCP, DevOps",
        "automation": "Automation Specialist — workflows, cron jobs, n8n, scheduling",
        "technical_writer": "Technical Writer — prompt engineering, documentation, content",
        "learning_mentor": "Learning Mentor — teaching, explanations, tutorials",
    }

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        user_input = state.get("user_input", "")
        raw_channel = state.get("discord_channel") or ""

        # Step 1: Normalize the Discord channel name
        channel = _normalize_channel(raw_channel)

        # Step 2: Map channel → Hermes profile
        selected_profile = CHANNEL_PROFILE_MAP.get(channel, DEFAULT_PROFILE)
        profile_role = self.PROFILE_ROLES.get(selected_profile, selected_profile)

        self.logger.info(
            "hermes_agent: channel='%s' → normalized='%s' → profile='%s'",
            raw_channel, channel, selected_profile,
        )

        # Step 3: Build a context-enriched prompt for Hermes
        # This tells Hermes which Discord channel it's responding in and its role
        enriched_prompt = (
            f"[CONTEXT] You are responding in Discord channel '#{raw_channel}'. "
            f"Your role: {profile_role}. "
            f"You are part of the Kalki Nexus multi-agent system with profiles: "
            f"{', '.join(CHANNEL_PROFILE_MAP.values())}. "
            f"Respond naturally as this agent role.\n\n"
            f"[USER MESSAGE] {user_input}"
        )

        # Step 4: Execute the prompt through the Hermes CLI
        run_tool = RunHermesProfileTool()
        output = await run_tool.run(profile=selected_profile, prompt=enriched_prompt)

        # Step 5: Handle empty or error responses
        if not output or (output.startswith("Hermes profile") and "timed out" in output):
            self.logger.warning("hermes_agent: profile '%s' returned no/error output, using fallback", selected_profile)
            return await self._default_llm_run(state)

        return AgentResult(
            agent=self.name,
            answer=output,
            confidence=0.95,
            metadata={
                "profile": selected_profile,
                "channel": channel,
                "container": "hermes-dashboard",
            },
        )
