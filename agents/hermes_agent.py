"""
Kalki Nexus - Hermes Agent Specialist

Specialist agent that bridges Kalki Nexus to self-hosted Hermes Agent profiles
running in Docker. Automatically lists active profiles and delegates execution
to the appropriate Hermes Agent profile.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent
from tools.hermes_tools import ListHermesProfilesTool, RunHermesProfileTool


CHANNEL_PROFILE_MAP: Dict[str, str] = {
    # Quant Sanctum -> quant_research
    "market-watch": "quant_research",
    "alpha-research": "quant_research",
    "backtesting": "quant_research",
    "analytics": "quant_research",
    "macro-news": "quant_research",
    "quant": "quant_research",
    # Engineering & Coding -> software_engineer
    "vajra-python": "software_engineer",
    "brahmastra-terminal": "software_engineer",
    "debug-zone": "software_engineer",
    "sandbox": "software_engineer",
    "bug-hunt": "software_engineer",
    "agnipariksha": "software_engineer",
    "python": "software_engineer",
    # Research & Akashic Library -> research_analyst
    "rajya-grantham": "research_analyst",
    "research": "research_analyst",
    "amarendra-ai": "research_analyst",
    "rag-vault": "research_analyst",
    "akashic-library": "research_analyst",
    # Machine Learning & AI Lab -> ml_engineer
    "model-training": "ml_engineer",
    "evaluations": "ml_engineer",
    "llm-lab": "ml_engineer",
    # Automation -> automation
    "garuda-automation": "automation",
    "automation": "automation",
    # Architecture & Core -> ai_architect / system_architect
    "simhasanam": "ai_architect",
    "katappa-core": "system_architect",
    "prompt-engineering": "technical_writer",
}


class HermesAgent(BaseAgent):
    name = "hermes_agent"
    description = "Delegates complex tasks to custom self-hosted Hermes Agent profiles (quant_research, software_engineer, research_analyst, etc.)."
    channel_hints: ClassVar[List[str]] = list(CHANNEL_PROFILE_MAP.keys())
    default_tool_categories: ClassVar[List[str]] = ["hermes", "terminal"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        user_input = state.get("user_input", "")
        discord_channel = (state.get("discord_channel") or "").lstrip("#").strip().lower()

        # Step 1: List available profiles from container
        list_tool = ListHermesProfilesTool()
        profiles = await list_tool.run()

        # Step 2: Select profile based on Discord channel mapping first, then prompt mention, then default
        selected_profile = CHANNEL_PROFILE_MAP.get(discord_channel)

        if not selected_profile or selected_profile not in profiles:
            for p in profiles:
                if p.lower() in user_input.lower():
                    selected_profile = p
                    break
            else:
                selected_profile = "default"

        # Step 3: Execute query through Hermes Agent profile
        run_tool = RunHermesProfileTool()
        output = await run_tool.run(profile=selected_profile, prompt=user_input)

        if not output or "error" in output.lower():
            return await self._default_llm_run(state)

        return AgentResult(
            agent=self.name,
            answer=output,
            confidence=0.9,
            metadata={"profile": selected_profile, "container": "hermes-dashboard"},
        )
