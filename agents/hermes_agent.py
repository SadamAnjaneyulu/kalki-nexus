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
    "katappa": "system_architect",
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

# Regex for parsing cross-profile delegation tags
DELEGATE_PATTERN = re.compile(r'DELEGATE\[([a-zA-Z0-9_-]+)\]:\s*(.+)', re.DOTALL)


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
    description = "Routes Discord messages to the correct Hermes Agent profile and enables cross-profile delegation."
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

        # Step 2: Map channel → Primary Hermes profile
        selected_profile = CHANNEL_PROFILE_MAP.get(channel, DEFAULT_PROFILE)
        profile_role = self.PROFILE_ROLES.get(selected_profile, selected_profile)

        self.logger.info(
            "hermes_agent: channel='%s' → normalized='%s' → profile='%s'",
            raw_channel, channel, selected_profile,
        )

        # Step 3: Build a context-enriched prompt for Hermes with Delegation Instructions
        available_profiles_str = ", ".join(sorted(set(CHANNEL_PROFILE_MAP.values())))
        enriched_prompt = (
            f"[CONTEXT] You are responding in Discord channel '#{raw_channel}'.\n"
            f"Your role: {profile_role}.\n"
            f"Available Hermes Agent profiles: {available_profiles_str}.\n"
            f"CROSS-AGENT DELEGATION INSTRUCTIONS:\n"
            f"If the user request requires specialized help from another profile (e.g. software_engineer for coding, quant_research for market stats, research_analyst for deep search), you can delegate sub-tasks by including this line on a new line:\n"
            f"DELEGATE[<profile_name>]: <sub_task_instructions>\n"
            f"The system will execute that profile and present its output.\n\n"
            f"[USER MESSAGE] {user_input}"
        )

        # Step 4: Check if user explicitly asked to call/involve another agent profile
        user_text_lower = user_input.lower()
        requested_profiles = []
        
        # Map keywords/mentions to target profile
        PROFILE_MENTION_MAP = {
            "software_engineer": "software_engineer",
            "software engineer": "software_engineer",
            "python agent": "software_engineer",
            "vajra-python": "software_engineer",
            "quant_research": "quant_research",
            "quant research": "quant_research",
            "quant agent": "quant_research",
            "market-watch": "quant_research",
            "research_analyst": "research_analyst",
            "research analyst": "research_analyst",
            "research agent": "research_analyst",
            "rajya-grantham": "research_analyst",
            "system_architect": "system_architect",
            "system architect": "system_architect",
            "docker agent": "system_architect",
            "katappa": "system_architect",
            "ai_architect": "ai_architect",
            "ai architect": "ai_architect",
            "simhasanam": "ai_architect",
            "ml_engineer": "ml_engineer",
            "ml engineer": "ml_engineer",
            "automation agent": "automation",
            "garuda-automation": "automation",
        }

        for keyword, target_p in PROFILE_MENTION_MAP.items():
            if keyword in user_text_lower and target_p != selected_profile:
                if target_p not in requested_profiles:
                    requested_profiles.append(target_p)

        # Step 5: Execute the primary prompt through the Hermes CLI
        run_tool = RunHermesProfileTool()
        output = await run_tool.run(profile=selected_profile, prompt=enriched_prompt)

        # Step 6: Handle empty, timeout, quota, or provider error responses
        is_error = (
            not output
            or "timed out" in output.lower()
            or "http 402" in output.lower()
            or "http 403" in output.lower()
            or "credits exhausted" in output.lower()
            or "openrouter reported" in output.lower()
            or "requires a subscription" in output.lower()
        )
        if is_error:
            self.logger.warning("hermes_agent: profile '%s' returned provider/quota error ('%s'), falling back to default LLM", selected_profile, output[:100])
            return await self._default_llm_run(state)

        # Step 7: Handle Agent-Driven Cross-Profile Delegation if emitted by the agent
        match = DELEGATE_PATTERN.search(output)
        delegated_profile = None
        if match:
            target_profile = match.group(1).strip().lower()
            sub_prompt = match.group(2).strip()

            if target_profile in self.PROFILE_ROLES and target_profile != selected_profile:
                self.logger.info("hermes_agent: Cross-delegating from '%s' to '%s'", selected_profile, target_profile)
                delegated_profile = target_profile
                sub_output = await run_tool.run(profile=target_profile, prompt=sub_prompt)

                # Replace delegation tag with sub-agent response
                replacement = (
                    f"\n\n🤝 **[Cross-Agent Handoff: @{target_profile}]**\n"
                    f"{sub_output}\n"
                )
                output = DELEGATE_PATTERN.sub(replacement, output)

        # Step 8: Handle User-Requested Cross-Agent Consultation (e.g. "call software_engineer")
        if requested_profiles and not match:
            for extra_profile in requested_profiles:
                self.logger.info("hermes_agent: User requested additional profile '%s'", extra_profile)
                extra_prompt = (
                    f"[CONTEXT] You are requested to assist in channel '#{raw_channel}' alongside @{selected_profile}.\n"
                    f"Your role: {self.PROFILE_ROLES.get(extra_profile, extra_profile)}.\n"
                    f"Provide your domain-specific input for this request:\n\n"
                    f"[USER MESSAGE] {user_input}"
                )
                extra_output = await run_tool.run(profile=extra_profile, prompt=extra_prompt)
                if extra_output and not extra_output.startswith("Hermes profile"):
                    output += f"\n\n🤝 **[Consulted Agent: @{extra_profile}]**\n{extra_output}"
                    delegated_profile = extra_profile

        return AgentResult(
            agent=self.name,
            answer=output,
            confidence=0.95,
            metadata={
                "profile": selected_profile,
                "delegated_profile": delegated_profile,
                "channel": channel,
                "container": "hermes-dashboard",
            },
        )
