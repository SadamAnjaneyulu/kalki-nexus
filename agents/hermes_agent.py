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


class HermesAgent(BaseAgent):
    name = "hermes_agent"
    description = "Delegates complex tasks to custom self-hosted Hermes Agent profiles and skills."
    channel_hints: ClassVar[List[str]] = ["hermes", "katappa-core", "simhasanam"]
    default_tool_categories: ClassVar[List[str]] = ["hermes", "terminal"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        user_input = state.get("user_input", "")

        # Step 1: List available profiles
        list_tool = ListHermesProfilesTool()
        profiles = await list_tool.run()

        # Step 2: Select profile if mentioned or run default
        selected_profile = "default"
        for p in profiles:
            if p.lower() in user_input.lower():
                selected_profile = p
                break

        # Step 3: Execute query through Hermes Agent profile
        run_tool = RunHermesProfileTool()
        output = await run_tool.run(profile=selected_profile, prompt=user_input)

        if not output or "error" in output.lower():
            # Fall back to standard LLM run if container run fails
            return await self._default_llm_run(state)

        return AgentResult(
            agent=self.name,
            answer=output,
            confidence=0.9,
            metadata={"profile": selected_profile, "container": "hermes-dashboard"},
        )
