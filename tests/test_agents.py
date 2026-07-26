"""
Unit tests for Kalki Nexus Agent Registry and BaseAgent.
"""
import pytest
from core.registry import discover_agents
from core.base_agent import AgentResult, BaseAgent
from agents.fallback_agent import FallbackAgent


def test_discover_agents():
    agents = discover_agents()
    assert isinstance(agents, dict)
    assert "fallback_agent" in agents
    assert "python_agent" in agents
    assert "research_agent" in agents
    assert "supervisor" in agents


@pytest.mark.asyncio
async def test_fallback_agent_run():
    agent = FallbackAgent()
    state = {
        "user_input": "Test query",
        "error": {"message": "Test failure message"},
    }
    result = await agent.run(state)
    assert isinstance(result, AgentResult)
    assert result.agent == "fallback_agent"
    assert "Test failure message" in result.answer
