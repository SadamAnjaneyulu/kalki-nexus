"""
Kalki Nexus - Agent Delegation Protocol

Allows any specialist agent to delegate a sub-task to another specialist
agent directly, without going back through the Supervisor. This enables
multi-hop chains like:

    Python Agent -> delegates research to -> Research Agent
    Research Agent -> delegates quant analysis to -> Quant Agent

Usage inside any agent's run() method:

    from core.delegation import delegate_to

    sub_result = await delegate_to("research_agent", state, sub_question)

The delegate_to() function:
  - Instantiates the target agent on-demand (using discover_agents)
  - Injects the sub_input as user_input in a derived state copy
  - Returns an AgentResult from the target agent
  - Raises AgentError if the target agent doesn't exist or fails
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.base_agent import AgentResult
from core.exceptions import AgentError
from core.observability import get_logger

logger = get_logger("kalki.delegation")


async def delegate_to(
    agent_name: str,
    parent_state: Dict[str, Any],
    sub_input: str,
    metadata_override: Optional[Dict[str, Any]] = None,
) -> AgentResult:
    """Delegate a sub-task to another specialist agent.

    Args:
        agent_name: Registry name of the target agent (e.g. 'research_agent').
        parent_state: The current graph state from the calling agent's run().
        sub_input: The sub-question or task for the target agent.
        metadata_override: Optional additional metadata to inject into the child state.

    Returns:
        AgentResult from the target agent.

    Raises:
        AgentError: If the target agent is not found or raises an exception.
    """
    from core.registry import discover_agents  # lazy import to avoid circular dependency

    agents = discover_agents()
    if agent_name not in agents:
        raise AgentError(agent_name, f"Agent '{agent_name}' not found in registry. Available: {sorted(agents)}")

    agent_cls = agents[agent_name]
    agent = agent_cls()

    # Build a derived child state for the sub-task
    child_state: Dict[str, Any] = {
        **parent_state,
        "user_input": sub_input,
        "route": [],
        "agent_results": {},
        "final_answer": None,
        **(metadata_override or {}),
    }

    logger.info("delegate_to: %s -> %s | sub_input=%r", parent_state.get("_caller", "?"), agent_name, sub_input[:80])

    try:
        result = await agent.run(child_state)
        result = await agent.post_process(child_state, result)
    except Exception as exc:
        raise AgentError(agent_name, f"Delegation to '{agent_name}' failed: {exc}") from exc

    return result
