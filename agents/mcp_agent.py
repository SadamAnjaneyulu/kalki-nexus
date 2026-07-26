"""
Kalki Nexus - MCP Agent

Bridges requests to Model Context Protocol (MCP) servers and tools. Unlike
the other specialists, its tools are discovered dynamically at runtime from
the MCPRegistry rather than bound statically via ToolLoader - so it
overrides run() directly instead of relying on _default_llm_run's
synchronous load_tools() path.
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import AgentResult, BaseAgent
from mcp.registry import get_mcp_registry


class McpAgent(BaseAgent):
    name = "mcp_agent"
    description = "Discovers and invokes tools exposed by connected MCP servers."
    channel_hints = ["mcp"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        registry = get_mcp_registry()
        capabilities = await registry.discover_tools()
        mcp_tools = await registry.load_tools()

        llm = self.settings.build_chat_model(temperature=self.temperature, tools=mcp_tools or None)
        capability_summary = "\n".join(
            f"- {cap.server_name}.{cap.tool_name}: {cap.description}" for cap in capabilities
        ) or "(no MCP servers currently registered)"

        messages = [
            SystemMessage(content=f"{self.load_prompt()}\n\nKnown MCP tool capabilities:\n{capability_summary}"),
            HumanMessage(content=state.get("user_input", "")),
        ]
        response = await llm.ainvoke(messages)
        tool_calls = [{"name": c.get("name"), "args": c.get("args")} for c in (response.tool_calls or [])]

        return AgentResult(
            agent=self.name,
            answer=str(response.content),
            tool_calls=tool_calls,
            metadata={"mcp_servers": sorted({cap.server_name for cap in capabilities})},
        )
