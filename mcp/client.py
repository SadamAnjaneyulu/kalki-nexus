"""
Kalki Nexus - MCP Client Transport

This is the one intentionally-placeholder piece of the MCP subsystem: the
actual network call to an MCP server. Everything upstream of this module
(MCPRegistry's server registration, capability caching, hot reload, and
resolution) is real; only the wire call is a TODO.

To make this real, install `langchain-mcp-adapters` and replace the bodies
below with something like:

    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def discover_server_tools(config: MCPServerConfig) -> list[dict]:
        client = MultiServerMCPClient({config.name: _connection_dict(config)})
        async with client.session(config.name) as session:
            response = await session.list_tools()
            return [
                {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                for t in response.tools
            ]

    async def load_server_tools(config: MCPServerConfig) -> list:
        client = MultiServerMCPClient({config.name: _connection_dict(config)})
        async with client.session(config.name) as session:
            return await load_mcp_tools(session)
"""
from __future__ import annotations

from typing import Any, Dict, List

from mcp.registry import MCPServerConfig  # noqa: F401 - re-exported for type hints in real implementations


async def discover_server_tools(config: "MCPServerConfig") -> List[Dict[str, Any]]:
    """Return this server's tool capabilities as plain dicts: {name, description, input_schema}.

    TODO: open a real session against `config` (stdio/sse/streamable_http)
    and call the MCP `tools/list` method. Left empty rather than raising so
    a Kalki Nexus deployment with zero MCP servers configured still runs
    cleanly end to end.
    """
    return []


async def load_server_tools(config: "MCPServerConfig") -> List[Any]:
    """Return live, callable LangChain-compatible tool objects for this server.

    TODO: open a real session against `config` and call
    `langchain_mcp_adapters.tools.load_mcp_tools(session)` (or equivalent).
    """
    return []
