"""
Kalki Nexus - MCP Registry

Server registration, dynamic tool discovery, capability caching (TTL-based),
hot reload, and tool resolution/selection for Model Context Protocol
servers. The actual network transport is deliberately isolated in
mcp/client.py behind `discover_server_tools()` / `load_server_tools()` - see
that module for the one real TODO in this subsystem. Everything else here
(the registry, the cache, multi-server support, hot reload, and the
agent-facing API `discover_tools()` / `load_tools()` / `resolve_tool()`) is
real and exercised by agents/mcp_agent.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from core.exceptions import MCPError
from core.observability import get_logger
from mcp.client import discover_server_tools, load_server_tools

logger = get_logger("kalki.mcp.registry")


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    transport: str  # "stdio" | "sse" | "streamable_http"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MCPToolCapability:
    """A single tool a server exposes, as last discovered."""

    server_name: str
    tool_name: str
    description: str
    input_schema: Dict[str, Any]
    cached_at: float


class MCPRegistry:
    """Registers MCP servers and manages discovery/caching/loading of their tools."""

    def __init__(self, cache_ttl_seconds: float = 300.0) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self._servers: Dict[str, MCPServerConfig] = {}
        self._capability_cache: Dict[str, List[MCPToolCapability]] = {}

    def register_server(self, config: MCPServerConfig) -> None:
        self._servers[config.name] = config
        self._capability_cache.pop(config.name, None)  # a re-registered server needs fresh discovery

    def unregister_server(self, name: str) -> None:
        self._servers.pop(name, None)
        self._capability_cache.pop(name, None)

    def servers(self) -> List[MCPServerConfig]:
        return [config for config in self._servers.values() if config.enabled]

    def _cache_is_fresh(self, server_name: str) -> bool:
        cached = self._capability_cache.get(server_name)
        if not cached:
            return False
        return (time.time() - cached[0].cached_at) < self.cache_ttl_seconds

    async def discover_tools(
        self, server_name: Optional[str] = None, force_refresh: bool = False
    ) -> List[MCPToolCapability]:
        """Discover (or return cached) tool capabilities for one server, or all servers."""
        targets = [self._servers[server_name]] if server_name else self.servers()
        all_capabilities: List[MCPToolCapability] = []

        for config in targets:
            if not force_refresh and self._cache_is_fresh(config.name):
                all_capabilities.extend(self._capability_cache[config.name])
                continue
            try:
                raw_tools = await discover_server_tools(config)
            except Exception as exc:  # noqa: BLE001 - one server's outage shouldn't break the others
                logger.warning("MCP discovery failed for server '%s': %s", config.name, exc)
                raise MCPError(config.name, f"tool discovery failed: {exc}") from exc

            now = time.time()
            capabilities = [
                MCPToolCapability(
                    server_name=config.name,
                    tool_name=raw["name"],
                    description=raw.get("description", ""),
                    input_schema=raw.get("input_schema", {}),
                    cached_at=now,
                )
                for raw in raw_tools
            ]
            self._capability_cache[config.name] = capabilities
            all_capabilities.extend(capabilities)

        return all_capabilities

    async def load_tools(self, server_names: Optional[List[str]] = None) -> List[Any]:
        """Load live, callable LangChain-compatible tools for the given servers (or all)."""
        targets = [self._servers[name] for name in server_names] if server_names else self.servers()
        tools: List[Any] = []
        for config in targets:
            try:
                tools.extend(await load_server_tools(config))
            except Exception as exc:  # noqa: BLE001 - degrade gracefully per server
                logger.warning("MCP tool loading failed for server '%s': %s", config.name, exc)
        return tools

    async def hot_reload(self) -> List[MCPToolCapability]:
        """Invalidate every cached capability and re-discover from scratch."""
        self._capability_cache.clear()
        return await self.discover_tools(force_refresh=True)

    def resolve_tool(self, tool_name: str) -> Optional[MCPToolCapability]:
        """Find a cached tool capability by exact name, or by `server.tool` qualified name."""
        for capabilities in self._capability_cache.values():
            for capability in capabilities:
                if capability.tool_name == tool_name or f"{capability.server_name}.{capability.tool_name}" == tool_name:
                    return capability
        return None


def default_servers() -> List[MCPServerConfig]:
    """Starter MCP server list. Add real servers here, or call
    MCPRegistry.register_server(...) at application startup."""
    return [
        # Example (disabled by default until a real command/url is supplied):
        # MCPServerConfig(name="filesystem", transport="stdio", command="npx",
        #                  args=["-y", "@modelcontextprotocol/server-filesystem", "."], enabled=False),
    ]


@lru_cache(maxsize=1)
def get_mcp_registry() -> MCPRegistry:
    """Process-wide MCPRegistry singleton, pre-populated with default_servers()."""
    registry = MCPRegistry()
    for config in default_servers():
        registry.register_server(config)
    return registry
