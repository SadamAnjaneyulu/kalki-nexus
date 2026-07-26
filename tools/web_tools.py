"""
Kalki Nexus - Web Tools

BaseTool wrappers for HTTP fetch and web search, registered under the "web"
category. fetch_url is fully implemented; web_search is a stub pending a
search provider (e.g. Tavily, Serper, Bing).
"""
from __future__ import annotations

import httpx

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

REQUEST_TIMEOUT_SECONDS = 15.0


@ToolRegistry.register()
class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = "Fetch a URL over HTTP(S) and return the response body as text."
    category = "web"
    required_permissions = [Permission.NETWORK]

    async def run(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text


@ToolRegistry.register()
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for a query and return up to max_results result snippets."
    category = "web"
    required_permissions = [Permission.NETWORK]

    async def run(self, query: str, max_results: int = 5) -> list:
        # TODO: call a search provider such as Tavily, Serper, or Bing Web Search here.
        raise NotImplementedError("web_search: wire up a search provider (e.g. Tavily or Serper).")
