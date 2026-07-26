"""
Kalki Nexus - Web Tools

BaseTool wrappers for HTTP fetch and web search, registered under the
"web" category. fetch_url is fully implemented. web_search uses DuckDuckGo
HTML scraping as a zero-cost, no-API-key-required default with an optional
upgrade path to Tavily (set TAVILY_API_KEY in .env).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

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
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0 KalkiNexus/1.0"})
            response.raise_for_status()
            return response.text


@ToolRegistry.register()
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for a query and return up to max_results result snippets with titles and URLs."
    category = "web"
    required_permissions = [Permission.NETWORK]

    async def run(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_key:
            return await self._tavily_search(query, max_results, tavily_key)
        return await self._ddg_search(query, max_results)

    async def _tavily_search(self, query: str, max_results: int, api_key: str) -> List[Dict[str, str]]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in data.get("results", [])[:max_results]
            ]

    async def _ddg_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Zero-dependency DuckDuckGo HTML fallback."""
        url = "https://html.duckduckgo.com/html/"
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = await client.post(url, data={"q": query}, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        results = []
        # Extract result titles and URLs from HTML using simple regex
        title_pattern = re.compile(r'<a class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
        snippet_pattern = re.compile(r'<a class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
        titles_urls = title_pattern.findall(resp.text)
        snippets = snippet_pattern.findall(resp.text)
        for i, (url, title) in enumerate(titles_urls[:max_results]):
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            results.append({"title": clean_title, "url": url, "snippet": snippet})
        return results
