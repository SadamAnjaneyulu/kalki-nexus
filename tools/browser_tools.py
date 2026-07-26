"""
Kalki Nexus - Browser Tools

Placeholder BaseTool wrappers for headless-browser automation (e.g.
Playwright), registered under the "browser" category. Install `playwright`
and run `playwright install` before wiring these up.
"""
from __future__ import annotations

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry


@ToolRegistry.register()
class OpenPageTool(BaseTool):
    name = "open_page"
    description = "Open a URL in a headless browser and return the rendered page title."
    category = "browser"
    required_permissions = [Permission.BROWSER, Permission.NETWORK]

    async def run(self, url: str) -> str:
        # TODO: launch Playwright (async_playwright), navigate to `url`, return page.title()
        raise NotImplementedError("open_page: wire up a headless browser (e.g. Playwright).")


@ToolRegistry.register()
class ClickSelectorTool(BaseTool):
    name = "click_selector"
    description = "Navigate to a URL and click the element matching a CSS selector."
    category = "browser"
    required_permissions = [Permission.BROWSER, Permission.NETWORK]

    async def run(self, url: str, selector: str) -> str:
        # TODO: launch Playwright, page.goto(url), then page.click(selector)
        raise NotImplementedError("click_selector: wire up a headless browser (e.g. Playwright).")


@ToolRegistry.register()
class ExtractTextTool(BaseTool):
    name = "extract_text"
    description = "Navigate to a URL and return the text content of a CSS selector."
    category = "browser"
    required_permissions = [Permission.BROWSER, Permission.NETWORK]

    async def run(self, url: str, selector: str = "body") -> str:
        # TODO: launch Playwright, page.goto(url), return page.inner_text(selector)
        raise NotImplementedError("extract_text: wire up a headless browser (e.g. Playwright).")
