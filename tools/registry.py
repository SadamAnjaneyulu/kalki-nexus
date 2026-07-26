"""
Kalki Nexus - Tool Registry, Loader, and Resolver

Tools register themselves once (via the @ToolRegistry.register() class
decorator) instead of every agent maintaining its own manual
`llm.bind_tools([...])` list. Agents request tools by *category*
(`ToolLoader.load_categories`) or by explicit name
(`ToolResolver.resolve_requested`, used for the `requested_tools` the
Supervisor was handed).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from core.base_tool import BaseTool
from core.permissions import SecurityContext


class ToolRegistry:
    """Global registry of every BaseTool subclass, keyed by tool name."""

    _tools: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls):
        """Class decorator: `@ToolRegistry.register()` above a BaseTool subclass."""

        def decorator(tool_cls: Type[BaseTool]) -> Type[BaseTool]:
            cls._tools[tool_cls.name] = tool_cls
            return tool_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        tool_cls = cls._tools.get(name)
        return tool_cls() if tool_cls else None

    @classmethod
    def by_category(cls, category: str) -> List[BaseTool]:
        return [tool_cls() for tool_cls in cls._tools.values() if tool_cls.category == category]

    @classmethod
    def all(cls) -> List[BaseTool]:
        return [tool_cls() for tool_cls in cls._tools.values()]

    @classmethod
    def categories(cls) -> List[str]:
        return sorted({tool_cls.category for tool_cls in cls._tools.values()})


class ToolLoader:
    """Loads every registered tool in a set of categories as LangChain tools."""

    @staticmethod
    def load_categories(categories: List[str], security_context: Optional[SecurityContext] = None) -> List[Any]:
        tools: List[Any] = []
        for category in categories:
            for tool in ToolRegistry.by_category(category):
                tools.append(tool.to_langchain_tool(security_context))
        return tools


class ToolResolver:
    """Resolves explicit tool names (e.g. state["requested_tools"]) to LangChain tools."""

    @staticmethod
    def resolve_requested(names: List[str], security_context: Optional[SecurityContext] = None) -> List[Any]:
        resolved: List[Any] = []
        for name in names:
            tool = ToolRegistry.get(name)
            if tool is not None:
                resolved.append(tool.to_langchain_tool(security_context))
        return resolved
