"""
Kalki Nexus - BaseTool

Every tool in `tools/` subclasses BaseTool instead of being a bare
`@tool`-decorated function. This gives every tool a uniform shape - a
declared name, description, category, and required permissions - and a
single place (`__call__`) where permission checks and structured errors are
enforced before the tool logic runs.
"""
from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Optional

from langchain_core.tools import StructuredTool

from core.exceptions import ToolError
from core.permissions import Permission, SecurityContext


class BaseTool(ABC):
    """Shared interface every Kalki Nexus tool implements."""

    name: ClassVar[str]
    description: ClassVar[str]
    category: ClassVar[str] = "general"
    required_permissions: ClassVar[List[Permission]] = []

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute the tool. Subclasses implement the actual side effect here."""

    async def __call__(self, *, security_context: Optional[SecurityContext] = None, **kwargs: Any) -> Any:
        """Permission-checked entrypoint used by ToolLoader-bound LangChain tools."""
        if security_context is not None:
            security_context.check(self.required_permissions, self.name)
        try:
            return await self.run(**kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize into a structured ToolError
            raise ToolError(self.name, str(exc)) from exc

    def to_langchain_tool(self, security_context: Optional[SecurityContext] = None) -> StructuredTool:
        """Bind this tool (with its permission check already applied) as a LangChain StructuredTool.

        Uses functools.wraps on the bound run() method so LangChain's
        signature introspection (used to build the tool's argument schema
        for the LLM) sees run()'s real parameters instead of a bare **kwargs.
        """
        bound_run = self.run

        @functools.wraps(bound_run)
        async def _invoke(**kwargs: Any) -> Any:
            if security_context is not None:
                security_context.check(self.required_permissions, self.name)
            try:
                return await bound_run(**kwargs)
            except ToolError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize into a structured ToolError
                raise ToolError(self.name, str(exc)) from exc

        return StructuredTool.from_function(
            coroutine=_invoke,
            name=self.name,
            description=self.description,
        )
