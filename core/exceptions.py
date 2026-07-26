"""
Kalki Nexus - Structured Exceptions

Every failure mode in the graph should raise one of these instead of a bare
Exception, so `core/resilience.py` and the graph's error node can make
informed retry/fallback decisions instead of treating all failures alike.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class KalkiError(Exception):
    """Base class for every structured Kalki Nexus exception."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


class AgentError(KalkiError):
    """Raised when a specialist agent fails to produce a result."""

    def __init__(self, agent_name: str, message: str, *, details: Optional[Dict[str, Any]] = None, retryable: bool = True) -> None:
        super().__init__(message, details={"agent": agent_name, **(details or {})}, retryable=retryable)
        self.agent_name = agent_name


class ToolError(KalkiError):
    """Raised when a tool call fails."""

    def __init__(self, tool_name: str, message: str, *, details: Optional[Dict[str, Any]] = None, retryable: bool = True) -> None:
        super().__init__(message, details={"tool": tool_name, **(details or {})}, retryable=retryable)
        self.tool_name = tool_name


class PermissionDeniedError(KalkiError):
    """Raised when a tool is invoked without the permissions it declares as required."""

    def __init__(self, tool_name: str, missing: Any) -> None:
        super().__init__(
            f"Tool '{tool_name}' requires permissions the caller was not granted: {missing}",
            details={"tool": tool_name, "missing_permissions": [str(p) for p in missing]},
            retryable=False,
        )
        self.tool_name = tool_name
        self.missing = missing


class RoutingError(KalkiError):
    """Raised when the Supervisor cannot produce a valid routing decision."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details=details, retryable=True)


class MemoryError_(KalkiError):
    """Raised on a memory backend failure. Named with a trailing underscore to avoid
    shadowing the built-in MemoryError."""

    def __init__(self, backend: str, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details={"backend": backend, **(details or {})}, retryable=True)
        self.backend = backend


class MCPError(KalkiError):
    """Raised on an MCP server/tool discovery or invocation failure."""

    def __init__(self, server_name: str, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details={"server": server_name, **(details or {})}, retryable=True)
        self.server_name = server_name


class RetryExhaustedError(KalkiError):
    """Raised by core.resilience.with_retry once every attempt has failed."""

    def __init__(self, operation: str, attempts: int, last_error: Optional[BaseException]) -> None:
        super().__init__(
            f"'{operation}' failed after {attempts} attempt(s): {last_error}",
            details={"operation": operation, "attempts": attempts, "last_error": str(last_error)},
            retryable=False,
        )
        self.last_error = last_error
