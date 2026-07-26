"""
Kalki Nexus - Observability

Rich logging, LangSmith tracing setup, and lightweight timing utilities used
by every agent and tool call. `state["metadata"]["timings"]` accumulates a
{node_name: seconds} map across a single graph run so slow nodes are visible
without attaching a profiler.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

from rich.logging import RichHandler

from config import Settings

_CONFIGURED = False


def configure_logging(settings: Settings) -> None:
    """Configure Rich-powered logging for the whole application. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. get_logger('kalki.agents.python_agent')."""
    return logging.getLogger(name)


def setup_langsmith(settings: Settings) -> bool:
    """Enable LangSmith tracing via environment variables if an API key is configured.

    Returns True if tracing was enabled. LangChain/LangGraph read these
    LANGCHAIN_* variables automatically - no code-level SDK call is required.
    """
    if not settings.langsmith_api_key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    return True


@dataclass
class _Timing:
    label: str
    start: float = field(default_factory=time.perf_counter)
    elapsed: float = 0.0


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[_Timing]:
    """Context manager that logs and records the wall-clock duration of a block.

    Usage:
        with timed(logger, "agent:python_agent") as timing:
            ...
        # timing.elapsed is now populated
    """
    timing = _Timing(label=label)
    try:
        yield timing
    finally:
        timing.elapsed = time.perf_counter() - timing.start
        logger.debug("%s took %.3fs", label, timing.elapsed)


def render_graph_mermaid(app: object, output_path: str = "graph.mmd") -> Optional[str]:
    """Write a Mermaid diagram of a compiled graph's structure. Best-effort.

    Requires nothing beyond langgraph itself for the `.mmd` text; a `.png`
    render additionally requires the optional `pygraphviz`/`grandalf`
    dependencies, so PNG export is attempted but never required.
    """
    try:
        mermaid = app.get_graph().draw_mermaid()  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - visualization is best-effort
        get_logger("kalki.observability").warning("could not render graph: %s", exc)
        return None

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(mermaid)

    try:
        png_bytes = app.get_graph().draw_mermaid_png()  # type: ignore[attr-defined]
        png_path = output_path.rsplit(".", 1)[0] + ".png"
        with open(png_path, "wb") as handle:
            handle.write(png_bytes)
    except Exception:  # noqa: BLE001 - PNG export needs extra deps; text export already succeeded
        pass

    return output_path


def extract_token_usage(response: Any) -> Dict[str, int]:
    """Extract token usage metrics (input, output, total) from an LLM response."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        return {
            "input_tokens": meta.get("input_tokens", 0),
            "output_tokens": meta.get("output_tokens", 0),
            "total_tokens": meta.get("total_tokens", 0),
        }
    if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
        usage = response.response_metadata.get("token_usage") or response.response_metadata.get("usage", {})
        if isinstance(usage, dict) and usage:
            return {
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                "total_tokens": usage.get("total_tokens", 0),
            }
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

