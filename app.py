"""
Kalki Nexus - Application Entrypoint

Boots Rich-powered logging + LangSmith tracing, loads configuration, and
either runs a sample graph invocation, renders the graph structure, or
launches the Discord bot. Also activates the background job scheduler.
"""
from __future__ import annotations

import asyncio
import sys

from config import get_settings
from core.observability import configure_logging, get_logger, render_graph_mermaid, setup_langsmith
from graph import compiled_graph, invoke

logger = get_logger("kalki.app")


async def run_example() -> None:
    """Run a single example request through the Kalki Nexus graph."""
    result = await invoke("Write a Python script that backtests a VWAP mean reversion strategy.")
    logger.info("Final answer: %s", result.get("final_answer"))


def render_graph() -> None:
    """Compile the graph and write graph.mmd (and graph.png, if renderable) without invoking it."""
    app = compiled_graph()
    path = render_graph_mermaid(app)
    if path:
        logger.info("Graph structure written to %s", path)
    else:
        logger.warning("Could not render the graph (see warning above).")


async def run_with_scheduler() -> None:
    """Run example query + background job scheduler concurrently."""
    import core.jobs  # noqa: F401 - registers all built-in cron jobs on import
    from core.scheduler import scheduler

    await asyncio.gather(
        run_example(),
        scheduler.start(),
    )


def main() -> None:
    """CLI entrypoint.

    --discord           launch the Discord bot instead of the example run
    --render-graph      write graph.mmd/graph.png and exit
    --with-scheduler    run background jobs alongside the example invocation
    """
    settings = get_settings()
    configure_logging(settings)
    if setup_langsmith(settings):
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)

    if "--discord" in sys.argv:
        from discord.bot import run as run_discord_bot  # local import: only needed for this path
        run_discord_bot()
    elif "--render-graph" in sys.argv:
        render_graph()
    elif "--with-scheduler" in sys.argv:
        asyncio.run(run_with_scheduler())
    else:
        asyncio.run(run_example())


if __name__ == "__main__":
    main()
