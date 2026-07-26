"""
Kalki Nexus - Built-in Background Jobs

Registers the default recurring jobs on the process-wide scheduler.
Import this module at startup (e.g. in app.py) to activate all jobs.

Jobs are controlled via environment variables in .env:
  SCHEDULE_HEALTH_CHECK=true       Run health check every 30 minutes
  SCHEDULE_GITHUB_SUMMARY=true     Run GitHub summary daily at 08:00 UTC
  SCHEDULE_MARKET_SCAN=true        Run market scan daily at 06:30 UTC
"""
from __future__ import annotations

import os

from core.observability import get_logger
from core.scheduler import cron_job

logger = get_logger("kalki.jobs")


if os.getenv("SCHEDULE_HEALTH_CHECK", "true").lower() == "true":
    @cron_job("*/30 * * * *")
    async def health_check() -> None:
        """Ping the graph and confirm it compiles cleanly every 30 minutes."""
        from graph import build_graph
        graph = build_graph()
        assert graph is not None
        logger.info("health_check: graph compiled OK")


if os.getenv("SCHEDULE_GITHUB_SUMMARY", "false").lower() == "true":
    @cron_job("0 8 * * *")
    async def daily_github_summary() -> None:
        """Run a daily GitHub activity summary via the full multi-agent graph."""
        from graph import invoke
        result = await invoke("Summarize the most important GitHub activity across all watched repos today.")
        logger.info("daily_github_summary completed: %s", str(result.get("final_answer", ""))[:200])


if os.getenv("SCHEDULE_MARKET_SCAN", "false").lower() == "true":
    @cron_job("30 6 * * 1-5")
    async def daily_market_scan() -> None:
        """Run a daily pre-market scan on weekdays at 06:30 UTC via the full graph."""
        from graph import invoke
        result = await invoke("Scan current market conditions and flag any notable pre-market movements or signals.")
        logger.info("daily_market_scan completed: %s", str(result.get("final_answer", ""))[:200])
