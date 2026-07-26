"""
Kalki Nexus - Background Jobs Scheduler

Lightweight asyncio-based cron scheduler for recurring tasks on your
Azure VM. Jobs are defined as async callables and registered with a
cron expression. The scheduler runs as a background asyncio task
alongside your main application.

Usage:
    from core.scheduler import scheduler, cron_job

    @cron_job("0 8 * * *")   # Run every day at 08:00 UTC
    async def daily_summary():
        result = await invoke("Summarize GitHub activity for today")
        ...

    asyncio.run(scheduler.start())

Built-in jobs (configurable in config.py / .env):
  - Daily GitHub summary        (SCHEDULE_GITHUB_SUMMARY=true)
  - Market scan                 (SCHEDULE_MARKET_SCAN=true)
  - Health check                (SCHEDULE_HEALTH_CHECK=true)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from core.observability import get_logger

logger = get_logger("kalki.scheduler")


def _cron_matches(expression: str, now: datetime) -> bool:
    """Evaluate a 5-field cron expression against the given UTC datetime.
    Supports: * (wildcard), comma-separated values, and /step syntax.
    """
    fields = expression.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields

    def _matches_field(field: str, value: int, min_v: int, max_v: int) -> bool:
        if field == "*":
            return True
        for part in field.split(","):
            if "/" in part:
                base, step = part.split("/", 1)
                base_val = min_v if base == "*" else int(base)
                if (value - base_val) % int(step) == 0 and value >= base_val:
                    return True
            elif "-" in part:
                lo, hi = part.split("-", 1)
                if int(lo) <= value <= int(hi):
                    return True
            elif int(part) == value:
                return True
        return False

    return (
        _matches_field(minute, now.minute, 0, 59)
        and _matches_field(hour, now.hour, 0, 23)
        and _matches_field(dom, now.day, 1, 31)
        and _matches_field(month, now.month, 1, 12)
        and _matches_field(dow, now.weekday(), 0, 6)
    )


class ScheduledJob:
    def __init__(self, name: str, cron: str, fn: Callable[[], Coroutine[Any, Any, None]]) -> None:
        self.name = name
        self.cron = cron
        self.fn = fn
        self.last_run: Optional[datetime] = None
        self.run_count = 0
        self.error_count = 0


class JobScheduler:
    """Asyncio-native background job scheduler."""

    def __init__(self) -> None:
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running = False

    def register(self, name: str, cron: str, fn: Callable[[], Coroutine[Any, Any, None]]) -> None:
        self._jobs[name] = ScheduledJob(name=name, cron=cron, fn=fn)
        logger.info("scheduler: registered job '%s' with cron '%s'", name, cron)

    async def _run_job(self, job: ScheduledJob) -> None:
        logger.info("scheduler: running job '%s'", job.name)
        try:
            await job.fn()
            job.run_count += 1
            job.last_run = datetime.now(timezone.utc)
            logger.info("scheduler: job '%s' completed (run #%d)", job.name, job.run_count)
        except Exception as exc:  # noqa: BLE001
            job.error_count += 1
            logger.error("scheduler: job '%s' failed (error #%d): %s", job.name, job.error_count, exc)

    async def start(self) -> None:
        """Start the scheduler loop. Checks cron expressions every 60 seconds."""
        self._running = True
        logger.info("scheduler: started with %d registered jobs", len(self._jobs))
        last_minute: Optional[int] = None
        while self._running:
            now = datetime.now(timezone.utc)
            current_minute = now.year * 100000 + now.month * 10000 + now.day * 100 + now.hour * 60 + now.minute
            if current_minute != last_minute:
                last_minute = current_minute
                for job in self._jobs.values():
                    if _cron_matches(job.cron, now):
                        asyncio.create_task(self._run_job(job))
            await asyncio.sleep(15)  # poll every 15s for sub-minute accuracy

    def stop(self) -> None:
        self._running = False
        logger.info("scheduler: stopped")

    def status(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": j.name,
                "cron": j.cron,
                "last_run": j.last_run.isoformat() if j.last_run else None,
                "run_count": j.run_count,
                "error_count": j.error_count,
            }
            for j in self._jobs.values()
        ]


# Process-wide singleton
scheduler = JobScheduler()


def cron_job(cron_expression: str):
    """Decorator to register an async function as a scheduled cron job."""
    def decorator(fn: Callable[[], Coroutine[Any, Any, None]]) -> Callable[[], Coroutine[Any, Any, None]]:
        scheduler.register(name=fn.__name__, cron=cron_expression, fn=fn)
        return fn
    return decorator
