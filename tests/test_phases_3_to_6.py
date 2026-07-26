"""
Tests for Phase 3 (Persistent Memory), Phase 4 (Delegation),
Phase 5 (Scheduler), and Phase 6 (RAG).
"""
import pytest
from memory.layers import ShortTermMemory, LongTermMemory, ExecutionHistory, UserPreferences
from memory.backends import InMemoryStorageBackend
from memory.factory import MemoryFactory
from core.scheduler import JobScheduler, _cron_matches
from datetime import datetime, timezone


# ── Phase 3: Layered Memory ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_short_term_memory():
    backend = InMemoryStorageBackend()
    mem = ShortTermMemory(backend, "test_scope", ttl_seconds=10)
    await mem.set("key1", "value1")
    assert await mem.get("key1") == "value1"
    keys = await mem.list_keys()
    assert "key1" in keys


@pytest.mark.asyncio
async def test_long_term_memory():
    backend = InMemoryStorageBackend()
    mem = LongTermMemory(backend, "long_term:test")
    await mem.set("preference", {"theme": "dark"})
    result = await mem.get("preference")
    assert result == {"theme": "dark"}


@pytest.mark.asyncio
async def test_execution_history_append_and_recent():
    backend = InMemoryStorageBackend()
    history = ExecutionHistory(backend, "history:test")
    await history.append({"agent": "python_agent", "duration": 1.2})
    await history.append({"agent": "quant_agent", "duration": 2.3})
    recent = await history.recent(n=10)
    assert len(recent) == 2


@pytest.mark.asyncio
async def test_user_preferences():
    backend = InMemoryStorageBackend()
    prefs = UserPreferences(backend, "prefs:user123")
    await prefs.set("language", "python")
    val = await prefs.get("language")
    assert val == "python"


def test_memory_factory_returns_correct_types():
    assert isinstance(MemoryFactory.short_term("test"), ShortTermMemory)
    assert isinstance(MemoryFactory.long_term("test"), LongTermMemory)
    assert isinstance(MemoryFactory.execution_history("test"), ExecutionHistory)
    assert isinstance(MemoryFactory.user_preferences("user42"), UserPreferences)


# ── Phase 5: Scheduler ────────────────────────────────────────────────────────

def test_cron_matches_every_minute():
    now = datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)
    assert _cron_matches("* * * * *", now) is True


def test_cron_matches_exact():
    now = datetime(2025, 1, 15, 8, 0, tzinfo=timezone.utc)
    assert _cron_matches("0 8 * * *", now) is True
    assert _cron_matches("0 9 * * *", now) is False


def test_cron_matches_step():
    now = datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)
    assert _cron_matches("*/30 * * * *", now) is True
    assert _cron_matches("*/15 * * * *", now) is True


def test_scheduler_register():
    sched = JobScheduler()

    async def dummy_job():
        pass

    sched.register("test_job", "* * * * *", dummy_job)
    status = sched.status()
    assert len(status) == 1
    assert status[0]["name"] == "test_job"
