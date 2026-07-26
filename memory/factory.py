"""
Kalki Nexus - Memory Factory (Updated)

Provides namespaced memory objects for all scopes:
  - agent_memory()        AgentMemory (persistent, SQLite by default)
  - short_term()          ShortTermMemory (TTL-based, in-memory)
  - long_term()           LongTermMemory (persistent, SQLite)
  - execution_history()   ExecutionHistory (append-only log, SQLite)
  - user_preferences()    UserPreferences (persistent per-user, SQLite)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from config import Settings
from core.base_memory import StorageBackend
from memory.agent_memory import AgentMemory
from memory.backends import InMemoryStorageBackend
from memory.layers import ExecutionHistory, LongTermMemory, ShortTermMemory, UserPreferences
from memory.sqlite_backend import SQLiteStorageBackend


@lru_cache(maxsize=1)
def _get_persistent_backend() -> StorageBackend:
    return SQLiteStorageBackend()


@lru_cache(maxsize=1)
def _get_volatile_backend() -> StorageBackend:
    return InMemoryStorageBackend()


class MemoryFactory:
    """Factory providing namespaced memory objects for all scopes and backends."""

    @staticmethod
    def agent_memory(agent_name: str, settings: Optional[Settings] = None) -> AgentMemory:
        """Persistent namespaced agent memory (SQLite)."""
        return AgentMemory(backend=_get_persistent_backend(), agent_name=agent_name)

    @staticmethod
    def short_term(scope: str, ttl_seconds: int = 3600) -> ShortTermMemory:
        """Volatile TTL-based working memory (in-process, resets on restart)."""
        return ShortTermMemory(
            backend=_get_volatile_backend(),
            namespace=f"short_term:{scope}",
            ttl_seconds=ttl_seconds,
        )

    @staticmethod
    def long_term(scope: str) -> LongTermMemory:
        """Persistent cross-session long-term memory (SQLite)."""
        return LongTermMemory(backend=_get_persistent_backend(), namespace=f"long_term:{scope}")

    @staticmethod
    def execution_history(scope: str = "global") -> ExecutionHistory:
        """Persistent append-only execution history log (SQLite)."""
        return ExecutionHistory(backend=_get_persistent_backend(), namespace=f"history:{scope}")

    @staticmethod
    def user_preferences(user_id: str) -> UserPreferences:
        """Persistent per-user preferences (SQLite)."""
        return UserPreferences(backend=_get_persistent_backend(), namespace=f"prefs:{user_id}")
