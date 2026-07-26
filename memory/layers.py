"""
Kalki Nexus - Memory Layers

Concrete memory layer implementations for different scopes:

  ShortTermMemory   - TTL-based working memory for a single run (default 1h)
  LongTermMemory    - Persistent memory for cross-session knowledge
  ExecutionHistory  - Append-only log of agent runs and results
  UserPreferences   - Persistent user settings per user ID
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.base_memory import BaseMemory, StorageBackend


class ShortTermMemory(BaseMemory):
    """TTL-based working memory that expires entries after a configurable window."""

    default_ttl_seconds: int = 3600  # 1 hour

    def __init__(self, backend: StorageBackend, namespace: str, ttl_seconds: int = 3600) -> None:
        super().__init__(backend, namespace)
        self.default_ttl_seconds = ttl_seconds

    async def get(self, key: str) -> Optional[Any]:
        return await self.backend.get(self.namespace, key)

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        await self.backend.set(self.namespace, key, value, ttl_seconds or self.default_ttl_seconds)

    async def delete(self, key: str) -> None:
        await self.backend.delete(self.namespace, key)

    async def list_keys(self, prefix: str = "") -> List[str]:
        return await self.backend.list_keys(self.namespace, prefix)


class LongTermMemory(BaseMemory):
    """Persistent memory that survives across sessions. No TTL by default."""

    async def get(self, key: str) -> Optional[Any]:
        return await self.backend.get(self.namespace, key)

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        await self.backend.set(self.namespace, key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        await self.backend.delete(self.namespace, key)

    async def list_keys(self, prefix: str = "") -> List[str]:
        return await self.backend.list_keys(self.namespace, prefix)


class ExecutionHistory(BaseMemory):
    """Append-only chronological log of agent run metadata."""

    async def get(self, key: str) -> Optional[Any]:
        return await self.backend.get(self.namespace, key)

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        await self.backend.set(self.namespace, key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        await self.backend.delete(self.namespace, key)

    async def list_keys(self, prefix: str = "") -> List[str]:
        return await self.backend.list_keys(self.namespace, prefix)

    async def append(self, entry: Dict[str, Any]) -> str:
        """Append a timestamped entry and return its key."""
        key = f"run:{time.time():.6f}"
        await self.backend.set(self.namespace, key, entry)
        return key

    async def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the N most recent history entries, newest first."""
        keys = sorted(await self.backend.list_keys(self.namespace, "run:"), reverse=True)
        entries = []
        for k in keys[:n]:
            val = await self.backend.get(self.namespace, k)
            if val is not None:
                entries.append(val)
        return entries


class UserPreferences(BaseMemory):
    """Persistent per-user preference store."""

    async def get(self, key: str) -> Optional[Any]:
        return await self.backend.get(self.namespace, key)

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        await self.backend.set(self.namespace, key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        await self.backend.delete(self.namespace, key)

    async def list_keys(self, prefix: str = "") -> List[str]:
        return await self.backend.list_keys(self.namespace, prefix)
