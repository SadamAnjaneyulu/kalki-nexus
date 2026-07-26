"""
Kalki Nexus - Storage Backends
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from core.base_memory import StorageBackend


class InMemoryStorageBackend(StorageBackend):
    """In-memory storage backend for development and testing."""

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, str], Tuple[Any, Optional[float]]] = {}

    async def get(self, namespace: str, key: str) -> Optional[Any]:
        entry = self._store.get((namespace, key))
        if not entry:
            return None
        val, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            del self._store[(namespace, key)]
            return None
        return val

    async def set(self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        expires_at = (time.time() + ttl_seconds) if ttl_seconds is not None else None
        self._store[(namespace, key)] = (value, expires_at)

    async def delete(self, namespace: str, key: str) -> None:
        self._store.pop((namespace, key), None)

    async def list_keys(self, namespace: str, prefix: str = "") -> List[str]:
        now = time.time()
        keys: List[str] = []
        expired: List[Tuple[str, str]] = []
        for (ns, k), (val, expires_at) in self._store.items():
            if ns == namespace and k.startswith(prefix):
                if expires_at is not None and now > expires_at:
                    expired.append((ns, k))
                else:
                    keys.append(k)
        for k in expired:
            del self._store[k]
        return keys
