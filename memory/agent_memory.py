"""
Kalki Nexus - Agent Memory
"""
from __future__ import annotations

from typing import Any, List, Optional
from core.base_memory import BaseMemory, StorageBackend


class AgentMemory(BaseMemory):
    """Namespaced memory for a single specialist agent."""

    def __init__(self, backend: StorageBackend, agent_name: str) -> None:
        super().__init__(backend, namespace=f"agent:{agent_name}")

    async def get(self, key: str) -> Optional[Any]:
        return await self.backend.get(self.namespace, key)

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        await self.backend.set(self.namespace, key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        await self.backend.delete(self.namespace, key)

    async def list_keys(self, prefix: str = "") -> List[str]:
        return await self.backend.list_keys(self.namespace, prefix)
