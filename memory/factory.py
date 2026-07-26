"""
Kalki Nexus - Memory Factory
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from config import Settings
from core.base_memory import StorageBackend
from memory.agent_memory import AgentMemory
from memory.backends import InMemoryStorageBackend


@lru_cache(maxsize=1)
def _get_shared_backend() -> StorageBackend:
    return InMemoryStorageBackend()


class MemoryFactory:
    """Factory creating namespaced memory objects for agents and graph components."""

    @staticmethod
    def agent_memory(agent_name: str, settings: Optional[Settings] = None) -> AgentMemory:
        backend = _get_shared_backend()
        return AgentMemory(backend=backend, agent_name=agent_name)
