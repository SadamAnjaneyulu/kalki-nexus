"""
Kalki Nexus - BaseMemory & StorageBackend

BaseMemory is the interface every memory layer (Working, Conversation,
Session, Agent, Shared, Long-Term) implements. StorageBackend is the
interface every physical store (SQLite, Postgres, Redis, Qdrant, Chroma)
implements. Memory layers hold a StorageBackend by composition, so swapping
SQLite for Postgres/Redis/Qdrant/Chroma is a config change, not an agent
code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class StorageBackend(ABC):
    """Physical storage interface: get/set/delete/list_keys against a namespaced store."""

    @abstractmethod
    async def get(self, namespace: str, key: str) -> Optional[Any]: ...

    @abstractmethod
    async def set(self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None: ...

    @abstractmethod
    async def delete(self, namespace: str, key: str) -> None: ...

    @abstractmethod
    async def list_keys(self, namespace: str, prefix: str = "") -> List[str]: ...


class BaseMemory(ABC):
    """Interface every memory layer implements, regardless of scope or backend."""

    namespace: str

    def __init__(self, backend: StorageBackend, namespace: str) -> None:
        self.backend = backend
        self.namespace = namespace

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]: ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> List[str]: ...
