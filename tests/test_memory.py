"""
Unit tests for Kalki Nexus Memory subsystem.
"""
import pytest
from memory.backends import InMemoryStorageBackend
from memory.agent_memory import AgentMemory
from memory.factory import MemoryFactory


@pytest.mark.asyncio
async def test_in_memory_storage_backend():
    backend = InMemoryStorageBackend()
    
    # Test set & get
    await backend.set("test_ns", "key1", "val1")
    val = await backend.get("test_ns", "key1")
    assert val == "val1"

    # Test list_keys
    await backend.set("test_ns", "key2", "val2")
    keys = await backend.list_keys("test_ns")
    assert "key1" in keys
    assert "key2" in keys

    # Test delete
    await backend.delete("test_ns", "key1")
    val_after_del = await backend.get("test_ns", "key1")
    assert val_after_del is None


@pytest.mark.asyncio
async def test_agent_memory():
    backend = InMemoryStorageBackend()
    mem = AgentMemory(backend, "test_agent")

    assert mem.namespace == "agent:test_agent"
    await mem.set("session_data", {"active": True})
    res = await mem.get("session_data")
    assert res == {"active": True}


def test_memory_factory():
    mem = MemoryFactory.agent_memory("research_agent")
    assert mem.namespace == "agent:research_agent"
