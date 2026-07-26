"""
Kalki Nexus - Plugin Registry (Agent Auto-Discovery)

Drop a new `agents/<name>_agent.py` defining a BaseAgent subclass and it is
picked up automatically - graph.py and the Supervisor never need to import
it by name. This is what "add an agent by dropping a file into agents/"
means in practice: `discover_agents()` walks the package, imports every
module, and collects every concrete BaseAgent subclass it finds.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from functools import lru_cache
from typing import Dict, Type

import agents as agents_package
from core.base_agent import BaseAgent


@lru_cache(maxsize=1)
def discover_agents() -> Dict[str, Type[BaseAgent]]:
    """Import every module under agents/ and return {agent.name: AgentClass}
    for every concrete BaseAgent subclass found, excluding the Supervisor
    and Aggregator (which are orchestration nodes, not routable specialists)."""
    registry: Dict[str, Type[BaseAgent]] = {}
    skip_modules = {"supervisor", "aggregator"}

    for module_info in pkgutil.iter_modules(agents_package.__path__):
        if module_info.name in skip_modules or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"agents.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseAgent)
                and obj is not BaseAgent
                and obj.__module__ == module.__name__
                and getattr(obj, "name", None)
            ):
                registry[obj.name] = obj

    return registry


def clear_agent_cache() -> None:
    """Invalidate the discovery cache (used by tests and hot-reload tooling)."""
    discover_agents.cache_clear()
