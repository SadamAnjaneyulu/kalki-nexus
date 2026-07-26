"""Kalki Nexus agents package.

Every module here (except supervisor.py and aggregator.py, which are
orchestration nodes rather than routable specialists) is auto-discovered by
core.registry.discover_agents(). Add a new specialist by dropping a
BaseAgent subclass in a new module here - nothing else needs to change.
"""
