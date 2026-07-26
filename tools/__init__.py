"""Kalki Nexus tools package: BaseTool implementations plus the ToolRegistry.

Importing this package eagerly imports every tool module so their
`@ToolRegistry.register()` decorators run and the registry is fully
populated before any agent calls ToolLoader.load_categories(...).
"""
from __future__ import annotations

from tools import (  # noqa: F401 - imported for registration side effects
    browser_tools,
    discord_tools,
    docker_tools,
    filesystem_tools,
    github_tools,
    terminal_tools,
    web_tools,
)
