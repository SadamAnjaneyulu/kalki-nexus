"""
Kalki Nexus - Filesystem Tools

Async filesystem access for agents, rooted at the current working directory,
registered under the "filesystem" category. Read/write/list each declare
their own permission so an agent granted only FS_READ cannot write files.
"""
from __future__ import annotations

from pathlib import Path

import aiofiles

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

# TODO: sandbox this to an allow-listed root directory before exposing it to
# agents in a production deployment.
WORKSPACE_ROOT = Path.cwd()


@ToolRegistry.register()
class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read and return the text contents of a file relative to the workspace root."
    category = "filesystem"
    required_permissions = [Permission.FS_READ]

    async def run(self, path: str) -> str:
        target = WORKSPACE_ROOT / path
        async with aiofiles.open(target, mode="r", encoding="utf-8") as handle:
            return await handle.read()


@ToolRegistry.register()
class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file relative to the workspace root, creating parent dirs."
    category = "filesystem"
    required_permissions = [Permission.FS_WRITE]

    async def run(self, path: str, content: str) -> str:
        target = WORKSPACE_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, mode="w", encoding="utf-8") as handle:
            await handle.write(content)
        return f"wrote {len(content)} bytes to {target}"


@ToolRegistry.register()
class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List file and directory names under a path, relative to the workspace root."
    category = "filesystem"
    required_permissions = [Permission.FS_READ]

    async def run(self, path: str = ".") -> list:
        target = WORKSPACE_ROOT / path
        return sorted(item.name for item in target.iterdir())


@ToolRegistry.register()
class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "Delete a file relative to the workspace root."
    category = "filesystem"
    required_permissions = [Permission.FS_DELETE]

    async def run(self, path: str) -> str:
        target = WORKSPACE_ROOT / path
        target.unlink(missing_ok=True)
        return f"deleted {target}"
