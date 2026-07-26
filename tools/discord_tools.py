"""
Kalki Nexus - Discord Tools

BaseTool wrapper for sending messages back to Discord from an agent,
registered under the "discord" category. Expects a running Discord client
instance to be attached at startup via bind_client().
"""
from __future__ import annotations

from typing import Optional

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

_client: Optional[object] = None


def bind_client(client: object) -> None:
    """Attach the running Discord client so these tools can send messages through it.
    Call this from discord/bot.py's on_ready handler."""
    global _client
    _client = client


@ToolRegistry.register()
class SendChannelMessageTool(BaseTool):
    name = "send_channel_message"
    description = "Send a message to a Discord channel by its numeric ID."
    category = "discord"
    required_permissions = [Permission.DISCORD]

    async def run(self, channel_id: int, content: str) -> str:
        if _client is None:
            raise RuntimeError("Discord client is not bound. Call bind_client() from discord/bot.py at startup.")
        channel = _client.get_channel(channel_id)  # type: ignore[attr-defined]
        if channel is None:
            raise ValueError(f"Discord channel {channel_id} not found or not cached.")
        await channel.send(content)
        return f"sent {len(content)} chars to channel {channel_id}"
