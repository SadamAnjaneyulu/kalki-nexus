"""
Kalki Nexus - Channel Adapter Protocol

Any chat surface (Discord today; Telegram, Slack, WhatsApp, a web UI, or
voice tomorrow) implements this Protocol and calls graph.py the same way
discord/bot.py does. The LangGraph graph itself has no knowledge of Discord
or any other transport - it only knows KalkiState.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class ChannelAdapter(Protocol):
    """The minimum a chat surface needs to implement to front the Kalki Nexus graph."""

    async def send(self, text: str) -> None:
        """Send a complete message to the user/channel."""
        ...

    async def stream(self, chunks: AsyncIterator[str]) -> None:
        """Stream a response incrementally (e.g. progressive message edits)."""
        ...

    async def typing(self) -> None:
        """Signal that a response is being generated (typing indicator equivalent)."""
        ...
