"""
Kalki Nexus - Hermes Discord Bot

Connects Hermes (Discord bot) directly to the Kalki Nexus multi-agent graph.
Listens for messages in text channels, invokes `graph.invoke()`, and sends the
synthesized final answer back to the channel.
"""
from __future__ import annotations

import asyncio
from typing import List

import discord

from config import get_settings
from core.observability import get_logger
from graph import invoke
from tools.discord_tools import bind_client

logger = get_logger("kalki.discord")


def _chunk_message(text: str, limit: int = 1990) -> List[str]:
    """Split long responses into Discord-compliant chunks (max 2000 chars)."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_idx = text.rfind("\n", 0, limit)
        if split_idx == -1:
            split_idx = limit
        chunks.append(text[:split_idx])
        text = text[split_idx:].lstrip("\n")
    return chunks


class HermesBot(discord.Client):
    def __init__(self, **options):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **options)

    async def on_ready(self):
        logger.info("Hermes Discord bot logged in as %s (ID: %s)", self.user, self.user.id)
        bind_client(self)

    async def on_message(self, message: discord.Message):
        if message.author == self.user or message.author.bot:
            return

        user_input = message.content.strip()
        if not user_input:
            return

        channel_name = getattr(message.channel, "name", "dm")
        logger.info("Received Discord message from %s in #%s: %r", message.author, channel_name, user_input[:50])

        async with message.channel.typing():
            try:
                attached_files = [att.filename for att in message.attachments]
                result = await invoke(
                    user_input=user_input,
                    discord_channel=channel_name,
                    attached_files=attached_files,
                    thread_id=f"discord_{message.channel.id}",
                )
                answer = result.get("final_answer") or "Sorry, no response was generated."

                for chunk in _chunk_message(answer):
                    await message.channel.send(chunk)

            except Exception as exc:  # noqa: BLE001
                logger.exception("Error processing Discord message")
                await message.channel.send(f"⚠️ An error occurred while processing your request: {exc}")


def run() -> None:
    """Launch the Hermes Discord bot. Requires DISCORD_TOKEN in .env."""
    settings = get_settings()
    if not settings.discord_token:
        logger.error("DISCORD_TOKEN is not configured in .env!")
        return
    bot = HermesBot()
    bot.run(settings.discord_token)
