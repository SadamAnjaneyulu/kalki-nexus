"""
Kalki Nexus - SQLite Storage Backend

Persistent key/value store using aiosqlite. Default backend when
MEMORY_BACKEND=sqlite (the default). DB file location can be
overridden by SQLITE_DB_PATH in .env (default: kalki_memory.db in
the project root).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List, Optional

import aiosqlite

from core.base_memory import StorageBackend

DB_PATH = Path(os.getenv("SQLITE_DB_PATH", "kalki_memory.db"))

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS kv_store (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    expires_at REAL,
    PRIMARY KEY (namespace, key)
)
"""


class SQLiteStorageBackend(StorageBackend):
    """Persistent storage backend backed by aiosqlite."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path

    async def _conn(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_path)
        await conn.execute(_CREATE_TABLE)
        await conn.commit()
        return conn

    async def get(self, namespace: str, key: str) -> Optional[Any]:
        async with await self._conn() as db:
            async with db.execute(
                "SELECT value, expires_at FROM kv_store WHERE namespace=? AND key=?",
                (namespace, key),
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            return None
        val_json, expires_at = row
        if expires_at is not None and time.time() > expires_at:
            await self.delete(namespace, key)
            return None
        return json.loads(val_json)

    async def set(self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        expires_at = (time.time() + ttl_seconds) if ttl_seconds else None
        val_json = json.dumps(value)
        async with await self._conn() as db:
            await db.execute(
                "INSERT OR REPLACE INTO kv_store (namespace, key, value, expires_at) VALUES (?, ?, ?, ?)",
                (namespace, key, val_json, expires_at),
            )
            await db.commit()

    async def delete(self, namespace: str, key: str) -> None:
        async with await self._conn() as db:
            await db.execute("DELETE FROM kv_store WHERE namespace=? AND key=?", (namespace, key))
            await db.commit()

    async def list_keys(self, namespace: str, prefix: str = "") -> List[str]:
        now = time.time()
        async with await self._conn() as db:
            async with db.execute(
                "SELECT key FROM kv_store WHERE namespace=? AND key LIKE ? AND (expires_at IS NULL OR expires_at > ?)",
                (namespace, f"{prefix}%", now),
            ) as cursor:
                rows = await cursor.fetchall()
        return [row[0] for row in rows]
