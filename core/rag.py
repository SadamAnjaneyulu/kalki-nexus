"""
Kalki Nexus - RAG (Retrieval Augmented Generation)

Simple, dependency-light RAG subsystem backed by SQLite + BM25-style
full-text search (no vector DB required out of the box). Agents can
index documents and retrieve relevant context before generating responses.

For production-grade semantic search, swap the SQLite backend for Qdrant
or Chroma by setting MEMORY_BACKEND=qdrant or MEMORY_BACKEND=chroma.

Architecture:
  Indexer  - chunks and stores documents
  Retriever - BM25 keyword search over indexed chunks
  RAGContext - injects retrieved context into agent prompts

Usage:
    from core.rag import Indexer, Retriever

    indexer = Indexer()
    await indexer.index("worldquant_docs", "VWAP is Volume Weighted Average...")

    retriever = Retriever()
    chunks = await retriever.retrieve("worldquant_docs", "What is VWAP?", top_k=3)
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

RAG_DB_PATH = Path(os.getenv("RAG_DB_PATH", "kalki_rag.db"))

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL,
    source TEXT,
    chunk_index INTEGER,
    text TEXT NOT NULL,
    indexed_at REAL
);
CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(
    text, collection UNINDEXED, chunk_id UNINDEXED,
    content='rag_chunks', content_rowid='id'
);
"""


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class Indexer:
    """Chunks and indexes documents into the RAG store."""

    def __init__(self, db_path: Path = RAG_DB_PATH) -> None:
        self.db_path = db_path

    async def _setup(self, db: aiosqlite.Connection) -> None:
        for stmt in _CREATE_SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                await db.execute(stmt)
        await db.commit()

    async def index(self, collection: str, text: str, source: str = "", chunk_size: int = 500) -> int:
        """Index a document into the collection. Returns number of chunks stored."""
        chunks = _chunk_text(text, chunk_size=chunk_size)
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await self._setup(db)
            for i, chunk in enumerate(chunks):
                cursor = await db.execute(
                    "INSERT INTO rag_chunks (collection, source, chunk_index, text, indexed_at) VALUES (?, ?, ?, ?, ?)",
                    (collection, source, i, chunk, now),
                )
                chunk_id = cursor.lastrowid
                await db.execute(
                    "INSERT INTO rag_fts (rowid, text, collection, chunk_id) VALUES (?, ?, ?, ?)",
                    (chunk_id, chunk, collection, chunk_id),
                )
            await db.commit()
        return len(chunks)

    async def clear_collection(self, collection: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await self._setup(db)
            await db.execute("DELETE FROM rag_chunks WHERE collection=?", (collection,))
            await db.commit()

    async def list_collections(self) -> List[str]:
        async with aiosqlite.connect(self.db_path) as db:
            await self._setup(db)
            async with db.execute("SELECT DISTINCT collection FROM rag_chunks") as cur:
                rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def count(self, collection: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            await self._setup(db)
            async with db.execute("SELECT COUNT(*) FROM rag_chunks WHERE collection=?", (collection,)) as cur:
                row = await cur.fetchone()
        return row[0] if row else 0


class Retriever:
    """Retrieves relevant chunks from the RAG store using FTS5 full-text search."""

    def __init__(self, db_path: Path = RAG_DB_PATH) -> None:
        self.db_path = db_path

    async def retrieve(self, collection: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k relevant chunks for query from the collection."""
        safe_query = re.sub(r'[^\w\s]', ' ', query).strip()
        words = [w for w in safe_query.split() if len(w) > 1]
        if not words:
            return []

        fts_query = " OR ".join(words)
        async with aiosqlite.connect(self.db_path) as db:
            rows = []
            try:
                async with db.execute(
                    """SELECT rc.text, rc.source, rc.chunk_index, rank
                       FROM rag_fts f
                       JOIN rag_chunks rc ON rc.id = f.rowid
                       WHERE rag_fts MATCH ? AND f.collection = ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, collection, top_k),
                ) as cur:
                    rows = await cur.fetchall()
            except Exception:
                rows = []

            if not rows:
                like_clauses = " OR ".join(["rc.text LIKE ?"] * len(words))
                params = [f"%{w}%" for w in words] + [collection, top_k]
                sql = f"""SELECT rc.text, rc.source, rc.chunk_index, 0.0
                          FROM rag_chunks rc
                          WHERE ({like_clauses}) AND rc.collection = ?
                          LIMIT ?"""
                try:
                    async with db.execute(sql, params) as cur:
                        rows = await cur.fetchall()
                except Exception:
                    rows = []

        return [
            {"text": r[0], "source": r[1], "chunk_index": r[2], "score": r[3]}
            for r in rows
        ]

    async def build_context(self, collection: str, query: str, top_k: int = 5) -> str:
        """Return retrieved chunks formatted as a prompt context block."""
        chunks = await self.retrieve(collection, query, top_k=top_k)
        if not chunks:
            return ""
        parts = [f"[Source: {c['source'] or 'unknown'}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks]
        return "Relevant context:\n\n" + "\n\n---\n\n".join(parts)
