"""
Kalki Nexus - Knowledge Base RAG Ingestion CLI

Indexes text files or markdown documentation into the RAG knowledge base.
Usage:
    python scripts/index_docs.py --collection kalki_knowledge --path ./docs/
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core.rag import Indexer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def index_path(collection: str, path: Path) -> None:
    indexer = Indexer()
    files_to_index = []

    if path.is_file():
        files_to_index.append(path)
    elif path.is_dir():
        for ext in ("*.md", "*.txt", "*.py", "*.json"):
            files_to_index.extend(path.rglob(ext))

    if not files_to_index:
        print(f"No matchable files found in {path}")
        return

    print(f"Indexing {len(files_to_index)} files into collection '{collection}'...")
    total_chunks = 0
    for file_path in files_to_index:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue
            rel_path = str(file_path.relative_to(PROJECT_ROOT)) if file_path.is_relative_to(PROJECT_ROOT) else str(file_path)
            num_chunks = await indexer.index(collection=collection, text=content, source=rel_path)
            total_chunks += num_chunks
            print(f"  [+] {rel_path} -> {num_chunks} chunks")
        except Exception as exc:  # noqa: BLE001
            print(f"  [!] Failed to index {file_path}: {exc}")

    print(f"Done! Indexed total of {total_chunks} chunks into '{collection}'.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/index_docs.py <path_to_file_or_dir> [collection_name]")
        print("Example: python scripts/index_docs.py ./README.md kalki_knowledge")
        return

    target_path = Path(sys.argv[1]).resolve()
    collection = sys.argv[2] if len(sys.argv) > 2 else "kalki_knowledge"

    asyncio.run(index_path(collection, target_path))


if __name__ == "__main__":
    main()
