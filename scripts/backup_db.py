"""
Kalki Nexus - Database Backup Script

Backs up SQLite databases (kalki_memory.db, kalki_rag.db) to a timestamped
backup folder in backups/. Can be scheduled via systemd timer or cron.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = PROJECT_ROOT / "backups"


def backup() -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / timestamp
    dest.mkdir(parents=True, exist_ok=True)

    db_files = ["kalki_memory.db", "kalki_rag.db"]
    copied = 0
    for filename in db_files:
        src = PROJECT_ROOT / filename
        if src.exists():
            shutil.copy2(src, dest / filename)
            print(f"Backed up {filename} -> {dest / filename}")
            copied += 1

    if copied == 0:
        print("No database files found to back up.")
    else:
        print(f"Successfully created backup in {dest}")


if __name__ == "__main__":
    backup()
