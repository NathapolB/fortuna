#!/usr/bin/env python3
"""Migration: add notion_page_id column to predictions table. SPEC §Enhancement-1.

Run once against an existing lab.db that was created before v2.2.
New databases created via get_or_init_db() already include this column
(it is in the DDL in fortuna/schema.py as of v2.2).

Usage:
    cd ~/projects/fortuna
    .venv/bin/python scripts/migrate_add_notion_page_id.py

    # Or against a specific DB path:
    .venv/bin/python scripts/migrate_add_notion_page_id.py --db data/lab.db

The script is idempotent — it checks whether the column already exists
before attempting ALTER TABLE, so it is safe to run multiple times.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if column already exists in table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    rows = cursor.fetchall()
    return any(row[1] == column for row in rows)


def migrate(db_path: Path) -> None:
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        if column_exists(conn, "predictions", "notion_page_id"):
            print(
                f"Column 'notion_page_id' already exists in predictions table of {db_path}. "
                "No migration needed."
            )
            return

        print(f"Adding column 'notion_page_id TEXT' to predictions table in {db_path} ...")
        conn.execute("ALTER TABLE predictions ADD COLUMN notion_page_id TEXT")
        conn.commit()
        print("Migration complete.")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add notion_page_id column to predictions table (v2.2 migration)"
    )
    default_db = Path(__file__).parent.parent / "data" / "lab.db"
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db,
        metavar="PATH",
        help=f"Path to lab.db (default: {default_db})",
    )
    args = parser.parse_args()
    migrate(args.db)


if __name__ == "__main__":
    main()
