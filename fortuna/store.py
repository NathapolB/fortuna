"""SQLite connection helper + JSONL append store + idempotent insert.

SPEC §2.2 (lab.db) + §2.1 (draws.jsonl rules).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterator

from fortuna.config import (
    DRAWS_CHECKSUM,
    DRAWS_JSONL,
    LAB_DB,
    check_not_icloud,
)
from fortuna.schema import DDL, Draw

# Maximum BLOB size stored directly in features table (per SPEC §2.2 note)
MAX_BLOB_BYTES = 4 * 1024  # 4 KB


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def get_connection(db_path: Path = LAB_DB) -> sqlite3.Connection:
    """Open (and optionally create) the SQLite database. Thread-unsafe — call
    once per process and pass the connection around, or open/close per task."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Enable WAL for better concurrency (cron jobs vs notebooks)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_db(conn: sqlite3.Connection) -> None:
    """Run DDL — idempotent (uses CREATE TABLE IF NOT EXISTS). SPEC §2.2."""
    conn.executescript(DDL)
    conn.commit()


def get_or_init_db(db_path: Path = LAB_DB) -> sqlite3.Connection:
    """Open connection and ensure schema exists."""
    check_not_icloud()
    conn = get_connection(db_path)
    initialize_db(conn)
    return conn


# ---------------------------------------------------------------------------
# JSONL draw store
# ---------------------------------------------------------------------------


class DrawStore:
    """Append-only JSONL store for draws. SPEC §2.1 + §3.3."""

    def __init__(
        self,
        jsonl_path: Path = DRAWS_JSONL,
        checksum_path: Path = DRAWS_CHECKSUM,
    ) -> None:
        check_not_icloud()
        self._path = jsonl_path
        self._checksum_path = checksum_path
        self._known_ids: set[str] = self._load_known_ids()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, draw: Draw) -> bool:
        """Append draw to JSONL. Returns True if written, False if duplicate.

        SPEC §2.1: draw_id uniqueness enforced here.
        """
        if draw.draw_id in self._known_ids:
            return False

        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = draw.model_dump_json() + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)

        self._known_ids.add(draw.draw_id)
        self._update_checksum()
        return True

    def iter_draws(self) -> Iterator[Draw]:
        """Iterate all draws in insertion order."""
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield Draw.model_validate_json(line)

    def all_draws(self) -> list[Draw]:
        return list(self.iter_draws())

    def count(self) -> int:
        return len(self._known_ids)

    def contains(self, draw_id: str) -> bool:
        return draw_id in self._known_ids

    def verify_checksum(self) -> bool:
        """Return True if stored checksum matches current file hash."""
        if not self._checksum_path.exists():
            return False
        stored = self._checksum_path.read_text().strip()
        actual = self._compute_checksum()
        return stored == actual

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_known_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self._path.exists():
            return ids
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        ids.add(obj["draw_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        return ids

    def _compute_checksum(self) -> str:
        if not self._path.exists():
            return hashlib.sha256(b"").hexdigest()
        h = hashlib.sha256()
        with self._path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _update_checksum(self) -> None:
        checksum = self._compute_checksum()
        self._checksum_path.parent.mkdir(parents=True, exist_ok=True)
        self._checksum_path.write_text(checksum + "\n")


# ---------------------------------------------------------------------------
# SQLite idempotent inserts
# ---------------------------------------------------------------------------


def insert_draw_to_db(conn: sqlite3.Connection, draw: Draw) -> None:
    """Insert draw metadata into a hypothetical draws table (if it existed).
    For now, canonical storage is JSONL. This stub exists for future use."""
    # Phase 1 uses JSONL as canonical store; lab.db holds features/predictions/outcomes
    # This function is reserved for Phase 2+ when we may want SQL joins on draws.
    pass


def insert_feature(
    conn: sqlite3.Connection,
    draw_id: str,
    feature_name: str,
    feature_value: float | None,
    feature_blob: bytes | None,
    computed_at: str,
    code_sha: str,
) -> bool:
    """Idempotent feature insert. Returns True if inserted, False if already existed."""
    if feature_blob is not None and len(feature_blob) > MAX_BLOB_BYTES:
        raise ValueError(
            f"feature_blob for ({draw_id}, {feature_name}) is {len(feature_blob)} bytes "
            f"— exceeds {MAX_BLOB_BYTES} byte limit. Store large embeddings in artifacts/."
        )
    try:
        conn.execute(
            """
            INSERT INTO features (draw_id, feature_name, feature_value, feature_blob, computed_at, code_sha)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (draw_id, feature_name, feature_value, feature_blob, computed_at, code_sha),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # PRIMARY KEY conflict = already exists, idempotent
        return False


def insert_prediction(
    conn: sqlite3.Connection,
    draw_id: str,
    model_id: str,
    prize_type: str,
    pick_value: str,
    pick_rank: int,
    confidence: float | None,
    purchased: bool,
    frozen_at: str,
    freeze_commit_sha: str | None = None,
) -> int | None:
    """Insert prediction. Returns pred_id or None if UNIQUE constraint hit."""
    try:
        cur = conn.execute(
            """
            INSERT INTO predictions
              (draw_id, model_id, prize_type, pick_value, pick_rank,
               confidence, purchased, frozen_at, freeze_commit_sha)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                draw_id, model_id, prize_type, pick_value, pick_rank,
                confidence, int(purchased), frozen_at, freeze_commit_sha,
            ),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def insert_outcome(
    conn: sqlite3.Connection,
    pred_id: int,
    hit: bool,
    payout_thb: int,
    cost_thb: int,
    settled_at: str,
) -> bool:
    """Idempotent outcome insert. Returns True if inserted."""
    if cost_thb not in (0, 80):
        raise ValueError(f"cost_thb must be 0 or 80, got {cost_thb}. SPEC §2.2.")
    try:
        conn.execute(
            """
            INSERT INTO outcomes (pred_id, hit, payout_thb, cost_thb, settled_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (pred_id, int(hit), payout_thb, cost_thb, settled_at),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_feature(
    conn: sqlite3.Connection, draw_id: str, feature_name: str
) -> sqlite3.Row | None:
    """Fetch a feature row by (draw_id, feature_name). Used by leakage tests (SPEC §7.3)."""
    row = conn.execute(
        "SELECT * FROM features WHERE draw_id = ? AND feature_name = ?",
        (draw_id, feature_name),
    ).fetchone()
    return row
