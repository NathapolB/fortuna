"""Deduplication tests for DrawStore and SQLite predictions. SPEC Phase 1.

Tests:
- Insert same draw twice → no duplication
- UNIQUE (draw_id, model_id, prize_type, pick_rank) constraint
- UNIQUE (draw_id, model_id, prize_type, pick_value) constraint
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from fortuna.schema import Draw, DDL
from fortuna.store import DrawStore, get_or_init_db, insert_prediction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_draw(draw_id: str = "2024-01-01") -> Draw:
    return Draw(
        draw_date=draw_id,
        draw_id=draw_id,
        first_prize="123456",
        first_prize_near=["123455", "123457"],
        three_digit_front=["123", "456"],
        three_digit_back=["789", "012"],
        two_digit_back="56",
        bonus_prizes={},
        source="news.sanook.com",
        source_url=f"https://news.sanook.com/lotto/{draw_id}/",
        scraped_at="2024-01-01T17:00:00+07:00",
        raw_html_sha256="abc123def456" * 5 + "ab",
        verified_against=["kapook.com"],
        schema_version=1,
    )


@pytest.fixture()
def tmp_jsonl(tmp_path: Path) -> Path:
    return tmp_path / "draws.jsonl"


@pytest.fixture()
def store(tmp_jsonl: Path, tmp_path: Path) -> DrawStore:
    checksum_path = tmp_path / "draws.checksum"
    return DrawStore(jsonl_path=tmp_jsonl, checksum_path=checksum_path)


@pytest.fixture()
def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# DrawStore dedup tests
# ---------------------------------------------------------------------------


class TestDrawStoreDedup:
    def test_insert_once_succeeds(self, store: DrawStore) -> None:
        draw = _make_draw("2024-01-01")
        result = store.append(draw)
        assert result is True
        assert store.count() == 1

    def test_insert_same_draw_twice_no_duplication(self, store: DrawStore) -> None:
        """Core dedup test: same draw_id must not be written twice."""
        draw = _make_draw("2024-01-01")
        first = store.append(draw)
        second = store.append(draw)

        assert first is True
        assert second is False
        assert store.count() == 1

        # Also verify file has exactly 1 line
        lines = [l for l in store._path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_insert_different_draws_both_accepted(self, store: DrawStore) -> None:
        draw1 = _make_draw("2024-01-01")
        draw2 = _make_draw("2024-01-16")
        assert store.append(draw1) is True
        assert store.append(draw2) is True
        assert store.count() == 2

    def test_contains_after_insert(self, store: DrawStore) -> None:
        draw = _make_draw("2024-01-01")
        store.append(draw)
        assert store.contains("2024-01-01") is True
        assert store.contains("2024-01-16") is False

    def test_iter_draws_returns_all(self, store: DrawStore) -> None:
        for day in ["2024-01-01", "2024-01-16", "2024-02-01"]:
            store.append(_make_draw(day))
        draws = list(store.iter_draws())
        assert len(draws) == 3

    def test_checksum_written_after_append(self, store: DrawStore, tmp_path: Path) -> None:
        draw = _make_draw("2024-01-01")
        store.append(draw)
        assert (tmp_path / "draws.checksum").exists()

    def test_checksum_verifies(self, store: DrawStore) -> None:
        draw = _make_draw("2024-01-01")
        store.append(draw)
        assert store.verify_checksum() is True

    def test_checksum_fails_after_manual_mutation(self, store: DrawStore) -> None:
        draw = _make_draw("2024-01-01")
        store.append(draw)
        # Manually corrupt the file
        store._path.write_text("tampered content\n")
        assert store.verify_checksum() is False

    def test_persistence_across_instances(self, tmp_jsonl: Path, tmp_path: Path) -> None:
        """A new DrawStore instance reading the same file should know about existing draws."""
        checksum_path = tmp_path / "draws.checksum"
        store1 = DrawStore(jsonl_path=tmp_jsonl, checksum_path=checksum_path)
        store1.append(_make_draw("2024-01-01"))

        store2 = DrawStore(jsonl_path=tmp_jsonl, checksum_path=checksum_path)
        assert store2.contains("2024-01-01") is True
        # Insert same draw via store2 → should be rejected
        assert store2.append(_make_draw("2024-01-01")) is False


# ---------------------------------------------------------------------------
# SQLite predictions UNIQUE constraint tests
# ---------------------------------------------------------------------------


class TestPredictionsUniqueConstraints:
    """SPEC §2.2 fix #1: Both UNIQUE constraints on predictions table."""

    def test_insert_prediction_succeeds(self, db_conn: sqlite3.Connection) -> None:
        pred_id = insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="test-model",
            prize_type="two_back",
            pick_value="56",
            pick_rank=1,
            confidence=0.015,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        assert pred_id is not None
        assert isinstance(pred_id, int)

    def test_unique_pick_rank_constraint(self, db_conn: sqlite3.Connection) -> None:
        """Same (draw_id, model_id, prize_type, pick_rank) → second insert returns None."""
        insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="test-model",
            prize_type="two_back",
            pick_value="56",
            pick_rank=1,
            confidence=0.015,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        # Different pick_value but same rank slot
        result = insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="test-model",
            prize_type="two_back",
            pick_value="78",  # different value
            pick_rank=1,     # same rank — should fail UNIQUE (draw_id, model_id, prize_type, pick_rank)
            confidence=0.012,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        assert result is None

    def test_unique_pick_value_constraint(self, db_conn: sqlite3.Connection) -> None:
        """Same (draw_id, model_id, prize_type, pick_value) → second insert returns None.

        This is the idempotency key for re-runs of predict.py before the freeze cutoff.
        SPEC §2.2 note.
        """
        insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="test-model",
            prize_type="two_back",
            pick_value="56",
            pick_rank=1,
            confidence=0.015,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        # Same pick_value but different rank — should fail UNIQUE (draw_id, model_id, prize_type, pick_value)
        result = insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="test-model",
            prize_type="two_back",
            pick_value="56",  # same value
            pick_rank=2,     # different rank
            confidence=0.014,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        assert result is None

    def test_different_model_same_pick_is_allowed(self, db_conn: sqlite3.Connection) -> None:
        """Different model_id with same pick_value and rank is allowed."""
        insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="model-a",
            prize_type="two_back",
            pick_value="56",
            pick_rank=1,
            confidence=0.015,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        result = insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="model-b",  # different model
            prize_type="two_back",
            pick_value="56",
            pick_rank=1,
            confidence=0.013,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        assert result is not None

    def test_different_draw_same_pick_is_allowed(self, db_conn: sqlite3.Connection) -> None:
        """Different draw_id with same pick is allowed."""
        insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="test-model",
            prize_type="two_back",
            pick_value="56",
            pick_rank=1,
            confidence=0.015,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        result = insert_prediction(
            db_conn,
            draw_id="2024-01-16",  # different draw
            model_id="test-model",
            prize_type="two_back",
            pick_value="56",
            pick_rank=1,
            confidence=0.015,
            purchased=False,
            frozen_at="2024-01-16T07:30:00+07:00",
        )
        assert result is not None

    def test_cost_thb_check_constraint(self, db_conn: sqlite3.Connection) -> None:
        """outcomes.cost_thb must be 0 or 80. SPEC §2.2."""
        from fortuna.store import insert_outcome
        # Insert a valid prediction first
        pred_id = insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="test-model",
            prize_type="two_back",
            pick_value="56",
            pick_rank=1,
            confidence=0.015,
            purchased=True,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        assert pred_id is not None

        # Valid cost: 80
        result = insert_outcome(
            db_conn,
            pred_id=pred_id,
            hit=False,
            payout_thb=0,
            cost_thb=80,
            settled_at="2024-01-01T18:00:00+07:00",
        )
        assert result is True

        # Invalid cost: 100 — should raise ValueError (enforced in code per SPEC §2.2)
        with pytest.raises(ValueError, match="cost_thb must be 0 or 80"):
            insert_outcome(
                db_conn,
                pred_id=pred_id,
                hit=False,
                payout_thb=0,
                cost_thb=100,  # no 100 THB path — Pao Tang is 80 THB (SPEC §13 Q2)
                settled_at="2024-01-01T18:00:00+07:00",
            )

    def test_schema_has_both_unique_constraints(self, db_conn: sqlite3.Connection) -> None:
        """Verify both UNIQUE constraints are enforced (behavioral test).

        Rather than parsing DDL text (brittle), we verify both constraints fire
        independently.
        """
        # Set up: insert rank=1, value="56"
        insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="chk-model",
            prize_type="first6",
            pick_value="123456",
            pick_rank=1,
            confidence=0.001,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )

        # Constraint 1: same rank, different value → None
        r1 = insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="chk-model",
            prize_type="first6",
            pick_value="999999",  # different value
            pick_rank=1,          # same rank
            confidence=0.001,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        assert r1 is None, "UNIQUE (draw_id, model_id, prize_type, pick_rank) not enforced"

        # Constraint 2: different rank, same value → None
        r2 = insert_prediction(
            db_conn,
            draw_id="2024-01-01",
            model_id="chk-model",
            prize_type="first6",
            pick_value="123456",  # same value
            pick_rank=2,          # different rank
            confidence=0.001,
            purchased=False,
            frozen_at="2024-01-01T07:30:00+07:00",
        )
        assert r2 is None, "UNIQUE (draw_id, model_id, prize_type, pick_value) not enforced"
